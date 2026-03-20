# Phase 7g: GPU VM Fault — ROOT CAUSE FOUND AND FIXED

## Status: RESOLVED — pinned host memory path causes use-after-unmap

## Root Cause

The CLR's **pinned host memory path** for hipMemcpy H2D/D2H creates a GPU VA
mapping for the user's host buffer, launches a DMA copy via SDMA, then **unpins
(clears PTEs) after the DMA completion signal fires**. On gfx8/no-atomics, the
SDMA engine still has in-flight reads from the pinned VA range when PTEs are
cleared, causing "page not present" VM faults.

### Evidence

Full KFD ioctl tracing (kernel 24) showed the exact sequence:

```
756.473099  KFD_ALLOC: va=41013ff000 size=401000 flags=d0000004  ← pin host buffer (4MB+4KB)
756.473186  KFD_MAP: mem_va=41013ff000 mem_size=401000            ← GPU VA mapped
756.473216  PTE_UPD: valid PTEs written for staging range
756.473409  PTE_UPD: PTEs CLEARED (flags=0x0, 0x480)             ← unpin starts
756.473424  KFD_FREE: handle=2f                                   ← unpin complete
756.473644  VM FAULT at 0x410145a000 (within freed staging range) ← DMA reads cleared PTEs
```

- **230µs from pin-to-unpin** for a 4MB copy that needs ~500µs at PCIe Gen2 bandwidth
- The DMA copy cannot complete before the buffer is unpinned
- Fault client is TC (texture cache / page walker), confirming PTE lookup failure

### Verification

Setting `GPU_PINNED_MIN_XFER_SIZE=1073741824` (disable pinning, force staging pool):

| Test | Without fix | With fix |
|------|-------------|----------|
| 1MB H2D+verify | PASS | PASS |
| 4MB H2D+verify | **VM FAULT** | PASS |
| 16MB H2D+verify | never reached | PASS |
| 32MB H2D+verify | never reached | PASS |
| 64MB H2D boundary | never reached | PASS |
| 128MB H2D boundary | never reached | PASS |
| 256MB H2D boundary | never reached | PASS |
| 512MB H2D boundary | never reached | PASS |

### Fix

Added `&& dev().info().pcie_atomics_` to the `doHostPinning` condition in
`DmaBlitManager::getBuffer()` (rocblit.cpp). This forces all copies through
the persistent staging buffer pool (managed_buffer_) which is never unmapped.

```cpp
// Before:
bool doHostPinning = enablePin && (size > MinSizeForPinnedXfer);
// After:
bool doHostPinning = enablePin && (size > MinSizeForPinnedXfer) && dev().info().pcie_atomics_;
```

## What Was NOT The Root Cause

All of the following were investigated and ruled out as red herrings:

| Theory | Test | Result |
|--------|------|--------|
| Invisible VRAM for PT BOs | CPU_ACCESS_REQUIRED → PT in visible VRAM | Still faults |
| GFX ring PTE write coherence | wait=true in unreserve_bo_and_vms | Still faults |
| CPU vs GPU PTE writes | amdgpu.vm_update_mode=3 | Still faults |
| SDMA-GFX L2 incoherence | amdgpu.num_sdma=0 | Still faults |
| TLB stale entries | HEAVYWEIGHT TLB flush | Still faults |
| PT-in-GTT | Force GTT domain for PT BOs | Deadlocks |

The PTEs were always being written correctly — they were just being **cleared too early**
by the pinned memory unpin path.

## Remaining Issues

1. **CP idle stall**: llama.cpp hangs during inference (no VM fault, just stall).
   This is the SLOT_BASED_WPTR / barrier dep issue, separate from memory.

2. **hipMemset on sub-allocated VRAM**: `hipMemset(d_bad, 0, 4)` may not take
   effect for tiny VRAM sub-allocations. Causes test_h2d_kernel 32MB+ to report
   `bad=0xABABABAB` (memset didn't zero the counter). Not a VM fault.

## Packages

- `linux-lts-rocm-polaris` 6.18.16-24: patches 0004-0006, 0008-0010 + debug traces
- `hsa-rocr-polaris` 7.2.0-13: patches 0001-0005
- `hip-runtime-amd-polaris` 7.2.0-6: Fix 1-4 (fine-grain staging, clflush, cpu_wait, **no pinning**)
- `rocblas-gfx803` 7.2.0-2: gfx803 target restored
- `llama-cpp-rocm-polaris` b7376-1: native gfx803 build

## Boot Parameters

```
amdgpu.sched_policy=2 amdgpu.runpm=0
```

No vm_update_mode, no num_sdma, no gartsize overrides needed.
