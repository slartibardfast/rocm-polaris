# Phase 7g: GPU VM Fault — ROOT CAUSE FOUND AND FIXED

## Status: RESOLVED — all transfers verified correct, 1MB to 1792MB

## Root Cause

The CLR's **pinned host memory path** for hipMemcpy H2D/D2H creates a temporary
GPU VA mapping for the user's host buffer, launches a DMA copy, then **unpins
(clears PTEs) before the DMA engine finishes reading**. The GPU page walker
faults "page not present" on the freed VA range.

This affects any hipMemcpy larger than `pinnedMinXferSize_` (128KB default) on
platforms without PCIe atomics. Smaller copies use the persistent staging buffer
pool which is never unmapped.

### Evidence

Full KFD ioctl + PTE tracing on kernel 24 captured the exact sequence:

```
756.473099  KFD_ALLOC: va=41013ff000 size=401000 flags=d0000004  ← pin host buffer (4MB+4KB)
756.473186  KFD_MAP:   mem_va=41013ff000 mem_size=401000          ← GPU VA mapped
756.473216  PTE_UPD:   valid system-memory PTEs written
756.473409  PTE_UPD:   PTEs CLEARED (flags=0x0, 0x480)           ← unpin starts
756.473424  KFD_FREE:  handle=2f                                  ← unpin complete
756.473644  VM FAULT   at 0x410145a000 (within freed range)       ← DMA reads hit cleared PTEs
```

**230µs from pin to unpin** — a 4MB DMA copy over PCIe Gen2 x16 needs ~500µs.
The buffer is freed before the copy can possibly complete.

### Fix

Disable pinning on no-atomics platforms in `DmaBlitManager::getBuffer()`:

```cpp
// Before:
bool doHostPinning = enablePin && (size > MinSizeForPinnedXfer);
// After:
bool doHostPinning = enablePin && (size > MinSizeForPinnedXfer) && dev().info().pcie_atomics_;
```

All copies now go through the persistent staging buffer pool (`managed_buffer_`),
which uses fine-grain memory (MTYPE=CC) and is never unmapped. Transfers larger
than the staging chunk size (4MB default) are split into multiple chunks with
per-chunk CPU waits.

### Verification

Every byte verified via H2D + D2H round-trip, pattern `0xCD`:

```
    1MB:          262,144 bytes  PASS
    2MB:          524,288 bytes  PASS
    4MB:        1,048,576 bytes  PASS
    8MB:        2,097,152 bytes  PASS
   16MB:        4,194,304 bytes  PASS
   32MB:        8,388,608 bytes  PASS
   64MB:       16,777,216 bytes  PASS
  128MB:       33,554,432 bytes  PASS
  256MB:       67,108,864 bytes  PASS
  512MB:      134,217,728 bytes  PASS
  768MB:      805,306,368 bytes  PASS
 1024MB:    1,073,741,824 bytes  PASS
 1280MB:    1,342,177,280 bytes  PASS
 1536MB:    1,610,612,736 bytes  PASS
 1792MB:    1,879,048,192 bytes  PASS
```

15/15 sizes pass. 1.75GB transferred on a 2GB card with zero corruption.

Additionally, `test_h2d_kernel` passes all 17 sub-tests (standalone 1-64MB,
sequential 1-64MB, same-size realloc 16-64MB) with GPU-side kernel verification
via `atomicAdd`.

## What Was NOT The Root Cause

All of the following were investigated and ruled out:

| Theory | Test | Result |
|--------|------|--------|
| Invisible VRAM for PT BOs | CPU_ACCESS_REQUIRED → PT in visible VRAM | Still faults |
| GFX ring PTE write coherence | wait=true in unreserve_bo_and_vms | Still faults |
| CPU vs GPU PTE writes | amdgpu.vm_update_mode=3 | Still faults |
| SDMA-GFX L2 incoherence | amdgpu.num_sdma=0 | Still faults |
| TLB stale entries | HEAVYWEIGHT TLB flush | Still faults |
| PT-in-GTT | Force GTT domain for PT BOs | Deadlocks |
| PTE content wrong | Read PDB/PTB from BAR | PTEs correct; just cleared too early |

The PTEs were always written correctly. The kernel's VM subsystem was never at
fault. The bug was entirely in CLR userspace — premature unpin of the host
buffer GPU mapping.

## How We Found It

1. Added `pr_err` traces to `amdgpu_vm_pte_update_flags()` — showed PTE writes
   but no valid VRAM PTEs for our allocations (red herring: those were clears)
2. Added `pr_err` to `amdgpu_vm_bo_update()` — confirmed `invalids` list was
   non-empty (mappings were being processed)
3. Added `KFD_ALLOC/MAP/FREE` traces to KFD ioctls — revealed the pinned host
   buffer (userptr) was allocated, mapped, and freed within 230µs
4. Timing analysis proved DMA couldn't complete in that window
5. `GPU_PINNED_MIN_XFER_SIZE=1073741824` env var disabled pinning — instant fix
6. Built the fix into `hip-runtime-amd-polaris` 7.2.0-6

## Remaining Issues

1. **CP idle stall**: llama.cpp hangs during prompt eval (no VM fault). The
   Command Processor stops processing AQL packets after encountering barrier
   dependencies. This is the SLOT_BASED_WPTR issue, separate from memory.

## Packages

- `linux-lts-rocm-polaris` 6.18.16-24: patches 0004-0006, 0008-0010
- `hsa-rocr-polaris` 7.2.0-13: patches 0001-0005
- `hip-runtime-amd-polaris` 7.2.0-6: Fix 1-4 (fine-grain staging, clflush, cpu_wait, no pinning)
- `rocblas-gfx803` 7.2.0-2: gfx803 target restored
- `llama-cpp-rocm-polaris` b7376-1: native gfx803 build

## Boot Parameters

```
amdgpu.sched_policy=2 amdgpu.runpm=0
```
