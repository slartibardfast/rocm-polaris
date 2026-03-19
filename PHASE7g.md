# Phase 7g: GPU VM Fault After H2D Blit ≥32MB

## Status: ROOT CAUSE UNKNOWN — SDMA THEORY DISPROVEN

## What We Know (Hard Facts)

### The fault
- GPU VM fault "Page not present" at ~48-53MB into VRAM address space
- Fault is in a GTT-mapped region: `0x4102fff000 - 0x4105000000 domain GTT`
- Fault address varies slightly between runs but always in this range
- Read from TC (texture cache) — GPU kernel or page walker trying to read

### What triggers it
- `hipMemcpy H2D` of ≥32MB followed by any GPU kernel dispatch
- The H2D blit causes internal structures at `0x4102fff000` to be **unmapped from VRAM and remapped to GTT** (kernel debug output confirms)
- The subsequent GPU kernel dispatch faults reading from the GTT-mapped region

### What does NOT trigger it
- `hipMemset` 32MB + GPU kernel verify: PASS (no blit, no remapping)
- `hipMemcpy H2D` 16MB + GPU kernel verify: PASS (no remapping at 16MB)
- `hipMemcpy H2D` 32MB + `hipMemcpy D2H` 32MB (blit-only, no GPU kernel): PASS at 512MB
- `hipMemcpy H2D` 32MB + `read_one<<<1,1>>>`: passed on kernel 13, faults on kernel 16

### Kernel debug output (clean boot, during fault)
```
Unmap VA 0x4100600000 - 0x4101600000    ← free previous 16MB alloc
Map VA 0x4100600000 - 0x4102600000 domain VRAM   ← new 32MB alloc
Map VA 0x4102fff000 - 0x4105000000 domain GTT     ← internal structures remapped to GTT
... VM fault at 0x41030a0000 (within GTT-mapped region) ...
```

## What We Ruled Out

| Hypothesis | Test | Result |
|---|---|---|
| SDMA-GFX L2 incoherence | `amdgpu.num_sdma=0` (disable SDMA entirely) | **Still faults** |
| TLB stale entries | Poll VM_INVALIDATE_RESPONSE | Still faults |
| KFD eviction race | flush_delayed_work in alloc/map ioctls | Still faults |
| CPU PTE mode (bypass SDMA) | `amdgpu.vm_update_mode=2` | Deadlocks (separate bug) |
| SNOOPED=0 on page table BOs | `if (ttm)` SNOOPED for all system memory | Still faults |
| GPU runtime PM | `amdgpu.runpm=0` | Still faults (but prevents BACO hangs) |

## What We Have NOT Tested

1. **Is the GTT mapping physically backed?** The PTEs map to GTT pages, but are those pages pinned in system memory? If TTM hasn't populated the physical pages, the GPU reads from an unmapped physical address.

2. **Is the BO validation complete before dispatch?** The `amdgpu_vm_bo_update()` submits PTE writes (via GFX ring with num_sdma=0). The dispatch fence should ensure completion, but maybe the fence isn't properly awaited by the KFD compute queue.

3. **Is the issue in how the internal BO is remapped?** The unmap at `0x4102fff000` followed by remap to a larger range (`0x4102fff000 - 0x4105000000`) might leave a gap or use a stale PDE if the page directory entry for that range wasn't updated.

4. **Is the fault address in the page directory or page table BO itself?** If the PDE pointing to the page table for the 0x41030xxxxx range was invalidated during remapping, the page walker can't find the PTE.

5. **Does the fault happen with a smaller VRAM allocation that doesn't trigger remapping?** If we can find a 32MB allocation pattern that doesn't cause the GTT remapping, we can confirm the remapping is the trigger.

## Recommended Next Steps (In Order)

### Step 1: Characterize the remapping trigger
What exactly is the BO at `0x4102fff000`? Is it a queue descriptor, scratch memory, code object? Enable kernel debug (`echo 'file amdgpu_amdkfd_gpuvm.c +p' | sudo tee /proc/dynamic_debug/control`) and trace which BO gets remapped and why.

### Step 2: Check if the BO is a KFD internal allocation
Cross-reference `0x4102fff000` with KFD's BO list. The BO might be allocated during `kfd_process_device_init_vm` and remapped during `hipMalloc`.

### Step 3: Test without the GTT remapping
If the BO at `0x4102fff000` is being moved from VRAM to GTT because of memory pressure, try reducing the allocation to avoid the move. Or try `hipMalloc(32MB)` without the preceding 16MB alloc/free cycle (the test does A=32MB memset, B=16MB H2D, then C=32MB H2D — B's 16MB alloc/free may fragment VRAM).

### Step 4: Test with standalone 32MB (no prior allocations)
Write a minimal test: single `hipMalloc(32MB)` + `hipMemcpy H2D` + `verify<<<1,256>>>` with no prior GPU operations. This isolates whether the fault requires the alloc/free cycling from tests A and B.

## Packages (Current)

- `linux-lts-rocm-polaris` 6.18.16-16: patches 0004-0006, 0008-0009 + TLB wait + eviction flush
- `hsa-rocr-polaris` 7.2.0-13: UC shared memory, bounce buffer, cpu-wait blit deps
- `hip-runtime-amd-polaris` 7.2.0-5: UC hostAlloc, synchronous flush, cpu_wait_for_signal
- `rocblas-gfx803` 7.2.0-2: gfx803 target restored
- `llama-cpp-rocm-polaris` b7376-1: native gfx803 build

## Boot Parameters

```
amdgpu.sched_policy=2 amdgpu.runpm=0
```

Removed: `gartsize=2048` (caused GART table BO eviction), `vm_update_mode=2` (deadlocks), `num_sdma=0` (diagnostic only, didn't help)
