# Phase 7g: GPU VM Fault After H2D Blit ≥32MB

## Status: ROOT CAUSE FOUND — FIX PROPOSED

## Root Cause: KFD Process Eviction/Restore Race

### The Bug

`hipMalloc(32MB)` triggers TTM VRAM pressure → KFD evicts the process (stops queues, unmaps BOs) → schedules a **delayed restore** (100ms timer) → `hipMalloc` returns to userspace → application dispatches a kernel → queues not yet restored → **VM fault**.

### Evidence Chain

1. **Kernel debug output** confirms eviction during `hipMalloc`:
   ```
   Unmap VA 0x4102fff000 - 0x4104000000
   Evicting process pid 2246 queues        ← QUEUES STOPPED
   Unmap VA 0x4100600000 - 0x4101600000
   Map VA 0x4100600000 - 0x4102600000      ← 32MB allocation mapped
   Map VA 0x4102fff000 - 0x4105000000      ← internal BOs remapped to GTT
   ```

2. **2-second usleep() between hipMalloc and kernel dispatch eliminates the fault** — the 100ms restore timer completes during the sleep.

3. **The restore delay is hardcoded:**
   ```c
   // kfd_priv.h:724
   #define PROCESS_RESTORE_TIME_MS 100
   ```
   The eviction worker (`evict_process_worker` in `kfd_process.c:2014`) is scheduled as `schedule_delayed_work(..., msecs_to_jiffies(PROCESS_RESTORE_TIME_MS))`.

4. **Fault address (53.7MB into VRAM)** is in the internal GPU structures region — code objects, queue descriptors, or page tables that were unmapped during eviction and not yet restored.

### The Eviction/Restore Flow

```
hipMalloc(32MB)
  → kfd_ioctl_alloc_memory_of_gpu
    → amdgpu_gem_object_create → TTM placement
      → TTM needs VRAM → evicts existing BOs
        → KFD eviction fence signals
          → schedule_delayed_work(&process->eviction_work, 100ms)
    → BO allocated, ioctl returns
  → kfd_ioctl_map_memory_to_gpu
    → amdgpu_vm_bo_update (maps new BO into GPU page tables)
    → ioctl returns

hipMemcpy H2D 32MB          ← blit queue may work (re-created internally)
verify<<<1,256>>>            ← compute queue NOT RESTORED → VM fault

... 100ms later ...

evict_process_worker fires:
  → restore_process_bos (validate all BOs, update page tables)
  → restore_process_queues (restart compute queues)
  → TOO LATE — process already faulted
```

### Why 16MB Works But 32MB Doesn't

At 16MB, the `hipMalloc` fits within existing VRAM without triggering BO eviction. No process eviction occurs. At 32MB, TTM must evict BOs to make contiguous space (even though total VRAM usage is well under 2GB), triggering the process eviction cascade.

### Why Blit Works But Compute Doesn't

The blit kernel runs on ROCR's internal queue. During `hipMemcpy H2D`, ROCR may re-create or re-validate its internal blit queue as part of the copy operation. The **user's compute queue** is managed by KFD's eviction/restore mechanism and isn't restored until the delayed worker fires.

### Why This Wasn't Seen on Coherent Platforms

On platforms with PCIe atomics and resizable BAR (typical modern setups), VRAM is fully visible (no 256MB BAR limit), eviction pressure is lower, and the TTM placement algorithm has more headroom. The 100ms restore delay is rarely triggered because BOs don't need to move between visible/invisible VRAM.

## Fix: Flush Eviction Work Before Returning from Alloc/Map Ioctls

### Kernel Patch (0010)

In `kfd_chardev.c`, after `kfd_ioctl_alloc_memory_of_gpu` and `kfd_ioctl_map_memory_to_gpu` complete, flush any pending eviction work to ensure the process is fully restored before returning to userspace.

```c
// At the end of kfd_ioctl_alloc_memory_of_gpu, before return:
if (p->eviction_work.work.func)
    flush_delayed_work(&p->eviction_work);

// Same for kfd_ioctl_map_memory_to_gpu
```

`flush_delayed_work()` is safe to call unconditionally:
- If no work is pending, returns immediately (no overhead)
- If work is pending, executes it synchronously and waits for completion
- If work is already running, waits for it to finish

This ensures that by the time `hipMalloc` returns to userspace, the process queues are restored and all BOs are re-validated. The 100ms delay is preserved for the normal eviction case (where no ioctl is actively waiting), but the ioctl path doesn't race against it.

### Alternative Considered: Reduce PROCESS_RESTORE_TIME_MS to 0

This would eliminate the delay entirely, but risks thrashing if multiple evictions happen in rapid succession. The `flush_delayed_work` approach is better because it only synchronizes when userspace is actively waiting for an ioctl to complete — the delayed path remains available for background eviction handling.

### Alternative Considered: ROCR/CLR Userspace Workaround

ROCR could `usleep(PROCESS_RESTORE_TIME_MS)` after alloc ioctls. But this is a userspace band-aid for a kernel bug — the kernel should not return from an ioctl leaving the process in an inconsistent state.

## Implementation

### Files Changed

**kernel/PKGBUILD** — sed injection in `kfd_chardev.c`:
- After `kfd_ioctl_alloc_memory_of_gpu`: flush eviction work
- After `kfd_ioctl_map_memory_to_gpu`: flush eviction work

### Test Plan

After cold boot with the fix:
```bash
ROC_CPU_WAIT_FOR_SIGNAL=1 ~/rocm-polaris/tests/test_h2d_kernel
```

Expected: all 10 tests (A-J) PASS, including:
- I: H2D 32MB + verify<<<1,256>>> (was faulting)
- J: H2D 32MB + verify<<<131072,256>>> (full grid)

Then run llama.cpp:
```bash
ROC_CPU_WAIT_FOR_SIGNAL=1 llama-cli -m ~/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -ngl 1 -p "2+2=" -n 8 --threads 4 --no-mmap
```

## Test Results History

| Test | Pre-7g (6.18.16-13) | Post-7g TLB wait (6.18.16-14) | Expected (flush fix) |
|------|---------------------|-------------------------------|---------------------|
| A: memset 32MB + verify | PASS | PASS | PASS |
| B: H2D 16MB + verify | PASS | PASS | PASS |
| C: H2D 32MB + read_one | PASS | VM FAULT | PASS |
| D-H: H2D 20-31MB + verify | untested | untested | PASS |
| I: H2D 32MB + verify<<<1,256>>> | VM FAULT | VM FAULT | **PASS** |
| J: H2D 32MB + verify<<<131072,256>>> | VM FAULT | VM FAULT | **PASS** |

## Previous Fix Attempts (Superseded)

- **TLB invalidation wait (VM_INVALIDATE_RESPONSE poll):** Correct for general robustness but didn't fix this bug. The issue is missing PTEs, not stale TLB entries. Keeping the TLB wait patch for defense-in-depth.

- **Phase 7f ACQUIRE_MEM L2 writeback:** Removed. Was addressing a different issue (GPU L2 coherency for system memory). Replaced by kernel SNOOPED=1 fix.
