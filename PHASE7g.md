# Phase 7g: SDMA-GFX L2 Incoherence — PTE Flush Debug Module

## Status: TESTING FLUSH OPTIONS

## Root Cause (Confirmed)

SDMA writes GPU page table entries (PTEs) to VRAM. The GFX engine's page walker reads PTEs from VRAM through a potentially stale cache. On gfx8, SDMA and GFX are not L2 cache coherent (known hardware limitation — Mesa disabled SDMA for Polaris entirely).

The kernel driver's TLB flush (`VM_INVALIDATE_REQUEST`) invalidates the TLB and VM L2 translation cache, but does NOT ensure the underlying **data** written by SDMA has reached VRAM's backing store. The page walker re-walks, reads from VRAM (or an intermediate cache), and gets stale PTE data.

## Approach: Runtime-Switchable Flush Modes

Add a kernel module parameter `gfx8_pte_flush` (writable at runtime via sysfs) that controls which cache flush operations run after SDMA PTE writes, before the TLB invalidation. This allows testing each approach without rebuilding or rebooting.

```
/sys/module/amdgpu/parameters/gfx8_pte_flush
```

### Flush Mode Bitflags

| Bit | Value | Mode | Mechanism | What it flushes |
|-----|-------|------|-----------|-----------------|
| 0 | 1 | CP_COHER | MMIO write to `CP_COHER_CNTL` with `TC_WB_ACTION_ENA \| TC_ACTION_ENA` | GFX shader L2 (TC) writeback + invalidate |
| 1 | 2 | HDP | MMIO write to `HDP_MEM_COHERENCY_FLUSH_CNTL` | Host Data Path — CPU↔VRAM coherency |
| 2 | 4 | VM_L2 | Toggle `VM_L2_CNTL.ENABLE_L2_CACHE` off/on | VM page table L2 cache (direct reset) |

Modes can be combined: `echo 5 > .../gfx8_pte_flush` = CP_COHER + VM_L2.

### Testing Protocol

After kexec with the debug kernel:

```bash
# Baseline — should fault at 32MB (confirms bug present)
echo 0 | sudo tee /sys/module/amdgpu/parameters/gfx8_pte_flush
ROC_CPU_WAIT_FOR_SIGNAL=1 timeout 30 ~/rocm-polaris/tests/test_h2d_kernel

# Option 1: CP_COHER TC writeback+invalidate
# Cold reboot required after each fault
echo 1 | sudo tee /sys/module/amdgpu/parameters/gfx8_pte_flush
ROC_CPU_WAIT_FOR_SIGNAL=1 timeout 30 ~/rocm-polaris/tests/test_h2d_kernel

# Option 2: HDP flush
echo 2 | sudo tee /sys/module/amdgpu/parameters/gfx8_pte_flush
ROC_CPU_WAIT_FOR_SIGNAL=1 timeout 30 ~/rocm-polaris/tests/test_h2d_kernel

# Option 3: VM L2 cache reset
echo 4 | sudo tee /sys/module/amdgpu/parameters/gfx8_pte_flush
ROC_CPU_WAIT_FOR_SIGNAL=1 timeout 30 ~/rocm-polaris/tests/test_h2d_kernel

# Option 4: CP_COHER + VM L2 (belt and suspenders)
echo 5 | sudo tee /sys/module/amdgpu/parameters/gfx8_pte_flush
ROC_CPU_WAIT_FOR_SIGNAL=1 timeout 30 ~/rocm-polaris/tests/test_h2d_kernel

# Option 5: All three
echo 7 | sudo tee /sys/module/amdgpu/parameters/gfx8_pte_flush
ROC_CPU_WAIT_FOR_SIGNAL=1 timeout 30 ~/rocm-polaris/tests/test_h2d_kernel
```

**Note:** After any VM fault, the GPU is in a degraded state. Must cold boot between tests that fault.

### Registers Used

**CP_COHER_CNTL (0xc07c):**
- `TC_WB_ACTION_ENA` (bit 18): L2 writeback — dirty lines written to VRAM
- `TC_ACTION_ENA` (bit 23): L2 invalidate — cache lines discarded
- `CP_COHER_SIZE` (0xc07d): Set to 0xFFFFFFFF for full range
- `CP_COHER_BASE` (0xc07e): Set to 0 for full range
- `CP_COHER_STATUS` (0xc07f): Poll bit 0 for completion

**HDP_MEM_COHERENCY_FLUSH_CNTL (0x5520):**
- Write any value to trigger HDP cache flush
- Normally used for CPU-VRAM coherency, but may also affect SDMA write buffers

**VM_L2_CNTL (0x500):**
- `ENABLE_L2_CACHE` (bit 0): Toggle off/on to reset VM L2 state
- Nuclear option — discards ALL cached page table data

### Overhead Estimates

| Mode | Mechanism | Estimated overhead | Notes |
|------|-----------|-------------------|-------|
| CP_COHER (1) | 4 MMIO writes + poll | ~2-10µs | May race with GFX ring if active |
| HDP (2) | 1 MMIO write | ~1µs | Cheapest option |
| VM_L2 (4) | 3 MMIO writes (read-modify-write) | ~2-5µs | Most disruptive, clears all translations |
| All (7) | All combined | ~5-15µs | Maximum coverage |

Per hipMalloc overhead. Negligible for compute workloads.

### What vm_update_mode=2 Taught Us

The CPU PTE write approach (`vm_update_mode=2`) deadlocked in `amdgpu_bo_sync_wait()` during the SDMA→CPU mode transition. The sync_wait tries to drain outstanding SDMA fences on the page directory BO, and at least one fence never signals. This suggests the SDMA engine itself has completion issues on our platform — which aligns with the broader SDMA incoherence theory.

### Previous Attempts (Superseded)

- **TLB invalidation wait (patch 0010):** Polls VM_INVALIDATE_RESPONSE. Correct but insufficient alone.
- **KFD eviction flush:** flush_delayed_work in alloc/map ioctls. Harmless belt-and-suspenders.
- **vm_update_mode=2:** Deadlocks on sync_wait. Cannot be used without fixing SDMA fence completion.
