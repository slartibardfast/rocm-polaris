# Phase 7g: SDMA-GFX L2 Incoherence for Page Table Updates

## Status: FIX UNDER TEST — SNOOPED=1 FOR ALL SYSTEM MEMORY

## Root Cause

SDMA and GFX engines are not L2 cache coherent on gfx8 (known hardware limitation — Mesa disabled SDMA for Polaris entirely). The kernel driver uses SDMA to write page table entries (PTEs). The GFX page walker reads stale PTE data, causing VM faults after hipMalloc triggers internal BO remapping.

On VI/gfx8, the SNOOPED PTE bit (bit 2) is the only per-page coherency control. It determines whether GPU accesses to system memory participate in the PCIe coherency protocol. Without SNOOPED=1, GPU reads may see stale data from SDMA writes.

## Current Fix: Unconditional SNOOPED=1 (kernel 6.18.16-16)

Page table BOs are created with `AMDGPU_GEM_CREATE_CPU_GTT_USWC` → `ttm_write_combined`. Our previous patch 0009 set SNOOPED only for `ttm_cached` and `ttm_uncached`, excluding `ttm_write_combined`. This meant page table BOs that fall back to GTT (system memory) had SNOOPED=0 — SDMA writes to them were invisible to the page walker.

**Fix:** Set `AMDGPU_PTE_SNOOPED` unconditionally for ALL system memory pages in `amdgpu_ttm_tt_pde_flags()`. No caching mode check. On VI where SNOOPED is the only coherency knob, it must always be on.

```c
// Before (ttm_write_combined excluded):
if (ttm && ttm->caching != ttm_write_combined)
    flags |= AMDGPU_PTE_SNOOPED;

// After (unconditional):
flags |= AMDGPU_PTE_SNOOPED;
```

## Contingency: If SNOOPED=1 Doesn't Fix 32MB

SNOOPED only affects system memory (GTT) pages. If page table BOs remain in VRAM (their preferred domain), SNOOPED has no effect — the SDMA-GFX incoherence is within the GPU's internal fabric. Contingency plans ranked by feasibility:

### Option A: Force page table BOs to GTT (highest confidence)

```c
// amdgpu_vm_pt.c — amdgpu_vm_pt_create()
// Current:
if (!adev->gmc.is_app_apu)
    bp.domain = AMDGPU_GEM_DOMAIN_VRAM;
// Change for small BAR:
if (!adev->gmc.is_app_apu && amdgpu_gmc_vram_full_visible(&adev->gmc))
    bp.domain = AMDGPU_GEM_DOMAIN_VRAM;
else
    bp.domain = AMDGPU_GEM_DOMAIN_GTT;
```

With PT BOs in GTT + SNOOPED=1, the page walker reads them through the PCIe coherency protocol. CPU can write PTEs directly (no SDMA). ~4MB of page tables in system memory is negligible.

**Why it works:** Eliminates SDMA from the PTE write path entirely. CPU writes to GTT are coherent with the page walker via SNOOPED. This is analogous to `vm_update_mode=2` (CPU PTE updates) but without the sync_wait deadlock — the PT BOs start in GTT, no SDMA→CPU mode transition needed.

### Option B: Disable SDMA rings (quick diagnostic)

Boot with `amdgpu.num_sdma=0`. If the driver falls back to GFX for all DMA operations, the SDMA-GFX incoherence vanishes. Quick 30-second test to confirm the theory. Not a production fix (disables all SDMA).

### Option C: Fix vm_update_mode=2 deadlock

The CPU PTE mode (`vm_update_mode=2`) deadlocked in `amdgpu_bo_sync_wait(vm->root.bo)` during `amdgpu_vm_make_compute`. The wait tries to drain outstanding SDMA fences before switching to CPU mode. If an SDMA fence never signals (broken SDMA on gfx8), it hangs forever.

Fix: skip or timeout the sync_wait when transitioning to CPU mode on gfx8. The stale SDMA fence is irrelevant — we're switching away from SDMA precisely because it's broken.

```c
// amdgpu_vm.c — amdgpu_vm_make_compute(), after setting use_cpu_for_update:
if (vm->use_cpu_for_update) {
    r = amdgpu_bo_sync_wait(vm->root.bo,
                            AMDGPU_FENCE_OWNER_UNDEFINED, true);
    // Add: skip wait failure on gfx8 — SDMA fences may be stale
    if (r && amdgpu_ip_version(adev, GC_HWIP, 0) < IP_VERSION(9, 0, 0))
        r = 0;  // ignore SDMA fence timeout on pre-gfx9
```

### Option D: Debug flush with hardcoded offsets (diagnostic)

Rebuild the runtime-switchable gfx8_pte_flush parameter using raw hex register offsets instead of macro names (avoid gca include crash):

```c
#define GFX8_CP_COHER_CNTL   0xc07c
#define GFX8_CP_COHER_SIZE   0xc07d
#define GFX8_CP_COHER_BASE   0xc07e
#define GFX8_CP_COHER_STATUS 0xc07f
```

Tests whether CP_COHER/HDP/VM_L2 flush resolves VRAM-internal incoherence at runtime without rebuilds.

### Option E: Force GFX ring for PTE writes (Mesa's approach)

Change the SDMA scheduler entity for gfx8 VM updates to use the GFX ring. Most comprehensive fix but deepest driver change. Mesa did the equivalent by disabling SDMA entirely.

## Test Protocol

After cold boot with each fix:

```bash
# Basic functionality
ROC_CPU_WAIT_FOR_SIGNAL=1 timeout 30 ~/rocm-polaris/tests/hip_smoke

# Phase 7g test battery (32MB H2D + compute kernel)
ROC_CPU_WAIT_FOR_SIGNAL=1 timeout 120 ~/rocm-polaris/tests/test_h2d_kernel

# Full boundary test (up to 512MB)
ROC_CPU_WAIT_FOR_SIGNAL=1 timeout 300 ~/rocm-polaris/tests/test_h2d_boundary

# llama.cpp
ROC_CPU_WAIT_FOR_SIGNAL=1 timeout 600 llama-cli \
  -m ~/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -ngl 1 -p "2+2=" -n 8 --threads 4 --no-mmap
```

## Previous Attempts

| Approach | Result | Why it failed |
|----------|--------|---------------|
| TLB wait (VM_INVALIDATE_RESPONSE poll) | No fix | Issue is data coherency, not TLB staleness |
| KFD eviction flush (flush_delayed_work) | No fix | Eviction not the cause; SDMA incoherence is |
| vm_update_mode=2 (CPU PTE writes) | Deadlock | sync_wait hangs on stale SDMA fences |
| gfx8_pte_flush debug param | Kernel crash | gca/gfx_8_0 header include conflicts |
| SNOOPED=1 for ttm_uncached only | Partial | Excluded ttm_write_combined PT BOs |

## References

- [RadeonSI Disables SDMA For Polaris](https://www.phoronix.com/news/RadeonSI-Disables-Polaris-SDMA)
- [Mesa 20: SDMA Disabled on RX-Series](https://linuxreviews.org/Mesa_20_Will_Have_SDMA_Disabled_On_AMD_RX-Series_GPUs)
- [AMDGPU Module Parameters: vm_update_mode](https://docs.kernel.org/gpu/amdgpu/module-parameters.html)
