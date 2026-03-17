# Phase 7g: GPU VM Fault After H2D Blit ≥32MB

## Status: INVESTIGATING

## Symptom

After `hipMemcpy H2D` of ≥32MB, any subsequent compute kernel dispatch causes a GPU VM fault ("Page not present"). The fault address is 10-25MB past the data allocation, in the 48-63MB range of VRAM address space — suggesting internal GPU structures (code objects, queue descriptors, or kernarg) are unmapped.

## Test Results (clean cold boot/kexec)

| Test | Result |
|------|--------|
| A: hipMemset 32MB + verify<<<1,256>>> | PASS |
| B: H2D 16MB + verify<<<1,256>>> | PASS |
| C: H2D 32MB + read_one<<<1,1>>> | PASS |
| D: H2D 32MB + verify<<<131072,256>>> | **VM FAULT** |
| test_h2d_boundary (blit-only) up to 512MB | PASS |
| test_h2d_gpu_verify 16MB | PASS (both cpu & gpu) |
| test_h2d_gpu_verify 32MB | **VM FAULT** |

**Key differentiator:** Test C vs D — both follow a 32MB H2D blit. `read_one<<<1,1>>>` (2 ptr args) passes; `verify<<<131072,256>>>` (4 args, large grid) faults. On a faulted GPU, even `verify<<<1,256>>>` after H2D faults — but post-fault results are unreliable.

## Hardware Context

- **GPU:** Radeon Pro WX 2100 (Polaris 12, gfx803, 2GB GDDR5)
- **Visible VRAM BAR:** 256MB (no resizable BAR on Westmere)
- **Total VRAM:** 2048MB
- **amdgpu_vram_mm size:** 268435456 (256MB — visible only)
- **Host:** Dual Xeon X5650 (Westmere), no PCIe atomics, no resizable BAR

## Root Cause Hypothesis: VRAM BO Eviction Past Visible BAR

### VRAM Layout

```
0x4100000000  ┌──────────────────────────┐
              │  Visible VRAM (256MB)    │  ← PCIe BAR window
              │  - internal BOs          │
              │  - code objects           │
              │  - queue ring buffers     │
              │  - user allocations       │
0x4110000000  ├──────────────────────────┤
              │  Invisible VRAM (1792MB) │  ← GPU-internal only
              │                          │
0x4180000000  └──────────────────────────┘
```

### Eviction Cascade

When `hipMemcpy H2D 32MB` runs:

1. CLR allocates staging buffer chunks (4MB each, 8 chunks for 32MB) from system memory (GTT, UC via Phase 7e)
2. Each chunk: CPU memcpy → staging, then `hsa_amd_memory_async_copy` (blit kernel) copies staging → VRAM
3. The blit kernel's own kernarg is from `system_allocator()` (system memory) — fine
4. **But:** The H2D copy may trigger TTM memory management operations that cause internal BOs to be **evicted from visible VRAM**

### TTM Eviction Logic

`amdgpu_ttm.c:135-150`:
```c
if (!amdgpu_gmc_vram_full_visible(&adev->gmc) &&
    !(abo->flags & AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED) &&
    amdgpu_res_cpu_visible(adev, bo->resource)) {
    // Evict to invisible VRAM (fpfn = visible_vram_size >> PAGE_SHIFT)
    abo->placements[0].fpfn = adev->gmc.visible_vram_size >> PAGE_SHIFT;
}
```

BOs without `CPU_ACCESS_REQUIRED` can be evicted from visible → invisible VRAM. If internal structures (code objects, queue descriptors) get evicted and the GPU VM page table isn't properly updated, the next dispatch faults.

### Why Blit Works But User Kernels Don't

- **Blit kernel:** kernarg from `system_allocator()` (GTT), code object pre-loaded at init (likely pinned in visible VRAM)
- **User kernel:** kernarg from CLR's `kern_arg_pool` (potentially fine-grain VRAM), code object loaded lazily

### Fault Address Analysis

Fault addresses across runs: 0x41030b2000, 0x41030c3000, 0x4103801000, 0x4103ee8000

All in the 48-63MB range of VRAM. This is well within the 256MB visible region. So the fault may not be about visible/invisible eviction at all — the internal structures ARE in visible VRAM, but their page table entries get invalidated during the H2D blit's VM operations.

### Alternative Hypothesis: TLB Invalidation During Blit

The H2D blit at 32MB involves 8 staging buffer chunks. Each chunk's `hsa_amd_memory_async_copy` may trigger GPU VM page table updates (mapping staging buffer pages for GPU access). On VI/gfx8, page table updates require TLB invalidation. If the TLB invalidation is too broad (full TLB flush instead of targeted), it could temporarily unmap pages for other BOs.

If a compute dispatch is submitted immediately after the blit completes, the CP may try to fetch the kernel code object or kernarg before the TLB has been repopulated — causing a "Page not present" fault.

## Investigation Plan

1. **Check dmesg for BO eviction during H2D** — is TTM actually evicting BOs?
2. **Check if kernarg pool is GTT or VRAM** — trace CLR's `kern_arg_pool` type on our platform
3. **Check GPU VM page table state** — use `amdgpu_vm_info` debugfs after H2D
4. **Test with explicit hipDeviceSynchronize + sleep** — does a delay between H2D and kernel dispatch help? (would confirm TLB repopulation timing)
5. **Test with smaller staging chunk size** — does reducing chunks (fewer VM operations) raise the threshold?

## Key Finding: Timing/Fence Issue Confirmed

A 2-second `usleep()` between `hipDeviceSynchronize()` and the verify kernel launch **eliminates the fault** — even on a GPU that has already faulted from prior tests. This proves the VM fault is a **page table / TLB timing issue**, not a permanent mapping error.

### GFX8 TLB Behavior

`amdgpu_vm.c:1133-1135`:
```c
/* On GFX8 and older any 8 PTE block with a valid bit set enters the TLB */
flush_tlb |= amdgpu_ip_version(adev, GC_HWIP, 0) < IP_VERSION(9, 0, 0);
```

GFX8 forces TLB flush on ALL page table updates. The H2D blit's staging buffer mapping triggers `amdgpu_vm_bo_update()` via SDMA, which writes PTEs and triggers `VM_INVALIDATE_REQUEST` for the process VMID. If the TLB invalidation hasn't fully propagated by the time the CP fetches the next dispatch packet's code object or kernarg, it reads a stale TLB entry → "Page not present."

### Root Cause

The `hsa_amd_memory_async_copy` path maps/unmaps staging buffer pages into the GPU VA space for each chunk. Each map/unmap updates PTEs via SDMA and triggers TLB invalidation (`gmc_v8_0_flush_gpu_tlb` → `WREG32(mmVM_INVALIDATE_REQUEST, 1 << vmid)`). The TLB invalidation is asynchronous — the MMIO write returns immediately, but the actual TLB flush takes time to propagate to all CUs. The CP may start fetching the next kernel's code object before the TLB flush completes, hitting stale entries.

## Fix Options

### Option A: Pin Internal BOs in Visible VRAM (kernel patch)
Mark code objects and queue descriptors with `AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED` to prevent eviction from visible VRAM.

### Option B: Force Kernarg to System Memory (CLR patch)
Change CLR's `kern_arg_pool` to use system memory (HSA kernarg region) instead of fine-grain VRAM on no-atomics platforms. Already UC from Phase 7e.

### Option C: Wait for TLB Invalidation Completion (kernel or ROCR)
After SDMA PTE updates, wait for `VM_INVALIDATE_REQUEST` to complete by polling `VM_INVALIDATE_RESPONSE` before allowing the next dispatch. This is the correct fix — the current code writes `VM_INVALIDATE_REQUEST` but may not wait for acknowledgment on gfx8.

### Option D: Avoid Staging Buffer Map/Unmap Per Chunk (ROCR)
Pre-map the staging buffer into GPU VA at init time and keep it mapped. Avoid per-chunk map/unmap that triggers TLB invalidation. This eliminates the PTE update → TLB flush → stale TLB race entirely.

### Option E: Reduce VRAM Pressure (CLR patch)
Ensure staging buffer uses GTT-only placement (no VRAM fallback). Already partially done in Phase 7e.

## Test Battery

Pre-built at `~/rocm-polaris/tests/test_h2d_kernel`:

```
A: memset 32MB + verify<<<1,256>>>          (baseline, no blit)
B: H2D 16MB + verify<<<1,256>>>             (under threshold)
C: H2D 32MB + read_one<<<1,1>>>             (trivial kernel after blit)
D: H2D 20MB + verify<<<1,256>>>             (threshold sweep)
E: H2D 24MB + verify<<<1,256>>>
F: H2D 28MB + verify<<<1,256>>>
G: H2D 30MB + verify<<<1,256>>>
H: H2D 31MB + verify<<<1,256>>>
I: H2D 32MB + verify<<<1,256>>>             (expected fault)
J: H2D 32MB + verify<<<131072,256>>>         (full grid)
```

Run after cold boot: `ROC_CPU_WAIT_FOR_SIGNAL=1 ~/rocm-polaris/tests/test_h2d_kernel`
