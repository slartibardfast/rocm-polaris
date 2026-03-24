# Phase 13: VRAM AQL Ring Buffer

## Status: FOUNDATION BUILT — kernel topology fix works, performance wash

## Goal

Move the AQL ring buffer from system memory (PCIe 2.0 reads, ~500ns) to
VRAM (local GDDR5 reads, ~100ns) so the GPU's CP fetches packets faster.
Eliminate the pkt[15] IOH readback (~2µs/dispatch) since PCIe downstream
ordering handles VRAM writes. Expected: save ~2ms/token → 80+ t/s.

## What We Built

### Kernel: Two-Part Fix for CPU-Visible VRAM on Small BAR

**Problem:** KFD blocks PUBLIC VRAM allocations on small-BAR systems AND
reports all VRAM as a single PRIVATE bank.

**Fix 1 — kfd_chardev.c:** Relaxed the small-BAR PUBLIC VRAM rejection to
allow allocations that fit within `local_mem_size_public` (256MB BAR).

**Fix 2 — kfd_crat.c:** Split VRAM into two topology banks:
- PUBLIC: 256MB (CPU-visible via BAR, heap_type=1)
- PRIVATE: 1.75GB (GPU-only, heap_type=2)

Both fixes are required together — without Fix 2, ROCR never creates a
CPU-accessible VRAM region. Without Fix 1, the kernel rejects the alloc.

Verified: `rocminfo` shows GPU Pool 1 as COARSE GRAINED 256MB, and the
VRAM allocation succeeds at 0x4100400000.

### ROCR: Double-Buffered Ring + Sideband

**Architecture:**
- System memory ring: CLR writes AQL packets here, ScanNewPackets reads
- VRAM shadow ring: CP reads from here (MQD points to VRAM)
- StoreRelaxed copies 64B per dispatch (header-last for WC atomicity)

**Explicit PUBLIC region search:** `coarsegrain_allocator()` picks the
PRIVATE pool (last enumerated). Our code iterates `agent_->regions()` to
find the coarse-grain VRAM region with `HostAccess=1`.

**Compute queues only:** Blit queues (≤256 slots) keep system memory ring
for HDP flush coherency during weight copies.

### CLR-Side Sentinel (Experimental)

Attempted to eliminate the double-buffer copy by having CLR write directly
to VRAM and save signals to a system-memory sideband. Required:
- Sideband pointer + sentinel handle stored in `amd_queue_.reserved2`
- CLR reads reserved2, writes signal to sideband, sentinel to VRAM ring
- ScanNewPackets reads sideband instead of ring (skip header check)
- Both `dispatchAqlPacket` and `dispatchBarrierPacket` injection

**Bug found:** Sideband slots not cleared after read → stale signals
re-tracked on ring wrap → double-Retain → reference corruption → hang.
Fixed with `sig_ptr->handle = 0` clear.

## Performance Results

| Approach | t/s | vs Baseline | Notes |
|----------|-----|-------------|-------|
| Baseline (system ring) | 70 | — | Phase 11/12 |
| VRAM double-buffer | 71 | +1% | Copy offsets readback savings |
| pkt[15] removal only | 70 | 0% | Readback was already ~0 effective cost |
| CLR-side sentinel | 66 | -6% | CLR overhead > ScanNewPackets savings |

## Why It Didn't Help

The per-dispatch overhead (~2.6µs) is an **ensemble of small fixed costs**:

| Source | Cost | Removable? |
|--------|------|------------|
| ScanNewPackets (lock + scan) | ~500ns | CLR sentinel: adds more than saves |
| Kernarg memcpy + sfence | ~300ns | No (correctness) |
| AQL packet fill + ring write | ~200ns | No (fundamental) |
| Doorbell MMIO write | ~100ns | No (fundamental) |
| GPU CP packet fetch | ~500ns | VRAM helps but copy offsets it |
| GPU sentinel processing | ~1µs | No (EOP interrupt mechanism) |

**Key insight:** On Westmere PCIe 2.0, BAR writes to VRAM cost ~200ns per
64B TLP — the same order as the IOH readback they replace. The VRAM ring
trades one overhead for another of equal cost.

**The structural floor is ~14ms/token (70 t/s) for per-dispatch ROCm on
this hardware.** Going higher requires dispatch batching (multiple AQL
packets per doorbell write), not faster individual dispatches.

## Discoveries Saved to Memory

1. **Small-BAR VRAM topology** — KFD reports ALL VRAM as PRIVATE on
   small-BAR. Must split PUBLIC+PRIVATE banks in CRAT AND relax KFD alloc
   check. Both fixes needed together.

2. **coarsegrain_allocator picks PRIVATE** — Last coarse-grain region
   enumerated wins. Must explicitly search for HostAccess=1 region.

3. **Sideband slots must be cleared** — Ring wrap re-reads stale signals
   → double-Retain → corruption. Clear after read.

4. **pkt[15] readback is effectively free** — IOH commits UC writes during
   sfence + subsequent code. The readback returns instantly.

## Packages

- `linux-lts-rocm-polaris` 6.18.16-36 (CRAT topology fix)
- `hsa-rocr-polaris` 7.2.0-37 (VRAM shadow + double-buffer, falls back gracefully)
- `hip-runtime-amd-polaris` 7.2.0-37 (stable, no CLR sentinel)

## What Would Actually Close the Gap

| Approach | Mechanism | Expected | Effort |
|----------|-----------|----------|--------|
| Dispatch batching | N packets per doorbell | 80-100 t/s | HIGH |
| sched_policy=0 (HWS) | Hardware queue scheduler | Unknown | MEDIUM |
| DOORBELL_BIF_DROP | PCIe write dedup | +2-5% | LOW |
| IQ_TIMER WAIT_TIME | CP sleep between polls | +2-5% | LOW |
| Vulkan backend | Already shipping | 118 t/s | DONE |
