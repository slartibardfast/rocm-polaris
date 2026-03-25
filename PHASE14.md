# Phase 14: Deferred Doorbell Batching

## Status: REVERTED — approach has subtle dispatch hang bug

## Goal

Batch N AQL dispatches per doorbell write to reduce per-dispatch overhead.
The CP gets N packets at once and processes them continuously. CLR already
has `dispatchGenericAqlPacketBatch()` for CUDA graph capture (N packets,
1 doorbell). Phase 14 attempted to bring this benefit to regular dispatch.

## Approach: ROCR Deferred Doorbell

In `StoreRelaxed`, defer the doorbell MMIO write. Accumulate dispatches.
Ring the doorbell every Nth dispatch or when `ProcessCompletions` needs
to check RPTR (which triggers `FlushDoorbell`).

### Implementation
- `deferred_doorbell_value_` / `deferred_count_` state in AqlQueue
- `FlushDoorbell()` method: sfence + legacy doorbell WPTR encoding
- Flush triggers: batch threshold, ProcessCompletions, destructor
- Blit queues (≤256 slots) flush immediately

### Code: ~25 lines in `hsa-rocr/PKGBUILD`

## Results

| Batch Size | t/s | vs Baseline | Notes |
|-----------|-----|-------------|-------|
| N=1 (always flush) | untested clean | — | GPU wedged from N=2 before testing |
| N=2 | **GPU WEDGE** | — | CP preemption timeout, unrecoverable |
| N=8 | 70.7 | +1% | Neutral — too small to matter |
| N=64 | 65.5 | **-7%** | GPU starves during 64-dispatch accumulation |

## Why It Failed

### N=64: GPU Starvation
The CPU writes 64 packets in ~64µs. During this time, the GPU sits idle
(no doorbell yet). When the doorbell fires, the GPU processes all 64
packets (~960µs). But the 64µs idle gap per batch × 14 batches/token =
~900µs of added GPU idle time. This exceeds the ~100µs saved from fewer
doorbells.

**Root cause:** The system is GPU-bound. The GPU processes each dispatch
at ~15µs. The CPU submits at ~1µs. With per-dispatch doorbells, the GPU
is always fed. Batching creates gaps.

### N=2: Dispatch Hang
Dispatching worked for ~20 tokens, then hung. Investigation showed:
- 577 `AMDKFD_IOC_WAIT_EVENTS` in 5 seconds (event IS firing)
- Signals never completing despite events
- GPU wedge persists across kexec (requires cold reboot)

The hang occurs during model loading where a single compute dispatch is
followed by an immediate WaitCurrent. The deferred doorbell delays the
submission; ProcessCompletions → FlushDoorbell should flush it, but
something in the timing breaks. The exact root cause is unresolved.

### N=8: Neutral
Small enough batch that GPU idle time is minimal (~8µs per batch).
But the doorbell savings (~100 fewer writes) is negligible. Net zero.

## The Fundamental Issue

Phase 12 proved the system is **GPU-bound, not CPU-serialization-bound**:
- Signal pool 64→1024: no effect (GPU keeps up)
- pkt[15] readback removal: no effect (readback was free)
- VRAM ring: no effect (BAR writes ≈ IOH readbacks)

Batching doorbells saves CPU overhead (~100ns each) but creates GPU
idle gaps (~1µs each batch boundary × N). The tradeoff is negative for
all tested N values.

## Discovery: CLR Already Has Batch Dispatch

`dispatchGenericAqlPacketBatch()` (rocvirtual.cpp:1287-1482):
- Reserves N slots atomically
- Writes N packets to ring in a loop
- Single doorbell for all N
- Adaptive batch size (1→2→4→...→256)
- Used ONLY in CUDA graph capture mode

This existing code is architecturally superior to our ROCR-level approach
because it eliminates CPU overhead per packet (kernarg, signal, etc.),
not just the doorbell. However, it requires CUDA graph support which
gfx803's HIP backend doesn't provide.

## What Would Make Batching Work

1. **CLR-level batching** — use `dispatchGenericAqlPacketBatch` for
   non-graph workloads. Requires buffering packets in CLR and flushing
   at sync points. Much more complex but saves ~2µs/dispatch (kernarg
   + signal + doorbell), not just ~100ns/dispatch (doorbell only).

2. **CUDA graph support for gfx803** — if HIP supported graph capture
   on gfx803, llama.cpp would automatically use the batch path. Requires
   ROCm-level work (graph capture + replay for gfx8 compute).

3. **Application-level batching** — modify ggml-hip to submit batches
   of independent kernels. Limited by data dependencies between layers.

## Packages

Phase 14 reverted. Stable state restored:
- `hsa-rocr-polaris` 7.2.0-37 (Phase 13, no batching)
- All other packages unchanged

## Note: GPU Wedge Persistence

The N=2 batch test left the GPU in a state that persists across kexec.
The `cp queue preemption time out` error appears immediately on next boot.
Simple HIP tests work but complex workloads (llama.cpp model loading) hang.
Full cold reboot required to clear this state.
