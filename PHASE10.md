# Phase 10: Interrupt-Driven Signal Completion

## Status: IN PROGRESS — interrupt mechanism WORKING

## Findings

### EOP Interrupts Work on gfx8/No-Atomics
- GPU fires 326 interrupts for 100 dispatches (verified via /proc/interrupts)
- InterruptSignal + single dispatch completes instantly
- 1000 dispatches complete in 237ms (0.237ms/dispatch)
- The interrupt fires via CP EOP → IH ring DMA → KFD ISR → event signal

### Changes Made (hsa-rocr-polaris 7.2.0-32)
- Removed HDP+L2 flush from compute queue StoreRelaxed (kept for blit queues ≤256 slots)
- Removed sched_yield from BusyWaitSignal polling loop
- Doorbell MMIO readback (Phase 9 fix) retained for correctness

### Changes Made (hip-runtime-amd-polaris 7.2.0-35)
- CreateSignal passes num_consumers=1 + GPU agent on no-atomics → creates InterruptSignal
- HwQueueTracker::Create passes interrupt=true on no-atomics
- InterruptSignal uses hsaKmtWaitOnEvent (sleeps until EOP interrupt) instead of tight polling

### Remaining Issue
llama.cpp shows 0.0 t/s despite interrupt mechanism working.
1000 simple dispatches: 237ms. But llama.cpp's 900-node forward pass
takes >60s. The issue is NOT the interrupt mechanism — it's either:
- Chat template prompt eval (100+ tokens × 900 ops = huge compute)
- HDP flush still firing on blit queues during model loading/D2H
- The ggml-hip kernels being much slower than simple test kernels

### Vulkan Baseline
SmolLM2-135M Q8_0: Prompt 848.8 t/s, Generation 83.3 t/s

## Packages
- `linux-lts-rocm-polaris` 6.18.16-34
- `hsa-rocr-polaris` 7.2.0-32
- `hip-runtime-amd-polaris` 7.2.0-35
- `llama-cpp-rocm-polaris` b7376-7
