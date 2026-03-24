# Phase 10: Interrupt-Driven Signal Completion

## Status: WORKING — 59.05 t/s generation (70.9% of Vulkan)

## Solution: Shared KFD Event (Option A)

One KFD event shared across all signals and queues. All BusyWaitSignals
sleep on this event instead of tight-polling. Any dispatch completion
fires the event via CP EOP interrupt → IH ring DMA → KFD ISR.

### Architecture
1. **Sentinel signal**: One static `DefaultSignal` with shared event's
   `event_id` and `event_mailbox_ptr`. Value=INT64_MAX (never consumed).
2. **ScanNewPackets**: Replaces each AQL packet's completion_signal with
   the sentinel before doorbell write. CP sees sentinel, fires interrupt.
3. **BusyWaitSignal::WaitRelaxed**: Sleeps on shared event via
   `hsaKmtWaitOnEvent_Ext(eop, 1ms)` instead of tight mwaitx polling.
4. **ProcessCompletions**: Doorbell MMIO readback on every call (no
   rate-limiting) to flush IOH buffered RPTR writes.

### Key Fixes
- **Phase 8 barrier CPU-wait**: Changed raw `hsa_signal_load_relaxed` +
  yield loop to `hsa_signal_wait_scacquire` so ROCR's
  ProcessAllBounceBuffers runs and decrements signals.
- **g_use_interrupt_wait=false** kept for no-atomics: all signals are
  BusyWaitSignal (no event pool pressure from InterruptSignal).

## Benchmarks

### ROCm (Phase 10)
SmolLM2-135M-Instruct Q8_0, ngl=99, fa=off:
- Prompt eval: 127.90 t/s (5 tokens)
- Generation: 59.05 t/s (99 tokens, 16.93 ms/token)

### Vulkan Baseline
SmolLM2-135M Q8_0:
- Prompt eval: 848.8 t/s
- Generation: 83.3 t/s

### Gap Analysis
- Generation: 59.05 / 83.3 = **70.9%** of Vulkan
- Prompt eval: 127.90 / 848.8 = **15.1%** of Vulkan (room for improvement)
- Remaining gap likely from: HDP flush on blit queues, event wait overhead,
  RPTR polling latency, IOH write commit delay

## Packages
- `linux-lts-rocm-polaris` 6.18.16-34
- `hsa-rocr-polaris` 7.2.0-34
- `hip-runtime-amd-polaris` 7.2.0-36
- `llama-cpp-rocm-polaris` b7376-7
