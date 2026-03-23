# Phase 10: Signal Completion Performance on No-Atomics Platforms

## Status: NOT STARTED

## Problem

Phase 9 fixed the correctness bug (queue wedge from stale RPTR), but GPU
inference is unusable due to catastrophic performance. SmolLM2-135M Q8_0:

| Backend | Prompt (t/s) | Generation (t/s) |
|---------|-------------|-------------------|
| Vulkan  | 848.8       | 83.3              |
| ROCm    | 0.0         | 0.0 (timeout)     |
| CPU     | ~50         | ~15               |

The ROCm path cannot complete a single forward pass (30 layers, ~900
kernel dispatches) within 5 minutes on the same hardware.

## Root Cause

On platforms with PCIe atomics, the GPU atomically decrements completion
signals via PCIe posted write. This is instant — zero CPU involvement.

On our no-atomics platform (Westmere PCIe 2.0), signal completion uses
the **bounce buffer path**:

1. GPU writes RPTR to system memory (PCIe posted write)
2. CPU polls RPTR via `ProcessAllBounceBuffers` → `ProcessCompletions`
3. `ProcessCompletions` calls `UpdateReadDispatchId` to convert RPTR
4. When `read_dispatch_id >= complete_idx`, CPU decrements signal
5. CLR's `WaitForSignal` loop detects signal == 0 and returns

The bottleneck: **step 2 runs in a tight poll loop** inside
`BusyWaitSignal::WaitRelaxed`. Each iteration calls
`ProcessAllBounceBuffers` for ALL active queues. With 64-entry signal
pool wrapping every 64 dispatches, `WaitCurrent` triggers ~14 times
per 900-dispatch forward pass. Each `WaitCurrent` spins in the poll
loop until the signal drops to 0.

The poll loop itself is fast, but `UpdateReadDispatchId` reads from
UC memory via PCIe, which has ~1-2μs latency per read. In a tight
loop, this serializes to thousands of PCIe reads per second instead
of actual GPU compute.

## Approaches

### 1. Interrupt-based signal completion (recommended)

Instead of busy-waiting in `WaitRelaxed`, use KFD's interrupt mechanism.
The GPU can fire an interrupt when a dispatch completes (EOP event).
ROCR already has `InterruptSignal` which uses `hsaKmtWaitOnEvent`.

**Approach:** Force interrupt-based signals for all dispatches on
no-atomics platforms. This avoids the poll loop entirely — the CPU
sleeps until the GPU signals completion via interrupt.

**Risk:** Interrupt latency (~10-50μs) may add overhead per dispatch.
But 900 dispatches × 50μs = 45ms per forward pass, which is acceptable.

### 2. Batch signal completion

Instead of waiting for each signal individually, submit all 900
dispatches with signal only on the LAST one. The intermediate
dispatches run without completion tracking.

**Approach:** In CLR's `ActiveSignal`, return `{0}` (no signal) for
most dispatches, only attach a real signal every N dispatches or on
the last dispatch of a graph.

**Risk:** CLR uses signal completion for queue flow control. Skipping
signals may cause queue overflow.

### 3. EOP fence-based completion

Use PM4 RELEASE_MEM with EVENT_INDEX(5) to write a completion token
to system memory at end of each dispatch. CPU polls this token instead
of RPTR. This is how the kernel driver tracks fence completion.

**Approach:** Allocate a per-queue completion buffer. Modify the AQL
dispatch to append a PM4 RELEASE_MEM that writes a monotonic value.
Poll the completion buffer instead of RPTR.

### 4. Bypass bounce buffer for synchronous dispatches

When `HIP_LAUNCH_BLOCKING=1` or CLR's `WaitCurrent` is called, skip
the bounce buffer path entirely. Instead, read RPTR directly and
compare against the expected value.

## Success Criteria

- [ ] SmolLM2-135M generates tokens at ≥10 t/s on gfx803
- [ ] Qwen2.5-0.5B completes prompt eval within 30 seconds
- [ ] No correctness regressions (test_800_dispatch, test_ggml_sched)

## Packages

- `linux-lts-rocm-polaris` 6.18.16-34
- `hsa-rocr-polaris` 7.2.0-30
- `hip-runtime-amd-polaris` 7.2.0-32
- `llama-cpp-rocm-polaris` b7376-7
