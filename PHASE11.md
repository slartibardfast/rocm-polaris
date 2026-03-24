# Phase 11/12: Close the ROCm-Vulkan Gap

## Status: 70 t/s ROCm, 118 t/s Vulkan — structural PCIe floor reached

## Phase 11: Remove Per-Dispatch Overhead

### Step 11a: Remove HDP flush from submitKernelInternal (hip-runtime-amd)
Removed `roc_device_.HdpFlushWait()` from entry of `submitKernelInternal`.
Compute kernels read VRAM (weights) and UC kernarg (fine-grain pool).
Neither goes through HDP. Blit queues still get HDP flush via ROCR
StoreRelaxed (queue size ≤ 256 check).

**Impact:** +22% generation speed (59 → 72 t/s). Largest single improvement
in this phase — removed 900 KFD ioctls per forward pass.

### Step 11b: Remove debug instrumentation (hip-runtime-amd)
Removed Phase 7h-debug: no-op doorbell replacement and AQL packet
verification (volatile reads of kernarg_address/header before every
doorbell). Root cause found and fixed in kernel TC_CFG L2 policy.

### Step 11c: Phase 7c force sync marker — SKIPPED
llama.cpp uses `hipStreamSynchronize`, not markers. `submitMarker` is
never called during inference. No performance impact.

### Step 11d: Event wait timeout tuning — SKIPPED
`hsaKmtWaitOnEvent_Ext` takes millisecond granularity (min 1ms). Interrupts
fire within ~370µs average, well before timeout. No improvement at this API.

### Step 11e: Vulkan package — SHIPPED
Created `llama-cpp-vulkan-polaris` b7376-1 with `-DGGML_VULKAN=ON`.
118 t/s generation, 180 t/s prompt eval out of the box.

## Phase 12: Find the Remaining Gap

### Discovery: ROCR async event threads burning 2 CPU cores
`perf` profiling revealed 224% CPU utilization during inference — two
ROCR async event threads (queue error handlers) in tight polling loops.
On no-atomics, queue error signals are `BusyWaitSignal` (no EOP event),
so `AsyncEventsLoop` falls back to CPU polling instead of interrupt wait.

**Fix:** Added `usleep(1000)` to the polling fallback path in
`Runtime::AsyncEventsLoop`. CPU usage dropped from 224% to 5%.
Generation speed unchanged (70 t/s) — the overhead was pure waste.

### Experiment: Remove kernarg readback — NO EFFECT
Hypothesis: AQL packet readback (`pkt[15]`) in ROCR already commits all
prior UC writes via IOH ordering. CLR kernarg readback is redundant.

Result: 69.4 t/s vs 70.5 baseline. No improvement. The readback was
already fast (~0 latency) because sfence + subsequent code provides
enough delay for IOH to commit. Reverted — no gain, added risk.

### Experiment: Larger signal pool (64→1024) — NO EFFECT
Hypothesis: WaitCurrent every 64 dispatches serializes CPU/GPU. Larger
pool = fewer stalls = more overlap.

Result: 69.5-69.9 t/s across all pool sizes. No improvement. The GPU
processes dispatches faster than the CPU submits them, so WaitCurrent
returns instantly at pool=64 already. System is GPU-bound, not
CPU-serialization-bound.

### Root Cause: GPU-Side PCIe Latency (Structural Floor)

The remaining 70→118 t/s gap is **GPU-side overhead** from PCIe 2.0:

| Per-dispatch overhead | ROCm (system memory) | Vulkan (VRAM) |
|-----------------------|----------------------|---------------|
| AQL/command fetch | ~500ns (PCIe read) | ~100ns (local) |
| Kernarg/push constant | ~500ns-2µs (PCIe) | ~100ns (local) |
| Signal processing | ~1µs (sentinel IH write) | 0 (batch semaphore) |
| Inter-dispatch gap | ~1µs (CP pipeline stall) | ~0.5µs |
| **Total per dispatch** | **~3-6µs** | **~0.5-1µs** |
| **900 dispatches** | **~3-5ms** | **~0.5-1ms** |

The AQL ring buffer and kernarg pool are in system memory (UC) because
gfx803/no-atomics requires CPU-visible bounce buffers. Vulkan command
buffers live in VRAM (local GDDR5 at 25.6 GB/s). This is a fundamental
architectural difference — per-dispatch ROCm on PCIe 2.0 Westmere
cannot match Vulkan's VRAM-local command path.

## Final Benchmarks (SmolLM2-135M-Instruct Q8_0, ngl=99, fa=off)

| Metric | Phase 10 | Phase 11/12 | Vulkan | ROCm/Vulkan |
|--------|----------|-------------|--------|-------------|
| Generation | 59.05 t/s | **70 t/s** | **118 t/s** | 59.3% |
| Prompt eval | 125 t/s | **143 t/s** | **180 t/s** | 79.4% |
| CPU usage | 224% | **5%** | ~100% | |
| ms/token | 16.93 | **14.2** | **8.5** | |

## Stability
- 499 tokens at 66.5 t/s — zero GPU faults, zero hangs
- Clean dmesg after all tests
- Slight t/s decrease on long sequences (KV cache growth, expected)

## Packages
- `hsa-rocr-polaris` 7.2.0-36 (Phase 12 async thread fix)
- `hip-runtime-amd-polaris` 7.2.0-37 (Phase 11a+11b HDP/debug removal)
- `llama-cpp-vulkan-polaris` b7376-1 (NEW — Vulkan backend)
- `llama-cpp-rocm-polaris` b7376-7 (unchanged)

## What Would Close the Gap Further

1. **Move AQL ring to VRAM** — eliminates PCIe fetch latency for packets.
   Requires kernel/ROCR changes to map ring in GPU-visible VRAM with
   CPU write access via BAR aperture. Complex, high impact.

2. **Move kernarg to VRAM** — eliminates PCIe fetch for kernel arguments.
   Requires GPU-side kernarg allocator + CPU BAR writes. Same complexity.

3. **Dispatch batching** — write N packets with 1 doorbell. Reduces CP
   pipeline stalls. Requires CLR deferred-doorbell mode. Medium complexity.

4. **Remove sentinel signal** — stop replacing completion_signal with
   sentinel. Save CP signal processing overhead. Requires alternative
   completion detection (pure RPTR-based, no interrupt). Breaks Phase 10.
