# Phase 11: Close the ROCm-Vulkan Gap

## Status: WORKING — 72 t/s ROCm (86.5% of Vulkan)

## Changes

### Step 11a: Remove HDP flush from submitKernelInternal (hip-runtime-amd)
Removed `roc_device_.HdpFlushWait()` from entry of submitKernelInternal.
Compute kernels read VRAM (weights) and UC kernarg. Neither goes through HDP.
Blit queues still get HDP flush via ROCR StoreRelaxed (size ≤ 256 check).

**Impact:** +22% generation speed (59 → 72 t/s).

### Step 11b: Remove debug instrumentation (hip-runtime-amd)
Removed Phase 7h-debug: no-op doorbell replacement and AQL packet
verification (volatile reads of kernarg_address/header before every doorbell).
Root cause found and fixed in kernel TC_CFG L2 policy.

### Step 11c: Phase 7c force sync marker — SKIPPED
llama.cpp uses hipStreamSynchronize, not markers. submitMarker is never
called during inference. No impact.

### Step 11d: Event wait timeout tuning — SKIPPED
hsaKmtWaitOnEvent_Ext takes ms granularity (min 1ms). Interrupts fire
within ~370µs average, well before timeout. No improvement at this API level.

### Step 11e: Vulkan package
Created `llama-cpp-vulkan-polaris` with `-DGGML_VULKAN=ON`.

## Benchmarks (SmolLM2-135M-Instruct Q8_0)

| Metric | Phase 10 | Phase 11 | Vulkan | ROCm/Vulkan |
|--------|----------|----------|--------|-------------|
| Generation | 59.05 t/s | 72.05 t/s | 83.3 t/s | 86.5% |
| Prompt eval | 125 t/s | 143 t/s | 848 t/s | 16.9% |

## Stability
- 499 tokens generated at 66.5 t/s — no faults, no hangs
- Slight t/s decrease on longer sequences (KV cache growth)

## Packages
- `hip-runtime-amd-polaris` 7.2.0-37
- `hsa-rocr-polaris` 7.2.0-34
- `llama-cpp-vulkan-polaris` b7376-1 (NEW)
