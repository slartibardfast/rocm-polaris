# Phase 16: Close the GPU Efficiency Gap

## Status: IN PROGRESS — profiling complete, fused SSM shader needed

## GPU Profiling Results (Qwen3.5-2B, per token)

Total eval: 85ms (23 t/s). GPU time: 67ms. CPU overhead: 18ms.

### Where 67ms of GPU Time Goes

| Category | Dispatches | Total µs | % GPU | Per-dispatch µs |
|----------|-----------|----------|-------|-----------------|
| **SCALE** | **36** | **20,800** | **31%** | **578** |
| **CPY** | **37** | **17,500** | **26%** | **473** |
| Output logits (q6_K 248K) | 1 | 10,100 | 15% | 10,100 |
| MUL_MAT_VEC (FFN/attn) | ~150 | 30,000 | 45% | 200 |
| GET_ROWS | 38 | 8,100 | 12% | 214 |
| GATED_DELTA_NET (fused SSM) | 18 | 2,000 | 3% | 111 |
| SILU | 36 | 1,500 | 2% | 42 |
| Flash attention | 6 | 1,200 | 2% | 200 |
| SIGMOID/SOFTPLUS/L2_NORM/etc | 96 | 1,500 | 2% | 16 |

### The Problem: 267 Tiny Dispatches

The fused SSM kernel (gated_delta_net.comp) is FAST — 2ms for 18 layers.
But 267 surrounding dispatches (SCALE, CPY, GET_ROWS, SIGMOID, SOFTPLUS,
SILU, L2_NORM, MUL) cost 38ms in GCN wave launch overhead.

Each dispatch on 5-CU Polaris 12 costs ~500µs regardless of tensor size.
A 2048-element SCALE takes 0.3µs of compute but 500µs of launch overhead.
**99.9% overhead, 0.1% compute.**

### CPU Overhead: 18ms

`perf` profiling revealed 40% of CPU time is a PAUSE spin loop polling
`getFenceStatus()`. Replaced with `vkWaitForFences` (kernel futex).
No speed improvement (GPU-bound) but frees CPU cycles.

The remaining CPU overhead is command buffer recording + descriptor
updates — happens serially before GPU starts.

## Fix: Extended Fused SSM Shader

### Current Architecture (per SSM layer)
```
dispatch: sigmoid(beta)           ~3µs compute, ~500µs launch
dispatch: softplus(alpha)         ~6µs compute, ~500µs launch
dispatch: mul(gate)               ~8µs compute, ~500µs launch
dispatch: ssm_conv                ~50µs compute, ~500µs launch
dispatch: silu(conv_output)       ~40µs compute, ~500µs launch
dispatch: l2_norm(q)              ~3µs compute, ~500µs launch
dispatch: l2_norm(k)              ~3µs compute, ~500µs launch
dispatch: GATED_DELTA_NET         ~111µs (fused, efficient)
dispatch: cpy(conv_state)         ~470µs (state copy)
dispatch: cpy(ssm_state)          ~470µs (state copy)
dispatch: rms_norm_mul            ~65µs compute, ~500µs launch
dispatch: silu(z)                 ~40µs compute, ~500µs launch
dispatch: mul(gated_output)       ~8µs compute, ~500µs launch
dispatch: scale(?)                ~5µs compute, ~500µs launch
... ≈ 15 dispatches per SSM layer × 18 layers = 270 dispatches
```

### Proposed: Single Fused SSM Layer Dispatch
```
dispatch: FUSED_SSM_LAYER         ~300µs (all ops combined)
... × 18 layers = 18 dispatches
```

**Savings: 252 dispatches eliminated. At 500µs each = 126ms saved.**
Even conservatively (some ops can't fuse): 150 dispatches eliminated =
75ms saved → eval drops from 85ms to ~30ms → **33 t/s**.

### Implementation Path

1. Write `fused_ssm_layer.comp` that combines:
   - sigmoid + softplus + gate_mul
   - ssm_conv + silu
   - l2_norm (q, k)
   - state copy (conv + ssm)
   - rms_norm_mul + silu + output_mul

2. Register the fused shader in ggml-vulkan dispatch logic

3. Modify qwen35.cpp to emit a single FUSED_SSM_LAYER op instead
   of 15 individual ops per SSM layer

4. Test harness: compare output of fused vs unfused (bit-exact)

5. A/B benchmark: measure dispatch count + eval time

### Files to Modify

- New: `vulkan-shaders/fused_ssm_layer.comp`
- Modify: `ggml-vulkan.cpp` (dispatch logic for fused op)
- Modify: `src/models/qwen35.cpp` (emit fused op)
- Modify: `ggml.h` / `ggml.c` (add GGML_OP_FUSED_SSM_LAYER)

### Packages

- `llama-cpp-vulkan-polaris` b8508-9 (fence fix + nodes_per_submit)
