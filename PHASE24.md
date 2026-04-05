# Phase 24: Complete SSSE3/SSE4.1 SIMD Coverage for Westmere

## Goal

Eliminate all scalar fallback paths on Westmere X5650 (SSSE3 + SSE4.1, no AVX).
Every function that has an AVX or AVX2 path must have an SSSE3/SSE4.1 equivalent.

## Hardware

- Intel Xeon X5650 (Westmere-EP), dual socket
- ISA: SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2 — NO AVX
- 12 cores total, 12 MB L3 per socket
- Target models: Qwen3.5-122B-MoE (Q4_K), Nemotron-3-Nano-30B

## Prior Work (This Session)

20 SSSE3 vec_dot kernels already committed:
q4_0, q4_1, q5_0, q5_1, q8_0, q2_K, q3_K, q4_K, q5_K, q6_K,
iq2_xs, iq2_s, iq3_xxs, iq3_s, iq1_s, iq1_m, iq4_nl, iq4_xs, tq1_0
+ helpers: mul_add_epi8_sse(), get_scale_shuffle_ssse3(), hsum_float_4x4()

## Remaining Gaps

### Tier 1: Per-Token Hot Path (activation quantization)

These run on EVERY token, EVERY layer. Currently scalar on Westmere.

| # | Function | File | Line | Current Paths | Notes |
|---|----------|------|------|---------------|-------|
| 1 | `quantize_row_q8_0` | quants.c | 313 | AVX2 → scalar | float→int8 with rounding, stores d scale |
| 2 | `quantize_row_q8_1` | quants.c | 411 | AVX2 → scalar | Same + computes sum per block |
| 3 | `quantize_row_q8_K` | quants.c | 516 | AVX2 → scalar | K-quant variant, 256-element superblock |

### Tier 2: Dot Product Gaps

| # | Function | File | Line | Current Paths | Notes |
|---|----------|------|------|---------------|-------|
| 4 | `ggml_vec_dot_tq2_0_q8_K` | quants.c | 1555 | AVX2 → scalar | Ternary quant, AVX2-only |
| 5 | `ggml_vec_dot_iq2_xxs_q8_K` | quants.c | 3179 | AVX2 → AVX → scalar | Has AVX path to downlevel |
| 6 | `ggml_vec_dot_mxfp4_q8_0` | quants.c | 812 | AVX2 → AVX | MX floating point 4-bit |

### Tier 3: BF16 Support (only if BF16 models used)

| # | Function | File | Line | Current Paths | Notes |
|---|----------|------|------|---------------|-------|
| 7 | `ggml_cpu_bf16_to_fp32` | ggml-cpu.c | 3459 | AVX2 → scalar | Shift-left conversion |
| 8 | `ggml_vec_dot_bf16` | vec.cpp | 139 | AVX2/AVX → scalar | BF16 dot product |

### Not In Scope

- **WKV6/WKV7/GLA kernels** (ops.cpp): RWKV/GLA architectures only, not used
- **llamafile sgemm**: Returns false without AVX, standard path used instead
- **repack.cpp GEMV/GEMM 8x8**: All have `_generic` fallbacks, only activate with AVX2+
- **ggml-quants.c validation**: Cold path (model load), scalar is fine
- **vec.h ggml_vec_add_f32**: Single AVX2 loop, auto-vectorizable scalar fallback
- **simd-mappings.h**: SSE3 SIMD macros already defined and working
- **simd-gemm.h**: Works with SSE3 at reduced tile size (RM=2, RN=2)

## Approach

Same as prior SSSE3 work: read AVX/AVX2 path, identify 256-bit ops, split to
128-bit equivalents. For AVX2-only functions (no AVX path), write from scratch
using the AVX2 body as template. Build, test, commit after each function.

### Validation

- `quantize_row_q8_*`: Indirect — if quantization is wrong, all downstream
  dot products produce garbage. Run inference and check output coherence.
- `vec_dot_*`: `test-backend-ops test -o MUL_MAT -p 'type_a=<type>'` where
  Vulkan shader exists. Manual review where no shader exists (tq2_0).
- BF16: Test with a BF16 model if available.

## Results

| # | Function | Status | Commit | Tests |
|---|----------|--------|--------|-------|
| 1 | quantize_row_q8_0 | | | |
| 2 | quantize_row_q8_1 | | | |
| 3 | quantize_row_q8_K | | | |
| 4 | vec_dot_tq2_0 | | | |
| 5 | vec_dot_iq2_xxs | | | |
| 6 | vec_dot_mxfp4 | | | |
| 7 | bf16_to_fp32 | | | |
| 8 | vec_dot_bf16 | | | |
