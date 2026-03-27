# Phase 15: CPU Maximum Effort — SSSE3, SSE4.1, fast_expf, -march=native

## Status: COMPLETE — CPU-only +109%, SmolLM2 Vulkan +26%

## Discovery

Every previous build compiled for generic x86-64 (SSE2 only) because
Arch's `makepkg.conf` sets `-march=x86-64` which overrides cmake's
`GGML_NATIVE=ON`. The Q4_K dequant kernels fell through to SCALAR
GENERIC C code — no SIMD at all. ALL CPU code paths (sampling, graph
scheduling, token processing) ran at 2004-era instruction speed on a
2010 Westmere CPU with SSE4.2 support.

## Changes

### Build System
- `-march=native` override in CFLAGS/CXXFLAGS (replaces `-march=x86-64`)
- `-ffast-math -fno-finite-math-only` (enables auto-vectorization of
  reductions while preserving infinity/NaN support)
- `GGML_NATIVE=ON` (single-variant build optimized for host CPU)

### Hand-Vectorized Kernels (PKGBUILD injections)
- **SSSE3 Q4_K dot product** (`arch/x86/quants.c`): hand-written
  `_mm_maddubs_epi16` + `_mm_shuffle_epi8` kernel replacing the scalar
  generic fallback. Same algorithm as AVX path but with paired `__m128`
  accumulators instead of `__m256`.
- **SSE `ggml_vec_max_f32`** (`vec.h`): 16-wide unrolled `_mm_max_ps`
  with horizontal reduction. Replaces scalar conditional max loop that
  GCC cannot auto-vectorize.
- **SSE4.1 `ggml_vec_argmax_f32`** (`vec.h`): simultaneous max + index
  tracking via `_mm_blendv_ps`. Branch-free conditional update. Used
  for sampling over 248K vocabulary.
- **Inline `ggml_fast_expf`** (`ops.cpp`): range reduction + 4th-order
  minimax polynomial. Replaces opaque `expf()` library call in CPU
  softmax. ~23-bit accuracy, fully inlineable.

### Vulkan Memory Fixes
- **Visible VRAM memory type skip**: prevents compute buffers from
  landing on the 256MB BAR-mapped visible VRAM heap (slow PCIe access).
  Filters `HOST_VISIBLE` heaps <1GB from `DEVICE_LOCAL`-only requests.
- **VRAM slow band pad**: pre-compute allocation pad that shifts heap
  boundary when predicted post-compute free VRAM lands in the 260-285MB
  danger zone. Partially effective — the slow band at 7168-9216 context
  persists for reasons deeper in RADV's memory manager.

## Results

| Model | Backend | Before | After | Change |
|-------|---------|--------|-------|--------|
| SmolLM2-135M | Vulkan | 93 t/s | **117 t/s** | **+26%** |
| SmolLM2-135M | ROCm | 75 t/s | **77 t/s** | +3% |
| Qwen3.5-2B 32K | Vulkan | 23 t/s | **23 t/s** | — |
| Qwen3.5-0.8B | Vulkan | 44 t/s | **43 t/s** | — |
| Qwen3-1.7B | Vulkan | 27 t/s | **26 t/s** | — |
| Qwen3-1.7B | ROCm | 12 t/s | **12 t/s** | — |
| CPU-only Qwen3.5-2B | — | 1.84 t/s | **3.85 t/s** | **+109%** |

### SIMD Instruction Count in Binary

| Instruction Class | Before (generic x86-64) | After (-march=native) |
|-------------------|------------------------|----------------------|
| SSSE3 (pmaddubsw, pshufb) | 0 | **696** |
| SSE4.1 (blendvps, pmaxsd) | 0 | **206** |
| SSE (maxps, addps, mulps) | 5045 | **1825** |
| AVX | 0 | 0 |

The SSE count dropped because the multi-variant dispatch (which compiled
multiple paths) was replaced by a single native-optimized path.

## Why Small Models Benefit Most

Phase 15 optimizes CPU-side overhead (sampling, graph scheduling, token
processing). For small models, this overhead is a larger fraction of
total token time:

- **SmolLM2-135M**: 8.5ms/token GPU + ~2ms/token CPU overhead = 26% CPU
- **Qwen3.5-2B**: 43ms/token GPU + ~2ms/token CPU overhead = 4% CPU

The 26% speedup on SmolLM2 Vulkan comes from eliminating that 2ms CPU
overhead (now <0.5ms with SIMD). Large models are GPU-bound — CPU
optimization has minimal impact.

## Vulkan Speed Band (Unresolved)

A 13MB "slow band" persists at 7168-9216 context for Qwen3.5-2B:
- Total GPU allocation 1768-1781 MB → 13 t/s (vs 23 t/s outside)
- Root cause is deeper in RADV/Mesa memory management, not visible VRAM spill
- The band shifted with `-march=native` (was at 49K-127K before)
- Workaround: use ctx=10240+ (naturally avoids the band)

## Packages

- `llama-cpp-vulkan-polaris` b8508-6
- `llama-cpp-rocm-polaris` b8508-3
