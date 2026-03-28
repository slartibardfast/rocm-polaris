# Phase 15: CPU Maximum Effort + VRAM Investigation

## Status: COMPLETE — CPU +109%, SmolLM2 Vulkan +26%, VRAM confirmed correct

## VRAM Investigation: False Alarm → Confirmed Correct

### The Scare

Late in Phase 15, we discovered `mem_info_vram_used` showed only 18.5 MB
during "inference." We concluded the model was in system RAM via GART,
not physical VRAM. This triggered extensive investigation:

- Tested `--no-mmap`, `--no-host`, env vars — no change
- Built `VK_EXT_external_memory_host` interceptor
- Built standalone Vulkan test harness (`test_vk_vram.c`)
- Binary-searched all Vulkan extensions for GTT triggers
- Attempted kernel TTM patches (contiguous-or-GTT, 24MB headroom)
- Investigated RADV suballocator architecture

### The Resolution

**Continuous VRAM monitoring** during llama.cpp startup revealed the truth:

```
18 MB  → (idle)
1234 MB → (model weights loaded into physical VRAM)
1777 MB → (compute buffer allocated)
1236 MB → (compute freed after inference)
18 MB  → (process exited, everything freed)
```

**The model WAS in physical VRAM the entire time.** Our earlier sysfs
readings were taken AFTER the process exited. The `sleep 5` + `sleep 8`
delays were too long — llama-completion had already finished and freed
all VRAM.

### Confirmation

- `vkAllocateMemory` interceptor proved type 0 (heap 0, invisible VRAM)
  is selected for the 1020 MB model allocation
- Standalone test harness confirmed type 0 gives physical VRAM at all
  sizes (128 MB through 1536 MB)
- Binary search of all Vulkan extensions: ALL give physical VRAM
- The original 48 GB/s GDDR5 bandwidth analysis was CORRECT

### What This Means

| What We Feared | Reality |
|----------------|---------|
| Model in system RAM (8 GB/s) | Model in physical VRAM (48 GB/s) |
| 3.6x L2 cache amplification | Normal GDDR5 bandwidth |
| 23 t/s = PCIe-limited | 23 t/s = 67% bandwidth utilization |
| Potential 6x speedup from VRAM | Already at VRAM speed |

The 67% bandwidth utilization (23 t/s vs 38 t/s theoretical) is from
compute overhead, Vulkan dispatch, and the SSM architecture's non-linear
memory access patterns. NOT from PCIe bottleneck.

### Lesson Learned

**Always measure continuously, never after the fact.** A `sleep N` before
reading sysfs can miss the entire allocation lifecycle if the process
completes within N seconds. Use background monitoring with timestamps.

## CPU Optimizations (Working)

### Build System
- `-march=native` replaces `-march=x86-64` (was compiling for SSE2 only!)
- `-ffast-math -fno-finite-math-only` enables auto-vectorization
- `GGML_NATIVE=ON` for single-variant native build

### Hand-Vectorized Kernels
- **SSSE3 Q4_K dot product**: `_mm_maddubs_epi16` + `_mm_shuffle_epi8`
- **SSE `ggml_vec_max_f32`**: 16-wide unrolled `_mm_max_ps`
- **SSE4.1 `ggml_vec_argmax_f32`**: branch-free `_mm_blendv_ps`
- **Inline `ggml_fast_expf`**: 6th-order Horner polynomial, 22.8-bit
  precision (full f32), 2.8x faster than libm, zero perplexity impact

### Results

| Model | Backend | Before | After | Change |
|-------|---------|--------|-------|--------|
| SmolLM2-135M | Vulkan | 93 t/s | **117 t/s** | **+26%** |
| SmolLM2-135M | ROCm | 75 t/s | **77 t/s** | +3% |
| Qwen3.5-2B 32K | Vulkan | 23 t/s | **23 t/s** | — |
| CPU-only Qwen3.5-2B | — | 1.84 t/s | **3.85 t/s** | **+109%** |

### SIMD Instruction Count

| Class | Before | After |
|-------|--------|-------|
| SSSE3 (pmaddubsw, pshufb) | 0 | **696** |
| SSE4.1 (blendvps, pmaxsd) | 0 | **206** |

## Vulkan Slow Band

A 13 MB performance dip exists at specific total VRAM usage (1748-1767 MB)
where inference drops from 23 to 13 t/s. Root cause unknown — tested
kernel TTM patches, Vulkan memory type filtering, VRAM padding, AMD
overallocation extension. None fixed it.

**Workaround:** Use F16 KV (default) for ctx 5120-24576, or ctx ≥ 10240
with Q8_0 KV. Both avoid the band.

## Test Artifacts

- `tests/test_vk_vram.c` — standalone Vulkan VRAM placement test
- `/tmp/vk_intercept.so` — vkAllocateMemory interceptor (LD_PRELOAD)
- `/tmp/test_vk_ext_search.c` — extension binary search harness

## Packages

- `llama-cpp-vulkan-polaris` b8508-7 (Phase 15 CPU optimizations)
- `llama-cpp-rocm-polaris` b8508-3 (Phase 15 CPU optimizations)
