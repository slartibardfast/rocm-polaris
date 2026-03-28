# Phase 15: CPU Maximum Effort + VRAM Discovery

## Status: COMPLETE — CPU +109%, SmolLM2 Vulkan +26%, VRAM architecture revealed

## Major Discovery: Model Is NOT In Physical VRAM

During investigation of the Vulkan "slow band," we discovered that
RADV (Mesa's Vulkan driver) places ALL allocations in system RAM via
GART on small-BAR GPUs like Polaris 12. Despite requesting `DEVICE_LOCAL`
memory and reporting 2GB VRAM, the kernel shows only **18.5 MB of
physical VRAM used** during inference — just GPU page tables.

The entire 1.2GB model, 51MB KV cache, 19MB SSM state, and 493MB
compute buffer reside in host system RAM (DDR3-1333, ~32 GB/s per
socket). The GPU accesses them through PCIe 2.0 x16 (8 GB/s).

### Why We Get 23 t/s on 8 GB/s PCIe

23 t/s × 1.27 GB/token = 29 GB/s — 3.6x the PCIe bandwidth. This is
possible because:

1. **GPU L2 cache amplification**: Polaris 12 has 512KB L2 cache (TCC).
   Q4_K weight blocks are 256 bytes. The L2 caches recently-accessed
   blocks, and the 64-wide wavefront reuses each block across all lanes.

2. **Quantized weights are small**: Q4_K_M at 5.4 bits per weight means
   less data to read per matmul than FP16.

3. **Batched PCIe reads**: The GPU's memory controller coalesces PCIe
   read requests, achieving higher effective bandwidth than raw latency
   would suggest.

### Why The Slow Band Exists

The "slow band" at 1748-1767 MB total mapped memory (25-44 MB from the
GART limit) is **GART page table TLB pressure**:

- The GPU TLB covers a fixed number of pages
- When the total mapped VA space crosses ~1748 MB, TLB coverage drops
- More TLB misses → more PCIe round-trips for page table walks
- Each page table walk costs ~500ns (PCIe non-posted read)
- At 1767+ MB, the mapping overflows to a different GART region with
  better TLB coverage (likely different page table level)

### What This Means

Our entire bandwidth analysis was wrong:

| What We Assumed | Reality |
|----------------|---------|
| Model in VRAM (48 GB/s) | Model in system RAM (8 GB/s PCIe) |
| Bandwidth-limited at 67% | PCIe-limited with 3.6x L2 amplification |
| 38 t/s theoretical max | 6.3 t/s raw PCIe, 23 t/s with caching |
| VRAM slow band from fragmentation | GART TLB pressure at specific VA sizes |

### Tested and Ruled Out

- `--no-mmap`: still 18.5 MB VRAM (RADV choice, not mmap-related)
- `--no-host`: still 18.5 MB VRAM
- `--no-mmap --no-host`: still 18.5 MB VRAM
- `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1`: env var doesn't exist in b8508
- Kernel TTM contiguous-or-GTT: made slow band wider
- Kernel TTM headroom: no effect (kernel sees only 18.5 MB)
- Vulkan memory type skip: no effect
- VRAM padding (0-50 MB): no effect

**RADV on small-BAR Polaris always uses GART.** This is a driver
architecture decision, not a configuration choice.

### Path to True VRAM

To get model weights into physical GDDR5 (48 GB/s), we would need to:
1. Modify RADV's memory allocator to prefer VRAM over GART on small-BAR
2. Or use a different Vulkan ICD (AMDVLK) that may behave differently
3. Or use ROCm/HIP which DOES use physical VRAM (our 75 t/s SmolLM2)

ROCm inference DOES use physical VRAM — that's why SmolLM2 at 135MB
fits easily and runs at 75 t/s (bandwidth-limited at GDDR5 speed).
The Qwen3.5-2B (1.2GB) doesn't fit in VRAM for ROCm because the SSM
kernels don't work on HIP/gfx803.

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
  precision, 2.8x faster than libm, zero perplexity impact

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

## Vulkan Slow Band (Understood, Not Fixed)

| Total GPU Mapped | Free GART | Speed | Cause |
|-----------------|-----------|-------|-------|
| < 1748 MB | > 44 MB | 23 t/s | TLB covers working set |
| 1748-1767 MB | 25-44 MB | 13 t/s | TLB pressure zone |
| > 1767 MB | < 25 MB | 23 t/s | Different GART region |

**Workaround**: Use F16 KV (default) for ctx 5120-24576 — all fast.
Or use ctx ≥ 10240 with any KV type.

## Packages

- `linux-lts-rocm-polaris` 6.18.16-40 (TTM fix reverted — not the cause)
- `llama-cpp-vulkan-polaris` b8508-7 (Phase 15 CPU optimizations)
- `llama-cpp-rocm-polaris` b8508-3 (Phase 15 CPU optimizations)
