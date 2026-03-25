# Benchmarks: LLM Inference on Radeon Pro WX 2100

**Hardware:** Radeon Pro WX 2100 (Polaris 12, gfx803, 2GB GDDR5)
**Host:** Dual Xeon X5650 (Westmere), no PCIe atomics, PCIe 2.0
**Kernel:** linux-lts-rocm-polaris 6.18.16-39
**llama.cpp:** b8508 (ROCm + Vulkan builds)
**Date:** 2026-03-25

## Headline Numbers

| Model | Params | Context | Backend | Gen t/s |
|-------|--------|---------|---------|---------|
| SmolLM2-135M Q8_0 | 135M | 4K | ROCm | **75** |
| SmolLM2-135M Q8_0 | 135M | 4K | Vulkan | **93** |
| Qwen3.5-0.8B Q4_K_M | 752M | 4K | Vulkan | **44** |
| **Qwen3.5-2B Q4_K_M** | **1.9B** | **16K** | **Vulkan** | **23** |
| Qwen3-1.7B Q4_K_M | 1.7B | 4K | Vulkan | **27** |

## Full Results

### SmolLM2-135M-Instruct Q8_0 (139MB)

| Backend | ngl | Gen t/s | PP t/s | ms/token |
|---------|-----|---------|--------|----------|
| ROCm b8508 | 99 | 75.2 | 146 | 13.3 |
| Vulkan b8508 | 99 | 93.5 | 152 | 10.7 |

### Qwen3.5-0.8B Q8_0 (764MB) — SSM Hybrid

| Backend | ngl | Gen t/s | PP t/s | Notes |
|---------|-----|---------|--------|-------|
| Vulkan | 99 | 35.4 | 145 | |
| ROCm | 99 | HANG | — | SSM kernels incompatible with gfx803 HIP |

### Qwen3.5-0.8B Q4_K_M (508MB) — SSM Hybrid

| Backend | ngl | Gen t/s | PP t/s | Notes |
|---------|-----|---------|--------|-------|
| Vulkan | 99 | **43.8** | 24 | Best small Qwen3.5 config |
| ROCm | 99 | HANG | — | SSM kernels incompatible |

### Qwen3.5-2B Q4_K_M (1.2GB) — SSM Hybrid, Target Model

| Backend | Context | ngl | Gen t/s | PP t/s | VRAM Used |
|---------|---------|-----|---------|--------|-----------|
| Vulkan | 4K | 25 | 14.4 | 31 | ~1900MB |
| Vulkan | 8K | 25 | **23.1** | — | ~1900MB |
| Vulkan | **16K** | **25** | **23.1** | **50** | **~1900MB** |
| Vulkan | 32K | 25 | 10.4 | — | ~1900MB |
| Vulkan | 64K | 25 | 10.4 | — | ~1900MB |
| Vulkan | 128K | 25 | 10.4 | — | ~1900MB |
| Vulkan | 4K | 0 | 3.2 | 4.8 | CPU only |
| ROCm | any | any | HANG | — | SSM kernels incompatible |

**Sweet spot: 8K-16K context at 23 t/s** with all 25 layers on GPU.

The 4K→8K speed jump (14→23 t/s) is from the Vulkan scheduler finding
a better compute buffer split. The 16K→32K drop (23→10 t/s) is KV cache
(384MB) crowding the compute buffer.

Qwen3.5's SSM hybrid architecture enables massive context: only 6 of 24
layers use KV cache (attention layers). The other 18 use fixed-size SSM
state (19MB RS buffer). 128K context still fits in 2GB VRAM.

### Qwen3-1.7B Q4_K_M (1.1GB) — Pure Transformer

| Backend | ngl | Gen t/s | PP t/s |
|---------|-----|---------|--------|
| ROCm b8508 | 99 | 11.7 | 6.3 |
| Vulkan b8508 | 99 | 26.6 | 11.5 |

### Qwen3.5-2B Q4_K_M — GPU Layer Scaling (Vulkan, ctx=4K)

| ngl | Layers on GPU | Gen t/s |
|-----|--------------|---------|
| 0 | 0 | 3.2 |
| 7 | 7 | 4.0 |
| 14 | 14 | 5.6 |
| 20 | 20 | 8.3 |
| 24 | 24 | 14.6 |
| 25 | 25 | 14.3 |

## Key Findings

1. **llama.cpp b8508 improved ROCm SmolLM2 from 70→75 t/s** — free
   upgrade from newer ggml-hip dispatch code.

2. **Qwen3.5 SSM architecture doesn't work on ROCm gfx803** — the fused
   Gated Delta Net kernels hang. Vulkan works fine (shader compilation
   handles SSM ops correctly).

3. **Qwen3.5-2B fits in 2GB with 16K context at 23 t/s** — the SSM
   hybrid architecture has tiny KV cache (6 attention layers / 24 total).
   128K context still fits but drops to 10 t/s.

4. **Vulkan is 1.2-2.3x faster than ROCm** across all models. The gap
   is largest on bigger models (2.3x on Qwen3-1.7B) due to Vulkan's
   batched command buffer dispatch.

5. **`-fit off` or context tuning needed** — b8508's autofit tries to
   keep 1024MB free, which is too conservative for 2GB VRAM. Use
   explicit `-ngl 25 -c 16384` for best Qwen3.5-2B performance.

## Packages

- `linux-lts-rocm-polaris` 6.18.16-39
- `hsa-rocr-polaris` 7.2.0-34 (Phase 10 shared event)
- `hip-runtime-amd-polaris` 7.2.0-37 (Phase 11 HDP removal)
- `llama-cpp-rocm-polaris` b8508-1
- `llama-cpp-vulkan-polaris` b8508-1
