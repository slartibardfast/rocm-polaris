# Benchmarks: LLM Inference on Radeon Pro WX 2100

**Hardware:** Radeon Pro WX 2100 (Polaris 12, gfx803, 2GB GDDR5)
**Host:** Dual Xeon X5650 (Westmere), no PCIe atomics, PCIe 2.0
**Kernel:** linux-lts-rocm-polaris 6.18.16-39
**llama.cpp:** b8508 (ROCm + Vulkan builds)
**Date:** 2026-03-26

## Recommended Configs

### Coding Assistant / Autocomplete
```
llama-completion -m Qwen3.5-2B-Q4_K_M.gguf -ngl 25 -fa on -c 8192 -fit off -ctk q8_0 -ctv q8_0
```
- **23 t/s** initial, **14 t/s** average at full fill
- 8K context = plenty for file + imports + conversation
- **STABILITY PROVEN:** 8,191 tokens, 10 min, exit 0, zero faults

### Long Conversation / Document Analysis
```
llama-completion -m Qwen3.5-2B-Q4_K_M.gguf -ngl 25 -fa on -c 32768 -fit off -ctk q8_0 -ctv q8_0
```
- **23 t/s** initial, **10.9 t/s** average at full fill
- 32K context = ~50 pages of text
- **STABILITY PROVEN:** 32,000 tokens, 50 min, exit 0, zero faults

## Qwen3.5-2B Context Scaling (Q8_0 KV, ngl=25, Vulkan)

### Speed vs Context Allocation (10-token test, empty context)

| Context | KV Cache | Initial t/s | Band |
|---------|----------|-------------|------|
| 2K-45K | 12-280MB | **23 t/s** | FAST |
| 49K-127K | 306-792MB | **9.5 t/s** | SLOW |
| 131K-196K | 816-1224MB | **23 t/s** | FAST |
| 262K | 1632MB | **9.4 t/s** | SLOW |

Speed bands are from Vulkan scheduler allocation decisions. The 131K-196K
recovery is the scheduler finding an efficient memory split.

### Speed vs Context Fill (generation degradation)

**8K context:**

| Tokens Generated | Fill % | Gen t/s |
|-----------------|--------|---------|
| 100 | 1% | 23.1 |
| 500 | 6% | 22.6 |
| 8,191 (full) | 100% | 14.0 avg |

**32K context:**

| Tokens Generated | Fill % | Gen t/s |
|-----------------|--------|---------|
| 100 | 0.3% | 23.2 |
| 500 | 1.5% | 22.9 |
| 2,000 | 6% | 22.0 |
| 5,000 | 15% | 20.1 |
| 10,000 | 31% | 17.4 |
| 32,000 (full) | 100% | 10.9 avg |

## All Models

| Model | Params | Quant | Backend | ngl | Gen t/s | Notes |
|-------|--------|-------|---------|-----|---------|-------|
| SmolLM2-135M | 135M | Q8_0 | ROCm | 99 | **75** | Benchmark model |
| SmolLM2-135M | 135M | Q8_0 | Vulkan | 99 | **93** | |
| Qwen3.5-0.8B | 752M | Q4_K_M | Vulkan | 99 | **44** | SSM hybrid |
| Qwen3.5-0.8B | 752M | Q8_0 | Vulkan | 99 | **35** | |
| **Qwen3.5-2B** | **1.9B** | **Q4_K_M** | **Vulkan** | **25** | **23** | **Target model** |
| Qwen3-1.7B | 1.7B | Q4_K_M | ROCm | 99 | **12** | Pure transformer |
| Qwen3-1.7B | 1.7B | Q4_K_M | Vulkan | 99 | **27** | |
| Qwen3.5-2B | 1.9B | Q4_K_M | ROCm | any | HANG | SSM kernels broken |

## Key Findings

1. **Qwen3.5-2B at 23 t/s with 32K context on a 2GB GPU from 2017.**
   The SSM hybrid architecture (6 attention + 18 SSM layers) has tiny
   KV cache — only 6 layers need KV. 128K context fits in 2GB VRAM.

2. **Q8_0 KV saves 50% KV with zero speed loss.** F16 KV: 192MB at 16K.
   Q8_0 KV: 102MB. Same 23 t/s. Q4_0 KV halves speed (dequant overhead).

3. **llama.cpp b8508 improved ROCm SmolLM2 from 70→75 t/s** over b7376.

4. **Qwen3.5 SSM doesn't work on ROCm gfx803.** Fused Gated Delta Net
   kernels hang. Vulkan works fine.

5. **`-fit off` required.** b8508's autofit keeps 1GB free (too
   conservative for 2GB VRAM). Override with explicit ngl + context.

6. **Layer splits don't help.** Partial GPU offload (attention-only,
   ngl<25) is always slower than all-GPU due to PCIe 2.0 transfer overhead.
   The Vulkan scheduler's fast bands are specific to ngl=25.

## Packages

- `linux-lts-rocm-polaris` 6.18.16-39
- `hsa-rocr-polaris` 7.2.0-37
- `hip-runtime-amd-polaris` 7.2.0-37
- `llama-cpp-rocm-polaris` b8508-1
- `llama-cpp-vulkan-polaris` b8508-1
