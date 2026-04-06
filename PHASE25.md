# Phase 25: 5 t/s on Largest Possible Model (CPU+GPU)

## Status: In Progress

## Goal

Reach 5 t/s sustained generation on the largest possible model using
dual Xeon X5650 (188 GB DDR3) + Radeon Pro WX 2100 (2 GB GDDR5).

**Primary candidate:** Qwen3.5-35B-A3B Q4_K_M — 35B MoE, ~2.6B active/token,
~1.43 GB active weight read per token.

## Hardware

- **CPU**: Dual Xeon X5650 (Westmere, 6 cores/socket, 12 total, SSSE3/SSE4.2, NO AVX)
- **RAM**: 188 GB DDR3-1333 (3ch/socket, ~25.6 GB/s/socket, ~51 GB/s aggregate)
- **QPI**: 6.4 GT/s (~12.8 GB/s between sockets)
- **GPU**: Radeon Pro WX 2100 (Polaris 12, gfx803, 2 GB GDDR5, 48 GB/s)
- **PCIe**: 2.0 x16 (~5 GB/s practical)
- **NUMA**: 2 nodes (96 GB each), distance 10/20

## Model Architecture (Qwen3.5-35B-A3B)

- 35B total params, ~2.6B active per token (7.4% activation)
- 40 layers: 30 GDN (linear attention) + 10 full attention (full_attention_interval=4)
- 256 experts, 8 routed + 1 shared active per token
- moe_intermediate_size=512, hidden_size=2048
- Per expert: 3 × (2048 × 512) = 3.15M params → ~1.73 MB at Q4_K_M
- Built-in MTP head (mtp_num_hidden_layers=1) — not supported for MoE in llama.cpp
- KV per token: 10 attn layers × 2 heads × 256 dim × 4 bytes = 20 KB

## Phase A: Baseline Smoke Test

### A1: Single node 1, 6 threads (Phase 23 analogue)

```bash
VK_ICD_FILENAMES=/dev/null numactl --membind=1 --cpunodebind=1 \
  llama-completion -t 6 --no-mmap -ngl 0 -c 512 -n 64 --simple-io -no-cnv \
  -m /home/llm/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -p "The theory of general relativity"
```

**Results:**

| Metric | Value |
|--------|-------|
| Prompt eval | **9.76 t/s** (102.43 ms/tok) |
| Generation | **4.93 t/s** (202.91 ms/tok) |
| Graph splits | 1 |
| Load time | 47.8 s |
| RAM usage | 21,555 MiB |
| Output quality | Coherent (relativity → spacetime curvature → Einstein field equations) |

**4.93 t/s on 6 cores — 1.4% below target!**

Effective throughput: 1.43 GB × 4.93 = **7.05 GB/s** (27.5% of 25.6 GB/s DDR3 peak).
Same utilization as 122B (26%) — access pattern bottleneck persists but active read is 2.2× smaller.

## Phase B: NUMA Strategy Sweep
