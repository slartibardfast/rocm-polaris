# Phase 16: Updated Approach — Actual Profiling vs. Original Plan

## Profiling Reality Check

The original Phase 16 plan targeted SCALE (20.8ms) and CPY (17.5ms) based
on profiling done BEFORE Phase 15/16 optimizations. After applying
vkWaitForFences + nodes_per_submit=2000, those costs collapsed:

| Category | Old (pre-fix) | Current (b8508-12) | Change |
|----------|--------------|---------------------|--------|
| SCALE | 20.8ms (24%) | 1.0ms (2%) | -95% |
| CPY | 17.5ms (21%) | 1.3ms (3%) | -92% |
| CPU/sync | 17.8ms (21%) | ~0ms (overlap) | eliminated |

**The fence wait + batching fixes already solved the dispatch overhead
problem.** The remaining 47ms/token is dominated by matmul bandwidth.

## Actual Bottleneck Breakdown (b8508-12, Qwen3.5-2B Q4_K_M, ctx=8192)

| Op | Time | % | Count | Per-call |
|----|------|---|-------|----------|
| MUL_MAT q4_K m=6144 (FFN down/gate/up) | 10.7ms | 21.5% | 48 | 224µs |
| MUL_MAT q6_K m=248K (output logits) | 10.1ms | 20.3% | 1 | 10.1ms |
| MUL_MAT q5_K m=6144 | 6.2ms | 12.4% | 18 | 344µs |
| MUL_MAT q6_K m=2048 | 3.9ms | 7.8% | 12 | 324µs |
| FLASH_ATTN | 3.3ms | 6.5% | 6 | 544µs |
| MUL_MAT q4_K m=2048 | 2.7ms | 5.4% | 25 | 108µs |
| MUL_MAT q5_K m=2048 | 2.1ms | 4.2% | 18 | 115µs |
| GATED_DELTA_NET | 1.9ms | 3.9% | 18 | 107µs |
| GET_ROWS | 1.5ms | 2.9% | 38 | 39µs |
| CPY | 1.3ms | 2.7% | 37 | 36µs |
| SCALE | 1.0ms | 2.0% | 36 | 28µs |
| Everything else | 5.0ms | 10% | — | — |
| **Total** | **~50ms** | | **~340 dispatches** | |

## The Real Target: Matmul Bandwidth

**Matmul is 72% of token time.** At 23 t/s, we're reading ~1.2 GB of
weights per token. That's 23 × 1.2 = ~27.6 GB/s achieved vs 48 GB/s
GDDR5 theoretical = **57% bandwidth utilization**.

The gap has three possible causes:

### 1. Output logits matmul (10.1ms, 20% of total)
Single q6_K 248320×2048 matmul. This reads 248K × 2048 × 6.5/8 bytes
= ~400 MB for ONE dispatch. At 48 GB/s that should take ~8.3ms, we
measure 10.1ms = 82% utilization. Reasonable but could improve.

**Options:**
- Use Q4_K for output layer (saves ~35% bandwidth, ~3.5ms)
- Split into 2 dispatches for better CU scheduling
- Speculative decoding with smaller draft model (avoids output logits entirely)

### 2. FFN matmuls inefficiency
48 FFN dispatches at 224µs each = 10.7ms. These are q4_K m=6144 k=2048
= 6144 × 2048 × 4.5/8 = ~7 MB each. At 48 GB/s that's ~0.15ms minimum,
we measure 0.22ms = 68% utilization. The 32% overhead is dispatch +
L2 cache cold-start.

**Options:**
- Fuse gate+up matmuls (2 matmuls with same input → 1 wider matmul)
- Persistent kernel (one dispatch processes multiple layers)

### 3. q5_K and q6_K mixed quant overhead
The q5_K and q6_K layers use less efficient dequant than q4_K, adding
decode ALU cost. Requantizing to Q4_K_S would trade perplexity for speed.

## Actionable Targets (by expected impact)

### A. Output logits quantization (high impact, low risk)
Requantize the output layer from q6_K to q4_K. This is the single
largest op at 10.1ms. Expected savings: ~3ms → **26 t/s**.

Implementation: `--override-tensor output.weight=q4_0` or requantize
the GGUF file with different output layer quant.

### B. Speculative decoding (high impact, moderate risk)
Use SmolLM2-135M as draft model. Most tokens skip the 248K output logits
entirely. At 59 t/s for SmolLM2, draft overhead is minimal. Expected:
**30-35 t/s** effective throughput with 70% acceptance rate.

### C. KV cache type optimization (moderate impact, low risk)
`--cache-type-k q8_0 --cache-type-v q8_0` reduces KV cache VRAM,
allowing more context or reducing memory pressure on the 2GB GPU.

### D. Gate+up matmul fusion (moderate impact, high complexity)
Fuse the two 6144×2048 FFN matmuls (gate and up projections, same input)
into one 12288×2048 matmul. Saves 24 dispatches × ~150µs overhead =
~3.6ms. Requires model graph changes in qwen35.cpp.

### E. Persistent matmul kernel (high impact, very high complexity)
Single dispatch processes entire layer sequence. Eliminates ALL dispatch
overhead between matmuls. Academic-level optimization.

## Recommended Order

1. **A** — try `--override-tensor` flag first (zero code changes)
2. **B** — speculative decoding (command-line only if supported)
3. **C** — KV cache types (command-line)
4. **D** — gate+up fusion (code change, moderate)
5. Skip E unless 30+ t/s not reached by A-D

## Current State

- b8508-12 INSTALLED: fused gate_prep working, 23.12 t/s
- Fused gated_norm and dual_l2_norm: wired but not used in model graph
- SCALE + CPY: no longer bottlenecks (2% + 3% of time)
