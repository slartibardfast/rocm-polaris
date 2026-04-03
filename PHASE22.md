# Phase 22: CPU Inference for Qwen3.5 (Speculative Decoding Target)

## Goal

Enable Qwen3.5 GDN models to run on CPU as the "target" model in a
GPU-draft / CPU-target speculative decoding pipeline.

- **Draft**: Qwen3.5-0.8B on GPU (Polaris 12, ~49 t/s)
- **Target**: Qwen3.5-9B/27B/35B-A3B on CPU (dual Xeon X5650, 96 GB)

## Status: In Progress

### What Works

- **Autoregressive (generation) path**: CORRECT
  - Greedy output (temp=0) is **identical** between GPU fused and CPU unfused
  - Verified on 0.8B with seed=42, 64 tokens
  - 9B produces coherent, high-quality output on CPU

- **CPU FUSED_GATE_PREP**: Implemented but guarded behind fused-only path
  - Unfused path uses separate ggml ops (add + softplus + mul) instead

- **Auto-detection**: Correctly disables fused GDN when ops scheduled on CPU
  - Added `ggml_backend_dev_type() == CPU` check in llama-context.cpp

### What's Broken

- **Chunked (prompt eval) path**: PPL REGRESSION
  - GPU fused: PPL = 16.8 (0.8B, wikitext-2, 2 chunks)
  - CPU unfused: PPL = 51.9 (same model, same data)
  - Root cause: unknown, under investigation
  - Autoregressive path is correct, so the bug is specific to
    `build_delta_net_chunking` in delta-net-base.cpp

### Fixes Applied (in llama-jit working tree)

1. **llama-context.cpp**: Auto-detect CPU device, disable fused GDN
   - Both autoregressive and chunked fused paths disabled when on CPU

2. **qwen35.cpp**: Conditional unfused path
   - Re-add `ggml_sigmoid(beta)` when fused GDN disabled
     (Phase 16 fused sigmoid into GPU GDN kernel)
   - Re-add `ggml_silu(conv_output)` when fused GDN disabled
     (Phase 16 fused SiLU into GPU ssm_conv shader)
   - Use `ggml_add + ggml_softplus + ggml_mul` instead of `ggml_fused_gate_prep`

3. **ggml-cpu.c**: CPU implementation of FUSED_GATE_PREP
   - `output[i] = softplus(alpha[i] + dt_bias[i % H]) * ssm_a[i % H]`
   - Currently unused (guarded behind fused-only path) but available

### Phase 16 Fusions That Affect CPU Path

Phase 16 fused several ops into GPU shaders. When running unfused on CPU,
these must be restored as separate ops:

| Fusion | GPU shader | CPU unfused equivalent | Status |
|--------|-----------|----------------------|--------|
| sigmoid(beta) | gated_delta_net.comp:140 | ggml_sigmoid(beta) | FIXED |
| SiLU(conv_out) | ssm_conv.comp | ggml_silu(conv_output) | FIXED |
| gate_prep | fused_gate_prep.comp | add+softplus+mul | FIXED |
| chunked GDN beta? | gated_delta_net.comp | ??? | INVESTIGATING |

### Next: PPL Validation

1. Diff `delta-net-base.cpp` against pre-Phase-16 to find chunked path changes
2. Check if `build_delta_net_chunking` was also modified by Phase 16
3. Fix chunked path PPL to match GPU within ±0.5
4. Run full PPL comparison (20+ chunks) on 0.8B
5. Benchmark 9B, 27B, 35B-A3B on CPU with validated path

### CPU Performance (preliminary)

| Model | Threads | NUMA | t/s (gen) | ms/token |
|-------|---------|------|-----------|----------|
| 0.8B | 6 | single | ~8 | ~125 |
| 9B | 6 | single | 1.24 | 805 |
| 9B | 12 | interleave | 1.02 | 980 |
| 9B | 6 | socket 0 | 1.20 | 832 |

### Models Downloaded

- Qwen3.5-9B-Q4_K_M.gguf (5.3 GB)
- Qwen3.5-27B-Q4_K_M.gguf (16 GB)
- Qwen3.5-35B-A3B-Q4_K_M.gguf (21 GB) — MoE, 3B active
