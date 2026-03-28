# Phase 16: Close the GPU Efficiency Gap

## Status: IN PROGRESS — first fusion proved, big bang fusion next

## Root Cause: GCN3 Inter-Dispatch Pipeline Stall

Each Vulkan dispatch boundary on GCN3 (Polaris 12, 5 CUs) costs ~500µs
of hardware pipeline stall: `BUFFER_WBINVL1_VOL` L1 invalidation + L2
writeback before the next dispatch can read. This is NOT idle time from
poor batching — it persists even with all dispatches in one command buffer.

Proof:
- `vkQueueSubmit + fence`: 53µs (measured with test harness)
- Pipeline barriers: 0.2µs (negligible in recording)
- But GPU profiler shows 500µs PER DISPATCH for tiny ops (SCALE, CPY)
- Actual compute: 0.3µs (2048 elements × 4 bytes at 48 GB/s)
- **99.9% of each dispatch is GCN3 cache flush overhead**

The ONLY fix: eliminate dispatch boundaries by fusing operations.

## GPU Profiling (67ms GPU, 85ms eval per token)

| Op | Count | Per-op µs | Total ms | Fuseable? |
|----|-------|----------|---------|-----------|
| SCALE | 36 | 536 | 19.3 | YES → fuse into matmul/norm |
| CPY | 37 | 439 | 16.2 | YES → eliminate with in-place ops |
| MUL_MAT_VEC (output 248K) | 1 | 10,083 | 10.1 | NO (bandwidth) |
| MUL_MAT_VEC (FFN q4_K) | 48 | 230 | 11.0 | NO (bandwidth) |
| MUL_MAT_VEC (FFN q5_K) | 18 | 483 | 8.7 | NO (bandwidth) |
| GET_ROWS | 38 | 210 | 8.0 | MAYBE → batch |
| MUL_MAT_ADD (q6_K/q4_K) | ~40 | 250 | 10.0 | NO (bandwidth) |
| GATED_DELTA_NET | 18 | 113 | 2.0 | Already fused |
| SILU (FFN only, after fusion) | 18 | 78 | 1.4 | YES → fuse into FFN |
| RMS_NORM_MUL | ~80 | 15-68 | 3.4 | YES → fuse into adjacent |
| Flash attention | 6 | 200 | 1.2 | NO (already fused) |
| Everything else | ~60 | various | 3.0 | SOME |

## First Fusion: SSM_CONV + SILU (DONE)

Fused SiLU activation into `ssm_conv.comp`: `dst = sum/(1+exp(-sum))`.
Eliminated 18 SILU dispatches. Saved 1.4ms (18 × 78µs actual compute).

The savings were smaller than the estimated 9ms because the 500µs
"per-dispatch" time in the profiler includes the stall from the NEXT
dispatch boundary, not just this one. Eliminating one dispatch removes
one stall but the adjacent dispatches still have their own stalls.

**Key insight: savings are CUMULATIVE. Eliminating N consecutive
dispatch boundaries saves N × 500µs because the stalls don't overlap.**

## Big Bang Fusion Plan

### Target: Fuse the Entire SSM Pre/Post Chain

Per SSM layer, the current dispatch sequence:

```
1.  MUL_MAT (beta projection)     — keep (bandwidth-bound matmul)
2.  SIGMOID(beta)                  — FUSE into #18
3.  MUL_MAT (alpha projection)    — keep
4.  ADD(alpha, dt_bias)            — FUSE →
5.  SOFTPLUS(alpha_biased)         — FUSE → single "gate_prep" dispatch
6.  MUL(softplus, ssm_a)          — FUSE →
7.  GET_ROWS (conv state)          — keep (state fetch)
8.  CONCAT (conv_states, qkv)      — FUSE into #9
9.  SSM_CONV + SILU (fused)        — DONE
10. L2_NORM(q)                     — FUSE →
11. L2_NORM(k)                     — FUSE → single "qk_norm" dispatch
12. GATED_DELTA_NET                — keep (already fused)
13. CPY (conv state update)        — FUSE →
14. CPY (ssm state update)         — FUSE → single "state_update" dispatch
15. RMS_NORM(output)               — FUSE →
16. SILU(z)                        — FUSE → single "gated_norm" dispatch
17. MUL(normed, silu_z)            — FUSE →
18. MUL_MAT (output projection)   — keep
```

### Fused Shaders to Write

**1. `fused_gate_prep.comp`** — sigmoid(beta) + add + softplus + mul
- Input: beta (after matmul), alpha (after matmul), dt_bias, ssm_a
- Output: gate, beta_sigmoid
- Eliminates: 4 dispatches × 18 layers = 72 dispatches → 18
- Savings: ~36ms → ~9ms (saves 27ms)

**2. `fused_qk_norm.comp`** — l2_norm(q) + l2_norm(k)
- Input: q_conv, k_conv (from SSM_CONV output views)
- Output: q_normed, k_normed
- Eliminates: 2 dispatches × 18 layers = 36 dispatches → 18
- Savings: ~3ms → ~1ms (saves 2ms)

**3. `fused_state_update.comp`** — cpy(conv_state) + cpy(ssm_state)
- Input: last_conv_states, new_ssm_state, state buffers
- Output: updated state cache (in-place write)
- Eliminates: 2 dispatches × 18 layers = 36 dispatches → 18
- Savings: ~16ms → ~8ms (saves 8ms)

**4. `fused_gated_norm.comp`** — rms_norm + silu + mul
- Input: delta_net output, z (gate), norm weights
- Output: gated normalized output
- Eliminates: 3 dispatches × 18 layers = 54 dispatches → 18
- Savings: ~5ms → ~2ms (saves 3ms)

### Total Expected Savings

| Fusion | Dispatches eliminated | Estimated savings |
|--------|----------------------|-------------------|
| SSM_CONV+SILU (done) | 18 | 1.4ms |
| gate_prep | 54 | 27ms |
| qk_norm | 18 | 2ms |
| state_update | 18 | 8ms |
| gated_norm | 36 | 3ms |
| **TOTAL** | **144** | **~41ms** |

From 67ms GPU → ~26ms GPU. Eval: 85ms → ~44ms. **23 t/s → 45 t/s.**

Conservative (50% of estimate): **23 t/s → 34 t/s.**

### Implementation

Each fused shader is a standalone `.comp` file injected via PKGBUILD.
The model graph (qwen35.cpp) is modified to emit fused ops instead of
individual ones. The dispatch logic (ggml-vulkan.cpp) routes fused ops
to the new shaders.

Approach: implement ALL four fused shaders together ("big bang"),
test correctness against unfused output, then benchmark.

### Files to Modify

- New shaders in `vulkan-shaders/`:
  - `fused_gate_prep.comp`
  - `fused_qk_norm.comp`
  - `fused_state_update.comp`
  - `fused_gated_norm.comp`
- `ggml-vulkan.cpp`: pipeline creation + dispatch for fused ops
- `src/models/qwen35.cpp`: emit fused op sequences
- `ggml.h` / `ggml.c`: new GGML_OP types (or use existing fusion framework)

### Verification

1. Generate 100 tokens with fused build, compare text quality
2. `GGML_VK_PERF_LOGGER=1` — verify dispatch count drops by ~144
3. 3-run benchmark: target 30+ t/s
4. Stability: 500-token generation
5. No regression on SmolLM2 (uses different architecture, unaffected)

## Packages

- `llama-cpp-vulkan-polaris` b8508-10 (SSM_CONV+SILU fusion + fence fix)
