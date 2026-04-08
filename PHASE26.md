# PHASE 26 — Closing the bandwidth ceiling gap

> **Predecessor:** PHASE25.md, which landed at 4.32 t/s on the 8K-fill
> OpenClaw bench (86.4% of the 5 t/s nominal target). Phase 26 picks up
> the carryover work and shifts focus from KV-side optimizations
> (which are mostly tapped out at 8K fill) to MoE-side bandwidth and
> production validation at the real 64K target context.

## Status entering Phase 26

- **Current best:** 4.32 t/s on 8K fill, `Qwen3.5-35B-A3B-Q4_K_M`,
  `-ctk tq_kv_1b -ctv tq_v_4b`, dual Xeon X5650 single-socket-bound
- **Target (nominal):** 5.0 t/s
- **Target (production):** 5.0 t/s **sustained at 64K fill** — the
  real OpenClaw production scenario
- **Bandwidth ceiling:** ~13 t/s on a single socket at 100% effective
  DDR3 streaming, ~25 t/s if we capture both sockets
- **Gap to nominal:** 0.68 t/s = 15.7% = 36 ms/tok at 8K fill
- **Gap to bandwidth ceiling:** ~3x at 8K fill — large headroom
  remains, but every additional bps requires touching the MoE-side
  scatter pattern that Phase 25's KV work didn't reach

## Active branch

- **`phase25-decode-perf`** (in `llama.cpp/src/llama.cpp-b8508`): the
  ongoing single branch carrying all in-flight Phase 25/26 work.
  Tip: `2fc8347bb` (TQ_KV_FUSED kernel).
- Phase 26 work continues on this branch unless a sub-experiment
  needs isolation.

## Carryover from Phase 25

### 1. Production benchmark at 64K fill (was Phase 25 Step 7) — **first move**

The entire Phase 25 perf history was measured against an 8K fill
fixture. The actual production target is 64K fill. We have not yet
measured 4.32 t/s under those conditions. **Most plausible
prediction is 3.5–4.0 t/s** at 64K because KV traffic dominates
once the cache fills.

Tasks:
- Run the existing `tests/openclaw_64k_bench.sh` against
  `Qwen3.5-35B-A3B-Q4_K_M.gguf` with the 32K and 64K fill fixtures
  (both already exist in `tests/`)
- Capture: prompt eval t/s, gen t/s, total wall time
- Establish the actual gap to 5 t/s under production conditions
- Add the result row to PHASE26.md before any further optimisation

### 2. Quality validation (was Phase 25 Step 7)

We've held a token-equivalence standard for every Phase 25 step
*against itself* (fused vs unfused), but we have never measured
absolute quality vs an unquantized baseline. The TQ_KV_1B + TQ_V_4B
config is aggressive (1-bit K, 4-bit V) and the failure mode would
be subtle output drift, not crashes.

Tasks:
- Run llama-perplexity on a small wikitext-2 chunk with our config
  (`-ctk tq_kv_1b -ctv tq_v_4b`) vs the same model with `-ctk f16
  -ctv f16` baseline
- Acceptance criterion: perplexity within 1% of the F16 baseline
- If quality fails, the production deployment can't use the
  aggressive KV config and we need to fall back to something safer
  (e.g. q4_0 KV) and re-baseline

### 3. Refined Step 5 — valid-range-aware split-KV (was reverted in Phase 25)

Step 5 split the K cache decode path into chunks for parallelism,
which regressed at partial fill because the chunking was based on
the *allocated* KV size (e.g. 64K), not the *valid* run. The
refined approach gates chunking on `valid_run` so a 4K-filled
context isn't split into 16x 4K chunks of mostly garbage.

Tasks:
- Re-read the reverted Step 5 commit (`83ecba512`) and the original
  attempt (`241b9621f`)
- Add a `valid_run >= chunk_size_min` gate before splitting
- Measure on both 8K and 32K fill
- Expected gain: +2–6% at 32K, marginal at 8K
- Token-equivalence test required (same kill-switch methodology
  as Step 4.75)

### 4. Spec decode — finish what Phase 25 Step 9 started

Phase 25 cherry-picked PR #20075 (recurrent state checkpointing) and
PR #20700 (MTP head support) but discovered the spec-decode path
doesn't activate at runtime due to two structural gaps:

1. The `common_speculative_is_compat` check feeds two tokens in a
   single batch and tries `seq_rm(0, 1, -1)`. PR #20075's checkpoint
   only fires at batch boundaries, so a single-batch test never
   creates a checkpoint to roll back to.
2. The 35B-A3B GGUF was converted before any of these PRs landed —
   it has no MTP tensors. The fallback gate
   `llama_model_n_mtp_layers > 0` returns 0 and rejects spec decode
   entirely.

Two sub-tasks:
- **4a. Compat-check fix.** Patch `common_speculative_is_compat`
  to use two separate `llama_decode` calls (one token each) so a
  checkpoint actually gets created at the batch boundary between
  them. This is a 2-line change and is the more honest test —
  it's what real speculative decode does. Once this fixes, the
  draft-model spec-decode path can be exercised via `llama-server`
  with `-md`.
- **4b. GGUF re-conversion.** Re-run the cherry-picked
  `convert_hf_to_gguf.py` against the source HF repo for
  `Qwen3.5-35B-A3B`, re-quantize to Q4_K_M, verify the resulting
  GGUF has MTP head tensors. This is hours of disk and CPU but is
  the only way to exercise `--spec-type mtp`.
- **Acceptance criterion:** end-to-end spec decode running in
  `llama-server` with measurable acceptance rate. Net t/s gain is
  bonus; the primary acceptance criterion is "spec decode is
  available, validated, and not silently broken."

### 5. MoE expert weight caching — **highest theoretical lever**

Phase 25 ground-truthed that MoE expert weight reads are 86% of
per-token bandwidth at 8K fill (and an even larger fraction at
64K). The router selects 8 experts + 1 shared per layer, and
consecutive autoregressive decode tokens have a high chance of
re-selecting the same experts on the same layers. Today every
token re-reads the expert weights from DRAM.

Sub-experiments to consider:
- **5a. Per-thread expert cache (LRU, ~32 MB budget).** Cache the
  most recently used expert weight tiles in a per-thread L3-sized
  buffer. On hit, skip the DRAM read and dequant from the cache.
  Measure hit rate on the OpenClaw fixture before deciding if it's
  worth the engineering. Hit rate < 30% means it's not worth it.
- **5b. Expert prefetch from router selection.** Once the router
  for layer `L` has selected experts `E_L`, kick off an explicit
  prefetch (mm_prefetch / madvise WILLNEED) for those experts so
  the actual matmul reads them from L2/L3 instead of DRAM.
- **5c. Pre-warmed weight pinning for the shared expert.** The
  shared expert (always-on) is a small fraction of the weights but
  every layer hits it. Pin it in a hugepage and verify it stays
  cache-resident.

### 6. NUMA model weight duplication

The host has dual Xeon X5650 with two NUMA nodes. Phase 25 found
that single-socket binding (`--cpunodebind=1 --membind=1`) beat
dual-socket interleaving by ~15% because the QPI cross-socket
latency dominates for MoE scatter. But that leaves half the
bandwidth on the floor.

Idea: duplicate the read-only model weights into BOTH NUMA nodes,
then run threads on both sockets, each thread reads from its
local node. Doubles the effective DRAM bandwidth at the cost of
2x memory.

Tasks:
- Verify mmap-able dual-loading works with `--no-mmap` workflow
  (we use --no-mmap for NUMA determinism, so this requires custom
  loader work)
- Test with a small model first to validate the technique
- Multi-day project; defer until expert caching is exhausted

### 7. Real CPU implementation of FUSED_GATE_PREP (latent debt)

Phase 25 cherry-picks introduced calls to `ggml_fused_gate_prep` in
`qwen35.cpp` and `qwen35moe.cpp` which had to be reverted because
the op only exists as a Vulkan kernel + CPU stub
(commit `0fba88dc5`). The unfused 3-step `add → softplus → mul`
materializes intermediate `alpha_softplus` to memory each call,
costing two extra full-tensor reads/writes per GDN layer.

Tasks:
- Cherry-pick `0fba88dc5` (the op definition, not the Vulkan
  shaders)
- Write a real CPU kernel for `FUSED_GATE_PREP` (not the stub)
  that fuses the three ops into one register-resident pass
- Re-enable the fused calls in `qwen35.cpp` and `qwen35moe.cpp`
- Token-equivalence test required
- Expected gain: +2–5% on the GDN gate path. Small per-layer but
  fires every GDN layer × every token

### 8. Fix the latent K stride bug (or document it harder)

`tq_kv_1b_attention_multi` and `tq_kv_fused_attention` both stride
K by `s * n_blocks` instead of `nbk1`. Half the real per-K-position
stride. Today this is invisible because Qwen3.5 GQA broadcasts
identical K across both KV heads. The first model that breaks the
GQA-broadcast assumption hits a real correctness bug.

Tasks:
- Write a unit test that constructs a synthetic KV cache with
  distinct values per KV head and asserts the correct head's K is
  read
- Fix BOTH helpers in lockstep to use `nbk1` as the per-K-position
  stride
- Token-equivalence test on the current Qwen3.5 model (should still
  produce identical output because the fix is a no-op for GQA
  broadcast layouts)
- Cost: half a day. Worth doing before any model swap.

## Sequencing

Recommended order, biased toward de-risking the production deployment
and preserving the option to bail out early if a measurement reveals
the gap is bigger than expected:

1. **#1 production benchmark** — establishes the real gap. May
   reveal that we're already at 5 t/s at 64K (unlikely but possible)
   or that we're at 3.0 t/s at 64K (would force a strategy reset).
2. **#2 quality validation** — must pass before any production
   deployment regardless of t/s.
3. **#3 refined Step 5** — small, contained, plan-already-exists,
   targets 32K+ context.
4. **#7 FUSED_GATE_PREP CPU kernel** — small, contained, every
   GDN layer benefits, no model-specific risk.
5. **#5a expert cache hit-rate measurement** — instrument-only,
   tells us if 5b/5c are worth pursuing. Can be done in a few hours.
6. **#5b/5c expert prefetch + shared-expert pinning** — gated on
   #5a result.
7. **#4 spec decode finish** — defer until non-spec t/s is locked
   in. Spec decode is a multiplier on the base; better to maximize
   the base first.
8. **#8 K stride bug fix** — defer until either a model swap is
   imminent or one of the above touches the K cache layout.
9. **#6 NUMA duplication** — last resort, highest risk, multi-day.

## Out of scope for Phase 26

- GCN 1.0/1.1 hardware
- Vulkan/JIT path optimisations (CPU is the production target now)
- Test infra changes
- Re-baselining against a different model (Qwen3.5-35B-A3B is fixed)
- Tensile/MIOpen tuning
