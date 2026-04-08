# PHASE 26 — Closing the bandwidth ceiling gap

> **Predecessor:** PHASE25.md, which landed at 4.32 t/s on the 8K-fill
> OpenClaw bench (86.4% of the 5 t/s nominal target). Phase 26 picks up
> the carryover work and shifts focus from KV-side optimizations
> (which are mostly tapped out at 8K fill) to MoE-side bandwidth and
> production validation at the real 64K target context.

## Status entering Phase 26 (historical, see "Current standing" below for live state)

- **Phase 25 final:** 4.32 t/s on 8K fill, `Qwen3.5-35B-A3B-Q4_K_M`,
  `-ctk tq_kv_1b -ctv tq_v_4b`, dual Xeon X5650 single-socket-bound
- **Target (nominal):** 5.0 t/s — **HIT** by NUMA mirror v1
- **Realistic ceiling on this hardware** (corrected after profile pass):
  ~10-15 t/s. The earlier "25 t/s" and "88 t/s" numbers in this doc
  were both wrong; see the bench results section for honest math.

## Current standing (2026-04-08)

- **Best decode rate:** **5.17 t/s** at 8K openclaw fill, with
  `--numa mirror -t 12 -ctk tq_kv_1b -ctv tq_v_4b`,
  `OMP_WAIT_POLICY=ACTIVE` (now the bench script default)
- **Cumulative gain over Phase 25 system baseline (3.37 t/s):** **+53.4%**
- **Cumulative gain over Phase 25 final (4.32 t/s):** **+19.7%**
- **Headroom to realistic compute ceiling (~10-15 t/s):** ~2-3×
- **Active branch tip:** `phase25-decode-perf` at `aa78a1381`
  (NUMA mirror v1: scaffolding + dual-mbind + read-side hoists +
  post-load replication, weights mirrored only)

What's landed:

| Commit | What |
|---|---|
| `73bf75c03` | `ggml-cpu: scaffolding for GGML_NUMA_STRATEGY_MIRROR` |
| `1675818ef` | `ggml-cpu: dual-mbind allocator and --numa mirror CLI` |
| `aa78a1381` | `ggml-cpu: read-side hoists for the NUMA mirror buffer type` |
| `cfd594d` (top-level) | `tests: openclaw bench — set OMP_WAIT_POLICY=ACTIVE by default` |

Status of the originally-planned Phase 26 #1 work:
- ✅ Mirror buffer type, dual-mbind allocator, CLI exposure
- ✅ Read-side hoists for matmul/mul_mat_id/flash_attn/get_rows
- ✅ Post-load replication (weights buffer finalize_load helper)
- ✅ Bench matrix run + profile pass + ceiling math correction
- ⏸️ KV cache mirroring (bumped to next, see below)
- ⏸️ After-op sync hook (lands with KV mirroring)
- ⏸️ Activation buffer placement (TBD if needed after KV)

## Active branch

- **`phase25-decode-perf`** (in `llama.cpp/src/llama.cpp-b8508`): the
  ongoing single branch carrying all in-flight Phase 25/26 work.
  Tip: `aa78a1381` (NUMA mirror read-side hoists).
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

## NUMA mirror — current state (2026-04-08, mid-implementation)

The Phase 26 #1 NUMA mirror work is partially landed on
`phase25-decode-perf` and **paused mid-stream** before benchmarking.
This section captures the state precisely so we can resume cleanly.

### Commits landed (in order)

- `73bf75c03` — `ggml-cpu: scaffolding for GGML_NUMA_STRATEGY_MIRROR`
  TLS NUMA node id, MIRROR case in set_numa_thread_affinity, mirror
  buffer type with single-copy fallback. No-op when MIRROR is inactive.
- `1675818ef` — `ggml-cpu: dual-mbind allocator and --numa mirror CLI`
  The mirror buft now actually allocates two physical regions via
  mbind to nodes 0 and 1. CLI exposes `--numa mirror`. supports_op
  dispatch is guarded against null contexts. libnuma linked.
- `aa78a1381` — `ggml-cpu: read-side hoists for the NUMA mirror buffer type`
  matmul, mul_mat_id, flash_attn_ext_f16, get_rows_{q,f16,bf16,f32}
  all hoist a NUMA-local pointer at function entry and use it in the
  inner loops. Post-load replication helper
  `ggml_backend_cpu_buffer_finalize_load` is wired into the model
  loader so weights end up in both copies after `file->read_raw`.

### What works

- Build clean.
- Model loads with `--numa mirror` without crashing.
- Output is byte-coherent on the smoke prompt — greedy generation
  produces real English text answering the question. (Verified with
  the OpenClaw smoke fixture; the model picks Paris as the capital
  of France and generates a multiple-choice question, etc.)

### What's NOT done yet (the "next chunk")

1. **KV cache mirroring.** `llama-kv-cache.cpp:119` and
   `llama-memory-recurrent.cpp:79` still hardcode
   `ggml_backend_cpu_buffer_type()` and bypass the buft_list. The
   KV cache lives on a single NUMA node and threads on the other
   socket read it via QPI for half their accesses.
2. **The after-op sync hook.** `ggml_backend_cpu_numa_mirror_after_op_sync`
   is implemented in `numa-mirror.cpp` but **not yet called from
   ggml_compute_forward**. Without this, KV writes (set_rows / cpy)
   only land in copy 0; copy 1 stays stale and reads return wrong
   data. KV mirror cannot ship until this hook is wired.
3. **Activation buffer placement.** Compute scratch and activation
   buffers also live on a single NUMA node. Half the threads pay
   QPI cost for activation reads/writes. The fix may be either
   first-touch via per-thread page binding, or also mirroring the
   activation buffer (which requires the after-op hook for every
   compute op, not just KV writes).

### The methodology problem we caught and parked

After landing the read-side hoists, the smoke test reported
**2.99 t/s** for `--numa mirror -t 12` vs **4.39 t/s** for
`-t 6 --cpunodebind=1 --membind=1`. I declared "mirror is 32% slower
than single-socket, that's bad" and started designing the next step.

**That comparison is not trustworthy** and should not be used to
make any decisions about the mirror's worth. Reasons:

- Both numbers came from `-c 512 -n 32` runs with a 5-token prompt.
  This is a SMOKE test, not a benchmark. Total runtime ~10 s, of
  which several seconds are warm-up and load-time amortization.
  Noise floor on a 32-token greedy generation is probably ±15%.
- The Phase 25 baseline numbers (4.25 → 4.32 t/s across the
  TQ_KV_FUSED commits) were measured on
  `tests/openclaw_64k_bench.sh` against `openclaw_8k_fill.txt`,
  with `-n 256` and the full warmup loop. **That's a different
  test point.** A smoke test rate cannot be directly compared to
  a bench rate, in either direction.
- The Phase 25 NUMA strategy sweep (memory file
  `finding_numa_single_node.md`) found dual-socket interleaved was
  **15% slower** than single-socket bound. My smoke comparison
  showed 38% slower for `--numa distribute`. The 23-point gap
  between those two numbers is by itself enough evidence that the
  smoke comparison isn't measuring what I thought it was.
- I never verified the dual NUMA placement actually happened.
  `numastat -p <pid>` was never run. The mbind call could have
  silently failed into the single-copy fallback path and I would
  not have noticed (the output would still be coherent because the
  fallback is the regular CPU buffer behavior).
- I never verified the read-side hoist is reaching the local copy.
  `perf stat -e mem_load_uops_retired.local_dram,mem_load_uops_retired.remote_dram`
  was never run.

**Lesson for next session:** before making ANY perf comparison
between configurations, run them on the SAME bench fixture with
the SAME script and ENOUGH tokens for the rate to stabilize. The
existing `tests/openclaw_64k_bench.sh` is the right tool. Smoke
tests are pass/fail gates, not measurement instruments. This is
already covered by the `feedback_benchmarking_rules.md` memory
but I broke the rule anyway in the heat of iteration.

### The bench pass to run when resuming

Before deciding what to do next on the NUMA mirror, run this
matrix on the SAME script and SAME fixture as Phase 25's
4.32 t/s baseline:

```
FILL=tests/openclaw_8k_fill.txt SKIP_SMOKE=1 \
  tests/openclaw_64k_bench.sh \
  llama.cpp/src/llama.cpp-b8508/build-tq/bin/llama-completion \
  -ctk tq_kv_1b -ctv tq_v_4b
```

Three configurations, sequentially (never parallel — see
`feedback_benchmarking_rules.md`):

| # | Config | Notes |
|---|---|---|
| A | `-t 6` with `numactl --cpunodebind=1 --membind=1` (existing bench script default) | Phase 25 single-socket reference. Should land at ~4.32 t/s. |
| B | `-t 12 --numa distribute` (drop the `numactl` wrapper) | 12-thread dual-socket without mirror. Apples-to-apples baseline for "what does ggml's existing dual-socket strategy give us on this workload". Expected: ~3.6 t/s based on Phase 25 NUMA sweep, but verify. |
| C | `-t 12 --numa mirror` (current state, weights mirrored only) | The thing we're actually evaluating. |

The current bench script hardcodes `numactl --cpunodebind=1 --membind=1`
in the run command. To run B and C, either modify the script or run the
inner command directly. Suggest a small generalization: accept a
`NUMA_MODE=mirror` env var that overrides the default `numactl`
wrapper with `--numa mirror -t 12` and similar for `distribute`.

### Verification steps (run alongside the bench)

While the bench is running for configurations B and C, in a separate
terminal window:

1. **Confirm dual-NUMA allocation.** `numastat -p $(pgrep llama-completion)`
   should show ~21 GB on each node when MIRROR is active. If it shows
   ~21 GB on one node only, the mbind allocator silently fell back to
   single-copy and the mirror is not actually happening.
2. **Confirm threads are pinned per socket.** `ps -mo pid,tid,psr,comm
   $(pgrep llama-completion)` shows each thread's assigned CPU. With
   MIRROR + `-t 12`, six threads should be on even-numbered CPUs
   (node 0) and six on odd-numbered (node 1).
3. **Confirm local DRAM reads dominate** (only worth doing if (1) and
   (2) check out). Re-run a short generation under `perf stat -e
   mem_load_uops_retired.local_dram,mem_load_uops_retired.remote_dram
   ./build-tq/bin/llama-completion ...`. The local count should be
   substantially higher than remote when MIRROR is active. If they're
   close, the read-side hoist isn't reaching the local copy.

### Decision tree based on bench results

- **If C ≥ A** (mirror beats single-socket on the real bench):
  ship the current state, then add KV cache mirroring + after-op
  hook for further gains. The +9% smoke-test number was probably
  underselling the actual win.
- **If C is within 10% of A**: KV cache mirror is the next move.
  KV is ~13.5% of bandwidth at 8K, so removing the cross-socket
  cost there can plausibly close the gap.
- **If C < 0.8 × A** (mirror is significantly worse than single-socket):
  there's a structural issue. Investigate via the verification steps
  above. Most likely candidates:
  - The mbind silently fell back to single-copy (look for it).
  - The read-side hoist is computing the wrong offset (perf counters
    will show this — remote DRAM reads stay high).
  - ggml's 12-thread coordination on dual-socket Westmere has too
    much overhead and it's the actual bottleneck (not bandwidth).
    In that case, mirror is doing its job but it's a single-process
    architecture problem and we'd need to look at the
    socket-isolated-worker pattern from
    `finding_socket_isolated_worker_pattern.md` instead.
- **If C is wildly different from B** in either direction: that
  itself is a useful signal. C should always be ≥ B if the mirror
  is doing anything. If C ≈ B, the mirror code is a no-op and
  there's a bug.

### Why we're pausing here

We have three commits that compile, produce correct output, and add
the read-side infrastructure. The next chunk (KV mirror + after-op
hook) is non-trivial and the right move is to first measure whether
what we've built so far is actually working before adding more code
on top of an unverified foundation. Adding KV mirror on top of a
broken weight mirror would just paper over the underlying issue
with more code.

When resuming, the first action is the bench pass above, NOT more
implementation. Only after the numbers come back do we decide which
direction to take.

### Bench matrix results (2026-04-08)

The proper bench pass against `tests/openclaw_64k_bench.sh` +
`openclaw_8k_fill.txt`, `-n 256`, full warmup:

| # | Config | Gen t/s | PP t/s | Δ vs Phase 25 (4.32) |
|---|---|---:|---:|---:|
| A | `-t 6 --cpunodebind=1 --membind=1` | 4.29 | 12.40 | -0.7% (within noise) |
| B | `-t 12 --numa distribute` | 3.85 | 21.58 | -10.9% |
| **C** | **`-t 12 --numa mirror`** | **5.01** | **21.71** | **+16.0%** |

**The 5.0 t/s nominal Phase 25 target is hit.** Cumulative gain
from the system baseline (3.37 t/s) is **+48.7%** — we crossed the
target with weights-only mirroring, before any KV cache or
activation work.

Key observations:

- **Mirror beats single-socket by +17% on gen.** The dual-socket
  bandwidth advantage is real and the read-side hoist captures it.
- **Mirror beats distribute by +30% on gen.** Without mirroring,
  12-thread dual-socket is *worse* than 6-thread single-socket
  because cross-socket weight reads dominate. With weight reads
  local to each socket, the bandwidth doubling shows up.
- **PP is +75% over single-socket** in both B and C. Prefill is
  compute-bound (large batched matmul, weights amortize across
  many tokens), so 12-thread parallelism helps directly and the
  mirror has no marginal effect — both B and C land at ~21.6 t/s.
- **The earlier smoke-test "2.99 t/s mirror is slower than baseline"
  comparison was a measurement artifact.** A 32-token greedy
  generation from a 5-token prompt is dominated by warm-up and
  load-time amortization — it doesn't measure steady-state decode
  rate. The methodology lesson stands and is documented above.

### Decision: ship as-is, then iterate

Per the decision tree above, **C ≥ A** clearly: ship the current
state as the new Phase 26 milestone, then profile to understand the
actual bottleneck before adding more code on top.

### Profile pass under --numa mirror (2026-04-08)

Captured pcm + perf record + numastat during a long-decode run
under --numa mirror. The findings completely reshape the Phase 26
priority list.

**Verification first:**
- `numastat -p $pid` shows ~21 GB on each NUMA node. Dual-mbind
  allocator is working as designed. Both copies of model weights
  are physically resident on their assigned sockets.
- The `set_numa_thread_affinity` MIRROR case is correctly pinning
  threads — verified via `ps -mo` showing even thread indices on
  node 0 CPUs and odd on node 1.

**pcm during decode steady-state:**
| Socket | Read GB/s | Write GB/s |
|---|---:|---:|
| 0 | 0.7–1.1 | 0.18–0.24 |
| 1 | 1.0–1.5 | 0.25–0.40 |
| **Total** | **~2.5** | **~0.6** |

**~2.5 GB/s total DRAM reads, against a ~44 GB/s dual-socket
peak streaming ceiling.** Bandwidth is not on the critical path
right now; mirroring more buffers won't move the needle much.

But the "5% of ceiling = 20× t/s headroom" math I initially did
is wrong. Dividing aggregate bandwidth by per-token demand only
gives a t/s ceiling if bandwidth is the binding constraint —
which it isn't here. To find the actual ceiling we have to look
at compute and op-overhead instead.

Honest model param math (Qwen3.5-35B-A3B observed at runtime):
- Total: 34.66B params, 20.49 GiB at 5.08 BPW
- MoE expert FFN per token: 8 active × 512 FF × 2048 embd × 2 ≈ 16.8M
- Plus shared expert per layer: +2.1M
- Plus attention QKVO per layer: ~16M
- × 40 layers + embeddings + norms ≈ **~1.4–1.6B active params/token**
- At 5.08 BPW: **~950 MB of weight reads per token** (NOT 1.875 GB)
- 5.01 t/s × 0.95 GB = 4.75 GB/s of weight demand
- pcm measured 2.5 GB/s DRAM, so cache is serving ~2.25 GB/s
- Cache hit rate by bytes: ~47% (more believable than the 75% I
  initially calculated from a wrong starting estimate)

Compute ceiling math:
- 1.5B active params × 2 ops/param (FMA) ≈ 3 GFLOPS/token
- Westmere int8/quant matmul realistic effective throughput on
  hand-tuned SSE4.2 kernels: ~25–40% of peak
- 12 cores × 2.67 GHz × ~16 INT-MACs/cycle peak ≈ 512 GINT-MACs/sec
- At 25–40% efficiency: ~130–200 GINT-MACs/sec usable
- Per-token compute: ~3 G ops / 130–200 G/sec ≈ 15–25 ms/token
- **Compute ceiling: ~40–65 t/s peak, ~10–15 t/s realistic**

5 t/s is **~35–50% of the realistic compute ceiling**, not 5% of
some absurd 88 t/s ceiling. There's 2–3× headroom available, not 17×.

**CPU utilization during decode:**
- Aggregate: ~46% (`ps` reports 554% / 1200%)
- Per-active-core: ~85% (pcm `Core UTIL` field)
- IPC: 2.05 (strong instruction-level parallelism when running)

So when cores ARE active they're computing efficiently. The
~54% aggregate idle is from cores being literally not-running,
not from memory stalls.

**perf record top hot functions (% of active cycles):**
| % | Function |
|---:|---|
| 34.8 | `ggml_vec_dot_q8_0_q8_0` (likely GDN linear projections) |
| 14.9 | `ggml_vec_dot_q4_K_q8_K` (MoE expert matmul) |
| 14.0 | `tq_v_4b_vec_mad_f32` (TQ V mad in FA loop) |
| 12.6 | `ggml_vec_dot_q5_K_q8_K` (mixed-quant matmul) |
| ~7  | libgomp (OMP barriers / runtime) |
| 4.0 | `tq_kv_fused_attention` |
| 3.3 | `ggml_vec_dot_f32` |
| 1.8 | `ggml_compute_forward_gated_delta_net` |
| ~6 | other compute / dispatch |

**~86% of active cycles is real compute. ~7% is libgomp barriers
inside the active phase.** The hot path is healthy when running.

### Thread scaling probe

Tested mirror at -t 6 (3 threads per socket) vs -t 12 (6 threads
per socket) on the same fixture:

| Config | Gen t/s | PP t/s |
|---|---:|---:|
| -t 6 mirror | 4.88 | 12.99 |
| -t 12 mirror | 5.01 | 21.71 |
| -t 12 mirror + OMP_WAIT_POLICY=ACTIVE | 5.17 | 22.39 |

**Decode 6→12 thread scaling is +3%. PP 6→12 thread scaling is +67%.**

Decode at batch=1 has tiny per-op work units. Each op is roughly
"matmul-vec for this single token" — thousands of MACs split
across N threads. At N=12 each thread gets ~hundreds of MACs and
the op-launch / barrier / wakeup overhead becomes comparable to
the op compute itself. Adding more threads doesn't help because
the work doesn't subdivide meaningfully.

**For decode at batch=1, the bottleneck is op-launch overhead per
op, multiplied by ~hundreds of ops per token, multiplied by 40
layers.** Not bandwidth, not compute, not memory placement.

### Reshaped Phase 26 priority list

The original Phase 26 plan put NUMA mirror first as the highest-EV
lever. After running it and profiling, we have:

1. **NUMA mirror v1 (weights only) — DONE.** +17% over single-socket
   reference. Captures the bandwidth-side win that was available.
   Beats single-socket because it makes 12 threads possible (which
   PP needs) without paying the dual-socket cross-QPI penalty for
   weight reads.

2. **NUMA mirror — KV cache + activations (the rest of #1).**
   *DEMOTED.* Profile shows we're at 5% of bandwidth ceiling.
   Removing the remaining cross-socket reads has a small ceiling.
   Realistic upside: **+1% to +3%**, not the +5–10% I estimated
   before profiling. Worth doing for code completeness but not
   as a perf lever.

3. **Speculative decoding — PROMOTED to highest priority.**
   Spec decode turns batch=1 into batch=N at the matmul level.
   With N=4, each per-op work unit is 4× larger, which makes the
   12-thread parallelism actually pay off for decode the way it
   already does for prefill (+67% from 6→12 threads). This
   directly attacks the op-overhead bottleneck. Phase 25's
   cherry-picks of PR #20075 + PR #20700 are the foundation;
   the structural compat-check fix and GGUF re-conversion are
   the remaining work. Realistic upside if N=4 with even modest
   acceptance rate: **+50–100% on gen t/s**, or roughly 7.5–10
   t/s on this hardware.

4. **Op fusion — second-priority parallel track.**
   Each fused op reduces the number of barriers and the number
   of times threads must coordinate. TQ_KV_FUSED already fused
   K Hamming + softmax + V mad in the FA loop. Concrete next
   targets:
   - `FUSED_GATE_PREP` CPU kernel (cherry-pick 0fba88dc5 from
     llama-jit and write a real CPU implementation, not the
     Vulkan-only stub). Currently every GDN gate is a 3-op
     `add → softplus → mul` sequence; fusing eliminates 2
     intermediate materializations per layer per token.
   - `rms_norm + residual_add` fusion (per-layer, ~2 saved
     per layer × 40 layers = 80 saved barriers per token).
   - SSM_CONV + SiLU fusion (already exists via Vulkan JIT;
     port to CPU).
   Per-target estimate: **+1% to +3%** each. Cumulative across
   3-4 fusions: maybe **+5% to +10%**.

5. **OMP tuning — exhausted at +3%.**
   `OMP_WAIT_POLICY=ACTIVE` gives ~3%. Adding `GOMP_SPINCOUNT=infinite`
   and `OMP_PROC_BIND=close` gave no further improvement. The
   barrier overhead is small; can't squeeze more from OMP config.

6. **Refined Step 5 split-KV chunking.**
   Same priority as before. Useful at higher contexts but
   doesn't address the op-overhead bottleneck.

7. **NUMA mirror — KV cache + activations.**
   Demoted to here. Code-quality finishing touch.

8. **K stride bug fix.** Same as before — defer until model swap.

9. **Production benchmark at 64K.** Last, locks in result.

### The honest revised target

Earlier in this doc I claimed the ceiling was 25 t/s (bandwidth
math) and then briefly 88 t/s (worse bandwidth math). Both are
wrong. Bandwidth is not the binding constraint at our current
operating point — compute and op-overhead are.

Realistic ceiling for this exact model on this exact hardware,
bound by compute on hand-tuned SSE4.2 quant kernels:
**~10–15 t/s realistic, ~25 t/s hard upper bound**.

5 t/s is **~35–50% of the realistic compute ceiling**, with
2–3× headroom — meaningful but not the 17× I claimed before
checking my arithmetic.

Realistic target with the levers below:
- **Spec decode landing (N=4, modest acceptance)**: 7–10 t/s
  (1.5–2× over current 5 t/s — amortizes op overhead and
  shifts the workload toward larger per-op work units that
  the matmul kernels handle well)
- **+ op fusion**: 8–11 t/s
- **+ small wins (KV mirror, refined chunking, bench script
  default OMP_WAIT_POLICY=ACTIVE)**: 9–12 t/s

Beyond ~12 t/s requires either:
- A different threading model (per-socket independent compute
  streams, i.e. the socket-isolated-worker pattern from
  `finding_socket_isolated_worker_pattern.md`) — but that gives
  throughput, not single-stream latency
- Tensor parallelism at the matmul level — major rewrite,
  uncertain payoff on this hardware
- A different model architecture (smaller MoE, fewer layers,
  or lower-bitwidth quantization)

**Spec decode is now the next coding move.** The Phase 25 work
on PR #20075 + PR #20700 was the right direction; we paused at
the wrong moment because the structural compat-check issue felt
hard. With the profile data in hand, we now know spec decode is
THE lever for this hardware, not a nice-to-have.

### Pre-flight for the next session

Before resuming spec decode, the work to do:

1. **(Optional) Run the OMP_WAIT_POLICY=ACTIVE bench again** for a
   confirmed +3% baseline. The 5.17 t/s number is from one run;
   noise is probably ±2%. Worth a second run for statistical
   confidence before treating it as the new floor.
2. **Add OMP_WAIT_POLICY=ACTIVE to the bench script default**
   so it lands in production builds without needing manual env
   var setting.
3. **Look at the spec-decode compat-check fix from Phase 25
   Step 9** — that's the 2-line patch to `common_speculative_is_compat`
   that feeds two tokens in two separate `llama_decode` calls
   instead of one. Then end-to-end test with `llama-server -md`
   against a small draft model (0.8B Qwen3.5).
4. **Decide whether to re-convert the GGUF for MTP.** The MTP
   path is a different spec decode strategy than draft-model.
   If draft-model spec decode lands first and gets us to 7-10 t/s,
   MTP becomes optional. If draft-model is hard to land, MTP
   might be easier despite the GGUF re-conversion cost.

## NUMA mirror — KV cache work plan (2026-04-08)

User direction: finish the NUMA mirror story before pivoting to spec
decode. Even though the profile shows KV mirror is a small win
(+1-3% expected, not the +5-10% I claimed pre-profile), it closes
out Phase 26 #1 cleanly and gives us a verified-correct mirror code
path that any future work (spec decode, op fusion, etc.) can lean on.

### Goal

Land KV cache mirroring + after-op sync hook on `phase25-decode-perf`
so that the KV cache and recurrent state buffers are physically
replicated across both NUMA nodes, threads on each socket read their
local copy, and writes from any thread propagate to both copies. The
result must be token-identical to the current weights-only mirror
(verified by greedy temp=0 generation diff).

### Touch list

**1. Runtime-aware buft selection helper.**

Add to `ggml/include/ggml-cpu.h` (public API):

```c
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type_for_runtime(void);
```

Returns the mirror buft when `g_state.numa.numa_strategy == GGML_NUMA_STRATEGY_MIRROR`,
else falls through to the regular `ggml_backend_cpu_buffer_type()`.
Implementation in `numa-mirror.cpp` (it has access to both buft
constructors and the runtime strategy via `ggml_cpu_get_numa_strategy()`).

**2. Use it at the KV cache and recurrent state allocation sites.**

`src/llama-kv-cache.cpp:119`:
```cpp
- ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
+ ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type_for_runtime();
```

`src/llama-memory-recurrent.cpp:79`: same one-line change.

These callsites currently bypass `make_cpu_buft_list` and the
extra-bufts mechanism that the model loader uses for weights. They
ask for the default CPU buft directly. The helper redirects them
to the mirror buft when MIRROR is active.

Add `#include "ggml-cpu.h"` to both files if not already present.

**3. Wire the after-op sync hook in `ggml_compute_forward`.**

In `ggml/src/ggml-cpu/ggml-cpu.c`, the dispatch switch ends at
line 2176. Right after the switch closes (and before the function
returns at line 2177), add:

```c
// NUMA mirror: replicate per-op writes to the secondary copy.
// No-op for non-mirror buffers (the helper checks the buft and
// returns immediately).
ggml_backend_cpu_numa_mirror_after_op_sync(tensor);
```

The hook is called by every thread that enters `ggml_compute_forward`
for this op. The hook itself uses `params->ith` to coordinate
parallelism — see step 4.

Add `#include "numa-mirror.h"` to `ggml-cpu.c`.

**4. Make the after-op hook actually do per-op narrow dirty-region sync.**

The current `ggml_backend_cpu_numa_mirror_after_op_sync` in
`numa-mirror.cpp` is a stub that does a full-tensor memcpy:

```cpp
const size_t nbytes = ggml_nbytes(tensor);
memcpy((char *) tensor->data + alt_off, tensor->data, nbytes);
```

For the KV cache (~63 MB at 8K context per layer), syncing the
whole tensor on every set_rows call (one per layer per token) is
catastrophic — ~2.5 GB/token of cross-socket memcpy traffic. Need
per-op narrow dispatch.

Replace the stub with:

```cpp
void ggml_backend_cpu_numa_mirror_after_op_sync(struct ggml_tensor * tensor) {
    if (!tensor || !tensor->buffer) return;
    const ptrdiff_t alt_off = ggml_backend_cpu_numa_mirror_alt_offset(tensor->buffer);
    if (alt_off == 0) return;  // not a mirror buffer

    switch (tensor->op) {
        case GGML_OP_SET_ROWS: {
            // Dirty region: rows in dst whose indices are in src[1].
            // Walk src[1] and memcpy each touched row from copy 0 to copy 1.
            const ggml_tensor * idxs = tensor->src[1];
            // ... read each index, compute row offset, memcpy row_size bytes ...
            // (Handles both i32 and i64 index types, mirroring the
            // template specialization in compute_forward_set_rows.)
            break;
        }
        case GGML_OP_CPY:
        case GGML_OP_DUP: {
            // Dirty region: the entire dst tensor view extent.
            memcpy((char*)tensor->data + alt_off, tensor->data, ggml_nbytes(tensor));
            break;
        }
        case GGML_OP_SCALE:
        case GGML_OP_SCALE_INPLACE: {
            // Same — entire dst view.
            memcpy((char*)tensor->data + alt_off, tensor->data, ggml_nbytes(tensor));
            break;
        }
        case GGML_OP_SSM_CONV:
        case GGML_OP_GATED_DELTA_NET: {
            // SSM/GDN write the new recurrent state to dst. Sync the
            // entire dst view (small per token — kilobytes scale).
            memcpy((char*)tensor->data + alt_off, tensor->data, ggml_nbytes(tensor));
            break;
        }
        default:
            // Unknown write op — fall back to full-tensor sync.
            // This is a correctness safety net: if we forgot to add
            // a case for a new write op, the model still produces
            // correct output (just with more cross-socket traffic).
            memcpy((char*)tensor->data + alt_off, tensor->data, ggml_nbytes(tensor));
            break;
    }
}
```

**Per-thread vs single-thread sync:** the simple version above runs
on EVERY thread that enters compute_forward — each thread does the
full memcpy redundantly. That's wasteful. Two options:

- **Option A (simpler):** gate on `params->ith == 0` so only the master
  thread does the sync. The other 11 threads just return. Still
  works because OpenMP barriers between ops sync all threads at the
  next op entry.
- **Option B (faster for big writes):** parallelize the memcpy across
  the calling threads using `params->ith / params->nth`. Each thread
  copies its slice of the dirty region.

For our case, KV writes are tiny (single row) so option A is fine.
For SSM_CONV and GATED_DELTA_NET, the dst view is larger but still
KB-scale. Option A works for everything.

**Adopt option A** in v1. If profiling shows it matters, switch to B.

**5. Read-only ops should not trigger sync.**

Many ops in the dispatch switch are read-only (compute output to a
fresh dst tensor that lives in scratch, not in a mirror buffer).
Their dst buffer is the compute scratch, not a mirror buffer, so
the helper's `alt_off == 0` check rejects them at the first branch.
No special handling needed at the call site.

**6. Kill switch for equivalence testing.**

Add an env var `LLAMA_NUMA_MIRROR_KV_DISABLE` (default 0). When set
to 1, the runtime-aware buft helper returns the regular CPU buft
even with `--numa mirror`. Used to A/B test KV mirroring without
rebuilding.

```c
ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type_for_runtime(void) {
    if (ggml_cpu_get_numa_strategy() != GGML_NUMA_STRATEGY_MIRROR) {
        return ggml_backend_cpu_buffer_type();
    }
    if (getenv("LLAMA_NUMA_MIRROR_KV_DISABLE") != NULL) {
        return ggml_backend_cpu_buffer_type();
    }
    return ggml_backend_cpu_numa_mirror_buffer_type();
}
```

The kill switch is removed in the same commit that lands the bench
results, after equivalence is verified.

### Verification gates

Each gate must pass before moving to the next.

**Gate A — Build clean.** All commits compile, no new warnings in
files I touched.

**Gate B — `--numa mirror` smoke test.** Same `tests/smoke_test.sh`
as for the weights mirror. Output is coherent English text. Process
exits cleanly.

**Gate C — Numastat verification.** While a `--numa mirror` bench
is running, `numastat -p $pid` should show:
- Approximately equal split across nodes for total memory (~21 GB
  weights × 2 + ~10 MB KV cache × 2 + activations on the dominant
  socket)
- The exact KV cache size depends on context length but should
  appear on BOTH nodes when mirror is active

**Gate D — Token-identical equivalence.** Greedy temp=0 generation
of 64 tokens with the OpenClaw smoke prompt:
- Run with `--numa mirror`
- Run with `--numa mirror LLAMA_NUMA_MIRROR_KV_DISABLE=1`
- `diff` the outputs — must be byte-identical
- Both should also match the single-socket reference output (same
  greedy seed, same model state)

**Gate E — Bench result.** Run `tests/openclaw_64k_bench.sh` with
`-ctk tq_kv_1b -ctv tq_v_4b`, three configs:
- Current weights-only mirror (the 5.17 t/s baseline)
- KV mirror with full-tensor sync stub (just to confirm correctness
  of the wiring before optimizing the dispatch)
- KV mirror with per-op narrow dispatch

Decision criteria:
- Per-op narrow dispatch must be ≥ weights-only baseline (no
  regression). Realistic +1% to +3% gain.
- Full-tensor sync stub will probably be slower due to the
  catastrophic cross-socket bandwidth. That's expected and
  ships only as an intermediate testing step, never as the
  final state.

### Commit boundaries

Three commits, each landing a coherent unit:

**Commit 1: KV mirror buft selection + full-tensor after-op sync.**
- New `ggml_backend_cpu_buffer_type_for_runtime()` helper
- KV cache and recurrent state callsites use it
- After-op hook wired in `ggml_compute_forward`, helper does
  full-tensor sync for any mirror dst (no per-op dispatch yet)
- Kill switch `LLAMA_NUMA_MIRROR_KV_DISABLE`
- Verification: gates A, B, D (equivalence test)

This is the minimum-correct intermediate state. KV mirror works,
but every write op syncs the full dst tensor. Bench will show
significant slowdown vs the weights-only baseline. That's fine
for this commit — we just need to confirm correctness.

**Commit 2: Per-op narrow dirty-region dispatch.**
- Replace the stub `after_op_sync` with the per-op switch dispatch
- For SET_ROWS: walk the index tensor and sync only the touched rows
- For CPY/DUP/SCALE/SSM_CONV/GATED_DELTA_NET: sync the dst view extent
- For unknown ops: fall back to full-tensor sync (safety net)
- Verification: gates A, B, D, E

This is the perf optimization that takes us from "correct but slow"
to "correct and fast".

**Commit 3: Remove kill switch + write up bench result.**
- Drop `LLAMA_NUMA_MIRROR_KV_DISABLE` from the helper
- PHASE26.md update with the actual bench numbers
- Mark Phase 26 #1 as DONE

### Risk surface

**Risk 1: I forget a write op type.**
The default case in the dispatch switch is a full-tensor sync,
which is correct (just slow). The model never produces wrong
output, just wastes cross-socket bandwidth on the unhandled op.
Mitigation: log the unhandled op type the first time it's hit
during bring-up, then add a case for it.

**Risk 2: SET_ROWS index walk is wrong.**
The compute_forward_set_rows iterates `(i03, i02, i in [ir0, ir1))`
and reads `*(idx_t*)(src1->data + i10*nb10 + i11*nb11 + i12*nb12)`.
The hook needs to do the same index iteration but over the FULL
range, not the per-thread slice. Need to be careful with the
template specialization (i32 vs i64 indices).

**Risk 3: Recurrent state checkpointing path.**
PR #20075 added `copy_cell` in `llama-memory-recurrent.cpp` which
uses `ggml_backend_tensor_copy` to clone an entire cell of recurrent
state. This goes through the buffer's `cpy_tensor` callback, which
the mirror buft already implements (it writes to both copies). So
this should "just work" without any changes to the hook.

**Risk 4: Activation buffer is single-copy and gets QPI traffic.**
The activation tensors live in compute scratch, which uses the
default CPU buft (not the mirror). Threads on socket 1 reading
activations produced by socket 0 still pay QPI cost. Profile says
activations are small (KB scale per layer per token) so this is
fine — but if profiling after KV mirror shows it matters, we
revisit.

**Risk 5: Op-launch overhead is the actual bottleneck.**
Already known from profile. KV mirror won't fix this. The +1-3%
expected gain is from removing the small bandwidth overhead, not
from making the bottleneck go away. Manage expectations
accordingly.

### Out of scope for this work

- Activation buffer mirroring (defer until after KV mirror is
  measured; profile says it's a small win)
- Spec decode (separate Phase 26 work, much higher leverage)
- Op fusion (separate Phase 26 work)
- Anything in PHASE26.md priority list 4-9
