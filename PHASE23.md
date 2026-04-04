# Phase 23: 122B MoE CPU Inference + Speculative Decoding

## Status: Planning

## Hardware Context

- **GPU**: Radeon Pro WX 2100 (Polaris 12, gfx803, 2 GB GDDR5, 10 CUs)
- **CPU**: Dual Xeon X5650 (Westmere, 6 cores/socket, 12 total, SSE4.2/SSSE3, NO AVX)
- **RAM**: 188 GB DDR3-1333 (3 channels/socket, ~25.6 GB/s/socket, ~51 GB/s aggregate)
- **QPI**: 6.4 GT/s (~12.8 GB/s between sockets)
- **NUMA**: 2 nodes, 96 GB each, distance 10/20
- **TLB**: L1 DTLB 64 entries, L2 DTLB 512 entries (2 MB reach at 4 KB pages)
- **Huge pages**: 2 MB and 1 GB supported (pdpe1gb flag confirmed)
- **PCIe**: 2.0 x16 (~5 GB/s practical)

## Goal

Run Qwen3.5-122B-A10B (MoE, 10B active/token) as CPU target model with
Qwen3.5-0.8B as GPU draft model via speculative decoding.

- **Primary target**: 64K context, 3-6 t/s effective output of 122B quality
- **Stretch goal**: 128K context via `-nkvo` (KV on CPU, PCIe penalty)

64K is the practical sweet spot: KV fits on GPU (Q8, 384 MB) with margin,
draft runs at full 49 t/s, prompt eval is long but tolerable (30 min baseline,
10-15 min after SSSE3 kernel), and 0.8B acceptance rate still reasonable.

128K pushes into diminishing returns: tight VRAM or PCIe penalty on draft,
1-2 hour prompt eval on target, and 0.8B acceptance rate degrades significantly
at that context length.

## Models

- **Draft (GPU)**: Qwen3.5-0.8B-Q8_0 (775 MB) — 35 t/s on Vulkan at 64K Q8 KV
- **Target (CPU)**: Qwen3.5-122B-A10B-UD-Q4_K_XL (77 GB, 3 shards)
  - Repo: unsloth/Qwen3.5-122B-A10B-GGUF
  - Downloaded to: /home/llm/models/UD-Q4_K_XL/

## Architecture Analysis

### Qwen3.5 Hybrid Attention+SSM

`full_attention_interval = 4` — only 1 in 4 layers uses attention (has KV cache).
The rest are SSM (GDN) with fixed-size state. This slashes KV costs by 75%.

| Model | Total layers | Attention layers | KV heads | Head dim | KV/token |
|-------|-------------|-----------------|----------|----------|----------|
| 0.8B  | 24          | 6               | 2        | 256      | 12 KB    |
| 122B  | 48          | 12              | 2        | 256      | 24 KB    |

### VRAM Budget (draft, 2 GB)

**Option A: KV on GPU** (standard)

| Context | Weights | KV (f16) | KV (Q8) | Total (f16) | Total (Q8) |
|---------|---------|----------|---------|-------------|------------|
| 64K     | 508 MB  | 768 MB   | 384 MB  | 1.3 GB      | 0.9 GB     |
| 128K    | 508 MB  | 1.5 GB   | 768 MB  | 2.0 GB      | 1.3 GB     |

128K at f16 KV = 2.0 GB (razor thin). Q8 KV recommended for headroom.

**Option B: KV on CPU (`-nkvo`)** — weights on GPU, KV cache in system RAM

VRAM usage: 508 MB (weights only), any context fits.
Tradeoff: GPU reads KV from CPU over PCIe 2.0 (~5 GB/s) each attention layer.

Per-token KV read = context × kv_heads × head_dim × 2(K+V) × 2(f16) × attn_layers
At 128K: 131072 × 2 × 256 × 2 × 2 × 6 = 1.5 GB per token

| KV quant | KV read/token (128K) | PCIe time | Draft t/s |
|----------|---------------------|-----------|-----------|
| f16      | 1.5 GB              | 300 ms    | ~3.3      |
| Q8       | 768 MB              | 154 ms    | ~6.5      |
| Q4       | 384 MB              | 77 ms     | ~13       |

With Q4 KV offload, draft achieves ~13 t/s at 128K context using only
508 MB VRAM. This enables unlimited context scaling on the draft model.
Quality impact on acceptance rate needs validation.

### RAM Budget (target, 188 GB)

| Context | Weights | KV (f16) | SSM state | Total  |
|---------|---------|----------|-----------|--------|
| 64K     | 77 GB   | 1.5 GB   | ~0.3 GB   | 79 GB  |
| 128K    | 77 GB   | 3.0 GB   | ~0.3 GB   | 80 GB  |
| 262K    | 77 GB   | 6.1 GB   | ~0.3 GB   | 83 GB  |

### Speculative Decoding Math

Target forward pass: ~3.2 GB active / 40-51 GB/s = 63-80 ms
Draft per token: ~20 ms (49 t/s)
Optimal draft length: K=1-2 (target is relatively fast vs draft)

Expected throughput (before SSSE3 optimization):
- Without spec decode (CPU only): 2-4 t/s
- With spec decode (K=2, p=0.65): 3-6 t/s effective

---

## Step 0: Draft Model Validation (KV on GPU at 64K)

Validate that 0.8B draft model works correctly at 64K context with Q8 KV
on GPU before touching the target model. This establishes the draft side
of the speculative decoding pipeline.

### Tests

**A. Verify 64K context fits in VRAM (Q8 KV)**

```bash
llama-completion \
  -m /home/llm/models/Qwen3.5-0.8B-Q4_K_M.gguf \
  -ngl 99 -c 65536 -ctk q8_0 -ctv q8_0 \
  --simple-io -no-cnv -n 32 \
  -p "Hello world"
```

Expected VRAM: 508 MB weights + 384 MB KV = 892 MB. Should fit with margin.

**B. Verify generation quality at 64K context**

Generate with a long prompt (fill context) and check output coherence.
Compare greedy output (temp=0, seed=42) at c512 vs c65536 with same prompt
to ensure no quality regression from KV quantization.

**C. Measure draft speed at 64K**

Confirm ~49 t/s generation is maintained with KV Q8 at 64K context.
If speed drops significantly, investigate whether Vulkan attention kernel
scales poorly with context length on gfx803.

**D. Test 128K stretch (two configurations)**

1. KV on GPU (f16): 508 + 1536 = 2044 MB. Will it OOM?
2. KV on CPU (`-nkvo`, Q8): 508 MB VRAM only. Measure PCIe penalty.

```bash
# 128K, KV on GPU f16
llama-completion \
  -m /home/llm/models/Qwen3.5-0.8B-Q4_K_M.gguf \
  -ngl 99 -c 131072 \
  --simple-io -no-cnv -n 32 \
  -p "Hello world"

# 128K, KV on CPU Q8
llama-completion \
  -m /home/llm/models/Qwen3.5-0.8B-Q4_K_M.gguf \
  -ngl 99 -nkvo -c 131072 -ctk q8_0 -ctv q8_0 \
  --simple-io -no-cnv -n 32 \
  -p "Hello world"
```

**Status**: DONE

### Results: Q8_0 Draft at Multiple Context Sizes

Q8_0 chosen for maximum acceptance rate in speculative decoding.

| Context | KV (Q8) | Total VRAM | Prompt eval | Generation | Splits |
|---------|---------|------------|-------------|------------|--------|
| c4096   | 26 MiB  | 1295 MiB   | 277 t/s     | 35.4 t/s   | 2      |
| c65536  | 408 MiB | 1678 MiB   | 331 t/s     | 35.2 t/s   | 2      |

**Findings (empty/sparse KV):**
- 64K fits: 1678 / 2021 MiB (83%), 343 MiB headroom
- Generation speed flat at ~35 t/s with sparse KV
- Only 2 graph splits (good — minimal CPU↔GPU data crossing)
- Output coherent at all context sizes
- 35 t/s vs 49 t/s for Q4_K_M (Q8 is 2x bytes, ~28% slower — expected)

### Results: Full Context Load Stability Test

Filled KV cache with 55K tokens (55000 random words) at c65536.

| KV fill | Prompt eval | Generation | VRAM used | VRAM free |
|---------|-------------|------------|-----------|-----------|
| ~40 tok | 331 t/s     | 35.2 t/s   | 1678 MiB  | 343 MiB   |
| 2118 tok| 395 t/s     | 34.0 t/s   | 1295 MiB  | 688 MiB   |
| **55001 tok** | **145.7 t/s** | **18.85 t/s** | **1857 MiB** | **191 MiB** |

**Key findings:**
- **STABLE at full context** — no OOM, no crash, coherent output
- Generation drops from 35→18.9 t/s at full KV (attention scanning 55K tokens)
- VRAM 91% full (191 MiB free) — tight but stable
- Prompt eval 6.86 ms/token at full context (145.7 t/s batch)
- Total prompt eval time: 377s (6.3 min) for 55K tokens

**Impact on speculative decoding:**
At full 64K context, draft runs at ~19 t/s. With target at ~1 t/s (scalar),
spec decode effective throughput ≈ 1.5-2 t/s. After SSSE3 kernel (target
→ 4+ t/s), spec decode becomes more effective as draft has more headroom.

TODO: 128K f16 (OOM test), 128K -nkvo Q8 (PCIe penalty measurement).

---

## Step 1: Baseline CPU Run

Establish raw CPU-only performance for 122B target model before any
optimizations. Smoke test with short generation, then measure t/s.

```bash
numactl --interleave=all llama-completion \
  -m /home/llm/models/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf \
  -t 12 --numa distribute --no-mmap --mlock \
  -c 512 -n 32 --simple-io -no-cnv \
  -p "The theory of general relativity"
```

**Success criteria**: Model loads, generates coherent output, t/s measured.

**Status**: DONE

### Results: CPU Target Baseline

| Context | Prompt eval | Generation | Graph splits (bs=1) |
|---------|-------------|------------|---------------------|
| c512    | 2.25 t/s    | 0.95 t/s   | 97                  |
| c4096   | 2.64 t/s    | 0.98 t/s   | 97                  |
| c65536  | 2.37 t/s    | 1.01 t/s   | 97                  |

- Generation flat at ~1 t/s across context sizes (bandwidth-bound on scalar)
- Vulkan GDN shaders help: 0.95 t/s with GPU GDN vs 0.84 t/s pure CPU
- 97 graph splits: scheduler routes GDN→GPU, MoE matmul→CPU (correct split)
- Load: 167s (77 GB, --no-mmap, numactl --interleave=all)
- VRAM: 1276 MiB compute only. RAM: 73.6 GB.

---

## Step 2: 1 GB Huge Pages

Westmere has NO L2 TLB for large pages — only L1 DTLB entries:

| Page size | TLB entries | TLB reach |
|-----------|-------------|-----------|
| 4 KB      | 64 L1 + 512 L2 | 2 MB   |
| 2 MB      | 32 L1 only     | 64 MB  |
| 1 GB      | 4 L1 only      | 4 GB   |

MoE access pattern: 8 of 256 experts per layer (~48 MB), scattered across
77 GB. With 4 KB pages, every expert jump = TLB miss. With 2 MB pages,
one layer barely fits in 32 L1 entries (64 MB) and thrashes on layer
transition. With 1 GB pages, 4 entries cover 4 GB — multiple layers'
active regions fit comfortably.

`pdpe1gb` confirmed on X5650. 77 GB model = 77 pages at 1 GB granularity.

### Implementation

```bash
# Pre-allocate 1 GB huge pages (must be done early, needs contiguous RAM)
# 77 GB model + headroom = 85 pages
echo 85 | sudo tee /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages

# Verify allocation succeeded
cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
cat /sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages

# Mount hugetlbfs for 1 GB pages
sudo mkdir -p /dev/hugepages1G
sudo mount -t hugetlbfs -o pagesize=1G none /dev/hugepages1G

# Also enable THP as fallback for smaller allocations
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
```

**Challenge**: llama.cpp does not use `MAP_HUGETLB` or `madvise(MADV_HUGEPAGE)`.
Options: (a) patch mmap path to add `MAP_HUGETLB`, (b) use `LD_PRELOAD`
interposer, (c) enable THP + `madvise(MADV_HUGEPAGE)` via patch. Option (a)
is cleanest.

**Expected impact**: 10-20% throughput improvement for MoE scattered access.

**Status**: DONE — marginal improvement

### Results

Successfully allocated 1GB huge pages for model weights via code patches:
- `ggml.c`: `ggml_aligned_malloc` uses `mmap(MAP_HUGETLB|MAP_HUGE_1GB)` for ≥512MB
- `ggml-vulkan.cpp`: redirect Vulkan_Host ≥256MB allocs to CPU buffer (enables huge pages)
- Model splits into ~76 buffers of ~960MB each, each gets 1 × 1GB page

| Config | Generation | vs Baseline |
|--------|-----------|-------------|
| Baseline (4KB pages) | 0.98 t/s | — |
| THP (2MB, transparent) | 1.06-1.12 t/s | +8-14% |
| **1GB huge pages** | **1.03 t/s** | **+5%** |

**Conclusion**: Huge pages give marginal improvement (~5-14%) on Westmere.
The bottleneck is scalar Q4_K dequant compute, not TLB misses. THP (2MB)
actually performed slightly better than 1GB pages due to less memory waste
(960MB buffers rounded up to 1GB waste 64MB each = ~5GB total).

Huge pages will matter more AFTER the SSSE3 kernel fix makes inference
bandwidth-bound instead of compute-bound.

---

## Step 3: NUMA Optimization

Optimal NUMA configuration for 77 GB MoE model spanning both nodes.

### Strategy

```bash
# Disable kernel NUMA balancing (prevents page migration overhead)
echo 0 | sudo tee /proc/sys/kernel/numa_balancing

# Drop page cache before loading (ensures clean NUMA placement)
echo 3 | sudo tee /proc/sys/vm/drop_caches

# Run with interleaved memory + distributed threads
numactl --interleave=all llama-completion \
  --numa distribute -t 12 --no-mmap --mlock \
  -m model.gguf ...
```

### Key flags

- `--no-mmap`: Read into malloc'd buffers (respects numactl policy).
  With mmap, page cache follows its own NUMA policy — unreliable.
- `--numa distribute`: Pin threads round-robin across NUMA nodes.
- `--mlock`: Prevent page-out during long 128K generation.
- `numactl --interleave=all`: Round-robin pages across both memory
  controllers for ~2x aggregate bandwidth.

### Alternative: single-node isolation

If the model were small enough (<90 GB with KV), `--membind=0 --cpunodebind=0`
with `--numa isolate -t 6` would avoid all QPI overhead. At 77 GB + 3 GB KV,
this is feasible but tight (80/96 GB), leaving only 16 GB for OS + buffers.
Test both.

**Expected impact**: 30-50% vs naive (no NUMA awareness).

**Status**: DONE

### Results: NUMA + Graph Splits + Allocation

Systematic benchmark progression on 122B at c4096, 16 tokens generated:

| Config | PP t/s | Gen t/s | Splits | Notes |
|--------|--------|---------|--------|-------|
| Baseline (scalar, 2N interleave, 12t) | 2.64 | 0.98 | 97 | Original Phase 23 Step 1 |
| + SSSE3 kernels | 3.38 | 1.07 | 97 | +9% gen from SIMD |
| + alloc (hugetlb, THP, prefault) | 3.07 | 1.06 | 97 | Within noise |
| Node 0 only, 6t | 2.61 | 1.23 | 97 | +15% from eliminating QPI |
| **Node 1 only, 6t** | **2.84** | **1.32** | **97** | +7% from clean node (no kernel) |
| Node 1, no Vulkan (VK_ICD=/dev/null) | 3.31 | **1.41** | **1** | **+7% from eliminating 97 splits** |

**Key findings:**
1. **NUMA: single node > interleave for MoE.** QPI latency on scattered expert
   access hurts more than 2x bandwidth helps. Node 1 (clean, no kernel) best.
2. **Graph splits: 97→1 = +7%.** Vulkan GDN on tiny batch=1 SSM ops costs more
   in sync overhead than it saves in compute. Pure CPU is faster.
3. **SSSE3: +50% prompt eval, +9% generation.** Compute-bound at batch>1,
   bandwidth-bound at batch=1.
4. **Allocation (hugetlb, THP, NUMA interleave, prefault):** Correctness
   improvement (mlock mutex, FADV_SEQUENTIAL) but negligible perf impact —
   MoE random access pattern dominates.
5. **Software prefetch:** No measurable impact. DDR3 DRAM row buffer misses
   are the bottleneck, not cache/TLB misses.

**Net: 0.98 → 1.41 t/s (+44%) from SSSE3 + NUMA node isolation + split elimination.**

**Optimal target command:**
```bash
VK_ICD_FILENAMES=/dev/null numactl --membind=1 --cpunodebind=1 \
  llama-completion -t 6 --no-mmap -ngl 0 -c 65536 \
  -m Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf
```

### Results: Speculative Decoding

| Draft model | K | Gen t/s | Accept | Notes |
|-------------|---|---------|--------|-------|
| 0.8B dense Q4_K_M (GPU) | 1 | 1.81 | 52% | Decent quality, thinking mismatch |
| 0.8B dense Q4_K_M (GPU) | 3 | 2.25 | 5% | Garbled output, SSM rollback bugs |
| 35B-A3B MoE IQ2_XXS (CPU) | 2 | 1.17 | 12% | Slower than standalone — both fight for bandwidth |

**After PR #20700 (MTP + recurrent state fixes):**

| Draft | K | t/s | Accept | Quality |
|-------|---|-----|--------|---------|
| 0.8B dense (CPU) | 1 | 1.63 | 73% | Good |
| 0.8B dense (CPU) | **2** | **1.76** | **42%** | **Good** |
| 0.8B dense (CPU) | 3 | 1.63 | 3% | Poor |
| 35B-A3B MoE (CPU) | 2 | 1.17 | 12% | Bandwidth contention |

PR #20700's recurrent state `copy_cell` + checkpoint rollback fixed the
garbled output at K=2 (was broken with PR #20075). Acceptance jumped from
11.5% → 42% at K=2.

**Best config: K=2, 0.8B dense draft, no Vulkan → 1.76 t/s (+25% over standalone)**

---

## Step 4: SSSE3 Q4_K Dequantization Kernel

**THE SINGLE BIGGEST OPTIMIZATION.** llama.cpp has no SSSE3-only path for
Q4_K dot products. Westmere (X5650) has SSSE3 but not AVX, falling through
to scalar C code — estimated 5-10x slower than AVX2.

### Current code path (ggml-cpu/arch/x86/quants.c)

```
#if defined __AVX2__     → fast (FMA + 256-bit)     — NOT AVAILABLE
#elif defined __AVX__    → medium (128+256 hybrid)   — NOT AVAILABLE
#else                    → scalar C (SLOW)           — THIS IS US
```

Westmere has `_mm_maddubs_epi16` (SSSE3), `_mm_madd_epi16` (SSE2), and all
128-bit SSE float ops. The building blocks exist for a proper SSSE3 path.

### Approach

Write `#elif defined __SSSE3__` path for `ggml_vec_dot_q4_K_q8_K` using:
- `_mm_maddubs_epi16` for uint8×int8 multiply-add (SSSE3)
- `_mm_madd_epi16` for horizontal pair-add (SSE2)
- `_mm_cvtepi32_ps` + `_mm_fmadd_ps` emulation via mul+add (SSE)
- 128-bit registers only (no 256-bit AVX)

### Target quant types

Priority order (by usage in 122B inference):
1. Q4_K (bulk of expert weights)
2. Q6_K (attention weights, output head)
3. Q5_K (some layers in UD dynamic quant)
4. Q8_K (upcast layers in UD XL)
5. Q2_K (aggressive layers in UD)

### Expected impact

3-5x speedup on all CPU matmul operations. This could push 122B target from
~2-3 t/s to ~8-12 t/s, making speculative decoding highly effective.

**Status**: DONE — SSSE3 Q4_K/Q5_K/Q6_K kernels implemented + allocation strategy

### Results: SSSE3 + Perfect Allocation

Backend tests: 105/105 pass (41 Q4_K, 11 Q5_K, 11 Q6_K, all sizes/shapes).

| Context | Prompt eval | Generation | vs Baseline |
|---------|-------------|------------|-------------|
| c512    | 3.38 t/s    | 1.07 t/s   | PP +50%, Gen +9% |
| c4096   | 3.07 t/s    | 1.06 t/s   | PP +36%, Gen +8% |

**Analysis**: SSSE3 improved prompt eval significantly (+50%) but generation
barely moved (+9%). At batch=1, generation is **memory-bandwidth-bound**:
MoE reads ~3.2 GB active weights/token, scattered across 77 GB. At 51 GB/s
DDR3 aggregate, theoretical floor is ~63 ms. We're at 930 ms — only 6.8% of
bandwidth due to random expert access thrashing DRAM row buffers.

Allocation improvements (MAP_POPULATE, NUMA interleave, THP, FADV_SEQUENTIAL)
are active but hard to isolate — generation is dominated by memory access
patterns, not page faults. SSSE3 will matter more for speculative decoding
where target verifies K>1 tokens in parallel (batch matmul benefits from SIMD).

---

## Step 5: KV Cache Quantization

Quantize KV cache on both draft and target to save memory and bandwidth.

### Draft (GPU)

```
-ctk q8_0 -ctv q8_0
```

Halves KV cache: 128K context drops from 1.5 GB to 768 MB, giving 700 MB
VRAM headroom. Minimal quality impact on draft (acceptance rate may drop
~1-2%, negligible).

### Target (CPU)

```
-ctk q8_0 -ctv q8_0
```

128K context KV drops from 3.0 GB to 1.5 GB. Not critical for RAM
(80 GB → 78.5 GB) but reduces memory bandwidth for attention layers.

**Expected impact**: Enables 128K context on draft; minor bandwidth savings
on target.

**Status**: TODO

---

## Step 6: Speculative Decoding Integration

Wire up draft (GPU) + target (CPU) speculative decoding pipeline.

### llama.cpp flags

```bash
numactl --interleave=all llama-speculative \
  --model /home/llm/models/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf \
  --model-draft /home/llm/models/Qwen3.5-0.8B-Q4_K_M.gguf \
  -ngl 0 -ngld 99 \
  --draft-max 2 \
  --numa distribute --no-mmap --mlock \
  -ctk q8_0 -ctv q8_0 \
  -c 4096 -n 256 --simple-io -no-cnv \
  -t 12 \
  -p "Explain the significance of the Higgs boson"
```

### Key parameters

- `-ngl 0`: Target model fully on CPU
- `-ngld 99`: Draft model fully on GPU
- `--draft-max 2`: K=2 draft tokens per cycle (optimal for our speed ratio)
- `-c 4096`: Start with moderate context, scale to 64K/128K after validation

### Tuning

1. Measure acceptance rate at K=1, 2, 3, 4
2. Find optimal K for maximum effective t/s
3. Scale context to 64K, then 128K
4. Compare 0.8B vs 2B draft (if 2B fits at target context)

### Concurrent pipeline

Check if llama-speculative overlaps draft generation with target verification
(async pipeline). If not, effective t/s = tokens_accepted / (draft_time + target_time).
If yes, effective t/s = tokens_accepted / max(draft_time, target_time).

**Expected impact**: 2-3x effective throughput vs CPU-only generation.

**Status**: DONE

### Results: Speculative Decoding (node 1, c512)

```
numactl --membind=1 --cpunodebind=1 llama-speculative \
  --model 122B-UD-Q4_K_XL --model-draft 0.8B-Q4_K_M \
  -ngl 0 -ngld 99 --draft-max K -t 6 --no-mmap -c 512
```

| K | Decoded t/s | Acceptance | vs baseline (1.32) |
|---|-------------|------------|---------------------|
| 1 | 1.81        | 52.4%      | +37%                |
| 2 | 1.93        | 11.5%      | +46%                |
| 3 | **2.25**    | 4.7%       | **+70%**            |
| 4 | 2.22        | 9.0%       | +68%                |

K=3 optimal at 2.25 t/s. Acceptance rate is low (0.8B is a weak draft for 122B)
but batch verification makes K>1 worthwhile — verifying K tokens in one forward
pass costs less than K separate passes.

Output quality: K=1 produces good text. K≥2 shows some repetition/degradation
due to low acceptance cascading through rejection sampling. For production use,
K=1 (1.81 t/s, 52% acceptance) is the safe choice.

---

## Execution Order

0. **Step 0** — Draft validation (get draft model right first)
1. **Step 1** — Baseline CPU run (establishes target ground truth)
2. **Step 2** — 1 GB huge pages (system config, no code changes)
3. **Step 3** — NUMA optimization (system config, no code changes)
4. **Step 4** — SSSE3 kernel (biggest code change, biggest impact)
5. **Step 5** — KV quantization (flags only)
6. **Step 6** — Speculative decoding (wire draft + target together)

Step 0 first: validate the draft model works at 64K before investing
in target-side optimization. Steps 2+3 are independent system tuning,
do them together. Step 4 is the high-effort/high-reward code change.
Steps 5+6 depend on 0-4 being validated.

## Performance Targets (revised with actuals)

| Milestone | Expected | **Actual** | Context | Notes |
|-----------|----------|------------|---------|-------|
| Step 0 draft | 18.9 | **18.9** | 64K | DONE: Q8_0, stable |
| Step 1 baseline | 0.95-1.01 | **0.98** | 4K | DONE: scalar, 97 splits |
| SSSE3 (4) | 8-12 | **1.07** | 4K | Gen bandwidth-bound, PP +50% |
| NUMA node 1 (3) | 3-5 | **1.32** | 4K | +35% cumulative |
| No Vulkan splits | — | **1.41** | 4K | +44% cumulative, 1 split |
| Spec decode (6) | 12-20 | **1.81** | 512 | K=1 dense→MoE only |

**Revised analysis:** Original estimates (8-12 t/s after SSSE3) assumed compute-bound
workload. Reality: 122B MoE at batch=1 is **DDR3 random-access bandwidth-bound**
(~5-8 GB/s effective vs 25.6 GB/s sequential due to scattered expert reads).
SSSE3 helps batch (prompt eval +50%) but not generation (+9%). The ceiling is
set by DRAM row buffer miss latency on MoE scatter access.

### Remaining Paths Forward

| Option | Effort | Expected gain | Notes |
|--------|--------|---------------|-------|
| Smaller quant (Q2_K_XL, 35 GB) | Zero (have file) | ~2x (halves reads) | Quality trade-off |
| Expert gather+pipeline | High | 20-50% | Contiguous staging buffer |
| Per-expert madvise(WILLNEED) | Medium | 10-30% | Kernel async prefetch |
| Upstream spec decode fixes | Wait | Unknown | PR #20075, #20700 |
| 16K | 55-65% | ~1.3x |
| 64K | 40-55% | ~1.1-1.2x |
| 128K | 30-45% | ~1.0-1.1x |
