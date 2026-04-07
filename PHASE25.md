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

All tests: VK_ICD_FILENAMES=/dev/null, --no-mmap, -ngl 0, c512, n64.
CPU governor set to `performance` (2.66 GHz locked).

| Config | Threads | Gen t/s | PP t/s | Splits | Notes |
|--------|---------|---------|--------|--------|-------|
| **B1: Node 1 only** | **6** | **4.97** | **10.06** | **1** | **BEST for gen (peak 4.99)** |
| B2: Asymmetric (data N1, cores both) | 12 | 3.54 | 11.09 | 1 | QPI penalty > core benefit |
| B3: Interleaved (data both) | 12 | 3.45 | 11.33 | 1 | Worst for gen |
| B4: Node 1, 5 threads | 5 | 4.93 | 8.57 | 1 | Same gen, slower PP |
| B4: Node 1, 4 threads | 4 | 4.63 | 7.48 | 1 | Fewer cores hurts |
| B5: GPU offload (ngl=5) | 6 | 4.50 | 8.85 | 56 | Graph splits kill it |
| **9B dense (comparison)** | **6** | **2.37** | **5.65** | **1** | **35B-A3B is 2.1× faster** |

### Burst vs Sustained (B1 config, performance governor)

| Tokens | Gen t/s | Notes |
|--------|---------|-------|
| 63 (3 runs) | 4.91, 4.96, **4.99** | Peak essentially AT 5 t/s |
| 255 | 4.79 | Slight degradation (thermal? KV growth?) |

### Key Findings

1. **Single node 6t wins** — same pattern as Phase 23 (122B). DRAM access pattern, not compute.
2. **Performance governor matters** — schedutil sat at ~2.0 GHz, locking to 2.66 GHz gave +0.09 t/s.
3. **35B-A3B (1.43 GB active) is 2.1× faster than 9B dense (5.3 GB all-weights)** — MoE advantage huge.
4. **Effective DDR3 throughput**: 1.43 GB × 4.97 = **7.1 GB/s** (28% of 25.6 GB/s peak).
5. **Burst 4.99 t/s, sustained 4.79 t/s** — 0.01-0.21 t/s short of 5 t/s target.

## Phase C: Realistic 64K Context (OpenClaw target)

The 4.97 t/s above is at empty context (6 prompt tokens). For OpenClaw at
64K context window, the question is how generation degrades with realistic
KV fill. Test fixture: Wikipedia wikitext-2 prompt of ~3681 tokens, 32
generation tokens, c=65536.

| KV K type | KV V type | KV size | Gen t/s | PP t/s | Notes |
|-----------|-----------|---------|---------|--------|-------|
| f16 (system) | f16 | 1280 MiB | 2.18 | 9.95 | baseline |
| q4_0 (system) | q4_0 | 360 MiB | **4.06** | 9.92 | best built-in |
| f16 (mybuild) | f16 | 1280 MiB | 1.90 | 6.43 | mybuild baseline |
| q4_0 (mybuild) | q4_0 | 360 MiB | 3.18 | 6.09 | |
| f16 (mybuild) | q4_0 | 820 MiB | 2.79 | 5.47 | V comp alone helps |
| **TQ_KV_1B** (mybuild, SSE4.2) | f16 | 700 MiB | 2.11 | 4.22 | K comp helps a little |
| **TQ_KV_1B** (mybuild, SSE4.2) | q4_0 | 240 MiB | **3.43** | 6.33 | best on mybuild |

### Key Findings (Phase C)

1. **The c65536 "cliff" was illusory** — context allocation alone has no effect.
   The 4.97 → 2.18 drop is from filled KV cache (3681 valid positions),
   not the c65536 allocation size. Empty c65536 stays at ~4.8 t/s.

2. **V dequant is the dominant cost at filled context, not K dot product**.
   Compressing V alone (`-ctv q4_0`) helps almost as much as compressing
   both K and V. Compressing K alone with TurboQuant on top of f16 V
   only buys ~10%, because the V vec_mad path's f16→f32 dequant
   (scalar table lookup, no F16C on Westmere) saturates the inner loop.

3. **TurboQuant 1-bit K with proper SSE4.2 (XOR + 64-bit POPCNT) works**.
   Output is coherent ("The Sun's gravitational field will deflect light
   passing near it by an angle of approximately 1.75 arcseconds" — correct
   GR prediction). The Hamming attention path in
   `tq_kv_1b_attention_multi` is dispatched directly from
   `ggml_compute_forward_flash_attn_ext_f16_one_chunk`, with the per-thread
   scratch buffer reserved at graph build time in `params->wdata`. Per
   call, the inner loop is `_mm_xor_si128` + `_mm_extract_epi64` + 64-bit
   POPCNT, ~6 cycles per K block.

4. **A scalar fallback was nearly shipped silently**: `ggml-base` does not
   pass `-march=native`, so without per-source-file flags my SSE4.2 inner
   loop was excluded by the preprocessor and fell to byte-by-byte
   `popcount8`. Fixed by adding targeted COMPILE_OPTIONS for
   `ggml-turbo-quant.c` only. Verified `popcntq` in object file.

5. **My build still trails the system binary by ~25%** even with
   SSE4.2/F16C cmake flags on, NUMA isolated, and all governors tuned.
   System binary q4_0+q4_0 = 4.06 t/s, my best (TQ_K + q4_0_V) = 3.43.
   Some optimization in the system PKGBUILD's flags that we haven't
   identified. Investigating.

### Best Path to OpenClaw 5 t/s @ 64K

Production target = 5 t/s sustained at 64K filled context.

**Today's best on system binary**: `-ctk q4_0 -ctv q4_0` → **4.06 t/s** at 3681 KV fill.
This is 81% of target. The remaining gap closes with:

1. **SSE4.1 vectorized fp16→fp32** (general ggml fix)
   Westmere has no F16C, so `ggml_cpu_fp16_to_fp32` falls to a scalar
   table lookup. The flash attention V dequant calls this 588960 times
   per token at 3681 fill (160 outer iters × 3681 valid KV × 1 v_to_float
   dispatch). Vectorising 8 elements at a time via PMOVZXWD + bit
   manipulation removes the lookup table and unlocks V-side throughput.
   Same fix benefits q4_0 V (which dequants per block before mad) and
   any code path that touches f16 conversion on Westmere.

2. **Port `arch/x86/repack.cpp` to SSE4.1**
   Currently the entire matmul repack module is gated on `#if defined(__AVX__)`.
   Westmere is locked out. This is a runtime data-layout optimization for
   quantized matmul that the system binary uses on AVX hosts but our
   target hardware can't even build. Porting the F16C macros
   (`GGML_F32Cx8_LOAD` etc.) to a SSE4.1 path requires the same
   PMOVZXWD-based fp16→fp32 vectoriser as #1.

3. **Extend TurboQuant to V quantization** (less impactful given #1)
   Currently TurboQuant compresses K only. V uses standard f16/q4_0.
   A V-side TurboQuant variant could pair with the K path, but most of
   the V cost is in the dequant call, not the cache footprint.

## Phase D: OpenClaw 8K Fill Baseline (Step 0 of gap-closing plan)

Earlier measurements used a 3681-token fill. For the OpenClaw target
(8192 tokens realistic), the baseline is tighter:

### System Binary at 8K fill

| Metric | Value |
|--------|-------|
| Binary | /usr/bin/llama-completion (b8637+Phase24) |
| Config | `-ctk q4_0 -ctv q4_0`, c65536, n=256 |
| Fill | 7834 tokens (34500 bytes wikitext) |
| Load | 48.07 s |
| PP | 8.30 t/s (120.45 ms/tok × 7834 tokens = 943 s) |
| **Gen** | **3.37 t/s** (297.14 ms/tok × 255 tokens = 75.77 s) |
| Total wall time | 1019.7 s (17.0 min) |

The production target is **5.0 t/s sustained** at this fill level.
**Gap = +48%** (absolute), not the +25% previously stated for the
3681 fill. KV-cache growth from 3681 → 7834 positions costs another
17% of decode speed on the baseline config.

### Gap decomposition

Per-token time at baseline: 297 ms
Target per-token time at 5 t/s: 200 ms
Required savings: **97 ms per token**

Median expectation based on plan steps: ~150 ms saved → 147 ms/token
→ 6.8 t/s. Pessimistic: ~75 ms saved → 222 ms/token → 4.5 t/s (below
target). Optimistic is capped around 100 ms by DDR3 bandwidth.
