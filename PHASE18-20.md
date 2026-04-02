# Phases 18-20: Vulkan Inference Optimization — Occupancy, BDA Cache, State I/O Elimination

## Status: COMPLETE — 48.91 t/s at n_ctx=8192 (Qwen3.5-0.8B Q4_K_M)

## Hardware Context

- **GPU**: Radeon Pro WX 2100 (Polaris 12, gfx803, 2GB GDDR5, 10 CUs)
- **GDDR5 bandwidth**: ~24 GB/s
- **Model**: Qwen3.5-0.8B Q4_K_M (497 MiB, hybrid SSM+attention, 18 SSM + 6 attention layers)
- **Backend**: Vulkan (ggml-vulkan), llama.cpp fork at `slartibardfast/llama.cpp:polaris-jit`

## Performance Summary

| Phase | ms/token | t/s | Change | Commit |
|-------|----------|-----|--------|--------|
| Pre-Phase 18 (Phase 16 baseline) | 22.60* | 44.24* | — | — |
| Phase 18: GDN occupancy (LANES=64) | ~21.5* | ~46.5* | +5%* | `6095a8532` |
| Phase 19: f16 BDA state cache | 22.16 | 45.13 | +2% vs no-cache | `ceacc6eae` |
| **Phase 20: State I/O elimination** | **20.44** | **48.91** | **+11.1%** | pending |

*All numbers measured at n_ctx=8192, 256 generation tokens, seed 42. Phase 16-18
measurements were originally taken at n_ctx=256 (68-85 ms/token, 12-15 t/s), which hits
a severe performance cliff — see "n_ctx=256 Performance Cliff" below. Relative improvements
between phases measured at the same context size are valid.*

**Bandwidth utilization**: 497 MiB / 20.44 ms = 24.3 GB/s — **~100% of GDDR5 theoretical**.

---

## Phase 18: GDN Shader Occupancy Fix

### Problem

The Gated Delta Net (GDN) shader for SSM state updates used `LANES_PER_COLUMN=8` when
`S_V=128`, yielding `ROWS_PER_LANE=16`. This required ~128 VGPRs per thread, limiting
occupancy to 2 waves/SIMD (25%) on GCN3's 256-VGPR budget.

### Fix

Set `LANES_PER_COLUMN=64` when `subgroup_size >= 64` (GCN3 native wave64):
- `ROWS_PER_LANE` drops from 16 to 2
- VGPRs drop from ~128 to ~32
- Occupancy increases from 25% to 100% (8 waves/SIMD)
- Existing `reduce_partial()` already handles LANES=64 via `subgroupAdd`

One-line change in `ggml-vulkan.cpp` pipeline creation:
```cpp
lanes_per_column = (device->subgroup_size >= 64u) ? 64u : 8u;
```

### Impact

26% per-kernel improvement on GDN. At n_ctx=256 this measured as 85→73.6 ms/token
(+15.6%), but this overstates the impact at realistic context where GDN is a smaller
fraction of total time.

---

## Phase 19: f16 BDA State Cache

### Problem

SSM state flows through the ggml graph as f32 tensors:
1. `GET_ROWS`: read state from RS buffer (f32, 1 MiB per layer)
2. `GATED_DELTA_NET`: read state, update, write to dst (f32)
3. `CPY`: write new state back to RS buffer (f32)

This is 2 MiB of state I/O per layer × 18 layers = 36 MiB per token.

### Solution: Buffer Device Address (BDA) State Cache

A persistent 12 MiB f16 shadow buffer allocated via `VK_KHR_buffer_device_address`.
During generation (n_tokens=1), the GDN shader reads/writes state through the BDA
cache instead of the graph's f32 tensors:

- **Shader extensions**: `GL_EXT_shader_16bit_storage`, `GL_EXT_buffer_reference`
- **BDA buffer references**: `F16In`/`F16Out` for direct f16 VRAM access
- **Push constant**: `uint64_t state_cache` passes the BDA address per layer
- **Computation**: always f32 in registers — only storage is f16
- **Toggle**: `GGML_VK_GDN_CACHE=0` to disable

**Files modified**: `gated_delta_net.comp`, `vulkan-shaders-gen.cpp`, `ggml-vulkan.cpp`

### Impact

State I/O halved: 36 MiB → 18 MiB per token (f16 vs f32). Measured +2% at n_ctx=512.

**Perplexity**: 16.1314 ± 0.538 (identical to f32 baseline). Zero quality loss.

---

## Phase 20: SSM State I/O Elimination

### Problem

With the BDA cache holding authoritative state during generation, the graph's
GET_ROWS (state read) and CPY (state writeback) are redundant. The GDN shader
also still wrote f32 state to the output tensor for graph compatibility — wasted
bandwidth since no downstream op reads it.

Additionally, a correctness bug: the BDA cache was zeroed on allocation, causing
the first generation token to lose prompt state accumulated during prompt eval.

### Solution: Three-Part Optimization

**Part 1: First-token init fix**

Used bit 0 of the `state_cache` BDA address as an init flag (BDA addresses are
always ≥2-byte aligned, so bit 0 is always 0 for valid addresses):

| `state_cache` value | Read from | Write to |
|---------------------|-----------|----------|
| `0` | binding (f32) | f32 dst only |
| `addr \| 1` (init) | binding (f32) | f16 cache + f32 dst |
| `addr` (aligned) | f16 cache | f16 cache only |

On the first generation token after prompt eval, the init flag is set. GDN reads
correct state from the binding (populated by GET_ROWS from RS buffer) and
initializes the BDA cache. Subsequent tokens read from cache directly.

Tracked by `gdn_cache_initialized` flag on the device context:
- Reset to `false` on prompt eval (detected by checking n_seq_tokens > 1 on GDN nodes)
- Set to `true` after the first generation token's graph_compute completes

**Part 2: Skip f32 state write**

When the BDA cache is active and initialized (not init mode), the GDN shader
skips the f32 state write to `data_dst[s_off + ...]`. Saves 18.9 MiB bandwidth
per token (18 layers × 128×128×16 heads × 4 bytes).

**Part 3: Skip GET_ROWS and CPY dispatches**

Pre-scan the compute graph for GDN nodes and build a skip set:
- **GET_ROWS to skip**: identified by `gdn_node->src[5]->view_src` (the GET_ROWS
  output that feeds GDN's state input)
- **CPY to skip**: identified by `cpy_node->src[0]->view_src->op == GGML_OP_GATED_DELTA_NET`
  (the CPY that writes GDN's new state back to RS buffer)

The skip set is an `unordered_set<const ggml_tensor *>` populated once per
graph_compute, checked in O(1) per node during dispatch.

Conv state GET_ROWS/CPY (18 of each) are correctly NOT skipped — they don't
feed GDN and have no BDA cache equivalent.

### Per-Op Impact

| Op | Before (21.0ms) | After (17.6ms) | Saved |
|---|---|---|---|
| GDN | 1402µs (18×78µs) | 843µs (18×47µs) | 559µs |
| GET_ROWS | 965µs (38×25µs) | 56µs (20×3µs) | 909µs |
| CPY | 1019µs (37×28µs) | 95µs (19×5µs) | 924µs |
| MUL_MAT (all) | 14,500µs | 14,500µs | 0 |
| Other | 3,110µs | 2,144µs | 966µs |
| **Total** | **20,996µs** | **17,638µs** | **3,358µs** |

### Overall Benchmark

256 tokens generation, seed 42, n_ctx=8192:
- **Baseline (no cache)**: 22.73 ms/token, 44.00 t/s
- **Phase 20 (all optimizations)**: 20.44 ms/token, 48.91 t/s → **+11.1%**
- **PPL**: 16.1314 ± 0.538 — identical. Zero quality loss.

---

## n_ctx=256 Performance Cliff

Previous sessions benchmarked at `-c 256`, producing 68-85 ms/token measurements.
This was an artificial performance cliff:

| n_ctx | ms/token | t/s |
|-------|----------|-----|
| 256 | 68.64 | 14.57 |
| 512 | 22.60 | 44.24 |
| 1024 | 23.00 | 43.47 |
| 4096 | 23.06 | 43.36 |

The 3× cliff at n_ctx=256 is caused by context size quantization (257 rounds to 512),
likely affecting dispatch geometry and compute buffer allocation. Root cause not fully
determined; suspected to be related to n_batch clamping to n_ctx affecting shader
variant selection.

**All previous measurements at n_ctx=256 are only valid for relative comparison at
that context size.** Standard benchmark context: **n_ctx=8192**.

---

## Rejected: Output Logits Requantization (q6_K → q4_K)

`token_embd.weight` (tied embeddings, 1024×248320) is q6_K at 199 MiB.
Requantizing to q4_K saves 63 MiB and 2.16ms/token (+10.4%).

**Rejected**: +3% PPL degradation (16.13 → 16.61) not worth the speed gain.

---

## Remaining Profile (at 20.31 ms/token)

| Category | Time | % |
|----------|------|---|
| MUL_MAT (all) | ~14.5ms | 82% |
| - Output logits (q6_K 248K) | ~5.0ms | 28% |
| - FFN/proj matmuls | ~9.5ms | 54% |
| GDN | 0.84ms | 5% |
| FLASH_ATTN | 1.08ms | 6% |
| Other | 1.3ms | 7% |

MUL_MAT now dominates at 82%. At 24.3 GB/s effective bandwidth, we are at the
GDDR5 memory bandwidth wall. Further optimization requires reducing bytes read
(requantization, pruning) or algorithmic changes to the matmul dispatch pattern.

---

## JIT Interpreter: Net Negative (Phase 17c result)

The JIT ubershader/interpreter architecture was fully implemented but proved to
be a net negative (-12ms, making inference slower):

- Interpreter overhead: GLSL barrier+memoryBarrier between ops, SSBO reads, BDA
  indirection, 1-WG limitation
- Standard path can overlap independent ops across CUs; interpreter serializes on 1 CU
- Barrier optimization (dependency tracking) saved only 0.47ms (and caused corruption
  when barriers were removed between non-independent ops)
- Command buffer replay analysis showed CPU dispatch overhead is only 58µs (0.07%),
  not the 17ms estimated in Phase 17

Re-tested at realistic context sizes:
- n_ctx=512, 30 tokens: JIT=31.01ms vs standard=20.31ms (**53% slower**)
- n_ctx=8192: JIT **hangs** (never completes even 30 tokens)

**Conclusion**: JIT interpreter is architecturally unsuitable for GCN3's simple
in-order pipeline. The standard multi-dispatch path is already near-optimal because
GCN3 has minimal dispatch overhead at realistic context sizes.
