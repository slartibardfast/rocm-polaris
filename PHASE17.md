# Phase 17: GCN3 Ubershader — GPU-Driven Persistent Inference

## Context

Phase 16 achieved 24.6 t/s (output-only q4_K model, b8508-13). Deep ISA analysis
proved the matmul-vec shaders are near-optimal (79% MAC fusion, DPP reduction,
proper load/ALU interleaving). The remaining bottleneck is **dispatch overhead**:

- 340 dispatches/token × ~50µs each = **~17ms overhead on 40.7ms token (42%)**
- Actual compute only needs ~24ms at 67% bandwidth utilization
- Theoretical without dispatch overhead: **42 t/s (+70%)**

Every attempt to improve per-shader efficiency (LARGE WG, NUM_ROWS tuning, GDN
occupancy) REGRESSED because it increased dispatch count. On 5 CUs, dispatch
cost dominates. The only way forward is **eliminating dispatches**.

## Design: Per-Layer Ubershader with Atomic Barriers

Replace ~340 per-op dispatches with ~25 per-layer ubershaders. Each ubershader
internally sequences operations using atomic barriers (500 cycles) instead of
dispatch boundaries (50,000 cycles) — **100× cheaper synchronization**.

### Architecture

```
Current:   [rms_norm] → dispatch → [matmul] → dispatch → [silu] → dispatch → ...
                        ↑ 50µs      ↑ 50µs     ↑ 50µs

Ubershader: [  rms_norm → barrier → silu → barrier → add → barrier → ...  ]
             ↑ single dispatch          ↑ 0.5µs each
```

### Key Enabling Technologies (all confirmed available on Polaris 12)

1. **VK_KHR_buffer_device_address**: Already implemented in codebase (`buf->bda_addr`).
   Eliminates descriptor sets — shader accesses any buffer by 64-bit pointer.
   **Faster than descriptors on AMD** (1 vs 2 indirections).

2. **GL_EXT_buffer_reference**: Already used in `im2col.comp`, `flash_attn_cm2.comp`.
   Pattern: `layout(buffer_reference) buffer Ptr { float d[]; };`

3. **Atomic global barrier**: 5 workgroups (1 per CU) = minimal contention.
   Cost: ~500 cycles vs ~50,000 per dispatch boundary. 100× improvement.

## Implementation Phases

### Phase 17a (MVP): Elementwise Ubershader

Fuse all non-matmul ops between matmul boundaries into one persistent dispatch.

**New file: `ubershader_elem.comp` (~300 lines)**

```glsl
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(buffer_reference, std430) buffer FloatBuf { float d[]; };

struct MegaCmd {
    uint op_type;
    uint element_count;
    uint64_t src0_ptr;    // BDA pointers
    uint64_t src1_ptr;
    uint64_t src2_ptr;
    uint64_t dst_ptr;
    uint params[4];       // op-specific (head_dim, eps, etc.)
};

layout(binding = 0) readonly buffer CmdBuf { MegaCmd cmds[]; };
layout(binding = 1) buffer BarrierBuf { uint counters[]; };

layout(push_constant) uniform PC {
    uint num_cmds;
    uint num_wgs;
};

void global_barrier(uint step) {
    memoryBarrierBuffer();
    if (gl_LocalInvocationID.x == 0) {
        atomicAdd(counters[step], 1u);
        while (atomicOr(counters[step], 0u) < num_wgs) {}
    }
    barrier();
    memoryBarrierBuffer();
}

void main() {
    uint wg = gl_WorkGroupID.x;

    for (uint i = 0; i < num_cmds; i++) {
        MegaCmd cmd = cmds[i];
        FloatBuf src0 = FloatBuf(cmd.src0_ptr);
        FloatBuf dst  = FloatBuf(cmd.dst_ptr);

        // Grid-stride loop: each WG handles a slice of elements
        uint stride = num_wgs * gl_WorkGroupSize.x;
        uint start = wg * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

        switch (cmd.op_type) {
            case 0: // ADD
                for (uint j = start; j < cmd.element_count; j += stride)
                    dst.d[j] = src0.d[j] + FloatBuf(cmd.src1_ptr).d[j];
                break;
            case 1: // SCALE
                float scale_val = uintBitsToFloat(cmd.params[0]);
                for (uint j = start; j < cmd.element_count; j += stride)
                    dst.d[j] = src0.d[j] * scale_val;
                break;
            case 2: // SILU
                for (uint j = start; j < cmd.element_count; j += stride) {
                    float x = src0.d[j];
                    dst.d[j] = x / (1.0 + exp(-x));
                }
                break;
            case 3: // GATE_PREP (add + softplus + mul)
                // ... fused implementation
                break;
            case 4: // RMS_NORM (needs shared mem reduction)
                // ... hierarchical reduction across WGs
                break;
        }

        global_barrier(i);
    }
}
```

**C++ integration (ggml-vulkan.cpp):**
- During graph walk, detect non-matmul op sequences
- Accumulate commands into GPU command buffer (via mapped SSBO)
- At matmul boundary: dispatch ubershader → dispatch matmul
- BDA pointers obtained from existing `buf->bda_addr` infrastructure

**Ops handled in Phase 17a:** ADD, MUL, SCALE, CPY, SILU, SIGMOID, CONT,
GATE_PREP, L2_NORM (element-wise portion), GLU/SWIGLU

**Ops remaining as individual dispatches:** MUL_MAT_VEC (all quant types),
FLASH_ATTN_EXT, GATED_DELTA_NET, SSM_CONV, GET_ROWS, SET_ROWS, RMS_NORM,
ROPE, CONCAT

**Expected savings:** ~100 dispatch boundaries eliminated × 50µs = ~5ms
→ **24.6 → ~28 t/s**

### Phase 17b: Batched Matmul-Vec

Batch same-format matmuls within a layer into a single dispatch. The shader
reads a command list of (weight_ptr, input_ptr, output_ptr, M, K) tuples
via BDA, processing them sequentially with atomic barriers.

**New file: `batched_mmv_q4_k.comp` (~200 lines)**
- Copies the q4_K matmul-vec inner loop from `mul_mat_vec_q4_k.comp`
- Wraps it in a command-reading outer loop
- Each command specifies weight BDA pointer, input/output BDA, dimensions
- Atomic barrier between commands

Per SSM layer: 5 matmuls (qkv, z, alpha, beta, out) → 1 batched dispatch
Per attention layer: 4 matmuls (qkv, out + 2 FFN) → 1 batched dispatch

**Expected savings:** ~80 dispatch boundaries × 50µs = ~4ms
→ **~28 → ~31 t/s**

### Phase 17c: Full Layer Ubershader

Combine Phase 17a and 17b: single dispatch per layer that alternates
matmul and elementwise phases with atomic barriers between them.

- 24 layers = 24 dispatches + 1 final logits = 25 total (from 340)
- Eliminates ~315 dispatch boundaries × 50µs = ~15.75ms
- GDN (128 VGPRs) included as a step — occupancy drops for that step
  but recovers for subsequent steps

**Expected result:** **~38-42 t/s** (approaching theoretical ceiling)

## Register Budget Analysis

The ubershader switch statement means ACO allocates VGPRs for the
heaviest branch. Element-wise ops need ~16-24 VGPRs. RMS_NORM needs
~24 VGPRs (shared mem reduction). Total max: ~32-40 VGPRs → 6-8
waves/SIMD. Excellent occupancy.

For Phase 17b (batched matmul): same as current matmul = 64 VGPRs →
4 waves/SIMD. No regression.

For Phase 17c (combined): matmul phase uses 64 VGPRs, element-wise
phase uses 32 VGPRs. ACO allocates for max = 64 VGPRs → 4 waves.
GDN phase uses 128 VGPRs → 2 waves (same as current, no regression).

## Global Barrier Cost Analysis

With 5 workgroups (1 per CU):
- 5 atomicAdd + 4 spins per barrier
- ~500 cycles at 1GHz = 0.5µs per barrier
- Phase 17c: ~20 barriers/layer × 24 layers = 480 barriers
- Total barrier cost: 480 × 0.5µs = **0.24ms** (vs 17ms dispatch overhead)

## Command Buffer Structure

```c
struct MegaCmd {          // 64 bytes, aligned
    uint32_t op_type;     // enum: ADD, SCALE, MATMUL_Q4K, RMS_NORM, GDN, ...
    uint32_t ne;          // element count or row count
    uint64_t src_ptrs[3]; // BDA pointers to source buffers
    uint64_t dst_ptr;     // BDA pointer to destination
    uint32_t params[4];   // op-specific parameters
};
```

- Per layer: ~20 commands × 64 bytes = 1280 bytes
- Per token: 24 layers × 1280 = ~30KB (fits easily in GPU SSBO)
- Command buffer is DETERMINISTIC per model — built once, reused every token
  (only intermediate buffer BDA addresses change, and those are stable)

## Correctness Verification

1. **Deterministic output match**: Reference from b8508-13 baseline
   ```
   diff <(GGML_VK_UBERSHADER_QWEN35=0 llama-completion ...) \
        <(GGML_VK_UBERSHADER_QWEN35=1 llama-completion ...)
   ```

2. **Backend ops test**: Run `test-backend-ops test -o MUL_MAT -b Vulkan0`
   (matmul path unchanged, should still pass 884/884)

3. **Perf logger comparison**: Verify same ops execute in same order

4. **Stress test**: 500-token generation, check for drift or corruption

## Key Files

- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
  - Line 920: `bda_addr` field on vk_buffer_struct
  - Line 2571: BDA buffer creation flags
  - Line 4584: BDA shader variant selection pattern (im2col)
  - Line 6499: `ggml_vk_dispatch_pipeline()` — current dispatch
  - Line 14400: Graph walk loop — insertion point for ubershader batching
- `ggml/src/ggml-vulkan/vulkan-shaders/im2col.comp` line 35: BDA pattern example
- `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q4_k.comp`: matmul inner loop to embed
- `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` line 1035: shader registration
- `llama-vulkan/PKGBUILD`: prepare() injection point

## Risk Mitigation

- **Fallback**: `GGML_VK_UBERSHADER_QWEN35=0` env var disables ubershader path entirely
- **Phased rollout**: Each phase is independently testable and revertable
- **Register pressure**: ACO pre-calculates occupancy; if >64 VGPRs, split the switch
- **Atomic barrier deadlock**: Only 5 WGs, all must reach barrier; if one crashes,
  detect timeout via CPU-side fence and fall back to standard dispatch
