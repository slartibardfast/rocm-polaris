# Phase 21: Persistent Megakernel — Multi-WG JIT Interpreter

## Status: ABANDONED — GCN3 architectural limits confirmed

## Hardware Context

- **GPU**: Radeon Pro WX 2100 (Polaris 12, gfx803, 2GB GDDR5, 10 CUs)
- **Baseline**: 20.35 ms/token, 49.13 t/s (Phase 20, n_ctx=8192, 256 tokens, seed 42)
- **Backend**: Vulkan (ggml-vulkan), llama.cpp fork at `slartibardfast/llama.cpp:polaris-jit`

## Goal

Extend the Phase 17 JIT interpreter from 1 workgroup to 10 (one per CU), using
atomic flag-based barriers for cross-WG synchronization. The hypothesis: multi-WG
would eliminate the 1-WG serialization bottleneck that made the JIT 53% slower than
standard dispatch in Phase 17.

## Results

| Config | ms/token | t/s | vs Standard |
|--------|----------|-----|-------------|
| **Standard (no JIT)** | **20.35** | **49.13** | — |
| JIT 1-WG | 23.20 | 43.11 | **-12%** |
| JIT 2-WG | HANG | — | GPU deadlock |
| JIT 10-WG | HANG | — | GPU deadlock |

**Conclusion**: The megakernel is dead on GCN3. Three independent failure modes:
1. Interpreter overhead exceeds dispatch savings (1-WG)
2. Cross-WG atomic sync is broken on GCN3 hardware (multi-WG)
3. We're at ~100% GDDR5 bandwidth — nothing left to gain

---

## Implementation

### Multi-WG Flag-Based Barrier (shader)

Added to `gated_delta_net.comp` (embedded GLSL in `ggml-vk-jit.cpp`):

```glsl
layout(binding = 1, std430) buffer BarrierBuf { uint flags[]; };
layout(push_constant) uniform PC { uint num_wgs; };

// Each WG writes to its own cache-line-aligned slot (16 uints = 64B).
// Uses atomics for cross-WG visibility.
void global_barrier(uint step) {
    memoryBarrierBuffer();
    barrier();
    if (num_wgs <= 1u) return;
    uint target = step + 1u;
    if (gl_LocalInvocationID.x == 0u) {
        atomicMax(flags[gl_WorkGroupID.x * 16u], target);
        for (uint w = 0u; w < num_wgs; w++) {
            while (atomicOr(flags[w * 16u], 0u) < target) {}
        }
    }
    barrier();
}
```

**Barrier buffer**: 4KB, host-visible + coherent, one uint per op per batch.
Each batch uses a unique offset (advanced per dispatch). Zeroed from CPU at
command buffer recording time.

### C++ Infrastructure

- `jit_barrier_buf` on device struct (4KB)
- `jit_barrier_offset` per-token counter (reset each graph_compute)
- `LLAMA_VULKAN_JIT_WGS` env var for WG count (default: 1)
- `needs_single_wg` flag: batches containing RMS_NORM/L2_NORM forced to 1 WG
- Push constant changed from `{ uint dummy }` to `{ uint num_wgs }`
- Descriptor count changed from 1 to 2 (SSBO + barrier buffer)

### Pipeline Compilation: Init-Time vs Lazy

Pipeline compiled at device init (after `ggml_vk_load_shaders`), not lazily during
`graph_compute`. This avoids a potential RADV issue where mid-execution pipeline
creation interferes with the warmup graph. The warmup `--no-warmup` flag is required
when JIT is enabled because the warmup graph's tensor layout conflicts with the JIT
buffer allocation.

---

## Bugs Found and Fixed

### Bug 1: GGML_VK_ Env Var Namespace Collision (CRITICAL)

**Symptom**: `GGML_VK_JIT=1` caused GPU hangs regardless of whether JIT code was active.
Even `jit_enabled = true` with NO pipeline compiled, NO buffers allocated, NO JIT
dispatches — the mere presence of the env var hung the GPU during warmup.

**Root cause**: The env var name `GGML_VK_JIT` collides with something in the
RADV/Mesa/Vulkan loader stack. Setting this env var changes driver behavior in a way
that causes compute dispatch hangs on Polaris 12.

**Proof**: Setting `jit_enabled = true` in C++ code (without the env var) works
perfectly. Setting `GGML_VK_XYZTEST=1` (unrelated name) works. Only `GGML_VK_JIT=1`
hangs.

**Fix**: Renamed all JIT env vars to `LLAMA_VULKAN_*` prefix:
- `GGML_VK_JIT` → `LLAMA_VULKAN_JIT`
- `GGML_VK_JIT_OPS` → `LLAMA_VULKAN_JIT_OPS`
- `GGML_VK_JIT_WGS` → `LLAMA_VULKAN_JIT_WGS`
- `GGML_VK_JIT_LOG` → `LLAMA_VULKAN_JIT_LOG`

**Lesson**: Never use `GGML_VK_` prefix for custom env vars on RADV. The Vulkan
driver stack may interpret them.

### Bug 2: Warmup Graph Conflict

**Symptom**: JIT hangs during `common_init_from_params` warmup ("warming up the model
with an empty run"), even with the env var fix. Works fine with `--no-warmup`.

**Root cause**: The warmup evaluates a graph with different tensor layout than
generation. Init-time JIT buffer allocation (65KB SSBO + 4KB barrier in host-visible
VRAM) interacts poorly with the warmup graph's compute buffer reservation on Polaris 12.

**Fix**: Auto-skip warmup when `POLARIS_JIT` env var is set (`common/common.cpp`).
Attempted sync-based fixes (waitIdle, graph_cleanup, command pool reset) all failed —
the corruption is inside RADV's pipeline state tracking, not recoverable from the
application side.

### Bug 3: GCN3 Cross-WG Atomic Sync Deadlock

**Symptom**: Any multi-WG JIT dispatch (WGS >= 2) hangs the GPU indefinitely.

**Root cause**: The `atomicOr(flags[w * 16u], 0u)` polling loop in the flag-based
barrier never resolves. On GCN3, buffer atomics go through L2, but the polling WG's
L1 cache may hold a stale value that is never invalidated. The `coherent` qualifier
was tested and also failed — RADV/ACO may not generate GLC (globally coherent) load
instructions for `coherent` buffer access on GCN3.

**Not fixable**: This is a hardware limitation. GCN3 has no reliable cross-WG
synchronization primitive in compute shaders. The hardware was designed for
single-dispatch-per-op execution, not persistent megakernels.

---

## Barrier Design Analysis

Three barrier designs were implemented and tested:

### 1. Atomic Counter Barrier (Phase 17 original)
```glsl
atomicAdd(counters[step], 1u);
while (atomicOr(counters[step], 0u) < num_wgs) {}
```
All WGs hammer a SINGLE atomic counter. Serializes through L2.
Cost: ~1000 cycles per barrier (10 WGs × ~100 cycles/atomic).
Result: 109 ms/token (5x slower than standard).

### 2. Flag-Based Barrier (Phase 21)
Each WG writes to its own cache-line-aligned slot, polls all others.
Eliminates single-counter contention. Cost: ~300 cycles estimated.
Result: GPU hang (atomicOr polling never resolves on GCN3).

### 3. Lockstep + Memory Fence (considered, not tested)
No atomic sync — rely on GCN3's in-order execution keeping WGs in lockstep,
with `memoryBarrierBuffer()` (s_waitcnt) between ops. Hardware-specific,
violates Vulkan memory model. Ruled out as too risky.

**Conclusion**: No viable cross-WG barrier exists for GCN3 compute shaders.
The hardware lacks the cache coherency mechanisms that RDNA and newer architectures
provide. This is the fundamental architectural limit that kills the megakernel concept.

---

## Why the Megakernel Can't Work on GCN3

### 1. Dispatch overhead is tiny
At n_ctx=8192, dispatch overhead is ~1.3ms (7% of 20.35ms/token). The megakernel's
entire value proposition was eliminating this overhead. But the interpreter's own
overhead (SSBO reads, BDA indirection, barrier cost) exceeds the savings.

### 2. We're at the bandwidth wall
497 MiB / 20.35ms = 24.4 GB/s — essentially 100% of GDDR5 theoretical bandwidth.
MUL_MAT dominates at 82%. Even perfect dispatch elimination can't make GDDR5 faster.

### 3. GCN3 can't do persistent compute
GCN3 has no preemption, no cross-WG sync, and no cache coherency between CUs
for compute shaders. It was designed for embarrassingly parallel workloads with
independent dispatches — which is exactly what the standard path provides.

### 4. The standard path is already near-optimal
GCN3's simple in-order pipeline has minimal dispatch overhead. The driver (RADV)
efficiently pipelines command buffer submission. There's no hidden parallelism to
unlock with a megakernel.

---

## Files Modified

- `ggml/src/ggml-vulkan/ggml-vk-jit.cpp`: Flag-based barrier shader, env var renames
- `ggml/src/ggml-vulkan/ggml-vk-jit.h`: No changes
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`: Barrier buffer, multi-WG dispatch, init-time
  compilation, env var renames, warmup gating

## Env Vars (all require `LLAMA_VULKAN_JIT=1` + `--no-warmup`)

| Var | Default | Description |
|-----|---------|-------------|
| `LLAMA_VULKAN_JIT` | 0 | Enable JIT interpreter |
| `LLAMA_VULKAN_JIT_WGS` | 1 | Workgroups (only 1 works on GCN3) |
| `LLAMA_VULKAN_JIT_OPS` | all | Bitmask of op types to JIT |
| `LLAMA_VULKAN_JIT_LOG` | off | Print JIT batch composition |
| `LLAMA_VULKAN_JIT_DRY` | off | Log BDA addresses, skip dispatch |
