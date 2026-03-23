# Phase 9: llama.cpp GPU Inference on gfx803

## Status: IN PROGRESS — ROOT CAUSE: KV cache read faults on 2nd pass

## Goal

Get llama.cpp to produce tokens using GPU compute on gfx803 (Polaris 12).
Model loads successfully; hangs during first inference dispatch.

## Known State

- **Model:** TinyLlama-1.1B-Q8_0 (1.1GB), Qwen2.5-0.5B-Q4_K_M (469MB)
- **Package:** `llama-cpp-rocm-polaris` b7376-1
- **GPU layers:** All layers load to GPU (H2D works, no VM fault)
- **CPU-only:** Works perfectly (TinyLlama: 15.1 t/s)
- **Hang point:** `ggml_cuda_mul_mat_vec_q` — quantized matrix-vector multiply
- **No GPU fault** in dmesg — GPU silently stops processing

## Investigation Results

### Hypotheses Eliminated

1. **rocBLAS** — GGML_HIP_NO_BLAS=1 still hangs. hipBLAS SGEMM works standalone.
2. **Flash attention** — `-fa off` still hangs (different kernel, same pattern).
3. **Memory pressure** — GPU_D2H_SAFE_THRESHOLD=0 still hangs. Only 19MB VRAM idle.
4. **Shared memory** — 8KB LDS kernels work (100 dispatches × 2 tests).
5. **Non-blocking streams** — 1000 dispatches on non-blocking stream pass.
6. **hipEvents** — 500 event record/wait cycles pass.
7. **H2D → compute transition** — 200MB H2D + 100 compute dispatches pass.
8. **hipBLAS + compute** — Full ggml-like pattern (24 layers norm+gemm+add) passes.
9. **Individual ggml ops** — add, mul_mat, rms_norm, softmax all pass via ggml API.
10. **Quantized MMVQ** — Q4_K, Q8_0, F32, F16 matrix-vector multiply all pass.
11. **ggml scheduler** — CPU+GPU backend sched with 50 iterations passes.
12. **test-backend-ops** — 217+ ops pass (GET_ROWS cross-backend has correctness issues).

### Backtrace (HIP_LAUNCH_BLOCKING=1, -fa off, -ngl 1)

```
#30 ggml_cuda_mul_mat_vec_q
#29 [ggml-hip dispatch]
#22-28 [CLR/HIP dispatch path]
#14-19 [ROCR signal wait — sched_yield polling]
#13 sched_yield
```

The GPU kernel dispatch goes into ROCR signal wait and never returns.
No GPU faults in dmesg — the kernel is accepted but never completes.

### ROOT CAUSE FOUND (2026-03-23)

With debug logging (`-ngl 99`):
- **First forward pass: ALL 797 GPU nodes complete** (22 layers, every op)
- **Second pass: HANGS at node 22 `kq-0 MUL_MAT`** — reads K cache from 1st pass

The `kq-0` operation does `MUL_MAT(K_cache, Q)` — it reads from the KV cache
populated during the first forward pass. VM faults at `0x4103xxx000` (TC read,
page not present) confirm the KV cache buffer is unmapped between passes.

With `-ngl 1`, the same pattern occurs: all 40 nodes complete, then VM fault.

**Root cause: KV cache buffer becomes unmapped between graph evaluations.**
This is either:
1. A ggml buffer reallocation that doesn't properly re-map on gfx803
2. Our hipMalloc→hipHostMalloc redirect corrupting buffer metadata
3. ggml's KV cache uses buffer views that lose their GPU mapping

## Next Steps

### Theory: Model loading leaves stale queue state

After H2D copies during model loading, CLR's internal queues may have
pending barrier deps or stale signal state. The first compute dispatch
on the same queue encounters this stale state and hangs.

Test: Rebuild llama.cpp with GGML_CUDA_DEBUG to log each operation,
or add fprintf to CLR's dispatch path to trace the exact AQL packet
sequence.

### Theory: Kernel resource requirements exceed gfx803 limits

The mmvq kernel for Q4_K with real model dimensions may require more
VGPRs/SGPRs than available, causing silent failure. Extract the gfx803
code object from libggml-hip.so and check resource requirements.

### Theory: fastdiv_values or integer division kernel helper

The mmvq kernel uses `init_fastdiv_values` and `fastdiv`/`fastmodulo`
which may compile to different code on gfx8. A division-by-zero or
infinite loop in the fast divider could hang the GPU.

## Success Criteria

- [ ] `llama-cli -p "Hello" -n 8 -ngl 99` produces 8 tokens on GPU
- [ ] Process exits cleanly (no crash, no hang)
- [ ] Token generation rate measured (baseline: 15.1 t/s CPU-only)
- [ ] 10 consecutive runs pass

## Packages

- `linux-lts-rocm-polaris` 6.18.16-34
- `hsa-rocr-polaris` 7.2.0-25
- `hip-runtime-amd-polaris` 7.2.0-30
- `rocblas-gfx803` 7.2.0-2
- `llama-cpp-rocm-polaris` b7376-1
