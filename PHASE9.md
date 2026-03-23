# Phase 9: llama.cpp GPU Inference on gfx803

## Status: NOT STARTED

## Goal

Get llama.cpp to produce tokens using GPU compute on gfx803 (Polaris 12).
Model loads successfully; crashes during first inference dispatch.

## Known State

- **Model:** Qwen2.5-0.5B-Instruct-Q4_K_M.gguf (469MB, fits in 2GB VRAM)
- **Package:** `llama-cpp-rocm-polaris` b7376-1
- **GPU layers:** All layers load to GPU successfully (no VM fault)
- **Crash point:** `ggml_backend_sched_synchronize` during first prompt eval
- **Backtrace:** libamdhip64.so -> libhsa-runtime64.so -> pthread_mutex_lock -> std::terminate

## Hypotheses

1. **rocBLAS kernel failure on gfx803** — rocBLAS may emit an unsupported
   instruction or hit a code object loading failure for gfx803. rocBLAS was
   rebuilt with gfx803 targets but Tensile-generated kernels may have issues.

2. **ggml-hip kernel incompatibility** — ggml's custom HIP kernels may use
   intrinsics or warp operations that behave differently on gfx8 (64-wide
   wavefronts, different instruction encoding).

3. **CP idle stall on barrier packets** — Phase 8 CPU-managed barriers may
   not cover all barrier paths exercised by llama.cpp's multi-stream dispatch.

4. **Memory pressure** — 2GB VRAM with hipMalloc->hipHostMalloc redirect may
   leave insufficient device memory for compute scratch/LDS.

## Investigation Plan

### Step 1: Isolate rocBLAS vs ggml-hip kernels
```bash
GGML_HIP_NO_BLAS=1 timeout 30 llama-cli \
  -m /opt/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -p "Hello" -n 4 -ngl 99
```
If this works -> rocBLAS is the problem.
If this crashes -> ggml-hip kernels are the problem.

### Step 2: Reduce GPU layers
```bash
timeout 30 llama-cli \
  -m /opt/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -p "Hello" -n 4 -ngl 1
```
Single GPU layer isolates whether the crash is layer-count dependent.

### Step 3: Enable HIP error reporting
```bash
AMD_LOG_LEVEL=3 timeout 15 llama-cli \
  -m /opt/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -p "Hello" -n 4 -ngl 99
```
Check for specific HIP/HSA error codes before the crash.

### Step 4: Check dmesg for GPU faults
```bash
sudo dmesg | grep -i 'amdgpu\|kfd\|gfx\|fault\|error'
```

### Step 5: Minimal HIP matmul test
Write a standalone test that dispatches a simple matrix multiply via
hipBLAS/rocBLAS on gfx803 to isolate whether rocBLAS kernels work at all.

## Success Criteria

- [ ] `llama-cli -p "Hello" -n 8 -ngl 99` produces 8 tokens on GPU
- [ ] Process exits cleanly (no crash, no hang)
- [ ] Token generation rate measured (baseline: 13.3 t/s CPU-only)
- [ ] 10 consecutive runs pass

## Packages

- `linux-lts-rocm-polaris` 6.18.16-34
- `hsa-rocr-polaris` 7.2.0-25
- `hip-runtime-amd-polaris` 7.2.0-30
- `rocblas-gfx803` 7.2.0-2
- `llama-cpp-rocm-polaris` b7376-1
