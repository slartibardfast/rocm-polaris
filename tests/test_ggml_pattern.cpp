// test_ggml_pattern.cpp — Reproduce ggml-hip's exact pattern:
// 1. Create hipBLAS handle
// 2. Create non-blocking stream
// 3. hipMalloc large buffers (model weights)
// 4. H2D copies (model loading)
// 5. Launch kernels + hipblasSgemm on the stream
// 6. hipStreamSynchronize
//
// Build: hipcc -o test_ggml_pattern test_ggml_pattern.cpp -lhipblas

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__global__ void rms_norm(float *out, const float *in, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        sum += in[i] * in[i];
    }
    smem[tid] = sum;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(smem[0] / n + 1e-6f);
    for (int i = tid; i < n; i += blockDim.x) {
        out[i] = in[i] * rms;
    }
}

__global__ void add_vectors(float *c, const float *a, const float *b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main(int argc, char **argv) {
    int hidden = 896; // Qwen2.5-0.5B hidden size
    int n_layers = argc > 1 ? atoi(argv[1]) : 5;

    // Step 1: Create hipBLAS handle
    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Step 2: Create non-blocking stream (like ggml-hip)
    hipStream_t stream;
    hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    hipblasSetStream(handle, stream);

    // Step 3: Allocate "model weights" (large buffers in VRAM)
    int weight_size = hidden * hidden * sizeof(float);
    float *d_wq, *d_wk, *d_wv, *d_wo;
    hipMalloc(&d_wq, weight_size);
    hipMalloc(&d_wk, weight_size);
    hipMalloc(&d_wv, weight_size);
    hipMalloc(&d_wo, weight_size);

    // Step 4: H2D copy (model loading)
    float *h_weight = (float*)malloc(weight_size);
    memset(h_weight, 0, weight_size);
    printf("Loading weights (%d bytes x 4)...\n", weight_size);
    hipMemcpy(d_wq, h_weight, weight_size, hipMemcpyHostToDevice);
    hipMemcpy(d_wk, h_weight, weight_size, hipMemcpyHostToDevice);
    hipMemcpy(d_wv, h_weight, weight_size, hipMemcpyHostToDevice);
    hipMemcpy(d_wo, h_weight, weight_size, hipMemcpyHostToDevice);
    hipDeviceSynchronize();
    printf("  Weights loaded.\n");

    // Inference buffers
    float *d_x, *d_norm, *d_q, *d_out;
    hipMalloc(&d_x, hidden * sizeof(float));
    hipMalloc(&d_norm, hidden * sizeof(float));
    hipMalloc(&d_q, hidden * sizeof(float));
    hipMalloc(&d_out, hidden * sizeof(float));
    hipMemset(d_x, 0, hidden * sizeof(float));

    // Step 5: Simulate inference — layers of norm + matmul + add
    printf("Running %d inference layers on stream...\n", n_layers);
    float alpha = 1.0f, beta = 0.0f;

    for (int layer = 0; layer < n_layers; layer++) {
        // RMS norm
        rms_norm<<<1, 256, 0, stream>>>(d_norm, d_x, hidden);

        // Q = Wq * norm (GEMV via GEMM with N=1)
        hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                     hidden, 1, hidden, &alpha,
                     d_wq, hidden, d_norm, hidden,
                     &beta, d_q, hidden);

        // out = Wo * q
        hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                     hidden, 1, hidden, &alpha,
                     d_wo, hidden, d_q, hidden,
                     &beta, d_out, hidden);

        // residual add: x = x + out
        add_vectors<<<(hidden + 255) / 256, 256, 0, stream>>>(
            d_x, d_x, d_out, hidden);
    }

    // Step 6: Synchronize
    hipError_t err = hipStreamSynchronize(stream);
    if (err != hipSuccess) {
        printf("  FAILED: %s\n", hipGetErrorString(err));
        return 1;
    }
    printf("  All %d layers PASS\n", n_layers);

    // Cleanup
    hipFree(d_wq); hipFree(d_wk); hipFree(d_wv); hipFree(d_wo);
    hipFree(d_x); hipFree(d_norm); hipFree(d_q); hipFree(d_out);
    free(h_weight);
    hipblasDestroy(handle);
    hipStreamDestroy(stream);
    return 0;
}
