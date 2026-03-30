// test_shared_mem.cpp — Test kernels with shared memory like ggml-hip uses
// ggml-hip mmq kernels use 8-32KB shared memory per workgroup
//
// Build: hipcc -o test_shared_mem test_shared_mem.cpp

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel with 8KB shared memory (typical ggml-hip)
__global__ void reduce_smem(const float *in, float *out, int n) {
    __shared__ float smem[2048]; // 8KB
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();

    // Simple reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = smem[0];
}

// Kernel with 16KB shared memory
__global__ void matmul_like(const float *A, const float *B, float *C,
                            int M, int N, int K) {
    __shared__ float As[32][32]; // 4KB
    __shared__ float Bs[32][32]; // 4KB (8KB total)

    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + 31) / 32; t++) {
        if (row < M && t * 32 + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * 32 + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * 32 + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * 32 + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();
        for (int k = 0; k < 32; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = sum;
}

int main(int argc, char **argv) {
    int n_iters = argc > 1 ? atoi(argv[1]) : 10;

    // Test 1: Reduction with 8KB shared memory
    {
        int n = 4096;
        float *d_in, *d_out;
        hipMalloc(&d_in, n * sizeof(float));
        hipMalloc(&d_out, (n / 256) * sizeof(float));

        // Initialize
        float *h_in = (float*)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) h_in[i] = 1.0f;
        hipMemcpy(d_in, h_in, n * sizeof(float), hipMemcpyHostToDevice);

        printf("Test 1: Reduction (8KB smem), %d iters...\n", n_iters);
        for (int i = 0; i < n_iters; i++) {
            reduce_smem<<<n / 256, 256>>>(d_in, d_out, n);
        }
        hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            printf("  FAILED: %s\n", hipGetErrorString(err));
            return 1;
        }
        printf("  PASS\n");

        free(h_in);
        hipFree(d_in);
        hipFree(d_out);
    }

    // Test 2: Matmul-like with 8KB shared memory, 32x32 blocks
    {
        int M = 128, N = 128, K = 128;
        float *d_A, *d_B, *d_C;
        hipMalloc(&d_A, M * K * sizeof(float));
        hipMalloc(&d_B, K * N * sizeof(float));
        hipMalloc(&d_C, M * N * sizeof(float));
        hipMemset(d_A, 0, M * K * sizeof(float));
        hipMemset(d_B, 0, K * N * sizeof(float));

        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (M + 31) / 32);

        printf("Test 2: Matmul-like (8KB smem, 32x32 blocks), %d iters...\n", n_iters);
        for (int i = 0; i < n_iters; i++) {
            matmul_like<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        }
        hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            printf("  FAILED: %s\n", hipGetErrorString(err));
            return 1;
        }
        printf("  PASS\n");

        hipFree(d_A);
        hipFree(d_B);
        hipFree(d_C);
    }

    printf("All tests PASSED\n");
    return 0;
}
