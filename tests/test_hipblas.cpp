// test_hipblas.cpp — Does hipblasSgemm work on gfx803?
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <stdio.h>

int main() {
    printf("Creating hipBLAS handle...\n");
    hipblasHandle_t handle;
    hipblasStatus_t stat = hipblasCreate(&handle);
    if (stat != HIPBLAS_STATUS_SUCCESS) {
        printf("hipblasCreate FAILED: %d\n", stat);
        return 1;
    }
    printf("  handle created OK\n");

    int M = 64, N = 64, K = 64;
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, M * K * sizeof(float));
    hipMalloc(&d_B, K * N * sizeof(float));
    hipMalloc(&d_C, M * N * sizeof(float));
    hipMemset(d_A, 0, M * K * sizeof(float));
    hipMemset(d_B, 0, K * N * sizeof(float));
    hipMemset(d_C, 0, M * N * sizeof(float));

    float alpha = 1.0f, beta = 0.0f;

    printf("Calling hipblasSgemm...\n");
    stat = hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                        M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
    if (stat != HIPBLAS_STATUS_SUCCESS) {
        printf("hipblasSgemm FAILED: %d\n", stat);
        return 1;
    }

    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("hipDeviceSynchronize FAILED: %s\n", hipGetErrorString(err));
        return 1;
    }
    printf("  SGEMM PASS\n");

    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    hipblasDestroy(handle);
    return 0;
}
