// test_800_dispatch.cpp — Reproduce CP stall after 800+ dispatches
// llama.cpp hangs after pass 1 (797 nodes) + sync + pass 2 start.
// Test: 800 dispatches, sync, then 10 more dispatches.
//
// Build: hipcc -o test_800_dispatch test_800_dispatch.cpp

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void add_one(float *buf, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) buf[i] += 1.0f;
}

int main(int argc, char **argv) {
    int pass1 = argc > 1 ? atoi(argv[1]) : 800;
    int pass2 = argc > 2 ? atoi(argv[2]) : 10;
    int n = 4096;

    float *d_buf;
    hipMalloc(&d_buf, n * sizeof(float));
    hipMemset(d_buf, 0, n * sizeof(float));

    hipStream_t stream;
    hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);

    printf("Pass 1: %d dispatches on non-blocking stream...\n", pass1);
    for (int i = 0; i < pass1; i++) {
        add_one<<<n / 256, 256, 0, stream>>>(d_buf, n);
    }

    hipError_t err = hipStreamSynchronize(stream);
    printf("  Sync: %s\n", err == hipSuccess ? "OK" : hipGetErrorString(err));

    printf("Pass 2: %d dispatches...\n", pass2);
    for (int i = 0; i < pass2; i++) {
        add_one<<<n / 256, 256, 0, stream>>>(d_buf, n);
        if (i < 3 || i == pass2 - 1)
            printf("  dispatch %d submitted\n", i);
    }

    err = hipStreamSynchronize(stream);
    printf("  Sync: %s\n", err == hipSuccess ? "OK" : hipGetErrorString(err));

    float *h_buf = (float*)malloc(n * sizeof(float));
    hipMemcpy(h_buf, d_buf, n * sizeof(float), hipMemcpyDeviceToHost);

    float expected = (float)(pass1 + pass2);
    int fails = 0;
    for (int i = 0; i < n; i++) {
        if (h_buf[i] != expected) { fails++; }
    }
    printf("Result: %d / %d correct (expected %.0f)\n", n - fails, n, expected);

    hipStreamDestroy(stream);
    free(h_buf);
    hipFree(d_buf);
    return fails > 0 ? 1 : 0;
}
