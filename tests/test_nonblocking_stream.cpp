// test_nonblocking_stream.cpp — Test hipStreamNonBlocking + matmul-like kernel
// ggml-hip creates streams with hipStreamNonBlocking flag.
//
// Build: hipcc -o test_nonblocking_stream test_nonblocking_stream.cpp

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void add_one(float *buf, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) buf[i] += 1.0f;
}

int main(int argc, char **argv) {
    int n_iters = argc > 1 ? atoi(argv[1]) : 100;
    int n = 4096;

    float *d_buf;
    hipMalloc(&d_buf, n * sizeof(float));
    hipMemset(d_buf, 0, n * sizeof(float));

    // Create non-blocking stream like ggml-hip does
    hipStream_t stream;
    hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);

    printf("Non-blocking stream, %d dispatches...\n", n_iters);

    for (int i = 0; i < n_iters; i++) {
        add_one<<<n / 256, 256, 0, stream>>>(d_buf, n);
    }

    hipError_t err = hipStreamSynchronize(stream);
    if (err != hipSuccess) {
        printf("FAILED: %s\n", hipGetErrorString(err));
        return 1;
    }

    float *h_buf = (float*)malloc(n * sizeof(float));
    hipMemcpy(h_buf, d_buf, n * sizeof(float), hipMemcpyDeviceToHost);

    int fails = 0;
    for (int i = 0; i < n; i++) {
        float expected = (float)n_iters;
        if (h_buf[i] != expected) {
            if (fails < 5)
                printf("  buf[%d] = %.0f (expected %.0f)\n", i, h_buf[i], expected);
            fails++;
        }
    }

    printf("Result: %d / %d correct\n", n - fails, n);
    hipStreamDestroy(stream);
    free(h_buf);
    hipFree(d_buf);
    return fails > 0 ? 1 : 0;
}
