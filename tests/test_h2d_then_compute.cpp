// test_h2d_then_compute.cpp — Does H2D memcpy followed by compute hang?
// Tests the exact pattern llama.cpp uses: large H2D transfers, then compute.
//
// Build: hipcc -o test_h2d_then_compute test_h2d_then_compute.cpp

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__global__ void add_one(float *buf, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) buf[i] += 1.0f;
}

int main(int argc, char **argv) {
    int n_buffers = argc > 1 ? atoi(argv[1]) : 10;
    int buf_size = argc > 2 ? atoi(argv[2]) : 1024 * 1024; // 1MB default
    int n_compute = argc > 3 ? atoi(argv[3]) : 100;

    printf("Phase 1: %d H2D copies of %d bytes each...\n", n_buffers, buf_size);

    float **d_bufs = (float**)malloc(n_buffers * sizeof(float*));
    float *h_buf = (float*)malloc(buf_size);
    memset(h_buf, 0x42, buf_size);

    for (int i = 0; i < n_buffers; i++) {
        hipMalloc(&d_bufs[i], buf_size);
        hipError_t err = hipMemcpy(d_bufs[i], h_buf, buf_size, hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            printf("  H2D copy %d FAILED: %s\n", i, hipGetErrorString(err));
            return 1;
        }
    }
    printf("  H2D copies done.\n");

    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("  Sync after H2D FAILED: %s\n", hipGetErrorString(err));
        return 1;
    }
    printf("  Sync OK.\n");

    printf("Phase 2: %d compute dispatches on buffer[0]...\n", n_compute);
    int n_elems = buf_size / sizeof(float);
    int blocks = (n_elems + 255) / 256;

    for (int i = 0; i < n_compute; i++) {
        add_one<<<blocks, 256>>>(d_bufs[0], n_elems);
    }

    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("  Compute sync FAILED: %s\n", hipGetErrorString(err));
        return 1;
    }
    printf("  Compute PASS\n");

    // Cleanup
    for (int i = 0; i < n_buffers; i++) hipFree(d_bufs[i]);
    free(d_bufs);
    free(h_buf);
    return 0;
}
