// test_multi_dispatch.cpp — Multiple kernel dispatches in sequence
// Mimics llama.cpp pattern: many kernels dispatched to same stream
// without explicit synchronization between them.
//
// Build: hipcc -o test_multi_dispatch test_multi_dispatch.cpp

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void add_one(int *buf, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) buf[i] += 1;
}

int main(int argc, char **argv) {
    int n_dispatches = argc > 1 ? atoi(argv[1]) : 10;
    int n = 256;

    int *d_buf;
    hipMalloc(&d_buf, n * sizeof(int));
    hipMemset(d_buf, 0, n * sizeof(int));

    printf("Dispatching %d kernels in sequence...\n", n_dispatches);

    for (int i = 0; i < n_dispatches; i++) {
        add_one<<<1, n>>>(d_buf, n);
    }

    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("hipDeviceSynchronize FAILED: %s (%d)\n",
               hipGetErrorString(err), err);
        return 1;
    }

    int *h_buf = (int*)malloc(n * sizeof(int));
    hipMemcpy(h_buf, d_buf, n * sizeof(int), hipMemcpyDeviceToHost);

    int fails = 0;
    for (int i = 0; i < n; i++) {
        if (h_buf[i] != n_dispatches) {
            if (fails < 5)
                printf("  buf[%d] = %d (expected %d)\n", i, h_buf[i], n_dispatches);
            fails++;
        }
    }

    printf("Result: %d / %d correct\n", n - fails, n);

    free(h_buf);
    hipFree(d_buf);
    return fails > 0 ? 1 : 0;
}
