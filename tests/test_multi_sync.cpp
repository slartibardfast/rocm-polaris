// test_multi_sync.cpp — Dispatch + sync pattern (mimics llama.cpp layers)
// Each "layer" = N kernel dispatches + hipDeviceSynchronize
//
// Build: hipcc -o test_multi_sync test_multi_sync.cpp

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void add_one(int *buf, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) buf[i] += 1;
}

int main(int argc, char **argv) {
    int n_layers = argc > 1 ? atoi(argv[1]) : 5;
    int n_ops_per_layer = argc > 2 ? atoi(argv[2]) : 10;
    int n = 256;

    int *d_buf;
    hipMalloc(&d_buf, n * sizeof(int));
    hipMemset(d_buf, 0, n * sizeof(int));

    printf("Layers: %d, ops/layer: %d, total dispatches: %d\n",
           n_layers, n_ops_per_layer, n_layers * n_ops_per_layer);

    for (int layer = 0; layer < n_layers; layer++) {
        for (int op = 0; op < n_ops_per_layer; op++) {
            add_one<<<1, n>>>(d_buf, n);
        }
        hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            printf("FAILED at layer %d: %s (%d)\n",
                   layer, hipGetErrorString(err), err);
            return 1;
        }
        printf("  layer %d OK\n", layer);
    }

    int *h_buf = (int*)malloc(n * sizeof(int));
    hipMemcpy(h_buf, d_buf, n * sizeof(int), hipMemcpyDeviceToHost);

    int total = n_layers * n_ops_per_layer;
    int fails = 0;
    for (int i = 0; i < n; i++) {
        if (h_buf[i] != total) {
            if (fails < 5)
                printf("  buf[%d] = %d (expected %d)\n", i, h_buf[i], total);
            fails++;
        }
    }

    printf("Result: %d / %d correct\n", n - fails, n);
    free(h_buf);
    hipFree(d_buf);
    return fails > 0 ? 1 : 0;
}
