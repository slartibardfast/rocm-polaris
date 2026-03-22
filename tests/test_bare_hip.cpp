// test_bare_hip.cpp — Absolute minimum HIP dispatch to reproduce corruption
// No hipMemset, no alloc/free, no interleaving. Just launch+sync+check.
//
// Build: hipcc --offload-arch=gfx803 -o test_bare_hip test_bare_hip.cpp
// Run:   ./test_bare_hip [iters]

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void record_val(int *out, int val) {
    if (threadIdx.x == 0) *out = val;
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 10000;

    int *d;
    hipMalloc(&d, sizeof(int));

    int fails = 0;
    for (int i = 0; i < iters; i++) {
        record_val<<<1, 1>>>(d, i * 7 + 13);
        hipDeviceSynchronize();

        int h;
        hipMemcpy(&h, d, sizeof(int), hipMemcpyDeviceToHost);
        if (h != i * 7 + 13) {
            fails++;
            if (fails <= 5)
                printf("iter %d: exp %d got %d (0x%x)\n", i, i*7+13, h, h);
        }
    }
    hipFree(d);
    printf("bare_hip: %d / %d (%.3f%%)\n", fails, iters, 100.0*fails/iters);
    return fails > 0 ? 1 : 0;
}
