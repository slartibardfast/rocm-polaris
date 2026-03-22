// test_bare_hip2.cpp — Single dispatch per iteration, no hipMemcpy
// Uses hipHostMalloc for CPU-visible result buffer.
//
// Build: hipcc --offload-arch=gfx803 -o test_bare_hip2 test_bare_hip2.cpp
// Run:   ./test_bare_hip2 [iters]

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void record_val(int *out, int val) {
    if (threadIdx.x == 0) *out = val;
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 10000;

    // Host-visible memory — CPU can read directly, no hipMemcpy needed
    int *d;
    hipHostMalloc(&d, sizeof(int), hipHostMallocDefault);

    int fails = 0;
    for (int i = 0; i < iters; i++) {
        record_val<<<1, 1>>>(d, i * 7 + 13);
        hipDeviceSynchronize();

        // Direct CPU read — no hipMemcpy dispatch
        if (*d != i * 7 + 13) {
            fails++;
            if (fails <= 5)
                printf("iter %d: exp %d got %d (0x%x)\n", i, i*7+13, *d, *d);
        }
    }
    hipHostFree(d);
    printf("bare_hip2: %d / %d (%.3f%%)\n", fails, iters, 100.0*fails/iters);
    return fails > 0 ? 1 : 0;
}
