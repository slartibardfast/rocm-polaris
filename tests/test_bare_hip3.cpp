// test_bare_hip3.cpp — Two user kernel dispatches per iteration
// No hipMemcpy, no hipMemset. Tests if any double-dispatch corrupts.
//
// Build: hipcc --offload-arch=gfx803 -o test_bare_hip3 test_bare_hip3.cpp

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void record_val(int *out, int val) {
    if (threadIdx.x == 0) *out = val;
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 10000;

    int *d;
    hipHostMalloc(&d, sizeof(int), hipHostMallocDefault);

    int *d2;
    hipHostMalloc(&d2, sizeof(int), hipHostMallocDefault);

    int fails = 0;
    for (int i = 0; i < iters; i++) {
        record_val<<<1, 1>>>(d, i * 7 + 13);
        record_val<<<1, 1>>>(d2, i * 3 + 5);
        hipDeviceSynchronize();

        if (*d != i * 7 + 13) {
            fails++;
            if (fails <= 5)
                printf("d1 iter %d: exp %d got %d (0x%x)\n", i, i*7+13, *d, *d);
        }
        if (*d2 != i * 3 + 5) {
            fails++;
            if (fails <= 5)
                printf("d2 iter %d: exp %d got %d (0x%x)\n", i, i*3+5, *d2, *d2);
        }
    }
    hipHostFree(d);
    hipHostFree(d2);
    printf("bare_hip3: %d / %d (%.3f%%)\n", fails, iters, 100.0*fails/iters);
    return fails > 0 ? 1 : 0;
}
