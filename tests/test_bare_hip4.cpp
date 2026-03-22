// test_bare_hip4.cpp — Isolate which HIP operation triggers corruption
//
// Build: hipcc --offload-arch=gfx803 -o test_bare_hip4 test_bare_hip4.cpp
// Run:   ./test_bare_hip4 <variant> [iters]
// Variants: none, memcpy_d2h, memcpy_h2d, memset, malloc_free

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

__global__ void record_val(int *out, int val) {
    if (threadIdx.x == 0) *out = val;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <variant> [iters]\n", argv[0]); return 1; }
    const char *v = argv[1];
    int iters = argc > 2 ? atoi(argv[2]) : 10000;

    int *d;
    hipHostMalloc(&d, sizeof(int), hipHostMallocDefault);

    int *d_tmp;
    hipMalloc(&d_tmp, 1024);

    int h_tmp[256];
    memset(h_tmp, 0xAB, sizeof(h_tmp));

    int fails = 0;
    for (int i = 0; i < iters; i++) {
        record_val<<<1, 1>>>(d, i * 7 + 13);
        hipDeviceSynchronize();

        // Intervention AFTER kernel, BEFORE check
        if (strcmp(v, "memcpy_d2h") == 0) {
            int x;
            hipMemcpy(&x, d_tmp, sizeof(int), hipMemcpyDeviceToHost);
        } else if (strcmp(v, "memcpy_h2d") == 0) {
            hipMemcpy(d_tmp, h_tmp, sizeof(h_tmp), hipMemcpyHostToDevice);
        } else if (strcmp(v, "memset") == 0) {
            hipMemset(d_tmp, 0, 1024);
        } else if (strcmp(v, "malloc_free") == 0) {
            int *tmp2;
            hipMalloc(&tmp2, 64);
            hipFree(tmp2);
        }
        // else: none

        if (*d != i * 7 + 13) {
            fails++;
            if (fails <= 5)
                printf("iter %d: exp %d got %d (0x%x)\n", i, i*7+13, *d, *d);
        }
    }
    hipHostFree(d);
    hipFree(d_tmp);
    printf("%-15s %d / %d (%.3f%%)\n", v, fails, iters, 100.0*fails/iters);
    return fails > 0 ? 1 : 0;
}
