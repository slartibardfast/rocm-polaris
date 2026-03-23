// test_d2h_sizes.cpp — Does D2H corruption depend on copy size?
// Build: hipcc --offload-arch=gfx803 -o test_d2h_sizes test_d2h_sizes.cpp
// Run:   ./test_d2h_sizes

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>

__global__ void fill_pattern(int *out, int n, int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = seed + i;
}

int main() {
    int sizes[] = {4, 16, 64, 256, 1024, 4096, 16384, 65536};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int si = 0; si < nsizes; si++) {
        int nbytes = sizes[si];
        int nints = nbytes / 4;
        int *d, *h;
        hipMalloc(&d, nbytes);
        h = (int*)malloc(nbytes);

        int fails = 0;
        for (int iter = 0; iter < 2000; iter++) {
            int seed = iter * 1000 + si;
            fill_pattern<<<(nints+63)/64, 64>>>(d, nints, seed);
            hipDeviceSynchronize();
            hipMemcpy(h, d, nbytes, hipMemcpyDeviceToHost);

            for (int j = 0; j < nints; j++) {
                if (h[j] != seed + j) {
                    fails++;
                    break; // count per-iteration, not per-element
                }
            }
        }
        printf("D2H %6d bytes: %3d / 2000 fails (%.1f%%)\n",
               nbytes, fails, 100.0*fails/2000);
        hipFree(d);
        free(h);
    }
}
