// test_bare_hip5.cpp — Is the corruption in hipMemcpy D2H reading stale data?
//
// Build: hipcc --offload-arch=gfx803 -o test_bare_hip5 test_bare_hip5.cpp

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void record_val(int *out, int val) {
    if (threadIdx.x == 0) *out = val;
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 10000;

    // VRAM buffer — can only read via hipMemcpy
    int *d_vram;
    hipMalloc(&d_vram, sizeof(int));

    // Host-visible buffer — can read directly
    int *d_host;
    hipHostMalloc(&d_host, sizeof(int), hipHostMallocDefault);

    int dispatch_fails = 0, memcpy_fails = 0;
    for (int i = 0; i < iters; i++) {
        // Write to BOTH buffers with the same kernel
        record_val<<<1, 1>>>(d_host, i * 7 + 13);
        record_val<<<1, 1>>>(d_vram, i * 7 + 13);
        hipDeviceSynchronize();

        // Check host buffer directly (no hipMemcpy)
        int host_val = *d_host;

        // Check VRAM buffer via hipMemcpy
        int vram_val;
        hipMemcpy(&vram_val, d_vram, sizeof(int), hipMemcpyDeviceToHost);

        if (host_val != i * 7 + 13) {
            dispatch_fails++;
            if (dispatch_fails <= 3)
                printf("DISPATCH iter %d: exp %d got %d\n", i, i*7+13, host_val);
        }
        if (vram_val != i * 7 + 13) {
            memcpy_fails++;
            if (memcpy_fails <= 3)
                printf("MEMCPY  iter %d: exp %d got %d (0x%x)\n", i, i*7+13, vram_val, vram_val);
        }
    }
    hipHostFree(d_host);
    hipFree(d_vram);
    printf("dispatch: %d / %d, memcpy: %d / %d\n", dispatch_fails, iters, memcpy_fails, iters);
    return (dispatch_fails + memcpy_fails) > 0 ? 1 : 0;
}
