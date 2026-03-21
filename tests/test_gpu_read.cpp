#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(e) do { hipError_t _e = (e); if (_e != hipSuccess) { \
    fprintf(stderr, "HIP error %d (%s) at %s:%d\n", _e, hipGetErrorString(_e), __FILE__, __LINE__); return 1; } } while(0)

// GPU kernel: copy first 16 ints from src to dst
__global__ void read_sample(const int* src, int* dst, int n) {
    int i = threadIdx.x;
    if (i < n) dst[i] = src[i];
}

int main() {
    size_t sz = 16UL << 20;
    int n = sz / sizeof(int);
    int *h = (int*)malloc(sz);
    int *d;
    for (int i = 0; i < n; i++) h[i] = 0xDEAD0000 | (i & 0xFFFF);

    CHECK(hipMalloc(&d, sz));

    // H2D
    CHECK(hipMemcpy(d, h, sz, hipMemcpyHostToDevice));

    // D2H readback (CPU verify)
    int *h2 = (int*)malloc(sz);
    CHECK(hipMemcpy(h2, d, sz, hipMemcpyDeviceToHost));
    int cpu_bad = 0;
    for (int i = 0; i < n; i++) if (h2[i] != h[i]) cpu_bad++;
    printf("CPU D2H verify: %d/%d bad\n", cpu_bad, n);

    // GPU kernel read: sample 16 ints from beginning, middle, end
    int *d_sample, h_sample[48];
    CHECK(hipMalloc(&d_sample, 48*sizeof(int)));
    read_sample<<<1, 16>>>(d, d_sample, 16);
    read_sample<<<1, 16>>>(d + n/2, d_sample + 16, 16);
    read_sample<<<1, 16>>>(d + n - 16, d_sample + 32, 16);
    CHECK(hipMemcpy(h_sample, d_sample, 48*sizeof(int), hipMemcpyDeviceToHost));

    printf("GPU reads from VRAM (expect 0xDEADxxxx):\n");
    printf("  Start: ");
    for (int i = 0; i < 4; i++) printf("0x%08X ", h_sample[i]);
    printf("\n  Mid:   ");
    for (int i = 16; i < 20; i++) printf("0x%08X ", h_sample[i]);
    printf("\n  End:   ");
    for (int i = 32; i < 36; i++) printf("0x%08X ", h_sample[i]);
    printf("\n");

    hipFree(d); hipFree(d_sample); free(h); free(h2);
    return 0;
}
