#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(e) do { hipError_t _e = (e); if (_e != hipSuccess) { \
    fprintf(stderr, "HIP error %d (%s) at %s:%d\n", _e, hipGetErrorString(_e), __FILE__, __LINE__); return 1; } } while(0)

// Trivial kernel that does nothing
__global__ void nop_kernel() {}

// Verify kernel
__global__ void verify(const unsigned char* d, unsigned char expected, int n, int* bad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && d[i] != expected) atomicAdd(bad, 1);
}

int main() {
    size_t sz = 65UL << 20;

    unsigned char *h = (unsigned char*)malloc(sz);
    std::memset(h, 0xAB, sz);
    unsigned char *d;
    CHECK(hipMalloc(&d, sz));

    // Test A: H2D + D2H only (no kernel between)
    CHECK(hipMemcpy(d, h, sz, hipMemcpyHostToDevice));
    unsigned char *h2 = (unsigned char*)malloc(sz);
    CHECK(hipMemcpy(h2, d, sz, hipMemcpyDeviceToHost));
    int bad_a = 0;
    for (size_t i = 0; i < sz; i++) if (h2[i] != 0xAB) bad_a++;
    printf("Test A (H2D+D2H, no kernel):    cpu_bad=%d %s\n", bad_a, bad_a==0?"PASS":"FAIL");
    free(h2);

    // Test B: H2D + nop kernel + D2H
    CHECK(hipMemcpy(d, h, sz, hipMemcpyHostToDevice));
    nop_kernel<<<1,1>>>();
    CHECK(hipDeviceSynchronize());
    h2 = (unsigned char*)malloc(sz);
    CHECK(hipMemcpy(h2, d, sz, hipMemcpyDeviceToHost));
    int bad_b = 0;
    for (size_t i = 0; i < sz; i++) if (h2[i] != 0xAB) bad_b++;
    printf("Test B (H2D+nop+D2H):            cpu_bad=%d %s\n", bad_b, bad_b==0?"PASS":"FAIL");
    free(h2);

    // Test C: H2D + verify kernel (reads d) + D2H
    int *d_bad;
    CHECK(hipMalloc(&d_bad, sizeof(int)));
    CHECK(hipMemset(d_bad, 0, sizeof(int)));
    CHECK(hipMemcpy(d, h, sz, hipMemcpyHostToDevice));
    verify<<<(sz+255)/256, 256>>>(d, 0xAB, sz, d_bad);
    CHECK(hipDeviceSynchronize());
    int gpu_bad = 0;
    CHECK(hipMemcpy(&gpu_bad, d_bad, sizeof(int), hipMemcpyDeviceToHost));
    h2 = (unsigned char*)malloc(sz);
    CHECK(hipMemcpy(h2, d, sz, hipMemcpyDeviceToHost));
    int bad_c = 0;
    for (size_t i = 0; i < sz; i++) if (h2[i] != 0xAB) bad_c++;
    printf("Test C (H2D+verify+D2H):         cpu_bad=%d gpu_bad=%d %s\n",
           bad_c, gpu_bad, (bad_c==0&&gpu_bad==0)?"PASS":"FAIL");
    free(h2);

    // Test D: H2D + verify kernel (65MB grid = lots of threads) + D2H of DIFFERENT buffer
    // Does the verify kernel's massive dispatch corrupt the blit state?
    unsigned char *d2;
    CHECK(hipMalloc(&d2, 1<<20));  // 1MB second buffer
    CHECK(hipMemset(d2, 0x42, 1<<20));
    CHECK(hipMemcpy(d, h, sz, hipMemcpyHostToDevice));
    CHECK(hipMemset(d_bad, 0, sizeof(int)));
    verify<<<(sz+255)/256, 256>>>(d, 0xAB, sz, d_bad);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(&gpu_bad, d_bad, sizeof(int), hipMemcpyDeviceToHost));
    // D2H the 1MB buffer (not the 65MB one)
    unsigned char *h3 = (unsigned char*)malloc(1<<20);
    CHECK(hipMemcpy(h3, d2, 1<<20, hipMemcpyDeviceToHost));
    int bad_d = 0;
    for (int i = 0; i < (1<<20); i++) if (h3[i] != 0x42) bad_d++;
    printf("Test D (verify 65MB, D2H 1MB):   cpu_bad=%d gpu_bad=%d %s\n",
           bad_d, gpu_bad, (bad_d==0&&gpu_bad==0)?"PASS":"FAIL");
    free(h3);

    hipFree(d_bad); hipFree(d2); hipFree(d); free(h);
    return 0;
}
