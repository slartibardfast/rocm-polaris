#include <cstring>
#include <hip/hip_runtime.h>
#include <cstdio>

#define CHECK(e) do { hipError_t _e = (e); if (_e != hipSuccess) { \
    fprintf(stderr, "HIP error %d (%s) at %s:%d\n", _e, hipGetErrorString(_e), __FILE__, __LINE__); return 1; } } while(0)

__global__ void verify(const unsigned char* d, unsigned char expected, int n, int* bad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && d[i] != expected) atomicAdd(bad, 1);
}

int main() {
    // Test 1: GPU memset + GPU kernel verify (no blit involved for H2D)
    printf("=== Test 1: hipMemset + GPU verify (no H2D blit) ===\n");
    for (size_t mb : {1, 4, 16, 64, 256}) {
        size_t sz = mb << 20;
        unsigned char *d; int *d_bad;
        CHECK(hipMalloc(&d, sz));
        CHECK(hipMalloc(&d_bad, sizeof(int)));
        CHECK(hipMemset(d_bad, 0, sizeof(int)));
        CHECK(hipMemset(d, 0x42, sz));
        CHECK(hipDeviceSynchronize());
        verify<<<(sz+255)/256, 256>>>(d, 0x42, sz, d_bad);
        int bad = 0;
        CHECK(hipMemcpy(&bad, d_bad, sizeof(int), hipMemcpyDeviceToHost));
        printf("  %3zuMB: bad=%d %s\n", mb, bad, bad==0?"PASS":"FAIL");
        hipFree(d); hipFree(d_bad);
        if (bad > 0) return 1;
    }

    // Test 2: H2D blit + D2H blit (CPU round-trip, no GPU kernel)
    printf("=== Test 2: H2D + D2H round-trip (CPU verify) ===\n");
    for (size_t mb : {1, 4, 16, 64, 256}) {
        size_t sz = mb << 20;
        int n = sz / sizeof(int);
        int *h = (int*)malloc(sz), *d, *h2 = (int*)malloc(sz);
        for (int i = 0; i < n; i++) h[i] = 0xDEAD0000 | (i & 0xFFFF);
        CHECK(hipMalloc(&d, sz));
        CHECK(hipMemcpy(d, h, sz, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(h2, d, sz, hipMemcpyDeviceToHost));
        int bad = 0;
        for (int i = 0; i < n; i++) if (h2[i] != h[i]) bad++;
        printf("  %3zuMB: bad=%d/%d %s\n", mb, bad, n, bad==0?"PASS":"FAIL");
        hipFree(d); free(h); free(h2);
        if (bad > 0) return 1;
    }

    // Test 3: H2D blit + GPU kernel verify (the failing case)
    printf("=== Test 3: H2D + GPU kernel verify ===\n");
    for (size_t mb : {1, 4, 16, 64}) {
        size_t sz = mb << 20;
        unsigned char *h = (unsigned char*)malloc(sz), *d;
        int *d_bad;
        std::memset(h, 0xAB, sz);
        CHECK(hipMalloc(&d, sz));
        CHECK(hipMalloc(&d_bad, sizeof(int)));
        CHECK(hipMemset(d_bad, 0, sizeof(int)));
        CHECK(hipMemcpy(d, h, sz, hipMemcpyHostToDevice));
        verify<<<(sz+255)/256, 256>>>(d, 0xAB, sz, d_bad);
        int bad = 0;
        CHECK(hipMemcpy(&bad, d_bad, sizeof(int), hipMemcpyDeviceToHost));
        printf("  %3zuMB: bad=%d/%zu %s\n", mb, bad, sz, bad==0?"PASS":"FAIL");
        hipFree(d); hipFree(d_bad); free(h);
        if (bad > 0) return 1;
    }

    printf("\nALL PASS\n");
    return 0;
}
