// Phase 7g: Hunt the alloc/free/realloc PTE bug.
// Standalone 32MB works. 16MB warmup + free + 32MB faults.
// Narrow down exactly what triggers the stale PTEs.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(e) do { hipError_t _e = (e); if (_e != hipSuccess) { \
    fprintf(stderr, "HIP error %d (%s) at %s:%d\n", _e, hipGetErrorString(_e), __FILE__, __LINE__); return 1; } } while(0)

__global__ void verify(const unsigned char* d, unsigned char expected, int n, int* bad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && d[i] != expected) atomicAdd(bad, 1);
}

static int h2d_verify(size_t mb, const char* label) {
    size_t sz = mb << 20;
    unsigned char *h = (unsigned char*)malloc(sz);
    std::memset(h, 0xAB, sz);
    unsigned char *d; int *d_bad;
    CHECK(hipMalloc(&d, sz));
    CHECK(hipMalloc(&d_bad, sizeof(int)));
    CHECK(hipMemset(d_bad, 0, sizeof(int)));
    CHECK(hipMemcpy(d, h, sz, hipMemcpyHostToDevice));
    CHECK(hipDeviceSynchronize());
    verify<<<1, 256>>>(d, 0xAB, 256, d_bad);
    hipError_t sync = hipDeviceSynchronize();
    int bad = -1;
    if (sync == hipSuccess)
        CHECK(hipMemcpy(&bad, d_bad, sizeof(int), hipMemcpyDeviceToHost));
    printf("%s: %zuMB H2D+verify at %p: %s (bad=%d)\n", label, mb, (void*)d,
           (sync==hipSuccess && bad==0)?"PASS":"FAIL", bad);
    fflush(stdout);
    CHECK(hipFree(d_bad)); CHECK(hipFree(d)); free(h);
    return (sync != hipSuccess || bad != 0);
}

int main() {
    printf("=== Hunt alloc/free/realloc PTE bug ===\n\n");

    // Test A: standalone 32MB (baseline, should pass)
    printf("A: Standalone 32MB\n");
    if (h2d_verify(32, "A")) return 1;

    // Test B: alloc+free 16MB, then 32MB
    printf("\nB: alloc+free 16MB, then 32MB\n");
    { void *tmp; CHECK(hipMalloc(&tmp, 16<<20)); CHECK(hipFree(tmp)); }
    if (h2d_verify(32, "B")) return 1;

    // Test C: alloc+free 32MB, then 32MB (same size reuse)
    printf("\nC: alloc+free 32MB, then 32MB\n");
    { void *tmp; CHECK(hipMalloc(&tmp, 32<<20)); CHECK(hipFree(tmp)); }
    if (h2d_verify(32, "C")) return 1;

    // Test D: alloc+free 1MB, then 32MB (tiny warmup)
    printf("\nD: alloc+free 1MB, then 32MB\n");
    { void *tmp; CHECK(hipMalloc(&tmp, 1<<20)); CHECK(hipFree(tmp)); }
    if (h2d_verify(32, "D")) return 1;

    // Test E: alloc+H2D+free 16MB, then 32MB (with actual data transfer)
    printf("\nE: alloc+H2D+free 16MB, then 32MB\n");
    {
        void *tmp; unsigned char *h16 = (unsigned char*)malloc(16<<20);
        std::memset(h16, 0xCD, 16<<20);
        CHECK(hipMalloc(&tmp, 16<<20));
        CHECK(hipMemcpy(tmp, h16, 16<<20, hipMemcpyHostToDevice));
        CHECK(hipDeviceSynchronize());
        CHECK(hipFree(tmp));
        free(h16);
    }
    if (h2d_verify(32, "E")) return 1;

    // Test F: alloc+memset+free 16MB, then 32MB (GPU kernel warmup, no blit)
    printf("\nF: alloc+memset+free 16MB, then 32MB\n");
    {
        void *tmp;
        CHECK(hipMalloc(&tmp, 16<<20));
        CHECK(hipMemset(tmp, 0x42, 16<<20));
        CHECK(hipDeviceSynchronize());
        CHECK(hipFree(tmp));
    }
    if (h2d_verify(32, "F")) return 1;

    // Test G: alloc 16MB (keep), then alloc 32MB + H2D + verify (no free before)
    printf("\nG: alloc 16MB (keep), then 32MB\n");
    {
        void *keep;
        CHECK(hipMalloc(&keep, 16<<20));
        int result = h2d_verify(32, "G");
        CHECK(hipFree(keep));
        if (result) return 1;
    }

    printf("\nALL PASS\n");
    return 0;
}
