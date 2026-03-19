// Phase 7g: Minimal reproducer + page table dump request.
// Does: alloc 32MB, free, alloc 32MB, H2D, verify.
// On fault, prints the VA and asks for /sys/kernel/debug analysis.
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

int main() {
    printf("=== Minimal alloc/free/realloc reproducer ===\n\n");

    // Step 1: alloc 32MB, touch it, free it
    void *first;
    CHECK(hipMalloc(&first, 32<<20));
    CHECK(hipMemset(first, 0x42, 32<<20));
    CHECK(hipDeviceSynchronize());
    printf("First alloc: %p\n", first);
    CHECK(hipFree(first));
    printf("First freed.\n");

    // Step 2: alloc 32MB again (same VA expected)
    unsigned char *d; int *d_bad;
    CHECK(hipMalloc(&d, 32<<20));
    CHECK(hipMalloc(&d_bad, sizeof(int)));
    printf("Second alloc: %p (d_bad: %p)\n", (void*)d, (void*)d_bad);

    // Step 3: H2D
    unsigned char *h = (unsigned char*)malloc(32<<20);
    std::memset(h, 0xAB, 32<<20);
    CHECK(hipMemset(d_bad, 0, sizeof(int)));
    CHECK(hipMemcpy(d, h, 32<<20, hipMemcpyHostToDevice));
    CHECK(hipDeviceSynchronize());
    printf("H2D done.\n");

    // Step 4: verify
    printf("Launching verify<<<1,256>>>...\n");
    fflush(stdout);
    verify<<<1, 256>>>(d, 0xAB, 256, d_bad);
    hipError_t sync = hipDeviceSynchronize();
    if (sync != hipSuccess) {
        printf("SYNC FAIL: %s\n", hipGetErrorString(sync));
        return 1;
    }
    int bad = 0;
    CHECK(hipMemcpy(&bad, d_bad, sizeof(int), hipMemcpyDeviceToHost));
    printf("verify: bad=%d %s\n", bad, bad==0?"PASS":"FAIL");

    CHECK(hipFree(d_bad)); CHECK(hipFree(d)); free(h);
    printf("ALL PASS\n");
    return 0;
}
