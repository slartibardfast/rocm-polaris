// Definitive alloc/free/realloc test — run on clean cold-booted GPU.
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
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(d, h, sz, hipMemcpyHostToDevice));
    CHECK(hipDeviceSynchronize());
    verify<<<1, 256>>>(d, 0xAB, 256, d_bad);
    hipError_t sync = hipDeviceSynchronize();
    int bad = -1;
    if (sync == hipSuccess)
        CHECK(hipMemcpy(&bad, d_bad, sizeof(int), hipMemcpyDeviceToHost));
    printf("%s: %zuMB ptr=%p %s (bad=%d)\n", label, mb, (void*)d,
           (sync==hipSuccess && bad==0)?"PASS":"FAIL", bad);
    fflush(stdout);
    CHECK(hipFree(d_bad)); CHECK(hipFree(d)); free(h);
    return (sync != hipSuccess || bad != 0);
}

int main() {
    printf("=== Cold boot definitive test ===\n\n");

    // 1. Standalone sizes (no prior alloc/free)
    for (size_t mb : {1, 4, 16, 32, 64}) {
        if (h2d_verify(mb, "standalone")) return 1;
    }

    // 2. Sequential increasing (each alloc/free before next)
    printf("\n--- Sequential increasing ---\n");
    for (size_t mb : {1, 4, 16, 32, 64}) {
        if (h2d_verify(mb, "sequential")) return 1;
    }

    // 3. Same-size realloc (the known trigger)
    printf("\n--- Same-size realloc ---\n");
    for (size_t mb : {16, 32, 64}) {
        char label[64];
        snprintf(label, sizeof(label), "realloc-%zu", mb);
        if (h2d_verify(mb, label)) return 1;
        if (h2d_verify(mb, label)) return 1;
    }

    printf("\nALL PASS\n");
    return 0;
}
