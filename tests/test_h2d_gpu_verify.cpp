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

int main(int argc, char** argv) {
    // Usage: test_h2d_gpu_verify [size_mb]
    // Default: run a sweep. If size given, test only that size.
    int sizes[] = {1, 4, 16, 32, 48, 56, 60, 62, 63, 64, 65, 66, 72, 80, 96, 128};
    int nsizes = sizeof(sizes)/sizeof(sizes[0]);
    int single = 0;
    if (argc > 1) { single = atoi(argv[1]); nsizes = 1; sizes[0] = single; }

    for (int si = 0; si < nsizes; si++) {
        size_t mb = sizes[si];
        size_t sz = mb << 20;
        unsigned char *h = (unsigned char*)malloc(sz);
        if (!h) { printf("%3zuMB: SKIP (host OOM)\n", mb); continue; }
        std::memset(h, 0xAB, sz);

        unsigned char *d = nullptr;
        hipError_t me = hipMalloc(&d, sz);
        if (me != hipSuccess) { printf("%3zuMB: SKIP (hipMalloc fail)\n", mb); free(h); continue; }

        int *d_bad;
        CHECK(hipMalloc(&d_bad, sizeof(int)));
        CHECK(hipMemset(d_bad, 0, sizeof(int)));
        CHECK(hipDeviceSynchronize());

        // H2D
        CHECK(hipMemcpy(d, h, sz, hipMemcpyHostToDevice));

        // CPU round-trip verify
        unsigned char *h2 = (unsigned char*)malloc(sz);
        CHECK(hipMemcpy(h2, d, sz, hipMemcpyDeviceToHost));
        int cpu_bad = 0;
        for (size_t i = 0; i < sz; i++) if (h2[i] != 0xAB) cpu_bad++;
        free(h2);

        // GPU verify
        CHECK(hipDeviceSynchronize());
        verify<<<(sz+255)/256, 256>>>(d, 0xAB, sz, d_bad);
        CHECK(hipDeviceSynchronize());
        int gpu_bad = 0;
        CHECK(hipMemcpy(&gpu_bad, d_bad, sizeof(int), hipMemcpyDeviceToHost));

        printf("%3zuMB: cpu_bad=%d gpu_bad=%d/%zu ptr=%p %s\n",
               mb, cpu_bad, gpu_bad, sz,
               (void*)d, (cpu_bad==0 && gpu_bad==0)?"PASS":"FAIL");

        hipFree(d_bad); hipFree(d); free(h);
        if (gpu_bad > 0 || cpu_bad > 0) return 1;
    }
    printf("ALL PASS\n");
    return 0;
}
