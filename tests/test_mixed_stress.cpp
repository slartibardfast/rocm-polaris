// test_mixed_stress.cpp — Diagnose H2D corruption under rapid alloc/free
//
// Isolated version of hip_barrier_test 3.3 with detailed diagnostics:
// - Logs GPU-read vs expected for first few bad bytes
// - Logs VA reuse (same pointer returned by hipMalloc)
// - Logs timing between free and next alloc
//
// Build: hipcc --offload-arch=gfx803 -o test_mixed_stress test_mixed_stress.cpp
// Run:   timeout 120 ./test_mixed_stress

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

#define HIP_CHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
                e, hipGetErrorString(e), __FILE__, __LINE__); \
        return 1; \
    } \
} while (0)

// GPU kernel: check pattern and report first 8 bad bytes
__global__ void verify_detailed(const unsigned char *d, unsigned char expected,
                                int n, int *bad_count,
                                int *bad_idx, unsigned char *bad_got, int max_report) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && d[i] != expected) {
        int slot = atomicAdd(bad_count, 1);
        if (slot < max_report) {
            bad_idx[slot] = i;
            bad_got[slot] = d[i];
        }
    }
}

int main() {
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("Device: %s (%s)\n\n", prop.name, prop.gcnArchName);

    const int MAX_REPORT = 16;
    unsigned int seed = 42;
    void *prev_ptr = nullptr;
    int reuse_count = 0;

    for (int op = 0; op < 500; op++) {
        seed = seed * 1103515245 + 12345;
        size_t sizes[] = {4096, 16384, 65536, 262144, 1<<20, 4<<20};
        size_t sz = sizes[seed % 6];
        seed = seed * 1103515245 + 12345;
        unsigned char val = (unsigned char)(seed & 0xFF);
        if (val == 0) val = 1;  // avoid ambiguity with zero-init

        unsigned char *h = (unsigned char *)malloc(sz);
        memset(h, val, sz);

        unsigned char *d;
        HIP_CHECK(hipMalloc(&d, sz));

        bool reused = ((void*)d == prev_ptr);
        if (reused) reuse_count++;

        // Alloc diagnostic buffers
        int *d_bad_count;
        int *d_bad_idx;
        unsigned char *d_bad_got;
        HIP_CHECK(hipMalloc(&d_bad_count, sizeof(int)));
        HIP_CHECK(hipMalloc(&d_bad_idx, MAX_REPORT * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_bad_got, MAX_REPORT));
        HIP_CHECK(hipMemset(d_bad_count, 0, sizeof(int)));

        // H2D
        HIP_CHECK(hipMemcpy(d, h, sz, hipMemcpyHostToDevice));

        // GPU verify
        int blocks = (sz + 255) / 256;
        verify_detailed<<<blocks, 256>>>(d, val, sz, d_bad_count,
                                         d_bad_idx, d_bad_got, MAX_REPORT);
        HIP_CHECK(hipDeviceSynchronize());

        int bad = 0;
        HIP_CHECK(hipMemcpy(&bad, d_bad_count, sizeof(int), hipMemcpyDeviceToHost));

        if (bad > 0) {
            // Read diagnostic data
            int h_idx[MAX_REPORT];
            unsigned char h_got[MAX_REPORT];
            int to_read = bad < MAX_REPORT ? bad : MAX_REPORT;
            HIP_CHECK(hipMemcpy(h_idx, d_bad_idx, to_read * sizeof(int), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(h_got, d_bad_got, to_read, hipMemcpyDeviceToHost));

            // Also D2H the actual buffer to see what CPU reads
            unsigned char *h2 = (unsigned char *)malloc(sz);
            HIP_CHECK(hipMemcpy(h2, d, sz, hipMemcpyDeviceToHost));

            printf("FAIL op %d: sz=%zu val=0x%02x bad=%d/%zu ptr=%p reused=%d\n",
                   op, sz, val, bad, sz, (void*)d, reused);
            printf("  GPU-read samples (first %d bad bytes):\n", to_read);
            for (int i = 0; i < to_read; i++) {
                printf("    [%d] expected=0x%02x gpu_got=0x%02x cpu_d2h=0x%02x\n",
                       h_idx[i], val, h_got[i], h2[h_idx[i]]);
            }

            // Check if it's all zeros (uninit VRAM) or a previous pattern
            int zero_count = 0, prev_pat = -1;
            for (int i = 0; i < to_read; i++) {
                if (h_got[i] == 0) zero_count++;
                if (prev_pat == -1 && h_got[i] != val && h_got[i] != 0)
                    prev_pat = h_got[i];
            }
            if (zero_count == to_read)
                printf("  Pattern: all zeros (VRAM not written / stale zero-init)\n");
            else if (prev_pat >= 0)
                printf("  Pattern: 0x%02x (possibly previous op's fill value)\n", prev_pat);

            free(h2);
            hipFree(d_bad_got); hipFree(d_bad_idx); hipFree(d_bad_count);
            hipFree(d); free(h);
            printf("  VA reuses so far: %d/%d\n", reuse_count, op + 1);
            return 1;
        }

        // Progress every 50 ops
        if ((op + 1) % 50 == 0)
            printf("  %d/500 OK (reuses=%d)\n", op + 1, reuse_count);

        prev_ptr = (void*)d;
        hipFree(d_bad_got); hipFree(d_bad_idx); hipFree(d_bad_count);
        HIP_CHECK(hipFree(d));
        free(h);
    }

    printf("\nALL 500 ops PASS (VA reuses: %d)\n", reuse_count);
    return 0;
}
