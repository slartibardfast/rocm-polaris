// test_alloc_free_race.cpp — Diagnose H2D/kernel corruption during rapid alloc/free
//
// Reproduces hip_barrier_test 3.2 failure with detailed diagnostics:
// - Verifies H2D immediately after copy (before kernel)
// - Verifies hipMemset(d_sum) is actually zero
// - Logs kernel args (pointers, n) visible to GPU
// - Checks d_out content after add_one (before sum_reduce)
// - Logs VA reuse and allocation addresses
//
// Build: hipcc --offload-arch=gfx803 -o test_alloc_free_race test_alloc_free_race.cpp
// Run:   timeout 120 ./test_alloc_free_race

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
                e, hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

__global__ void add_one(const int *in, int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] + 1;
}

__global__ void sum_reduce(const int *in, int *out, int n) {
    int s = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        s += in[i];
    atomicAdd(out, s);
}

// GPU kernel to snapshot first 8 elements for diagnostics
__global__ void snapshot(const int *src, int *dst, int n) {
    int i = threadIdx.x;
    if (i < n) dst[i] = src[i];
}

int main() {
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("Device: %s (%s)\n\n", prop.name, prop.gcnArchName);

    const int N = 256;
    const int SNAP = 8;  // snapshot first 8 elements

    // Diagnostic buffer (persistent across iterations)
    int *d_snap;
    HIP_CHECK(hipMalloc(&d_snap, SNAP * sizeof(int)));

    void *prev_in = nullptr, *prev_out = nullptr, *prev_sum = nullptr;

    for (int i = 0; i < 500; i++) {
        int *d_in, *d_out, *d_sum;
        int h_in[N];

        for (int j = 0; j < N; j++) h_in[j] = i + j;

        HIP_CHECK(hipMalloc(&d_in, N * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_out, N * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_sum, sizeof(int)));
        HIP_CHECK(hipMemset(d_sum, 0, sizeof(int)));
        HIP_CHECK(hipMemcpy(d_in, h_in, N * sizeof(int), hipMemcpyHostToDevice));

        // Step 1: Verify H2D worked (read back d_in)
        int h_check_in[SNAP];
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_check_in, d_in, SNAP * sizeof(int), hipMemcpyDeviceToHost));

        bool h2d_ok = true;
        for (int j = 0; j < SNAP; j++) {
            if (h_check_in[j] != i + j) { h2d_ok = false; break; }
        }

        // Step 2: Verify d_sum is zero
        int h_sum_before = -1;
        HIP_CHECK(hipMemcpy(&h_sum_before, d_sum, sizeof(int), hipMemcpyDeviceToHost));

        // Step 3: Run add_one
        add_one<<<1, N>>>(d_in, d_out, N);
        HIP_CHECK(hipDeviceSynchronize());

        // Step 4: Snapshot d_out (GPU-side read, avoids D2H coherency issues)
        HIP_CHECK(hipMemset(d_snap, 0xFF, SNAP * sizeof(int)));
        snapshot<<<1, SNAP>>>(d_out, d_snap, SNAP);
        HIP_CHECK(hipDeviceSynchronize());
        int h_snap_out[SNAP];
        HIP_CHECK(hipMemcpy(h_snap_out, d_snap, SNAP * sizeof(int), hipMemcpyDeviceToHost));

        // Step 5: Run sum_reduce
        sum_reduce<<<1, N>>>(d_out, d_sum, N);
        HIP_CHECK(hipDeviceSynchronize());

        // Step 6: Read result
        int h_sum = 0;
        HIP_CHECK(hipMemcpy(&h_sum, d_sum, sizeof(int), hipMemcpyDeviceToHost));

        int expected = 0;
        for (int j = 0; j < N; j++) expected += i + j + 1;

        if (h_sum != expected) {
            printf("FAIL iter %d: expected %d, got %d\n", i, expected, h_sum);
            printf("  Pointers: d_in=%p d_out=%p d_sum=%p\n",
                   (void*)d_in, (void*)d_out, (void*)d_sum);
            printf("  Reuse: in=%d out=%d sum=%d\n",
                   (void*)d_in == prev_in, (void*)d_out == prev_out,
                   (void*)d_sum == prev_sum);
            printf("  H2D check: %s  d_in[0..3]=%d,%d,%d,%d (expect %d,%d,%d,%d)\n",
                   h2d_ok ? "OK" : "FAIL",
                   h_check_in[0], h_check_in[1], h_check_in[2], h_check_in[3],
                   i, i+1, i+2, i+3);
            printf("  d_sum before kernels: %d (expect 0)\n", h_sum_before);
            printf("  d_out after add_one (GPU snapshot): %d,%d,%d,%d,%d,%d,%d,%d\n",
                   h_snap_out[0], h_snap_out[1], h_snap_out[2], h_snap_out[3],
                   h_snap_out[4], h_snap_out[5], h_snap_out[6], h_snap_out[7]);
            printf("  Expected d_out[0..7]: %d,%d,%d,%d,%d,%d,%d,%d\n",
                   i+1, i+2, i+3, i+4, i+5, i+6, i+7, i+8);

            // Also read d_out via D2H for comparison
            int h_d2h_out[SNAP];
            HIP_CHECK(hipMemcpy(h_d2h_out, d_out, SNAP * sizeof(int), hipMemcpyDeviceToHost));
            printf("  d_out D2H readback:   %d,%d,%d,%d,%d,%d,%d,%d\n",
                   h_d2h_out[0], h_d2h_out[1], h_d2h_out[2], h_d2h_out[3],
                   h_d2h_out[4], h_d2h_out[5], h_d2h_out[6], h_d2h_out[7]);

            // Check how many d_out elements are correct
            int correct = 0, zero = 0, other = 0;
            int h_full_out[N];
            HIP_CHECK(hipMemcpy(h_full_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost));
            for (int j = 0; j < N; j++) {
                if (h_full_out[j] == i + j + 1) correct++;
                else if (h_full_out[j] == 0) zero++;
                else other++;
            }
            printf("  d_out element breakdown: %d correct, %d zero, %d other (of %d)\n",
                   correct, zero, other, N);
            if (other > 0) {
                // Find first "other" value
                for (int j = 0; j < N; j++) {
                    if (h_full_out[j] != i + j + 1 && h_full_out[j] != 0) {
                        printf("  First 'other' at [%d]: got %d (0x%08x), expect %d\n",
                               j, h_full_out[j], h_full_out[j], i + j + 1);
                        break;
                    }
                }
            }

            hipFree(d_sum); hipFree(d_out); hipFree(d_in); hipFree(d_snap);
            return 1;
        }

        if ((i + 1) % 100 == 0)
            printf("  %d/500 OK\n", i + 1);

        prev_in = (void*)d_in;
        prev_out = (void*)d_out;
        prev_sum = (void*)d_sum;

        HIP_CHECK(hipFree(d_sum));
        HIP_CHECK(hipFree(d_out));
        HIP_CHECK(hipFree(d_in));
    }

    hipFree(d_snap);
    printf("\nALL 500 iters PASS\n");
    return 0;
}
