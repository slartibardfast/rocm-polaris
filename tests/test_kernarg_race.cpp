// test_kernarg_race.cpp — Prove kernarg corruption in sum_reduce
//
// The previous diagnostic showed: d_out has 256 correct elements,
// d_sum starts at 0, but sum_reduce returns 128 instead of the correct sum.
// Theory: sum_reduce's kernel argument `in` pointer is stale/wrong.
//
// This test: have sum_reduce write what it ACTUALLY sees as input to a
// diagnostic buffer, so we can compare against what d_out actually contains.
//
// Build: hipcc --offload-arch=gfx803 -o test_kernarg_race test_kernarg_race.cpp
// Run:   timeout 120 ./test_kernarg_race

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

// Instrumented sum_reduce: writes its actual `in` pointer and first elements
// to a diagnostic buffer before summing
__global__ void sum_reduce_diag(const int *in, int *out, int n,
                                 unsigned long long *diag_ptr,
                                 int *diag_data, int diag_max) {
    // Thread 0 records what pointer we actually received
    if (threadIdx.x == 0) {
        diag_ptr[0] = (unsigned long long)in;
        diag_ptr[1] = (unsigned long long)out;
        diag_ptr[2] = (unsigned long long)n;
        for (int j = 0; j < diag_max && j < n; j++)
            diag_data[j] = in[j];
    }
    __syncthreads();

    int s = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        s += in[i];
    atomicAdd(out, s);
}

int main() {
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("Device: %s (%s)\n\n", prop.name, prop.gcnArchName);

    const int N = 256;
    const int DIAG = 16;

    // Persistent diagnostic buffers
    unsigned long long *d_diag_ptr;
    int *d_diag_data;
    HIP_CHECK(hipMalloc(&d_diag_ptr, 3 * sizeof(unsigned long long)));
    HIP_CHECK(hipMalloc(&d_diag_data, DIAG * sizeof(int)));

    void *prev_in = nullptr, *prev_out = nullptr;

    for (int i = 0; i < 500; i++) {
        int *d_in, *d_out, *d_sum;
        int h_in[N];

        for (int j = 0; j < N; j++) h_in[j] = i + j;

        HIP_CHECK(hipMalloc(&d_in, N * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_out, N * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_sum, sizeof(int)));
        HIP_CHECK(hipMemset(d_sum, 0, sizeof(int)));
        HIP_CHECK(hipMemcpy(d_in, h_in, N * sizeof(int), hipMemcpyHostToDevice));

        add_one<<<1, N>>>(d_in, d_out, N);
        HIP_CHECK(hipDeviceSynchronize());

        // Clear diag buffers
        HIP_CHECK(hipMemset(d_diag_ptr, 0, 3 * sizeof(unsigned long long)));
        HIP_CHECK(hipMemset(d_diag_data, 0xFF, DIAG * sizeof(int)));

        // Instrumented sum_reduce
        sum_reduce_diag<<<1, N>>>(d_out, d_sum, N,
                                   d_diag_ptr, d_diag_data, DIAG);
        HIP_CHECK(hipDeviceSynchronize());

        int h_sum = 0;
        HIP_CHECK(hipMemcpy(&h_sum, d_sum, sizeof(int), hipMemcpyDeviceToHost));

        int expected = 0;
        for (int j = 0; j < N; j++) expected += i + j + 1;

        if (h_sum != expected) {
            // Read diagnostics
            unsigned long long h_ptrs[3];
            int h_diag[DIAG];
            HIP_CHECK(hipMemcpy(h_ptrs, d_diag_ptr, 3 * sizeof(unsigned long long),
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(h_diag, d_diag_data, DIAG * sizeof(int),
                                hipMemcpyDeviceToHost));

            printf("FAIL iter %d: expected %d, got %d\n", i, expected, h_sum);
            printf("\n  Allocated pointers:\n");
            printf("    d_in  = %p\n", (void*)d_in);
            printf("    d_out = %p  (this should be sum_reduce's 'in')\n", (void*)d_out);
            printf("    d_sum = %p  (this should be sum_reduce's 'out')\n", (void*)d_sum);
            printf("    prev_in=%p prev_out=%p\n", prev_in, prev_out);

            printf("\n  sum_reduce ACTUALLY received:\n");
            printf("    in  = 0x%llx  %s\n", h_ptrs[0],
                   h_ptrs[0] == (unsigned long long)d_out ? "(CORRECT = d_out)" :
                   h_ptrs[0] == (unsigned long long)d_in ? "(WRONG = d_in!)" :
                   h_ptrs[0] == (unsigned long long)prev_out ? "(STALE = prev d_out!)" :
                   h_ptrs[0] == (unsigned long long)prev_in ? "(STALE = prev d_in!)" :
                   "(UNKNOWN)");
            printf("    out = 0x%llx  %s\n", h_ptrs[1],
                   h_ptrs[1] == (unsigned long long)d_sum ? "(CORRECT = d_sum)" :
                   "(WRONG)");
            printf("    n   = %llu  %s\n", h_ptrs[2],
                   h_ptrs[2] == N ? "(CORRECT = 256)" : "(WRONG!)");

            printf("\n  Data sum_reduce read from its 'in' pointer:\n    ");
            for (int j = 0; j < DIAG; j++)
                printf("%d ", h_diag[j]);
            printf("\n");

            printf("  Expected (d_out[0..15]):\n    ");
            for (int j = 0; j < DIAG; j++)
                printf("%d ", i + j + 1);
            printf("\n");

            // Also read actual d_out via D2H
            int h_actual_out[DIAG];
            HIP_CHECK(hipMemcpy(h_actual_out, d_out, DIAG * sizeof(int),
                                hipMemcpyDeviceToHost));
            printf("  Actual d_out via D2H:\n    ");
            for (int j = 0; j < DIAG; j++)
                printf("%d ", h_actual_out[j]);
            printf("\n");

            hipFree(d_sum); hipFree(d_out); hipFree(d_in);
            hipFree(d_diag_data); hipFree(d_diag_ptr);
            return 1;
        }

        if ((i + 1) % 100 == 0)
            printf("  %d/500 OK\n", i + 1);

        prev_in = (void*)d_in;
        prev_out = (void*)d_out;

        HIP_CHECK(hipFree(d_sum));
        HIP_CHECK(hipFree(d_out));
        HIP_CHECK(hipFree(d_in));
    }

    hipFree(d_diag_data); hipFree(d_diag_ptr);
    printf("\nALL 500 iters PASS\n");
    return 0;
}
