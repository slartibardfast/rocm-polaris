// test_mfence_placement.cpp — Find WHERE mfence is needed
//
// The dispatch flow is:
//   hipMemset → [sync: signal read] → [return to user] → record_val launch
//                                                        → [allocKernArg]
//                                                        → [memcpy kernarg]
//                                                        → [sfence]
//                                                        → [write AQL]
//                                                        → [doorbell]
//
// We place mfence at different points to isolate which transition matters.
//
// Build: hipcc --offload-arch=gfx803 -o test_mfence_placement test_mfence_placement.cpp
// Run:   ./test_mfence_placement <variant> [iters]
//
// Variants:
//   baseline      — no intervention
//   pre_memset    — mfence BEFORE hipMemset (before previous dispatch)
//   post_memset   — mfence AFTER hipMemset returns (after sync completion)
//   pre_launch    — mfence just before kernel<<<>>> (after hipMemset sync, before new dispatch)
//   post_sync     — mfence after hipDeviceSynchronize of record_val
//   between_only  — mfence ONLY between hipMemset and launch (same as winning variant)
//   no_memset     — skip hipMemset entirely (no prior dispatch)

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <x86intrin.h>
#include <chrono>

__global__ void record_val(int *out, int val) {
    if (threadIdx.x == 0) *out = val;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <variant> [iters]\n", argv[0]);
        return 1;
    }
    const char *v = argv[1];
    int iters = argc > 2 ? atoi(argv[2]) : 2000;

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);

    int *d_result;
    hipMalloc(&d_result, sizeof(int));

    auto t0 = std::chrono::steady_clock::now();
    int fails = 0;

    for (int i = 0; i < iters; i++) {
        if (strcmp(v, "pre_memset") == 0) _mm_mfence();

        if (strcmp(v, "no_memset") != 0) {
            hipMemset(d_result, 0xFF, sizeof(int));
        }

        if (strcmp(v, "post_memset") == 0 ||
            strcmp(v, "between_only") == 0 ||
            strcmp(v, "pre_launch") == 0) _mm_mfence();

        record_val<<<1, 1>>>(d_result, i * 7 + 13);

        if (strcmp(v, "post_sync") == 0) {
            hipDeviceSynchronize();
            _mm_mfence();
        } else {
            hipDeviceSynchronize();
        }

        int h;
        hipMemcpy(&h, d_result, sizeof(int), hipMemcpyDeviceToHost);
        if (h != i * 7 + 13) {
            fails++;
            if (fails <= 3)
                printf("  iter %d: expected %d, got %d\n", i, i*7+13, h);
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    hipFree(d_result);
    printf("%-20s %3d / %d  (%.0f dispatch/s)\n", v, fails, iters, iters/(ms/1000.0));
    return fails > 0 ? 1 : 0;
}
