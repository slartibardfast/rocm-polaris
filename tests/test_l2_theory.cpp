// test_l2_theory.cpp — Verify GPU L2 stale read theory for kernarg corruption
//
// Runs rapid scalar arg dispatch with and without a 512KB L2 scrub.
// If scrub = 0 failures and no-scrub = failures → L2 confirmed.
//
// Build: hipcc --offload-arch=gfx803 -o test_l2_theory test_l2_theory.cpp
// Run:   ./test_l2_theory

#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void record_val(int *out, int val) {
    if (threadIdx.x == 0) *out = val;
}

static int run_test(const char *label, void *scrub, int iters) {
    int *d_result;
    hipMalloc(&d_result, sizeof(int));

    int fails = 0;
    for (int i = 0; i < iters; i++) {
        // Optional L2 scrub: 512KB memset fills entire L2
        if (scrub) hipMemset(scrub, 0, 512 * 1024);

        hipMemset(d_result, 0xFF, sizeof(int));
        record_val<<<1, 1>>>(d_result, i * 7 + 13);
        hipDeviceSynchronize();

        int h;
        hipMemcpy(&h, d_result, sizeof(int), hipMemcpyDeviceToHost);
        if (h != i * 7 + 13) {
            if (fails < 5)
                printf("  FAIL iter %d: expected %d, got %d\n", i, i*7+13, h);
            fails++;
        }
    }
    hipFree(d_result);
    printf("  %s: %d failures / %d\n", label, fails, iters);
    return fails;
}

int main() {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("Device: %s (%s)\n\n", prop.name, prop.gcnArchName);

    void *scrub;
    hipMalloc(&scrub, 512 * 1024);

    // Run WITHOUT scrub first (expect failures)
    printf("Test A: NO L2 scrub (2000 iters)\n");
    int fails_no_scrub = run_test("no-scrub", nullptr, 2000);

    // Run WITH scrub (expect 0 failures if L2 is root cause)
    printf("\nTest B: WITH L2 scrub (2000 iters)\n");
    int fails_with_scrub = run_test("with-scrub", scrub, 2000);

    printf("\n=== VERDICT ===\n");
    if (fails_no_scrub > 0 && fails_with_scrub == 0) {
        printf("L2 THEORY CONFIRMED: scrub eliminates corruption\n");
    } else if (fails_no_scrub == 0 && fails_with_scrub == 0) {
        printf("INCONCLUSIVE: no failures in either mode (run again)\n");
    } else if (fails_with_scrub > 0) {
        printf("L2 THEORY DISPROVED: scrub does not fix corruption\n");
    }

    hipFree(scrub);
    return (fails_no_scrub > 0 && fails_with_scrub == 0) ? 0 : 1;
}
