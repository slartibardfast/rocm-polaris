// test_corruption_pattern.cpp — Characterize corruption values
//
// Records ALL failure details: iteration, expected value, got value,
// delta, and the relationship to previous iterations' values.
//
// Build: hipcc --offload-arch=gfx803 -o test_corruption_pattern test_corruption_pattern.cpp
// Run:   ./test_corruption_pattern [iters]

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

__global__ void record_val(int *out, int val) {
    if (threadIdx.x == 0) *out = val;
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 10000;

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("Device: %s (%s)\n", prop.name, prop.gcnArchName);
    printf("Pattern: val = i * 7 + 13\n\n");

    int *d_result;
    hipMalloc(&d_result, sizeof(int));

    int fails = 0;
    int prev_expected = -1;
    int prev_got = -1;
    int prev_fail_iter = -1;

    printf("%-8s %-12s %-12s %-12s %-8s %-20s\n",
           "iter", "expected", "got", "delta", "got_hex", "analysis");

    for (int i = 0; i < iters; i++) {
        int expected = i * 7 + 13;
        hipMemset(d_result, 0xFF, sizeof(int));
        record_val<<<1, 1>>>(d_result, expected);
        hipDeviceSynchronize();

        int h;
        hipMemcpy(&h, d_result, sizeof(int), hipMemcpyDeviceToHost);

        if (h != expected) {
            // Analyze the corrupted value
            int delta = h - expected;

            // Check if 'got' matches any prior iteration's expected value
            int match_iter = -1;
            if (h >= 13 && (h - 13) % 7 == 0) {
                match_iter = (h - 13) / 7;
            }

            // Check if 'got' is 0xFFFFFFFF (memset pattern)
            const char *analysis = "";
            if (h == (int)0xFFFFFFFF) analysis = "memset_pattern";
            else if (h == 0) analysis = "ZERO";
            else if (match_iter >= 0 && match_iter < i) {
                static char buf[64];
                snprintf(buf, sizeof(buf), "STALE iter %d (age=%d)", match_iter, i - match_iter);
                analysis = buf;
            }
            else if (h == prev_got) analysis = "REPEAT_prev_fail";
            else if (h > 0 && h < 256) analysis = "small_value";
            else analysis = "unknown";

            printf("%-8d %-12d %-12d %-12d 0x%08x %s\n",
                   i, expected, h, delta, (unsigned)h, analysis);

            if (fails < 200) {
                prev_fail_iter = i;
                prev_got = h;
            }
            fails++;
        }
        prev_expected = expected;
    }

    hipFree(d_result);
    printf("\nTotal: %d fails / %d (%.3f%%)\n", fails, iters, 100.0*fails/iters);
    return fails > 0 ? 1 : 0;
}
