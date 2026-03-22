// test_isolate_fix.cpp — Isolate what the "L2 scrub" was actually fixing
//
// Runs rapid scalar arg dispatch with different interventions between
// iterations to identify which aspect eliminates corruption.
//
// Build: hipcc --offload-arch=gfx803 -o test_isolate_fix test_isolate_fix.cpp
// Run:   ./test_isolate_fix

#include <hip/hip_runtime.h>
#include <cstdio>
#include <unistd.h>

__global__ void record_val(int *out, int val) {
    if (threadIdx.x == 0) *out = val;
}

__global__ void nop_kernel(int *out) {
    if (threadIdx.x == 0) *out = 1;
}

static int run_variant(const char *label, int iters,
                       void *scrub, int scrub_size,
                       int *nop_buf, int usleep_us, bool do_nop) {
    int *d_result;
    hipMalloc(&d_result, sizeof(int));

    int fails = 0;
    for (int i = 0; i < iters; i++) {
        // Intervention
        if (usleep_us > 0) usleep(usleep_us);
        if (do_nop) { nop_kernel<<<1,1>>>(nop_buf); hipDeviceSynchronize(); }
        if (scrub && scrub_size > 0) { hipMemset(scrub, 0, scrub_size); }

        // The actual test
        hipMemset(d_result, 0xFF, sizeof(int));
        record_val<<<1, 1>>>(d_result, i * 7 + 13);
        hipDeviceSynchronize();

        int h;
        hipMemcpy(&h, d_result, sizeof(int), hipMemcpyDeviceToHost);
        if (h != i * 7 + 13) {
            fails++;
            if (fails <= 3)
                printf("    iter %d: expected %d, got %d\n", i, i*7+13, h);
        }
    }
    hipFree(d_result);
    printf("  %-40s %d fails / %d\n", label, fails, iters);
    return fails;
}

int main() {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("Device: %s (%s)\n\n", prop.name, prop.gcnArchName);

    void *scrub;
    hipMalloc(&scrub, 512 * 1024);
    int *nop_buf;
    hipMalloc(&nop_buf, sizeof(int));

    const int N = 2000;

    printf("A: No intervention (baseline)\n");
    run_variant("baseline", N, nullptr, 0, nullptr, 0, false);

    printf("B: usleep(10) — 10us CPU delay\n");
    run_variant("usleep(10)", N, nullptr, 0, nullptr, 10, false);

    printf("C: usleep(100) — 100us CPU delay\n");
    run_variant("usleep(100)", N, nullptr, 0, nullptr, 100, false);

    printf("D: usleep(1000) — 1ms CPU delay\n");
    run_variant("usleep(1000)", N, nullptr, 0, nullptr, 1000, false);

    printf("E: Empty kernel + sync\n");
    run_variant("nop kernel+sync", N, nullptr, 0, nop_buf, 0, true);

    printf("F: hipMemset 64B\n");
    run_variant("memset 64B", N, scrub, 64, nullptr, 0, false);

    printf("G: hipMemset 4KB\n");
    run_variant("memset 4KB", N, scrub, 4096, nullptr, 0, false);

    printf("H: hipMemset 512KB (original scrub)\n");
    run_variant("memset 512KB", N, scrub, 512*1024, nullptr, 0, false);

    hipFree(nop_buf);
    hipFree(scrub);
    return 0;
}
