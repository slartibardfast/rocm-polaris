// test_rocr_store_flush.cpp — Verify ROCR StoreRelaxed HDP flush path
//
// This test uses HIP but proves the ROCR-internal flush is sufficient.
// It does NOT call hsa_amd_hdp_flush_wait directly — the flush happens
// inside AqlQueue::StoreRelaxed before the doorbell write.
//
// If this test passes, the ROCR StoreRelaxed flush path is working
// correctly for gfx8/no-atomics platforms.
//
// Build: hipcc --offload-arch=gfx803 -o test_rocr_store_flush test_rocr_store_flush.cpp
// Run:   timeout 300 ./test_rocr_store_flush

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <chrono>

#define HIP_CHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
                e, hipGetErrorString(e), __FILE__, __LINE__); \
        return false; \
    } \
} while (0)

static volatile sig_atomic_t timed_out = 0;
static void alarm_handler(int) { timed_out = 1; }

__global__ void store_val(int *out, int val) {
    if (threadIdx.x == 0) *out = val;
}

// Test A: Rapid dispatches with no interleaved alloc/free.
// Pure StoreRelaxed path — no CLR kernarg reuse pressure.
static bool test_rapid_dispatch() {
    printf("Test A: Rapid dispatch (5000 iters, no alloc/free)...\n");
    int *d_result;
    HIP_CHECK(hipMalloc(&d_result, sizeof(int)));

    int fails = 0;
    for (int i = 0; i < 5000; i++) {
        int expected = i * 11 + 7;
        HIP_CHECK(hipMemset(d_result, 0xFF, sizeof(int)));
        store_val<<<1, 1>>>(d_result, expected);
        HIP_CHECK(hipDeviceSynchronize());

        int h;
        HIP_CHECK(hipMemcpy(&h, d_result, sizeof(int), hipMemcpyDeviceToHost));
        if (h != expected) {
            printf("  FAIL iter %d: expected %d, got %d (0x%08x)\n",
                   i, expected, h, h);
            fails++;
            if (fails >= 5) break;
        }
        if (timed_out) { hipFree(d_result); return false; }
    }
    hipFree(d_result);
    printf("  %d failures / 5000\n", fails);
    return fails == 0;
}

// Test B: Alloc/free interleave to stress kernarg pool and HDP coherency.
// This is the pattern that triggers corruption without proper flush.
static bool test_alloc_free_interleave() {
    printf("Test B: Alloc/free interleave (3000 iters)...\n");
    int *d_result;
    HIP_CHECK(hipMalloc(&d_result, sizeof(int)));

    int fails = 0;
    for (int i = 0; i < 3000; i++) {
        int *d_tmp;
        HIP_CHECK(hipMalloc(&d_tmp, 1024));

        int expected = i * 13 + 5;
        HIP_CHECK(hipMemset(d_result, 0xFF, sizeof(int)));
        store_val<<<1, 1>>>(d_result, expected);
        HIP_CHECK(hipDeviceSynchronize());

        int h;
        HIP_CHECK(hipMemcpy(&h, d_result, sizeof(int), hipMemcpyDeviceToHost));
        if (h != expected) {
            printf("  FAIL iter %d: expected %d, got %d (0x%08x)\n",
                   i, expected, h, h);
            fails++;
            if (fails >= 5) { hipFree(d_tmp); break; }
        }
        HIP_CHECK(hipFree(d_tmp));
        if (timed_out) { hipFree(d_result); return false; }
    }
    hipFree(d_result);
    printf("  %d failures / 3000\n", fails);
    return fails == 0;
}

// Test C: Multiple kernels per iteration (back-to-back doorbell writes).
static bool test_back_to_back() {
    printf("Test C: Back-to-back dispatches (2000 iters, 3 kernels each)...\n");
    int *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_b, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_c, sizeof(int)));

    int fails = 0;
    for (int i = 0; i < 2000; i++) {
        int va = i * 3, vb = i * 3 + 1, vc = i * 3 + 2;
        store_val<<<1, 1>>>(d_a, va);
        store_val<<<1, 1>>>(d_b, vb);
        store_val<<<1, 1>>>(d_c, vc);
        HIP_CHECK(hipDeviceSynchronize());

        int ha, hb, hc;
        HIP_CHECK(hipMemcpy(&ha, d_a, sizeof(int), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&hb, d_b, sizeof(int), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&hc, d_c, sizeof(int), hipMemcpyDeviceToHost));

        if (ha != va || hb != vb || hc != vc) {
            printf("  FAIL iter %d: a=%d(exp %d) b=%d(exp %d) c=%d(exp %d)\n",
                   i, ha, va, hb, vb, hc, vc);
            fails++;
            if (fails >= 5) break;
        }
        if (timed_out) break;
    }
    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    printf("  %d failures / 2000\n", fails);
    return fails == 0;
}

int main() {
    signal(SIGALRM, alarm_handler);
    alarm(300);

    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
        fprintf(stderr, "hipGetDeviceProperties failed\n");
        return 1;
    }
    printf("Device: %s (%s)\n\n", prop.name, prop.gcnArchName);

    struct { const char *name; bool (*fn)(); } tests[] = {
        {"Rapid dispatch (5000)",           test_rapid_dispatch},
        {"Alloc/free interleave (3000)",    test_alloc_free_interleave},
        {"Back-to-back dispatches (2000)",  test_back_to_back},
    };

    int pass = 0, fail = 0;
    for (auto &t : tests) {
        fflush(stdout);
        auto t0 = std::chrono::steady_clock::now();
        bool ok = t.fn();
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        if (timed_out) { printf("  TIMEOUT\n"); ok = false; }
        printf("  %s: %s (%.1fs)\n\n", t.name, ok ? "PASS" : "FAIL", secs);
        fflush(stdout);
        if (ok) pass++; else fail++;
    }

    printf("=== Results: %d/%d PASS ===\n", pass, pass + fail);
    return fail > 0 ? 1 : 0;
}
