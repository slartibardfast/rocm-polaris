// test_gpu_isolation.cpp — Resource isolation and lifecycle correctness
//
// Tests every gap identified in the Phase 7g investigation:
// 1. VA reuse after hipFree (stale GPU L2/TLB at recycled address)
// 2. Staging buffer chunk rotation (ManagedBuffer pool overflow)
// 3. Signal pool lifecycle (leak detection under sustained use)
// 4. hipMemcpy after hipFree at same VA (exact failure path)
// 5. Sustained mixed pattern (alloc→H2D→kernel→D2H→verify→free x1000)
// 6. Large buffer VA reuse stress
//
// Build: hipcc --offload-arch=gfx803 -o test_gpu_isolation test_gpu_isolation.cpp
// Run:   timeout 600 ./test_gpu_isolation

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

__global__ void verify_val(const unsigned char *d, unsigned char expected,
                           int n, int *bad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && d[i] != expected) atomicAdd(bad, 1);
}

__global__ void fill_val(unsigned char *d, unsigned char val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = val;
}

__global__ void add_one_int(const int *in, int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] + 1;
}

__global__ void sum_first_n(const int *data, int n, int *out) {
    if (threadIdx.x == 0) {
        int s = 0;
        for (int i = 0; i < n; i++) s += data[i];
        *out = s;
    }
}

__global__ void record_arg(int val, int *result) {
    if (threadIdx.x == 0) result[0] = val;
}

// ============================================================
// Test 1: VA reuse after hipFree — stale data detection
// ============================================================
static bool test_va_reuse() {
    printf("Test 1: VA reuse stale data detection (500 iters)...\n");
    const size_t SZ = 4096;
    int reuses = 0;

    for (int i = 0; i < 500; i++) {
        unsigned char val_a = 0xAA;
        unsigned char val_b = (unsigned char)((i % 200) + 0x10);
        if (val_b == val_a) val_b = 0xBB;

        unsigned char *d_a;
        HIP_CHECK(hipMalloc(&d_a, SZ));
        fill_val<<<(SZ+255)/256, 256>>>(d_a, val_a, SZ);
        HIP_CHECK(hipDeviceSynchronize());
        void *va_a = (void*)d_a;
        HIP_CHECK(hipFree(d_a));

        unsigned char *d_b;
        HIP_CHECK(hipMalloc(&d_b, SZ));
        bool reused = ((void*)d_b == va_a);
        if (reused) reuses++;

        unsigned char *h = (unsigned char *)malloc(SZ);
        memset(h, val_b, SZ);
        HIP_CHECK(hipMemcpy(d_b, h, SZ, hipMemcpyHostToDevice));

        int *d_bad;
        HIP_CHECK(hipMalloc(&d_bad, sizeof(int)));
        HIP_CHECK(hipMemset(d_bad, 0, sizeof(int)));
        verify_val<<<(SZ+255)/256, 256>>>(d_b, val_b, SZ, d_bad);
        HIP_CHECK(hipDeviceSynchronize());

        int bad = -1;
        HIP_CHECK(hipMemcpy(&bad, d_bad, sizeof(int), hipMemcpyDeviceToHost));

        if (bad != 0) {
            unsigned char h2[4];
            HIP_CHECK(hipMemcpy(h2, d_b, 4, hipMemcpyDeviceToHost));
            printf("  FAIL iter %d: bad=%d reused=%d expected=0x%02x got=0x%02x\n",
                   i, bad, reused, val_b, h2[0]);
            hipFree(d_bad); hipFree(d_b); free(h);
            return false;
        }

        hipFree(d_bad); hipFree(d_b); free(h);
        if (timed_out) return false;
    }
    printf("    500 OK (VA reuses: %d)\n", reuses);
    return true;
}

// ============================================================
// Test 2: Staging buffer chunk rotation (force pool overflow)
// ============================================================
static bool test_staging_rotation() {
    printf("Test 2: Staging chunk rotation (100 iters, 128KB each)...\n");
    const size_t SZ = 128 * 1024;

    for (int i = 0; i < 100; i++) {
        unsigned char val = (unsigned char)((i % 250) + 1);
        unsigned char *h = (unsigned char *)malloc(SZ);
        memset(h, val, SZ);

        unsigned char *d;
        int *d_bad;
        HIP_CHECK(hipMalloc(&d, SZ));
        HIP_CHECK(hipMalloc(&d_bad, sizeof(int)));
        HIP_CHECK(hipMemset(d_bad, 0, sizeof(int)));

        HIP_CHECK(hipMemcpy(d, h, SZ, hipMemcpyHostToDevice));
        verify_val<<<(SZ+255)/256, 256>>>(d, val, SZ, d_bad);
        HIP_CHECK(hipDeviceSynchronize());

        int bad = -1;
        HIP_CHECK(hipMemcpy(&bad, d_bad, sizeof(int), hipMemcpyDeviceToHost));
        if (bad != 0) {
            printf("  FAIL iter %d: bad=%d val=0x%02x\n", i, bad, val);
            hipFree(d_bad); hipFree(d); free(h);
            return false;
        }

        unsigned char *h2 = (unsigned char *)malloc(SZ);
        HIP_CHECK(hipMemcpy(h2, d, SZ, hipMemcpyDeviceToHost));
        for (size_t j = 0; j < SZ; j++) {
            if (h2[j] != val) {
                printf("  FAIL iter %d: D2H mismatch at byte %zu\n", i, j);
                free(h2); hipFree(d_bad); hipFree(d); free(h);
                return false;
            }
        }

        free(h2); hipFree(d_bad); hipFree(d); free(h);
        if (timed_out) return false;
    }
    return true;
}

// ============================================================
// Test 3: Signal pool lifecycle (10000 dispatches)
// ============================================================
static bool test_signal_lifecycle() {
    printf("Test 3: Signal pool lifecycle (10000 dispatches)...\n");
    int *d_result;
    HIP_CHECK(hipMalloc(&d_result, sizeof(int)));

    for (int i = 0; i < 10000; i++) {
        int expected = i * 7 + 3;
        record_arg<<<1, 1>>>(expected, d_result);

        if (i % 500 == 499) {
            HIP_CHECK(hipDeviceSynchronize());
            int h;
            HIP_CHECK(hipMemcpy(&h, d_result, sizeof(int), hipMemcpyDeviceToHost));
            if (h != expected) {
                printf("  FAIL at dispatch %d: expected %d, got %d\n", i, expected, h);
                hipFree(d_result);
                return false;
            }
        }
        if (timed_out) { hipFree(d_result); return false; }
    }

    HIP_CHECK(hipDeviceSynchronize());
    int h;
    HIP_CHECK(hipMemcpy(&h, d_result, sizeof(int), hipMemcpyDeviceToHost));
    if (h != 9999 * 7 + 3) {
        printf("  FAIL final: expected %d, got %d\n", 9999*7+3, h);
        hipFree(d_result);
        return false;
    }
    hipFree(d_result);
    return true;
}

// ============================================================
// Test 4: hipMemcpy after hipFree at same VA (1000 iters)
// ============================================================
static bool test_memcpy_after_free() {
    printf("Test 4: hipMemcpy after hipFree at same VA (1000 iters)...\n");
    const size_t SZ = 1024;
    void *prev_va = nullptr;
    int reuses = 0;

    for (int i = 0; i < 1000; i++) {
        unsigned char val = (unsigned char)((i % 250) + 1);
        unsigned char *d;
        HIP_CHECK(hipMalloc(&d, SZ));

        if ((void*)d == prev_va) reuses++;

        unsigned char *h = (unsigned char *)malloc(SZ);
        memset(h, val, SZ);
        HIP_CHECK(hipMemcpy(d, h, SZ, hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());

        int *d_bad;
        HIP_CHECK(hipMalloc(&d_bad, sizeof(int)));
        HIP_CHECK(hipMemset(d_bad, 0, sizeof(int)));
        verify_val<<<(SZ+255)/256, 256>>>(d, val, SZ, d_bad);
        HIP_CHECK(hipDeviceSynchronize());

        int bad = -1;
        HIP_CHECK(hipMemcpy(&bad, d_bad, sizeof(int), hipMemcpyDeviceToHost));
        if (bad != 0) {
            printf("  FAIL iter %d: bad=%d val=0x%02x reused=%d\n",
                   i, bad, val, (void*)d == prev_va);
            hipFree(d_bad); hipFree(d); free(h);
            return false;
        }

        prev_va = (void*)d;
        hipFree(d_bad); hipFree(d); free(h);
        if (timed_out) return false;
    }
    printf("    1000 OK (VA reuses: %d)\n", reuses);
    return true;
}

// ============================================================
// Test 5: Sustained mixed (alloc→H2D→kernel→D2H→verify→free x1000)
// ============================================================
static bool test_sustained_mixed() {
    printf("Test 5: Sustained mixed pattern (1000 iters)...\n");
    const int N = 256;

    for (int i = 0; i < 1000; i++) {
        int *d_in, *d_out, *d_sum;
        HIP_CHECK(hipMalloc(&d_in, N * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_out, N * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_sum, sizeof(int)));

        int h_in[N];
        for (int j = 0; j < N; j++) h_in[j] = i + j;
        HIP_CHECK(hipMemcpy(d_in, h_in, N * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_sum, 0, sizeof(int)));

        add_one_int<<<1, N>>>(d_in, d_out, N);
        sum_first_n<<<1, 1>>>(d_out, N, d_sum);
        HIP_CHECK(hipDeviceSynchronize());

        int h_sum;
        HIP_CHECK(hipMemcpy(&h_sum, d_sum, sizeof(int), hipMemcpyDeviceToHost));
        int expected = 0;
        for (int j = 0; j < N; j++) expected += i + j + 1;

        if (h_sum != expected) {
            printf("  FAIL iter %d: expected %d, got %d\n", i, expected, h_sum);
            hipFree(d_sum); hipFree(d_out); hipFree(d_in);
            return false;
        }

        HIP_CHECK(hipFree(d_sum));
        HIP_CHECK(hipFree(d_out));
        HIP_CHECK(hipFree(d_in));

        if ((i + 1) % 200 == 0) printf("    %d/1000 OK\n", i + 1);
        if (timed_out) return false;
    }
    return true;
}

// ============================================================
// Test 6: Large buffer VA reuse (1MB)
// ============================================================
static bool test_large_va_reuse() {
    printf("Test 6: Large buffer VA reuse (200 iters, 1MB)...\n");
    const size_t SZ = 1 << 20;

    for (int i = 0; i < 200; i++) {
        unsigned char val = (unsigned char)((i % 250) + 1);
        unsigned char *d;
        HIP_CHECK(hipMalloc(&d, SZ));

        unsigned char *h = (unsigned char *)malloc(SZ);
        memset(h, val, SZ);
        HIP_CHECK(hipMemcpy(d, h, SZ, hipMemcpyHostToDevice));

        int *d_bad;
        HIP_CHECK(hipMalloc(&d_bad, sizeof(int)));
        HIP_CHECK(hipMemset(d_bad, 0, sizeof(int)));
        verify_val<<<(SZ+255)/256, 256>>>(d, val, SZ, d_bad);
        HIP_CHECK(hipDeviceSynchronize());

        int bad = -1;
        HIP_CHECK(hipMemcpy(&bad, d_bad, sizeof(int), hipMemcpyDeviceToHost));
        if (bad != 0) {
            printf("  FAIL iter %d: bad=%d\n", i, bad);
            hipFree(d_bad); hipFree(d); free(h);
            return false;
        }

        hipFree(d_bad); hipFree(d); free(h);
        if (timed_out) return false;
    }
    return true;
}

// ============================================================
// Main
// ============================================================

int main() {
    signal(SIGALRM, alarm_handler);
    alarm(600);

    hipDeviceProp_t prop;
    hipError_t e = hipGetDeviceProperties(&prop, 0);
    if (e != hipSuccess) {
        fprintf(stderr, "hipGetDeviceProperties failed: %d\n", e);
        return 1;
    }
    printf("Device: %s (%s)\n\n", prop.name, prop.gcnArchName);

    struct { const char *name; bool (*fn)(); } tests[] = {
        {"VA reuse stale data (500)",          test_va_reuse},
        {"Staging chunk rotation (100x128KB)", test_staging_rotation},
        {"Signal pool lifecycle (10000)",      test_signal_lifecycle},
        {"hipMemcpy after hipFree (1000)",     test_memcpy_after_free},
        {"Sustained mixed pattern (1000)",     test_sustained_mixed},
        {"Large buffer VA reuse (200x1MB)",    test_large_va_reuse},
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
    fflush(stdout);
    return fail > 0 ? 1 : 0;
}
