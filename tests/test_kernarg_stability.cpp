// test_kernarg_stability.cpp — Exhaustive kernarg corruption detection
//
// Root cause: kernel arguments (pointers, scalars) are intermittently
// corrupted to zero or wrong values during rapid alloc/free cycling.
// Likely a kernarg pool recycling race in CLR.
//
// This test suite exercises every axis that could trigger the race:
// - Varying kernel argument counts (1 arg, 3 args, 6 args)
// - Varying argument sizes (int, float, pointer, struct-like)
// - Back-to-back dispatches vs interleaved with sync
// - With and without alloc/free between dispatches
// - With and without hipMemcpy between dispatches
// - Multi-kernel chains (A→B→C using each other's output)
// - High kernarg pressure (many unique arg values per dispatch)
//
// Each test verifies that the GPU kernel received the EXACT arguments
// that were passed at launch time, using GPU-side recording.
//
// Build: hipcc --offload-arch=gfx803 -o test_kernarg_stability test_kernarg_stability.cpp
// Run:   timeout 300 ./test_kernarg_stability

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

// ============================================================
// Kernels that record their own arguments for verification
// ============================================================

// 1-arg: just an int
__global__ void record_1arg(int val, int *result) {
    if (threadIdx.x == 0) result[0] = val;
}

// 3-arg: pointer + int + int
__global__ void record_3arg(const int *src, int n, int offset, int *result) {
    if (threadIdx.x == 0) {
        result[0] = n;
        result[1] = offset;
        // Read from src to verify pointer is valid
        result[2] = (n > 0) ? src[0] : -999;
    }
}

// 6-arg: heavy kernarg load
__global__ void record_6arg(const int *a, const int *b, int *out,
                            int n, int alpha, int beta, int *result) {
    if (threadIdx.x == 0) {
        result[0] = n;
        result[1] = alpha;
        result[2] = beta;
        result[3] = (n > 0) ? a[0] : -999;
        result[4] = (n > 0) ? b[0] : -999;
        // Actually compute so we can verify output
        for (int i = 0; i < n && i < 256; i++)
            out[i] = alpha * a[i] + beta * b[i];
    }
}

// Compute kernel with pointer + scalar args
__global__ void scale_add(int *data, int n, int scale, int addend) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = data[i] * scale + addend;
}

// Sum kernel for verification
__global__ void sum_first_n(const int *data, int n, int *out) {
    if (threadIdx.x == 0) {
        int s = 0;
        for (int i = 0; i < n; i++) s += data[i];
        *out = s;
    }
}

// ============================================================
// Test 1: Rapid 1-arg kernel dispatch (1000 iterations)
// Simplest possible kernarg — single int. If this fails,
// the pool is fundamentally broken.
// ============================================================
static bool test_1_rapid_1arg() {
    printf("Test 1: Rapid 1-arg dispatch (1000 iters)...\n");
    int *d_result;
    HIP_CHECK(hipMalloc(&d_result, sizeof(int)));

    for (int i = 0; i < 1000; i++) {
        HIP_CHECK(hipMemset(d_result, 0xFF, sizeof(int)));
        record_1arg<<<1, 1>>>(i * 7 + 13, d_result);
        HIP_CHECK(hipDeviceSynchronize());
        int h;
        HIP_CHECK(hipMemcpy(&h, d_result, sizeof(int), hipMemcpyDeviceToHost));
        if (h != i * 7 + 13) {
            printf("  FAIL iter %d: expected %d, got %d\n", i, i*7+13, h);
            hipFree(d_result);
            return false;
        }
        if (timed_out) { hipFree(d_result); return false; }
    }
    hipFree(d_result);
    return true;
}

// ============================================================
// Test 2: 3-arg dispatch with alloc/free per iteration
// This is the pattern that triggers the race.
// ============================================================
static bool test_2_3arg_alloc_free() {
    printf("Test 2: 3-arg + alloc/free (500 iters)...\n");
    int *d_result;
    HIP_CHECK(hipMalloc(&d_result, 4 * sizeof(int)));

    for (int i = 0; i < 500; i++) {
        int *d_src;
        HIP_CHECK(hipMalloc(&d_src, 64 * sizeof(int)));
        int h_src = i * 3;
        HIP_CHECK(hipMemcpy(d_src, &h_src, sizeof(int), hipMemcpyHostToDevice));

        HIP_CHECK(hipMemset(d_result, 0xFF, 4 * sizeof(int)));
        record_3arg<<<1, 1>>>(d_src, 64, i, d_result);
        HIP_CHECK(hipDeviceSynchronize());

        int h[3];
        HIP_CHECK(hipMemcpy(h, d_result, 3 * sizeof(int), hipMemcpyDeviceToHost));
        if (h[0] != 64 || h[1] != i || h[2] != i * 3) {
            printf("  FAIL iter %d: n=%d(exp 64) offset=%d(exp %d) src[0]=%d(exp %d)\n",
                   i, h[0], h[1], i, h[2], i*3);
            hipFree(d_src); hipFree(d_result);
            return false;
        }
        HIP_CHECK(hipFree(d_src));
        if (timed_out) { hipFree(d_result); return false; }
    }
    hipFree(d_result);
    return true;
}

// ============================================================
// Test 3: 6-arg dispatch with alloc/free (heavy kernarg load)
// 6 args = larger kernarg allocation, more pressure on pool.
// ============================================================
static bool test_3_6arg_alloc_free() {
    printf("Test 3: 6-arg + alloc/free (500 iters)...\n");
    int *d_result;
    HIP_CHECK(hipMalloc(&d_result, 8 * sizeof(int)));

    for (int i = 0; i < 500; i++) {
        int *d_a, *d_b, *d_out;
        const int SZ = 64;
        HIP_CHECK(hipMalloc(&d_a, SZ * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_b, SZ * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_out, SZ * sizeof(int)));

        int h_a[SZ], h_b[SZ];
        for (int j = 0; j < SZ; j++) { h_a[j] = i + j; h_b[j] = 100 + j; }
        HIP_CHECK(hipMemcpy(d_a, h_a, SZ * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_b, h_b, SZ * sizeof(int), hipMemcpyHostToDevice));

        int alpha = 2, beta = 3;
        HIP_CHECK(hipMemset(d_result, 0xFF, 8 * sizeof(int)));
        record_6arg<<<1, 1>>>(d_a, d_b, d_out, SZ, alpha, beta, d_result);
        HIP_CHECK(hipDeviceSynchronize());

        int h[5];
        HIP_CHECK(hipMemcpy(h, d_result, 5 * sizeof(int), hipMemcpyDeviceToHost));
        if (h[0] != SZ || h[1] != alpha || h[2] != beta ||
            h[3] != i || h[4] != 100) {
            printf("  FAIL iter %d: n=%d alpha=%d beta=%d a[0]=%d b[0]=%d\n",
                   i, h[0], h[1], h[2], h[3], h[4]);
            printf("    expected: n=%d alpha=%d beta=%d a[0]=%d b[0]=%d\n",
                   SZ, alpha, beta, i, 100);
            hipFree(d_out); hipFree(d_b); hipFree(d_a); hipFree(d_result);
            return false;
        }

        // Verify output
        int h_out;
        HIP_CHECK(hipMemcpy(&h_out, d_out, sizeof(int), hipMemcpyDeviceToHost));
        int expected_out = alpha * i + beta * 100;
        if (h_out != expected_out) {
            printf("  FAIL iter %d: out[0]=%d expected %d\n", i, h_out, expected_out);
            hipFree(d_out); hipFree(d_b); hipFree(d_a); hipFree(d_result);
            return false;
        }

        HIP_CHECK(hipFree(d_out));
        HIP_CHECK(hipFree(d_b));
        HIP_CHECK(hipFree(d_a));
        if (timed_out) { hipFree(d_result); return false; }
    }
    hipFree(d_result);
    return true;
}

// ============================================================
// Test 4: Back-to-back multi-kernel chain (no sync between)
// A→B→C where each kernel's args include previous output pointer.
// Tests kernarg ordering under rapid submission.
// ============================================================
static bool test_4_kernel_chain_no_sync() {
    printf("Test 4: 3-kernel chain without intermediate sync (200 iters)...\n");
    const int N = 256;
    int *d_data, *d_sum;
    HIP_CHECK(hipMalloc(&d_data, N * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_sum, sizeof(int)));

    for (int i = 0; i < 200; i++) {
        // Fill with i
        int h_data[N];
        for (int j = 0; j < N; j++) h_data[j] = i;
        HIP_CHECK(hipMemcpy(d_data, h_data, N * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_sum, 0, sizeof(int)));

        // Chain: scale_add(x2, +1) → scale_add(x1, +3) → sum
        // Expected per element: (i*2 + 1)*1 + 3 = 2i + 4
        scale_add<<<1, N>>>(d_data, N, 2, 1);
        scale_add<<<1, N>>>(d_data, N, 1, 3);
        sum_first_n<<<1, 1>>>(d_data, N, d_sum);
        HIP_CHECK(hipDeviceSynchronize());

        int h_sum;
        HIP_CHECK(hipMemcpy(&h_sum, d_sum, sizeof(int), hipMemcpyDeviceToHost));
        int expected = N * (2 * i + 4);
        if (h_sum != expected) {
            printf("  FAIL iter %d: expected %d, got %d\n", i, expected, h_sum);
            hipFree(d_sum); hipFree(d_data);
            return false;
        }
        if (timed_out) { hipFree(d_sum); hipFree(d_data); return false; }
    }
    hipFree(d_sum); hipFree(d_data);
    return true;
}

// ============================================================
// Test 5: Alternating alloc/free with multi-kernel chains
// The exact pattern that triggers the bug: alloc, dispatch chain, free.
// ============================================================
static bool test_5_chain_with_alloc_free() {
    printf("Test 5: Multi-kernel chain + alloc/free (500 iters)...\n");
    for (int i = 0; i < 500; i++) {
        const int N = 128;
        int *d_data, *d_sum;
        HIP_CHECK(hipMalloc(&d_data, N * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_sum, sizeof(int)));

        int h_data[N];
        for (int j = 0; j < N; j++) h_data[j] = i;
        HIP_CHECK(hipMemcpy(d_data, h_data, N * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_sum, 0, sizeof(int)));

        scale_add<<<1, N>>>(d_data, N, 1, 1);  // +1
        sum_first_n<<<1, 1>>>(d_data, N, d_sum);
        HIP_CHECK(hipDeviceSynchronize());

        int h_sum;
        HIP_CHECK(hipMemcpy(&h_sum, d_sum, sizeof(int), hipMemcpyDeviceToHost));
        int expected = N * (i + 1);
        if (h_sum != expected) {
            printf("  FAIL iter %d: expected %d, got %d (delta=%d)\n",
                   i, expected, h_sum, h_sum - expected);
            hipFree(d_sum); hipFree(d_data);
            return false;
        }

        HIP_CHECK(hipFree(d_sum));
        HIP_CHECK(hipFree(d_data));
        if (timed_out) return false;
    }
    return true;
}

// ============================================================
// Test 6: Kernarg with large struct-like args
// Passes 8 int args to stress kernarg slot size.
// ============================================================
__global__ void record_8int(int a, int b, int c, int d,
                            int e, int f, int g, int h, int *result) {
    if (threadIdx.x == 0) {
        result[0] = a; result[1] = b; result[2] = c; result[3] = d;
        result[4] = e; result[5] = f; result[6] = g; result[7] = h;
    }
}

static bool test_6_large_args() {
    printf("Test 6: 8-int args + alloc/free (500 iters)...\n");
    int *d_result;
    HIP_CHECK(hipMalloc(&d_result, 8 * sizeof(int)));

    for (int i = 0; i < 500; i++) {
        // Alloc/free to trigger pool recycling
        int *d_tmp;
        HIP_CHECK(hipMalloc(&d_tmp, 1024));
        HIP_CHECK(hipFree(d_tmp));

        int a = i, b = i+1, c = i+2, d = i+3;
        int e = i+4, f = i+5, g = i+6, h = i+7;
        HIP_CHECK(hipMemset(d_result, 0xFF, 8 * sizeof(int)));
        record_8int<<<1, 1>>>(a, b, c, d, e, f, g, h, d_result);
        HIP_CHECK(hipDeviceSynchronize());

        int hr[8];
        HIP_CHECK(hipMemcpy(hr, d_result, 8 * sizeof(int), hipMemcpyDeviceToHost));
        bool ok = true;
        for (int j = 0; j < 8; j++) {
            if (hr[j] != i + j) { ok = false; break; }
        }
        if (!ok) {
            printf("  FAIL iter %d: got %d,%d,%d,%d,%d,%d,%d,%d\n",
                   i, hr[0],hr[1],hr[2],hr[3],hr[4],hr[5],hr[6],hr[7]);
            printf("    expected: %d,%d,%d,%d,%d,%d,%d,%d\n",
                   i,i+1,i+2,i+3,i+4,i+5,i+6,i+7);
            hipFree(d_result);
            return false;
        }
        if (timed_out) { hipFree(d_result); return false; }
    }
    hipFree(d_result);
    return true;
}

// ============================================================
// Test 7: Mixed alloc sizes + kernel dispatch (the 3.3 pattern)
// Random buffer sizes with kernel verification each iteration.
// ============================================================
static bool test_7_mixed_sizes_with_kernel() {
    printf("Test 7: Mixed sizes + kernel verify (300 iters)...\n");
    unsigned int seed = 12345;
    int *d_result;
    HIP_CHECK(hipMalloc(&d_result, sizeof(int)));

    for (int i = 0; i < 300; i++) {
        seed = seed * 1103515245 + 12345;
        size_t sizes[] = {64, 256, 1024, 4096, 16384};
        int n = sizes[seed % 5] / sizeof(int);
        seed = seed * 1103515245 + 12345;
        int val = (seed % 1000) + 1;

        int *d_data;
        HIP_CHECK(hipMalloc(&d_data, n * sizeof(int)));

        // Fill on host, H2D
        int *h = (int *)malloc(n * sizeof(int));
        for (int j = 0; j < n; j++) h[j] = val;
        HIP_CHECK(hipMemcpy(d_data, h, n * sizeof(int), hipMemcpyHostToDevice));

        // Sum via kernel
        HIP_CHECK(hipMemset(d_result, 0, sizeof(int)));
        sum_first_n<<<1, 1>>>(d_data, n, d_result);
        HIP_CHECK(hipDeviceSynchronize());

        int h_sum;
        HIP_CHECK(hipMemcpy(&h_sum, d_result, sizeof(int), hipMemcpyDeviceToHost));
        int expected = n * val;
        if (h_sum != expected) {
            printf("  FAIL iter %d: n=%d val=%d expected=%d got=%d\n",
                   i, n, val, expected, h_sum);
            free(h); hipFree(d_data); hipFree(d_result);
            return false;
        }

        free(h);
        HIP_CHECK(hipFree(d_data));
        if (timed_out) { hipFree(d_result); return false; }
    }
    hipFree(d_result);
    return true;
}

// ============================================================
// Test 8: Sustained kernarg pressure without alloc/free
// Dispatches 5000 kernels with unique args, no alloc/free.
// If this passes but alloc/free tests fail, confirms the
// alloc/free is the trigger, not kernarg exhaustion alone.
// ============================================================
static bool test_8_sustained_no_alloc() {
    printf("Test 8: 5000 dispatches, no alloc/free...\n");
    int *d_result;
    HIP_CHECK(hipMalloc(&d_result, sizeof(int)));

    for (int i = 0; i < 5000; i++) {
        record_1arg<<<1, 1>>>(i * 13 + 7, d_result);
        if (i % 100 == 99) {
            HIP_CHECK(hipDeviceSynchronize());
            int h;
            HIP_CHECK(hipMemcpy(&h, d_result, sizeof(int), hipMemcpyDeviceToHost));
            if (h != i * 13 + 7) {
                printf("  FAIL iter %d: expected %d, got %d\n", i, i*13+7, h);
                hipFree(d_result);
                return false;
            }
        }
        if (timed_out) { hipFree(d_result); return false; }
    }
    hipFree(d_result);
    return true;
}

// ============================================================
// Test 9: hipMemcpy interleaved with kernel dispatch
// Simulates model loading pattern: H2D → kernel → H2D → kernel...
// ============================================================
static bool test_9_memcpy_kernel_interleave() {
    printf("Test 9: H2D + kernel interleave (500 iters, 4KB each)...\n");
    const int N = 1024;

    for (int i = 0; i < 500; i++) {
        int *d_data, *d_sum;
        HIP_CHECK(hipMalloc(&d_data, N * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_sum, sizeof(int)));

        int *h = (int *)malloc(N * sizeof(int));
        for (int j = 0; j < N; j++) h[j] = i + 1;
        HIP_CHECK(hipMemcpy(d_data, h, N * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_sum, 0, sizeof(int)));

        sum_first_n<<<1, 1>>>(d_data, N, d_sum);
        HIP_CHECK(hipDeviceSynchronize());

        int h_sum;
        HIP_CHECK(hipMemcpy(&h_sum, d_sum, sizeof(int), hipMemcpyDeviceToHost));
        int expected = N * (i + 1);
        if (h_sum != expected) {
            printf("  FAIL iter %d: expected %d, got %d\n", i, expected, h_sum);
            free(h); hipFree(d_sum); hipFree(d_data);
            return false;
        }

        free(h);
        HIP_CHECK(hipFree(d_sum));
        HIP_CHECK(hipFree(d_data));
        if (timed_out) return false;
    }
    return true;
}

// ============================================================
// Main
// ============================================================

int main() {
    signal(SIGALRM, alarm_handler);
    alarm(300);

    hipDeviceProp_t prop;
    hipError_t e = hipGetDeviceProperties(&prop, 0);
    if (e != hipSuccess) {
        fprintf(stderr, "hipGetDeviceProperties failed: %d\n", e);
        return 1;
    }
    printf("Device: %s (%s)\n\n", prop.name, prop.gcnArchName);

    struct { const char *name; bool (*fn)(); } tests[] = {
        {"1. Rapid 1-arg (1000 iters)",              test_1_rapid_1arg},
        {"2. 3-arg + alloc/free (500)",               test_2_3arg_alloc_free},
        {"3. 6-arg + alloc/free (500)",               test_3_6arg_alloc_free},
        {"4. 3-kernel chain no sync (200)",           test_4_kernel_chain_no_sync},
        {"5. Chain + alloc/free (500)",                test_5_chain_with_alloc_free},
        {"6. 8-int args + alloc/free (500)",          test_6_large_args},
        {"7. Mixed sizes + kernel (300)",             test_7_mixed_sizes_with_kernel},
        {"8. 5000 dispatches no alloc/free",          test_8_sustained_no_alloc},
        {"9. H2D + kernel interleave (500)",          test_9_memcpy_kernel_interleave},
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
