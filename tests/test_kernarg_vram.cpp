// test_kernarg_vram.cpp — Verify kernarg coherency after VRAM migration
//
// Tests the EXACT failure mode we identified: GPU L2 retains stale
// kernarg data when the kernarg pool is in system memory and addresses
// are reused.  VRAM kernarg should fix this since GPU L2 is coherent
// with its own VRAM.
//
// These tests must ALL pass with 0 failures on 10 consecutive runs.
// If any fail, the VRAM kernarg migration didn't fix the root cause.
//
// Build: hipcc --offload-arch=gfx803 -o test_kernarg_vram test_kernarg_vram.cpp
// Run:   timeout 300 ./test_kernarg_vram

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

// Record scalar arg — the exact failure mode (got 39 instead of 2855)
__global__ void record_1arg(int val, int *result) {
    if (threadIdx.x == 0) result[0] = val;
}

// Record 3 args — pointer dereference + scalars
__global__ void record_3arg(const int *src, int n, int offset, int *result) {
    if (threadIdx.x == 0) {
        result[0] = n;
        result[1] = offset;
        result[2] = (n > 0) ? src[0] : -999;
    }
}

// Compute kernel — actual work, not just recording
__global__ void add_one(const int *in, int *out, int n) {
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

// ============================================================
// Test 1: Rapid scalar arg (the smoking gun test)
// 2000 iterations of record_1arg with unique values.
// Previously failed at ~iter 150-400 with stale values.
// ============================================================
static bool test_scalar_arg_rapid() {
    printf("Test 1: Rapid scalar arg dispatch (2000 iters)...\n");
    int *d_result;
    HIP_CHECK(hipMalloc(&d_result, sizeof(int)));

    for (int i = 0; i < 2000; i++) {
        int expected = i * 7 + 13;
        HIP_CHECK(hipMemset(d_result, 0xFF, sizeof(int)));
        record_1arg<<<1, 1>>>(expected, d_result);
        HIP_CHECK(hipDeviceSynchronize());
        int h;
        HIP_CHECK(hipMemcpy(&h, d_result, sizeof(int), hipMemcpyDeviceToHost));
        if (h != expected) {
            printf("  FAIL iter %d: expected %d, got %d (0x%08x)\n",
                   i, expected, h, h);
            hipFree(d_result);
            return false;
        }
        if (timed_out) { hipFree(d_result); return false; }
    }
    hipFree(d_result);
    return true;
}

// ============================================================
// Test 2: Pointer + scalar args with alloc/free per iteration
// Forces kernarg pool reset via releaseGpuMemoryFence on each hipMemcpy.
// ============================================================
static bool test_ptr_arg_with_alloc() {
    printf("Test 2: Pointer+scalar args + alloc/free (1000 iters)...\n");
    int *d_result;
    HIP_CHECK(hipMalloc(&d_result, 4 * sizeof(int)));

    for (int i = 0; i < 1000; i++) {
        int *d_src;
        HIP_CHECK(hipMalloc(&d_src, 64 * sizeof(int)));
        int h_val = i * 3;
        HIP_CHECK(hipMemcpy(d_src, &h_val, sizeof(int), hipMemcpyHostToDevice));

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
// Test 3: Kernel chain + alloc/free (the 3.2 failure pattern)
// alloc→H2D→add_one→sum_reduce→D2H→verify→free, 1000 iterations.
// Previously returned "got 128" (half-computed sum).
// ============================================================
static bool test_chain_alloc_free() {
    printf("Test 3: Kernel chain + alloc/free (1000 iters)...\n");
    for (int i = 0; i < 1000; i++) {
        const int N = 256;
        int *d_in, *d_out, *d_sum;
        int h_in[N];
        for (int j = 0; j < N; j++) h_in[j] = i + j;

        HIP_CHECK(hipMalloc(&d_in, N * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_out, N * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_sum, sizeof(int)));
        HIP_CHECK(hipMemset(d_sum, 0, sizeof(int)));
        HIP_CHECK(hipMemcpy(d_in, h_in, N * sizeof(int), hipMemcpyHostToDevice));

        add_one<<<1, N>>>(d_in, d_out, N);
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
// Test 4: H2D + kernel interleave (the test 9 failure pattern)
// 1000 iterations of alloc→H2D→sum→D2H→verify→free.
// Previously returned "got 0" (H2D failed silently).
// ============================================================
static bool test_h2d_kernel_interleave() {
    printf("Test 4: H2D + kernel interleave (1000 iters)...\n");
    const int N = 1024;

    for (int i = 0; i < 1000; i++) {
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
        if ((i + 1) % 200 == 0) printf("    %d/1000 OK\n", i + 1);
        if (timed_out) return false;
    }
    return true;
}

// ============================================================
// Test 5: Sustained mixed pattern (the full llama.cpp pattern)
// alloc→H2D→kernel→kernel→D2H→verify→free, 2000 iterations.
// ============================================================
static bool test_sustained_mixed() {
    printf("Test 5: Sustained mixed pattern (2000 iters)...\n");
    const int N = 256;

    for (int i = 0; i < 2000; i++) {
        int *d_in, *d_out, *d_sum;
        HIP_CHECK(hipMalloc(&d_in, N * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_out, N * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_sum, sizeof(int)));

        int h_in[N];
        for (int j = 0; j < N; j++) h_in[j] = i + j;
        HIP_CHECK(hipMemcpy(d_in, h_in, N * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_sum, 0, sizeof(int)));

        add_one<<<1, N>>>(d_in, d_out, N);
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
        if ((i + 1) % 500 == 0) printf("    %d/2000 OK\n", i + 1);
        if (timed_out) return false;
    }
    return true;
}

int main() {
    signal(SIGALRM, alarm_handler);
    alarm(300);

    hipDeviceProp_t prop;
    hipError_t e = hipGetDeviceProperties(&prop, 0);
    if (e != hipSuccess) { fprintf(stderr, "hipGetDeviceProperties failed\n"); return 1; }
    printf("Device: %s (%s)\n\n", prop.name, prop.gcnArchName);

    struct { const char *name; bool (*fn)(); } tests[] = {
        {"Rapid scalar arg (2000)",          test_scalar_arg_rapid},
        {"Pointer+scalar + alloc/free (1000)", test_ptr_arg_with_alloc},
        {"Kernel chain + alloc/free (1000)",   test_chain_alloc_free},
        {"H2D + kernel interleave (1000)",     test_h2d_kernel_interleave},
        {"Sustained mixed (2000)",             test_sustained_mixed},
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
