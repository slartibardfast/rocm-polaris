// hip_barrier_test.cpp — Exhaustive barrier path acceptance tests
//
// Tests every CLR dispatchBarrierPacket() call site to verify
// CPU-managed barriers work on no-atomics platforms.
//
// Build: hipcc --offload-arch=gfx803 -o hip_barrier_test hip_barrier_test.cpp
// Run:   timeout 900 ./hip_barrier_test

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <chrono>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

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

// --- Kernels ---

__global__ void set_val(int *out, int val) {
    out[threadIdx.x] = val;
}

__global__ void add_one(const int *in, int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] + 1;
}

__global__ void fill_pattern(unsigned char *out, int n, unsigned char val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = val;
}

__global__ void verify_pattern(const unsigned char *d, unsigned char expected,
                               int n, int *bad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && d[i] != expected) atomicAdd(bad, 1);
}

__global__ void sum_reduce(const int *in, int *out, int n) {
    int s = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        s += in[i];
    atomicAdd(out, s);
}

// ============================================================
// Tier 2: Barrier-specific tests
// ============================================================

// 2.1: Kernel -> memcpy -> kernel chain (releaseGpuMemoryFence barriers)
static bool test_2_1_kernel_memcpy_chain() {
    printf("Test 2.1: Kernel->memcpy->kernel chain (100 iters)...\n");
    int *d_buf, h_val;
    HIP_CHECK(hipMalloc(&d_buf, sizeof(int)));

    for (int i = 0; i < 100; i++) {
        set_val<<<1, 1>>>(d_buf, i * 3 + 7);
        HIP_CHECK(hipMemcpy(&h_val, d_buf, sizeof(int), hipMemcpyDeviceToHost));
        if (h_val != i * 3 + 7) {
            printf("  FAIL iter %d: expected %d, got %d\n", i, i*3+7, h_val);
            hipFree(d_buf);
            return false;
        }
        // Launch another kernel that depends on the readback being done
        set_val<<<1, 1>>>(d_buf, h_val + 1);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(&h_val, d_buf, sizeof(int), hipMemcpyDeviceToHost));
        if (h_val != i * 3 + 8) {
            printf("  FAIL iter %d pass 2: expected %d, got %d\n", i, i*3+8, h_val);
            hipFree(d_buf);
            return false;
        }
        if (timed_out) { hipFree(d_buf); return false; }
    }
    hipFree(d_buf);
    return true;
}

// 2.2: hipEventRecord + hipEventSynchronize (marker barrier path)
static bool test_2_2_event_sync() {
    printf("Test 2.2: hipEventRecord + hipEventSynchronize (500 iters)...\n");
    int *d_buf, h_val;
    hipEvent_t ev;
    HIP_CHECK(hipMalloc(&d_buf, sizeof(int)));
    HIP_CHECK(hipEventCreate(&ev));

    for (int i = 0; i < 500; i++) {
        set_val<<<1, 1>>>(d_buf, i);
        HIP_CHECK(hipEventRecord(ev, 0));
        HIP_CHECK(hipEventSynchronize(ev));
        HIP_CHECK(hipMemcpy(&h_val, d_buf, sizeof(int), hipMemcpyDeviceToHost));
        if (h_val != i) {
            printf("  FAIL iter %d: expected %d, got %d\n", i, i, h_val);
            hipEventDestroy(ev); hipFree(d_buf);
            return false;
        }
        if (timed_out) { hipEventDestroy(ev); hipFree(d_buf); return false; }
    }
    hipEventDestroy(ev);
    hipFree(d_buf);
    return true;
}

// 2.3: hipStreamSynchronize under load (flush barrier)
static bool test_2_3_stream_sync_load() {
    printf("Test 2.3: hipStreamSynchronize under load (10 rounds x 100 kernels)...\n");
    int *d_buf, h_val;
    hipStream_t stream;
    HIP_CHECK(hipMalloc(&d_buf, sizeof(int)));
    HIP_CHECK(hipStreamCreate(&stream));

    for (int round = 0; round < 10; round++) {
        for (int i = 0; i < 100; i++) {
            set_val<<<1, 1, 0, stream>>>(d_buf, round * 100 + i);
        }
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipMemcpy(&h_val, d_buf, sizeof(int), hipMemcpyDeviceToHost));
        if (h_val != round * 100 + 99) {
            printf("  FAIL round %d: expected %d, got %d\n",
                   round, round*100+99, h_val);
            hipStreamDestroy(stream); hipFree(d_buf);
            return false;
        }
        if (timed_out) { hipStreamDestroy(stream); hipFree(d_buf); return false; }
    }
    hipStreamDestroy(stream);
    hipFree(d_buf);
    return true;
}

// 2.4: Cross-stream event wait (dispatchBlockingWait barrier)
static bool test_2_4_cross_stream_event() {
    printf("Test 2.4: Cross-stream event wait (100 iters)...\n");
    int *d_a, *d_b, h_val;
    hipStream_t sa, sb;
    hipEvent_t ev;
    HIP_CHECK(hipMalloc(&d_a, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_b, sizeof(int)));
    HIP_CHECK(hipStreamCreate(&sa));
    HIP_CHECK(hipStreamCreate(&sb));
    HIP_CHECK(hipEventCreate(&ev));

    for (int i = 0; i < 100; i++) {
        // Stream A writes a value
        set_val<<<1, 1, 0, sa>>>(d_a, i * 5);
        HIP_CHECK(hipEventRecord(ev, sa));

        // Stream B waits for stream A's event, then reads A's result
        HIP_CHECK(hipStreamWaitEvent(sb, ev, 0));
        // Copy A's result to B's buffer via kernel (proves ordering)
        add_one<<<1, 1, 0, sb>>>(d_a, d_b, 1);
        HIP_CHECK(hipStreamSynchronize(sb));

        HIP_CHECK(hipMemcpy(&h_val, d_b, sizeof(int), hipMemcpyDeviceToHost));
        if (h_val != i * 5 + 1) {
            printf("  FAIL iter %d: expected %d, got %d\n", i, i*5+1, h_val);
            hipEventDestroy(ev);
            hipStreamDestroy(sb); hipStreamDestroy(sa);
            hipFree(d_b); hipFree(d_a);
            return false;
        }
        if (timed_out) break;
    }
    hipEventDestroy(ev);
    hipStreamDestroy(sb); hipStreamDestroy(sa);
    hipFree(d_b); hipFree(d_a);
    return !timed_out;
}

// 2.5: hipMemcpyAsync + hipStreamSynchronize (staging + marker barriers)
static bool test_2_5_async_memcpy_sweep() {
    printf("Test 2.5: hipMemcpyAsync + sync sweep (1-16 MB)...\n");
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    for (size_t mb = 1; mb <= 16; mb *= 2) {
        size_t sz = mb << 20;
        unsigned char *h = (unsigned char *)malloc(sz);
        unsigned char *d;
        int *d_bad;
        memset(h, 0xCD, sz);
        HIP_CHECK(hipMalloc(&d, sz));
        HIP_CHECK(hipMalloc(&d_bad, sizeof(int)));
        HIP_CHECK(hipMemset(d_bad, 0, sizeof(int)));

        HIP_CHECK(hipMemcpyAsync(d, h, sz, hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        int blocks = (sz + 255) / 256;
        verify_pattern<<<blocks, 256, 0, stream>>>(d, 0xCD, sz, d_bad);
        HIP_CHECK(hipStreamSynchronize(stream));

        int bad = -1;
        HIP_CHECK(hipMemcpy(&bad, d_bad, sizeof(int), hipMemcpyDeviceToHost));
        printf("    %zuMB: %s (bad=%d)\n", mb, bad == 0 ? "PASS" : "FAIL", bad);
        hipFree(d_bad); hipFree(d); free(h);
        if (bad != 0) return false;
        if (timed_out) return false;
    }
    hipStreamDestroy(stream);
    return true;
}

// 2.6: Rapid hipMalloc/hipFree cycle (queue drain barriers)
static bool test_2_6_rapid_alloc_free() {
    printf("Test 2.6: Rapid alloc/free cycle (200 iters, 1MB each)...\n");
    const size_t SZ = 1 << 20;
    unsigned char *h = (unsigned char *)malloc(SZ);

    for (int i = 0; i < 200; i++) {
        unsigned char val = (unsigned char)(i & 0xFF);
        memset(h, val, SZ);
        unsigned char *d;
        int *d_bad;
        HIP_CHECK(hipMalloc(&d, SZ));
        HIP_CHECK(hipMalloc(&d_bad, sizeof(int)));
        HIP_CHECK(hipMemset(d_bad, 0, sizeof(int)));
        HIP_CHECK(hipMemcpy(d, h, SZ, hipMemcpyHostToDevice));

        int blocks = (SZ + 255) / 256;
        verify_pattern<<<blocks, 256>>>(d, val, SZ, d_bad);
        HIP_CHECK(hipDeviceSynchronize());

        int bad = -1;
        HIP_CHECK(hipMemcpy(&bad, d_bad, sizeof(int), hipMemcpyDeviceToHost));
        HIP_CHECK(hipFree(d_bad));
        HIP_CHECK(hipFree(d));

        if (bad != 0) {
            printf("  FAIL iter %d: bad=%d\n", i, bad);
            free(h);
            return false;
        }
        if (timed_out) { free(h); return false; }
    }
    free(h);
    return true;
}

// 2.7: Back-to-back hipDeviceSynchronize (global barrier path)
static bool test_2_7_device_sync() {
    printf("Test 2.7: Back-to-back hipDeviceSynchronize (500 iters)...\n");
    int *d_buf, h_val;
    HIP_CHECK(hipMalloc(&d_buf, sizeof(int)));

    for (int i = 0; i < 500; i++) {
        set_val<<<1, 1>>>(d_buf, i);
        HIP_CHECK(hipDeviceSynchronize());
        if (timed_out) { hipFree(d_buf); return false; }
    }
    HIP_CHECK(hipMemcpy(&h_val, d_buf, sizeof(int), hipMemcpyDeviceToHost));
    hipFree(d_buf);
    if (h_val != 499) {
        printf("  FAIL: expected 499, got %d\n", h_val);
        return false;
    }
    return true;
}

// ============================================================
// Tier 3: Stress tests
// ============================================================

// 3.1: Multi-stream sustained
static bool test_3_1_multi_stream_sustained() {
    printf("Test 3.1: Multi-stream sustained (4 streams x 200 kernels)...\n");
    const int NS = 4, NK = 200;
    hipStream_t streams[NS];
    int *d_out[NS], h_val[NS];

    for (int s = 0; s < NS; s++) {
        HIP_CHECK(hipStreamCreate(&streams[s]));
        HIP_CHECK(hipMalloc(&d_out[s], sizeof(int)));
    }

    for (int i = 0; i < NK; i++) {
        for (int s = 0; s < NS; s++) {
            set_val<<<1, 1, 0, streams[s]>>>(d_out[s], s * 1000 + i);
        }
        if (timed_out) break;
    }

    bool ok = true;
    for (int s = 0; s < NS; s++) {
        HIP_CHECK(hipStreamSynchronize(streams[s]));
        HIP_CHECK(hipMemcpy(&h_val[s], d_out[s], sizeof(int), hipMemcpyDeviceToHost));
        if (h_val[s] != s * 1000 + NK - 1) {
            printf("  FAIL stream %d: expected %d, got %d\n",
                   s, s*1000+NK-1, h_val[s]);
            ok = false;
        }
    }

    for (int s = 0; s < NS; s++) {
        hipStreamDestroy(streams[s]);
        hipFree(d_out[s]);
    }
    return ok && !timed_out;
}

// 3.2: Alloc/compute/free pipeline (500 iterations)
static bool test_3_2_alloc_compute_free() {
    printf("Test 3.2: Alloc/compute/free pipeline (500 iters)...\n");
    for (int i = 0; i < 500; i++) {
        int *d_in, *d_out, *d_sum;
        const int N = 256;
        int h_in[N], h_sum = 0;

        for (int j = 0; j < N; j++) h_in[j] = i + j;

        HIP_CHECK(hipMalloc(&d_in, N * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_out, N * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_sum, sizeof(int)));
        HIP_CHECK(hipMemset(d_sum, 0, sizeof(int)));
        HIP_CHECK(hipMemcpy(d_in, h_in, N * sizeof(int), hipMemcpyHostToDevice));

        add_one<<<1, N>>>(d_in, d_out, N);
        sum_reduce<<<1, N>>>(d_out, d_sum, N);
        HIP_CHECK(hipDeviceSynchronize());
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
        if (timed_out) return false;
    }
    return true;
}

// 3.3: Mixed sizes sustained (300 ops)
static bool test_3_3_mixed_sizes() {
    printf("Test 3.3: Mixed sizes sustained (300 ops, 4KB-4MB)...\n");
    unsigned int seed = 42;

    for (int op = 0; op < 300; op++) {
        seed = seed * 1103515245 + 12345;
        // Sizes: 4KB, 16KB, 64KB, 256KB, 1MB, 4MB
        size_t sizes[] = {4096, 16384, 65536, 262144, 1<<20, 4<<20};
        size_t sz = sizes[seed % 6];
        seed = seed * 1103515245 + 12345;
        unsigned char val = (unsigned char)(seed & 0xFF);

        unsigned char *h = (unsigned char *)malloc(sz);
        unsigned char *d;
        int *d_bad;
        memset(h, val, sz);

        HIP_CHECK(hipMalloc(&d, sz));
        HIP_CHECK(hipMalloc(&d_bad, sizeof(int)));
        HIP_CHECK(hipMemset(d_bad, 0, sizeof(int)));

        // H2D
        HIP_CHECK(hipMemcpy(d, h, sz, hipMemcpyHostToDevice));

        // GPU verify
        int blocks = (sz + 255) / 256;
        verify_pattern<<<blocks, 256>>>(d, val, sz, d_bad);
        HIP_CHECK(hipDeviceSynchronize());

        int bad = -1;
        HIP_CHECK(hipMemcpy(&bad, d_bad, sizeof(int), hipMemcpyDeviceToHost));

        // D2H verify
        unsigned char *h2 = (unsigned char *)malloc(sz);
        HIP_CHECK(hipMemcpy(h2, d, sz, hipMemcpyDeviceToHost));
        int d2h_bad = 0;
        for (size_t j = 0; j < sz; j++) {
            if (h2[j] != val) { d2h_bad++; break; }
        }

        hipFree(d_bad); hipFree(d); free(h); free(h2);

        if (bad != 0 || d2h_bad != 0) {
            printf("  FAIL op %d: sz=%zu val=0x%02x gpu_bad=%d d2h_bad=%d\n",
                   op, sz, val, bad, d2h_bad);
            return false;
        }
        if (timed_out) return false;
    }
    return true;
}

// 3.4: Queue pressure (4096 tiny kernels, no intermediate sync)
static bool test_3_4_queue_pressure() {
    printf("Test 3.4: Queue pressure (4096 kernels, then sync)...\n");
    int *d_buf, h_val;
    HIP_CHECK(hipMalloc(&d_buf, sizeof(int)));
    HIP_CHECK(hipMemset(d_buf, 0, sizeof(int)));

    for (int i = 0; i < 4096; i++) {
        set_val<<<1, 1>>>(d_buf, i);
        if (timed_out) { hipFree(d_buf); return false; }
    }
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(&h_val, d_buf, sizeof(int), hipMemcpyDeviceToHost));
    hipFree(d_buf);
    if (h_val != 4095) {
        printf("  FAIL: expected 4095, got %d\n", h_val);
        return false;
    }
    return true;
}

// ============================================================
// Tier 4: Integration tests
// ============================================================

// 4.1: Model load pattern (sequential alloc + H2D + verify)
static bool test_4_1_model_load() {
    printf("Test 4.1: Model load pattern (50 x 1MB tensor loads)...\n");
    const int NTENSORS = 50;
    const size_t TSIZ = 1 << 20;
    unsigned char *h = (unsigned char *)malloc(TSIZ);
    void *tensors[NTENSORS];

    for (int t = 0; t < NTENSORS; t++) {
        unsigned char val = (unsigned char)(t + 0x10);
        memset(h, val, TSIZ);

        unsigned char *d;
        HIP_CHECK(hipMalloc(&d, TSIZ));
        tensors[t] = d;
        HIP_CHECK(hipMemcpy(d, h, TSIZ, hipMemcpyHostToDevice));

        // Verify on GPU
        int *d_bad;
        HIP_CHECK(hipMalloc(&d_bad, sizeof(int)));
        HIP_CHECK(hipMemset(d_bad, 0, sizeof(int)));
        int blocks = (TSIZ + 255) / 256;
        verify_pattern<<<blocks, 256>>>(d, val, TSIZ, d_bad);
        HIP_CHECK(hipDeviceSynchronize());
        int bad = -1;
        HIP_CHECK(hipMemcpy(&bad, d_bad, sizeof(int), hipMemcpyDeviceToHost));
        hipFree(d_bad);

        if (bad != 0) {
            printf("  FAIL tensor %d: bad=%d\n", t, bad);
            for (int j = 0; j <= t; j++) hipFree(tensors[j]);
            free(h);
            return false;
        }
        if (timed_out) {
            for (int j = 0; j <= t; j++) hipFree(tensors[j]);
            free(h);
            return false;
        }
    }
    // Free all
    for (int t = 0; t < NTENSORS; t++) hipFree(tensors[t]);
    free(h);
    return true;
}

// 4.2: Inference pattern (kernel chain with shared buffer)
static bool test_4_2_inference_chain() {
    printf("Test 4.2: Inference pattern (100 x 10-kernel chains)...\n");
    const int N = 256;
    int *d_a, *d_b;
    HIP_CHECK(hipMalloc(&d_a, N * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_b, N * sizeof(int)));

    for (int iter = 0; iter < 100; iter++) {
        // Initialize d_a with iter value
        int h_init[N];
        for (int j = 0; j < N; j++) h_init[j] = iter;
        HIP_CHECK(hipMemcpy(d_a, h_init, N * sizeof(int), hipMemcpyHostToDevice));

        // 10-kernel chain: a->b->a->b...  each adds 1
        for (int k = 0; k < 10; k++) {
            if (k % 2 == 0)
                add_one<<<1, N>>>(d_a, d_b, N);
            else
                add_one<<<1, N>>>(d_b, d_a, N);
        }
        HIP_CHECK(hipDeviceSynchronize());

        // Result should be in d_a (10 is even, last write was to d_b? No:
        // k=0: a->b, k=1: b->a, k=2: a->b, ... k=9: b->a
        // So result is in d_a
        int h_out[N];
        HIP_CHECK(hipMemcpy(h_out, d_a, N * sizeof(int), hipMemcpyDeviceToHost));
        // Each element: iter + 10
        bool ok = true;
        for (int j = 0; j < N; j++) {
            if (h_out[j] != iter + 10) {
                printf("  FAIL iter %d elem %d: expected %d, got %d\n",
                       iter, j, iter+10, h_out[j]);
                ok = false;
                break;
            }
        }
        if (!ok) { hipFree(d_b); hipFree(d_a); return false; }
        if (timed_out) { hipFree(d_b); hipFree(d_a); return false; }
    }
    hipFree(d_b); hipFree(d_a);
    return true;
}

// ============================================================
// Tier 5: Cleanup tests
// ============================================================

// 5.1: Clean process exit (tested by running this binary and exiting)
// 5.2: Exit with pending work (separate test, needs fork)

static bool test_5_1_clean_exit() {
    printf("Test 5.1: Clean exit (kernel + sync + verify)...\n");
    int *d, h = 0;
    HIP_CHECK(hipMalloc(&d, sizeof(int)));
    set_val<<<1, 1>>>(d, 12345);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(&h, d, sizeof(int), hipMemcpyDeviceToHost));
    hipFree(d);
    if (h != 12345) {
        printf("  FAIL: got %d\n", h);
        return false;
    }
    return true;
}

static bool test_5_2_exit_pending_work() {
    printf("Test 5.2: Exit with pending work (fork + 1000 async kernels)...\n");
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
        // Child: launch lots of async work, then _exit immediately
        int *d;
        if (hipMalloc(&d, sizeof(int)) != hipSuccess) _exit(99);
        for (int i = 0; i < 1000; i++)
            set_val<<<1, 1>>>(d, i);
        _exit(0);
    }
    // Parent: wait for child with timeout
    int status = 0;
    auto t0 = std::chrono::steady_clock::now();
    while (true) {
        pid_t r = waitpid(pid, &status, WNOHANG);
        if (r == pid) break;
        auto elapsed = std::chrono::steady_clock::now() - t0;
        if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() > 10) {
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            printf("  FAIL: child didn't exit within 10s\n");
            return false;
        }
        usleep(50000);
    }
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) return true;
    printf("  FAIL: child exit status %d\n", WEXITSTATUS(status));
    return false;
}

// ============================================================
// Main
// ============================================================

int main() {
    signal(SIGALRM, alarm_handler);
    alarm(900);  // 15 minute global timeout

    hipDeviceProp_t prop;
    hipError_t e = hipGetDeviceProperties(&prop, 0);
    if (e != hipSuccess) {
        fprintf(stderr, "hipGetDeviceProperties failed: %d\n", e);
        return 1;
    }
    printf("Device: %s (%s)\n\n", prop.name, prop.gcnArchName);

    struct { const char *name; bool (*fn)(); } tests[] = {
        // Tier 2: Barrier-specific
        {"2.1 Kernel->memcpy->kernel chain",    test_2_1_kernel_memcpy_chain},
        {"2.2 Event record+sync",               test_2_2_event_sync},
        {"2.3 StreamSync under load",            test_2_3_stream_sync_load},
        {"2.4 Cross-stream event wait",          test_2_4_cross_stream_event},
        {"2.5 Async memcpy sweep",               test_2_5_async_memcpy_sweep},
        {"2.6 Rapid alloc/free cycle",           test_2_6_rapid_alloc_free},
        {"2.7 Back-to-back DeviceSync",          test_2_7_device_sync},
        // Tier 3: Stress
        {"3.1 Multi-stream sustained",           test_3_1_multi_stream_sustained},
        {"3.2 Alloc/compute/free pipeline",      test_3_2_alloc_compute_free},
        {"3.3 Mixed sizes sustained",            test_3_3_mixed_sizes},
        {"3.4 Queue pressure 4096",              test_3_4_queue_pressure},
        // Tier 4: Integration
        {"4.1 Model load pattern",               test_4_1_model_load},
        {"4.2 Inference kernel chain",           test_4_2_inference_chain},
        // Tier 5: Cleanup
        {"5.1 Clean exit",                       test_5_1_clean_exit},
        {"5.2 Exit with pending work",           test_5_2_exit_pending_work},
    };

    int pass = 0, fail = 0;
    for (auto &t : tests) {
        fflush(stdout);
        auto t0 = std::chrono::steady_clock::now();
        bool ok = t.fn();
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        if (timed_out) {
            printf("  GLOBAL TIMEOUT\n");
            ok = false;
        }
        printf("  %s: %s (%.1fs)\n\n", t.name, ok ? "PASS" : "FAIL", secs);
        fflush(stdout);
        if (ok) pass++; else fail++;
    }

    printf("=== Results: %d/%d PASS ===\n", pass, pass + fail);
    fflush(stdout);
    return fail > 0 ? 1 : 0;
}
