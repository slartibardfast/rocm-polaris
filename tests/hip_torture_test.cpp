// hip_torture_test.cpp — Torture test for sustained GPU dispatch + completion
//
// Tests 1-6:  dispatch correctness (SLOT_BASED_WPTR=0 fix)
// Tests 7-12: completion visibility (UC shared memory fix)
//
// Build: hipcc --offload-arch=gfx803 -o hip_torture_test hip_torture_test.cpp
// Run:   HSA_OVERRIDE_GFX_VERSION=8.0.3 ROC_CPU_WAIT_FOR_SIGNAL=1 \
//          timeout 600 ./hip_torture_test

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
        exit(1); \
    } \
} while (0)

static volatile sig_atomic_t timed_out = 0;
static void alarm_handler(int) { timed_out = 1; }

// Trivial kernel: write a known value
__global__ void set_val(int *out, int val) {
    out[threadIdx.x] = val;
}

// Kernel that reads input and writes output (for compute+memcpy test)
__global__ void add_one(const int *in, int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] + 1;
}

// Kernel that fills a buffer with a deterministic pattern
__global__ void fill_pattern(int *out, int n, int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = seed * 7 + i;
}

// ============================================================
// Tests 1-6: Dispatch correctness (Phase 7a — SLOT_BASED_WPTR)
// ============================================================

static bool test1_rapid_serial() {
    printf("Test 1: Rapid serial dispatch (1000 kernels, 1 stream)...\n");
    int *d_out;
    HIP_CHECK(hipMalloc(&d_out, sizeof(int)));
    for (int i = 0; i < 1000; i++) {
        set_val<<<1, 1, 0, 0>>>(d_out, i);
        if (timed_out) { hipFree(d_out); return false; }
    }
    HIP_CHECK(hipDeviceSynchronize());
    int result;
    HIP_CHECK(hipMemcpy(&result, d_out, sizeof(int), hipMemcpyDeviceToHost));
    hipFree(d_out);
    if (result != 999) {
        printf("  FAIL: expected 999, got %d\n", result);
        return false;
    }
    return true;
}

static bool test2_multi_stream() {
    printf("Test 2: Multi-stream dispatch (4 streams x 100 kernels)...\n");
    const int NSTREAMS = 4, NKERNELS = 100;
    hipStream_t streams[NSTREAMS];
    int *d_out[NSTREAMS];
    for (int s = 0; s < NSTREAMS; s++) {
        HIP_CHECK(hipStreamCreate(&streams[s]));
        HIP_CHECK(hipMalloc(&d_out[s], sizeof(int)));
    }
    for (int i = 0; i < NKERNELS; i++) {
        for (int s = 0; s < NSTREAMS; s++) {
            set_val<<<1, 1, 0, streams[s]>>>(d_out[s], i);
            if (timed_out) goto cleanup2;
        }
    }
    HIP_CHECK(hipDeviceSynchronize());
    for (int s = 0; s < NSTREAMS; s++) {
        int result;
        HIP_CHECK(hipMemcpy(&result, d_out[s], sizeof(int), hipMemcpyDeviceToHost));
        if (result != NKERNELS - 1) {
            printf("  FAIL: stream %d expected %d, got %d\n", s, NKERNELS - 1, result);
            goto cleanup2;
        }
    }
    for (int s = 0; s < NSTREAMS; s++) {
        hipFree(d_out[s]);
        hipStreamDestroy(streams[s]);
    }
    return true;
cleanup2:
    for (int s = 0; s < NSTREAMS; s++) {
        hipFree(d_out[s]);
        hipStreamDestroy(streams[s]);
    }
    return false;
}

static bool test3_interleaved() {
    printf("Test 3: Interleaved compute+memcpy (500 iterations)...\n");
    const int N = 256;
    int h_in[N], h_out[N];
    int *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, N * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_out, N * sizeof(int)));
    for (int iter = 0; iter < 500; iter++) {
        for (int i = 0; i < N; i++) h_in[i] = iter;
        HIP_CHECK(hipMemcpy(d_in, h_in, N * sizeof(int), hipMemcpyHostToDevice));
        add_one<<<1, N>>>(d_in, d_out, N);
        HIP_CHECK(hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost));
        if (h_out[0] != iter + 1 || h_out[N - 1] != iter + 1) {
            printf("  FAIL: iter %d expected %d, got [0]=%d [%d]=%d\n",
                   iter, iter + 1, h_out[0], N - 1, h_out[N - 1]);
            hipFree(d_in); hipFree(d_out);
            return false;
        }
        if (timed_out) { hipFree(d_in); hipFree(d_out); return false; }
    }
    hipFree(d_in); hipFree(d_out);
    return true;
}

static bool test4_rapid_fire() {
    printf("Test 4: Rapid fire no sync (500 kernels, then sync)...\n");
    int *d_out;
    HIP_CHECK(hipMalloc(&d_out, sizeof(int)));
    for (int i = 0; i < 500; i++) {
        set_val<<<1, 1>>>(d_out, i);
        if (timed_out) { hipFree(d_out); return false; }
    }
    HIP_CHECK(hipDeviceSynchronize());
    int result;
    HIP_CHECK(hipMemcpy(&result, d_out, sizeof(int), hipMemcpyDeviceToHost));
    hipFree(d_out);
    if (result != 499) {
        printf("  FAIL: expected 499, got %d\n", result);
        return false;
    }
    return true;
}

static bool test5_events() {
    printf("Test 5: Multi-stream with events (fork/join)...\n");
    const int N = 256;
    hipStream_t s0, s1, s2, s3;
    HIP_CHECK(hipStreamCreate(&s0));
    HIP_CHECK(hipStreamCreate(&s1));
    HIP_CHECK(hipStreamCreate(&s2));
    HIP_CHECK(hipStreamCreate(&s3));
    hipEvent_t produced;
    HIP_CHECK(hipEventCreate(&produced));

    int *d_in, *d_out1, *d_out2, *d_out3;
    HIP_CHECK(hipMalloc(&d_in, N * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_out1, N * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_out2, N * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_out3, N * sizeof(int)));

    int h_in[N];
    for (int i = 0; i < N; i++) h_in[i] = i;
    HIP_CHECK(hipMemcpyAsync(d_in, h_in, N * sizeof(int), hipMemcpyHostToDevice, s0));

    // Producer: add_one on s0
    add_one<<<1, N, 0, s0>>>(d_in, d_in, N);
    HIP_CHECK(hipEventRecord(produced, s0));

    // Consumers: wait for producer, then each does add_one
    HIP_CHECK(hipStreamWaitEvent(s1, produced, 0));
    HIP_CHECK(hipStreamWaitEvent(s2, produced, 0));
    HIP_CHECK(hipStreamWaitEvent(s3, produced, 0));
    add_one<<<1, N, 0, s1>>>(d_in, d_out1, N);
    add_one<<<1, N, 0, s2>>>(d_in, d_out2, N);
    add_one<<<1, N, 0, s3>>>(d_in, d_out3, N);

    HIP_CHECK(hipDeviceSynchronize());

    int h_out[N];
    bool pass = true;
    HIP_CHECK(hipMemcpy(h_out, d_out1, N * sizeof(int), hipMemcpyDeviceToHost));
    if (h_out[0] != 1 || h_out[N - 1] != N) { printf("  FAIL: stream 1\n"); pass = false; }
    HIP_CHECK(hipMemcpy(h_out, d_out2, N * sizeof(int), hipMemcpyDeviceToHost));
    if (h_out[0] != 1 || h_out[N - 1] != N) { printf("  FAIL: stream 2\n"); pass = false; }
    HIP_CHECK(hipMemcpy(h_out, d_out3, N * sizeof(int), hipMemcpyDeviceToHost));
    if (h_out[0] != 1 || h_out[N - 1] != N) { printf("  FAIL: stream 3\n"); pass = false; }

    hipFree(d_in); hipFree(d_out1); hipFree(d_out2); hipFree(d_out3);
    hipEventDestroy(produced);
    hipStreamDestroy(s0); hipStreamDestroy(s1);
    hipStreamDestroy(s2); hipStreamDestroy(s3);
    return pass;
}

static bool test6_stress_duration() {
    printf("Test 6: Stress duration (100 batches x 100 kernels)...\n");
    int *d_out;
    HIP_CHECK(hipMalloc(&d_out, sizeof(int)));
    for (int batch = 0; batch < 100; batch++) {
        for (int i = 0; i < 100; i++) {
            set_val<<<1, 1>>>(d_out, batch * 100 + i);
            if (timed_out) { hipFree(d_out); return false; }
        }
        HIP_CHECK(hipDeviceSynchronize());
    }
    int result;
    HIP_CHECK(hipMemcpy(&result, d_out, sizeof(int), hipMemcpyDeviceToHost));
    hipFree(d_out);
    if (result != 9999) {
        printf("  FAIL: expected 9999, got %d\n", result);
        return false;
    }
    return true;
}

// ============================================================
// Tests 7-12: Completion visibility (Phase 7b — UC memory)
// ============================================================

// Test 7: Signal storm — 1000 individual per-stream syncs.
// Each hipStreamSynchronize reads a completion signal from the signal pool.
// With WB-cached signal memory and no clflush in the polling loop,
// the CPU may see a stale signal value and spin forever.
static bool test7_signal_storm() {
    printf("Test 7: Signal storm (1000 per-stream syncs)...\n");
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    int *d_out;
    HIP_CHECK(hipMalloc(&d_out, sizeof(int)));
    for (int i = 0; i < 1000; i++) {
        set_val<<<1, 1, 0, stream>>>(d_out, i);
        HIP_CHECK(hipStreamSynchronize(stream));
        if (timed_out) { hipFree(d_out); hipStreamDestroy(stream); return false; }
    }
    int result;
    HIP_CHECK(hipMemcpy(&result, d_out, sizeof(int), hipMemcpyDeviceToHost));
    hipFree(d_out);
    hipStreamDestroy(stream);
    if (result != 999) {
        printf("  FAIL: expected 999, got %d\n", result);
        return false;
    }
    return true;
}

// Test 8: D2H integrity sweep — multiple transfer sizes.
// Exercises both staging path (<16KB) and pinned path (>=16KB).
// Each size is tested 50 times to catch intermittent cache stale reads.
// GPU fills buffer with deterministic pattern, D2H, verify every sample.
static bool test8_d2h_sweep() {
    printf("Test 8: D2H integrity sweep (6 sizes x 50 reps)...\n");
    const int sizes[] = {64, 4096, 16384, 65536, 262144, 1048576};
    const int NSIZES = sizeof(sizes) / sizeof(sizes[0]);
    const int REPS = 50;

    for (int si = 0; si < NSIZES; si++) {
        int n = sizes[si] / (int)sizeof(int);
        int *d_buf;
        HIP_CHECK(hipMalloc(&d_buf, sizes[si]));
        int *h_buf = (int *)malloc(sizes[si]);

        for (int rep = 0; rep < REPS; rep++) {
            int seed = si * REPS + rep;
            int blocks = (n + 255) / 256;
            fill_pattern<<<blocks, 256>>>(d_buf, n, seed);
            HIP_CHECK(hipMemcpy(h_buf, d_buf, sizes[si], hipMemcpyDeviceToHost));

            // Verify first, last, and 8 distributed spots
            int check_indices[] = {0, n/7, n/5, n/3, n/2, n*2/3, n*4/5, n*6/7, n-1};
            for (int ci = 0; ci < 9; ci++) {
                int idx = check_indices[ci];
                if (idx < 0 || idx >= n) continue;
                int expected = seed * 7 + idx;
                if (h_buf[idx] != expected) {
                    printf("  FAIL: size=%d rep=%d idx=%d expected=%d got=%d\n",
                           sizes[si], rep, idx, expected, h_buf[idx]);
                    free(h_buf); hipFree(d_buf);
                    return false;
                }
            }
            if (timed_out) { free(h_buf); hipFree(d_buf); return false; }
        }
        printf("    %7d bytes: %d/%d OK\n", sizes[si], REPS, REPS);
        free(h_buf);
        hipFree(d_buf);
    }
    return true;
}

// Test 9: Concurrent D2H — 4 streams each doing D2H simultaneously.
// Exercises contention on staging buffers (per-VirtualGPU) and
// signal pool (shared). 100 rounds.
static bool test9_concurrent_d2h() {
    printf("Test 9: Concurrent D2H (4 streams x 100 rounds)...\n");
    const int NSTREAMS = 4, ROUNDS = 100, N = 1024;
    hipStream_t streams[NSTREAMS];
    int *d_buf[NSTREAMS];
    int *h_buf[NSTREAMS];

    for (int s = 0; s < NSTREAMS; s++) {
        HIP_CHECK(hipStreamCreate(&streams[s]));
        HIP_CHECK(hipMalloc(&d_buf[s], N * sizeof(int)));
        h_buf[s] = (int *)malloc(N * sizeof(int));
    }

    for (int round = 0; round < ROUNDS; round++) {
        // All streams fill their buffers concurrently
        for (int s = 0; s < NSTREAMS; s++) {
            int seed = round * NSTREAMS + s;
            fill_pattern<<<(N + 255) / 256, 256, 0, streams[s]>>>(d_buf[s], N, seed);
        }
        // All streams D2H concurrently
        for (int s = 0; s < NSTREAMS; s++) {
            HIP_CHECK(hipMemcpyAsync(h_buf[s], d_buf[s], N * sizeof(int),
                                     hipMemcpyDeviceToHost, streams[s]));
        }
        // Sync all and verify
        HIP_CHECK(hipDeviceSynchronize());
        for (int s = 0; s < NSTREAMS; s++) {
            int seed = round * NSTREAMS + s;
            for (int idx = 0; idx < N; idx += N / 8) {
                int expected = seed * 7 + idx;
                if (h_buf[s][idx] != expected) {
                    printf("  FAIL: round=%d stream=%d idx=%d expected=%d got=%d\n",
                           round, s, idx, expected, h_buf[s][idx]);
                    for (int i = 0; i < NSTREAMS; i++) {
                        free(h_buf[i]); hipFree(d_buf[i]); hipStreamDestroy(streams[i]);
                    }
                    return false;
                }
            }
        }
        if (timed_out) {
            for (int s = 0; s < NSTREAMS; s++) {
                free(h_buf[s]); hipFree(d_buf[s]); hipStreamDestroy(streams[s]);
            }
            return false;
        }
    }

    printf("    %d rounds x %d streams: all correct\n", ROUNDS, NSTREAMS);
    for (int s = 0; s < NSTREAMS; s++) {
        free(h_buf[s]); hipFree(d_buf[s]); hipStreamDestroy(streams[s]);
    }
    return true;
}

// Test 10: hipFree under load — the exact llama.cpp crash pattern.
// hipFree calls SyncAllStreams → awaitCompletion → reads signal.
// With WB-cached signals, SyncAllStreams may spin forever.
static bool test10_free_under_load() {
    printf("Test 10: hipFree under load (100 alloc/use/free cycles)...\n");
    hipStream_t bg_stream;
    HIP_CHECK(hipStreamCreate(&bg_stream));
    int *d_bg;
    HIP_CHECK(hipMalloc(&d_bg, sizeof(int)));

    for (int i = 0; i < 100; i++) {
        // Background: keep bg_stream busy
        set_val<<<1, 1, 0, bg_stream>>>(d_bg, i);

        // Foreground: alloc, use on default stream, free
        int *d_tmp;
        HIP_CHECK(hipMalloc(&d_tmp, 4096));
        set_val<<<1, 1>>>(d_tmp, i);
        HIP_CHECK(hipDeviceSynchronize());
        hipFree(d_tmp);  // This calls SyncAllStreams internally

        if (timed_out) { hipFree(d_bg); hipStreamDestroy(bg_stream); return false; }
    }

    HIP_CHECK(hipStreamSynchronize(bg_stream));
    int result;
    HIP_CHECK(hipMemcpy(&result, d_bg, sizeof(int), hipMemcpyDeviceToHost));
    hipFree(d_bg);
    hipStreamDestroy(bg_stream);
    if (result != 99) {
        printf("  FAIL: expected 99, got %d\n", result);
        return false;
    }
    return true;
}

// Test 11: Ring buffer wrap — 8192 packets to force RPTR wrap.
// Default AQL queue is 4096 packets. Each sync reads RPTR via bounce
// buffer. Verifies RPTR monotonicity across wrap and no lost completions.
static bool test11_ring_wrap() {
    printf("Test 11: Ring buffer wrap (16 batches x 512 = 8192 kernels)...\n");
    const int BATCHES = 16, PER_BATCH = 512;
    int *d_out;
    HIP_CHECK(hipMalloc(&d_out, sizeof(int)));

    for (int b = 0; b < BATCHES; b++) {
        for (int i = 0; i < PER_BATCH; i++) {
            set_val<<<1, 1>>>(d_out, b * PER_BATCH + i);
            if (timed_out) { hipFree(d_out); return false; }
        }
        HIP_CHECK(hipDeviceSynchronize());

        int result;
        HIP_CHECK(hipMemcpy(&result, d_out, sizeof(int), hipMemcpyDeviceToHost));
        int expected = b * PER_BATCH + PER_BATCH - 1;
        if (result != expected) {
            printf("  FAIL: batch %d expected %d, got %d\n", b, expected, result);
            hipFree(d_out);
            return false;
        }
    }

    printf("    %d kernels across wrap: all correct\n", BATCHES * PER_BATCH);
    hipFree(d_out);
    return true;
}

// Test 12: Completion latency — time individual launch→sync cycles.
// With UC memory: RPTR/signal read is ~200ns DRAM access, consistent.
// With WB + stale cache: may see multi-ms delays from extra polling.
// Fail if >5% of syncs exceed 50ms.
static bool test12_completion_latency() {
    printf("Test 12: Completion latency (500 launch+sync, <50ms each)...\n");
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    int *d_out;
    HIP_CHECK(hipMalloc(&d_out, sizeof(int)));

    // Warmup
    set_val<<<1, 1, 0, stream>>>(d_out, 0);
    HIP_CHECK(hipStreamSynchronize(stream));

    int outliers = 0;
    double max_ms = 0;
    double total_ms = 0;
    const int N = 500;
    const double THRESHOLD_MS = 50.0;

    for (int i = 0; i < N; i++) {
        auto t0 = std::chrono::steady_clock::now();
        set_val<<<1, 1, 0, stream>>>(d_out, i);
        HIP_CHECK(hipStreamSynchronize(stream));
        auto t1 = std::chrono::steady_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (ms > max_ms) max_ms = ms;
        if (ms > THRESHOLD_MS) {
            if (outliers < 5)
                printf("    outlier: iter %d took %.1f ms\n", i, ms);
            outliers++;
        }
        if (timed_out) { hipFree(d_out); hipStreamDestroy(stream); return false; }
    }

    printf("    avg=%.2f ms  max=%.2f ms  outliers(>%.0fms)=%d/%d\n",
           total_ms / N, max_ms, THRESHOLD_MS, outliers, N);

    hipFree(d_out);
    hipStreamDestroy(stream);

    if (outliers > N / 20) {  // >5% outliers = fail
        printf("  FAIL: too many outliers (%d/%d > 5%%)\n", outliers, N);
        return false;
    }
    return true;
}

// ============================================================

int main() {
    signal(SIGALRM, alarm_handler);
    alarm(600);  // 10 minute global timeout

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("Device: %s (%s)\n\n", prop.name, prop.gcnArchName);

    struct { const char *name; bool (*fn)(); } tests[] = {
        // Phase 7a: dispatch correctness
        {"Rapid serial dispatch",      test1_rapid_serial},
        {"Multi-stream dispatch",      test2_multi_stream},
        {"Interleaved compute+memcpy", test3_interleaved},
        {"Rapid fire no sync",         test4_rapid_fire},
        {"Multi-stream events",        test5_events},
        {"Stress duration",            test6_stress_duration},
        // Phase 7b: completion visibility
        {"Signal storm",               test7_signal_storm},
        {"D2H integrity sweep",        test8_d2h_sweep},
        {"Concurrent D2H",             test9_concurrent_d2h},
        {"hipFree under load",         test10_free_under_load},
        {"Ring buffer wrap",           test11_ring_wrap},
        {"Completion latency",         test12_completion_latency},
    };

    int pass = 0, fail = 0;
    for (auto &t : tests) {
        fflush(stdout);
        bool ok = t.fn();
        if (timed_out) {
            printf("  TIMEOUT\n");
            ok = false;
        }
        printf("  %s: %s\n\n", t.name, ok ? "PASS" : "FAIL");
        fflush(stdout);
        if (ok) pass++; else fail++;
    }

    printf("Results: %d/%d PASS\n", pass, pass + fail);
    fflush(stdout);
    _exit(fail > 0 ? 1 : 0);  // skip HIP cleanup (hangs on stream destroy)
}
