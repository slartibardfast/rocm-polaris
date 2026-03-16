// hip_torture_test.cpp — Torture test for sustained GPU dispatch
// Mimics llama.cpp dispatch patterns to verify SLOT_BASED_WPTR=0 fix.
// Build: hipcc -o hip_torture_test hip_torture_test.cpp
// Run: timeout 120 ./hip_torture_test

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>

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
    printf("Test 2: Multi-stream dispatch (8 streams x 100 kernels)...\n");
    const int NSTREAMS = 8, NKERNELS = 100;
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

int main() {
    signal(SIGALRM, alarm_handler);
    alarm(90);  // 90 second global timeout

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("Device: %s (gfx%d)\n\n", prop.name, prop.gcnArchName[3] ? atoi(prop.gcnArchName + 3) : 0);

    struct { const char *name; bool (*fn)(); } tests[] = {
        {"Rapid serial dispatch",  test1_rapid_serial},
        {"Multi-stream dispatch",  test2_multi_stream},
        {"Interleaved compute+memcpy", test3_interleaved},
        {"Rapid fire no sync",     test4_rapid_fire},
        {"Multi-stream events",    test5_events},
        {"Stress duration",        test6_stress_duration},
    };

    int pass = 0, fail = 0;
    for (auto &t : tests) {
        bool ok = t.fn();
        if (timed_out) {
            printf("  TIMEOUT\n");
            ok = false;
        }
        printf("  %s: %s\n\n", t.name, ok ? "PASS" : "FAIL");
        if (ok) pass++; else fail++;
    }

    printf("Results: %d/%d PASS\n", pass, pass + fail);
    return fail > 0 ? 1 : 0;
}
