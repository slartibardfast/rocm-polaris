// test_h2d_degradation.cpp — Binary search for H2D degradation threshold
//
// Runs N burn operations (kernel+sync+alloc/free+event+memcpy mix),
// then tests H2D correctness.  Finds the exact point where hipMemcpy
// H2D silently fails after sustained use in a single process.
//
// Build: hipcc --offload-arch=gfx803 -o test_h2d_degradation test_h2d_degradation.cpp
// Run:   timeout 300 ./test_h2d_degradation

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

#define HIP_CHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
                e, hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

__global__ void set_val(int *out, int val) {
    out[threadIdx.x] = val;
}

__global__ void verify_pattern(const unsigned char *d, unsigned char expected,
                               int n, int *bad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && d[i] != expected) atomicAdd(bad, 1);
}

static int g_total_ops = 0;

// Burn N operations: mix of kernel dispatch, memcpy, alloc/free, event
static void burn(int n) {
    int *d_buf;
    HIP_CHECK(hipMalloc(&d_buf, 256 * sizeof(int)));

    hipEvent_t ev;
    HIP_CHECK(hipEventCreate(&ev));

    for (int i = 0; i < n; i++) {
        int kind = i % 10;
        if (kind < 6) {
            set_val<<<1, 256>>>(d_buf, i);
            if (kind == 0)
                HIP_CHECK(hipDeviceSynchronize());
        } else if (kind < 8) {
            int *tmp;
            HIP_CHECK(hipMalloc(&tmp, 4096));
            set_val<<<1, 1>>>(tmp, i);
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipFree(tmp));
        } else if (kind == 8) {
            set_val<<<1, 1>>>(d_buf, i);
            HIP_CHECK(hipEventRecord(ev, 0));
            HIP_CHECK(hipEventSynchronize(ev));
        } else {
            int h = i;
            HIP_CHECK(hipMemcpy(d_buf, &h, sizeof(int), hipMemcpyHostToDevice));
        }
    }

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventDestroy(ev));
    HIP_CHECK(hipFree(d_buf));
    g_total_ops += n;
}

// Test H2D correctness with GPU and CPU verification
static bool test_h2d(size_t sz, unsigned char val) {
    unsigned char *h = (unsigned char *)malloc(sz);
    memset(h, val, sz);

    unsigned char *d;
    int *d_bad;
    HIP_CHECK(hipMalloc(&d, sz));
    HIP_CHECK(hipMalloc(&d_bad, sizeof(int)));
    HIP_CHECK(hipMemset(d_bad, 0, sizeof(int)));

    HIP_CHECK(hipMemcpy(d, h, sz, hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    int blocks = (sz + 255) / 256;
    verify_pattern<<<blocks, 256>>>(d, val, sz, d_bad);
    HIP_CHECK(hipDeviceSynchronize());

    int bad = -1;
    HIP_CHECK(hipMemcpy(&bad, d_bad, sizeof(int), hipMemcpyDeviceToHost));

    unsigned char *h2 = (unsigned char *)malloc(sz);
    HIP_CHECK(hipMemcpy(h2, d, sz, hipMemcpyDeviceToHost));

    bool ok = (bad == 0);
    if (!ok) {
        printf("    FAIL: sz=%zu val=0x%02x gpu_bad=%d/%zu "
               "d2h[0..3]=0x%02x%02x%02x%02x (total_ops=%d)\n",
               sz, val, bad, sz,
               h2[0], h2[1], h2[2], h2[3], g_total_ops);
    }

    hipFree(d_bad); hipFree(d); free(h2); free(h);
    return ok;
}

static bool test_h2d_battery() {
    size_t sizes[] = {1024, 4096, 65536, 1 << 20};
    unsigned char vals[] = {0xAB, 0xCD, 0x42, 0x77};
    for (int i = 0; i < 4; i++) {
        if (!test_h2d(sizes[i], vals[i]))
            return false;
    }
    return true;
}

int main() {
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("Device: %s (%s)\n\n", prop.name, prop.gcnArchName);

    // Cumulative burn: each round adds more ops, testing H2D after each
    int increments[] = {
        500, 500, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
        2000, 2000, 2000, 5000, 5000, 5000, 10000, 10000
    };
    int n = sizeof(increments) / sizeof(increments[0]);

    printf("Cumulative burn → H2D test (threshold search):\n\n");

    for (int t = 0; t < n; t++) {
        auto t0 = std::chrono::steady_clock::now();
        burn(increments[t]);
        auto t1 = std::chrono::steady_clock::now();
        double burn_s = std::chrono::duration<double>(t1 - t0).count();

        bool ok = test_h2d_battery();
        auto t2 = std::chrono::steady_clock::now();
        double test_s = std::chrono::duration<double>(t2 - t1).count();

        printf("  cumulative=%5d (+%d): %s  (burn=%.1fs test=%.1fs)\n",
               g_total_ops, increments[t],
               ok ? "PASS" : "FAIL", burn_s, test_s);
        fflush(stdout);

        if (!ok) {
            printf("\n  Degradation at ~%d cumulative ops\n", g_total_ops);
            return 1;
        }
    }

    printf("\nALL PASS through %d cumulative ops\n", g_total_ops);
    return 0;
}
