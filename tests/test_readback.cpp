// test_readback.cpp — Test DRAM readback as IOH write drain
//
// Build: hipcc --offload-arch=gfx803 -o test_readback test_readback.cpp
// Run:   ./test_readback <variant> [iters]

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <x86intrin.h>
#include <chrono>

__global__ void record_val(int *out, int val) {
    if (threadIdx.x == 0) *out = val;
}

// We need access to the kernarg buffer to readback from it.
// Since we can't easily access CLR internals from a test,
// we'll allocate our OWN UC system memory and use it as
// a proxy for what the fix would do inside CLR.
//
// The test writes to a UC buffer, optionally reads it back,
// then launches a kernel that reads from GPU memory.
// The kernarg is written by CLR internally — we can't readback
// CLR's kernarg. But we CAN test whether a readback from ANY
// UC DRAM address forces the IOH to drain.

// Actually, the simplest test: just add a volatile read of
// a UC variable after hipMemset returns and before kernel launch.
// hipMemset writes to d_result (VRAM), so reading d_result
// wouldn't test DRAM. We need to read from system memory.
//
// Approach: allocate host UC memory, write to it, readback,
// then launch kernel. If readback drains IOH for ALL pending
// DRAM writes (not just the one we read), it fixes corruption.

static volatile char *g_uc_anchor = nullptr;

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <variant> [iters]\n", argv[0]); return 1; }
    const char *v = argv[1];
    int iters = argc > 2 ? atoi(argv[2]) : 10000;

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);

    // Allocate UC host memory as readback anchor
    void *anchor;
    hipHostMalloc(&anchor, 64, hipHostMallocNonCoherent);
    g_uc_anchor = (volatile char*)anchor;
    *g_uc_anchor = 0;

    int *d_result;
    hipMalloc(&d_result, sizeof(int));

    auto t0 = std::chrono::steady_clock::now();
    int fails = 0;
    for (int i = 0; i < iters; i++) {
        hipMemset(d_result, 0xFF, sizeof(int));

        if (strcmp(v, "readback") == 0) {
            // Write then read UC system memory — forces IOH DRAM drain
            *g_uc_anchor = (char)i;
            _mm_sfence();
            volatile char x = *g_uc_anchor;
            (void)x;
        }
        else if (strcmp(v, "readback_mfence") == 0) {
            *g_uc_anchor = (char)i;
            _mm_mfence();
            volatile char x = *g_uc_anchor;
            (void)x;
        }
        else if (strcmp(v, "just_read") == 0) {
            // Just read UC memory — no write first
            volatile char x = *g_uc_anchor;
            (void)x;
        }
        else if (strcmp(v, "write_only") == 0) {
            // Just write UC memory — no readback
            *g_uc_anchor = (char)i;
            _mm_sfence();
        }
        // else: baseline

        record_val<<<1, 1>>>(d_result, i * 7 + 13);
        hipDeviceSynchronize();
        int h;
        hipMemcpy(&h, d_result, sizeof(int), hipMemcpyDeviceToHost);
        if (h != i * 7 + 13) fails++;
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    hipFree(d_result);
    hipHostFree(anchor);
    printf("%-25s %3d / %d  (%.0f dispatch/s)\n", v, fails, iters, iters/(ms/1000.0));
    return fails > 0 ? 1 : 0;
}
