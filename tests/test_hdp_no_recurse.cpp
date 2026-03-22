// test_hdp_no_recurse.cpp — Verify HDP flush has no recursion/hang
//
// Rapid-fire 10000 dispatches to stress the recursion guard in
// AqlQueue::StoreRelaxed. If the guard fails, HdpFlushWait calls
// StoreRelaxed which calls HdpFlushWait → stack overflow / hang.
//
// Build: hipcc --offload-arch=gfx803 -o test_hdp_no_recurse test_hdp_no_recurse.cpp
// Run:   timeout 120 ./test_hdp_no_recurse

#include <hip/hip_runtime.h>
#include <cstdio>
#include <csignal>
#include <chrono>

static volatile sig_atomic_t timed_out = 0;
static void alarm_handler(int) { timed_out = 1; }

__global__ void nop_kernel(int *out) {
    if (threadIdx.x == 0) *out = 1;
}

int main() {
    signal(SIGALRM, alarm_handler);
    alarm(120);

    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
        fprintf(stderr, "hipGetDeviceProperties failed\n");
        return 1;
    }
    printf("Device: %s (%s)\n", prop.name, prop.gcnArchName);
    printf("Test: 10000 rapid dispatches (recursion guard check)...\n");
    fflush(stdout);

    int *d_out;
    hipMalloc(&d_out, sizeof(int));

    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < 10000; i++) {
        nop_kernel<<<1, 1>>>(d_out);
        // Sync every 100 to avoid queue overflow, but keep pressure high
        if ((i + 1) % 100 == 0) {
            hipDeviceSynchronize();
            if (timed_out) {
                printf("TIMEOUT at iter %d — possible recursion hang\n", i);
                hipFree(d_out);
                return 1;
            }
        }
    }
    hipDeviceSynchronize();

    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    hipFree(d_out);

    if (timed_out) {
        printf("TIMEOUT — possible recursion hang\n");
        return 1;
    }

    printf("Completed 10000 dispatches in %.1fs (%.0f dispatch/s)\n",
           secs, 10000.0 / secs);
    printf("PASS: No recursion hang detected\n");
    return 0;
}
