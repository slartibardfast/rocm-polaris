// test_hip_events.cpp — Does hipEventRecord + hipStreamWaitEvent hang?
// This mimics ggml-hip's per-node synchronization pattern.
//
// Build: hipcc -o test_hip_events test_hip_events.cpp

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void add_one(int *buf, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) buf[i] += 1;
}

int main(int argc, char **argv) {
    int n_ops = argc > 1 ? atoi(argv[1]) : 10;
    int n = 256;

    int *d_buf;
    hipMalloc(&d_buf, n * sizeof(int));
    hipMemset(d_buf, 0, n * sizeof(int));

    hipStream_t stream;
    hipStreamCreate(&stream);

    hipEvent_t event;
    hipEventCreateWithFlags(&event, hipEventDisableTiming);

    printf("hipEvent test: %d ops with event record/wait...\n", n_ops);

    for (int i = 0; i < n_ops; i++) {
        add_one<<<1, n, 0, stream>>>(d_buf, n);

        // Record event after kernel
        hipError_t err = hipEventRecord(event, stream);
        if (err != hipSuccess) {
            printf("hipEventRecord FAILED at op %d: %s\n", i, hipGetErrorString(err));
            return 1;
        }

        // Wait on event (same stream — this is what ggml-hip does for dependencies)
        err = hipStreamWaitEvent(stream, event, 0);
        if (err != hipSuccess) {
            printf("hipStreamWaitEvent FAILED at op %d: %s\n", i, hipGetErrorString(err));
            return 1;
        }

        if (i % 100 == 0 && i > 0) printf("  op %d OK\n", i);
    }

    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("hipDeviceSynchronize FAILED: %s\n", hipGetErrorString(err));
        return 1;
    }

    int *h_buf = (int*)malloc(n * sizeof(int));
    hipMemcpy(h_buf, d_buf, n * sizeof(int), hipMemcpyDeviceToHost);

    int fails = 0;
    for (int i = 0; i < n; i++) {
        if (h_buf[i] != n_ops) {
            if (fails < 5)
                printf("  buf[%d] = %d (expected %d)\n", i, h_buf[i], n_ops);
            fails++;
        }
    }

    printf("Result: %d / %d correct\n", n - fails, n);

    hipEventDestroy(event);
    hipStreamDestroy(stream);
    free(h_buf);
    hipFree(d_buf);
    return fails > 0 ? 1 : 0;
}
