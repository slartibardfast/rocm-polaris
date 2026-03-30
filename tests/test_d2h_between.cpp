// test_d2h_between.cpp — Does D2H copy between dispatch passes trigger CP stall?
// llama.cpp reads back logits between forward passes via hipMemcpy D2H.
//
// Build: hipcc -o test_d2h_between test_d2h_between.cpp

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void add_one(float *buf, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) buf[i] += 1.0f;
}

int main(int argc, char **argv) {
    int pass1 = argc > 1 ? atoi(argv[1]) : 800;
    int pass2 = argc > 2 ? atoi(argv[2]) : 10;
    int n = 4096;

    float *d_buf;
    hipMalloc(&d_buf, n * sizeof(float));
    hipMemset(d_buf, 0, n * sizeof(float));

    float *h_buf = (float*)malloc(n * sizeof(float));

    hipStream_t stream;
    hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);

    printf("Pass 1: %d dispatches...\n", pass1);
    for (int i = 0; i < pass1; i++) {
        add_one<<<n / 256, 256, 0, stream>>>(d_buf, n);
    }
    hipStreamSynchronize(stream);
    printf("  Pass 1 sync OK\n");

    // D2H copy between passes (like llama.cpp reading logits)
    printf("D2H copy (like logits readback)...\n");
    hipMemcpy(h_buf, d_buf, n * sizeof(float), hipMemcpyDeviceToHost);
    printf("  D2H OK, buf[0]=%.0f\n", h_buf[0]);

    // hipDeviceSynchronize to match llama.cpp's behavior
    hipDeviceSynchronize();
    printf("  DeviceSync OK\n");

    printf("Pass 2: %d dispatches...\n", pass2);
    for (int i = 0; i < pass2; i++) {
        add_one<<<n / 256, 256, 0, stream>>>(d_buf, n);
        if (i % 3 == 0) printf("  dispatch %d OK\n", i);
    }

    hipError_t err = hipStreamSynchronize(stream);
    printf("  Pass 2 sync: %s\n", err == hipSuccess ? "OK" : hipGetErrorString(err));

    printf("PASS\n");
    hipStreamDestroy(stream);
    free(h_buf);
    hipFree(d_buf);
    return 0;
}
