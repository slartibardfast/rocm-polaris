#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void write_val(int *out, int val) {
    if (threadIdx.x == 0) *out = val;
}

int main() {
    int *d;
    hipMalloc(&d, sizeof(int));
    
    // 512KB scrub buffer — enough to fill entire L2
    void *scrub;
    hipMalloc(&scrub, 512 * 1024);

    int fail = 0;
    for (int i = 0; i < 2000; i++) {
        int *tmp;
        hipMalloc(&tmp, 1024);
        int h_in = i * 3;
        hipMemcpy(tmp, &h_in, sizeof(int), hipMemcpyHostToDevice);

        // L2 scrub: memset 512KB to force L2 eviction of all stale lines
        hipMemset(scrub, 0, 512 * 1024);

        write_val<<<1,1>>>(d, i);
        hipDeviceSynchronize();
        int h;
        hipMemcpy(&h, d, sizeof(int), hipMemcpyDeviceToHost);
        if (h != i) {
            printf("FAIL %d: got %d\n", i, h);
            fail++;
            if (fail >= 3) { hipFree(tmp); break; }
        }
        hipFree(tmp);
    }
    printf("%d fail / 2000\n", fail);
    hipFree(scrub);
    hipFree(d);
}
