#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>

#define CHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        printf("FAIL: %s returned %d (%s)\n", #cmd, e, hipGetErrorString(e)); \
        return 1; \
    } \
} while (0)

__global__ void set_kernel(int *p, int val) {
    *p = val;
}

int main() {
    printf("=== HIP Smoke Test ===\n\n");

    // 5a: Init + device count
    printf("[5a] hipInit + hipGetDeviceCount\n");
    CHECK(hipInit(0));
    int count = 0;
    CHECK(hipGetDeviceCount(&count));
    printf("     devices: %d\n", count);
    if (count < 1) { printf("FAIL: no devices\n"); return 1; }
    printf("     PASS\n\n");

    // 5b: Device properties
    printf("[5b] hipGetDeviceProperties\n");
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, 0));
    printf("     name: %s\n", props.name);
    printf("     gcnArchName: %s\n", props.gcnArchName);
    printf("     totalGlobalMem: %zu MB\n", props.totalGlobalMem / (1024*1024));
    printf("     maxThreadsPerBlock: %d\n", props.maxThreadsPerBlock);
    printf("     PASS\n\n");

    // 5c: Malloc + Free
    printf("[5c] hipSetDevice + hipMalloc + hipFree\n");
    CHECK(hipSetDevice(0));
    void *devptr = nullptr;
    CHECK(hipMalloc(&devptr, 4096));
    printf("     devptr: %p\n", devptr);
    CHECK(hipFree(devptr));
    printf("     PASS\n\n");

    // 5d-alt: Memcpy round-trip FIRST (isolate from memset)
    printf("[5d] hipMemcpy round-trip\n");
    int h_src[64], h_dst[64];
    memset(h_dst, 0xFF, sizeof(h_dst));
    for (int i = 0; i < 64; i++) h_src[i] = i * 7 + 3;
    int *d_mc = nullptr;
    CHECK(hipMalloc(&d_mc, sizeof(h_src)));
    CHECK(hipMemcpy(d_mc, h_src, sizeof(h_src), hipMemcpyHostToDevice));
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(h_dst, d_mc, sizeof(h_dst), hipMemcpyDeviceToHost));
    CHECK(hipDeviceSynchronize());
    printf("     h_src[0..3]: %d %d %d %d\n", h_src[0], h_src[1], h_src[2], h_src[3]);
    printf("     h_dst[0..3]: %d %d %d %d\n", h_dst[0], h_dst[1], h_dst[2], h_dst[3]);
    int cpy_ok = (memcmp(h_src, h_dst, sizeof(h_src)) == 0);
    printf("     round-trip: %s\n", cpy_ok ? "match" : "MISMATCH");
    if (!cpy_ok) {
        printf("     first 16 dst bytes:");
        unsigned char *p = (unsigned char*)h_dst;
        for (int i = 0; i < 16; i++) printf(" %02x", p[i]);
        printf("\n");
    }
    CHECK(hipFree(d_mc));
    printf("     %s\n\n", cpy_ok ? "PASS" : "FAIL");
    if (!cpy_ok) return 1;

    // 5e: Memset
    printf("[5e] hipMemset\n");
    int *d_buf = nullptr;
    CHECK(hipMalloc(&d_buf, 256));
    CHECK(hipMemset(d_buf, 0x42, 256));
    CHECK(hipDeviceSynchronize());
    unsigned char h_buf[256];
    memset(h_buf, 0xFF, 256);
    CHECK(hipMemcpy(h_buf, d_buf, 256, hipMemcpyDeviceToHost));
    CHECK(hipDeviceSynchronize());
    int memset_ok = 1;
    for (int i = 0; i < 256; i++) {
        if (h_buf[i] != 0x42) { memset_ok = 0; break; }
    }
    printf("     memset check: %s\n", memset_ok ? "all 0x42" : "MISMATCH");
    if (!memset_ok) {
        printf("     first 32 bytes:");
        for (int i = 0; i < 32; i++) printf(" %02x", h_buf[i]);
        printf("\n");
        printf("FAIL\n"); return 1;
    }
    printf("     PASS\n\n");

    // 5f: Kernel launch
    printf("[5f] Kernel launch\n");
    int *d_val = nullptr;
    CHECK(hipMalloc(&d_val, sizeof(int)));
    CHECK(hipMemset(d_val, 0, sizeof(int)));
    set_kernel<<<1, 1>>>(d_val, 42);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    int h_val = 0;
    CHECK(hipMemcpy(&h_val, d_val, sizeof(int), hipMemcpyDeviceToHost));
    printf("     kernel result: %d (expected 42)\n", h_val);
    CHECK(hipFree(d_val));
    if (h_val != 42) { printf("FAIL\n"); return 1; }
    printf("     PASS\n\n");

    printf("=== All tests passed ===\n");
    return 0;
}
