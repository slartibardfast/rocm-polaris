// test_isolate_single.cpp — Single-variant isolation test
//
// Usage: ./test_isolate_single <variant> [iters]
// Variants: baseline, usleep1, usleep5, usleep10, sfence, mfence,
//           hdp_read, hdp_flush_read, hdp_flush_read_x2, ioctl
//
// Build: hipcc --offload-arch=gfx803 -o test_isolate_single test_isolate_single.cpp \
//        -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <x86intrin.h>
#include <chrono>

__global__ void record_val(int *out, int val) {
    if (threadIdx.x == 0) *out = val;
}

static volatile uint32_t *g_hdp_flush = nullptr;
static volatile uint32_t *g_hdp_read = nullptr;
static hsa_agent_t g_gpu = {};

static hsa_status_t find_gpu(hsa_agent_t agent, void *) {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type != HSA_DEVICE_TYPE_GPU) return HSA_STATUS_SUCCESS;
    g_gpu = agent;
    hsa_amd_hdp_flush_t hdp = {0};
    hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_HDP_FLUSH, &hdp);
    g_hdp_flush = hdp.HDP_MEM_FLUSH_CNTL;
    g_hdp_read = hdp.HDP_REG_FLUSH_CNTL;
    return HSA_STATUS_SUCCESS;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <variant> [iters]\n", argv[0]);
        return 1;
    }
    const char *variant = argv[1];
    int iters = argc > 2 ? atoi(argv[2]) : 2000;

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    hsa_iterate_agents(find_gpu, nullptr);

    int *d_result;
    hipMalloc(&d_result, sizeof(int));

    auto t0 = std::chrono::steady_clock::now();
    int fails = 0;
    for (int i = 0; i < iters; i++) {
        hipMemset(d_result, 0xFF, sizeof(int));

        // Intervention
        if (strcmp(variant, "usleep1") == 0) usleep(1);
        else if (strcmp(variant, "usleep5") == 0) usleep(5);
        else if (strcmp(variant, "usleep10") == 0) usleep(10);
        else if (strcmp(variant, "sfence") == 0) _mm_sfence();
        else if (strcmp(variant, "mfence") == 0) _mm_mfence();
        else if (strcmp(variant, "hdp_read") == 0) {
            if (g_hdp_read) { volatile uint32_t x = *g_hdp_read; (void)x; }
        }
        else if (strcmp(variant, "hdp_flush_read") == 0) {
            if (g_hdp_flush) *g_hdp_flush = 1;
            if (g_hdp_read) { volatile uint32_t x = *g_hdp_read; (void)x; }
        }
        else if (strcmp(variant, "hdp_flush_read_x2") == 0) {
            if (g_hdp_flush) *g_hdp_flush = 1;
            if (g_hdp_read) { volatile uint32_t x = *g_hdp_read; (void)x; }
            if (g_hdp_flush) *g_hdp_flush = 1;
            if (g_hdp_read) { volatile uint32_t x = *g_hdp_read; (void)x; }
        }
        else if (strcmp(variant, "ioctl") == 0) {
            hsa_amd_hdp_flush_wait(g_gpu, 10000);
        }
        // else: baseline (nothing)

        record_val<<<1, 1>>>(d_result, i * 7 + 13);
        hipDeviceSynchronize();

        int h;
        hipMemcpy(&h, d_result, sizeof(int), hipMemcpyDeviceToHost);
        if (h != i * 7 + 13) {
            fails++;
            if (fails <= 3)
                printf("  iter %d: expected %d, got %d\n", i, i*7+13, h);
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    hipFree(d_result);
    printf("%-25s %3d / %d  (%.0f dispatch/s)\n", variant, fails, iters, iters/(ms/1000.0));
    return fails > 0 ? 1 : 0;
}
