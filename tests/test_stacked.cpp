// test_stacked.cpp — Stack multiple mechanisms and measure combined failure rate
//
// Build: hipcc --offload-arch=gfx803 -o test_stacked test_stacked.cpp \
//        -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64
// Run:   ./test_stacked <variant> [iters]
//
// Variants:
//   baseline           — nothing
//   mfence             — _mm_mfence only
//   mfence_hdp         — mfence + HDP flush+read
//   mfence_usleep1     — mfence + usleep(1)
//   mfence_usleep5     — mfence + usleep(5)
//   all_no_sleep       — mfence + HDP flush+read x2
//   all_usleep1        — mfence + HDP flush+read + usleep(1)
//   usleep100          — usleep(100) alone
//   usleep1000         — usleep(1000) alone

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

static hsa_status_t find_gpu(hsa_agent_t agent, void *) {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type != HSA_DEVICE_TYPE_GPU) return HSA_STATUS_SUCCESS;
    hsa_amd_hdp_flush_t hdp = {0};
    hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_HDP_FLUSH, &hdp);
    g_hdp_flush = hdp.HDP_MEM_FLUSH_CNTL;
    g_hdp_read = hdp.HDP_REG_FLUSH_CNTL;
    return HSA_STATUS_SUCCESS;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <variant> [iters]\n", argv[0]); return 1; }
    const char *v = argv[1];
    int iters = argc > 2 ? atoi(argv[2]) : 10000;

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    hsa_iterate_agents(find_gpu, nullptr);

    int *d_result;
    hipMalloc(&d_result, sizeof(int));

    auto t0 = std::chrono::steady_clock::now();
    int fails = 0;
    for (int i = 0; i < iters; i++) {
        hipMemset(d_result, 0xFF, sizeof(int));

        if (strstr(v, "mfence")) _mm_mfence();
        if (strstr(v, "hdp")) {
            if (g_hdp_flush) *g_hdp_flush = 1;
            if (g_hdp_read) { volatile uint32_t x = *g_hdp_read; (void)x; }
        }
        if (strcmp(v, "usleep100") == 0) usleep(100);
        else if (strcmp(v, "usleep1000") == 0) usleep(1000);
        else if (strstr(v, "usleep1") && !strstr(v, "usleep10")) usleep(1);
        else if (strstr(v, "usleep5")) usleep(5);

        record_val<<<1, 1>>>(d_result, i * 7 + 13);
        hipDeviceSynchronize();
        int h;
        hipMemcpy(&h, d_result, sizeof(int), hipMemcpyDeviceToHost);
        if (h != i * 7 + 13) fails++;
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    hipFree(d_result);
    printf("%-25s %3d / %d  (%.0f dispatch/s)\n", v, fails, iters, iters/(ms/1000.0));
    return fails > 0 ? 1 : 0;
}
