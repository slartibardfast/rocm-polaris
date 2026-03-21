// trace_hdp_fix.cpp — Test HDP flush fixing kernarg corruption
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <cstdio>

static uint32_t *g_hdp_flush = nullptr;

void init_hdp() {
    hsa_agent_t gpu = {};
    auto cb = [](hsa_agent_t a, void *d) -> hsa_status_t {
        hsa_device_type_t t;
        hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
        if (t == HSA_DEVICE_TYPE_GPU) *(hsa_agent_t*)d = a;
        return HSA_STATUS_SUCCESS;
    };
    hsa_iterate_agents(cb, &gpu);
    hsa_amd_hdp_flush_t hdp;
    if (HSA_STATUS_SUCCESS == hsa_agent_get_info(gpu,
            (hsa_agent_info_t)HSA_AMD_AGENT_INFO_HDP_FLUSH, &hdp))
        g_hdp_flush = hdp.HDP_MEM_FLUSH_CNTL;
    printf("HDP flush: %p\n", g_hdp_flush);
}

__global__ void write_val(int *out, int val) {
    if (threadIdx.x == 0) *out = val;
}

int main() {
    int *d;
    hipMalloc(&d, sizeof(int));
    init_hdp();
    if (!g_hdp_flush) { printf("No HDP flush\n"); return 1; }

    int fail = 0;
    for (int i = 0; i < 2000; i++) {
        int *tmp;
        hipMalloc(&tmp, 1024);
        int h_in = i * 3;
        hipMemcpy(tmp, &h_in, sizeof(int), hipMemcpyHostToDevice);

        // HDP flush BEFORE kernel dispatch
        *g_hdp_flush = 1u;

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
    hipFree(d);
}
