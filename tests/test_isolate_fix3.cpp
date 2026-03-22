// test_isolate_fix3.cpp — Compare HDP flush mechanisms for perf + correctness
//
// Build: hipcc --offload-arch=gfx803 -o test_isolate_fix3 test_isolate_fix3.cpp \
//        -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64
// Run:   ./test_isolate_fix3

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <cstdio>
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

typedef void (*intervention_fn)(void);

static void do_nothing(void) {}

// Userspace MMIO: write HDP flush + read back (~1us, no syscall)
static void do_mmio_flush_read(void) {
    if (g_hdp_flush) *g_hdp_flush = 1;
    if (g_hdp_read) { volatile uint32_t x = *g_hdp_read; (void)x; }
}

// Userspace MMIO: just the write, no read (~100ns)
static void do_mmio_flush_only(void) {
    if (g_hdp_flush) *g_hdp_flush = 1;
}

// HSA API ioctl (~10us, syscall overhead)
static void do_hsa_api(void) {
    hsa_amd_hdp_flush_wait(g_gpu, 10000);
}

// Userspace MMIO flush + read, twice
static void do_mmio_flush_read_x2(void) {
    if (g_hdp_flush) *g_hdp_flush = 1;
    if (g_hdp_read) { volatile uint32_t x = *g_hdp_read; (void)x; }
    if (g_hdp_flush) *g_hdp_flush = 1;
    if (g_hdp_read) { volatile uint32_t x = *g_hdp_read; (void)x; }
}

static int run_variant(const char *label, int iters, intervention_fn fn) {
    int *d_result;
    hipMalloc(&d_result, sizeof(int));

    auto t0 = std::chrono::steady_clock::now();
    int fails = 0;
    for (int i = 0; i < iters; i++) {
        hipMemset(d_result, 0xFF, sizeof(int));
        fn();
        record_val<<<1, 1>>>(d_result, i * 7 + 13);
        hipDeviceSynchronize();
        int h;
        hipMemcpy(&h, d_result, sizeof(int), hipMemcpyDeviceToHost);
        if (h != i * 7 + 13) fails++;
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double rate = iters / (ms / 1000.0);

    hipFree(d_result);
    printf("  %-40s %3d fails / %d  (%.0f dispatch/s, %.1f ms)\n",
           label, fails, iters, rate, ms);
    return fails;
}

int main() {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("Device: %s (%s)\n\n", prop.name, prop.gcnArchName);

    hsa_iterate_agents(find_gpu, nullptr);
    printf("HDP MMIO: flush=%p read=%p\n\n", (void*)g_hdp_flush, (void*)g_hdp_read);

    const int N = 2000;
    struct { const char *name; intervention_fn fn; } tests[] = {
        {"A: baseline",                 do_nothing},
        {"B: MMIO flush only (no read)", do_mmio_flush_only},
        {"C: MMIO flush + read",         do_mmio_flush_read},
        {"D: MMIO flush + read x2",      do_mmio_flush_read_x2},
        {"E: HSA API ioctl",             do_hsa_api},
    };

    for (auto &t : tests) {
        run_variant(t.name, N, t.fn);
    }
    return 0;
}
