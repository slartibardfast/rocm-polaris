// test_isolate_fix2.cpp — Find minimum effective PCIe write drain mechanism
//
// Build: hipcc --offload-arch=gfx803 -o test_isolate_fix2 test_isolate_fix2.cpp \
//        -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64
// Run:   ./test_isolate_fix2

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <cstdio>
#include <unistd.h>
#include <sched.h>
#include <x86intrin.h>

__global__ void record_val(int *out, int val) {
    if (threadIdx.x == 0) *out = val;
}

// Get HDP flush pointers (MMIO BAR mapped)
static volatile uint32_t *g_hdp_flush = nullptr;
static volatile uint32_t *g_hdp_read = nullptr;

static hsa_status_t find_hdp(hsa_agent_t agent, void *) {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type != HSA_DEVICE_TYPE_GPU) return HSA_STATUS_SUCCESS;
    hsa_amd_hdp_flush_t hdp = {0};
    hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_HDP_FLUSH, &hdp);
    g_hdp_flush = hdp.HDP_MEM_FLUSH_CNTL;
    g_hdp_read = hdp.HDP_REG_FLUSH_CNTL;
    return HSA_STATUS_SUCCESS;
}

typedef void (*intervention_fn)(void);

static void do_nothing(void) {}
static void do_usleep1(void) { usleep(1); }
static void do_usleep5(void) { usleep(5); }
static void do_usleep10(void) { usleep(10); }
static void do_sfence(void) { _mm_sfence(); }
static void do_mfence(void) { _mm_mfence(); }
static void do_lfence(void) { _mm_lfence(); }
static void do_yield(void) { sched_yield(); }

// MMIO read from GPU BAR (forces PCIe posted write drain per spec)
static void do_hdp_read(void) {
    if (g_hdp_read) { volatile uint32_t x = *g_hdp_read; (void)x; }
}

// MMIO write+read (HDP flush + drain)
static void do_hdp_flush_and_read(void) {
    if (g_hdp_flush) *g_hdp_flush = 1;
    if (g_hdp_read) { volatile uint32_t x = *g_hdp_read; (void)x; }
}

// mfence + sfence combo
static void do_mfence_sfence(void) { _mm_sfence(); _mm_mfence(); }

// Double mfence
static void do_double_mfence(void) { _mm_mfence(); _mm_mfence(); }

static int run_variant(const char *label, int iters, intervention_fn fn) {
    int *d_result;
    hipMalloc(&d_result, sizeof(int));

    int fails = 0;
    for (int i = 0; i < iters; i++) {
        hipMemset(d_result, 0xFF, sizeof(int));

        fn();  // <-- intervention BEFORE kernel launch

        record_val<<<1, 1>>>(d_result, i * 7 + 13);
        hipDeviceSynchronize();

        int h;
        hipMemcpy(&h, d_result, sizeof(int), hipMemcpyDeviceToHost);
        if (h != i * 7 + 13) {
            fails++;
        }
    }
    hipFree(d_result);
    printf("  %-40s %3d fails / %d\n", label, fails, iters);
    return fails;
}

int main() {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("Device: %s (%s)\n\n", prop.name, prop.gcnArchName);

    // Get HDP MMIO pointers
    hsa_iterate_agents(find_hdp, nullptr);
    printf("HDP flush MMIO: %p, read MMIO: %p\n\n",
           (void*)g_hdp_flush, (void*)g_hdp_read);

    const int N = 2000;

    struct { const char *name; intervention_fn fn; } tests[] = {
        {"A: baseline (nothing)",           do_nothing},
        {"B: usleep(1)",                    do_usleep1},
        {"C: usleep(5)",                    do_usleep5},
        {"D: usleep(10)",                   do_usleep10},
        {"E: _mm_sfence",                   do_sfence},
        {"F: _mm_mfence",                   do_mfence},
        {"G: _mm_lfence",                   do_lfence},
        {"H: sched_yield",                  do_yield},
        {"I: MMIO read (GPU BAR)",          do_hdp_read},
        {"J: HDP flush + MMIO read",       do_hdp_flush_and_read},
        {"K: sfence + mfence",              do_mfence_sfence},
        {"L: double mfence",               do_double_mfence},
    };

    for (auto &t : tests) {
        run_variant(t.name, N, t.fn);
    }

    return 0;
}
