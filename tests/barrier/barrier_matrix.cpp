// Barrier resolution test matrix — isolates cross-queue and memory ordering
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/amd_hsa_signal.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <unistd.h>

#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#endif

#define HSA_CHECK(call) do { hsa_status_t s = (call); \
    if (s != HSA_STATUS_SUCCESS) { fprintf(stderr, "HSA err %d at %s:%d\n", s, __FILE__, __LINE__); _exit(1); } } while(0)

static hsa_agent_t gpu_agent;
static hsa_status_t find_gpu(hsa_agent_t agent, void*) {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_GPU) { gpu_agent = agent; return HSA_STATUS_INFO_BREAK; }
    return HSA_STATUS_SUCCESS;
}

static void submit_barrier(hsa_queue_t* q, hsa_signal_t dep, hsa_signal_t comp) {
    uint64_t idx = hsa_queue_load_write_index_relaxed(q);
    hsa_barrier_and_packet_t* pkt = (hsa_barrier_and_packet_t*)
        ((char*)q->base_address + (idx & (q->size - 1)) * 64);
    memset(pkt, 0, sizeof(*pkt));
    pkt->dep_signal[0] = dep;
    pkt->completion_signal = comp;
    __atomic_store_n((uint16_t*)pkt,
                     (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
                     (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE),
                     __ATOMIC_RELEASE);
    hsa_queue_store_write_index_relaxed(q, idx + 1);
    hsa_signal_store_relaxed(q->doorbell_signal, idx);
}

static void signal_sub_clflush(hsa_signal_t sig) {
    // Mimic bounce buffer: atomic sub + clflush + mfence
    amd_signal_t* ams = reinterpret_cast<amd_signal_t*>(sig.handle);
    __atomic_sub_fetch(&ams->value, 1, __ATOMIC_RELEASE);
    _mm_clflush(const_cast<void*>(static_cast<volatile void*>(&ams->value)));
    _mm_mfence();
}

static bool wait_signal(hsa_signal_t sig, uint64_t timeout_ns) {
    hsa_signal_value_t v = hsa_signal_wait_scacquire(
        sig, HSA_SIGNAL_CONDITION_EQ, 0, timeout_ns, HSA_WAIT_STATE_ACTIVE);
    return v == 0;
}

static bool run_test(const char* name, bool cross_queue, bool use_sub_clflush,
                     int pre_packets, int delay_ms, bool re_ring) {
    printf("  %-45s ", name);
    fflush(stdout);

    hsa_queue_t *q1, *q2;
    HSA_CHECK(hsa_queue_create(gpu_agent, 128, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX, UINT32_MAX, &q1));
    if (cross_queue)
        HSA_CHECK(hsa_queue_create(gpu_agent, 128, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX, UINT32_MAX, &q2));
    else
        q2 = q1;

    hsa_signal_t dep, comp;
    HSA_CHECK(hsa_signal_create(1, 0, NULL, &dep));
    HSA_CHECK(hsa_signal_create(1, 0, NULL, &comp));

    // Optionally submit pre-packets (nop barriers with no deps) to give queue history
    for (int i = 0; i < pre_packets; i++) {
        hsa_signal_t dummy;
        HSA_CHECK(hsa_signal_create(1, 0, NULL, &dummy));
        submit_barrier(q1, {0}, dummy);
        // Wait for it
        wait_signal(dummy, 1000000000ULL);
        hsa_signal_destroy(dummy);
    }

    // Submit barrier on q1 that depends on dep signal
    submit_barrier(q1, dep, comp);

    if (delay_ms > 0) usleep(delay_ms * 1000);

    // Decrement dep signal
    if (use_sub_clflush)
        signal_sub_clflush(dep);
    else
        hsa_signal_store_screlease(dep, 0);

    if (re_ring) {
        // Re-ring q1 doorbell
        uint64_t wid = hsa_queue_load_write_index_relaxed(q1);
        hsa_signal_store_relaxed(q1->doorbell_signal, wid - 1);
    }

    bool pass = wait_signal(comp, 3000000000ULL);
    printf("%s\n", pass ? "PASS" : "FAIL");

    hsa_signal_destroy(dep);
    hsa_signal_destroy(comp);
    hsa_queue_destroy(q1);
    if (cross_queue) hsa_queue_destroy(q2);
    return pass;
}

int main() {
    setbuf(stdout, NULL);
    HSA_CHECK(hsa_init());
    hsa_iterate_agents(find_gpu, NULL);

    char name[64];
    hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_NAME, name);
    printf("GPU: %s\n\n", name);

    int fail = 0;
    //                                            name                                          cross  sub    pre  delay rering
    fail += !run_test("A1: same-queue, store_screlease",                                        false, false, 0,   0,    false);
    fail += !run_test("A2: same-queue, sub+clflush",                                            false, true,  0,   0,    false);
    fail += !run_test("B1: cross-queue, store_screlease",                                       true,  false, 0,   0,    false);
    fail += !run_test("B2: cross-queue, sub+clflush",                                           true,  true,  0,   0,    false);
    fail += !run_test("C1: same-queue, 10 pre-packets, store_screlease",                        false, false, 10,  0,    false);
    fail += !run_test("C2: same-queue, 10 pre-packets, sub+clflush",                            false, true,  10,  0,    false);
    fail += !run_test("D1: same-queue, 100ms delay, store_screlease",                           false, false, 0,   100,  false);
    fail += !run_test("D2: same-queue, 100ms delay, sub+clflush",                               false, true,  0,   100,  false);
    fail += !run_test("D3: cross-queue, 100ms delay, sub+clflush",                              true,  true,  0,   100,  false);
    fail += !run_test("E1: same-queue, 100ms delay, sub+clflush, re-ring",                      false, true,  0,   100,  true);
    fail += !run_test("E2: cross-queue, 100ms delay, sub+clflush, re-ring",                     true,  true,  0,   100,  true);
    fail += !run_test("F1: cross-queue, 10 pre-pkts, 100ms delay, sub+clflush",                 true,  true,  10,  100,  false);
    fail += !run_test("F2: cross-queue, 10 pre-pkts, 100ms delay, sub+clflush, re-ring",        true,  true,  10,  100,  true);

    printf("\n%d/%d passed\n", 13-fail, 13);
    hsa_shut_down();
    return fail;
}
