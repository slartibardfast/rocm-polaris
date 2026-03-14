// Test: after barrier dep is resolved, does RPTR advance?
// (Don't check completion_signal — it's always stuck at 1 on no-atomics)
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/amd_hsa_signal.h>
#include <x86intrin.h>
#include <cstdio>
#include <cstring>
#include <unistd.h>

#define HSA_CHECK(call) do { hsa_status_t s = (call); if (s) { fprintf(stderr, "HSA err %d at %s:%d\n", s, __FILE__, __LINE__); _exit(1); } } while(0)

static hsa_agent_t gpu;
static hsa_status_t find_gpu(hsa_agent_t a, void*) {
    hsa_device_type_t t; hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
    if (t == HSA_DEVICE_TYPE_GPU) { gpu = a; return HSA_STATUS_INFO_BREAK; }
    return HSA_STATUS_SUCCESS;
}

int main() {
    setbuf(stdout, NULL);
    HSA_CHECK(hsa_init());
    hsa_iterate_agents(find_gpu, NULL);

    hsa_queue_t* q;
    HSA_CHECK(hsa_queue_create(gpu, 128, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX, UINT32_MAX, &q));

    hsa_signal_t dep, comp;
    HSA_CHECK(hsa_signal_create(1, 0, NULL, &dep));
    HSA_CHECK(hsa_signal_create(1, 0, NULL, &comp));

    // Submit barrier with dep
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

    printf("Barrier submitted. read_idx before stall: %lu\n",
           hsa_queue_load_read_index_relaxed(q));

    // Wait for CP to stall
    usleep(200000);
    uint64_t rid_stalled = hsa_queue_load_read_index_relaxed(q);
    printf("After 200ms: read_idx=%lu (expect 0 if barrier stalled)\n", rid_stalled);

    // Set dep=0 from CPU using sub+clflush (like bounce buffer)
    printf("Setting dep=0 via sub+clflush...\n");
    {
        amd_signal_t* ams = reinterpret_cast<amd_signal_t*>(dep.handle);
        __atomic_sub_fetch(&ams->value, 1, __ATOMIC_RELEASE);
        _mm_clflush(const_cast<void*>(static_cast<volatile void*>(&ams->value)));
        _mm_mfence();
    }

    // Poll read_idx via hsa_queue_load_read_index (goes through bounce buffer UpdateReadDispatchId)
    printf("Polling read_idx for 5s...\n");
    for (int i = 0; i < 50; i++) {
        uint64_t rid = hsa_queue_load_read_index_relaxed(q);
        if (rid > rid_stalled) {
            printf("PASS: RPTR advanced! read_idx=%lu after %dms\n", rid, (i+1)*100);
            goto done;
        }
        usleep(100000);
    }
    printf("FAIL: read_idx still %lu after 5s\n", hsa_queue_load_read_index_relaxed(q));

    // Re-ring doorbell
    printf("Re-ringing doorbell...\n");
    hsa_signal_store_relaxed(q->doorbell_signal, idx);
    for (int i = 0; i < 30; i++) {
        uint64_t rid = hsa_queue_load_read_index_relaxed(q);
        if (rid > rid_stalled) {
            printf("PASS after re-ring: read_idx=%lu after %dms\n", rid, (i+1)*100);
            goto done;
        }
        usleep(100000);
    }
    printf("FAIL even after re-ring: read_idx=%lu\n", hsa_queue_load_read_index_relaxed(q));

done:
    hsa_signal_destroy(dep);
    hsa_signal_destroy(comp);
    hsa_queue_destroy(q);
    hsa_shut_down();
    return 0;
}
