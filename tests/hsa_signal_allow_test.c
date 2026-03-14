// Test: explicitly allow GPU access to signal memory before dispatch
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

static hsa_status_t find_gpu(hsa_agent_t agent, void *data) {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_GPU) {
        *(hsa_agent_t*)data = agent;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

int main() {
    setbuf(stdout, NULL);
    printf("=== Signal Allow-Access Test ===\n");

    hsa_init();
    hsa_agent_t gpu = {0};
    hsa_iterate_agents(find_gpu, &gpu);
    if (!gpu.handle) { printf("No GPU\n"); return 1; }

    hsa_signal_t signal;
    hsa_signal_create(1, 0, NULL, &signal);
    printf("Signal: 0x%lx, value=%ld\n", signal.handle,
        *((int64_t*)signal.handle + 1));

    // KEY: Explicitly allow GPU access to the signal memory region
    hsa_amd_pointer_info_t info;
    info.size = sizeof(info);
    hsa_amd_pointer_info((void*)signal.handle, &info, NULL, NULL, NULL);
    printf("Signal region: base=%p, size=%zu\n",
        info.agentBaseAddress, info.sizeInBytes);

    hsa_status_t s;
    s = hsa_amd_agents_allow_access(1, &gpu, NULL, info.agentBaseAddress);
    printf("hsa_amd_agents_allow_access(GPU, signal_base): %d\n", s);

    // Create queue
    hsa_queue_t *queue = NULL;
    s = hsa_queue_create(gpu, 1024, HSA_QUEUE_TYPE_MULTI,
                         NULL, NULL, 0, 0, &queue);
    printf("Queue: status=%d id=%u\n", s, s==0 ? queue->id : 0);
    if (s != HSA_STATUS_SUCCESS) { _exit(1); }

    // Dispatch barrier
    uint64_t wi = hsa_queue_add_write_index_relaxed(queue, 1);
    hsa_barrier_and_packet_t *pkt =
        (hsa_barrier_and_packet_t*)((char*)queue->base_address +
        (wi & (queue->size-1)) * 64);
    memset(pkt, 0, sizeof(*pkt));
    pkt->completion_signal = signal;

    uint16_t header = HSA_PACKET_TYPE_BARRIER_AND |
        (1 << HSA_PACKET_HEADER_BARRIER) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    __atomic_store_n((uint16_t*)pkt, header, __ATOMIC_RELEASE);
    hsa_signal_store_relaxed(queue->doorbell_signal,
        hsa_queue_load_write_index_relaxed(queue));

    printf("Dispatched, waiting 3s...\n");
    int64_t *sig_val = (int64_t*)signal.handle + 1;
    struct timespec ts_start, ts_now;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    while (1) {
        int64_t val = __atomic_load_n(sig_val, __ATOMIC_ACQUIRE);
        if (val == 0) { printf("PASS! Signal decremented to %ld!\n", val); break; }
        clock_gettime(CLOCK_MONOTONIC, &ts_now);
        if ((ts_now.tv_sec - ts_start.tv_sec) > 3) {
            printf("TIMEOUT signal=%ld\n", val);
            break;
        }
    }

    hsa_signal_destroy(signal);
    hsa_queue_destroy(queue);
    hsa_shut_down();
    printf("Done.\n");
    return 0;
}
