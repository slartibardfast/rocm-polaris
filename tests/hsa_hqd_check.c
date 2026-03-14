// Submit a barrier packet and hold queue open for HQD register inspection
#include <hsa/hsa.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

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
    printf("=== HQD Register Check ===\n");
    hsa_init();

    hsa_agent_t gpu = {0};
    hsa_iterate_agents(find_gpu, &gpu);

    hsa_queue_t *queue;
    hsa_queue_create(gpu, 1024, HSA_QUEUE_TYPE_MULTI, NULL, NULL, 0, 0, &queue);
    printf("Queue id=%u\n", queue->id);

    // Submit barrier packet
    hsa_signal_t signal;
    hsa_signal_create(1, 0, NULL, &signal);

    uint64_t write_idx = hsa_queue_add_write_index_relaxed(queue, 1);
    uint64_t mask = queue->size - 1;
    hsa_barrier_and_packet_t *pkt =
        (hsa_barrier_and_packet_t*)((char*)queue->base_address +
        (write_idx & mask) * 64);

    memset(pkt, 0, sizeof(*pkt));
    pkt->completion_signal = signal;

    uint16_t header = HSA_PACKET_TYPE_BARRIER_AND;
    header |= (1 << HSA_PACKET_HEADER_BARRIER);
    header |= (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE);
    header |= (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    __atomic_store_n((uint16_t*)pkt, header, __ATOMIC_RELEASE);

    printf("Wrote packet at idx %lu\n", write_idx);
    printf("write_dispatch_id before doorbell: %lu\n",
           hsa_queue_load_write_index_relaxed(queue));

    // Ring doorbell
    hsa_signal_store_relaxed(queue->doorbell_signal, write_idx);
    printf("Doorbell rung with %lu\n", write_idx);

    printf("\nSleeping 15s — run: sudo cat /sys/kernel/debug/kfd/hqds\n");
    printf("Look for CP_HQD_PQ_WPTR value and PQ_RPTR\n");
    fflush(stdout);

    for (int i = 0; i < 15; i++) {
        sleep(1);
        printf("  [%2ds] read_idx=%lu write_idx=%lu signal=%ld\n",
               i+1,
               hsa_queue_load_read_index_relaxed(queue),
               hsa_queue_load_write_index_relaxed(queue),
               (long)hsa_signal_load_relaxed(signal));
        fflush(stdout);
    }

    hsa_signal_destroy(signal);
    hsa_queue_destroy(queue);
    hsa_shut_down();
    return 0;
}
