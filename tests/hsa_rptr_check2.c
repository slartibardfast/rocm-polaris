#include <hsa/hsa.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

static hsa_status_t find_gpu(hsa_agent_t agent, void *data) {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_GPU) { *(hsa_agent_t*)data = agent; return HSA_STATUS_INFO_BREAK; }
    return HSA_STATUS_SUCCESS;
}

int main() {
    printf("=== RPTR Check v2 ===\n");
    hsa_init();
    hsa_agent_t gpu = {0};
    hsa_iterate_agents(find_gpu, &gpu);
    hsa_queue_t *queue = NULL;
    hsa_queue_create(gpu, 1024, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX, UINT32_MAX, &queue);

    hsa_signal_t signal;
    hsa_signal_create(1, 0, NULL, &signal);

    uint64_t idx = hsa_queue_add_write_index_relaxed(queue, 1);
    hsa_barrier_and_packet_t *b = (hsa_barrier_and_packet_t*)
        ((char*)queue->base_address + (idx & (queue->size-1)) * 64);
    memset(b, 0, sizeof(*b));
    b->completion_signal = signal;
    uint16_t hdr = HSA_PACKET_TYPE_BARRIER_AND |
                   (1 << HSA_PACKET_HEADER_BARRIER) |
                   (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                   (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    __atomic_store_n((uint16_t*)b, hdr, __ATOMIC_RELEASE);

    uint64_t new_idx = hsa_queue_load_write_index_relaxed(queue);
    printf("Ringing doorbell with %lu\n", new_idx);
    hsa_signal_store_relaxed(queue->doorbell_signal, new_idx);

    printf("Sleeping 15s — dump HQDs now!\n");
    printf("read_dispatch_id=%lu signal=%ld\n",
           hsa_queue_load_read_index_relaxed(queue),
           hsa_signal_load_relaxed(signal));
    fflush(stdout);
    sleep(15);

    printf("After sleep: read_dispatch_id=%lu signal=%ld\n",
           hsa_queue_load_read_index_relaxed(queue),
           hsa_signal_load_relaxed(signal));
    hsa_signal_destroy(signal);
    hsa_queue_destroy(queue);
    hsa_shut_down();
    return 0;
}
