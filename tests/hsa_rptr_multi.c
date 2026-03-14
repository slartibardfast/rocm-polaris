/* Test: submit multiple barrier packets to exceed RPTR_BLOCK_SIZE threshold.
 * RPTR_BLOCK_SIZE=5 means report every 2^5=32 dwords = 2 AQL packets.
 * We submit 4 packets to ensure we cross the threshold. */
#include <hsa/hsa.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
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
    printf("=== Multi-packet RPTR Test ===\n");

    hsa_status_t s = hsa_init();
    if (s != HSA_STATUS_SUCCESS) { printf("hsa_init: %d\n", s); return 1; }

    hsa_agent_t gpu = {0};
    hsa_iterate_agents(find_gpu, &gpu);
    if (!gpu.handle) { printf("No GPU\n"); return 1; }

    hsa_queue_t *queue = NULL;
    s = hsa_queue_create(gpu, 1024, HSA_QUEUE_TYPE_MULTI, NULL, NULL, 0, 0, &queue);
    if (s != HSA_STATUS_SUCCESS) { printf("hsa_queue_create: %d\n", s); return 1; }

    const int N = 4;  /* Submit 4 barrier packets */
    hsa_signal_t signals[4];

    for (int i = 0; i < N; i++) {
        hsa_signal_create(1, 0, NULL, &signals[i]);

        uint64_t write_idx = hsa_queue_add_write_index_relaxed(queue, 1);
        uint64_t mask = queue->size - 1;
        hsa_barrier_and_packet_t *barrier =
            (hsa_barrier_and_packet_t*)((char*)queue->base_address +
            (write_idx & mask) * 64);
        memset(barrier, 0, sizeof(*barrier));
        barrier->completion_signal = signals[i];

        uint16_t header = HSA_PACKET_TYPE_BARRIER_AND;
        header |= (1 << HSA_PACKET_HEADER_BARRIER);
        header |= (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE);
        header |= (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
        __atomic_store_n((uint16_t*)barrier, header, __ATOMIC_RELEASE);
    }

    printf("Submitted %d barrier packets\n", N);
    printf("Ringing doorbell (write_idx=%lu)...\n",
           (unsigned long)hsa_queue_load_write_index_relaxed(queue));
    hsa_signal_store_relaxed(queue->doorbell_signal,
                             hsa_queue_load_write_index_relaxed(queue));

    /* Sleep for CP to process */
    printf("Sleeping 2s...\n");
    sleep(2);

    uint64_t read_idx = hsa_queue_load_read_index_relaxed(queue);
    uint64_t write_idx = hsa_queue_load_write_index_relaxed(queue);
    printf("\nread_dispatch_id: %lu\n", (unsigned long)read_idx);
    printf("write_dispatch_id: %lu\n", (unsigned long)write_idx);

    for (int i = 0; i < N; i++) {
        printf("signal[%d] = %ld\n", i,
               (long)hsa_signal_load_relaxed(signals[i]));
        hsa_signal_destroy(signals[i]);
    }

    hsa_queue_destroy(queue);
    hsa_shut_down();
    printf("Done.\n");
    return 0;
}
