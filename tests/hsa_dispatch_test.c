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
    printf("=== HSA Barrier Dispatch Test ===\n");

    hsa_status_t s = hsa_init();
    printf("hsa_init: %d\n", s);
    if (s != HSA_STATUS_SUCCESS) return 1;

    hsa_agent_t gpu = {0};
    hsa_iterate_agents(find_gpu, &gpu);
    if (!gpu.handle) { printf("No GPU\n"); return 1; }

    char name[64];
    hsa_agent_get_info(gpu, HSA_AGENT_INFO_NAME, name);
    printf("GPU: %s\n", name);

    // Create queue
    hsa_queue_t *queue = NULL;
    s = hsa_queue_create(gpu, 1024, HSA_QUEUE_TYPE_MULTI, NULL, NULL, 0, 0, &queue);
    printf("hsa_queue_create: %d (id=%u)\n", s, s == 0 ? queue->id : 0);
    if (s != HSA_STATUS_SUCCESS) return 1;

    // Create a completion signal
    hsa_signal_t signal;
    s = hsa_signal_create(1, 0, NULL, &signal);
    printf("hsa_signal_create: %d\n", s);
    if (s != HSA_STATUS_SUCCESS) return 1;

    // Write a barrier-AND packet (simplest AQL packet type)
    // This completes when all dependent signals are satisfied (none in our case)
    uint64_t write_idx = hsa_queue_add_write_index_relaxed(queue, 1);
    uint64_t mask = queue->size - 1;
    hsa_barrier_and_packet_t *barrier =
        (hsa_barrier_and_packet_t*)((char*)queue->base_address +
        (write_idx & mask) * 64);

    // Zero the packet first
    memset(barrier, 0, sizeof(*barrier));

    // Set completion signal
    barrier->completion_signal = signal;

    // Set header LAST (marks packet as valid)
    // type = HSA_PACKET_TYPE_BARRIER_AND (4), barrier bit = 1, acquire = SYSTEM, release = SYSTEM
    uint16_t header = HSA_PACKET_TYPE_BARRIER_AND;
    header |= (1 << HSA_PACKET_HEADER_BARRIER);
    header |= (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE);
    header |= (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    __atomic_store_n((uint16_t*)barrier, header, __ATOMIC_RELEASE);

    // Ring the doorbell with the OLD write index (= last valid packet index).
    // ROCR convention: doorbell value = index of last valid packet.
    // ReleaseWriteIndex(new, count) stores (new - 1) internally.
    printf("Ringing doorbell (write_idx=%lu)...\n", write_idx);
    hsa_signal_store_relaxed(queue->doorbell_signal, write_idx);

    // Wait for completion using hsa_signal_wait (triggers bounce buffer processing)
    printf("Waiting for GPU (3 second timeout)...\n");
    hsa_signal_value_t val = hsa_signal_wait_scacquire(signal,
        HSA_SIGNAL_CONDITION_LT, 1, 3000000000ULL, HSA_WAIT_STATE_ACTIVE);
    if (val < 1) {
        printf("GPU completed barrier! signal=%ld\n", (long)val);
    } else {
        printf("TIMEOUT! signal=%ld (GPU did not respond)\n", (long)val);
        printf("Queue read_dispatch_id: %lu\n",
               hsa_queue_load_read_index_relaxed(queue));
        printf("Queue write_dispatch_id: %lu\n",
               hsa_queue_load_write_index_relaxed(queue));
    }

    hsa_signal_destroy(signal);
    hsa_queue_destroy(queue);
    hsa_shut_down();
    printf("Done.\n");
    return 0;
}
