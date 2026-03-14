#include <hsa/hsa.h>
#include <stdio.h>
#include <stdint.h>
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
    printf("=== RPTR & Signal Write Check ===\n");
    hsa_init();
    hsa_agent_t gpu = {0};
    hsa_iterate_agents(find_gpu, &gpu);

    hsa_queue_t *queue = NULL;
    hsa_queue_create(gpu, 1024, HSA_QUEUE_TYPE_SINGLE,
                     NULL, NULL, UINT32_MAX, UINT32_MAX, &queue);
    printf("Queue id=%lu, base=%p, size=%u\n", queue->id, queue->base_address, queue->size);

    hsa_signal_t signal;
    hsa_signal_create(1, 0, NULL, &signal);
    printf("Signal handle=0x%lx, initial value=%ld\n",
           signal.handle, hsa_signal_load_relaxed(signal));

    /* Enqueue barrier-AND */
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

    printf("Packet at %p, header=0x%04x\n", b, *(volatile uint16_t*)b);
    printf("  completion_signal in packet = 0x%lx\n", b->completion_signal.handle);

    /* Ring doorbell */
    uint64_t new_idx = hsa_queue_load_write_index_relaxed(queue);
    printf("Ringing doorbell with %lu\n", new_idx);
    hsa_signal_store_relaxed(queue->doorbell_signal, new_idx);

    /* Poll read_dispatch_id and signal for 5 seconds */
    for (int i = 0; i < 50; i++) {
        usleep(100000);  /* 100ms */
        uint64_t rdi = hsa_queue_load_read_index_relaxed(queue);
        int64_t sig = hsa_signal_load_relaxed(signal);
        if (i < 5 || i % 10 == 0 || rdi > 0 || sig != 1) {
            printf("  [%4dms] read_dispatch_id=%lu signal=%ld\n",
                   (i+1)*100, rdi, sig);
        }
        if (sig == 0) {
            printf("  SUCCESS! Barrier completed!\n");
            break;
        }
    }

    printf("\nFinal: read_dispatch_id=%lu write_dispatch_id=%lu signal=%ld\n",
           hsa_queue_load_read_index_relaxed(queue),
           hsa_queue_load_write_index_relaxed(queue),
           hsa_signal_load_relaxed(signal));

    hsa_signal_destroy(signal);
    hsa_queue_destroy(queue);
    hsa_shut_down();
    return 0;
}
