#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
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
    printf("=== WPTR Debug Test ===\n");

    hsa_init();
    hsa_agent_t gpu = {0};
    hsa_iterate_agents(find_gpu, &gpu);

    hsa_queue_t *queue = NULL;
    hsa_status_t s = hsa_queue_create(gpu, 1024, HSA_QUEUE_TYPE_SINGLE,
                                       NULL, NULL, UINT32_MAX, UINT32_MAX, &queue);
    printf("Queue created: status=%d\n", s);
    if (s != HSA_STATUS_SUCCESS) return 1;

    printf("queue ptr          = %p\n", (void*)queue);
    printf("queue->base_address= %p\n", queue->base_address);
    printf("queue->size        = %u\n", queue->size);
    printf("queue->id          = %lu\n", queue->id);
    printf("queue->doorbell_signal.handle = 0x%lx\n", queue->doorbell_signal.handle);

    /* Read the AMD-specific queue fields.
     * write_dispatch_id is at a known offset in the AMD queue structure.
     * Probe to find it by checking hsa_queue_load_write_index_relaxed.
     */
    printf("\nProbing write_dispatch_id location...\n");
    volatile uint64_t *base = (volatile uint64_t*)queue;
    uint64_t cur_wdi = hsa_queue_load_write_index_relaxed(queue);
    for (int i = 0; i < 32; i++) {
        if (base[i] == cur_wdi && cur_wdi == 0) {
            /* Write a unique value to find the right one */
        }
        printf("  offset 0x%02x: 0x%016lx", i*8, base[i]);
        if (i*8 == 0x40) printf("  <-- typical write_dispatch_id offset");
        if (i*8 == 0x48) printf("  <-- typical read_dispatch_id offset");
        printf("\n");
    }

    /* Use the standard API accessors */
    printf("\nhsa_queue write_idx = %lu\n", hsa_queue_load_write_index_relaxed(queue));
    printf("hsa_queue read_idx  = %lu\n", hsa_queue_load_read_index_relaxed(queue));

    /* Create a completion signal */
    hsa_signal_t signal;
    hsa_signal_create(1, 0, NULL, &signal);

    /* Write a barrier-AND packet */
    uint64_t write_idx = hsa_queue_add_write_index_relaxed(queue, 1);
    uint64_t mask = queue->size - 1;
    hsa_barrier_and_packet_t *barrier =
        (hsa_barrier_and_packet_t*)((char*)queue->base_address +
        (write_idx & mask) * 64);
    memset(barrier, 0, sizeof(*barrier));
    barrier->completion_signal = signal;

    uint16_t header = HSA_PACKET_TYPE_BARRIER_AND;
    header |= (1 << HSA_PACKET_HEADER_BARRIER);
    header |= (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE);
    header |= (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    __atomic_store_n((uint16_t*)barrier, header, __ATOMIC_RELEASE);

    printf("\nAfter enqueue, before doorbell:\n");
    printf("  hsa write_idx = %lu\n", hsa_queue_load_write_index_relaxed(queue));

    /* Ring doorbell with NEW write index */
    uint64_t new_idx = hsa_queue_load_write_index_relaxed(queue);
    printf("Ringing doorbell with value %lu...\n", new_idx);
    hsa_signal_store_relaxed(queue->doorbell_signal, new_idx);

    printf("Sleeping 10s — dump HQDs now...\n");
    fflush(stdout);
    sleep(10);

    printf("\nAfter wait:\n");
    printf("  hsa write_idx = %lu\n", hsa_queue_load_write_index_relaxed(queue));
    printf("  hsa read_idx  = %lu\n", hsa_queue_load_read_index_relaxed(queue));
    printf("  signal value  = %ld\n", hsa_signal_load_relaxed(signal));

    hsa_signal_destroy(signal);
    hsa_queue_destroy(queue);
    hsa_shut_down();
    printf("Done.\n");
    return 0;
}
