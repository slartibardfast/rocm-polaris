#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

/* From amd_hsa_signal.h — signal struct layout */
typedef struct {
    int64_t kind;
    union {
        volatile int64_t value;
        volatile uint64_t *hardware_doorbell_ptr;
    };
    /* ... more fields we don't need */
} amd_signal_t;

static hsa_status_t find_gpu(hsa_agent_t agent, void *data) {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_GPU) {
        *(hsa_agent_t *)data = agent;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

int main(int argc, char **argv) {
    int use_32bit = (argc > 1 && strcmp(argv[1], "--32") == 0);

    printf("=== HSA Doorbell Test (write=%d-bit) ===\n", use_32bit ? 32 : 64);

    hsa_status_t s = hsa_init();
    printf("hsa_init: %d\n", s);
    if (s != HSA_STATUS_SUCCESS) return 1;

    hsa_agent_t gpu = {0};
    hsa_iterate_agents(find_gpu, &gpu);
    if (!gpu.handle) { printf("No GPU\n"); return 1; }

    char name[64];
    hsa_agent_get_info(gpu, HSA_AGENT_INFO_NAME, name);
    printf("GPU: %s\n", name);

    /* Create queue */
    hsa_queue_t *queue = NULL;
    s = hsa_queue_create(gpu, 1024, HSA_QUEUE_TYPE_MULTI, NULL, NULL, 0, 0, &queue);
    printf("hsa_queue_create: %d (id=%u)\n", s, s == 0 ? queue->id : 0);
    if (s != HSA_STATUS_SUCCESS) return 1;

    /* Peek at the doorbell pointer */
    amd_signal_t *sig = (amd_signal_t *)(queue->doorbell_signal.handle);
    volatile uint64_t *db_ptr = sig->hardware_doorbell_ptr;
    printf("doorbell kind: %ld\n", sig->kind);
    printf("doorbell ptr: %p\n", (void *)db_ptr);

    /* Create completion signal */
    hsa_signal_t signal;
    s = hsa_signal_create(1, 0, NULL, &signal);
    if (s != HSA_STATUS_SUCCESS) { printf("signal_create failed\n"); return 1; }

    /* Write barrier-AND packet */
    uint64_t write_idx = hsa_queue_add_write_index_relaxed(queue, 1);
    uint64_t mask = queue->size - 1;
    hsa_barrier_and_packet_t *barrier =
        (hsa_barrier_and_packet_t *)((char *)queue->base_address +
                                     (write_idx & mask) * 64);

    memset(barrier, 0, sizeof(*barrier));
    barrier->completion_signal = signal;

    uint16_t header = HSA_PACKET_TYPE_BARRIER_AND;
    header |= (1 << HSA_PACKET_HEADER_BARRIER);
    header |= (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE);
    header |= (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    __atomic_store_n((uint16_t *)barrier, header, __ATOMIC_RELEASE);

    /* Ring doorbell — either 32-bit or 64-bit */
    printf("Ringing doorbell (write_idx=%lu, %d-bit write)...\n",
           write_idx, use_32bit ? 32 : 64);

    __asm__ volatile("sfence" ::: "memory");

    if (use_32bit) {
        volatile uint32_t *db32 = (volatile uint32_t *)db_ptr;
        *db32 = (uint32_t)write_idx;
    } else {
        *db_ptr = (uint64_t)write_idx;
    }

    /* Wait with timeout */
    printf("Waiting for GPU (3 second timeout)...\n");
    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);

    hsa_signal_value_t val;
    while (1) {
        val = hsa_signal_load_relaxed(signal);
        if (val == 0) {
            printf("SUCCESS! GPU completed barrier. signal=%ld\n", val);
            break;
        }
        clock_gettime(CLOCK_MONOTONIC, &now);
        double elapsed = (now.tv_sec - start.tv_sec) +
                         (now.tv_nsec - start.tv_nsec) / 1e9;
        if (elapsed > 3.0) {
            printf("TIMEOUT after %.1fs! signal=%ld\n", elapsed, val);
            printf("Queue read_dispatch_id: %lu\n",
                   hsa_queue_load_read_index_relaxed(queue));
            printf("Queue write_dispatch_id: %lu\n",
                   hsa_queue_load_write_index_relaxed(queue));
            break;
        }
    }

    hsa_signal_destroy(signal);
    hsa_queue_destroy(queue);
    hsa_shut_down();
    printf("Done.\n");
    return 0;
}
