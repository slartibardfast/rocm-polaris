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
    setbuf(stdout, NULL); // unbuffered
    printf("=== Signal Test ===\n");

    hsa_init();
    hsa_agent_t gpu = {0};
    hsa_iterate_agents(find_gpu, &gpu);
    if (!gpu.handle) { printf("No GPU\n"); return 1; }

    hsa_signal_t signal;
    hsa_signal_create(1, 0, NULL, &signal);
    printf("Signal: 0x%lx\n", signal.handle);

    uint64_t *sig_ptr = (uint64_t*)signal.handle;
    printf("  kind=%ld value=%ld at %p/%p\n", sig_ptr[0], (int64_t)sig_ptr[1], &sig_ptr[0], &sig_ptr[1]);

    hsa_queue_t *queue = NULL;
    hsa_status_t s = hsa_queue_create(gpu, 1024, HSA_QUEUE_TYPE_MULTI,
                                       NULL, NULL, 0, 0, &queue);
    printf("Queue: status=%d id=%u base=%p\n", s, s==0 ? queue->id : 0, s==0 ? queue->base_address : NULL);
    if (s != HSA_STATUS_SUCCESS) { printf("QUEUE FAILED\n"); _exit(1); }

    // Dispatch barrier
    uint64_t wi = hsa_queue_add_write_index_relaxed(queue, 1);
    hsa_barrier_and_packet_t *pkt =
        (hsa_barrier_and_packet_t*)((char*)queue->base_address + (wi & (queue->size-1)) * 64);
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

    struct timespec ts_start, ts_now;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    while (1) {
        int64_t val = __atomic_load_n(&sig_ptr[1], __ATOMIC_ACQUIRE);
        if (val == 0) { printf("PASS! signal=%ld\n", val); break; }
        clock_gettime(CLOCK_MONOTONIC, &ts_now);
        if ((ts_now.tv_sec - ts_start.tv_sec) > 3) {
            printf("TIMEOUT signal=%ld kind=%ld\n", val, sig_ptr[0]);
            // Also check raw memory around signal for any writes
            printf("Signal memory dump:\n");
            for (int i = -2; i < 10; i++)
                printf("  [%+d] = 0x%016lx\n", i, sig_ptr[i]);
            break;
        }
    }

    printf("Cleaning up... ");
    // Use _exit to avoid crashes in destructors
    hsa_signal_destroy(signal);
    printf("signal destroyed. ");
    hsa_queue_destroy(queue);
    printf("queue destroyed. ");
    hsa_shut_down();
    printf("done.\n");
    return 0;
}
