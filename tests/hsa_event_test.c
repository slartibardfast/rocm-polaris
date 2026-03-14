// Test: check if EOP interrupt/event mechanism works
// even when atomic signal write fails
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
    printf("=== EOP Event/Mailbox Test ===\n");

    hsa_init();
    hsa_agent_t gpu = {0};
    hsa_iterate_agents(find_gpu, &gpu);

    // Create signal - check the event mailbox fields
    hsa_signal_t signal;
    hsa_signal_create(1, 0, NULL, &signal);

    // The signal handle points to amd_signal_t:
    // offset 0:  kind (8 bytes)
    // offset 8:  value (8 bytes)
    // offset 16: event_mailbox_ptr (8 bytes)
    // offset 24: event_id (4 bytes)
    uint64_t *sig = (uint64_t*)signal.handle;
    printf("Signal at %p:\n", (void*)signal.handle);
    printf("  kind             = %ld\n", (int64_t)sig[0]);
    printf("  value            = %ld\n", (int64_t)sig[1]);
    printf("  event_mailbox_ptr= 0x%lx\n", sig[2]);
    printf("  event_id         = %u\n", *(uint32_t*)&sig[3]);

    // Also try with hsa_amd_signal_create and InterruptSignal
    hsa_signal_t isignal;
    hsa_status_t s = hsa_amd_signal_create(1, 0, NULL, 0, &isignal);
    uint64_t *isig = (uint64_t*)isignal.handle;
    printf("\nInterrupt signal at %p:\n", (void*)isignal.handle);
    printf("  kind             = %ld\n", (int64_t)isig[0]);
    printf("  value            = %ld\n", (int64_t)isig[1]);
    printf("  event_mailbox_ptr= 0x%lx\n", isig[2]);
    printf("  event_id         = %u\n", *(uint32_t*)&isig[3]);

    // Create queue
    hsa_queue_t *queue = NULL;
    s = hsa_queue_create(gpu, 1024, HSA_QUEUE_TYPE_MULTI,
                         NULL, NULL, 0, 0, &queue);
    printf("\nQueue: status=%d id=%u\n", s, s==0 ? queue->id : 0);
    if (s != HSA_STATUS_SUCCESS) { _exit(1); }

    // Check signal AFTER queue creation (ROCR might update event fields)
    printf("\nAfter queue creation:\n");
    printf("  signal  event_mailbox=0x%lx event_id=%u\n", sig[2], *(uint32_t*)&sig[3]);
    printf("  isignal event_mailbox=0x%lx event_id=%u\n", isig[2], *(uint32_t*)&isig[3]);

    // Save mailbox ptr content before dispatch
    uint64_t *mailbox1 = NULL, *mailbox2 = NULL;
    if (sig[2]) {
        mailbox1 = (uint64_t*)sig[2];
        printf("  signal  mailbox content before: 0x%lx\n", *mailbox1);
    }
    if (isig[2]) {
        mailbox2 = (uint64_t*)isig[2];
        printf("  isignal mailbox content before: 0x%lx\n", *mailbox2);
    }

    // Dispatch barrier with the interrupt signal
    printf("\nDispatching barrier with interrupt signal...\n");
    uint64_t wi = hsa_queue_add_write_index_relaxed(queue, 1);
    hsa_barrier_and_packet_t *pkt =
        (hsa_barrier_and_packet_t*)((char*)queue->base_address +
        (wi & (queue->size-1)) * 64);
    memset(pkt, 0, sizeof(*pkt));
    pkt->completion_signal = isignal;

    uint16_t header = HSA_PACKET_TYPE_BARRIER_AND |
        (1 << HSA_PACKET_HEADER_BARRIER) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    __atomic_store_n((uint16_t*)pkt, header, __ATOMIC_RELEASE);
    hsa_signal_store_relaxed(queue->doorbell_signal,
        hsa_queue_load_write_index_relaxed(queue));

    // Wait a bit and check
    usleep(500000);  // 500ms

    printf("\nAfter dispatch (500ms):\n");
    printf("  isignal kind=%ld value=%ld\n", (int64_t)isig[0], (int64_t)isig[1]);
    printf("  isignal mailbox=0x%lx event_id=%u\n", isig[2], *(uint32_t*)&isig[3]);
    if (mailbox2) {
        printf("  isignal mailbox content after: 0x%lx\n", *mailbox2);
    }

    // Check raw memory around signal for any changes
    printf("\nFull signal memory dump:\n");
    for (int i = 0; i < 8; i++)
        printf("  [%d] = 0x%016lx\n", i, isig[i]);

    // Try using hsa_signal_wait with interrupt mode
    printf("\nTrying hsa_signal_wait_scacquire (2s timeout)...\n");
    hsa_signal_value_t val = hsa_signal_wait_scacquire(
        isignal, HSA_SIGNAL_CONDITION_LT, 1,
        2000000000ULL, HSA_WAIT_STATE_BLOCKED);
    printf("Wait returned: %ld\n", val);

    hsa_signal_destroy(signal);
    hsa_signal_destroy(isignal);
    hsa_queue_destroy(queue);
    hsa_shut_down();
    printf("Done.\n");
    return 0;
}
