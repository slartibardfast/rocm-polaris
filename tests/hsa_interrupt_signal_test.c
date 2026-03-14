// Test: use interrupt-based signal wait instead of memory polling
// The GPU interrupt fires regardless of PCIe atomics
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
    printf("=== Interrupt-Based Signal Test ===\n");

    hsa_init();
    hsa_agent_t gpu = {0};
    hsa_iterate_agents(find_gpu, &gpu);

    // Create signal with GPU as consumer
    hsa_signal_t signal;
    hsa_status_t s = hsa_signal_create(1, 1, &gpu, &signal);
    printf("Signal: handle=0x%lx, status=%d\n", signal.handle, s);

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

    printf("Dispatched. Using hsa_signal_wait_scacquire (interrupt-based)...\n");

    // Use the proper wait API which should use interrupts
    hsa_signal_value_t val = hsa_signal_wait_scacquire(
        signal,
        HSA_SIGNAL_CONDITION_LT,  // wait for value < 1
        1,
        3000000000ULL,  // 3 second timeout in ns
        HSA_WAIT_STATE_BLOCKED);  // use interrupt, not busy-wait

    printf("Wait returned: %ld\n", val);
    if (val == 0) {
        printf("PASS! Signal completed via interrupt!\n");
    } else {
        printf("TIMEOUT/FAIL: signal value=%ld\n", val);
        // Also check raw memory
        int64_t *sig_ptr = (int64_t*)signal.handle;
        printf("  Raw kind=%ld, value=%ld\n", sig_ptr[0], sig_ptr[1]);
    }

    hsa_signal_destroy(signal);
    hsa_queue_destroy(queue);
    hsa_shut_down();
    printf("Done.\n");
    return 0;
}
