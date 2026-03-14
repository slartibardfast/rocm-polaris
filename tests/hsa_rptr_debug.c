/* Debug tool: inspect the first 16 uint64_t values at rptr_gpu_buf_ location.
 * We can't directly read rptr_gpu_buf_ from outside ROCR, but we CAN read
 * the amd_queue_ structure which is exposed via the HSA queue handle.
 *
 * The amd_queue_ layout (from hsa_ext_amd.h) has:
 *   offset 0x00: hsa_queue_t (public part)
 *   offset 0x??: read_dispatch_id
 *   offset 0x??: write_dispatch_id
 *
 * We also use hsa_queue_load_read_index which calls UpdateReadDispatchId().
 */
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
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
    printf("=== RPTR Debug Tool ===\n");

    hsa_status_t s = hsa_init();
    if (s != HSA_STATUS_SUCCESS) { printf("hsa_init: %d\n", s); return 1; }

    hsa_agent_t gpu = {0};
    hsa_iterate_agents(find_gpu, &gpu);
    if (!gpu.handle) { printf("No GPU\n"); return 1; }

    char name[64];
    hsa_agent_get_info(gpu, HSA_AGENT_INFO_NAME, name);
    printf("GPU: %s\n", name);

    hsa_queue_t *queue = NULL;
    s = hsa_queue_create(gpu, 1024, HSA_QUEUE_TYPE_MULTI, NULL, NULL, 0, 0, &queue);
    printf("hsa_queue_create: %d\n", s);
    if (s != HSA_STATUS_SUCCESS) return 1;

    /* The amd_queue_ struct starts at the container of queue.
     * read_dispatch_id offset is stored at
     * amd_queue_.read_dispatch_id_field_base_byte_offset which is
     * at a fixed offset in the AMD extension structure.
     * For now, use the HSA API to read it. */

    printf("\n--- Before dispatch ---\n");
    printf("read_dispatch_id (via API): %lu\n",
           (unsigned long)hsa_queue_load_read_index_relaxed(queue));
    printf("write_dispatch_id (via API): %lu\n",
           (unsigned long)hsa_queue_load_write_index_relaxed(queue));

    /* Submit a barrier-AND packet */
    hsa_signal_t signal;
    hsa_signal_create(1, 0, NULL, &signal);

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

    printf("\nRinging doorbell (write_idx=%lu)...\n",
           (unsigned long)hsa_queue_load_write_index_relaxed(queue));
    hsa_signal_store_relaxed(queue->doorbell_signal,
                             hsa_queue_load_write_index_relaxed(queue));

    /* Wait a bit for CP to process */
    printf("Sleeping 2s for CP to process...\n");
    sleep(2);

    printf("\n--- After dispatch + 2s ---\n");
    printf("read_dispatch_id (via API): %lu\n",
           (unsigned long)hsa_queue_load_read_index_relaxed(queue));
    printf("write_dispatch_id (via API): %lu\n",
           (unsigned long)hsa_queue_load_write_index_relaxed(queue));
    printf("signal value: %ld\n",
           (long)hsa_signal_load_relaxed(signal));

    /* Also try to read the raw amd_queue_ memory around read_dispatch_id.
     * In the AMD AQL queue struct, read_dispatch_id is at a known offset.
     * The queue pointer IS the hsa_queue_t at the START of amd_queue_t.
     * amd_queue_t has read_dispatch_id after hsa_queue_t fields.
     * From hsa_ext_amd.h:
     *   typedef struct hsa_amd_queue_s {
     *     hsa_queue_t hsa_queue;          // offset 0, size 0x40
     *     ...
     *     uint64_t write_dispatch_id;     // offset 0x40
     *     ...
     *     uint64_t read_dispatch_id;      // offset 0x80 or nearby
     *   }
     * The exact offset depends on the struct layout. Let's dump around it.
     */
    uint64_t *raw = (uint64_t*)queue;
    printf("\n--- Raw amd_queue_ dump (first 32 uint64_t) ---\n");
    for (int i = 0; i < 32; i++) {
        printf("  [%3d] offset 0x%03x: 0x%016lx\n", i, i*8, (unsigned long)raw[i]);
    }

    hsa_signal_destroy(signal);
    hsa_queue_destroy(queue);
    hsa_shut_down();
    printf("Done.\n");
    return 0;
}
