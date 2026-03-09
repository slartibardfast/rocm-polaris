#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

/* amd_queue_t layout (from amd_hsa_queue.h) */
typedef struct {
    uint32_t hsa_queue_[8];  /* hsa_queue_t: 64 bytes but we need specific offsets */
} hsa_queue_raw_t;

static hsa_status_t find_gpu(hsa_agent_t agent, void *data) {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_GPU) {
        *(hsa_agent_t *)data = agent;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

int main() {
    printf("=== HSA Write Pointer Debug ===\n");

    hsa_status_t s = hsa_init();
    if (s != HSA_STATUS_SUCCESS) return 1;

    hsa_agent_t gpu = {0};
    hsa_iterate_agents(find_gpu, &gpu);
    if (!gpu.handle) return 1;

    hsa_queue_t *queue = NULL;
    s = hsa_queue_create(gpu, 1024, HSA_QUEUE_TYPE_MULTI, NULL, NULL, 0, 0, &queue);
    if (s != HSA_STATUS_SUCCESS) return 1;

    /* hsa_queue_t public fields */
    printf("queue->base_address: %p\n", queue->base_address);
    printf("queue->size: %u\n", queue->size);
    printf("queue->id: %u\n", queue->id);

    /* The write_dispatch_id and read_dispatch_id are at known offsets
     * in the amd_queue_v2_t structure. The public hsa_queue_t is at the
     * beginning. After it come AMD-specific fields.
     *
     * hsa_queue_t is 40 bytes. Then:
     *   offset 40: reserved (8 bytes)
     *   offset 48: write_dispatch_id (8 bytes)
     *   offset 56: group_segment_aperture_base_hi (4 bytes)
     *   offset 60: private_segment_aperture_base_hi (4 bytes)
     *   offset 64: max_cu_id (4 bytes)
     *   offset 68: max_wave_id (4 bytes)
     *   offset 72: max_legacy_doorbell_dispatch_id_plus_1 (8 bytes)
     *   offset 80: legacy_doorbell_lock (4 bytes)
     *   offset 84: reserved2 (12 bytes)
     *   offset 96: read_dispatch_id (8 bytes)
     *
     * Actually, let me just dump raw bytes around the queue struct.
     */

    uint8_t *qbase = (uint8_t *)queue;
    printf("\nRaw queue memory (first 128 bytes):\n");
    for (int i = 0; i < 128; i += 8) {
        uint64_t val = *(uint64_t *)(qbase + i);
        printf("  [%3d] 0x%016lx", i, val);
        if (i == 0) printf("  (type/features/base_address)");
        if (i == 16) printf("  (base_address)");
        if (i == 24) printf("  (doorbell_signal)");
        if (i == 32) printf("  (size/reserved/id)");
        printf("\n");
    }

    /* The write_dispatch_id address that KFD/ROCR gives to the CP */
    printf("\nhsa_queue_load_write_index: %lu\n",
           hsa_queue_load_write_index_relaxed(queue));
    printf("hsa_queue_load_read_index: %lu\n",
           hsa_queue_load_read_index_relaxed(queue));

    /* Check if write_dispatch_id is at offset 48 from queue start */
    uint64_t *write_dispatch_ptr = (uint64_t *)(qbase + 48);
    printf("\n*(queue + 48) [candidate write_dispatch_id]: %lu\n", *write_dispatch_ptr);

    /* After adding a write index, check again */
    hsa_queue_add_write_index_relaxed(queue, 1);
    printf("After add_write_index(1): %lu\n", *write_dispatch_ptr);
    printf("hsa_queue_load_write_index: %lu\n",
           hsa_queue_load_write_index_relaxed(queue));

    /* Print the virtual address of write_dispatch_id */
    printf("\nVirtual address of write_dispatch_id: %p\n", (void *)write_dispatch_ptr);
    printf("Expected in MQD wptr_addr: should map to this GPU VA\n");

    hsa_queue_destroy(queue);
    hsa_shut_down();
    return 0;
}
