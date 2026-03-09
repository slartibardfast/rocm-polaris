#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {
    int64_t kind;
    union {
        volatile int64_t value;
        volatile uint64_t *hardware_doorbell_ptr;
    };
    uint64_t event_mailbox_ptr;
    uint32_t event_id;
    uint32_t reserved1;
    uint64_t start_ts;
    uint64_t end_ts;
    union {
        void *queue_ptr;
        uint64_t reserved2;
    };
    uint32_t reserved3[2];
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

int main() {
    printf("=== HSA Doorbell Debug ===\n");

    hsa_status_t s = hsa_init();
    if (s != HSA_STATUS_SUCCESS) { printf("hsa_init failed: %d\n", s); return 1; }

    hsa_agent_t gpu = {0};
    hsa_iterate_agents(find_gpu, &gpu);
    if (!gpu.handle) { printf("No GPU\n"); return 1; }

    char name[64];
    hsa_agent_get_info(gpu, HSA_AGENT_INFO_NAME, name);
    printf("GPU: %s\n", name);

    /* Create two queues to compare doorbell addresses */
    hsa_queue_t *q1 = NULL, *q2 = NULL;
    s = hsa_queue_create(gpu, 1024, HSA_QUEUE_TYPE_MULTI, NULL, NULL, 0, 0, &q1);
    printf("Queue 1: status=%d id=%u\n", s, s == 0 ? q1->id : 0);

    s = hsa_queue_create(gpu, 1024, HSA_QUEUE_TYPE_MULTI, NULL, NULL, 0, 0, &q2);
    printf("Queue 2: status=%d id=%u\n", s, s == 0 ? q2->id : 0);

    if (q1) {
        amd_signal_t *sig1 = (amd_signal_t *)(q1->doorbell_signal.handle);
        printf("\nQueue 1 doorbell:\n");
        printf("  signal handle: 0x%lx\n", q1->doorbell_signal.handle);
        printf("  kind: %ld\n", sig1->kind);
        printf("  hardware_doorbell_ptr: %p\n", (void *)sig1->hardware_doorbell_ptr);
        printf("  event_mailbox_ptr: 0x%lx\n", sig1->event_mailbox_ptr);
        printf("  event_id: %u\n", sig1->event_id);
        printf("  queue base: %p\n", q1->base_address);
        printf("  queue size: %u\n", q1->size);
    }

    if (q2) {
        amd_signal_t *sig2 = (amd_signal_t *)(q2->doorbell_signal.handle);
        printf("\nQueue 2 doorbell:\n");
        printf("  signal handle: 0x%lx\n", q2->doorbell_signal.handle);
        printf("  kind: %ld\n", sig2->kind);
        printf("  hardware_doorbell_ptr: %p\n", (void *)sig2->hardware_doorbell_ptr);
        printf("  event_mailbox_ptr: 0x%lx\n", sig2->event_mailbox_ptr);
        printf("  event_id: %u\n", sig2->event_id);
    }

    /* Dump /proc/self/maps around the doorbell address */
    if (q1) {
        amd_signal_t *sig1 = (amd_signal_t *)(q1->doorbell_signal.handle);
        uintptr_t db_addr = (uintptr_t)sig1->hardware_doorbell_ptr;
        uintptr_t db_page = db_addr & ~0xFFFUL;

        printf("\n--- /proc/self/maps entries near doorbell (0x%lx) ---\n", db_addr);
        FILE *f = fopen("/proc/self/maps", "r");
        if (f) {
            char line[512];
            while (fgets(line, sizeof(line), f)) {
                unsigned long start, end;
                if (sscanf(line, "%lx-%lx", &start, &end) == 2) {
                    /* Show entries containing or near the doorbell */
                    if (start <= db_addr + 0x10000 && end >= (db_page > 0x10000 ? db_page - 0x10000 : 0)) {
                        printf("  %s", line);
                    }
                }
            }
            fclose(f);
        }
    }

    /* Also show KFD topology info */
    printf("\n--- KFD queue id to doorbell stride ---\n");
    if (q1 && q2) {
        amd_signal_t *sig1 = (amd_signal_t *)(q1->doorbell_signal.handle);
        amd_signal_t *sig2 = (amd_signal_t *)(q2->doorbell_signal.handle);
        uintptr_t d1 = (uintptr_t)sig1->hardware_doorbell_ptr;
        uintptr_t d2 = (uintptr_t)sig2->hardware_doorbell_ptr;
        printf("  queue1 id=%u doorbell=0x%lx\n", q1->id, d1);
        printf("  queue2 id=%u doorbell=0x%lx\n", q2->id, d2);
        printf("  stride: %ld bytes (q2 - q1)\n", (long)(d2 - d1));
        printf("  expected for gfx8 (4-byte): %d bytes\n",
               (int)(q2->id - q1->id) * 4);
    }

    if (q2) hsa_queue_destroy(q2);
    if (q1) hsa_queue_destroy(q1);
    hsa_shut_down();
    printf("\nDone.\n");
    return 0;
}
