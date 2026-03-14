// Hold an AQL queue open so we can inspect MQD state via debugfs
#include <hsa/hsa.h>
#include <stdio.h>
#include <unistd.h>

int main() {
    hsa_init();

    hsa_agent_t gpu = {0};
    hsa_status_t (*cb)(hsa_agent_t a, void *d) = ({
        hsa_status_t fn(hsa_agent_t a, void *d) {
            hsa_device_type_t t;
            hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
            if (t == HSA_DEVICE_TYPE_GPU) { *(hsa_agent_t*)d = a; return HSA_STATUS_INFO_BREAK; }
            return HSA_STATUS_SUCCESS;
        } fn;
    });
    hsa_iterate_agents(cb, &gpu);

    uint32_t qsize = 0;
    hsa_agent_get_info(gpu, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &qsize);

    hsa_queue_t *q;
    hsa_status_t s = hsa_queue_create(gpu, qsize, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX, UINT32_MAX, &q);
    printf("Queue created: status=%d, id=%lu\n", s, q->id);
    printf("Inspect with: sudo cat /sys/kernel/debug/kfd/hqds\n");
    printf("Sleeping 10s — inspect now...\n");
    fflush(stdout);
    sleep(10);

    hsa_queue_destroy(q);
    hsa_shut_down();
    return 0;
}
