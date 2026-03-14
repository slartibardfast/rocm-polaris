// Keep HSA context alive so we can inspect GPUVM state
#include <hsa/hsa.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
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
    hsa_init();
    hsa_agent_t gpu = {0};
    hsa_iterate_agents(find_gpu, &gpu);

    hsa_signal_t signal;
    hsa_signal_create(1, 0, NULL, &signal);

    hsa_queue_t *queue = NULL;
    hsa_queue_create(gpu, 1024, HSA_QUEUE_TYPE_MULTI, NULL, NULL, 0, 0, &queue);

    uint64_t *sig_ptr = (uint64_t*)signal.handle;
    printf("PID: %d\n", getpid());
    printf("Signal handle: 0x%lx (kind=%ld, value=%ld)\n", signal.handle, sig_ptr[0], (int64_t)sig_ptr[1]);
    printf("Queue base: %p, doorbell: 0x%lx\n", queue->base_address, queue->doorbell_signal.handle);
    printf("\nInspect with:\n");
    printf("  sudo cat /sys/kernel/debug/kfd/proc/%d/vm\n", getpid());
    printf("  sudo cat /sys/kernel/debug/dri/1/amdgpu_vm_info\n");
    printf("  sudo cat /sys/kernel/debug/kfd/hqds\n");
    printf("\nSleeping 15s...\n");
    fflush(stdout);
    sleep(15);

    hsa_queue_destroy(queue);
    hsa_signal_destroy(signal);
    hsa_shut_down();
    return 0;
}
