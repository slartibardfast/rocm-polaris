// Diagnostic: dump signal handle address and check GPU VA mapping
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
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

static hsa_status_t find_cpu(hsa_agent_t agent, void *data) {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_CPU) {
        *(hsa_agent_t*)data = agent;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

static hsa_status_t find_region(hsa_region_t region, void *data) {
    hsa_region_global_flag_t flags;
    hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
    if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
        *(hsa_region_t*)data = region;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

int main() {
    printf("=== HSA Signal Address Diagnostic ===\n");

    hsa_status_t s = hsa_init();
    if (s != HSA_STATUS_SUCCESS) { printf("hsa_init failed: %d\n", s); return 1; }

    hsa_agent_t gpu = {0}, cpu = {0};
    hsa_iterate_agents(find_gpu, &gpu);
    hsa_iterate_agents(find_cpu, &cpu);
    if (!gpu.handle) { printf("No GPU\n"); return 1; }

    // Create signal and dump its handle
    hsa_signal_t signal;
    s = hsa_signal_create(1, 0, NULL, &signal);
    printf("Signal handle: 0x%lx\n", signal.handle);
    printf("Signal value:  %ld\n", hsa_signal_load_relaxed(signal));

    // The handle IS the CPU address of amd_signal_t.
    // amd_signal_t layout: kind (8 bytes) + value (8 bytes)
    uint64_t *sig_ptr = (uint64_t*)signal.handle;
    printf("Signal kind:   0x%lx (at %p)\n", sig_ptr[0], &sig_ptr[0]);
    printf("Signal value:  %ld (at %p)\n", (int64_t)sig_ptr[1], &sig_ptr[1]);

    // Check what /proc/self/maps says about this region
    printf("\n--- Memory mapping for signal region ---\n");
    char cmd[256];
    snprintf(cmd, sizeof(cmd),
        "grep '%lx' /proc/%d/maps || echo 'Not found in maps'",
        (unsigned long)signal.handle & ~0xFFFUL, getpid());
    system(cmd);

    // Also dump all GPU-mapped regions
    printf("\n--- All KFD/GPU mappings (look for kfd) ---\n");
    snprintf(cmd, sizeof(cmd), "cat /proc/%d/maps | grep -i 'kfd\\|gpu\\|amd\\|drm' || echo 'No KFD maps found'", getpid());
    system(cmd);

    // Check /proc/self/maps for the full layout
    printf("\n--- Full process memory map ---\n");
    snprintf(cmd, sizeof(cmd), "cat /proc/%d/maps", getpid());
    system(cmd);

    // Create queue and dispatch barrier
    hsa_queue_t *queue = NULL;
    s = hsa_queue_create(gpu, 1024, HSA_QUEUE_TYPE_MULTI, NULL, NULL, 0, 0, &queue);
    printf("\n=== Queue created: status=%d, id=%u ===\n", s, s == 0 ? queue->id : 0);
    if (s != HSA_STATUS_SUCCESS) goto done;

    printf("Queue base_address: %p\n", queue->base_address);
    printf("Queue doorbell:     0x%lx\n", queue->doorbell_signal.handle);

    // Write barrier packet
    uint64_t write_idx = hsa_queue_add_write_index_relaxed(queue, 1);
    uint64_t mask = queue->size - 1;
    hsa_barrier_and_packet_t *barrier =
        (hsa_barrier_and_packet_t*)((char*)queue->base_address +
        (write_idx & mask) * 64);

    memset(barrier, 0, sizeof(*barrier));
    barrier->completion_signal = signal;

    // Dump the raw packet bytes to see what completion_signal looks like
    printf("\nPacket at %p:\n", barrier);
    uint64_t *pkt = (uint64_t*)barrier;
    for (int i = 0; i < 8; i++)
        printf("  dw[%d]: 0x%016lx\n", i, pkt[i]);

    // The completion_signal in the packet should be at offset 48 (dw[6])
    printf("completion_signal in packet: 0x%lx\n", pkt[6]);
    printf("This should match signal.handle: 0x%lx\n", signal.handle);

    // Set header
    uint16_t header = HSA_PACKET_TYPE_BARRIER_AND;
    header |= (1 << HSA_PACKET_HEADER_BARRIER);
    header |= (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE);
    header |= (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    __atomic_store_n((uint16_t*)barrier, header, __ATOMIC_RELEASE);

    // Ring doorbell
    uint64_t new_write_idx = hsa_queue_load_write_index_relaxed(queue);
    hsa_signal_store_relaxed(queue->doorbell_signal, new_write_idx);

    printf("\nWaiting 3s for signal (polling raw memory)...\n");
    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);
    while (1) {
        int64_t val = __atomic_load_n(&sig_ptr[1], __ATOMIC_ACQUIRE);
        if (val == 0) {
            printf("Signal decremented! value=%ld\n", val);
            break;
        }
        clock_gettime(CLOCK_MONOTONIC, &now);
        double elapsed = (now.tv_sec - start.tv_sec) + (now.tv_nsec - start.tv_nsec) / 1e9;
        if (elapsed > 3.0) {
            printf("TIMEOUT! Signal value still %ld\n", val);
            printf("Signal kind: 0x%lx\n", sig_ptr[0]);
            printf("read_dispatch_id: %lu\n", hsa_queue_load_read_index_relaxed(queue));
            break;
        }
    }

    hsa_queue_destroy(queue);
done:
    hsa_signal_destroy(signal);
    hsa_shut_down();
    printf("Done.\n");
    return 0;
}
