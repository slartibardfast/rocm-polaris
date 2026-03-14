// Test: create signal with GPU-local memory to avoid PCIe atomics
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

static hsa_agent_t g_gpu = {0}, g_cpu = {0};

static hsa_status_t find_agent(hsa_agent_t agent, void *data) {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_GPU && !g_gpu.handle) g_gpu = agent;
    if (type == HSA_DEVICE_TYPE_CPU && !g_cpu.handle) g_cpu = agent;
    return HSA_STATUS_SUCCESS;
}

static hsa_status_t find_vram_pool(hsa_amd_memory_pool_t pool, void *data) {
    hsa_amd_segment_t seg;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
    if (seg == HSA_AMD_SEGMENT_GLOBAL) {
        uint32_t flags;
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
        // Coarse-grained VRAM (not the "uncacheable" pool)
        if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) {
            *(hsa_amd_memory_pool_t*)data = pool;
            return HSA_STATUS_INFO_BREAK;
        }
    }
    return HSA_STATUS_SUCCESS;
}

int main() {
    setbuf(stdout, NULL);
    printf("=== GPU-Local Signal Test ===\n");

    hsa_init();
    hsa_iterate_agents(find_agent, NULL);
    if (!g_gpu.handle) { printf("No GPU\n"); return 1; }

    // Find VRAM pool
    hsa_amd_memory_pool_t vram_pool = {0};
    hsa_amd_agent_iterate_memory_pools(g_gpu, find_vram_pool, &vram_pool);
    if (!vram_pool.handle) { printf("No VRAM pool\n"); return 1; }

    // Allocate 4K in VRAM
    void *vram_buf = NULL;
    hsa_status_t s = hsa_amd_memory_pool_allocate(vram_pool, 4096, 0, &vram_buf);
    printf("VRAM alloc: status=%d, addr=%p\n", s, vram_buf);
    if (s != HSA_STATUS_SUCCESS) { printf("VRAM alloc failed\n"); return 1; }

    // Allow CPU access so we can read/write the signal value
    s = hsa_amd_agents_allow_access(1, &g_cpu, NULL, vram_buf);
    printf("Allow CPU access: %d\n", s);

    // Set up fake amd_signal_t at the start of the buffer
    // Layout: kind (8 bytes) | value (8 bytes) | event_mailbox_ptr (8) | event_id (4) | ...
    volatile int64_t *sig_mem = (volatile int64_t*)vram_buf;
    sig_mem[0] = 1;   // kind = AMD_SIGNAL_KIND_USER
    sig_mem[1] = 1;   // value = 1
    sig_mem[2] = 0;   // event_mailbox_ptr
    printf("VRAM signal: kind=%ld, value=%ld at %p\n", sig_mem[0], sig_mem[1], vram_buf);

    // Create fake hsa_signal_t pointing to VRAM
    hsa_signal_t signal;
    signal.handle = (uint64_t)vram_buf;

    // Create queue
    hsa_queue_t *queue = NULL;
    s = hsa_queue_create(g_gpu, 1024, HSA_QUEUE_TYPE_MULTI,
                         NULL, NULL, 0, 0, &queue);
    printf("Queue: status=%d id=%u base=%p\n", s, s==0 ? queue->id : 0,
           s==0 ? queue->base_address : NULL);
    if (s != HSA_STATUS_SUCCESS) { _exit(1); }

    // Dispatch barrier with VRAM signal
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

    printf("Dispatched barrier to VRAM signal, waiting 3s...\n");
    struct timespec ts_start, ts_now;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    while (1) {
        int64_t val = sig_mem[1];
        if (val == 0) {
            printf("PASS! Signal decremented to %ld!\n", val);
            printf("kind=%ld\n", sig_mem[0]);
            break;
        }
        clock_gettime(CLOCK_MONOTONIC, &ts_now);
        if ((ts_now.tv_sec - ts_start.tv_sec) > 3) {
            printf("TIMEOUT signal=%ld\n", val);
            printf("Memory dump:\n");
            for (int i = 0; i < 8; i++)
                printf("  [%d] = 0x%016lx\n", i, (uint64_t)sig_mem[i]);
            break;
        }
    }

    hsa_queue_destroy(queue);
    hsa_amd_memory_pool_free(vram_buf);
    hsa_shut_down();
    printf("Done.\n");
    return 0;
}
