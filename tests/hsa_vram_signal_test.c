// Test: allocate signal in VRAM pool instead of default system memory
// and see if barrier completion signal write works
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

static hsa_agent_t g_gpu = {0};

static hsa_status_t find_gpu(hsa_agent_t agent, void *data) {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_GPU) {
        *(hsa_agent_t*)data = agent;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

// Find a GPU-local (VRAM) memory pool
static hsa_status_t find_vram_pool(hsa_amd_memory_pool_t pool, void *data) {
    hsa_amd_segment_t segment;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
    if (segment == HSA_AMD_SEGMENT_GLOBAL) {
        uint32_t flags;
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
        if (!(flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)) {
            // Coarse-grained VRAM pool
            *(hsa_amd_memory_pool_t*)data = pool;
            return HSA_STATUS_INFO_BREAK;
        }
    }
    return HSA_STATUS_SUCCESS;
}

// Find fine-grained system memory pool
static hsa_status_t find_system_pool(hsa_amd_memory_pool_t pool, void *data) {
    hsa_amd_segment_t segment;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
    if (segment == HSA_AMD_SEGMENT_GLOBAL) {
        uint32_t flags;
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
        if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) {
            *(hsa_amd_memory_pool_t*)data = pool;
            return HSA_STATUS_INFO_BREAK;
        }
    }
    return HSA_STATUS_SUCCESS;
}

int main() {
    printf("=== HSA Signal Memory Location Test ===\n");

    hsa_init();
    hsa_iterate_agents(find_gpu, &g_gpu);
    if (!g_gpu.handle) { printf("No GPU\n"); return 1; }

    // Find memory pools
    hsa_amd_memory_pool_t vram_pool = {0}, sys_pool = {0};
    hsa_amd_agent_iterate_memory_pools(g_gpu, find_vram_pool, &vram_pool);
    hsa_amd_agent_iterate_memory_pools(g_gpu, find_system_pool, &sys_pool);

    printf("VRAM pool: 0x%lx\n", vram_pool.handle);
    printf("System pool: 0x%lx\n", sys_pool.handle);

    // Create default signal and check its address
    hsa_signal_t signal;
    hsa_signal_create(1, 0, NULL, &signal);
    printf("\nDefault signal handle: 0x%lx\n", signal.handle);
    printf("  kind=%ld, value=%ld\n",
        *(int64_t*)signal.handle, *((int64_t*)signal.handle + 1));

    // Create queue
    hsa_queue_t *queue = NULL;
    hsa_status_t s = hsa_queue_create(g_gpu, 1024, HSA_QUEUE_TYPE_MULTI,
                                       NULL, NULL, 0, 0, &queue);
    if (s != HSA_STATUS_SUCCESS) { printf("Queue create failed: %d\n", s); return 1; }
    printf("Queue created: id=%u, base=%p\n", queue->id, queue->base_address);

    // Test 1: Barrier with default signal
    printf("\n--- Test 1: Default signal ---\n");
    {
        uint64_t write_idx = hsa_queue_add_write_index_relaxed(queue, 1);
        uint64_t mask = queue->size - 1;
        hsa_barrier_and_packet_t *pkt =
            (hsa_barrier_and_packet_t*)((char*)queue->base_address +
            (write_idx & mask) * 64);
        memset(pkt, 0, sizeof(*pkt));
        pkt->completion_signal = signal;

        uint16_t header = HSA_PACKET_TYPE_BARRIER_AND |
            (1 << HSA_PACKET_HEADER_BARRIER) |
            (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
            (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
        __atomic_store_n((uint16_t*)pkt, header, __ATOMIC_RELEASE);
        hsa_signal_store_relaxed(queue->doorbell_signal,
            hsa_queue_load_write_index_relaxed(queue));

        struct timespec ts_start, ts_now;
        clock_gettime(CLOCK_MONOTONIC, &ts_start);
        int64_t val;
        while (1) {
            val = hsa_signal_load_relaxed(signal);
            if (val == 0) { printf("  PASS! signal decremented\n"); break; }
            clock_gettime(CLOCK_MONOTONIC, &ts_now);
            if ((ts_now.tv_sec - ts_start.tv_sec) > 2) {
                printf("  FAIL! signal=%ld (timeout)\n", val);
                break;
            }
        }
    }

    // Test 2: Try with a manually-allocated signal in VRAM
    // Allocate 128 bytes in VRAM for a fake amd_signal_t
    if (vram_pool.handle) {
        printf("\n--- Test 2: Manual VRAM signal ---\n");
        void *vram_sig = NULL;
        s = hsa_amd_memory_pool_allocate(vram_pool, 4096, 0, &vram_sig);
        if (s != HSA_STATUS_SUCCESS) {
            printf("  VRAM alloc failed: %d\n", s);
        } else {
            printf("  VRAM signal buffer at: %p\n", vram_sig);
            // Allow CPU access
            hsa_agent_t agents[1] = {g_gpu};
            s = hsa_amd_agents_allow_access(1, agents, NULL, vram_sig);
            printf("  Allow GPU access: %d\n", s);

            // Try to set up a fake amd_signal_t
            // kind=1 (USER), value=1
            int64_t *sig_mem = (int64_t*)vram_sig;
            sig_mem[0] = 1;  // kind = AMD_SIGNAL_KIND_USER
            sig_mem[1] = 1;  // value = 1
            printf("  Set kind=%ld, value=%ld at %p\n",
                sig_mem[0], sig_mem[1], vram_sig);

            // Create a fake hsa_signal_t with this address
            hsa_signal_t fake_sig;
            fake_sig.handle = (uint64_t)vram_sig;

            uint64_t write_idx = hsa_queue_add_write_index_relaxed(queue, 1);
            uint64_t mask = queue->size - 1;
            hsa_barrier_and_packet_t *pkt =
                (hsa_barrier_and_packet_t*)((char*)queue->base_address +
                (write_idx & mask) * 64);
            memset(pkt, 0, sizeof(*pkt));
            pkt->completion_signal = fake_sig;

            uint16_t header = HSA_PACKET_TYPE_BARRIER_AND |
                (1 << HSA_PACKET_HEADER_BARRIER) |
                (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
            __atomic_store_n((uint16_t*)pkt, header, __ATOMIC_RELEASE);
            hsa_signal_store_relaxed(queue->doorbell_signal,
                hsa_queue_load_write_index_relaxed(queue));

            struct timespec ts_start, ts_now;
            clock_gettime(CLOCK_MONOTONIC, &ts_start);
            while (1) {
                int64_t val = sig_mem[1];  // read value directly
                if (val == 0) { printf("  PASS! VRAM signal decremented\n"); break; }
                clock_gettime(CLOCK_MONOTONIC, &ts_now);
                if ((ts_now.tv_sec - ts_start.tv_sec) > 2) {
                    printf("  FAIL! value=%ld (timeout)\n", val);
                    printf("  kind=%ld\n", sig_mem[0]);
                    break;
                }
            }
            hsa_amd_memory_pool_free(vram_sig);
        }
    }

    hsa_signal_destroy(signal);
    hsa_queue_destroy(queue);
    hsa_shut_down();
    printf("Done.\n");
    return 0;
}
