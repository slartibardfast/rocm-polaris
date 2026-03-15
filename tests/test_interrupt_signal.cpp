// Test: interrupt-based signal completion on no-atomics platforms.
//
// Verifies that the GPU's event_mailbox_ptr write + s_sendmsg interrupt
// work correctly when PCIe AtomicOps are not available. The signal VALUE
// won't be decremented by the GPU (atomic fails), but the interrupt
// should still fire, waking hsaKmtWaitOnEvent.
//
// This test does NOT use HSA_ENABLE_INTERRUPT=0 — it tests the interrupt
// path specifically.

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/amd_hsa_signal.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <unistd.h>

#define HSA_CHECK(call) do { hsa_status_t s = (call); \
    if (s != HSA_STATUS_SUCCESS) { \
        fprintf(stderr, "HSA error %d at %s:%d\n", s, __FILE__, __LINE__); \
        return 1; \
    } } while(0)

static hsa_agent_t gpu_agent, cpu_agent;
static bool found_gpu = false, found_cpu = false;

static hsa_status_t find_agents(hsa_agent_t agent, void*) {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_GPU && !found_gpu) { gpu_agent = agent; found_gpu = true; }
    if (type == HSA_DEVICE_TYPE_CPU && !found_cpu) { cpu_agent = agent; found_cpu = true; }
    return HSA_STATUS_SUCCESS;
}

int main() {
    setbuf(stdout, NULL);
    int failures = 0;

    printf("=== Test: Interrupt-Based Signal Completion ===\n\n");

    HSA_CHECK(hsa_init());
    hsa_iterate_agents(find_agents, NULL);
    if (!found_gpu) { fprintf(stderr, "No GPU agent found\n"); return 1; }

    char name[64];
    hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_NAME, name);
    printf("GPU: %s\n", name);

    // Check if platform has PCIe atomics
    hsa_amd_memory_pool_t fine_pool = {0};
    auto pool_cb = [](hsa_amd_memory_pool_t pool, void* data) -> hsa_status_t {
        hsa_amd_segment_t seg;
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
        if (seg == HSA_AMD_SEGMENT_GLOBAL) {
            bool fine;
            hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &fine);
            // Just find any pool
            *(hsa_amd_memory_pool_t*)data = pool;
        }
        return HSA_STATUS_SUCCESS;
    };
    hsa_amd_agent_iterate_memory_pools(cpu_agent, pool_cb, &fine_pool);

    // === Test 1: Verify interrupt counter increases ===
    printf("\nTest 1: GPU interrupt delivery...\n");
    {
        // Read interrupt count from /proc/interrupts
        FILE* f = fopen("/proc/interrupts", "r");
        long count_before = 0;
        if (f) {
            char line[1024];
            while (fgets(line, sizeof(line), f)) {
                if (strstr(line, "amdgpu")) {
                    // Sum all CPU columns
                    char* p = line;
                    while (*p == ' ') p++;
                    p = strchr(p, ':'); // skip IRQ number
                    if (p) {
                        p++;
                        long val;
                        while (sscanf(p, "%ld", &val) == 1) {
                            count_before += val;
                            while (*p == ' ') p++;
                            while (*p && *p != ' ') p++;
                        }
                    }
                    break;
                }
            }
            fclose(f);
        }

        // Do a GPU operation
        hsa_queue_t* queue;
        HSA_CHECK(hsa_queue_create(gpu_agent, 128, HSA_QUEUE_TYPE_SINGLE,
                                    NULL, NULL, UINT32_MAX, UINT32_MAX, &queue));
        hsa_signal_t sig;
        HSA_CHECK(hsa_signal_create(1, 0, NULL, &sig));

        // Submit barrier
        uint64_t idx = hsa_queue_load_write_index_relaxed(queue);
        hsa_barrier_and_packet_t* pkt = (hsa_barrier_and_packet_t*)
            ((char*)queue->base_address + (idx & (queue->size - 1)) * 64);
        memset(pkt, 0, sizeof(*pkt));
        pkt->completion_signal = sig;
        __atomic_store_n((uint16_t*)pkt,
                         (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
                         (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE),
                         __ATOMIC_RELEASE);
        hsa_queue_store_write_index_relaxed(queue, idx + 1);
        hsa_signal_store_relaxed(queue->doorbell_signal, idx);

        usleep(100000); // 100ms for interrupt to fire

        // Read interrupt count again
        long count_after = 0;
        f = fopen("/proc/interrupts", "r");
        if (f) {
            char line[1024];
            while (fgets(line, sizeof(line), f)) {
                if (strstr(line, "amdgpu")) {
                    char* p = line;
                    while (*p == ' ') p++;
                    p = strchr(p, ':');
                    if (p) {
                        p++;
                        long val;
                        while (sscanf(p, "%ld", &val) == 1) {
                            count_after += val;
                            while (*p == ' ') p++;
                            while (*p && *p != ' ') p++;
                        }
                    }
                    break;
                }
            }
            fclose(f);
        }

        long delta = count_after - count_before;
        printf("  Interrupt delta: %ld", delta);
        if (delta > 0) {
            printf(" PASS\n");
        } else {
            printf(" FAIL (no interrupts received)\n");
            failures++;
        }

        hsa_signal_destroy(sig);
        hsa_queue_destroy(queue);
    }

    // === Test 2: Check signal value after interrupt ===
    printf("\nTest 2: Signal value after GPU dispatch...\n");
    {
        hsa_queue_t* queue;
        HSA_CHECK(hsa_queue_create(gpu_agent, 128, HSA_QUEUE_TYPE_SINGLE,
                                    NULL, NULL, UINT32_MAX, UINT32_MAX, &queue));
        hsa_signal_t sig;
        HSA_CHECK(hsa_signal_create(1, 0, NULL, &sig));

        // Submit barrier with completion signal
        uint64_t idx = hsa_queue_load_write_index_relaxed(queue);
        hsa_barrier_and_packet_t* pkt = (hsa_barrier_and_packet_t*)
            ((char*)queue->base_address + (idx & (queue->size - 1)) * 64);
        memset(pkt, 0, sizeof(*pkt));
        pkt->completion_signal = sig;
        __atomic_store_n((uint16_t*)pkt,
                         (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
                         (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE),
                         __ATOMIC_RELEASE);
        hsa_queue_store_write_index_relaxed(queue, idx + 1);
        hsa_signal_store_relaxed(queue->doorbell_signal, idx);

        usleep(100000); // Wait for CP to process

        int64_t val = hsa_signal_load_scacquire(sig);
        printf("  Signal value: %ld", val);
        if (val == 0) {
            printf(" (GPU has PCIe atomics — decremented signal directly)\n");
        } else if (val == 1) {
            printf(" (No PCIe atomics — signal NOT decremented by GPU)\n");
            printf("  This is EXPECTED on Westmere. The bounce buffer must decrement.\n");
        } else {
            printf(" UNEXPECTED value\n");
            failures++;
        }

        hsa_signal_destroy(sig);
        hsa_queue_destroy(queue);
    }

    // === Test 3: Check event_mailbox_ptr in signal ===
    printf("\nTest 3: Signal has event_mailbox_ptr...\n");
    {
        hsa_signal_t sig;
        HSA_CHECK(hsa_signal_create(1, 0, NULL, &sig));

        amd_signal_t* ams = (amd_signal_t*)sig.handle;
        printf("  event_mailbox_ptr: 0x%lx\n", ams->event_mailbox_ptr);
        printf("  event_id: %u\n", ams->event_id);

        if (ams->event_mailbox_ptr != 0) {
            printf("  PASS (signal has event mailbox for interrupt delivery)\n");
        } else {
            printf("  INFO (no event mailbox — interrupt signals may not be allocated)\n");
            printf("  This may be because g_use_interrupt_wait=false suppresses event allocation.\n");
        }

        hsa_signal_destroy(sig);
    }

    // === Test 4: Verify RPTR advances (bounce buffer basic) ===
    printf("\nTest 4: RPTR advances after barrier dispatch...\n");
    {
        hsa_queue_t* queue;
        HSA_CHECK(hsa_queue_create(gpu_agent, 128, HSA_QUEUE_TYPE_SINGLE,
                                    NULL, NULL, UINT32_MAX, UINT32_MAX, &queue));
        hsa_signal_t sig;
        HSA_CHECK(hsa_signal_create(1, 0, NULL, &sig));

        uint64_t rid_before = hsa_queue_load_read_index_relaxed(queue);

        // Submit barrier
        uint64_t idx = hsa_queue_load_write_index_relaxed(queue);
        hsa_barrier_and_packet_t* pkt = (hsa_barrier_and_packet_t*)
            ((char*)queue->base_address + (idx & (queue->size - 1)) * 64);
        memset(pkt, 0, sizeof(*pkt));
        pkt->completion_signal = sig;
        __atomic_store_n((uint16_t*)pkt,
                         (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
                         (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE),
                         __ATOMIC_RELEASE);
        hsa_queue_store_write_index_relaxed(queue, idx + 1);
        hsa_signal_store_relaxed(queue->doorbell_signal, idx);

        usleep(200000); // Wait for CP + RPTR update

        uint64_t rid_after = hsa_queue_load_read_index_relaxed(queue);
        printf("  read_dispatch_id: %lu -> %lu", rid_before, rid_after);
        if (rid_after > rid_before) {
            printf(" PASS (RPTR advanced)\n");
        } else {
            printf(" FAIL (RPTR did not advance)\n");
            failures++;
        }

        hsa_signal_destroy(sig);
        hsa_queue_destroy(queue);
    }

    printf("\n=== Results: %d failures ===\n", failures);

    hsa_shut_down();
    return failures;
}
