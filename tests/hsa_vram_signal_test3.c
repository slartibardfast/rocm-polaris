// Test: can CP write to VRAM vs system memory?
// Place a fake signal struct in VRAM, submit barrier-AND,
// use hsa_memory_copy to read back VRAM → system memory.
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <unistd.h>

static hsa_agent_t gpu_agent = {0};
static hsa_agent_t cpu_agent = {0};

static hsa_status_t find_agents(hsa_agent_t agent, void *data) {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_GPU && !gpu_agent.handle)
        gpu_agent = agent;
    if (type == HSA_DEVICE_TYPE_CPU && !cpu_agent.handle)
        cpu_agent = agent;
    return HSA_STATUS_SUCCESS;
}

// amd_signal_t layout (must match ROCR's definition)
// Total size 256 bytes (per SharedSignal, page-aligned pool)
typedef struct __attribute__((aligned(64))) {
    uint64_t kind;                // offset 0: AMD_SIGNAL_KIND_USER=0
    int64_t  value;               // offset 8: signal value
    uint64_t event_mailbox_ptr;   // offset 16
    uint32_t event_id;            // offset 24
    uint32_t reserved1;           // offset 28
    uint64_t start_ts;            // offset 32
    uint64_t end_ts;              // offset 40
    uint64_t queue_ptr;           // offset 48
    uint32_t reserved2[4];        // offset 56 (pad to 72)
} fake_signal_t;

static hsa_amd_memory_pool_t vram_pool = {0};
static hsa_amd_memory_pool_t sys_pool = {0};

static hsa_status_t find_gpu_pools(hsa_amd_memory_pool_t pool, void *data) {
    hsa_amd_segment_t segment;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
    if (segment == HSA_AMD_SEGMENT_GLOBAL) {
        hsa_amd_memory_pool_global_flag_t flags;
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
        size_t sz = 0;
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SIZE, &sz);
        printf("  GPU pool: size=%zuMB flags=0x%x\n", sz/(1024*1024), flags);
        if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED)
            vram_pool = pool;
    }
    return HSA_STATUS_SUCCESS;
}

static hsa_status_t find_sys_pools(hsa_amd_memory_pool_t pool, void *data) {
    hsa_amd_segment_t segment;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
    if (segment == HSA_AMD_SEGMENT_GLOBAL) {
        hsa_amd_memory_pool_global_flag_t flags;
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
        if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)
            sys_pool = pool;
    }
    return HSA_STATUS_SUCCESS;
}

static double elapsed_since(struct timespec *start) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (now.tv_sec - start->tv_sec) + (now.tv_nsec - start->tv_nsec) / 1e9;
}

int main() {
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    printf("=== VRAM vs System Memory Signal Test ===\n");

    hsa_status_t s = hsa_init();
    if (s != HSA_STATUS_SUCCESS) { printf("hsa_init failed: %d\n", s); return 1; }

    hsa_iterate_agents(find_agents, NULL);
    if (!gpu_agent.handle || !cpu_agent.handle) {
        printf("No GPU or CPU agent\n"); return 1;
    }

    printf("Finding pools...\n");
    hsa_amd_agent_iterate_memory_pools(gpu_agent, find_gpu_pools, NULL);
    hsa_amd_agent_iterate_memory_pools(cpu_agent, find_sys_pools, NULL);

    if (!vram_pool.handle) { printf("No VRAM pool\n"); return 1; }
    if (!sys_pool.handle) { printf("No system pool\n"); return 1; }

    // Allocate VRAM for fake signal (must be GPU-writable)
    void *vram_sig = NULL;
    s = hsa_amd_memory_pool_allocate(vram_pool, 4096, 0, &vram_sig);
    printf("VRAM signal alloc: %d gpu_va=%p\n", s, vram_sig);
    if (s != HSA_STATUS_SUCCESS) return 1;

    // Allocate system memory for readback
    void *readback = NULL;
    s = hsa_amd_memory_pool_allocate(sys_pool, 4096, 0, &readback);
    printf("System readback alloc: %d ptr=%p\n", s, readback);
    if (s != HSA_STATUS_SUCCESS) return 1;

    // Allow GPU to access system readback buffer
    s = hsa_amd_agents_allow_access(1, &gpu_agent, NULL, readback);
    printf("GPU access to readback: %d\n", s);

    // Initialize fake signal in VRAM via system buffer + copy
    fake_signal_t init_sig;
    memset(&init_sig, 0, sizeof(init_sig));
    init_sig.kind = 0;   // AMD_SIGNAL_KIND_USER
    init_sig.value = 1;  // Will be decremented to 0 by barrier-AND
    init_sig.event_mailbox_ptr = 0;
    init_sig.event_id = 0;

    // Copy init data to VRAM
    s = hsa_memory_copy(vram_sig, &init_sig, sizeof(init_sig));
    printf("Init copy to VRAM: %d\n", s);
    if (s != HSA_STATUS_SUCCESS) {
        printf("Cannot init VRAM signal — trying memset via allow_access\n");
        // Try allowing CPU access and writing directly
        s = hsa_amd_agents_allow_access(1, &cpu_agent, NULL, vram_sig);
        printf("CPU access to VRAM: %d\n", s);
        if (s == HSA_STATUS_SUCCESS) {
            memcpy(vram_sig, &init_sig, sizeof(init_sig));
            printf("Direct VRAM write succeeded\n");
        } else {
            printf("Cannot initialize VRAM signal, aborting\n");
            return 1;
        }
    }

    // Verify: copy VRAM back to readback and check
    memset(readback, 0xFF, 4096);
    s = hsa_memory_copy(readback, vram_sig, sizeof(init_sig));
    printf("Verify readback: %d\n", s);
    fake_signal_t *rb = (fake_signal_t*)readback;
    printf("  VRAM signal: kind=%lu value=%ld mailbox=%lx\n",
           rb->kind, rb->value, rb->event_mailbox_ptr);

    if (rb->value != 1) {
        printf("ERROR: VRAM signal value mismatch, expected 1 got %ld\n", rb->value);
        return 1;
    }

    // Build hsa_signal_t handle pointing to VRAM signal
    hsa_signal_t fake_sig;
    fake_sig.handle = (uint64_t)vram_sig;
    printf("\nFake signal handle: 0x%lx (VRAM)\n", fake_sig.handle);

    // Also create a real system-memory signal for comparison
    hsa_signal_t real_sig;
    s = hsa_signal_create(1, 0, NULL, &real_sig);
    printf("Real signal handle: 0x%lx (system memory)\n", real_sig.handle);

    // Create queue
    hsa_queue_t *queue = NULL;
    s = hsa_queue_create(gpu_agent, 1024, HSA_QUEUE_TYPE_MULTI, NULL, NULL, 0, 0, &queue);
    printf("Queue create: %d id=%u\n", s, s == 0 ? queue->id : 0);
    if (s != HSA_STATUS_SUCCESS) return 1;

    uint16_t header = HSA_PACKET_TYPE_BARRIER_AND;
    header |= (1 << HSA_PACKET_HEADER_BARRIER);
    header |= (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE);
    header |= (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    uint64_t mask = queue->size - 1;

    // ---- TEST 1: Barrier-AND with VRAM signal ----
    printf("\n=== TEST 1: Barrier-AND → VRAM signal ===\n");
    uint64_t write_idx = hsa_queue_add_write_index_relaxed(queue, 1);
    hsa_barrier_and_packet_t *barrier =
        (hsa_barrier_and_packet_t*)((char*)queue->base_address +
        (write_idx & mask) * 64);
    memset(barrier, 0, sizeof(*barrier));
    barrier->completion_signal = fake_sig;
    __atomic_store_n((uint16_t*)barrier, header, __ATOMIC_RELEASE);
    hsa_signal_store_relaxed(queue->doorbell_signal, write_idx);
    printf("Submitted, doorbell=%lu\n", write_idx);

    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
    int vram_ok = 0;
    while (elapsed_since(&start) < 3.0) {
        // Read VRAM signal value via hsa_memory_copy
        int64_t val = -999;
        hsa_memory_copy(&val, (char*)vram_sig + 8, sizeof(val));  // offset 8 = value field
        if (val == 0) {
            printf("VRAM signal DECREMENTED to 0! (%.3fs)\n", elapsed_since(&start));
            vram_ok = 1;
            break;
        }
        usleep(1000);
    }
    if (!vram_ok) {
        int64_t val = -999;
        hsa_memory_copy(&val, (char*)vram_sig + 8, sizeof(val));
        printf("VRAM signal TIMEOUT! value=%ld\n", val);
        printf("  read_idx=%lu write_idx=%lu\n",
               hsa_queue_load_read_index_relaxed(queue),
               hsa_queue_load_write_index_relaxed(queue));
    }

    // ---- TEST 2: Barrier-AND with system memory signal ----
    printf("\n=== TEST 2: Barrier-AND → system memory signal ===\n");
    write_idx = hsa_queue_add_write_index_relaxed(queue, 1);
    barrier = (hsa_barrier_and_packet_t*)((char*)queue->base_address +
        (write_idx & mask) * 64);
    memset(barrier, 0, sizeof(*barrier));
    barrier->completion_signal = real_sig;
    __atomic_store_n((uint16_t*)barrier, header, __ATOMIC_RELEASE);
    hsa_signal_store_relaxed(queue->doorbell_signal, write_idx);
    printf("Submitted, doorbell=%lu\n", write_idx);

    clock_gettime(CLOCK_MONOTONIC, &start);
    int sys_ok = 0;
    while (elapsed_since(&start) < 3.0) {
        hsa_signal_value_t val = hsa_signal_load_relaxed(real_sig);
        if (val == 0) {
            printf("System signal DECREMENTED to 0! (%.3fs)\n", elapsed_since(&start));
            sys_ok = 1;
            break;
        }
        usleep(1000);
    }
    if (!sys_ok) {
        printf("System signal TIMEOUT! value=%ld\n", hsa_signal_load_relaxed(real_sig));
    }

    printf("\n=== RESULTS ===\n");
    printf("VRAM signal:   %s\n", vram_ok ? "PASS (CP can write VRAM)" : "FAIL");
    printf("System signal: %s\n", sys_ok ? "PASS (CP can write system mem)" : "FAIL");
    if (vram_ok && !sys_ok)
        printf(">>> CONFIRMED: CP can write VRAM but NOT system memory <<<\n");
    else if (!vram_ok && !sys_ok)
        printf(">>> CP cannot write either — queue may not be processing packets <<<\n");
    else if (vram_ok && sys_ok)
        printf(">>> Both work — issue may be elsewhere <<<\n");

    hsa_signal_destroy(real_sig);
    hsa_amd_memory_pool_free(vram_sig);
    hsa_amd_memory_pool_free(readback);
    hsa_queue_destroy(queue);
    hsa_shut_down();
    printf("Done.\n");
    return 0;
}
