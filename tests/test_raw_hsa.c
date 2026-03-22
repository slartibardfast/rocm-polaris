// test_raw_hsa.c — Raw HSA dispatch bypassing CLR entirely
//
// Loads a pre-compiled .hsaco, creates a queue, writes AQL packets
// manually, rings doorbell, and verifies results. If corruption
// disappears → CLR bug. If persists → GPU hardware issue.
//
// Build: gcc -O2 -o test_raw_hsa test_raw_hsa.c \
//        -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 \
//        -Wl,-rpath,/opt/rocm/lib
// Run:   ./test_raw_hsa [iters]

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static hsa_agent_t g_gpu = {0};
static hsa_agent_t g_cpu = {0};
static hsa_amd_memory_pool_t g_kernarg_pool = {0};
static hsa_amd_memory_pool_t g_gpu_pool = {0};
static hsa_amd_memory_pool_t g_sys_pool = {0};

static hsa_status_t find_agents(hsa_agent_t agent, void *data) {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_GPU && g_gpu.handle == 0) g_gpu = agent;
    if (type == HSA_DEVICE_TYPE_CPU && g_cpu.handle == 0) g_cpu = agent;
    return HSA_STATUS_SUCCESS;
}

static hsa_status_t find_pools(hsa_amd_memory_pool_t pool, void *data) {
    hsa_amd_segment_t seg;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
    if (seg != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;

    uint32_t flags;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);

    hsa_agent_t agent = *(hsa_agent_t*)data;
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);

    if (type == HSA_DEVICE_TYPE_GPU) {
        if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED)
            g_gpu_pool = pool;
    } else {
        if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT)
            g_kernarg_pool = pool;
        else if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)
            g_sys_pool = pool;
    }
    return HSA_STATUS_SUCCESS;
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 10000;
    hsa_status_t s;

    s = hsa_init();
    if (s != HSA_STATUS_SUCCESS) { fprintf(stderr, "hsa_init failed\n"); return 1; }

    hsa_iterate_agents(find_agents, NULL);
    if (!g_gpu.handle) { fprintf(stderr, "No GPU\n"); return 1; }

    // Find memory pools
    hsa_amd_agent_iterate_memory_pools(g_gpu, find_pools, &g_gpu);
    hsa_amd_agent_iterate_memory_pools(g_cpu, find_pools, &g_cpu);

    // Load code object
    FILE *f = fopen("record_val_gfx803.co", "rb");
    if (!f) { fprintf(stderr, "Cannot open record_val.hsaco\n"); return 1; }
    fseek(f, 0, SEEK_END);
    size_t co_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *co_data = malloc(co_size);
    fread(co_data, 1, co_size, f);
    fclose(f);

    hsa_code_object_reader_t reader;
    s = hsa_code_object_reader_create_from_memory(co_data, co_size, &reader);
    if (s != HSA_STATUS_SUCCESS) { fprintf(stderr, "reader create failed: %d\n", s); return 1; }

    hsa_executable_t exec;
    s = hsa_executable_create_alt(HSA_PROFILE_BASE, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                   NULL, &exec);
    if (s != HSA_STATUS_SUCCESS) { fprintf(stderr, "exec create failed: %d\n", s); return 1; }

    s = hsa_executable_load_agent_code_object(exec, g_gpu, reader, NULL, NULL);
    if (s != HSA_STATUS_SUCCESS) { fprintf(stderr, "load failed: %d\n", s); return 1; }

    s = hsa_executable_freeze(exec, NULL);
    if (s != HSA_STATUS_SUCCESS) { fprintf(stderr, "freeze failed: %d\n", s); return 1; }

    // Get kernel symbol
    hsa_executable_symbol_t sym;
    s = hsa_executable_get_symbol_by_name(exec, "record_val.kd", &g_gpu, &sym);
    if (s != HSA_STATUS_SUCCESS) { fprintf(stderr, "symbol not found: %d\n", s); return 1; }

    uint64_t kernel_object;
    hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object);

    uint32_t kernarg_size;
    hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kernarg_size);

    uint32_t group_size;
    hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &group_size);

    uint32_t private_size;
    hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &private_size);

    printf("Kernel: object=0x%lx kernarg=%u group=%u private=%u\n",
           kernel_object, kernarg_size, group_size, private_size);

    // Create queue
    hsa_queue_t *queue;
    s = hsa_queue_create(g_gpu, 1024, HSA_QUEUE_TYPE_SINGLE, NULL, NULL,
                         UINT32_MAX, UINT32_MAX, &queue);
    if (s != HSA_STATUS_SUCCESS) { fprintf(stderr, "queue create failed: %d\n", s); return 1; }

    // Allocate result in system memory (CPU+GPU accessible)
    int *d_result;
    s = hsa_amd_memory_pool_allocate(g_sys_pool, sizeof(int), 0, (void**)&d_result);
    if (s != HSA_STATUS_SUCCESS) { fprintf(stderr, "sys alloc for result failed: %d\n", s); return 1; }
    hsa_amd_agents_allow_access(1, &g_gpu, NULL, d_result);

    // Allocate kernarg (from kernarg pool or fine-grain)
    void *kernarg;
    hsa_amd_memory_pool_t ka_pool = g_kernarg_pool.handle ? g_kernarg_pool : g_sys_pool;
    s = hsa_amd_memory_pool_allocate(ka_pool, kernarg_size * 2, 0, &kernarg);
    if (s != HSA_STATUS_SUCCESS) { fprintf(stderr, "kernarg alloc failed: %d\n", s); return 1; }
    hsa_amd_agents_allow_access(1, &g_gpu, NULL, kernarg);

    // Create completion signal
    hsa_signal_t signal;
    hsa_signal_create(1, 0, NULL, &signal);

    printf("Running %d dispatches (raw HSA, no CLR)...\n", iters);

    int fails = 0;
    for (int i = 0; i < iters; i++) {
        int expected = i * 7 + 13;

        // Clear result
        *d_result = 0xFFFFFFFF;

        // Write kernarg: {pointer(8B), int(4B)}
        *(uint64_t*)((char*)kernarg) = (uint64_t)(uintptr_t)d_result;
        *(int*)((char*)kernarg + 8) = expected;

        // Reset signal
        hsa_signal_store_relaxed(signal, 1);

        // Write AQL packet
        uint64_t index = hsa_queue_load_write_index_relaxed(queue);
        hsa_kernel_dispatch_packet_t *pkt =
            &((hsa_kernel_dispatch_packet_t*)queue->base_address)[index & (queue->size - 1)];

        // Write body first (header invalid)
        pkt->setup = 1;  // 1 dimension
        pkt->workgroup_size_x = 1;
        pkt->workgroup_size_y = 1;
        pkt->workgroup_size_z = 1;
        pkt->grid_size_x = 1;
        pkt->grid_size_y = 1;
        pkt->grid_size_z = 1;
        pkt->kernel_object = kernel_object;
        pkt->kernarg_address = kernarg;
        pkt->private_segment_size = private_size;
        pkt->group_segment_size = group_size;
        pkt->completion_signal = signal;

        // Write header last (makes packet valid)
        uint16_t header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
        header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
        header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
        __atomic_store_n(&pkt->header, header | (pkt->setup << 16), __ATOMIC_RELEASE);

        // Increment write index and ring doorbell
        hsa_queue_store_write_index_relaxed(queue, index + 1);
        hsa_signal_store_screlease(queue->doorbell_signal, index);

        // Wait for completion
        while (hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1,
                                          UINT64_MAX, HSA_WAIT_STATE_ACTIVE) != 0) {}

        // Check result
        if (*d_result != expected) {
            fails++;
            if (fails <= 10)
                printf("  iter %d: expected %d, got %d (0x%08x)\n",
                       i, expected, *d_result, *d_result);
        }
    }

    printf("Total: %d fails / %d (%.3f%%)\n", fails, iters, 100.0 * fails / iters);

    hsa_signal_destroy(signal);
    hsa_amd_memory_pool_free(kernarg);
    hsa_amd_memory_pool_free(d_result);
    hsa_queue_destroy(queue);
    hsa_executable_destroy(exec);
    hsa_code_object_reader_destroy(reader);
    free(co_data);
    hsa_shut_down();

    return fails > 0 ? 1 : 0;
}
