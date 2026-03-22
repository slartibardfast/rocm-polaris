// test_raw_hsa_variants.c — Add CLR behaviors to raw HSA test one at a time
//
// Usage: ./test_raw_hsa_variants <variant> [iters]
// Variants:
//   baseline       — field-by-field write (proven clean)
//   nt_stores      — use _mm_stream for kernarg write
//   struct_copy    — use struct copy for AQL packet (header=0)
//   interleaved    — dispatch two kernels per iteration
//   pool_reuse     — rotating kernarg pool with periodic reset
//   signal_reuse   — pool of 16 signals, reused
//
// Build: gcc -O2 -msse2 -o test_raw_hsa_variants test_raw_hsa_variants.c \
//        -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 \
//        -Wl,-rpath,/opt/rocm/lib
// Run:   ./test_raw_hsa_variants baseline 10000

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <x86intrin.h>

static hsa_agent_t g_gpu = {0}, g_cpu = {0};
static hsa_amd_memory_pool_t g_sys_pool = {0}, g_ka_pool = {0};

static hsa_status_t find_agents(hsa_agent_t agent, void *data) {
    (void)data;
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_GPU && !g_gpu.handle) g_gpu = agent;
    if (type == HSA_DEVICE_TYPE_CPU && !g_cpu.handle) g_cpu = agent;
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
    if (type == HSA_DEVICE_TYPE_CPU) {
        if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) g_ka_pool = pool;
        else if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) g_sys_pool = pool;
    }
    return HSA_STATUS_SUCCESS;
}

static void dispatch_one(hsa_queue_t *queue, uint64_t kernel_object,
                          void *kernarg, hsa_signal_t signal,
                          uint32_t private_size, uint32_t group_size,
                          const char *variant) {
    uint64_t index = hsa_queue_load_write_index_relaxed(queue);
    hsa_kernel_dispatch_packet_t *pkt =
        &((hsa_kernel_dispatch_packet_t*)queue->base_address)[index & (queue->size - 1)];

    if (strcmp(variant, "struct_copy") == 0) {
        // CLR-style: fill struct on stack, then struct copy
        hsa_kernel_dispatch_packet_t tmp = {0};
        tmp.setup = 1;
        tmp.workgroup_size_x = 1; tmp.workgroup_size_y = 1; tmp.workgroup_size_z = 1;
        tmp.grid_size_x = 1; tmp.grid_size_y = 1; tmp.grid_size_z = 1;
        tmp.kernel_object = kernel_object;
        tmp.kernarg_address = kernarg;
        tmp.private_segment_size = private_size;
        tmp.group_segment_size = group_size;
        tmp.completion_signal = signal;
        // header stays 0 (INVALID) in tmp — matches CLR behavior
        *pkt = tmp;  // struct copy
    } else {
        // Field-by-field (baseline — what raw HSA test does)
        pkt->setup = 1;
        pkt->workgroup_size_x = 1; pkt->workgroup_size_y = 1; pkt->workgroup_size_z = 1;
        pkt->grid_size_x = 1; pkt->grid_size_y = 1; pkt->grid_size_z = 1;
        pkt->kernel_object = kernel_object;
        pkt->kernarg_address = kernarg;
        pkt->private_segment_size = private_size;
        pkt->group_segment_size = group_size;
        pkt->completion_signal = signal;
    }

    // Header last with atomic release (same for all variants)
    uint16_t header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
    __atomic_store_n((uint32_t*)pkt, header | (1 << 16), __ATOMIC_RELEASE);

    hsa_queue_store_write_index_relaxed(queue, index + 1);
    hsa_signal_store_screlease(queue->doorbell_signal, index);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <variant> [iters]\n", argv[0]); return 1; }
    const char *variant = argv[1];
    int iters = argc > 2 ? atoi(argv[2]) : 10000;

    hsa_init();
    hsa_iterate_agents(find_agents, NULL);
    hsa_amd_agent_iterate_memory_pools(g_cpu, find_pools, &g_cpu);

    // Load kernel
    FILE *f = fopen("record_val_gfx803.co", "rb");
    if (!f) { fprintf(stderr, "Cannot open record_val_gfx803.co\n"); return 1; }
    fseek(f, 0, SEEK_END); size_t co_size = ftell(f); fseek(f, 0, SEEK_SET);
    void *co_data = malloc(co_size);
    fread(co_data, 1, co_size, f); fclose(f);

    hsa_code_object_reader_t reader;
    hsa_code_object_reader_create_from_memory(co_data, co_size, &reader);
    hsa_executable_t exec;
    hsa_executable_create_alt(HSA_PROFILE_BASE, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, NULL, &exec);
    hsa_executable_load_agent_code_object(exec, g_gpu, reader, NULL, NULL);
    hsa_executable_freeze(exec, NULL);

    hsa_executable_symbol_t sym;
    hsa_executable_get_symbol_by_name(exec, "record_val.kd", &g_gpu, &sym);
    uint64_t kernel_object;
    hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object);
    uint32_t kernarg_size, group_size, private_size;
    hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kernarg_size);
    hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &group_size);
    hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &private_size);

    // Create queue
    hsa_queue_t *queue;
    hsa_queue_create(g_gpu, 1024, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX, UINT32_MAX, &queue);

    // Allocate result buffer (system memory)
    int *d_result;
    hsa_amd_memory_pool_t pool = g_ka_pool.handle ? g_ka_pool : g_sys_pool;
    hsa_amd_memory_pool_allocate(pool, sizeof(int), 0, (void**)&d_result);
    hsa_amd_agents_allow_access(1, &g_gpu, NULL, d_result);

    // Kernarg: single buffer or pool depending on variant
    int ka_align = 16;
    int ka_stride = (kernarg_size + ka_align - 1) & ~(ka_align - 1);
    int pool_entries = 256;
    void *ka_base;
    hsa_amd_memory_pool_allocate(pool, ka_stride * pool_entries, 0, &ka_base);
    hsa_amd_agents_allow_access(1, &g_gpu, NULL, ka_base);

    // Signals
    int num_signals = (strcmp(variant, "signal_reuse") == 0) ? 16 : 1;
    hsa_signal_t signals[16];
    for (int i = 0; i < num_signals; i++) hsa_signal_create(1, 0, NULL, &signals[i]);

    printf("Running %d dispatches (variant: %s)...\n", iters, variant);

    int fails = 0;
    int pool_offset = 0;

    for (int i = 0; i < iters; i++) {
        int expected = i * 7 + 13;
        *d_result = 0xFFFFFFFF;

        // Kernarg selection
        void *kernarg;
        if (strcmp(variant, "pool_reuse") == 0) {
            kernarg = (char*)ka_base + (pool_offset % pool_entries) * ka_stride;
            pool_offset++;
            if (pool_offset >= pool_entries) pool_offset = 0;  // reset = reuse
        } else {
            kernarg = ka_base;  // same buffer every time
        }

        // Write kernarg
        if (strcmp(variant, "nt_stores") == 0) {
            _mm_stream_si64((long long*)kernarg, (long long)(uintptr_t)d_result);
            _mm_stream_si32((int*)((char*)kernarg + 8), expected);
            _mm_sfence();
        } else {
            *(uint64_t*)kernarg = (uint64_t)(uintptr_t)d_result;
            *(int*)((char*)kernarg + 8) = expected;
        }

        // Signal
        hsa_signal_t sig = signals[i % num_signals];
        hsa_signal_store_relaxed(sig, 1);

        // Interleaved: dispatch a dummy "memset" kernel first
        if (strcmp(variant, "interleaved") == 0) {
            // Write d_result = 0xFF via a kernel dispatch (simulating hipMemset)
            void *ka2 = (char*)ka_base + ka_stride;  // use second slot
            *(uint64_t*)ka2 = (uint64_t)(uintptr_t)d_result;
            *(int*)((char*)ka2 + 8) = 0xFFFFFFFF;
            hsa_signal_t sig2;
            hsa_signal_create(1, 0, NULL, &sig2);
            dispatch_one(queue, kernel_object, ka2, sig2, private_size, group_size, "baseline");
            hsa_signal_wait_scacquire(sig2, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
            hsa_signal_destroy(sig2);
        }

        // Main dispatch
        dispatch_one(queue, kernel_object, kernarg, sig, private_size, group_size, variant);
        hsa_signal_wait_scacquire(sig, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);

        if (*d_result != expected) {
            fails++;
            if (fails <= 5)
                printf("  iter %d: expected %d, got %d (0x%08x)\n", i, expected, *d_result, *d_result);
        }
    }

    printf("%-20s %3d / %d (%.3f%%)\n", variant, fails, iters, 100.0*fails/iters);

    for (int i = 0; i < num_signals; i++) hsa_signal_destroy(signals[i]);
    hsa_amd_memory_pool_free(ka_base);
    hsa_amd_memory_pool_free(d_result);
    hsa_queue_destroy(queue);
    hsa_executable_destroy(exec);
    hsa_code_object_reader_destroy(reader);
    free(co_data);
    hsa_shut_down();
    return fails > 0 ? 1 : 0;
}
