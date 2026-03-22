// Test: allocate kernarg from fine_grain_pool with AllocateUncached (like CLR does)
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static hsa_agent_t g_gpu={0}, g_cpu={0};
static hsa_amd_memory_pool_t g_fg_pool={0};

static hsa_status_t find_agents(hsa_agent_t a, void *d) {
    (void)d; hsa_device_type_t t;
    hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
    if (t==HSA_DEVICE_TYPE_GPU && !g_gpu.handle) g_gpu=a;
    if (t==HSA_DEVICE_TYPE_CPU && !g_cpu.handle) g_cpu=a;
    return HSA_STATUS_SUCCESS;
}
static hsa_status_t find_fg(hsa_amd_memory_pool_t p, void *d) {
    (void)d; hsa_amd_segment_t seg;
    hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
    if (seg != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
    uint32_t f;
    hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &f);
    if (f & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) g_fg_pool = p;
    return HSA_STATUS_SUCCESS;
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 10000;
    hsa_init();
    hsa_iterate_agents(find_agents, NULL);
    hsa_amd_agent_iterate_memory_pools(g_cpu, find_fg, NULL);

    // Load kernel
    FILE *f = fopen("record_val_gfx803.co", "rb");
    fseek(f,0,SEEK_END); size_t sz=ftell(f); fseek(f,0,SEEK_SET);
    void *co=malloc(sz); fread(co,1,sz,f); fclose(f);
    hsa_code_object_reader_t rdr;
    hsa_code_object_reader_create_from_memory(co, sz, &rdr);
    hsa_executable_t exe;
    hsa_executable_create_alt(HSA_PROFILE_BASE, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, NULL, &exe);
    hsa_executable_load_agent_code_object(exe, g_gpu, rdr, NULL, NULL);
    hsa_executable_freeze(exe, NULL);
    hsa_executable_symbol_t sym;
    hsa_executable_get_symbol_by_name(exe, "record_val.kd", &g_gpu, &sym);
    uint64_t kobj; uint32_t kasiz, gsiz, psiz;
    hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kobj);
    hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kasiz);
    hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &gsiz);
    hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &psiz);

    hsa_queue_t *q;
    hsa_queue_create(g_gpu, 1024, HSA_QUEUE_TYPE_MULTI, NULL, NULL, UINT32_MAX, UINT32_MAX, &q);

    // Allocate result from fine_grain with UNCACHED (like CLR hostAlloc)
    int *result;
    hsa_amd_memory_pool_allocate(g_fg_pool, sizeof(int),
        HSA_AMD_MEMORY_POOL_UNCACHED_FLAG, (void**)&result);
    hsa_amd_agents_allow_access(1, &g_gpu, NULL, result);

    // Allocate kernarg from fine_grain with UNCACHED (like CLR ManagedBuffer)
    void *ka;
    hsa_amd_memory_pool_allocate(g_fg_pool, 4096,
        HSA_AMD_MEMORY_POOL_UNCACHED_FLAG, &ka);
    hsa_amd_agents_allow_access(1, &g_gpu, NULL, ka);

    hsa_signal_t sig;
    hsa_signal_create(1, 0, NULL, &sig);

    printf("Using fine_grain+UNCACHED (like CLR). %d iters...\n", iters);
    int fails = 0;
    for (int i = 0; i < iters; i++) {
        int expected = i * 7 + 13;
        *result = 0xFFFFFFFF;
        *(uint64_t*)ka = (uint64_t)(uintptr_t)result;
        *(int*)((char*)ka + 8) = expected;

        hsa_signal_store_relaxed(sig, 1);
        uint64_t idx = hsa_queue_add_write_index_screlease(q, 1);
        hsa_kernel_dispatch_packet_t *pkt =
            &((hsa_kernel_dispatch_packet_t*)q->base_address)[idx & (q->size-1)];
        hsa_kernel_dispatch_packet_t tmp = {0};
        tmp.setup=1; tmp.workgroup_size_x=1; tmp.workgroup_size_y=1; tmp.workgroup_size_z=1;
        tmp.grid_size_x=1; tmp.grid_size_y=1; tmp.grid_size_z=1;
        tmp.kernel_object=kobj; tmp.kernarg_address=ka;
        tmp.private_segment_size=psiz; tmp.group_segment_size=gsiz;
        tmp.completion_signal=sig;
        *pkt = tmp;
        uint16_t hdr = HSA_PACKET_TYPE_KERNEL_DISPATCH |
            (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
            (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
        __atomic_store_n((uint32_t*)pkt, hdr | (1<<16), __ATOMIC_RELEASE);
        hsa_signal_store_screlease(q->doorbell_signal, idx);
        hsa_signal_wait_scacquire(sig, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);

        if (*result != expected) {
            fails++;
            if (fails <= 5) printf("  iter %d: exp %d got %d\n", i, expected, *result);
        }
    }
    printf("fg_uncached: %d / %d (%.3f%%)\n", fails, iters, 100.0*fails/iters);
    hsa_signal_destroy(sig);
    hsa_amd_memory_pool_free(ka);
    hsa_amd_memory_pool_free(result);
    hsa_queue_destroy(q);
    hsa_executable_destroy(exe);
    hsa_code_object_reader_destroy(rdr);
    free(co);
    hsa_shut_down();
    return fails>0?1:0;
}
