// test_two_queues.c — Does using TWO HSA queues cause corruption?
// Raw HSA test with two queues dispatching the same kernel.
// Queue A: dispatches record_val, writes to host-visible buffer
// Queue B: dispatches record_val, writes to a DIFFERENT host-visible buffer
// CPU checks both results.
//
// Build: gcc -O2 -o test_two_queues test_two_queues.c \
//        -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -Wl,-rpath,/opt/rocm/lib

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static hsa_agent_t g_gpu = {0}, g_cpu = {0};
static hsa_amd_memory_pool_t g_sys_pool = {0}, g_ka_pool = {0};

static hsa_status_t find_agents(hsa_agent_t a, void *d) {
    (void)d; hsa_device_type_t t;
    hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
    if (t == HSA_DEVICE_TYPE_GPU && !g_gpu.handle) g_gpu = a;
    if (t == HSA_DEVICE_TYPE_CPU && !g_cpu.handle) g_cpu = a;
    return HSA_STATUS_SUCCESS;
}
static hsa_status_t find_pools(hsa_amd_memory_pool_t p, void *d) {
    (void)d; hsa_amd_segment_t seg;
    hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
    if (seg != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
    uint32_t f;
    hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &f);
    if (f & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) g_ka_pool = p;
    else if (f & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) g_sys_pool = p;
    return HSA_STATUS_SUCCESS;
}

static void dispatch(hsa_queue_t *q, uint64_t kobj, void *ka, hsa_signal_t sig,
                     uint32_t psiz, uint32_t gsiz) {
    uint64_t idx = hsa_queue_load_write_index_relaxed(q);
    hsa_kernel_dispatch_packet_t *p =
        &((hsa_kernel_dispatch_packet_t*)q->base_address)[idx & (q->size-1)];
    p->setup = 1;
    p->workgroup_size_x = 1; p->workgroup_size_y = 1; p->workgroup_size_z = 1;
    p->grid_size_x = 1; p->grid_size_y = 1; p->grid_size_z = 1;
    p->kernel_object = kobj;
    p->kernarg_address = ka;
    p->private_segment_size = psiz;
    p->group_segment_size = gsiz;
    p->completion_signal = sig;
    uint16_t hdr = HSA_PACKET_TYPE_KERNEL_DISPATCH |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    __atomic_store_n((uint32_t*)p, hdr | (1<<16), __ATOMIC_RELEASE);
    hsa_queue_store_write_index_relaxed(q, idx + 1);
    hsa_signal_store_screlease(q->doorbell_signal, idx);
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 10000;
    hsa_init();
    hsa_iterate_agents(find_agents, NULL);
    hsa_amd_agent_iterate_memory_pools(g_cpu, find_pools, NULL);

    FILE *f = fopen("record_val_gfx803.co", "rb");
    if (!f) { fprintf(stderr, "No .co file\n"); return 1; }
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

    // Two queues
    hsa_queue_t *qA, *qB;
    hsa_queue_create(g_gpu, 1024, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX, UINT32_MAX, &qA);
    hsa_queue_create(g_gpu, 1024, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX, UINT32_MAX, &qB);

    // Two result buffers
    int *rA, *rB;
    hsa_amd_memory_pool_t pool = g_ka_pool.handle ? g_ka_pool : g_sys_pool;
    hsa_amd_memory_pool_allocate(pool, sizeof(int), 0, (void**)&rA);
    hsa_amd_memory_pool_allocate(pool, sizeof(int), 0, (void**)&rB);
    hsa_amd_agents_allow_access(1, &g_gpu, NULL, rA);
    hsa_amd_agents_allow_access(1, &g_gpu, NULL, rB);

    // Two kernarg buffers
    void *kaA, *kaB;
    hsa_amd_memory_pool_allocate(pool, 16, 0, &kaA);
    hsa_amd_memory_pool_allocate(pool, 16, 0, &kaB);
    hsa_amd_agents_allow_access(1, &g_gpu, NULL, kaA);
    hsa_amd_agents_allow_access(1, &g_gpu, NULL, kaB);

    hsa_signal_t sigA, sigB;
    hsa_signal_create(1, 0, NULL, &sigA);
    hsa_signal_create(1, 0, NULL, &sigB);

    printf("Two queues, %d iters...\n", iters);
    int failsA = 0, failsB = 0;

    for (int i = 0; i < iters; i++) {
        int expA = i * 7 + 13, expB = i * 3 + 5;
        *rA = 0xFFFFFFFF; *rB = 0xFFFFFFFF;

        *(uint64_t*)kaA = (uint64_t)(uintptr_t)rA;
        *(int*)((char*)kaA + 8) = expA;
        *(uint64_t*)kaB = (uint64_t)(uintptr_t)rB;
        *(int*)((char*)kaB + 8) = expB;

        hsa_signal_store_relaxed(sigA, 1);
        hsa_signal_store_relaxed(sigB, 1);

        dispatch(qA, kobj, kaA, sigA, psiz, gsiz);
        dispatch(qB, kobj, kaB, sigB, psiz, gsiz);

        hsa_signal_wait_scacquire(sigA, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
        hsa_signal_wait_scacquire(sigB, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);

        if (*rA != expA) failsA++;
        if (*rB != expB) failsB++;
    }

    printf("qA: %d fails, qB: %d fails / %d\n", failsA, failsB, iters);

    hsa_signal_destroy(sigA); hsa_signal_destroy(sigB);
    hsa_amd_memory_pool_free(kaA); hsa_amd_memory_pool_free(kaB);
    hsa_amd_memory_pool_free(rA); hsa_amd_memory_pool_free(rB);
    hsa_queue_destroy(qA); hsa_queue_destroy(qB);
    hsa_executable_destroy(exe);
    hsa_code_object_reader_destroy(rdr);
    free(co);
    hsa_shut_down();
    return (failsA + failsB) > 0 ? 1 : 0;
}
