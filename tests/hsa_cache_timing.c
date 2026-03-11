#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static hsa_agent_t cpu_agent;
static hsa_amd_memory_pool_t sys_pool;

static hsa_status_t find_cpu(hsa_agent_t agent, void *data) {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_CPU) cpu_agent = agent;
    return HSA_STATUS_SUCCESS;
}
static hsa_status_t find_kernarg(hsa_amd_memory_pool_t pool, void *data) {
    hsa_amd_segment_t seg;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
    if (seg == HSA_AMD_SEGMENT_GLOBAL) {
        hsa_amd_memory_pool_global_flag_t flags;
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
        if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) sys_pool = pool;
    }
    return HSA_STATUS_SUCCESS;
}

static long long time_reads(volatile unsigned int *p, int count, int stride) {
    struct timespec t0, t1;
    unsigned int sink = 0;
    // Warm up
    for (int i = 0; i < count; i++) sink += p[i * stride / 4];
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int rep = 0; rep < 100; rep++)
        for (int i = 0; i < count; i++) sink += p[i * stride / 4];
    clock_gettime(CLOCK_MONOTONIC, &t1);
    (void)sink;
    return (t1.tv_sec - t0.tv_sec) * 1000000000LL + (t1.tv_nsec - t0.tv_nsec);
}

int main() {
    hsa_init();
    hsa_iterate_agents(find_cpu, NULL);
    hsa_amd_agent_iterate_memory_pools(cpu_agent, find_kernarg, NULL);

    // Allocate 64KB from kernarg pool
    void *hsa_buf = NULL;
    hsa_amd_memory_pool_allocate(sys_pool, 65536, 0, &hsa_buf);
    memset(hsa_buf, 0xAA, 65536);

    // Allocate 64KB from heap (normal WB memory)
    void *heap_buf = malloc(65536);
    memset(heap_buf, 0xBB, 65536);

    int count = 1024;  // 1024 reads per iteration
    int stride = 64;   // cache line stride

    long long t_hsa = time_reads((volatile unsigned int *)hsa_buf, count, stride);
    long long t_heap = time_reads((volatile unsigned int *)heap_buf, count, stride);

    printf("HSA kernarg pool: %lld ns for %d reads (%.1f ns/read)\n",
           t_hsa, count * 100, (double)t_hsa / (count * 100));
    printf("Heap (WB cached):  %lld ns for %d reads (%.1f ns/read)\n",
           t_heap, count * 100, (double)t_heap / (count * 100));
    printf("Ratio: %.1fx\n", (double)t_hsa / t_heap);
    printf("If ratio > 10x, HSA pool is likely UC/WC (not cached)\n");
    printf("If ratio ~1x, HSA pool is cached (WB)\n");

    hsa_amd_memory_pool_free(hsa_buf);
    free(heap_buf);
    hsa_shut_down();
    return 0;
}
