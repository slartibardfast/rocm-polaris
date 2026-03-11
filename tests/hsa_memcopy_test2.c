#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdio.h>
#include <string.h>

static hsa_agent_t gpu_agent, cpu_agent;
static hsa_amd_memory_pool_t sys_pool, vram_pool;

static hsa_status_t find_gpu(hsa_agent_t agent, void *data) {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_GPU) { gpu_agent = agent; }
    if (type == HSA_DEVICE_TYPE_CPU) { cpu_agent = agent; }
    return HSA_STATUS_SUCCESS;
}

static hsa_status_t find_pool(hsa_amd_memory_pool_t pool, void *data) {
    hsa_amd_segment_t seg;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
    if (seg == HSA_AMD_SEGMENT_GLOBAL) {
        hsa_amd_memory_pool_global_flag_t flags;
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
        if (data == &sys_pool && (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT))
            sys_pool = pool;
        if (data == &vram_pool && !(flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT))
            vram_pool = pool;
    }
    return HSA_STATUS_SUCCESS;
}

int main() {
    printf("1. hsa_init\n"); fflush(stdout);
    hsa_init();
    hsa_iterate_agents(find_gpu, NULL);
    hsa_amd_agent_iterate_memory_pools(cpu_agent, find_pool, &sys_pool);
    hsa_amd_agent_iterate_memory_pools(gpu_agent, find_pool, &vram_pool);

    printf("2. allocating\n"); fflush(stdout);
    void *sys_src = NULL, *sys_dst = NULL, *vram_buf = NULL;
    hsa_amd_memory_pool_allocate(sys_pool, 4096, 0, &sys_src);
    hsa_amd_memory_pool_allocate(sys_pool, 4096, 0, &sys_dst);
    hsa_amd_memory_pool_allocate(vram_pool, 4096, 0, &vram_buf);
    printf("   sys_src=%p sys_dst=%p vram=%p\n", sys_src, sys_dst, vram_buf); fflush(stdout);

    printf("3. allow_access\n"); fflush(stdout);
    hsa_agent_t agents[2] = {cpu_agent, gpu_agent};
    hsa_amd_agents_allow_access(2, agents, NULL, sys_src);
    hsa_amd_agents_allow_access(2, agents, NULL, sys_dst);
    hsa_amd_agents_allow_access(2, agents, NULL, vram_buf);

    printf("4. fill src\n"); fflush(stdout);
    unsigned char *src = (unsigned char *)sys_src;
    for (int i = 0; i < 256; i++) src[i] = (unsigned char)(i ^ 0xAA);

    printf("5. sys->vram copy\n"); fflush(stdout);
    hsa_status_t s = hsa_memory_copy(vram_buf, sys_src, 256);
    printf("   status=%d\n", s); fflush(stdout);

    printf("6. vram->sys copy\n"); fflush(stdout);
    memset(sys_dst, 0xFF, 256);
    s = hsa_memory_copy(sys_dst, vram_buf, 256);
    printf("   status=%d\n", s); fflush(stdout);

    printf("7. checking\n"); fflush(stdout);
    unsigned char *dst = (unsigned char *)sys_dst;
    int ok = 1;
    for (int i = 0; i < 256; i++) {
        if (dst[i] != (unsigned char)(i ^ 0xAA)) {
            printf("   MISMATCH at %d: got 0x%02x expected 0x%02x\n",
                   i, dst[i], (unsigned char)(i ^ 0xAA));
            ok = 0; break;
        }
    }
    printf("   result: %s\n", ok ? "PASS" : "FAIL"); fflush(stdout);

    printf("8. cleanup\n"); fflush(stdout);
    hsa_amd_memory_pool_free(vram_buf);
    hsa_amd_memory_pool_free(sys_src);
    hsa_amd_memory_pool_free(sys_dst);
    hsa_shut_down();
    printf("Done.\n");
    return ok ? 0 : 1;
}
