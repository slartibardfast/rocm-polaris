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
    printf("=== HSA Memory Copy Test ===\n");
    hsa_init();
    hsa_iterate_agents(find_gpu, NULL);
    hsa_amd_agent_iterate_memory_pools(cpu_agent, find_pool, &sys_pool);
    hsa_amd_agent_iterate_memory_pools(gpu_agent, find_pool, &vram_pool);

    // Allocate system memory (kernarg pool, GPU-accessible)
    void *sys_src = NULL, *sys_dst = NULL;
    hsa_amd_memory_pool_allocate(sys_pool, 4096, 0, &sys_src);
    hsa_amd_memory_pool_allocate(sys_pool, 4096, 0, &sys_dst);
    printf("sys_src=%p sys_dst=%p\n", sys_src, sys_dst);

    // Fill source with known pattern
    unsigned char *src = (unsigned char *)sys_src;
    for (int i = 0; i < 256; i++) src[i] = (unsigned char)(i ^ 0xAA);

    // Test 1: System-to-system copy (should use memcpy path)
    memset(sys_dst, 0xFF, 256);
    hsa_status_t s = hsa_memory_copy(sys_dst, sys_src, 256);
    printf("sys->sys copy: status=%d\n", s);
    unsigned char *dst = (unsigned char *)sys_dst;
    int ok = 1;
    for (int i = 0; i < 256; i++) {
        if (dst[i] != (unsigned char)(i ^ 0xAA)) { ok = 0; break; }
    }
    printf("  result: %s\n", ok ? "PASS" : "MISMATCH");
    if (!ok) {
        printf("  first 16:");
        for (int i = 0; i < 16; i++) printf(" %02x", dst[i]);
        printf("\n");
    }

    // Allocate VRAM
    void *vram_buf = NULL;
    hsa_amd_memory_pool_allocate(vram_pool, 4096, 0, &vram_buf);
    printf("vram_buf=%p\n", vram_buf);

    // Allow GPU and CPU to access all buffers
    hsa_agent_t agents[2] = {cpu_agent, gpu_agent};
    hsa_amd_agents_allow_access(2, agents, NULL, sys_src);
    hsa_amd_agents_allow_access(2, agents, NULL, sys_dst);
    hsa_amd_agents_allow_access(2, agents, NULL, vram_buf);

    // Test 2: System->VRAM copy
    s = hsa_memory_copy(vram_buf, sys_src, 256);
    printf("sys->vram copy: status=%d\n", s);

    // Test 3: VRAM->System copy
    memset(sys_dst, 0xFF, 256);
    s = hsa_memory_copy(sys_dst, vram_buf, 256);
    printf("vram->sys copy: status=%d\n", s);
    ok = 1;
    for (int i = 0; i < 256; i++) {
        if (dst[i] != (unsigned char)(i ^ 0xAA)) { ok = 0; break; }
    }
    int fail_idx = -1;
    for (int i = 0; i < 256; i++) {
        if (dst[i] != (unsigned char)(i ^ 0xAA)) { fail_idx = i; break; }
    }
    printf("  round-trip result: %s\n", fail_idx < 0 ? "PASS" : "MISMATCH");
    if (fail_idx >= 0) {
        printf("  first mismatch at byte %d: got 0x%02x expected 0x%02x\n",
               fail_idx, dst[fail_idx], (unsigned char)(fail_idx ^ 0xAA));
        int start = (fail_idx / 16) * 16;
        printf("  bytes %d-%d:", start, start+15);
        for (int i = start; i < start+16 && i < 256; i++) printf(" %02x", dst[i]);
        printf("\n  expected:   ");
        for (int i = start; i < start+16 && i < 256; i++) printf(" %02x", (unsigned char)(i ^ 0xAA));
        printf("\n");
    }

    // Test 4: Fill VRAM with hsa_amd_memory_fill
    s = hsa_amd_memory_fill(vram_buf, 0xDEADBEEF, 64);
    printf("vram fill: status=%d\n", s);
    memset(sys_dst, 0, 256);
    s = hsa_memory_copy(sys_dst, vram_buf, 256);
    printf("fill->sys copy: status=%d\n", s);
    unsigned int *u32 = (unsigned int *)sys_dst;
    printf("  direct read:  0x%08x 0x%08x 0x%08x 0x%08x\n",
           u32[0], u32[1], u32[2], u32[3]);

    // Test if clflush helps (CPU cache coherency test)
    for (int i = 0; i < 256; i += 64)
        asm volatile("clflush (%0)" :: "r"((char*)sys_dst + i) : "memory");
    asm volatile("mfence" ::: "memory");
    printf("  after clflush: 0x%08x 0x%08x 0x%08x 0x%08x\n",
           u32[0], u32[1], u32[2], u32[3]);

    // Also try reading via a Lock'd stack buffer (known-working path)
    unsigned char vram_check[256];
    memset(vram_check, 0x55, 256);
    s = hsa_memory_copy(vram_check, vram_buf, 256);
    printf("  via Lock'd buf: 0x%08x 0x%08x 0x%08x 0x%08x\n",
           ((unsigned int*)vram_check)[0], ((unsigned int*)vram_check)[1],
           ((unsigned int*)vram_check)[2], ((unsigned int*)vram_check)[3]);

    hsa_amd_memory_pool_free(vram_buf);
    hsa_amd_memory_pool_free(sys_src);
    hsa_amd_memory_pool_free(sys_dst);
    hsa_shut_down();
    printf("Done.\n");
    return 0;
}
