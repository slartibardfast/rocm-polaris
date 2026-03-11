#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static hsa_agent_t gpu_agent, cpu_agent;
static hsa_amd_memory_pool_t sys_pool;

static hsa_status_t find_agents(hsa_agent_t agent, void *data) {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_GPU) gpu_agent = agent;
    if (type == HSA_DEVICE_TYPE_CPU) cpu_agent = agent;
    return HSA_STATUS_SUCCESS;
}

static hsa_status_t find_kernarg(hsa_amd_memory_pool_t pool, void *data) {
    hsa_amd_segment_t seg;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
    if (seg == HSA_AMD_SEGMENT_GLOBAL) {
        hsa_amd_memory_pool_global_flag_t flags;
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
        if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT)
            sys_pool = pool;
    }
    return HSA_STATUS_SUCCESS;
}

static void check_smaps(void *ptr) {
    char cmd[256];
    unsigned long addr = (unsigned long)ptr;
    // Read smaps to find the VMA containing this address
    FILE *f = fopen("/proc/self/smaps", "r");
    if (!f) { perror("smaps"); return; }
    char line[512];
    int found = 0;
    while (fgets(line, sizeof(line), f)) {
        unsigned long start, end;
        if (sscanf(line, "%lx-%lx", &start, &end) == 2) {
            found = (addr >= start && addr < end);
            if (found) printf("  VMA: %s", line);
        } else if (found) {
            if (strncmp(line, "VmFlags:", 8) == 0 ||
                strncmp(line, "Size:", 5) == 0)
                printf("  %s", line);
        }
        if (found && strncmp(line, "VmFlags:", 8) == 0) break;
    }
    fclose(f);
}

int main() {
    hsa_init();
    hsa_iterate_agents(find_agents, NULL);
    hsa_amd_agent_iterate_memory_pools(cpu_agent, find_kernarg, NULL);

    void *ptr = NULL;
    hsa_amd_memory_pool_allocate(sys_pool, 4096, 0, &ptr);
    printf("kernarg alloc: %p\n", ptr);
    check_smaps(ptr);

    // Test: write from CPU, read back (should work)
    volatile unsigned int *p = (volatile unsigned int *)ptr;
    p[0] = 0xCAFEBABE;
    unsigned int v = p[0];
    printf("  CPU write/read: wrote 0xCAFEBABE, read 0x%08x → %s\n",
           v, v == 0xCAFEBABE ? "OK" : "FAIL");

    // Also check stack memory (known WB)
    int stack_var = 42;
    printf("\nstack var: %p\n", &stack_var);
    check_smaps(&stack_var);

    hsa_amd_memory_pool_free(ptr);
    hsa_shut_down();
    return 0;
}
