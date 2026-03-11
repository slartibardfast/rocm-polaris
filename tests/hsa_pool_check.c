#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdio.h>
#include <stdlib.h>

static hsa_agent_t gpu_agent, cpu_agent;

static hsa_status_t find_agents(hsa_agent_t agent, void *data) {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_GPU) gpu_agent = agent;
    if (type == HSA_DEVICE_TYPE_CPU) cpu_agent = agent;
    return HSA_STATUS_SUCCESS;
}

static hsa_status_t check_pool(hsa_amd_memory_pool_t pool, void *data) {
    hsa_amd_segment_t seg;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
    if (seg != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;

    hsa_amd_memory_pool_global_flag_t flags;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);

    const char *name = "other";
    if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) name = "kernarg";
    else if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) name = "fine_grain";
    else if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) name = "coarse_grain";

    // Check GPU access
    hsa_amd_memory_pool_access_t access;
    hsa_amd_agent_memory_pool_get_info(gpu_agent, pool,
        HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);

    const char *access_str = "unknown";
    switch (access) {
        case HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT: access_str = "ALLOWED_BY_DEFAULT"; break;
        case HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT: access_str = "DISALLOWED_BY_DEFAULT"; break;
        case HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED: access_str = "NEVER_ALLOWED"; break;
    }

    printf("  pool type=%-12s flags=0x%x gpu_access=%s\n", name, flags, access_str);

    // If it's kernarg, try allocating and checking PtrInfo
    if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) {
        void *ptr = NULL;
        hsa_amd_memory_pool_allocate(pool, 4096, 0, &ptr);
        printf("    allocated ptr=%p\n", ptr);

        hsa_amd_pointer_info_t info = {};
        info.size = sizeof(info);
        uint32_t count = 0;
        hsa_agent_t *accessible = NULL;
        hsa_amd_pointer_info(ptr, &info, NULL, &count, &accessible);
        printf("    PtrInfo: type=%d agentOwner=%lu accessible_count=%u\n",
               info.type, info.agentOwner.handle, count);
        for (uint32_t i = 0; i < count; i++) {
            hsa_device_type_t dt;
            hsa_agent_get_info(accessible[i], HSA_AGENT_INFO_DEVICE, &dt);
            printf("      agent[%u]: handle=%lu type=%s\n", i, accessible[i].handle,
                   dt == HSA_DEVICE_TYPE_GPU ? "GPU" : "CPU");
        }
        free(accessible);
        hsa_amd_memory_pool_free(ptr);
    }

    return HSA_STATUS_SUCCESS;
}

int main() {
    hsa_init();
    hsa_iterate_agents(find_agents, NULL);

    printf("CPU pools:\n");
    hsa_amd_agent_iterate_memory_pools(cpu_agent, check_pool, NULL);
    printf("GPU pools:\n");
    hsa_amd_agent_iterate_memory_pools(gpu_agent, check_pool, NULL);

    hsa_shut_down();
    return 0;
}
