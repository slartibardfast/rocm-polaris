// List all memory pools and their properties
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdio.h>

static int pool_idx = 0;

static hsa_status_t show_pool(hsa_amd_memory_pool_t pool, void *data) {
    hsa_amd_segment_t seg;
    uint32_t flags = 0;
    size_t size = 0;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SIZE, &size);
    if (seg == HSA_AMD_SEGMENT_GLOBAL)
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);

    const char *seg_name = seg == HSA_AMD_SEGMENT_GLOBAL ? "GLOBAL" :
                            seg == HSA_AMD_SEGMENT_GROUP ? "GROUP" :
                            seg == HSA_AMD_SEGMENT_READONLY ? "READONLY" : "?";
    printf("  Pool %d: seg=%s size=%zuMB flags=0x%x", pool_idx++, seg_name, size/(1024*1024), flags);
    if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) printf(" KERNARG");
    if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) printf(" FINE_GRAINED");
    if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) printf(" COARSE_GRAINED");

    // Check if GPU-accessible
    hsa_amd_memory_pool_access_t access;
    hsa_agent_t *agent = (hsa_agent_t*)data;
    if (agent->handle) {
        hsa_amd_agent_memory_pool_get_info(*agent, pool,
            HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
        printf(" gpu_access=%d", access);
    }
    printf(" handle=0x%lx\n", pool.handle);
    return HSA_STATUS_SUCCESS;
}

static hsa_status_t show_agent(hsa_agent_t agent, void *data) {
    char name[64];
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    printf("\nAgent: %s (type=%d)\n", name, type);
    pool_idx = 0;

    hsa_agent_t *gpu = (hsa_agent_t*)data;
    hsa_amd_agent_iterate_memory_pools(agent, show_pool, gpu);
    return HSA_STATUS_SUCCESS;
}

int main() {
    setbuf(stdout, NULL);
    hsa_init();

    // Find GPU first
    hsa_agent_t gpu = {0};
    hsa_status_t (*find_gpu)(hsa_agent_t, void*) = ({
        hsa_status_t fn(hsa_agent_t a, void *d) {
            hsa_device_type_t t;
            hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
            if (t == HSA_DEVICE_TYPE_GPU) { *(hsa_agent_t*)d = a; return HSA_STATUS_INFO_BREAK; }
            return HSA_STATUS_SUCCESS;
        } fn;
    });
    hsa_iterate_agents(find_gpu, &gpu);

    printf("=== Memory Pool Info ===\n");
    hsa_iterate_agents(show_agent, &gpu);

    // Now create a signal and check which pool it's in
    hsa_signal_t signal;
    hsa_signal_create(1, 0, NULL, &signal);
    printf("\nSignal handle: 0x%lx\n", signal.handle);
    printf("Signal is at CPU VA %p\n", (void*)signal.handle);

    // Try to identify which pool owns this memory
    hsa_amd_pointer_info_t info;
    info.size = sizeof(info);
    hsa_status_t s = hsa_amd_pointer_info((void*)signal.handle, &info, NULL, NULL, NULL);
    printf("hsa_amd_pointer_info: status=%d\n", s);
    if (s == HSA_STATUS_SUCCESS) {
        printf("  type=%d (0=unknown, 1=HSA, 2=locked, 3=ipc, 4=graphics)\n", info.type);
        printf("  agentBase=%p, hostBase=%p\n", info.agentBaseAddress, info.hostBaseAddress);
        printf("  sizeInBytes=%zu\n", info.sizeInBytes);
        printf("  agentOwner.handle=0x%lx\n", info.agentOwner.handle);
    }

    hsa_signal_destroy(signal);
    hsa_shut_down();
    return 0;
}
