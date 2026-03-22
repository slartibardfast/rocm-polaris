#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdio.h>
#include <stdint.h>

hsa_status_t agent_cb(hsa_agent_t agent, void *data) {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type != HSA_DEVICE_TYPE_GPU) return HSA_STATUS_SUCCESS;

    hsa_amd_hdp_flush_t hdp = {0};
    hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_HDP_FLUSH, &hdp);
    
    printf("HDP_MEM_FLUSH_CNTL (now = GPU_HDP_FLUSH_REQ): %p\n", hdp.HDP_MEM_FLUSH_CNTL);
    printf("HDP_REG_FLUSH_CNTL (now = GPU_HDP_FLUSH_DONE): %p\n", hdp.HDP_REG_FLUSH_CNTL);
    
    if (!hdp.HDP_MEM_FLUSH_CNTL || !hdp.HDP_REG_FLUSH_CNTL) {
        printf("FAIL: pointers are NULL\n");
        return HSA_STATUS_SUCCESS;
    }
    
    volatile uint32_t *req = hdp.HDP_MEM_FLUSH_CNTL;
    volatile uint32_t *done = hdp.HDP_REG_FLUSH_CNTL;
    
    // Read DONE before REQ — should show current state
    uint32_t done_before = *done;
    printf("DONE register before REQ: 0x%08x\n", done_before);
    
    // Write REQ for all compute CPs
    uint32_t mask = 0x3FC;
    printf("Writing REQ mask: 0x%08x\n", mask);
    *req = mask;
    
    // Poll DONE
    int polls = 0;
    while ((*done & mask) != mask) {
        polls++;
        if (polls > 1000000) {
            printf("TIMEOUT: DONE=0x%08x after %d polls (BIF remap may not forward reads)\n",
                   *done, polls);
            return HSA_STATUS_SUCCESS;
        }
    }
    uint32_t done_after = *done;
    printf("DONE register after REQ: 0x%08x (polls=%d)\n", done_after, polls);
    printf("SUCCESS: HDP flush REQ/DONE protocol works from userspace!\n");
    
    return HSA_STATUS_SUCCESS;
}

int main() {
    hsa_init();
    hsa_iterate_agents(agent_cb, NULL);
    hsa_shut_down();
}
