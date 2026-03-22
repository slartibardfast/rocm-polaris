// test_hsa_hdp_flush.c — Verify hsa_amd_hdp_flush_wait() HSA API export
//
// Tests:
//   3a: Symbol resolves (link succeeds)
//   3b: Returns HSA_STATUS_SUCCESS on GPU agent
//   3c: Returns HSA_STATUS_ERROR_INVALID_AGENT on CPU agent
//   3d: nm -D shows symbol exported (manual check)
//
// Build: gcc -O2 -o test_hsa_hdp_flush test_hsa_hdp_flush.c \
//        -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 \
//        -Wl,-rpath,/opt/rocm/lib
// Run:   ./test_hsa_hdp_flush

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdio.h>
#include <stdlib.h>

static hsa_agent_t g_gpu = {0};
static hsa_agent_t g_cpu = {0};

static hsa_status_t find_agents(hsa_agent_t agent, void *data) {
    (void)data;
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_GPU && g_gpu.handle == 0)
        g_gpu = agent;
    else if (type == HSA_DEVICE_TYPE_CPU && g_cpu.handle == 0)
        g_cpu = agent;
    return HSA_STATUS_SUCCESS;
}

int main() {
    hsa_status_t s;
    int pass = 0, fail = 0;

    s = hsa_init();
    if (s != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "hsa_init failed: %d\n", s);
        return 1;
    }

    hsa_iterate_agents(find_agents, NULL);
    if (g_gpu.handle == 0) {
        fprintf(stderr, "No GPU agent found\n");
        hsa_shut_down();
        return 1;
    }

    // Test 3a: Symbol resolved (we got here, link succeeded)
    printf("Test 3a: Symbol resolves (link)... PASS\n");
    pass++;

    // Test 3b: Returns SUCCESS on GPU agent
    s = hsa_amd_hdp_flush_wait(g_gpu, 10000);
    if (s == HSA_STATUS_SUCCESS) {
        printf("Test 3b: GPU agent returns SUCCESS... PASS\n");
        pass++;
    } else {
        printf("Test 3b: GPU agent returns %d (expected SUCCESS)... FAIL\n", s);
        fail++;
    }

    // Test 3c: Returns ERROR_INVALID_AGENT on CPU agent
    if (g_cpu.handle != 0) {
        s = hsa_amd_hdp_flush_wait(g_cpu, 10000);
        if (s == HSA_STATUS_ERROR_INVALID_AGENT) {
            printf("Test 3c: CPU agent returns ERROR_INVALID_AGENT... PASS\n");
            pass++;
        } else {
            printf("Test 3c: CPU agent returns %d (expected ERROR_INVALID_AGENT)... FAIL\n", s);
            fail++;
        }
    } else {
        printf("Test 3c: No CPU agent found, skipped\n");
    }

    hsa_shut_down();

    printf("\n=== Results: %d PASS, %d FAIL ===\n", pass, fail);
    printf("Manual check: nm -D /opt/rocm/lib/libhsa-runtime64.so | grep hsa_amd_hdp_flush_wait\n");
    return fail > 0 ? 1 : 0;
}
