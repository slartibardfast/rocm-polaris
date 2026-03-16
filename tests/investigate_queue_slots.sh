#!/bin/bash
# Phase 7 Investigation: Which MEC pipe/queue slots work?
#
# On clean boot, creates queues one at a time and tests each.
# Dumps HQD registers to compare working vs non-working slots.
#
# Run immediately after cold reboot before any GPU resets accumulate.

set -euo pipefail
cd /home/llm/rocm-polaris

echo "=========================================="
echo "Phase 7: MEC Queue Slot Investigation"
echo "=========================================="

# Step 0: Verify clean state
echo ""
echo "=== Step 0: GPU State ==="
uname -r
RESETS=$(sudo dmesg 2>/dev/null | grep -c "GART.*enabled" || echo "?")
echo "GPU resets since boot: $RESETS"
if [ "$RESETS" -gt 3 ]; then
    echo "WARNING: GPU has been reset $RESETS times. Results may be unreliable."
    echo "Consider a cold reboot first."
fi

# Step 1: Build the queue probe test
echo ""
echo "=== Step 1: Build queue probe ==="
cat > /tmp/queue_probe.c << 'EOF'
// Creates N queues and tests each with a barrier dispatch.
// Reports which queues work and which hang.
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>

static hsa_agent_t gpu;
static hsa_status_t find_gpu(hsa_agent_t a, void* d) {
    hsa_device_type_t t;
    hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
    if (t == HSA_DEVICE_TYPE_GPU) { gpu = a; return HSA_STATUS_INFO_BREAK; }
    return HSA_STATUS_SUCCESS;
}

static volatile int timed_out = 0;
static void alarm_handler(int sig) { timed_out = 1; }

int main(int argc, char** argv) {
    int max_queues = argc > 1 ? atoi(argv[1]) : 8;

    setbuf(stdout, NULL);
    signal(SIGALRM, alarm_handler);

    hsa_init();
    hsa_iterate_agents(find_gpu, NULL);

    char name[64];
    hsa_agent_get_info(gpu, HSA_AGENT_INFO_NAME, name);
    printf("GPU: %s, testing %d queues\n\n", name, max_queues);

    for (int i = 0; i < max_queues; i++) {
        hsa_queue_t* q;
        hsa_status_t s = hsa_queue_create(gpu, 128, HSA_QUEUE_TYPE_SINGLE,
                                           NULL, NULL, UINT32_MAX, UINT32_MAX, &q);
        if (s != HSA_STATUS_SUCCESS) {
            printf("Queue %d: CREATE FAILED (status=%d)\n", i, s);
            continue;
        }

        // Submit a barrier with completion signal
        hsa_signal_t sig;
        hsa_signal_create(1, 0, NULL, &sig);

        uint64_t idx = hsa_queue_load_write_index_relaxed(q);
        hsa_barrier_and_packet_t* pkt = (hsa_barrier_and_packet_t*)
            ((char*)q->base_address + (idx & (q->size - 1)) * 64);
        memset(pkt, 0, sizeof(*pkt));
        pkt->completion_signal = sig;
        __atomic_store_n((uint16_t*)pkt,
                         (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
                         (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE),
                         __ATOMIC_RELEASE);
        hsa_queue_store_write_index_relaxed(q, idx + 1);
        hsa_signal_store_relaxed(q->doorbell_signal, idx);

        // Wait with 2-second timeout
        timed_out = 0;
        alarm(2);
        hsa_signal_value_t val = hsa_signal_wait_scacquire(
            sig, HSA_SIGNAL_CONDITION_EQ, 0, 2000000000ULL, HSA_WAIT_STATE_ACTIVE);
        alarm(0);

        uint64_t rid = hsa_queue_load_read_index_relaxed(q);

        if (val == 0 && !timed_out) {
            printf("Queue %d: PASS (read_idx=%lu)\n", i, rid);
        } else {
            printf("Queue %d: FAIL (signal=%ld, read_idx=%lu, timeout=%d)\n",
                   i, val, rid, timed_out);
        }

        hsa_signal_destroy(sig);
        // Don't destroy queue — keep it allocated to test slot exhaustion
    }

    printf("\nDumping HQD state...\n");
    fflush(stdout);
    system("sudo cat /sys/kernel/debug/kfd/hqds 2>/dev/null");

    printf("\nDone.\n");
    hsa_shut_down();
    return 0;
}
EOF

gcc -o /tmp/queue_probe /tmp/queue_probe.c \
    -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -Wl,-rpath,/opt/rocm/lib
echo "Built."

# Step 2: Run the probe
echo ""
echo "=== Step 2: Queue Probe ==="
HSA_ENABLE_INTERRUPT=0 timeout 30 /tmp/queue_probe 8

# Step 3: Also test with hipMemcpy + kernel to see which CLR queue hangs
echo ""
echo "=== Step 3: HIP Queue Test ==="
cat > /tmp/hip_queue_test.cpp << 'CPPEOF'
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <signal.h>
#include <unistd.h>
__global__ void set42(int* p) { *p = 42; }
static volatile int step = 0;
static void handler(int) { fprintf(stderr, "HANG at step %d\n", step); _exit(2); }
int main() {
    setbuf(stdout, NULL);
    signal(SIGALRM, handler);
    int *d, h;
    hipMalloc(&d, sizeof(int));

    // Test 1: hipMemset (blit queue)
    step = 1; alarm(5);
    hipMemset(d, 0, sizeof(int));
    alarm(0);
    printf("Step 1 (hipMemset): OK\n");

    // Test 2: hipMemcpy H2D (blit queue)
    step = 2; alarm(5);
    h = 99;
    hipMemcpy(d, &h, sizeof(int), hipMemcpyHostToDevice);
    alarm(0);
    printf("Step 2 (hipMemcpy H2D): OK\n");

    // Test 3: kernel launch (compute queue)
    step = 3; alarm(5);
    set42<<<1,1>>>(d);
    hipDeviceSynchronize();
    alarm(0);
    printf("Step 3 (kernel launch): OK\n");

    // Test 4: hipMemcpy D2H (blit queue)
    step = 4; alarm(5);
    hipMemcpy(&h, d, sizeof(int), hipMemcpyDeviceToHost);
    alarm(0);
    printf("Step 4 (hipMemcpy D2H): OK, result=%d (expect 42)\n", h);

    hipFree(d);
    printf("All steps passed!\n");
    _exit(0);
}
CPPEOF
/opt/rocm/bin/hipcc --offload-arch=gfx803 -o /tmp/hip_queue_test /tmp/hip_queue_test.cpp 2>/dev/null
echo "Built."
timeout -k 2 30 /tmp/hip_queue_test

echo ""
echo "=== Step 4: Final GPU State ==="
RESETS_AFTER=$(sudo dmesg 2>/dev/null | grep -c "GART.*enabled" || echo "?")
echo "GPU resets: $RESETS -> $RESETS_AFTER"
