// Reproduce the exact three-queue cross-barrier hang pattern.
//
// From the debug log, the failing pattern is:
//   Queue B: barrier(no dep, sig=S1) → dispatch(sig=S2) → barrier(no dep, sig=S3) → dispatch(sig=S4)
//   Queue C: barrier(dep=S1, no sig) → dispatch(sig=X) → barrier(dep=S3, no sig) → dispatch(sig=Y)
//
// Queue C hangs because:
//   1. S1 isn't decremented by GPU (no PCIe atomics)
//   2. Our bounce buffer decrements S1 after RPTR advances on Queue B
//   3. Queue C's barrier should then resolve, but RPTR doesn't advance
//
// This test polls RPTR directly (like the bounce buffer does) to see if
// it advances after we manually decrement the dep signal.

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/amd_hsa_signal.h>
#include <hsa/amd_hsa_queue.h>
#include <x86intrin.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <unistd.h>

#define HSA_CHECK(call) do { hsa_status_t s = (call); \
    if (s) { fprintf(stderr, "HSA err %d at %s:%d\n", s, __FILE__, __LINE__); _exit(1); } } while(0)

static hsa_agent_t gpu, cpu;
static hsa_status_t find_agents(hsa_agent_t a, void*) {
    hsa_device_type_t t;
    hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
    if (t == HSA_DEVICE_TYPE_GPU) gpu = a;
    if (t == HSA_DEVICE_TYPE_CPU) cpu = a;
    return HSA_STATUS_SUCCESS;
}

static void submit_barrier(hsa_queue_t* q, hsa_signal_t dep, hsa_signal_t comp) {
    uint64_t idx = hsa_queue_load_write_index_relaxed(q);
    hsa_barrier_and_packet_t* pkt = (hsa_barrier_and_packet_t*)
        ((char*)q->base_address + (idx & (q->size - 1)) * 64);
    memset(pkt, 0, sizeof(*pkt));
    pkt->dep_signal[0] = dep;
    pkt->completion_signal = comp;
    __atomic_store_n((uint16_t*)pkt,
                     (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
                     (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE),
                     __ATOMIC_RELEASE);
    hsa_queue_store_write_index_relaxed(q, idx + 1);
    hsa_signal_store_relaxed(q->doorbell_signal, idx);
}

static void cpu_decrement(hsa_signal_t sig) {
    amd_signal_t* s = reinterpret_cast<amd_signal_t*>(sig.handle);
    __atomic_sub_fetch(&s->value, 1, __ATOMIC_RELEASE);
    _mm_clflush(const_cast<void*>(static_cast<volatile void*>(&s->value)));
    _mm_mfence();
}

// Read RPTR from the queue's read_dispatch_id (which our bounce buffer updates)
static uint64_t read_dispatch_id(hsa_queue_t* q) {
    return hsa_queue_load_read_index_relaxed(q);
}

int main() {
    setbuf(stdout, NULL);
    HSA_CHECK(hsa_init());
    hsa_iterate_agents(find_agents, NULL);

    char name[64];
    hsa_agent_get_info(gpu, HSA_AGENT_INFO_NAME, name);
    printf("GPU: %s\n", name);

    // Create two queues (mimicking CLR's blit + utility queues)
    hsa_queue_t *qB, *qC;
    HSA_CHECK(hsa_queue_create(gpu, 128, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX, UINT32_MAX, &qB));
    HSA_CHECK(hsa_queue_create(gpu, 128, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX, UINT32_MAX, &qC));

    printf("Queue B: %p, Queue C: %p\n", qB, qC);

    // Signals
    hsa_signal_t S1, S2, S3, S4;  // Queue B signals
    hsa_signal_t X, Y;            // Queue C dispatch signals
    HSA_CHECK(hsa_signal_create(1, 0, NULL, &S1));
    HSA_CHECK(hsa_signal_create(1, 0, NULL, &S2));
    HSA_CHECK(hsa_signal_create(1, 0, NULL, &S3));
    HSA_CHECK(hsa_signal_create(1, 0, NULL, &S4));
    HSA_CHECK(hsa_signal_create(1, 0, NULL, &X));
    HSA_CHECK(hsa_signal_create(1, 0, NULL, &Y));

    // === Round 1 ===
    printf("\n=== Round 1 ===\n");

    // Queue B: barrier(no dep, sig=S1)
    printf("QB: submit barrier(sig=S1)\n");
    submit_barrier(qB, {0}, S1);

    // Wait for QB's barrier to be processed (RPTR should advance)
    usleep(50000);
    printf("QB read_idx=%lu (expect 1)\n", read_dispatch_id(qB));
    printf("S1 value=%ld (expect 1 on no-atomics)\n", hsa_signal_load_scacquire(S1));

    // Queue C: barrier(dep=S1, no sig) + dispatch(sig=X)
    // This matches the real pattern from the debug log:
    //   idx=0: type=3 (BARRIER_AND) sig=0x0 dep0=S1
    //   idx=1: type=2 (KERNEL_DISPATCH) sig=X
    // We use a barrier as a dispatch stand-in since we don't have a kernel
    printf("QC: submit barrier(dep=S1, no sig) + barrier(sig=X) [4-pkt pattern]\n");
    {
        uint64_t idx = hsa_queue_load_write_index_relaxed(qC);
        // Packet 0: barrier, dep=S1, no completion signal
        hsa_barrier_and_packet_t* p0 = (hsa_barrier_and_packet_t*)
            ((char*)qC->base_address + (idx & (qC->size - 1)) * 64);
        memset(p0, 0, sizeof(*p0));
        p0->dep_signal[0] = S1;

        // Packet 1: barrier (stand-in for dispatch), sig=X
        hsa_barrier_and_packet_t* p1 = (hsa_barrier_and_packet_t*)
            ((char*)qC->base_address + ((idx+1) & (qC->size - 1)) * 64);
        memset(p1, 0, sizeof(*p1));
        p1->completion_signal = X;

        uint16_t hdr = (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
                       (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
        __atomic_store_n((uint16_t*)p1, hdr, __ATOMIC_RELEASE);
        __atomic_store_n((uint16_t*)p0, hdr, __ATOMIC_RELEASE);

        hsa_queue_store_write_index_relaxed(qC, idx + 2);
        hsa_signal_store_relaxed(qC->doorbell_signal, idx + 1);
    }

    usleep(50000);
    printf("QC read_idx=%lu (expect 0 — stalled on S1)\n", read_dispatch_id(qC));

    // Now decrement S1 from CPU (mimicking bounce buffer)
    printf("CPU: decrement S1\n");
    cpu_decrement(S1);
    printf("S1 value=%ld (expect 0)\n", hsa_signal_load_scacquire(S1));

    // Wait for QC to resolve
    printf("Waiting for X (QC completion, 3s)...\n");
    hsa_signal_value_t xv = hsa_signal_wait_scacquire(X, HSA_SIGNAL_CONDITION_EQ, 0,
                                                       3000000000ULL, HSA_WAIT_STATE_ACTIVE);
    printf("X value=%ld, QC read_idx=%lu\n", xv, read_dispatch_id(qC));

    if (xv != 0) {
        printf("FAIL: QC barrier didn't resolve after S1 decrement!\n");
        printf("  Trying to re-ring QC doorbell...\n");
        hsa_signal_store_relaxed(qC->doorbell_signal,
                                 hsa_queue_load_write_index_relaxed(qC) - 1);
        xv = hsa_signal_wait_scacquire(X, HSA_SIGNAL_CONDITION_EQ, 0,
                                        3000000000ULL, HSA_WAIT_STATE_ACTIVE);
        printf("  After re-ring: X=%ld, QC read_idx=%lu\n", xv, read_dispatch_id(qC));
        if (xv == 0)
            printf("PASS (needed re-ring)\n");
        else
            printf("FAIL even after re-ring\n");
    } else {
        printf("PASS: QC resolved after CPU decrement of S1\n");
    }

    // === Round 2 (chain) ===
    printf("\n=== Round 2 (chained) ===\n");

    // Queue B: barrier(no dep, sig=S3)
    printf("QB: submit barrier(sig=S3)\n");
    submit_barrier(qB, {0}, S3);
    usleep(50000);
    printf("QB read_idx=%lu, S3=%ld\n", read_dispatch_id(qB), hsa_signal_load_scacquire(S3));

    // Queue C: barrier(dep=S3) + barrier(sig=Y)
    printf("QC: submit barrier(dep=S3) + barrier(sig=Y)\n");
    {
        uint64_t idx = hsa_queue_load_write_index_relaxed(qC);
        hsa_barrier_and_packet_t* p0 = (hsa_barrier_and_packet_t*)
            ((char*)qC->base_address + (idx & (qC->size - 1)) * 64);
        memset(p0, 0, sizeof(*p0));
        p0->dep_signal[0] = S3;

        hsa_barrier_and_packet_t* p1 = (hsa_barrier_and_packet_t*)
            ((char*)qC->base_address + ((idx+1) & (qC->size - 1)) * 64);
        memset(p1, 0, sizeof(*p1));
        p1->completion_signal = Y;

        uint16_t hdr = (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
                       (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
        __atomic_store_n((uint16_t*)p1, hdr, __ATOMIC_RELEASE);
        __atomic_store_n((uint16_t*)p0, hdr, __ATOMIC_RELEASE);

        hsa_queue_store_write_index_relaxed(qC, idx + 2);
        hsa_signal_store_relaxed(qC->doorbell_signal, idx + 1);
    }

    usleep(50000);
    printf("QC read_idx=%lu (expect 2 — stalled on S3)\n", read_dispatch_id(qC));

    printf("CPU: decrement S3\n");
    cpu_decrement(S3);

    printf("Waiting for Y (3s)...\n");
    hsa_signal_value_t yv = hsa_signal_wait_scacquire(Y, HSA_SIGNAL_CONDITION_EQ, 0,
                                                       3000000000ULL, HSA_WAIT_STATE_ACTIVE);
    printf("Y=%ld, QC read_idx=%lu\n", yv, read_dispatch_id(qC));
    printf("%s\n", yv == 0 ? "PASS" : "FAIL");

    // Cleanup
    hsa_signal_destroy(S1); hsa_signal_destroy(S2);
    hsa_signal_destroy(S3); hsa_signal_destroy(S4);
    hsa_signal_destroy(X); hsa_signal_destroy(Y);
    hsa_queue_destroy(qB); hsa_queue_destroy(qC);
    hsa_shut_down();
    return (xv != 0 || yv != 0);
}
