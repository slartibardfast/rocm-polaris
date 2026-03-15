#!/bin/bash
# Post-reboot test plan for Phase 6c validation
# Run this after a CLEAN COLD REBOOT (not kexec)
#
# Prerequisites:
# - linux-lts-rocm-polaris 6.18.16-10 booted
# - hsa-rocr-polaris 7.2.0-9 installed (clean 0004, no NOP kick, with interrupt + Phase 6a sed)
# - hip-runtime-amd-polaris 7.2.0-2 installed (CLR fixes + cpu_wait_for_signal)
#
# Expected: 0 GPU resets at start

set -e
cd /home/llm/rocm-polaris

echo "=========================================="
echo "Post-Reboot Phase 6c Validation Test Suite"
echo "=========================================="
echo ""

# Step 0: Verify clean state
echo "=== Step 0: Clean GPU state ==="
uname -r
GPU_RESETS=$(sudo dmesg 2>/dev/null | grep -c "GART.*enabled" || echo "?")
echo "GPU resets since boot: $GPU_RESETS"
rocm-smi --showmeminfo vram 2>&1 | grep Used
echo ""

# Step 1: Rebuild test binaries (lost after reboot)
echo "=== Step 1: Build test binaries ==="
cat > /tmp/hip_sweep_test.cpp << 'CPPEOF'
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define HIP_CHECK(call) do { hipError_t err = (call); if (err != hipSuccess) { fprintf(stderr, "HIP error %d (%s)\n", err, hipGetErrorString(err)); exit(1); } } while(0)
int main() {
    size_t sizes[] = {64,128,256,512,1024,4096,8192,16384,32768,65536,131072,262144,1048576};
    int n=13, pass=0;
    hipDeviceProp_t p; HIP_CHECK(hipGetDeviceProperties(&p,0));
    printf("Device: %s\nTesting %d D2H sizes...\n\n",p.name,n);
    for(int i=0;i<n;i++){
        size_t sz=sizes[i]; printf("[%2d/%d] %7zu bytes: ",i+1,n,sz); fflush(stdout);
        unsigned char *d,*h=(unsigned char*)malloc(sz),*pat=(unsigned char*)malloc(sz);
        HIP_CHECK(hipMalloc(&d,sz));
        for(size_t j=0;j<sz;j++) pat[j]=(unsigned char)(j%251);
        HIP_CHECK(hipMemcpy(d,pat,sz,hipMemcpyHostToDevice));
        memset(h,0xBB,sz);
        HIP_CHECK(hipMemcpy(h,d,sz,hipMemcpyDeviceToHost));
        int errs=0; for(size_t j=0;j<sz;j++) if(h[j]!=pat[j]){errs++;break;}
        printf("%s\n",errs==0?"PASS":"FAIL"); if(!errs)pass++;
        free(pat);free(h);HIP_CHECK(hipFree(d));
    }
    printf("\n=== Results: %d/%d passed ===\n",pass,n);
    return pass!=n;
}
CPPEOF

cat > /tmp/stress_long.cpp << 'CPPEOF'
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <signal.h>
static volatile int op_num = 0;
static void alarm_handler(int) { fprintf(stderr, "\n*** HANG at op %d ***\n", op_num); _exit(2); }
#define HIP_CHECK(call) do { hipError_t err = (call); if (err != hipSuccess) { fprintf(stderr, "HIP error %d (%s) at op %d\n", err, hipGetErrorString(err), op_num); _exit(1); } } while(0)
int main() {
    setbuf(stdout, NULL);
    ::signal(SIGALRM, alarm_handler);
    void* d; HIP_CHECK(hipMalloc(&d, 4*1024*1024));
    char* h = (char*)malloc(3*1024*1024); memset(h, 0x42, 3*1024*1024);
    size_t sizes[] = {3584, 551936, 78848, 78848, 551936, 3584, 512, 512, 3584, 2996224, 2451456};
    int nsizes = sizeof(sizes)/sizeof(sizes[0]);
    for (int round = 0; round < 50; round++) {
        for (int i = 0; i < nsizes; i++) {
            size_t sz = sizes[i]; if (sz > 3*1024*1024) sz = 3*1024*1024;
            op_num = round * nsizes + i + 1;
            alarm(5);
            HIP_CHECK(hipMemcpy(d, h, sz, hipMemcpyHostToDevice));
            alarm(0);
        }
        printf("round %d done (%d ops)\n", round+1, op_num);
    }
    printf("All %d ops done!\n", op_num);
    free(h); HIP_CHECK(hipFree(d)); _exit(0);
}
CPPEOF

cat > /tmp/mix_compute_memcpy.cpp << 'CPPEOF'
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <signal.h>
#include <unistd.h>
__global__ void fill(float* p, float v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}
static volatile int op = 0;
static void handler(int) { fprintf(stderr, "HANG at op %d\n", op); _exit(2); }
int main() {
    setbuf(stdout, NULL);
    signal(SIGALRM, handler);
    const int N = 1024;
    float *d, *h = (float*)malloc(N * sizeof(float));
    hipMalloc(&d, N * sizeof(float));
    for (int i = 0; i < 200; i++) {
        op = i + 1;
        alarm(3);
        fill<<<(N+255)/256, 256>>>(d, (float)(i+1), N);
        hipMemcpy(h, d, N * sizeof(float), hipMemcpyDeviceToHost);
        alarm(0);
        if (h[0] != (float)(i+1)) { printf("[%d] FAIL: h[0]=%.1f expect %.1f\n", i+1, h[0], (float)(i+1)); _exit(1); }
        memset(h, 0, N * sizeof(float));
        hipMemcpy(d, h, N * sizeof(float), hipMemcpyHostToDevice);
        if ((i+1) % 50 == 0) printf("[%3d] ok\n", i+1);
    }
    printf("All 200 mixed ops done!\n");
    free(h); hipFree(d); _exit(0);
}
CPPEOF

/opt/rocm/bin/hipcc --offload-arch=gfx803 -o /tmp/hip_sweep_test /tmp/hip_sweep_test.cpp
/opt/rocm/bin/hipcc --offload-arch=gfx803 -o /tmp/stress_long /tmp/stress_long.cpp
/opt/rocm/bin/hipcc --offload-arch=gfx803 -o /tmp/mix_compute_memcpy /tmp/mix_compute_memcpy.cpp
g++ -o /tmp/test_interrupt_signal tests/test_interrupt_signal.cpp -I/opt/rocm/include -L/opt/rocm/lib -lhsa-runtime64 -Wl,-rpath,/opt/rocm/lib
echo "All binaries built."
echo ""

# Step 2: Interrupt signal test
echo "=== Step 2: Interrupt Signal Test ==="
/tmp/test_interrupt_signal
echo ""

# Step 3: HSA dispatch
echo "=== Step 3: HSA Dispatch ==="
HSA_ENABLE_INTERRUPT=0 timeout 10 tests/hsa_dispatch_test
echo ""

# Step 4: D2H sweep x3
echo "=== Step 4: D2H Sweep x3 ==="
for i in 1 2 3; do
    echo "--- Run $i ---"
    timeout 60 /tmp/hip_sweep_test
done
echo ""

# Step 5: Long stress (550 ops)
echo "=== Step 5: Long Stress (550 ops) ==="
timeout 180 /tmp/stress_long
echo ""

# Step 6: Mixed kernel+memcpy
echo "=== Step 6: Mixed Kernel+Memcpy (200 ops) ==="
timeout 60 /tmp/mix_compute_memcpy
echo ""

# Step 7: GPU reset count
echo "=== Step 7: Post-test GPU state ==="
GPU_RESETS_AFTER=$(sudo dmesg 2>/dev/null | grep -c "GART.*enabled" || echo "?")
echo "GPU resets: $GPU_RESETS -> $GPU_RESETS_AFTER"
rocm-smi --showmeminfo vram 2>&1 | grep Used
echo ""

# Step 8: llama.cpp (if all above pass)
echo "=== Step 8: llama.cpp GPU inference ==="
echo "Running with 5 minute timeout..."
timeout 300 llama-cli -m /home/llm/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf -ngl 1 -p "2+2=" -n 16 --no-display-prompt --simple-io -e --log-disable -c 64 2>&1
LLAMA_RC=$?
echo "llama.cpp exit code: $LLAMA_RC"
echo ""

echo "=========================================="
echo "Test suite complete."
echo "=========================================="
