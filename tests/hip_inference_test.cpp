// hip_inference_test.cpp — Inference-pattern stress tests
//
// Targets the exact dispatch patterns ggml-hip uses:
// - Large thread grids (not just <<<1,1>>>)
// - Many different kernel functions per "layer"
// - Deep dispatch chains without intermediate sync
// - Scratch memory (large register pressure)
// - Kernarg pool pressure (many launches with different args)
// - Mixed memcpy + compute
// - hipStreamSynchronize after deep chains
// - hipDeviceSynchronize under sustained load
//
// Build: hipcc --offload-arch=gfx803 -o hip_inference_test hip_inference_test.cpp
// Run:   timeout 600 ./hip_inference_test

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <chrono>
#include <cmath>

#define HIP_CHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
                e, hipGetErrorString(e), __FILE__, __LINE__); \
        return false; \
    } \
} while (0)

static volatile sig_atomic_t timed_out = 0;
static void alarm_handler(int) { timed_out = 1; }

// --- Kernels mimicking ggml-hip operations ---

// RMS norm (ggml pattern: reduce + normalize)
__global__ void rms_norm(const float *x, float *out, int n, float eps) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x)
        sum += x[i] * x[i];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / n + eps);
    for (int i = tid; i < n; i += blockDim.x)
        out[i] = x[i] * rms;
}

// Element-wise multiply (ggml pattern: apply weights)
__global__ void elem_mul(const float *a, const float *b, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

// Element-wise add
__global__ void elem_add(const float *a, const float *b, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// SiLU activation (ggml pattern: MLP)
__global__ void silu(const float *x, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        out[i] = v / (1.0f + expf(-v));
    }
}

// Softmax (ggml pattern: attention)
__global__ void softmax_kernel(float *x, int n) {
    __shared__ float smax[256];
    __shared__ float ssum[256];
    int tid = threadIdx.x;

    float mx = -1e30f;
    for (int i = tid; i < n; i += blockDim.x)
        mx = fmaxf(mx, x[i]);
    smax[tid] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smax[tid] = fmaxf(smax[tid], smax[tid + s]);
        __syncthreads();
    }
    mx = smax[0];

    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        x[i] = expf(x[i] - mx);
        sum += x[i];
    }
    ssum[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) ssum[tid] += ssum[tid + s];
        __syncthreads();
    }
    sum = ssum[0];
    for (int i = tid; i < n; i += blockDim.x)
        x[i] /= sum;
}

// Simple matmul (ggml pattern: linear layer, small for testing)
__global__ void matmul_naive(const float *A, const float *B, float *C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// Copy kernel (ggml pattern: reshape/permute)
__global__ void copy_kernel(const float *src, float *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// Scale kernel
__global__ void scale_kernel(float *x, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
}

// Verify kernel
__global__ void check_finite(const float *x, int n, int *bad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && (std::isnan(x[i]) || std::isinf(x[i]))) atomicAdd(bad, 1);
}

// ============================================================
// Test 1: Deep dispatch chain (many kernels, no intermediate sync)
// Simulates a transformer layer: norm → matmul → activation → add
// ============================================================
static bool test_deep_dispatch_chain() {
    printf("Test 1: Deep dispatch chain (50 layers, ~350 kernels)...\n");
    const int DIM = 512;  // embedding dimension
    const int N = DIM;
    int blocks = (N + 255) / 256;

    float *d_x, *d_w, *d_tmp, *d_tmp2;
    HIP_CHECK(hipMalloc(&d_x, DIM * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_w, DIM * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_tmp, DIM * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_tmp2, DIM * sizeof(float)));

    // Init: x = 1.0, w = 0.5
    float *h = (float *)malloc(DIM * sizeof(float));
    for (int i = 0; i < DIM; i++) h[i] = 1.0f;
    HIP_CHECK(hipMemcpy(d_x, h, DIM * sizeof(float), hipMemcpyHostToDevice));
    for (int i = 0; i < DIM; i++) h[i] = 0.5f;
    HIP_CHECK(hipMemcpy(d_w, h, DIM * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    // 50 "layers" — each is ~7 kernel dispatches, NO sync between layers
    for (int layer = 0; layer < 50; layer++) {
        rms_norm<<<1, 256>>>(d_x, d_tmp, DIM, 1e-5f);          // 1
        elem_mul<<<blocks, 256>>>(d_tmp, d_w, d_tmp2, N);       // 2
        silu<<<blocks, 256>>>(d_tmp2, d_tmp, N);                // 3
        elem_add<<<blocks, 256>>>(d_x, d_tmp, d_tmp2, N);       // 4
        copy_kernel<<<blocks, 256>>>(d_tmp2, d_x, N);           // 5
        scale_kernel<<<blocks, 256>>>(d_x, 0.999f, N);          // 6 (prevent explosion)
        if (timed_out) break;
    }

    // Single sync at the end (ggml pattern: sync after full forward pass)
    HIP_CHECK(hipDeviceSynchronize());

    // Verify result is finite
    int *d_bad;
    HIP_CHECK(hipMalloc(&d_bad, sizeof(int)));
    HIP_CHECK(hipMemset(d_bad, 0, sizeof(int)));
    check_finite<<<blocks, 256>>>(d_x, N, d_bad);
    HIP_CHECK(hipDeviceSynchronize());
    int bad = -1;
    HIP_CHECK(hipMemcpy(&bad, d_bad, sizeof(int), hipMemcpyDeviceToHost));

    // Read back and check
    HIP_CHECK(hipMemcpy(h, d_x, DIM * sizeof(float), hipMemcpyDeviceToHost));
    bool finite = (bad == 0);
    bool nonzero = false;
    for (int i = 0; i < DIM; i++) {
        if (h[i] != 0.0f) { nonzero = true; break; }
    }

    printf("    finite=%d nonzero=%d x[0]=%.6f\n", finite, nonzero, h[0]);
    hipFree(d_bad); hipFree(d_tmp2); hipFree(d_tmp); hipFree(d_w); hipFree(d_x);
    free(h);
    return finite && nonzero && !timed_out;
}

// ============================================================
// Test 2: Kernarg pool pressure (many launches with different args)
// ============================================================
static bool test_kernarg_pressure() {
    printf("Test 2: Kernarg pool pressure (2000 launches, varying args)...\n");
    const int N = 1024;
    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_b, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_c, N * sizeof(float)));

    float *h = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h[i] = 1.0f;
    HIP_CHECK(hipMemcpy(d_a, h, N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h, N * sizeof(float), hipMemcpyHostToDevice));

    int blocks = (N + 255) / 256;

    // 2000 launches with different scale values (each = new kernarg alloc)
    for (int i = 0; i < 2000; i++) {
        float scale = 1.0f + (float)i * 0.0001f;
        elem_mul<<<blocks, 256>>>(d_a, d_b, d_c, N);
        scale_kernel<<<blocks, 256>>>(d_c, scale, N);
        if (timed_out) break;
    }
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h, d_c, N * sizeof(float), hipMemcpyDeviceToHost));
    bool ok = (h[0] != 0.0f && !std::isnan(h[0]));
    printf("    c[0]=%.6f\n", h[0]);

    hipFree(d_c); hipFree(d_b); hipFree(d_a); free(h);
    return ok && !timed_out;
}

// ============================================================
// Test 3: Mixed memcpy + compute (model load + inference pattern)
// ============================================================
static bool test_mixed_memcpy_compute() {
    printf("Test 3: Mixed memcpy + compute (20 layers: load weights + compute)...\n");
    const int DIM = 512;
    int blocks = (DIM + 255) / 256;

    float *d_x, *d_out;
    HIP_CHECK(hipMalloc(&d_x, DIM * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, DIM * sizeof(float)));

    float *h_x = (float *)malloc(DIM * sizeof(float));
    float *h_w = (float *)malloc(DIM * sizeof(float));
    for (int i = 0; i < DIM; i++) h_x[i] = 1.0f;
    HIP_CHECK(hipMemcpy(d_x, h_x, DIM * sizeof(float), hipMemcpyHostToDevice));

    for (int layer = 0; layer < 20; layer++) {
        // "Load weights" for this layer (H2D during compute)
        float *d_w;
        HIP_CHECK(hipMalloc(&d_w, DIM * sizeof(float)));
        for (int i = 0; i < DIM; i++) h_w[i] = 0.99f;
        HIP_CHECK(hipMemcpy(d_w, h_w, DIM * sizeof(float), hipMemcpyHostToDevice));

        // Compute: norm → mul → activation → residual
        rms_norm<<<1, 256>>>(d_x, d_out, DIM, 1e-5f);
        elem_mul<<<blocks, 256>>>(d_out, d_w, d_out, DIM);
        silu<<<blocks, 256>>>(d_out, d_out, DIM);
        elem_add<<<blocks, 256>>>(d_x, d_out, d_x, DIM);

        HIP_CHECK(hipFree(d_w));
        if (timed_out) break;
    }

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_x, d_x, DIM * sizeof(float), hipMemcpyDeviceToHost));

    bool ok = !std::isnan(h_x[0]) && h_x[0] != 0.0f;
    printf("    x[0]=%.6f\n", h_x[0]);

    hipFree(d_out); hipFree(d_x); free(h_w); free(h_x);
    return ok && !timed_out;
}

// ============================================================
// Test 4: Stream sync after deep chains (hipStreamSynchronize)
// ============================================================
static bool test_stream_sync_deep() {
    printf("Test 4: Stream sync after deep chains (10 rounds x 100 kernels)...\n");
    const int N = 1024;
    int blocks = (N + 255) / 256;
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    float *d_a, *d_b;
    HIP_CHECK(hipMalloc(&d_a, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_b, N * sizeof(float)));

    float *h = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h[i] = 1.0f;
    HIP_CHECK(hipMemcpy(d_a, h, N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h, N * sizeof(float), hipMemcpyHostToDevice));

    for (int round = 0; round < 10; round++) {
        for (int i = 0; i < 100; i++) {
            elem_mul<<<blocks, 256, 0, stream>>>(d_a, d_b, d_a, N);
            scale_kernel<<<blocks, 256, 0, stream>>>(d_a, 1.0f, N);  // identity
        }
        HIP_CHECK(hipStreamSynchronize(stream));

        HIP_CHECK(hipMemcpy(h, d_a, sizeof(float), hipMemcpyDeviceToHost));
        if (std::isnan(h[0]) || h[0] == 0.0f) {
            printf("  FAIL round %d: a[0]=%.6f\n", round, h[0]);
            hipStreamDestroy(stream); hipFree(d_b); hipFree(d_a); free(h);
            return false;
        }
        if (timed_out) break;
    }

    printf("    a[0]=%.6f\n", h[0]);
    hipStreamDestroy(stream); hipFree(d_b); hipFree(d_a); free(h);
    return !timed_out;
}

// ============================================================
// Test 5: Small matmul chain (attention pattern)
// ============================================================
static bool test_matmul_chain() {
    printf("Test 5: Small matmul chain (32x32, 50 iterations)...\n");
    const int M = 32, N = 32, K = 32;
    size_t sz = M * N * sizeof(float);

    float *d_A, *d_B, *d_C, *d_tmp;
    HIP_CHECK(hipMalloc(&d_A, sz));
    HIP_CHECK(hipMalloc(&d_B, sz));
    HIP_CHECK(hipMalloc(&d_C, sz));
    HIP_CHECK(hipMalloc(&d_tmp, sz));

    float *h = (float *)malloc(sz);
    for (int i = 0; i < M * N; i++) h[i] = (float)(i % 7) * 0.1f;
    HIP_CHECK(hipMemcpy(d_A, h, sz, hipMemcpyHostToDevice));
    for (int i = 0; i < M * N; i++) h[i] = (float)(i % 5) * 0.1f;
    HIP_CHECK(hipMemcpy(d_B, h, sz, hipMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    for (int iter = 0; iter < 50; iter++) {
        matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);      // Q*K
        softmax_kernel<<<M, 256>>>(d_C, N);                          // softmax
        matmul_naive<<<grid, block>>>(d_C, d_B, d_tmp, M, N, K);    // attn*V
        copy_kernel<<<(M*N+255)/256, 256>>>(d_tmp, d_A, M * N);     // feedback
        if (timed_out) break;
    }
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h, d_A, sz, hipMemcpyDeviceToHost));
    bool ok = !std::isnan(h[0]) && !std::isinf(h[0]);
    printf("    A[0]=%.6f\n", h[0]);

    hipFree(d_tmp); hipFree(d_C); hipFree(d_B); hipFree(d_A); free(h);
    return ok && !timed_out;
}

// ============================================================
// Test 6: hipMemsetAsync + compute interleave
// ============================================================
static bool test_memset_compute() {
    printf("Test 6: hipMemsetAsync + compute interleave (200 rounds)...\n");
    const int N = 4096;
    int blocks = (N + 255) / 256;

    float *d_a, *d_b;
    HIP_CHECK(hipMalloc(&d_a, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_b, N * sizeof(float)));

    float *h = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h[i] = 2.0f;
    HIP_CHECK(hipMemcpy(d_b, h, N * sizeof(float), hipMemcpyHostToDevice));

    for (int i = 0; i < 200; i++) {
        HIP_CHECK(hipMemsetAsync(d_a, 0, N * sizeof(float), 0));
        elem_add<<<blocks, 256>>>(d_a, d_b, d_a, N);
        scale_kernel<<<blocks, 256>>>(d_a, 1.0f, N);
        if (timed_out) break;
    }
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h, d_a, sizeof(float), hipMemcpyDeviceToHost));
    // Each round: 0 + 2.0 * 1.0 = 2.0 (memset resets each time)
    bool ok = fabsf(h[0] - 2.0f) < 0.01f;
    printf("    a[0]=%.6f (expected 2.0)\n", h[0]);

    hipFree(d_b); hipFree(d_a); free(h);
    return ok && !timed_out;
}

// ============================================================
// Test 7: Event-based pipeline (record event, then more work)
// ============================================================
static bool test_event_pipeline() {
    printf("Test 7: Event-based pipeline (100 rounds: compute→event→compute)...\n");
    const int N = 1024;
    int blocks = (N + 255) / 256;

    float *d_a, *d_b;
    hipEvent_t ev;
    HIP_CHECK(hipMalloc(&d_a, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_b, N * sizeof(float)));
    HIP_CHECK(hipEventCreate(&ev));

    float *h = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h[i] = 1.0f;
    HIP_CHECK(hipMemcpy(d_a, h, N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h, N * sizeof(float), hipMemcpyHostToDevice));

    for (int i = 0; i < 100; i++) {
        // Phase 1: compute
        elem_mul<<<blocks, 256>>>(d_a, d_b, d_a, N);
        scale_kernel<<<blocks, 256>>>(d_a, 1.0f, N);

        // Record event
        HIP_CHECK(hipEventRecord(ev, 0));

        // Phase 2: more compute after event
        elem_add<<<blocks, 256>>>(d_a, d_b, d_a, N);
        silu<<<blocks, 256>>>(d_a, d_a, N);

        // Sync on event (NOT DeviceSynchronize — just this event)
        HIP_CHECK(hipEventSynchronize(ev));

        if (timed_out) break;
    }
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h, d_a, sizeof(float), hipMemcpyDeviceToHost));
    bool ok = !std::isnan(h[0]) && h[0] != 0.0f;
    printf("    a[0]=%.6f\n", h[0]);

    hipEventDestroy(ev); hipFree(d_b); hipFree(d_a); free(h);
    return ok && !timed_out;
}

// ============================================================
// Test 8: Sustained inference simulation (full forward pass x 20)
// ============================================================
static bool test_sustained_inference() {
    printf("Test 8: Sustained inference (20 forward passes, 6 layers each)...\n");
    const int DIM = 256;
    const int HIDDEN = 512;
    int blocks_dim = (DIM + 255) / 256;
    int blocks_hid = (HIDDEN + 255) / 256;

    // Allocate "model weights" (persistent)
    float *d_w1, *d_w2, *d_w_norm;
    HIP_CHECK(hipMalloc(&d_w1, DIM * HIDDEN * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_w2, HIDDEN * DIM * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_w_norm, DIM * sizeof(float)));

    // Init weights
    float *h = (float *)malloc(DIM * HIDDEN * sizeof(float));
    for (int i = 0; i < DIM * HIDDEN; i++) h[i] = 0.01f * ((i % 17) - 8);
    HIP_CHECK(hipMemcpy(d_w1, h, DIM * HIDDEN * sizeof(float), hipMemcpyHostToDevice));
    for (int i = 0; i < HIDDEN * DIM; i++) h[i] = 0.01f * ((i % 13) - 6);
    HIP_CHECK(hipMemcpy(d_w2, h, HIDDEN * DIM * sizeof(float), hipMemcpyHostToDevice));
    for (int i = 0; i < DIM; i++) h[i] = 1.0f;
    HIP_CHECK(hipMemcpy(d_w_norm, h, DIM * sizeof(float), hipMemcpyHostToDevice));

    // Working buffers
    float *d_x, *d_normed, *d_hidden, *d_proj, *d_residual;
    HIP_CHECK(hipMalloc(&d_x, DIM * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_normed, DIM * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_hidden, HIDDEN * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_proj, DIM * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_residual, DIM * sizeof(float)));

    dim3 mm1_block(16, 16);
    dim3 mm1_grid((HIDDEN + 15) / 16, 1);  // [1 x HIDDEN] = [1 x DIM] * [DIM x HIDDEN]
    dim3 mm2_grid((DIM + 15) / 16, 1);

    for (int pass = 0; pass < 20; pass++) {
        // Init input for this pass
        for (int i = 0; i < DIM; i++) h[i] = 0.1f * (pass + 1);
        HIP_CHECK(hipMemcpy(d_x, h, DIM * sizeof(float), hipMemcpyHostToDevice));

        // 6 transformer layers
        for (int layer = 0; layer < 6; layer++) {
            // Layer norm
            rms_norm<<<1, 256>>>(d_x, d_normed, DIM, 1e-5f);
            elem_mul<<<blocks_dim, 256>>>(d_normed, d_w_norm, d_normed, DIM);

            // MLP up-project: [1,DIM] → [1,HIDDEN]
            matmul_naive<<<mm1_grid, mm1_block>>>(d_normed, d_w1, d_hidden, 1, HIDDEN, DIM);

            // Activation
            silu<<<blocks_hid, 256>>>(d_hidden, d_hidden, HIDDEN);

            // MLP down-project: [1,HIDDEN] → [1,DIM]
            matmul_naive<<<mm2_grid, mm1_block>>>(d_hidden, d_w2, d_proj, 1, DIM, HIDDEN);

            // Residual
            elem_add<<<blocks_dim, 256>>>(d_x, d_proj, d_x, DIM);
            scale_kernel<<<blocks_dim, 256>>>(d_x, 0.95f, DIM);  // dampen
        }

        // Sync after each forward pass (ggml pattern)
        HIP_CHECK(hipDeviceSynchronize());

        // D2H readback (token logits)
        HIP_CHECK(hipMemcpy(h, d_x, DIM * sizeof(float), hipMemcpyDeviceToHost));
        if (std::isnan(h[0])) {
            printf("  FAIL pass %d: NaN\n", pass);
            goto cleanup;
        }
        if (timed_out) goto cleanup;
    }

    printf("    x[0]=%.6f after 20 passes\n", h[0]);

cleanup:
    hipFree(d_residual); hipFree(d_proj); hipFree(d_hidden);
    hipFree(d_normed); hipFree(d_x);
    hipFree(d_w_norm); hipFree(d_w2); hipFree(d_w1);
    free(h);
    return !timed_out;
}

// ============================================================
// Test 9: Rapid alloc/compute/free with large buffers
// ============================================================
static bool test_rapid_large_alloc() {
    printf("Test 9: Rapid alloc/compute/free (100 rounds, 4MB buffers)...\n");
    const int N = 1 << 20;  // 1M floats = 4MB
    int blocks = (N + 255) / 256;

    float *h = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < 100; i++) {
        float *d_a, *d_b;
        HIP_CHECK(hipMalloc(&d_a, N * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_b, N * sizeof(float)));

        for (int j = 0; j < N; j++) h[j] = (float)(i + 1);
        HIP_CHECK(hipMemcpy(d_a, h, N * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_b, h, N * sizeof(float), hipMemcpyHostToDevice));

        elem_mul<<<blocks, 256>>>(d_a, d_b, d_a, N);
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipMemcpy(h, d_a, sizeof(float), hipMemcpyDeviceToHost));
        float expected = (float)(i + 1) * (float)(i + 1);
        if (fabsf(h[0] - expected) > 0.1f) {
            printf("  FAIL round %d: expected %.1f, got %.6f\n", i, expected, h[0]);
            hipFree(d_b); hipFree(d_a); free(h);
            return false;
        }

        HIP_CHECK(hipFree(d_b));
        HIP_CHECK(hipFree(d_a));
        if (timed_out) { free(h); return false; }
    }
    free(h);
    return true;
}

// ============================================================
// Main
// ============================================================

int main() {
    signal(SIGALRM, alarm_handler);
    alarm(600);  // 10 minute global timeout

    hipDeviceProp_t prop;
    hipError_t e = hipGetDeviceProperties(&prop, 0);
    if (e != hipSuccess) {
        fprintf(stderr, "hipGetDeviceProperties failed: %d\n", e);
        return 1;
    }
    printf("Device: %s (%s)\n\n", prop.name, prop.gcnArchName);

    struct { const char *name; bool (*fn)(); } tests[] = {
        {"Deep dispatch chain (350 kernels)",    test_deep_dispatch_chain},
        {"Kernarg pool pressure (4000 launches)", test_kernarg_pressure},
        {"Mixed memcpy + compute",               test_mixed_memcpy_compute},
        {"Stream sync after deep chains",        test_stream_sync_deep},
        {"Matmul chain (attention pattern)",      test_matmul_chain},
        {"hipMemsetAsync + compute",             test_memset_compute},
        {"Event-based pipeline",                 test_event_pipeline},
        {"Sustained inference (20 passes)",      test_sustained_inference},
        {"Rapid large alloc/compute/free",       test_rapid_large_alloc},
    };

    int pass = 0, fail = 0;
    for (auto &t : tests) {
        fflush(stdout);
        auto t0 = std::chrono::steady_clock::now();
        bool ok = t.fn();
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        if (timed_out) {
            printf("  GLOBAL TIMEOUT\n");
            ok = false;
        }
        printf("  %s: %s (%.1fs)\n\n", t.name, ok ? "PASS" : "FAIL", secs);
        fflush(stdout);
        if (ok) pass++; else fail++;
    }

    printf("=== Results: %d/%d PASS ===\n", pass, pass + fail);
    fflush(stdout);
    return fail > 0 ? 1 : 0;
}
