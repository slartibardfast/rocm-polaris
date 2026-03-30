// test_ggml_sched.cpp — Test ggml_backend_sched with CPU+GPU split
// This is what llama.cpp uses. The scheduler splits the compute graph
// between CPU and GPU backends, using events for synchronization.
//
// Build: g++ -o test_ggml_sched test_ggml_sched.cpp \
//        -I/path/to/ggml/include -lggml -lggml-base -lggml-hip \
//        -lhipblas -L/opt/rocm/lib -lhsa-runtime64 -lamdhip64

#include "ggml.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main() {
    // Init GPU backend
    ggml_backend_t gpu_backend = NULL;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (strstr(ggml_backend_dev_name(dev), "ROCm")) {
            gpu_backend = ggml_backend_dev_init(dev, NULL);
            break;
        }
    }
    if (!gpu_backend) { printf("No GPU\n"); return 1; }

    // Init CPU backend
    ggml_backend_t cpu_backend = ggml_backend_init_by_name("CPU", NULL);
    if (!cpu_backend) { printf("No CPU\n"); return 1; }

    printf("GPU: %s, CPU: %s\n",
           ggml_backend_name(gpu_backend), ggml_backend_name(cpu_backend));

    // Create scheduler with both backends
    ggml_backend_t backends[] = {gpu_backend, cpu_backend};
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, NULL, 2, 4096, false, true);

    // Build a simple graph: a chain of ops
    struct ggml_init_params params = {
        .mem_size = 64 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    struct ggml_context *ctx = ggml_init(params);

    // Simulate a multi-layer transformer: 24 layers of norm + matmul + add
    int hidden = 896; // Qwen2.5 hidden size
    int n_layers = 200;

    struct ggml_tensor *x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden);
    ggml_set_name(x, "x");
    ggml_backend_sched_set_tensor_backend(sched, x, gpu_backend);

    // Create weights for all layers
    struct ggml_tensor *ws[200], *bs[200];
    for (int l = 0; l < n_layers; l++) {
        char name[32];
        ws[l] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, hidden);
        snprintf(name, sizeof(name), "w%d", l);
        ggml_set_name(ws[l], name);
        ggml_backend_sched_set_tensor_backend(sched, ws[l], gpu_backend);

        bs[l] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden);
        snprintf(name, sizeof(name), "b%d", l);
        ggml_set_name(bs[l], name);
        ggml_backend_sched_set_tensor_backend(sched, bs[l], gpu_backend);
    }

    // Build the graph: chain of layers
    struct ggml_tensor *cur = x;
    for (int l = 0; l < n_layers; l++) {
        struct ggml_tensor *norm = ggml_rms_norm(ctx, cur, 1e-5f);
        struct ggml_tensor *mm = ggml_mul_mat(ctx, ws[l], norm);
        cur = ggml_add(ctx, mm, bs[l]);
    }
    struct ggml_tensor *result = cur;
    struct ggml_tensor *w = ws[0], *b = bs[0]; // for data init below
    ggml_set_name(result, "result");

    struct ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    // Allocate via scheduler
    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        printf("Failed to alloc graph\n");
        return 1;
    }

    // Set input data
    float *data = (float*)calloc(hidden * hidden, sizeof(float));
    for (int i = 0; i < hidden; i++) data[i] = 0.01f;
    ggml_backend_tensor_set(x, data, 0, hidden * sizeof(float));
    for (int l = 0; l < n_layers; l++) {
        for (int i = 0; i < hidden * hidden; i++) data[i] = 0.001f;
        ggml_backend_tensor_set(ws[l], data, 0, hidden * hidden * sizeof(float));
        for (int i = 0; i < hidden; i++) data[i] = 0.0f;
        ggml_backend_tensor_set(bs[l], data, 0, hidden * sizeof(float));
    }

    // Pass 1 — like llama.cpp's warmup/system prompt
    printf("Pass 1: Computing 24-layer graph...\n");
    enum ggml_status status = ggml_backend_sched_graph_compute(sched, graph);
    printf("  status: %d %s\n", status, status == 0 ? "PASS" : "FAIL");
    if (status != 0) return 1;

    // D2H readback (like llama.cpp reading logits)
    float readback[896];
    ggml_backend_tensor_get(result, readback, 0, hidden * sizeof(float));
    printf("  readback[0] = %f\n", readback[0]);

    // Pass 2 — like llama.cpp's actual inference
    printf("Pass 2: Computing 24-layer graph again...\n");
    ggml_backend_sched_reset(sched);
    for (int l = 0; l < n_layers; l++) {
        ggml_backend_sched_set_tensor_backend(sched, ws[l], gpu_backend);
        ggml_backend_sched_set_tensor_backend(sched, bs[l], gpu_backend);
    }
    ggml_backend_sched_set_tensor_backend(sched, x, gpu_backend);
    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        printf("  Failed to alloc graph for pass 2\n");
        return 1;
    }
    // Re-set input data
    for (int i = 0; i < hidden; i++) data[i] = 0.01f;
    ggml_backend_tensor_set(x, data, 0, hidden * sizeof(float));
    for (int l = 0; l < n_layers; l++) {
        for (int i = 0; i < hidden * hidden; i++) data[i] = 0.001f;
        ggml_backend_tensor_set(ws[l], data, 0, hidden * hidden * sizeof(float));
        for (int i = 0; i < hidden; i++) data[i] = 0.0f;
        ggml_backend_tensor_set(bs[l], data, 0, hidden * sizeof(float));
    }
    status = ggml_backend_sched_graph_compute(sched, graph);
    printf("  status: %d %s\n", status, status == 0 ? "PASS" : "FAIL");

    float out[128];
    ggml_backend_tensor_get(result, out, 0, hidden * sizeof(float));
    printf("  result[0] = %f\n", out[0]);

    free(data);
    ggml_backend_sched_free(sched);
    ggml_backend_free(gpu_backend);
    ggml_backend_free(cpu_backend);
    ggml_free(ctx);
    return status != 0 ? 1 : 0;
}
