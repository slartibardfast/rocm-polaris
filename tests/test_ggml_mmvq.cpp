// test_ggml_mmvq.cpp — Test mul_mat with vector (ncols=1) via ggml
// This is exactly what hangs in llama.cpp: Q4_K matrix × f32 vector
//
// Build: g++ -o test_ggml_mmvq test_ggml_mmvq.cpp \
//        -I/path/to/ggml/include -lggml -lggml-base -lggml-hip \
//        -lhipblas -L/opt/rocm/lib -lhsa-runtime64 -lamdhip64

#include "ggml.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main() {
    ggml_backend_t backend = NULL;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (strstr(ggml_backend_dev_name(dev), "ROCm")) {
            backend = ggml_backend_dev_init(dev, NULL);
            break;
        }
    }
    if (!backend) { printf("No GPU\n"); return 1; }

    // Test different quantization types used by Qwen/TinyLlama
    struct { const char *name; enum ggml_type type; int hidden; } tests[] = {
        {"Q4_K 1024x1", GGML_TYPE_Q4_K, 1024},
        {"Q8_0 896x1",  GGML_TYPE_Q8_0, 896},
        {"Q4_K 2048x1", GGML_TYPE_Q4_K, 2048},
        {"Q8_0 2048x1", GGML_TYPE_Q8_0, 2048},
        {"F32 896x1",   GGML_TYPE_F32,  896},
        {"F16 896x1",   GGML_TYPE_F16,  896},
        {NULL, GGML_TYPE_F32, 0}
    };

    for (int t = 0; tests[t].name; t++) {
        struct ggml_init_params params = {
            .mem_size = 64 * 1024 * 1024,
            .mem_buffer = NULL,
            .no_alloc = true,
        };
        struct ggml_context *ctx = ggml_init(params);

        int hidden = tests[t].hidden;
        // Matrix: hidden x hidden, quantized
        struct ggml_tensor *w = ggml_new_tensor_2d(ctx, tests[t].type, hidden, hidden);
        // Vector: hidden x 1, f32
        struct ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, 1);
        // Result: hidden x 1
        struct ggml_tensor *result = ggml_mul_mat(ctx, w, x);

        struct ggml_cgraph *graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

        // Zero-init
        size_t w_size = ggml_nbytes(w);
        void *wdata = calloc(1, w_size);
        ggml_backend_tensor_set(w, wdata, 0, w_size);
        free(wdata);

        float *xdata = (float*)calloc(hidden, sizeof(float));
        for (int i = 0; i < hidden; i++) xdata[i] = 0.1f;
        ggml_backend_tensor_set(x, xdata, 0, hidden * sizeof(float));
        free(xdata);

        printf("Test %s... ", tests[t].name);
        fflush(stdout);
        enum ggml_status status = ggml_backend_graph_compute(backend, graph);
        printf("status=%d %s\n", status, status == 0 ? "PASS" : "FAIL");

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        if (status != 0) return 1;
    }

    printf("All MMVQ tests PASS\n");
    ggml_backend_free(backend);
    return 0;
}
