// test_ggml_dispatch.cpp — Link against libggml-hip.so and call its backend
// This tests whether ggml-hip's own code objects work on gfx803.
//
// Build: g++ -o test_ggml_dispatch test_ggml_dispatch.cpp \
//        -I/home/llm/rocm-polaris/llama.cpp/src/llama.cpp-b7376/ggml/include \
//        -lggml -lggml-base -lggml-hip -lggml-cpu -lhipblas \
//        -L/opt/rocm/lib -lhsa-runtime64 -lamdhip64 \
//        -Wl,-rpath,/usr/lib -Wl,-rpath,/opt/rocm/lib

#include "ggml.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main() {
    printf("Initializing ggml backend...\n");

    // Get the CUDA/HIP backend
    ggml_backend_t backend = NULL;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char* name = ggml_backend_dev_name(dev);
        printf("  device %zu: %s\n", i, name);
        if (strstr(name, "ROCm") || strstr(name, "CUDA") || strstr(name, "GPU") || strstr(name, "cuda")) {
            backend = ggml_backend_dev_init(dev, NULL);
            printf("  -> initialized backend for %s\n", name);
        }
    }

    if (!backend) {
        printf("No CUDA/HIP backend found!\n");
        return 1;
    }

    // Test multiple operations that llama.cpp uses
    const char* ops[] = {"add", "mul_mat", "rms_norm", "softmax", "mul_mat_q8", "mul_mat_q4k", "rope", NULL};

    for (int test = 0; ops[test]; test++) {
        struct ggml_init_params params = {
            .mem_size = 64 * 1024 * 1024,
            .mem_buffer = NULL,
            .no_alloc = true,
        };
        struct ggml_context *ctx = ggml_init(params);

        struct ggml_tensor *result = NULL;
        struct ggml_tensor *a, *b;

        if (strcmp(ops[test], "add") == 0) {
            a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
            b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
            result = ggml_add(ctx, a, b);
        } else if (strcmp(ops[test], "mul_mat") == 0) {
            a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 64);
            b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 32);
            result = ggml_mul_mat(ctx, a, b);
        } else if (strcmp(ops[test], "rms_norm") == 0) {
            a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
            b = NULL;
            result = ggml_rms_norm(ctx, a, 1e-5f);
        } else if (strcmp(ops[test], "softmax") == 0) {
            a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4);
            b = NULL;
            result = ggml_soft_max(ctx, a);
        } else if (strcmp(ops[test], "mul_mat_q8") == 0) {
            a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 128, 64);
            b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 4);
            result = ggml_mul_mat(ctx, a, b);
        } else if (strcmp(ops[test], "mul_mat_q4k") == 0) {
            a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 256, 64);
            b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 4);
            result = ggml_mul_mat(ctx, a, b);
        } else if (strcmp(ops[test], "rope") == 0) {
            a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 4, 1);
            b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
            result = ggml_rope(ctx, a, b, 64, 0);
        }

        struct ggml_cgraph *graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

        // Zero-init inputs
        size_t a_size = ggml_nbytes(a);
        float *data = (float*)calloc(1, a_size);
        for (size_t i = 0; i < a_size / sizeof(float); i++) data[i] = 0.1f;
        ggml_backend_tensor_set(a, data, 0, a_size);
        if (b) {
            size_t b_size = ggml_nbytes(b);
            float *bdata = (float*)calloc(1, b_size);
            for (size_t i = 0; i < b_size / sizeof(float); i++) bdata[i] = 0.2f;
            ggml_backend_tensor_set(b, bdata, 0, b_size);
            free(bdata);
        }
        free(data);

        printf("Test %s... ", ops[test]);
        fflush(stdout);
        enum ggml_status status = ggml_backend_graph_compute(backend, graph);
        printf("status=%d %s\n", status, status == 0 ? "PASS" : "FAIL");

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);

        if (status != 0) return 1;
    }

    printf("All ops PASS\n");

    ggml_backend_free(backend);
    return 0;
}
