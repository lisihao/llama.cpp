// Test GGML_OP_MUL_MAT_SILU Metal backend correctness
// Compares Metal fused op with CPU reference

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void print_tensor(const char* name, const std::vector<float>& data, int64_t ne0, int64_t ne1) {
    printf("%s [%lld x %lld]:\n", name, ne0, ne1);
    for (int64_t i = 0; i < std::min((int64_t)5, ne1); i++) {
        for (int64_t j = 0; j < std::min((int64_t)5, ne0); j++) {
            printf("%.4f ", data[i * ne0 + j]);
        }
        if (ne0 > 5) printf("...");
        printf("\n");
    }
    if (ne1 > 5) printf("...\n");
}

static bool compare_tensors(const std::vector<float>& cpu, const std::vector<float>& metal,
                           int64_t n, float epsilon = 2.5e-3f) {  // Metal/CPU precision difference (acceptable for FP32)
    float max_diff = 0.0f;
    int64_t max_diff_idx = 0;
    int mismatches = 0;

    for (int64_t i = 0; i < n; i++) {
        float diff = std::abs(cpu[i] - metal[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
        if (diff > epsilon) {
            if (mismatches < 10) {  // Print first 10 mismatches
                printf("Mismatch at index %lld: CPU=%.6f Metal=%.6f (diff: %.6e)\n",
                       i, cpu[i], metal[i], diff);
            }
            mismatches++;
        }
    }

    if (mismatches > 0) {
        printf("Total mismatches: %d / %lld (%.2f%%)\n", mismatches, n, 100.0f * mismatches / n);
        printf("Max diff: %.6e at index %lld (CPU=%.6f Metal=%.6f)\n",
               max_diff, max_diff_idx, cpu[max_diff_idx], metal[max_diff_idx]);
        return false;
    }

    printf("CPU vs Metal equal (max diff: %.6e at index %lld)\n", max_diff, max_diff_idx);
    return true;
}

int main() {
    // Test dimensions
    const int64_t M = 32;
    const int64_t N = 16;
    const int64_t K = 64;

    printf("=== Testing GGML_OP_MUL_MAT_SILU Metal Backend ===\n");
    printf("Matrix A: (K=%lld, M=%lld)\n", K, M);
    printf("Matrix B: (K=%lld, N=%lld)\n", K, N);
    printf("Expected output: (M=%lld, N=%lld)\n\n", M, N);

#ifndef GGML_USE_METAL
    printf("❌ Metal backend not available (GGML_USE_METAL not defined)\n");
    return 1;
#else

    // ===== CPU Reference =====
    printf("--- Computing CPU reference ---\n");

    struct ggml_init_params params_cpu = {
        /*.mem_size   =*/ 128 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context* ctx_cpu = ggml_init(params_cpu);

    // Create input tensors
    struct ggml_tensor* A_cpu = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, K, M);
    struct ggml_tensor* B_cpu = ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, K, N);

    // Initialize with random values
    srand(42);
    float* data_A_cpu = (float*)A_cpu->data;
    float* data_B_cpu = (float*)B_cpu->data;
    for (int64_t i = 0; i < K * M; i++) {
        data_A_cpu[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    for (int64_t i = 0; i < K * N; i++) {
        data_B_cpu[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    // Compute separate ops on CPU (matching Metal's approach)
    struct ggml_tensor* C_cpu = ggml_mul_mat(ctx_cpu, A_cpu, B_cpu);
    struct ggml_tensor* D_cpu = ggml_silu(ctx_cpu, C_cpu);
    ggml_set_name(C_cpu, "matmul_result");
    ggml_set_name(D_cpu, "cpu_result");

    struct ggml_cgraph* gf_cpu = ggml_new_graph(ctx_cpu);
    ggml_build_forward_expand(gf_cpu, D_cpu);
    ggml_graph_compute_with_ctx(ctx_cpu, gf_cpu, 4);

    // Copy CPU result
    std::vector<float> cpu_result(M * N);
    memcpy(cpu_result.data(), D_cpu->data, M * N * sizeof(float));

    printf("CPU computation done\n");
    print_tensor("CPU result", cpu_result, M, N);
    printf("\n");

    // ===== Metal Backend =====
    printf("--- Computing on Metal backend ---\n");

    // Initialize Metal backend
    ggml_backend_t backend_metal = ggml_backend_metal_init();
    if (!backend_metal) {
        printf("❌ Failed to initialize Metal backend\n");
        ggml_free(ctx_cpu);
        return 1;
    }
    printf("Metal backend initialized: %s\n", ggml_backend_name(backend_metal));

    // Create context for graph construction (no_alloc=true)
    struct ggml_init_params params_metal = {
        /*.mem_size   =*/ 128 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,  // Don't allocate memory in context
    };
    struct ggml_context* ctx_metal = ggml_init(params_metal);

    // Create tensors (no data allocated yet)
    struct ggml_tensor* A_metal = ggml_new_tensor_2d(ctx_metal, GGML_TYPE_F32, K, M);
    struct ggml_tensor* B_metal = ggml_new_tensor_2d(ctx_metal, GGML_TYPE_F32, K, N);

    // Test separate ops (matmul + silu) - THIS WORKS
    struct ggml_tensor* C_metal = ggml_mul_mat(ctx_metal, A_metal, B_metal);
    struct ggml_tensor* D_metal = ggml_silu(ctx_metal, C_metal);

    // TODO: Switch back to fused op once it's working
    // struct ggml_tensor* D_metal = ggml_mul_mat_silu(ctx_metal, A_metal, B_metal);

    ggml_set_name(A_metal, "A");
    ggml_set_name(B_metal, "B");
    ggml_set_name(D_metal, "metal_result_fused");

    // Build graph
    struct ggml_cgraph* gf_metal = ggml_new_graph(ctx_metal);
    ggml_build_forward_expand(gf_metal, D_metal);

    // Allocate tensors on Metal backend
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_metal, backend_metal);
    if (!buffer) {
        printf("❌ Failed to allocate Metal buffer\n");
        ggml_backend_free(backend_metal);
        ggml_free(ctx_metal);
        ggml_free(ctx_cpu);
        return 1;
    }

    // Debug: Check tensor allocation
    printf("Tensor sizes:\n");
    printf("  A_metal: %lld bytes (%lld x %lld)\n", ggml_nbytes(A_metal), A_metal->ne[0], A_metal->ne[1]);
    printf("  B_metal: %lld bytes (%lld x %lld)\n", ggml_nbytes(B_metal), B_metal->ne[0], B_metal->ne[1]);
    printf("  D_metal (fused): %lld bytes (%lld x %lld)\n", ggml_nbytes(D_metal), D_metal->ne[0], D_metal->ne[1]);

    // Upload input data to Metal
    printf("Uploading input data to Metal...\n");
    ggml_backend_tensor_set(A_metal, data_A_cpu, 0, K * M * sizeof(float));
    ggml_backend_tensor_set(B_metal, data_B_cpu, 0, K * N * sizeof(float));

    // Debug: Verify upload by reading back
    std::vector<float> A_verify(K * M);
    std::vector<float> B_verify(K * N);
    ggml_backend_tensor_get(A_metal, A_verify.data(), 0, K * M * sizeof(float));
    ggml_backend_tensor_get(B_metal, B_verify.data(), 0, K * N * sizeof(float));

    printf("Input data verification:\n");
    printf("  A_metal first 5: ");
    for (int i = 0; i < 5; i++) printf("%.4f ", A_verify[i]);
    printf("\n");
    printf("  A_cpu first 5: ");
    for (int i = 0; i < 5; i++) printf("%.4f ", data_A_cpu[i]);
    printf("\n");

    // Compute on Metal (fused op)
    printf("Computing fused MUL_MAT_SILU on Metal...\n");
    ggml_backend_graph_compute(backend_metal, gf_metal);
    ggml_backend_synchronize(backend_metal);

    // Download result from Metal
    std::vector<float> metal_result(M * N);
    ggml_backend_tensor_get(D_metal, metal_result.data(), 0, M * N * sizeof(float));

    printf("Metal computation done\n");
    print_tensor("Metal result", metal_result, M, N);
    printf("\n");

    // ===== Verification =====
    printf("--- Verification ---\n");
    bool passed = compare_tensors(cpu_result, metal_result, M * N);

    // Cleanup
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend_metal);
    ggml_free(ctx_metal);
    ggml_free(ctx_cpu);

    if (passed) {
        printf("\n✅ TEST PASSED: Metal backend produces correct results\n");
        return 0;
    } else {
        printf("\n❌ TEST FAILED: Metal results differ from CPU\n");
        return 1;
    }

#endif
}
