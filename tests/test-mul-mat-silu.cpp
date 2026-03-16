// Test GGML_OP_MUL_MAT_SILU correctness
// Compares fused op with separate matmul + silu

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void print_tensor(const char* name, const ggml_tensor* t) {
    printf("%s [%lld x %lld]:\n", name, t->ne[0], t->ne[1]);
    float* data = (float*)t->data;
    for (int64_t i = 0; i < std::min((int64_t)5, t->ne[1]); i++) {
        for (int64_t j = 0; j < std::min((int64_t)5, t->ne[0]); j++) {
            printf("%.4f ", data[i * t->ne[0] + j]);
        }
        if (t->ne[0] > 5) printf("...");
        printf("\n");
    }
    if (t->ne[1] > 5) printf("...\n");
}

static bool tensors_equal(const ggml_tensor* a, const ggml_tensor* b, float epsilon = 1e-5f) {
    if (a->ne[0] != b->ne[0] || a->ne[1] != b->ne[1]) {
        printf("Dimension mismatch: [%lld x %lld] vs [%lld x %lld]\n",
               a->ne[0], a->ne[1], b->ne[0], b->ne[1]);
        return false;
    }

    float* data_a = (float*)a->data;
    float* data_b = (float*)b->data;
    int64_t n = a->ne[0] * a->ne[1];

    float max_diff = 0.0f;
    int64_t max_diff_idx = 0;

    for (int64_t i = 0; i < n; i++) {
        float diff = std::abs(data_a[i] - data_b[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
        if (diff > epsilon) {
            printf("Mismatch at index %lld: %.6f vs %.6f (diff: %.6e)\n",
                   i, data_a[i], data_b[i], diff);
            return false;
        }
    }

    printf("Tensors equal (max diff: %.6e at index %lld)\n", max_diff, max_diff_idx);
    return true;
}

int main() {
    // Test dimensions
    const int64_t M = 32;   // output rows
    const int64_t N = 16;   // output cols
    const int64_t K = 64;   // shared dimension

    printf("=== Testing GGML_OP_MUL_MAT_SILU ===\n");
    printf("Matrix A: %lld x %lld\n", M, K);
    printf("Matrix B: %lld x %lld\n", K, N);
    printf("Expected output: %lld x %lld\n\n", M, N);

    // Create context
    struct ggml_init_params params = {
        /*.mem_size   =*/ 128 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context* ctx = ggml_init(params);

    // Create input matrices
    // ggml matmul convention: a(K, M) @ b(K, N) = c(M, N)
    // a.ne[0] must equal b.ne[0] (both are K)
    struct ggml_tensor* A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);  // (K=64, M=32)
    struct ggml_tensor* B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);  // (K=64, N=16)

    // Initialize with random values
    srand(42);
    float* data_A = (float*)A->data;
    float* data_B = (float*)B->data;
    for (int64_t i = 0; i < K * M; i++) {  // A is (K, M)
        data_A[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // [-1, 1]
    }
    for (int64_t i = 0; i < K * N; i++) {  // B is (K, N)
        data_B[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    printf("Input matrices initialized\n\n");

    // ===== Method 1: Separate ops (reference) =====
    printf("--- Method 1: Separate ops (matmul + silu) ---\n");

    struct ggml_tensor* C_ref = ggml_mul_mat(ctx, A, B);
    ggml_set_name(C_ref, "matmul_result");

    struct ggml_tensor* D_ref = ggml_silu(ctx, C_ref);
    ggml_set_name(D_ref, "silu_result");

    // Build computation graph
    struct ggml_cgraph* gf_ref = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf_ref, D_ref);

    // Compute
    int n_threads = 4;
    ggml_graph_compute_with_ctx(ctx, gf_ref, n_threads);

    printf("Reference computation done\n");
    print_tensor("C_ref (after matmul)", C_ref);
    print_tensor("D_ref (after silu)", D_ref);
    printf("\n");

    // ===== Method 2: Fused op =====
    printf("--- Method 2: Fused op (mul_mat_silu) ---\n");

    struct ggml_tensor* D_fused = ggml_mul_mat_silu(ctx, A, B);
    ggml_set_name(D_fused, "fused_result");

    // Build computation graph
    struct ggml_cgraph* gf_fused = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf_fused, D_fused);

    // Compute
    ggml_graph_compute_with_ctx(ctx, gf_fused, n_threads);

    printf("Fused computation done\n");
    print_tensor("D_fused", D_fused);
    printf("\n");

    // ===== Verification =====
    printf("--- Verification ---\n");
    bool passed = tensors_equal(D_ref, D_fused);

    if (passed) {
        printf("\n✅ TEST PASSED: Fused op produces identical results\n");
    } else {
        printf("\n❌ TEST FAILED: Results differ\n");
    }

    // Cleanup
    ggml_free(ctx);

    return passed ? 0 : 1;
}
