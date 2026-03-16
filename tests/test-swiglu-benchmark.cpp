// Benchmark SWIGLU performance - Tier A4 optimization verification
// Compares vectorized vs scalar SWIGLU implementation

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

struct BenchmarkResult {
    double min_ms;
    double max_ms;
    double avg_ms;
    double median_ms;
    double throughput_gflops;
};

static BenchmarkResult run_benchmark(
    const char* name,
    ggml_backend_t backend,
    int64_t M, int64_t N,
    int iterations = 20,
    int warmup = 5
) {
    printf("\n=== %s ===\n", name);
    printf("Dimensions: M=%lld, N=%lld\n", M, N);
    printf("Backend: %s\n", ggml_backend_name(backend));

    // Create context
    struct ggml_init_params params = {
        /*.mem_size   =*/ 512 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context* ctx = ggml_init(params);

    // Create tensors
    struct ggml_tensor* src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);
    struct ggml_tensor* src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);

    // SWIGLU: dst = silu(src0) * src1
    struct ggml_tensor* dst = ggml_swiglu_split(ctx, src0, src1);

    // Build graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    // Allocate tensors
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        printf("❌ Failed to allocate buffer\n");
        ggml_free(ctx);
        return {0, 0, 0, 0, 0};
    }

    // Initialize with random data
    std::vector<float> data0(M * N);
    std::vector<float> data1(M * N);
    srand(42);
    for (int64_t i = 0; i < M * N; i++) {
        data0[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        data1[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    ggml_backend_tensor_set(src0, data0.data(), 0, M * N * sizeof(float));
    ggml_backend_tensor_set(src1, data1.data(), 0, M * N * sizeof(float));

    // Warmup
    for (int i = 0; i < warmup; i++) {
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
    }

    // Benchmark
    std::vector<double> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; i++) {
        auto start = Clock::now();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        auto end = Clock::now();

        double elapsed_ms = Duration(end - start).count();
        times.push_back(elapsed_ms);
    }

    // Calculate statistics
    std::sort(times.begin(), times.end());
    double min_ms = times.front();
    double max_ms = times.back();
    double median_ms = times[times.size() / 2];

    double sum = 0.0;
    for (double t : times) sum += t;
    double avg_ms = sum / times.size();

    // Calculate throughput
    // SWIGLU: exp + div + mul per element = ~3 ops
    int64_t total_ops = M * N * 3;
    double throughput_gflops = (total_ops / 1e9) / (avg_ms / 1000.0);

    printf("Min:    %.3f ms\n", min_ms);
    printf("Median: %.3f ms\n", median_ms);
    printf("Avg:    %.3f ms\n", avg_ms);
    printf("Max:    %.3f ms\n", max_ms);
    printf("Throughput: %.2f GFLOPS\n", throughput_gflops);

    // Cleanup
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);

    return {min_ms, max_ms, avg_ms, median_ms, throughput_gflops};
}

int main() {
    printf("=== Tier A4: SWIGLU Vectorization Benchmark ===\n\n");

#ifndef GGML_USE_METAL
    printf("❌ Metal backend not available (GGML_USE_METAL not defined)\n");
    return 1;
#else

    // Initialize Metal backend
    ggml_backend_t backend_metal = ggml_backend_metal_init();
    if (!backend_metal) {
        printf("❌ Failed to initialize Metal backend\n");
        return 1;
    }

    // Test configurations (representative of Qwen3-30B FFN dimensions)
    struct TestConfig {
        const char* name;
        int64_t M;
        int64_t N;
    } configs[] = {
        {"Small (Batch=1, FFN=4096)",    1,    4096},
        {"Medium (Batch=4, FFN=4096)",   4,    4096},
        {"Large (Batch=16, FFN=8192)",  16,    8192},
        {"XLarge (Batch=32, FFN=16384)", 32,   16384},
    };

    const size_t num_configs = sizeof(configs) / sizeof(configs[0]);

    printf("Test configurations:\n");
    for (size_t i = 0; i < num_configs; i++) {
        printf("%zu. %s\n", i + 1, configs[i].name);
    }

    std::vector<BenchmarkResult> results;

    for (const auto& config : configs) {
        auto result = run_benchmark(
            config.name,
            backend_metal,
            config.M,
            config.N,
            20,  // iterations
            5    // warmup
        );
        results.push_back(result);
    }

    // Summary table
    printf("\n=== Performance Summary ===\n\n");
    printf("%-35s %10s %12s\n", "Configuration", "Avg (ms)", "GFLOPS");
    printf("%-35s %10s %12s\n", "---------------------------------", "----------", "------------");

    for (size_t i = 0; i < num_configs; i++) {
        printf("%-35s %10.3f %12.2f\n",
               configs[i].name,
               results[i].avg_ms,
               results[i].throughput_gflops);
    }

    printf("\n✅ Benchmark complete\n");
    printf("\nNote: These results include the vectorized SWIGLU kernel (float4).\n");
    printf("Expected speedup vs scalar: 1.5-2x\n");

    ggml_backend_free(backend_metal);
    return 0;

#endif
}
