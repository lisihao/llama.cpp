/**
 * Simplified Integration Test: KV Cache Pipeline Quantization
 *
 * Tests pipeline quantization with realistic KV cache dimensions
 * (Qwen3-30B model parameters) without loading the full model.
 *
 * Features:
 * - Memory availability check before running
 * - Waits 30 seconds if insufficient memory
 * - Safe for concurrent testing scenarios
 */

#include "ggml.h"
#include "ggml-metal/ggml-metal-device.h"
#include "check_memory.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

// Forward declare from ggml-metal-ops.h
extern "C" void ggml_metal_quantize_kv_cache_q8_pipeline_cpu(
        void *             dev,
        const ggml_fp16_t * src_cpu,
        int8_t            * dst_cpu,
        ggml_fp16_t       * scales_cpu,
        int                 n_layers,
        int                 elements_per_layer,
        int                 group_size);

// Helper: Compute max absolute error
static float compute_max_error(
        const ggml_fp16_t * original,
        const int8_t      * quantized,
        const ggml_fp16_t * scales,
        size_t              n_elements,
        int                 group_size) {

    float max_error = 0.0f;
    int n_groups = (n_elements + group_size - 1) / group_size;

    for (size_t i = 0; i < n_elements; i++) {
        int group_id = i / group_size;
        float scale = ggml_fp16_to_fp32(scales[group_id]);
        float orig = ggml_fp16_to_fp32(original[i]);
        float dequant = quantized[i] * scale;
        float error = fabsf(orig - dequant) / (fabsf(orig) + 1e-6f);
        max_error = fmaxf(max_error, error);
    }

    return max_error;
}

int main() {
    fprintf(stderr, "=== KV Cache Pipeline Quantization Integration Test ===\n");
    fprintf(stderr, "Model: Qwen3-30B (simulated dimensions)\n\n");

    // Qwen3-30B parameters
    const int n_layers = 64;           // 64 layers
    const int n_embd_head = 192;       // 192 per head
    const int n_head_kv = 8;           // 8 KV heads (GQA)
    const int n_ctx_used = 2048;       // 2K context tokens

    // Calculate memory requirements
    const int n_embd_k_gqa = n_embd_head * n_head_kv;
    const int elements_per_layer = n_embd_k_gqa * n_ctx_used;
    const int total_elements = n_layers * elements_per_layer;

    // Memory needed (MB):
    // - src_k: total_elements * 2 bytes (FP16)
    // - src_v: total_elements * 2 bytes (FP16)
    // - dst_k: total_elements * 1 byte (INT8)
    // - dst_v: total_elements * 1 byte (INT8)
    // - scales_k: (total_elements / 64) * 2 bytes
    // - scales_v: (total_elements / 64) * 2 bytes
    // Total ≈ (total_elements * 6.06) bytes
    size_t required_memory_mb = (size_t)((total_elements * 6.1) / (1024.0 * 1024.0));

    // Add 500 MB buffer for Metal, system overhead, etc.
    required_memory_mb += 500;

    // Check memory availability
    if (!check_memory_availability(required_memory_mb, "KV Cache Pipeline Quantization")) {
        fprintf(stderr, "Test aborted due to insufficient memory.\n");
        return 1;
    }

    fprintf(stderr, "--- Model Dimensions ---\n");
    fprintf(stderr, "Layers: %d\n", n_layers);
    fprintf(stderr, "KV Heads: %d\n", n_head_kv);
    fprintf(stderr, "Head Dim: %d\n", n_embd_head);
    fprintf(stderr, "K/V GQA Embedding: %d\n", n_embd_k_gqa);
    fprintf(stderr, "Context Used: %d tokens\n", n_ctx_used);
    fprintf(stderr, "\n--- KV Cache Size ---\n");
    fprintf(stderr, "Elements per Layer: %d (%.2f MB)\n",
            elements_per_layer, elements_per_layer * 2.0 / 1024 / 1024);
    fprintf(stderr, "Total K Elements: %d (%.2f MB)\n",
            total_elements, total_elements * 2.0 / 1024 / 1024);
    fprintf(stderr, "Total K+V Elements: %d (%.2f MB)\n",
            total_elements * 2, total_elements * 2 * 2.0 / 1024 / 1024);

    // Allocate test data
    fprintf(stderr, "\n--- Preparing Test Data ---\n");
    std::vector<ggml_fp16_t> src_k(total_elements);
    std::vector<ggml_fp16_t> src_v(total_elements);

    // Initialize with random FP16 values
    srand(42);
    for (int i = 0; i < total_elements; i++) {
        float val_k = ((rand() % 2000) - 1000) / 1000.0f; // [-1, 1]
        float val_v = ((rand() % 2000) - 1000) / 1000.0f;
        src_k[i] = ggml_fp32_to_fp16(val_k);
        src_v[i] = ggml_fp32_to_fp16(val_v);
    }

    fprintf(stderr, "Test data initialized (random FP16)\n");

    // Allocate output buffers
    const int group_size = 64;
    const int groups_per_layer = (elements_per_layer + group_size - 1) / group_size;
    const int total_groups = n_layers * groups_per_layer;

    std::vector<int8_t> dst_k(total_elements);
    std::vector<int8_t> dst_v(total_elements);
    std::vector<ggml_fp16_t> scales_k(total_groups);
    std::vector<ggml_fp16_t> scales_v(total_groups);

    fprintf(stderr, "Output buffers allocated\n");
    fprintf(stderr, "  Groups per Layer: %d\n", groups_per_layer);
    fprintf(stderr, "  Total Groups: %d (%.2f MB scales)\n",
            total_groups, total_groups * 2.0 / 1024 / 1024);

    // Get Metal device
    void * dev = ggml_metal_device_get(0);
    if (!dev) {
        fprintf(stderr, "Error: failed to get Metal device\n");
        return 1;
    }

    fprintf(stderr, "\n--- Running Pipeline Quantization ---\n");

    // Quantize K cache
    auto t_start_k = std::chrono::high_resolution_clock::now();

    ggml_metal_quantize_kv_cache_q8_pipeline_cpu(
        dev,
        src_k.data(),
        dst_k.data(),
        scales_k.data(),
        n_layers,
        elements_per_layer,
        group_size
    );

    auto t_end_k = std::chrono::high_resolution_clock::now();
    double time_k_ms = std::chrono::duration<double, std::milli>(t_end_k - t_start_k).count();

    fprintf(stderr, "\nK Cache Quantization:\n");
    fprintf(stderr, "  Time: %.2f ms\n", time_k_ms);
    fprintf(stderr, "  Throughput: %.2f GB/s\n",
            (total_elements * 2.0 / 1024 / 1024 / 1024) / (time_k_ms / 1000));

    // Quantize V cache
    auto t_start_v = std::chrono::high_resolution_clock::now();

    ggml_metal_quantize_kv_cache_q8_pipeline_cpu(
        dev,
        src_v.data(),
        dst_v.data(),
        scales_v.data(),
        n_layers,
        elements_per_layer,
        group_size
    );

    auto t_end_v = std::chrono::high_resolution_clock::now();
    double time_v_ms = std::chrono::duration<double, std::milli>(t_end_v - t_start_v).count();

    fprintf(stderr, "\nV Cache Quantization:\n");
    fprintf(stderr, "  Time: %.2f ms\n", time_v_ms);
    fprintf(stderr, "  Throughput: %.2f GB/s\n",
            (total_elements * 2.0 / 1024 / 1024 / 1024) / (time_v_ms / 1000));

    // Verify correctness (sample 1% to save time)
    fprintf(stderr, "\n--- Verifying Correctness (sample 1%%) ---\n");

    int sample_size = total_elements / 100;
    float max_error_k = 0.0f;
    float max_error_v = 0.0f;

    // Print first 10 samples for debugging
    fprintf(stderr, "Debug: First 10 samples (K cache):\n");
    for (int i = 0; i < 10 && i < sample_size; i++) {
        int idx = (i * 100) % total_elements;
        int layer_id = idx / elements_per_layer;
        int pos_in_layer = idx % elements_per_layer;
        int group_in_layer = pos_in_layer / group_size;
        int group_id = layer_id * groups_per_layer + group_in_layer;

        float scale_k = ggml_fp16_to_fp32(scales_k[group_id]);
        float orig_k = ggml_fp16_to_fp32(src_k[idx]);
        int8_t quant_k = dst_k[idx];
        float dequant_k = quant_k * scale_k;
        float error_k = fabsf(orig_k - dequant_k) / (fabsf(orig_k) + 1e-6f);

        fprintf(stderr, "[%d] idx=%d, layer=%d, group=%d: orig=%.6f, quant=%d, scale=%.6f, dequant=%.6f, error=%.6f\n",
                i, idx, layer_id, group_id, orig_k, quant_k, scale_k, dequant_k, error_k);
    }

    int n_high_errors_k = 0;
    int n_high_errors_v = 0;
    float max_abs_error_k = 0.0f;
    float max_abs_error_v = 0.0f;

    for (int i = 0; i < sample_size; i++) {
        int idx = (i * 100) % total_elements;
        int layer_id = idx / elements_per_layer;
        int pos_in_layer = idx % elements_per_layer;
        int group_in_layer = pos_in_layer / group_size;
        int group_id = layer_id * groups_per_layer + group_in_layer;

        float scale_k = ggml_fp16_to_fp32(scales_k[group_id]);
        float orig_k = ggml_fp16_to_fp32(src_k[idx]);
        int8_t quant_k = dst_k[idx];
        float dequant_k = quant_k * scale_k;

        // Use absolute error for very small values (< 0.01), relative error otherwise
        float abs_error_k = fabsf(orig_k - dequant_k);
        float rel_error_k = abs_error_k / (fabsf(orig_k) + 1e-6f);
        float effective_error_k = (fabsf(orig_k) > 0.01f) ? rel_error_k : abs_error_k;

        if (effective_error_k > 0.1f && n_high_errors_k < 5) {
            fprintf(stderr, "High error K [%d]: idx=%d, layer=%d, orig=%.6f, quant=%d, scale=%.6f, dequant=%.6f, abs_err=%.6f, rel_err=%.6f\n",
                    n_high_errors_k, idx, layer_id, orig_k, quant_k, scale_k, dequant_k, abs_error_k, rel_error_k);
            n_high_errors_k++;
        }

        max_error_k = fmaxf(max_error_k, effective_error_k);
        max_abs_error_k = fmaxf(max_abs_error_k, abs_error_k);

        float scale_v = ggml_fp16_to_fp32(scales_v[group_id]);
        float orig_v = ggml_fp16_to_fp32(src_v[idx]);
        int8_t quant_v = dst_v[idx];
        float dequant_v = quant_v * scale_v;

        float abs_error_v = fabsf(orig_v - dequant_v);
        float rel_error_v = abs_error_v / (fabsf(orig_v) + 1e-6f);
        float effective_error_v = (fabsf(orig_v) > 0.01f) ? rel_error_v : abs_error_v;

        if (effective_error_v > 0.1f && n_high_errors_v < 5) {
            fprintf(stderr, "High error V [%d]: idx=%d, layer=%d, orig=%.6f, quant=%d, scale=%.6f, dequant=%.6f, abs_err=%.6f, rel_err=%.6f\n",
                    n_high_errors_v, idx, layer_id, orig_v, quant_v, scale_v, dequant_v, abs_error_v, rel_error_v);
            n_high_errors_v++;
        }

        max_error_v = fmaxf(max_error_v, effective_error_v);
        max_abs_error_v = fmaxf(max_abs_error_v, abs_error_v);
    }

    fprintf(stderr, "High errors found (effective > 0.1): K=%d, V=%d\n", n_high_errors_k, n_high_errors_v);
    fprintf(stderr, "Max Absolute Error: K=%.6f, V=%.6f\n", max_abs_error_k, max_abs_error_v);

    fprintf(stderr, "K Cache Max Error: %.6f (sampled %d elements)\n", max_error_k, sample_size);
    fprintf(stderr, "V Cache Max Error: %.6f (sampled %d elements)\n", max_error_v, sample_size);

    // Use absolute error threshold (INT8 quantization has high relative error for small values)
    // Absolute error < 0.01 is acceptable for KV cache
    bool passed = (max_abs_error_k < 0.01f) && (max_abs_error_v < 0.01f);

    // Summary
    fprintf(stderr, "\n=== Test Summary ===\n");
    fprintf(stderr, "Model: Qwen3-30B (64 layers, 2048 tokens)\n");
    fprintf(stderr, "K Quantization: %.2f ms (%.2f GB/s)\n",
            time_k_ms, (total_elements * 2.0 / 1024 / 1024 / 1024) / (time_k_ms / 1000));
    fprintf(stderr, "V Quantization: %.2f ms (%.2f GB/s)\n",
            time_v_ms, (total_elements * 2.0 / 1024 / 1024 / 1024) / (time_v_ms / 1000));
    fprintf(stderr, "Total K+V Time: %.2f ms (%.2f GB/s)\n",
            time_k_ms + time_v_ms,
            (total_elements * 2 * 2.0 / 1024 / 1024 / 1024) / ((time_k_ms + time_v_ms) / 1000));
    fprintf(stderr, "Max Error: %.6f (K) / %.6f (V)\n", max_error_k, max_error_v);
    fprintf(stderr, "Status: %s\n", passed ? "✓ PASSED" : "✗ FAILED");

    fprintf(stderr, "\n--- Memory Savings ---\n");
    fprintf(stderr, "Original (FP16):  %.2f MB (K+V)\n",
            total_elements * 2 * 2.0 / 1024 / 1024);
    fprintf(stderr, "Quantized (INT8): %.2f MB (data) + %.2f MB (scales)\n",
            total_elements * 2 * 1.0 / 1024 / 1024,
            total_groups * 2 * 2.0 / 1024 / 1024);
    fprintf(stderr, "Savings: %.1fx reduction\n",
            (total_elements * 2 * 2.0) / (total_elements * 2 * 1.0 + total_groups * 2 * 2.0));

    return passed ? 0 : 1;
}
