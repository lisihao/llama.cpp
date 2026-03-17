// ============================================================================
// Debug Logging Usage Examples
// ============================================================================
//
// Compile: g++ -o debug-example ggml-debug-example.cpp
// Run:     GGML_DEBUG=1 ./debug-example
//
// ============================================================================

#include "ggml-debug.h"
#include <vector>

// Mock ggml_tensor structure for demonstration
struct ggml_tensor {
    int type;
    int64_t ne[4];
    size_t nb[4];
};

void example_basic_logging() {
    DEBUG_LOG("Starting basic logging example");

    int value = 42;
    DEBUG_LOG_INT("test_value", value);

    float pi = 3.14159f;
    DEBUG_LOG_FLOAT("pi", pi);

    void* ptr = (void*)0xdeadbeef;
    DEBUG_LOG_PTR("some_pointer", ptr);
}

void example_tagged_logging() {
    DEBUG_LOG_TAG("METAL", "Metal kernel launched");
    DEBUG_LOG_TAG("GRAPH", "Building computation graph");
    DEBUG_LOG_TAG("CACHE", "KV cache hit rate: %.2f%%", 85.5f);
}

void example_conditional_logging() {
    int error_count = 0;
    DEBUG_LOG_IF(error_count > 0, "Errors detected: %d", error_count);

    error_count = 3;
    DEBUG_LOG_IF(error_count > 0, "Errors detected: %d", error_count);
}

void example_tensor_logging() {
    ggml_tensor tensor = {
        .type = 0,
        .ne = {512, 4096, 1, 1},
        .nb = {2, 1024, 4194304, 4194304}
    };

    DEBUG_LOG_TENSOR("my_tensor", &tensor);
}

void example_buffer_logging() {
    unsigned char buffer[32];
    for (int i = 0; i < 32; i++) {
        buffer[i] = i * 8;
    }

    DEBUG_LOG_BUFFER("test_buffer", buffer, 32, 16);
}

void example_timing() {
    DEBUG_TIMER_START(computation);

    // Simulate some work
    volatile long sum = 0;
    for (int i = 0; i < 1000000; i++) {
        sum += i;
    }

    DEBUG_TIMER_END(computation);
}

void example_complex_scenario() {
    DEBUG_LOG("=== Complex Scenario Example ===");

    // Simulate Metal kernel setup
    DEBUG_LOG_TAG("METAL_SETUP", "Configuring kernel parameters");

    int batch_size = 32;
    int seq_len = 2048;
    DEBUG_LOG("Batch size: %d, Sequence length: %d", batch_size, seq_len);

    // Simulate buffer allocation
    DEBUG_TIMER_START(buffer_alloc);
    std::vector<float> buffer(batch_size * seq_len);
    DEBUG_TIMER_END(buffer_alloc);

    DEBUG_LOG_PTR("buffer_address", buffer.data());
    DEBUG_LOG_INT("buffer_size", buffer.size());

    // Simulate error checking
    bool has_error = false;
    DEBUG_LOG_IF(has_error, "ERROR: Buffer allocation failed");
    DEBUG_LOG_IF(!has_error, "SUCCESS: Buffer allocated");
}

int main() {
    printf("=== ThunderLLAMA Debug Logging Examples ===\n");
    printf("Note: Set GGML_DEBUG=1 to see debug output\n\n");

    if (!ggml_debug_enabled()) {
        printf("⚠️  Debug logging is DISABLED\n");
        printf("   Run with: GGML_DEBUG=1 %s\n\n", "debug-example");
    } else {
        printf("✅ Debug logging is ENABLED\n\n");
    }

    example_basic_logging();
    example_tagged_logging();
    example_conditional_logging();
    example_tensor_logging();
    example_buffer_logging();
    example_timing();
    example_complex_scenario();

    printf("\n=== Examples Complete ===\n");
    return 0;
}
