#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Debug Logging Utilities
// ============================================================================
//
// Purpose: Generic debug logging framework for ThunderLLAMA
// Usage:
//   - Set environment variable: export GGML_DEBUG=1
//   - Include this header: #include "ggml-debug.h"
//   - Use macros: DEBUG_LOG("message: %d", value);
//
// Features:
//   - Timestamp with microseconds
//   - File and line number
//   - Automatic flush to stderr
//   - Environment variable control
//
// ============================================================================

// Check if debug is enabled (cached for performance)
static inline int ggml_debug_enabled(void) {
    static int checked = 0;
    static int enabled = 0;

    if (!checked) {
        const char* env = getenv("GGML_DEBUG");
        enabled = (env != NULL && env[0] == '1');
        checked = 1;
    }

    return enabled;
}

// Get current timestamp with microseconds
static inline void ggml_debug_timestamp(char* buf, size_t size) {
    struct timeval tv;
    gettimeofday(&tv, NULL);

    struct tm* tm_info = localtime(&tv.tv_sec);
    int len = strftime(buf, size, "%H:%M:%S", tm_info);
    snprintf(buf + len, size - len, ".%06ld", (long)tv.tv_usec);
}

// ============================================================================
// Core Macros
// ============================================================================

// Generic debug log with timestamp and location
#define DEBUG_LOG(fmt, ...) \
    do { \
        if (ggml_debug_enabled()) { \
            char timestamp[32]; \
            ggml_debug_timestamp(timestamp, sizeof(timestamp)); \
            fprintf(stderr, "[%s] [%s:%d] " fmt "\n", \
                    timestamp, __FILE__, __LINE__, ##__VA_ARGS__); \
            fflush(stderr); \
        } \
    } while(0)

// Debug log with custom tag
#define DEBUG_LOG_TAG(tag, fmt, ...) \
    do { \
        if (ggml_debug_enabled()) { \
            char timestamp[32]; \
            ggml_debug_timestamp(timestamp, sizeof(timestamp)); \
            fprintf(stderr, "[%s] [%s] [%s:%d] " fmt "\n", \
                    timestamp, tag, __FILE__, __LINE__, ##__VA_ARGS__); \
            fflush(stderr); \
        } \
    } while(0)

// Debug log without location (cleaner output)
#define DEBUG_LOG_SIMPLE(fmt, ...) \
    do { \
        if (ggml_debug_enabled()) { \
            fprintf(stderr, fmt "\n", ##__VA_ARGS__); \
            fflush(stderr); \
        } \
    } while(0)

// ============================================================================
// Specialized Macros
// ============================================================================

// Log tensor information
#define DEBUG_LOG_TENSOR(name, tensor) \
    do { \
        if (ggml_debug_enabled() && (tensor) != NULL) { \
            DEBUG_LOG("Tensor %s: type=%d, ne=[%lld,%lld,%lld,%lld], nb=[%zu,%zu,%zu,%zu]", \
                      name, \
                      (tensor)->type, \
                      (tensor)->ne[0], (tensor)->ne[1], (tensor)->ne[2], (tensor)->ne[3], \
                      (tensor)->nb[0], (tensor)->nb[1], (tensor)->nb[2], (tensor)->nb[3]); \
        } \
    } while(0)

// Log pointer address
#define DEBUG_LOG_PTR(name, ptr) \
    DEBUG_LOG("%s = %p", name, (void*)(ptr))

// Log integer value
#define DEBUG_LOG_INT(name, val) \
    DEBUG_LOG("%s = %d", name, (int)(val))

// Log float value
#define DEBUG_LOG_FLOAT(name, val) \
    DEBUG_LOG("%s = %.6f", name, (float)(val))

// Log buffer content (first N bytes in hex)
#define DEBUG_LOG_BUFFER(name, buf, size, max_bytes) \
    do { \
        if (ggml_debug_enabled() && (buf) != NULL) { \
            DEBUG_LOG("Buffer %s (%d bytes):", name, (int)(size)); \
            int n = (max_bytes) < (size) ? (max_bytes) : (size); \
            fprintf(stderr, "  "); \
            for (int i = 0; i < n; i++) { \
                fprintf(stderr, "%02x ", ((unsigned char*)(buf))[i]); \
                if ((i + 1) % 16 == 0) fprintf(stderr, "\n  "); \
            } \
            fprintf(stderr, "\n"); \
            fflush(stderr); \
        } \
    } while(0)

// ============================================================================
// Conditional Debug (only log if condition is true)
// ============================================================================

#define DEBUG_LOG_IF(cond, fmt, ...) \
    do { \
        if (ggml_debug_enabled() && (cond)) { \
            DEBUG_LOG(fmt, ##__VA_ARGS__); \
        } \
    } while(0)

// ============================================================================
// Performance Timing
// ============================================================================

// Start timing
#define DEBUG_TIMER_START(name) \
    struct timeval _debug_timer_start_##name; \
    if (ggml_debug_enabled()) { \
        gettimeofday(&_debug_timer_start_##name, NULL); \
    }

// End timing and log duration
#define DEBUG_TIMER_END(name) \
    do { \
        if (ggml_debug_enabled()) { \
            struct timeval _debug_timer_end; \
            gettimeofday(&_debug_timer_end, NULL); \
            long long elapsed_us = (_debug_timer_end.tv_sec - _debug_timer_start_##name.tv_sec) * 1000000LL + \
                                   (_debug_timer_end.tv_usec - _debug_timer_start_##name.tv_usec); \
            DEBUG_LOG("Timer %s: %.3f ms", #name, elapsed_us / 1000.0); \
        } \
    } while(0)

// ============================================================================
// Assert with custom message
// ============================================================================

#define DEBUG_ASSERT(cond, fmt, ...) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "[ASSERT FAILED] [%s:%d] " fmt "\n", \
                    __FILE__, __LINE__, ##__VA_ARGS__); \
            fflush(stderr); \
            abort(); \
        } \
    } while(0)

#ifdef __cplusplus
}
#endif
