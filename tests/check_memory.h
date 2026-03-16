/**
 * Memory availability checker for macOS
 */

#pragma once

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <mach/mach.h>
#include <sys/sysctl.h>

// Get available memory in MB
static size_t get_available_memory_mb() {
    mach_port_t host_port = mach_host_self();
    vm_size_t page_size;
    host_page_size(host_port, &page_size);

    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t host_size = sizeof(vm_stat) / sizeof(integer_t);

    if (host_statistics64(host_port, HOST_VM_INFO64, (host_info64_t)&vm_stat, &host_size) != KERN_SUCCESS) {
        fprintf(stderr, "Warning: Failed to get VM statistics\n");
        return 0;
    }

    // Available = free + inactive
    size_t free_bytes = (vm_stat.free_count + vm_stat.inactive_count) * page_size;
    return free_bytes / (1024 * 1024);
}

// Get total memory in MB
static size_t get_total_memory_mb() {
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    int64_t total_bytes = 0;
    size_t length = sizeof(total_bytes);

    if (sysctl(mib, 2, &total_bytes, &length, NULL, 0) != 0) {
        fprintf(stderr, "Warning: Failed to get total memory\n");
        return 0;
    }

    return total_bytes / (1024 * 1024);
}

/**
 * Check if enough memory is available
 * @param required_mb Required memory in MB
 * @param test_name Name of the test (for logging)
 * @return true if enough memory, false otherwise
 */
static bool check_memory_availability(size_t required_mb, const char* test_name) {
    size_t available_mb = get_available_memory_mb();
    size_t total_mb = get_total_memory_mb();

    fprintf(stderr, "\n=== Memory Check ===\n");
    fprintf(stderr, "Test: %s\n", test_name);
    fprintf(stderr, "Total Memory: %zu MB\n", total_mb);
    fprintf(stderr, "Available Memory: %zu MB\n", available_mb);
    fprintf(stderr, "Required Memory: %zu MB\n", required_mb);

    if (available_mb < required_mb) {
        fprintf(stderr, "\n⚠️  WARNING: Insufficient memory!\n");
        fprintf(stderr, "   Available: %zu MB\n", available_mb);
        fprintf(stderr, "   Required:  %zu MB\n", required_mb);
        fprintf(stderr, "   Shortage:  %zu MB\n", required_mb - available_mb);
        fprintf(stderr, "\n");
        fprintf(stderr, "Possible reasons:\n");
        fprintf(stderr, "  - Another process is using memory\n");
        fprintf(stderr, "  - Another test is running\n");
        fprintf(stderr, "  - System caches are occupying memory\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "Waiting 30 seconds for memory to be freed...\n");
        fprintf(stderr, "(You can terminate with Ctrl+C if needed)\n");
        fprintf(stderr, "\n");

        // Sleep 30 seconds
        for (int i = 30; i > 0; i--) {
            fprintf(stderr, "\rWaiting: %d seconds remaining...   ", i);
            fflush(stderr);
            sleep(1);
        }
        fprintf(stderr, "\n\n");

        // Re-check after waiting
        available_mb = get_available_memory_mb();
        fprintf(stderr, "After waiting:\n");
        fprintf(stderr, "  Available Memory: %zu MB\n", available_mb);

        if (available_mb < required_mb) {
            fprintf(stderr, "\n❌ Still insufficient memory after waiting!\n");
            fprintf(stderr, "   Please free up memory and try again.\n");
            fprintf(stderr, "   Suggestions:\n");
            fprintf(stderr, "   - Close other applications\n");
            fprintf(stderr, "   - Wait for other tests to complete\n");
            fprintf(stderr, "   - Use a smaller model\n");
            fprintf(stderr, "\n");
            return false;
        } else {
            fprintf(stderr, "✅ Memory now available, proceeding...\n\n");
            return true;
        }
    } else {
        fprintf(stderr, "✅ Sufficient memory available\n");
        fprintf(stderr, "   Margin: %zu MB\n\n", available_mb - required_mb);
        return true;
    }
}
