/*
 * Simple benchmark to test Metal GPU acceleration
 * This version focuses on testing the Metal infrastructure
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "MetalKZGAccelerator.h"

// Timer utility
static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

// Test field element operations
void test_field_operations(void) {
    printf("\n=== Testing Metal Field Operations ===\n");
    
    metal_kzg_handle handle = metal_kzg_init();
    if (!handle) {
        fprintf(stderr, "Failed to initialize Metal\n");
        return;
    }
    
    size_t count = 10000;
    gpu_field_element_t *a = calloc(count, sizeof(gpu_field_element_t));
    gpu_field_element_t *b = calloc(count, sizeof(gpu_field_element_t));
    gpu_field_element_t *result = calloc(count, sizeof(gpu_field_element_t));
    
    // Initialize with test data
    for (size_t i = 0; i < count; i++) {
        for (int j = 0; j < 6; j++) {
            a[i].limbs[j] = (uint64_t)(i * 7 + j);
            b[i].limbs[j] = (uint64_t)(i * 13 + j * 5);
        }
    }
    
    // Test field addition
    printf("Testing batch field addition (%zu elements)...\n", count);
    double start = get_time_ms();
    int ret = metal_field_add_batch(handle, a, b, result, count);
    double elapsed = get_time_ms() - start;
    
    if (ret == 0) {
        printf("  Field addition completed in %.3f ms\n", elapsed);
        printf("  Throughput: %.2f million ops/sec\n", (count / elapsed) / 1000.0);
    } else {
        printf("  Field addition failed\n");
    }
    
    // Test field multiplication
    printf("Testing batch field multiplication (%zu elements)...\n", count);
    start = get_time_ms();
    ret = metal_field_mul_batch(handle, a, b, result, count);
    elapsed = get_time_ms() - start;
    
    if (ret == 0) {
        printf("  Field multiplication completed in %.3f ms\n", elapsed);
        printf("  Throughput: %.2f million ops/sec\n", (count / elapsed) / 1000.0);
    } else {
        printf("  Field multiplication failed\n");
    }
    
    free(a);
    free(b);
    free(result);
    metal_kzg_cleanup(handle);
}

// Test FFT operations
void test_fft_operations(void) {
    printf("\n=== Testing Metal FFT Operations ===\n");
    
    metal_kzg_handle handle = metal_kzg_init();
    if (!handle) {
        fprintf(stderr, "Failed to initialize Metal\n");
        return;
    }
    
    // Test different FFT sizes
    size_t sizes[] = {256, 512, 1024, 2048, 4096};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    printf("FFT Performance (averaged over 10 iterations):\n");
    printf("Size\t\tTime (ms)\tThroughput (Mpoints/sec)\n");
    printf("----\t\t---------\t------------------------\n");
    
    for (int i = 0; i < num_sizes; i++) {
        double gpu_time = metal_benchmark_fft(handle, sizes[i], 10);
        double throughput = (sizes[i] / gpu_time) / 1000.0; // Million points per second
        printf("%zu\t\t%.3f\t\t%.2f\n", sizes[i], gpu_time, throughput);
    }
    
    metal_kzg_cleanup(handle);
}

// Test basic Metal functionality
void test_metal_basic(void) {
    printf("\n=== Testing Basic Metal Functionality ===\n");
    
    printf("Initializing Metal...\n");
    metal_kzg_handle handle = metal_kzg_init();
    
    if (handle) {
        printf("✓ Metal initialized successfully\n");
        printf("✓ Metal device available\n");
        printf("✓ Compute pipelines created\n");
        
        // Simple test
        gpu_field_element_t a[1], b[1], result[1];
        memset(a, 0x42, sizeof(a));
        memset(b, 0x13, sizeof(b));
        
        int ret = metal_field_add_batch(handle, a, b, result, 1);
        if (ret == 0) {
            printf("✓ Basic computation successful\n");
        } else {
            printf("✗ Basic computation failed\n");
        }
        
        metal_kzg_cleanup(handle);
        printf("✓ Metal cleanup successful\n");
    } else {
        printf("✗ Failed to initialize Metal\n");
        printf("  Make sure you're running on an Apple Silicon Mac\n");
    }
}

int main(int argc, char *argv[]) {
    printf("=== Metal GPU Acceleration Test Suite ===\n");
    printf("Running on: ");
    system("sysctl -n machdep.cpu.brand_string");
    printf("\n");
    
    // Test basic functionality first
    test_metal_basic();
    
    // Test field operations
    test_field_operations();
    
    // Test FFT operations  
    test_fft_operations();
    
    printf("\n=== All tests complete ===\n");
    return 0;
}