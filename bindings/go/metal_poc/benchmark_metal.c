/*
 * Benchmark program to compare CPU vs GPU performance for c-kzg-4844
 * Compile with: 
 * clang++ -O3 -framework Metal -framework Foundation -framework CoreGraphics \
 *   benchmark_metal.c kzg_metal_integration.c MetalKZGAccelerator.mm \
 *   -I../../src -L../../lib -lblst -lckzg -o benchmark_metal
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "MetalKZGAccelerator.h"
#include "../../src/ckzg.h"

// External functions from integration layer
extern int init_metal_acceleration(void);
extern void cleanup_metal_acceleration(void);
extern C_KZG_RET gpu_compute_cells_and_kzg_proofs(
    Cell *cells, KZGProof *proofs, const Blob *blob, const KZGSettings *s);

// Timer utility
static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

// Generate random blob for testing
static void generate_random_blob(Blob *blob) {
    uint8_t *bytes = blob->bytes;
    for (size_t i = 0; i < BYTES_PER_BLOB; i++) {
        bytes[i] = rand() % 256;
    }
    // Ensure the blob is valid by clearing the top bits
    for (size_t i = 0; i < FIELD_ELEMENTS_PER_BLOB; i++) {
        size_t offset = i * BYTES_PER_FIELD_ELEMENT;
        bytes[offset] &= 0x1F; // Clear top 3 bits to ensure < modulus
    }
}

// Benchmark a single run
static double benchmark_single_run(
    const char* name,
    C_KZG_RET (*compute_func)(Cell*, KZGProof*, const Blob*, const KZGSettings*),
    const Blob* blob,
    const KZGSettings* s,
    int iterations
) {
    Cell *cells = calloc(CELLS_PER_EXT_BLOB, sizeof(Cell));
    KZGProof *proofs = calloc(CELLS_PER_EXT_BLOB, sizeof(KZGProof));
    
    if (!cells || !proofs) {
        fprintf(stderr, "Memory allocation failed\n");
        return -1;
    }
    
    printf("Running %s benchmark (%d iterations)...\n", name, iterations);
    
    double total_time = 0;
    for (int i = 0; i < iterations; i++) {
        double start = get_time_ms();
        
        C_KZG_RET ret = compute_func(cells, proofs, blob, s);
        
        double end = get_time_ms();
        
        if (ret != C_KZG_OK) {
            fprintf(stderr, "Computation failed with error: %d\n", ret);
            free(cells);
            free(proofs);
            return -1;
        }
        
        double elapsed = end - start;
        total_time += elapsed;
        
        if (i == 0) {
            printf("  First run: %.2f ms\n", elapsed);
        }
    }
    
    double avg_time = total_time / iterations;
    printf("  Average time: %.2f ms\n", avg_time);
    
    free(cells);
    free(proofs);
    return avg_time;
}

// Benchmark FFT operations specifically
static void benchmark_fft_operations(void) {
    printf("\n=== FFT Operation Benchmarks ===\n");
    
    metal_kzg_handle handle = metal_kzg_init();
    if (!handle) {
        fprintf(stderr, "Failed to initialize Metal for FFT benchmark\n");
        return;
    }
    
    // Test different FFT sizes
    size_t sizes[] = {256, 512, 1024, 2048, 4096, 8192};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    printf("\nFFT Performance (10 iterations each):\n");
    printf("Size\t\tGPU (ms)\n");
    printf("----\t\t--------\n");
    
    for (int i = 0; i < num_sizes; i++) {
        double gpu_time = metal_benchmark_fft(handle, sizes[i], 10);
        printf("%zu\t\t%.3f\n", sizes[i], gpu_time);
    }
    
    // Benchmark field multiplication
    printf("\nField Multiplication Performance (1M operations):\n");
    double mul_time = metal_benchmark_field_mul(handle, 1000000, 10);
    printf("GPU batch multiplication: %.3f ms per million ops\n", mul_time);
    
    metal_kzg_cleanup(handle);
}

int main(int argc, char *argv[]) {
    printf("=== c-kzg-4844 Metal GPU Acceleration Benchmark ===\n");
    printf("Running on Apple M1 with Metal GPU acceleration\n\n");
    
    // Initialize random seed
    srand(time(NULL));
    
    // Load trusted setup
    printf("Loading trusted setup...\n");
    KZGSettings *s = malloc(sizeof(KZGSettings));
    if (!s) {
        fprintf(stderr, "Failed to allocate settings\n");
        return 1;
    }
    
    const char* trusted_setup = "../../src/trusted_setup.txt";
    FILE *fp = fopen(trusted_setup, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open trusted setup file\n");
        free(s);
        return 1;
    }
    
    // Initialize with precompute level
    size_t precompute = 8; // Match your preference
    C_KZG_RET ret = load_trusted_setup(s, fp, precompute);
    fclose(fp);
    
    if (ret != C_KZG_OK) {
        fprintf(stderr, "Failed to load trusted setup: %d\n", ret);
        free(s);
        return 1;
    }
    
    printf("Trusted setup loaded (precompute=%zu)\n\n", precompute);
    
    // Generate random blob for testing
    Blob blob;
    generate_random_blob(&blob);
    
    // Run benchmarks
    int iterations = 5;
    
    printf("=== ComputeCellsAndKZGProofs Benchmark ===\n");
    printf("Iterations per test: %d\n", iterations);
    printf("Blob size: %d bytes\n", BYTES_PER_BLOB);
    printf("Cells per blob: %d\n", CELLS_PER_EXT_BLOB);
    printf("Field elements per cell: %d\n\n", FIELD_ELEMENTS_PER_CELL);
    
    // CPU Benchmark
    double cpu_time = benchmark_single_run(
        "CPU (baseline)",
        compute_cells_and_kzg_proofs,
        &blob,
        s,
        iterations
    );
    
    // Initialize Metal acceleration
    if (init_metal_acceleration() == 0) {
        // GPU Benchmark
        double gpu_time = benchmark_single_run(
            "GPU (Metal)",
            gpu_compute_cells_and_kzg_proofs,
            &blob,
            s,
            iterations
        );
        
        // Calculate speedup
        if (cpu_time > 0 && gpu_time > 0) {
            double speedup = cpu_time / gpu_time;
            printf("\n=== Performance Summary ===\n");
            printf("CPU Time: %.2f ms\n", cpu_time);
            printf("GPU Time: %.2f ms\n", gpu_time);
            printf("Speedup: %.2fx\n", speedup);
            
            if (speedup > 1.0) {
                printf("GPU acceleration achieved %.1f%% improvement!\n", 
                       (speedup - 1.0) * 100);
            } else {
                printf("GPU was slower by %.1f%%\n", 
                       (1.0 - speedup) * 100);
            }
        }
        
        // Run FFT-specific benchmarks
        benchmark_fft_operations();
        
        // Cleanup Metal
        cleanup_metal_acceleration();
    } else {
        fprintf(stderr, "Metal acceleration not available\n");
    }
    
    // Cleanup
    free_trusted_setup(s);
    free(s);
    
    printf("\nBenchmark complete!\n");
    return 0;
}