#ifndef METAL_KZG_ACCELERATOR_H
#define METAL_KZG_ACCELERATOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

// Field element structure matching the Metal shader
typedef struct {
    uint64_t limbs[6];  // 384 bits for BLS12-381
} gpu_field_element_t;

// Opaque handle to the Metal accelerator
typedef void* metal_kzg_handle;

// Initialize the Metal accelerator
// Returns NULL on failure
metal_kzg_handle metal_kzg_init(void);

// Cleanup and release resources
void metal_kzg_cleanup(metal_kzg_handle handle);

// GPU-accelerated field operations
int metal_field_add_batch(
    metal_kzg_handle handle,
    const gpu_field_element_t* a,
    const gpu_field_element_t* b,
    gpu_field_element_t* result,
    size_t count
);

int metal_field_mul_batch(
    metal_kzg_handle handle,
    const gpu_field_element_t* a,
    const gpu_field_element_t* b,
    gpu_field_element_t* result,
    size_t count
);

// GPU-accelerated FFT
int metal_fft_fr(
    metal_kzg_handle handle,
    gpu_field_element_t* data,
    const gpu_field_element_t* roots,
    size_t n,
    bool inverse
);

// GPU-accelerated MSM (simplified interface)
int metal_msm_g1(
    metal_kzg_handle handle,
    const gpu_field_element_t* scalars,
    const void* points,  // g1_t points
    void* result,         // g1_t result
    size_t count
);

// Benchmark functions
double metal_benchmark_fft(metal_kzg_handle handle, size_t size, int iterations);
double metal_benchmark_field_mul(metal_kzg_handle handle, size_t count, int iterations);

// Convert between c-kzg-4844 fr_t and gpu_field_element_t
void fr_to_gpu_element(const void* fr, gpu_field_element_t* gpu_elem);
void gpu_element_to_fr(const gpu_field_element_t* gpu_elem, void* fr);

#ifdef __cplusplus
}
#endif

#endif // METAL_KZG_ACCELERATOR_H