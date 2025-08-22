/*
 * Integration layer between c-kzg-4844 and Metal GPU acceleration
 * This demonstrates how to accelerate compute_cells_and_kzg_proofs using GPU
 */

#include "MetalKZGAccelerator.h"
#include "../../src/ckzg.h"
#include "../../src/common/fr.h"
#include "../../src/eip7594/eip7594.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

// Global Metal accelerator handle
static metal_kzg_handle g_metal_handle = NULL;

// Initialize Metal acceleration
int init_metal_acceleration(void) {
    if (g_metal_handle != NULL) {
        return 0; // Already initialized
    }
    
    g_metal_handle = metal_kzg_init();
    if (g_metal_handle == NULL) {
        fprintf(stderr, "Failed to initialize Metal acceleration\n");
        return -1;
    }
    
    printf("Metal acceleration initialized successfully\n");
    return 0;
}

// Cleanup Metal acceleration
void cleanup_metal_acceleration(void) {
    if (g_metal_handle != NULL) {
        metal_kzg_cleanup(g_metal_handle);
        g_metal_handle = NULL;
        printf("Metal acceleration cleaned up\n");
    }
}

// GPU-accelerated FFT for field elements
C_KZG_RET gpu_fr_fft(fr_t *out, const fr_t *in, size_t n, const KZGSettings *s) {
    if (g_metal_handle == NULL) {
        fprintf(stderr, "Metal not initialized, falling back to CPU\n");
        return fr_fft(out, in, n, s);
    }
    
    // Convert fr_t array to GPU format
    gpu_field_element_t* gpu_data = calloc(n, sizeof(gpu_field_element_t));
    gpu_field_element_t* gpu_roots = calloc(n, sizeof(gpu_field_element_t));
    
    if (!gpu_data || !gpu_roots) {
        free(gpu_data);
        free(gpu_roots);
        return C_KZG_MALLOC;
    }
    
    // Copy input data
    for (size_t i = 0; i < n; i++) {
        fr_to_gpu_element(&in[i], &gpu_data[i]);
    }
    
    // Copy roots of unity (simplified - would need proper root extraction)
    size_t roots_stride = FIELD_ELEMENTS_PER_EXT_BLOB / n;
    for (size_t i = 0; i < n; i++) {
        fr_to_gpu_element(&s->roots_of_unity[i * roots_stride], &gpu_roots[i]);
    }
    
    // Perform GPU FFT
    int ret = metal_fft_fr(g_metal_handle, gpu_data, gpu_roots, n, false);
    
    if (ret == 0) {
        // Copy results back
        for (size_t i = 0; i < n; i++) {
            gpu_element_to_fr(&gpu_data[i], &out[i]);
        }
    }
    
    free(gpu_data);
    free(gpu_roots);
    
    return (ret == 0) ? C_KZG_OK : C_KZG_ERROR;
}

// GPU-accelerated compute_cells_and_kzg_proofs
C_KZG_RET gpu_compute_cells_and_kzg_proofs(
    Cell *cells,
    KZGProof *proofs,
    const Blob *blob,
    const KZGSettings *s
) {
    C_KZG_RET ret;
    fr_t *poly_monomial = NULL;
    fr_t *poly_lagrange = NULL;
    fr_t *data_fr = NULL;
    g1_t *proofs_g1 = NULL;
    
    printf("Using GPU-accelerated compute_cells_and_kzg_proofs\n");
    
    // Initialize Metal if not already done
    if (g_metal_handle == NULL) {
        if (init_metal_acceleration() != 0) {
            // Fall back to CPU implementation
            return compute_cells_and_kzg_proofs(cells, proofs, blob, s);
        }
    }
    
    // Allocate space for arrays
    ret = new_fr_array(&poly_monomial, FIELD_ELEMENTS_PER_EXT_BLOB);
    if (ret != C_KZG_OK) goto out;
    ret = new_fr_array(&poly_lagrange, FIELD_ELEMENTS_PER_EXT_BLOB);
    if (ret != C_KZG_OK) goto out;
    
    // Convert blob to polynomial
    ret = blob_to_polynomial(poly_lagrange, blob);
    if (ret != C_KZG_OK) goto out;
    
    // GPU-accelerated Lagrange to monomial conversion (via FFT)
    printf("Performing GPU-accelerated polynomial conversion...\n");
    
    // Bit-reverse the input
    fr_t *temp = calloc(FIELD_ELEMENTS_PER_BLOB, sizeof(fr_t));
    memcpy(temp, poly_lagrange, FIELD_ELEMENTS_PER_BLOB * sizeof(fr_t));
    ret = bit_reversal_permutation(temp, sizeof(fr_t), FIELD_ELEMENTS_PER_BLOB);
    if (ret != C_KZG_OK) {
        free(temp);
        goto out;
    }
    
    // Use GPU for inverse FFT
    ret = gpu_fr_fft(poly_monomial, temp, FIELD_ELEMENTS_PER_BLOB, s);
    free(temp);
    if (ret != C_KZG_OK) goto out;
    
    // Zero out the upper half
    for (size_t i = FIELD_ELEMENTS_PER_BLOB; i < FIELD_ELEMENTS_PER_EXT_BLOB; i++) {
        poly_monomial[i] = FR_ZERO;
    }
    
    if (cells != NULL) {
        // Allocate space for data points
        ret = new_fr_array(&data_fr, FIELD_ELEMENTS_PER_EXT_BLOB);
        if (ret != C_KZG_OK) goto out;
        
        // GPU-accelerated forward FFT
        printf("Computing cells via GPU FFT...\n");
        ret = gpu_fr_fft(data_fr, poly_monomial, FIELD_ELEMENTS_PER_EXT_BLOB, s);
        if (ret != C_KZG_OK) goto out;
        
        // Bit-reverse the data points
        ret = bit_reversal_permutation(data_fr, sizeof(fr_t), FIELD_ELEMENTS_PER_EXT_BLOB);
        if (ret != C_KZG_OK) goto out;
        
        // Convert cells to byte form
        for (size_t i = 0; i < CELLS_PER_EXT_BLOB; i++) {
            for (size_t j = 0; j < FIELD_ELEMENTS_PER_CELL; j++) {
                size_t index = i * FIELD_ELEMENTS_PER_CELL + j;
                size_t offset = j * BYTES_PER_FIELD_ELEMENT;
                bytes_from_bls_field((Bytes32 *)&cells[i].bytes[offset], &data_fr[index]);
            }
        }
    }
    
    if (proofs != NULL) {
        // For proofs, we'd need GPU-accelerated FK20
        // For now, fall back to CPU for this part
        printf("Computing proofs (CPU fallback for FK20)...\n");
        
        ret = new_g1_array(&proofs_g1, CELLS_PER_EXT_BLOB);
        if (ret != C_KZG_OK) goto out;
        
        ret = compute_fk20_cell_proofs(proofs_g1, poly_monomial, s);
        if (ret != C_KZG_OK) goto out;
        
        ret = bit_reversal_permutation(proofs_g1, sizeof(g1_t), CELLS_PER_EXT_BLOB);
        if (ret != C_KZG_OK) goto out;
        
        for (size_t i = 0; i < CELLS_PER_EXT_BLOB; i++) {
            bytes_from_g1(&proofs[i], &proofs_g1[i]);
        }
    }
    
    printf("GPU-accelerated computation complete\n");
    
out:
    c_kzg_free(poly_monomial);
    c_kzg_free(poly_lagrange);
    c_kzg_free(data_fr);
    c_kzg_free(proofs_g1);
    return ret;
}