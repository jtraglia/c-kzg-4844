/*
 * Copyright 2024 Benjamin Edgington
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifdef EXPERIMENTAL_METAL_SUPPORT

#include "common/fr.h"
#include "common/ret.h"

#include <stdbool.h>
#include <stddef.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
// Types
////////////////////////////////////////////////////////////////////////////////////////////////////

/** Opaque handle to Metal context (implementation details hidden in .m file). */
typedef struct MetalContext MetalContext;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Public Functions
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize the Metal context.
 *
 * @param[out]  ctx Pointer to store the created context
 *
 * @retval C_KZG_OK     Success
 * @retval C_KZG_ERROR  Metal not available or initialization failed
 */
C_KZG_RET metal_init(MetalContext **ctx);

/**
 * Free the Metal context.
 *
 * @param[in]   ctx The context to free (can be NULL)
 */
void metal_free(MetalContext *ctx);

/**
 * Check if Metal is available on this system.
 *
 * @retval true   Metal is available
 * @retval false  Metal is not available
 */
bool metal_is_available(void);

/**
 * Perform FFT on field elements using Metal GPU.
 *
 * @param[in]   ctx             The Metal context
 * @param[out]  out             Output array, length n
 * @param[in]   in              Input array, length n
 * @param[in]   n               Size of the FFT (must be power of 2)
 * @param[in]   roots           Roots of unity array
 * @param[in]   roots_stride    Stride in the roots array
 * @param[in]   inverse         If true, perform inverse FFT
 *
 * @retval C_KZG_OK      Success
 * @retval C_KZG_ERROR   GPU operation failed
 */
C_KZG_RET metal_fr_fft(
    MetalContext *ctx,
    fr_t *out,
    const fr_t *in,
    size_t n,
    const fr_t *roots,
    size_t roots_stride,
    bool inverse
);

/**
 * Perform batch FFT on multiple arrays of field elements.
 *
 * @param[in]   ctx             The Metal context
 * @param[out]  out             Output arrays (array of n-element arrays)
 * @param[in]   in              Input arrays (array of n-element arrays)
 * @param[in]   batch_count     Number of FFTs to perform
 * @param[in]   n               Size of each FFT (must be power of 2)
 * @param[in]   roots           Roots of unity array
 * @param[in]   roots_stride    Stride in the roots array
 * @param[in]   inverse         If true, perform inverse FFT
 *
 * @retval C_KZG_OK      Success
 * @retval C_KZG_ERROR   GPU operation failed
 */
C_KZG_RET metal_fr_fft_batch(
    MetalContext *ctx,
    fr_t **out,
    const fr_t **in,
    size_t batch_count,
    size_t n,
    const fr_t *roots,
    size_t roots_stride,
    bool inverse
);

#ifdef __cplusplus
}
#endif

#endif /* EXPERIMENTAL_METAL_SUPPORT */
