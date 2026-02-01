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

#ifdef EXPERIMENTAL_METAL_SUPPORT

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal/metal_api.h"
#include "common/alloc.h"

#include <stdlib.h>
#include <string.h>

/* Use our own min macro to avoid GNU extension warnings from Foundation's MIN */
#undef MIN
static inline NSUInteger metal_min(NSUInteger a, NSUInteger b) {
    return a < b ? a : b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Constants
////////////////////////////////////////////////////////////////////////////////////////////////////

/** Maximum number of elements we can process in a single batch. */
#define MAX_BUFFER_ELEMENTS (1024 * 1024)

/** Threshold below which CPU is likely faster than GPU. */
#define MIN_GPU_FFT_SIZE 256

////////////////////////////////////////////////////////////////////////////////////////////////////
// Metal Context Structure
////////////////////////////////////////////////////////////////////////////////////////////////////

struct MetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;

    /* Compute pipeline states for each kernel */
    id<MTLComputePipelineState> fftButterflyPipeline;
    id<MTLComputePipelineState> fftRadix2StagePipeline;
    id<MTLComputePipelineState> bitReversalPipeline;
    id<MTLComputePipelineState> scaleElementsPipeline;
    id<MTLComputePipelineState> copyElementsPipeline;

    /* Reusable buffers for roots of unity */
    id<MTLBuffer> rootsBuffer;
    size_t rootsBufferSize;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper Functions
////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Calculate log2 of a power-of-two value.
 */
static uint32_t log2_uint(uint32_t n) {
    uint32_t result = 0;
    while (n > 1) {
        n >>= 1;
        result++;
    }
    return result;
}

/**
 * Create a compute pipeline state from a function name.
 */
static id<MTLComputePipelineState> createPipeline(
    id<MTLDevice> device, id<MTLLibrary> library, NSString *functionName, NSError **error
) {
    id<MTLFunction> function = [library newFunctionWithName:functionName];
    if (function == nil) {
        if (error) {
            *error = [NSError
                errorWithDomain:@"MetalKZG"
                           code:1
                       userInfo:@{NSLocalizedDescriptionKey : @"Failed to find kernel function"}];
        }
        return nil;
    }

    return [device newComputePipelineStateWithFunction:function error:error];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Public API Implementation
////////////////////////////////////////////////////////////////////////////////////////////////////

bool metal_is_available(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device != nil;
    }
}

C_KZG_RET metal_init(MetalContext **ctx) {
    @autoreleasepool {
        /* Allocate context */
        MetalContext *context = (MetalContext *)calloc(1, sizeof(MetalContext));
        if (context == NULL) {
            return C_KZG_MALLOC;
        }

        /* Get default Metal device */
        context->device = MTLCreateSystemDefaultDevice();
        if (context->device == nil) {
            free(context);
            return C_KZG_ERROR;
        }

        /* Create command queue */
        context->commandQueue = [context->device newCommandQueue];
        if (context->commandQueue == nil) {
            free(context);
            return C_KZG_ERROR;
        }

        /* Load Metal library from default location */
        NSError *error = nil;

        /* Try to load from compiled metallib first */
        NSString *libraryPath =
            [[NSBundle mainBundle] pathForResource:@"kzg_kernels" ofType:@"metallib"];

        if (libraryPath != nil) {
            NSURL *libraryURL = [NSURL fileURLWithPath:libraryPath];
            context->library = [context->device newLibraryWithURL:libraryURL error:&error];
        }

        /* If that fails, try to compile from source */
        if (context->library == nil) {
            /* Get the path to the metal source file */
            /* In production, the metallib should be pre-compiled */
            NSString *sourcePath = nil;

            /* Try several possible locations */
            NSArray *searchPaths = @[
                @"src/metal/kzg_kernels.metal",
                @"../src/metal/kzg_kernels.metal",
                @"kzg_kernels.metal"
            ];

            NSFileManager *fm = [NSFileManager defaultManager];
            for (NSString *path in searchPaths) {
                if ([fm fileExistsAtPath:path]) {
                    sourcePath = path;
                    break;
                }
            }

            if (sourcePath != nil) {
                NSString *source = [NSString stringWithContentsOfFile:sourcePath
                                                             encoding:NSUTF8StringEncoding
                                                                error:&error];
                if (source != nil) {
                    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
                    options.mathMode = MTLMathModeFast;
#else
                    options.fastMathEnabled = YES;
#endif
                    context->library = [context->device newLibraryWithSource:source
                                                                     options:options
                                                                       error:&error];
                }
            }
        }

        if (context->library == nil) {
            NSLog(@"Failed to load Metal library: %@", error);
            free(context);
            return C_KZG_ERROR;
        }

        /* Create compute pipelines */
        context->fftButterflyPipeline =
            createPipeline(context->device, context->library, @"fft_butterfly", &error);
        if (context->fftButterflyPipeline == nil) {
            NSLog(@"Failed to create fft_butterfly pipeline: %@", error);
            free(context);
            return C_KZG_ERROR;
        }

        context->fftRadix2StagePipeline =
            createPipeline(context->device, context->library, @"fft_radix2_stage", &error);
        if (context->fftRadix2StagePipeline == nil) {
            NSLog(@"Failed to create fft_radix2_stage pipeline: %@", error);
            free(context);
            return C_KZG_ERROR;
        }

        context->bitReversalPipeline =
            createPipeline(context->device, context->library, @"bit_reversal", &error);
        if (context->bitReversalPipeline == nil) {
            NSLog(@"Failed to create bit_reversal pipeline: %@", error);
            free(context);
            return C_KZG_ERROR;
        }

        context->scaleElementsPipeline =
            createPipeline(context->device, context->library, @"scale_elements", &error);
        if (context->scaleElementsPipeline == nil) {
            NSLog(@"Failed to create scale_elements pipeline: %@", error);
            free(context);
            return C_KZG_ERROR;
        }

        context->copyElementsPipeline =
            createPipeline(context->device, context->library, @"copy_elements", &error);
        if (context->copyElementsPipeline == nil) {
            NSLog(@"Failed to create copy_elements pipeline: %@", error);
            free(context);
            return C_KZG_ERROR;
        }

        context->rootsBuffer = nil;
        context->rootsBufferSize = 0;

        *ctx = context;
        return C_KZG_OK;
    }
}

void metal_free(MetalContext *ctx) {
    if (ctx == NULL) return;

    @autoreleasepool {
        /* Release Metal objects */
        ctx->fftButterflyPipeline = nil;
        ctx->fftRadix2StagePipeline = nil;
        ctx->bitReversalPipeline = nil;
        ctx->scaleElementsPipeline = nil;
        ctx->copyElementsPipeline = nil;
        ctx->rootsBuffer = nil;
        ctx->library = nil;
        ctx->commandQueue = nil;
        ctx->device = nil;

        free(ctx);
    }
}

C_KZG_RET metal_fr_fft(
    MetalContext *ctx,
    fr_t *out,
    const fr_t *in,
    size_t n,
    const fr_t *roots,
    size_t roots_stride,
    bool inverse
) {
    @autoreleasepool {
        if (ctx == NULL || out == NULL || in == NULL || roots == NULL) {
            return C_KZG_BADARGS;
        }

        /* For small sizes, CPU is likely faster */
        if (n < MIN_GPU_FFT_SIZE) {
            /* Fall back to CPU - caller should handle this */
            return C_KZG_ERROR;
        }

        uint32_t log_n = log2_uint((uint32_t)n);

        /* Create buffers */
        size_t data_size = n * sizeof(fr_t);
        size_t roots_size = n * sizeof(fr_t); /* We need at most n roots */

        id<MTLBuffer> dataBuffer =
            [ctx->device newBufferWithBytes:in length:data_size options:MTLResourceStorageModeShared];

        if (dataBuffer == nil) {
            return C_KZG_MALLOC;
        }

        /* Copy roots to a contiguous buffer, respecting stride */
        fr_t *roots_contiguous = (fr_t *)malloc(roots_size);
        if (roots_contiguous == NULL) {
            return C_KZG_MALLOC;
        }

        for (size_t i = 0; i < n; i++) {
            roots_contiguous[i] = roots[i * roots_stride];
        }

        id<MTLBuffer> rootsBuffer = [ctx->device newBufferWithBytes:roots_contiguous
                                                             length:roots_size
                                                            options:MTLResourceStorageModeShared];
        free(roots_contiguous);

        if (rootsBuffer == nil) {
            return C_KZG_MALLOC;
        }

        /* Create command buffer */
        id<MTLCommandBuffer> commandBuffer = [ctx->commandQueue commandBuffer];
        if (commandBuffer == nil) {
            return C_KZG_ERROR;
        }

        /* Perform bit-reversal permutation first */
        {
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:ctx->bitReversalPipeline];
            [encoder setBuffer:dataBuffer offset:0 atIndex:0];

            uint32_t n_uint = (uint32_t)n;
            [encoder setBytes:&n_uint length:sizeof(uint32_t) atIndex:1];
            [encoder setBytes:&log_n length:sizeof(uint32_t) atIndex:2];

            MTLSize gridSize = MTLSizeMake(n, 1, 1);
            NSUInteger threadGroupSize =
                metal_min(ctx->bitReversalPipeline.maxTotalThreadsPerThreadgroup, n);
            MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];
        }

        /* Perform FFT stages */
        for (uint32_t stage = 0; stage < log_n; stage++) {
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:ctx->fftRadix2StagePipeline];
            [encoder setBuffer:dataBuffer offset:0 atIndex:0];
            [encoder setBuffer:rootsBuffer offset:0 atIndex:1];

            uint32_t n_uint = (uint32_t)n;
            uint32_t roots_stride_base = (uint32_t)(n / 2);

            [encoder setBytes:&n_uint length:sizeof(uint32_t) atIndex:2];
            [encoder setBytes:&stage length:sizeof(uint32_t) atIndex:3];
            [encoder setBytes:&roots_stride_base length:sizeof(uint32_t) atIndex:4];

            uint32_t num_butterflies = (uint32_t)(n / 2);
            MTLSize gridSize = MTLSizeMake(num_butterflies, 1, 1);
            NSUInteger threadGroupSize =
                metal_min(ctx->fftRadix2StagePipeline.maxTotalThreadsPerThreadgroup, num_butterflies);
            MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];
        }

        /* For inverse FFT, scale by 1/n */
        if (inverse) {
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:ctx->scaleElementsPipeline];
            [encoder setBuffer:dataBuffer offset:0 atIndex:0];

            /* Compute 1/n in the field */
            /* This needs to be the Montgomery form of 1/n */
            /* For now, we pass n and let the kernel compute the inverse */
            /* Actually, we need to precompute this on CPU */
            fr_t inv_n;
            fr_from_uint64(&inv_n, n);
            blst_fr_eucl_inverse(&inv_n, &inv_n);

            [encoder setBytes:&inv_n length:sizeof(fr_t) atIndex:1];

            uint32_t n_uint = (uint32_t)n;
            [encoder setBytes:&n_uint length:sizeof(uint32_t) atIndex:2];

            MTLSize gridSize = MTLSizeMake(n, 1, 1);
            NSUInteger threadGroupSize =
                metal_min(ctx->scaleElementsPipeline.maxTotalThreadsPerThreadgroup, n);
            MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];
        }

        /* Commit and wait */
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (commandBuffer.status == MTLCommandBufferStatusError) {
            NSLog(@"Metal command buffer error: %@", commandBuffer.error);
            return C_KZG_ERROR;
        }

        /* Copy results back */
        memcpy(out, dataBuffer.contents, data_size);

        return C_KZG_OK;
    }
}

C_KZG_RET metal_fr_fft_batch(
    MetalContext *ctx,
    fr_t **out,
    const fr_t **in,
    size_t batch_count,
    size_t n,
    const fr_t *roots,
    size_t roots_stride,
    bool inverse
) {
    /* For batch FFT, we process each FFT sequentially for now */
    /* A more optimized version would process them in parallel on the GPU */
    for (size_t i = 0; i < batch_count; i++) {
        C_KZG_RET ret = metal_fr_fft(ctx, out[i], in[i], n, roots, roots_stride, inverse);
        if (ret != C_KZG_OK) {
            return ret;
        }
    }
    return C_KZG_OK;
}

#endif /* EXPERIMENTAL_METAL_SUPPORT */
