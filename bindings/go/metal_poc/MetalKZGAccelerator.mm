#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <simd/simd.h>
#include "MetalKZGAccelerator.h"
#include <chrono>
#include <string.h>

struct MetalKZGContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;
    
    // Compute pipelines
    id<MTLComputePipelineState> fieldAddPipeline;
    id<MTLComputePipelineState> fieldMulPipeline;
    id<MTLComputePipelineState> fftButterflyPipeline;
    id<MTLComputePipelineState> fftRadix2Pipeline;
    id<MTLComputePipelineState> msmAccumulatePipeline;
    
    // Buffers
    id<MTLBuffer> scratchBuffer;
    size_t scratchBufferSize;
};

// Helper function to create compute pipeline
static id<MTLComputePipelineState> createComputePipeline(id<MTLDevice> device, 
                                                         id<MTLLibrary> library,
                                                         NSString* functionName) {
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:functionName];
    if (!function) {
        NSLog(@"Failed to find function: %@", functionName);
        return nil;
    }
    
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                error:&error];
    if (!pipeline) {
        NSLog(@"Failed to create pipeline for %@: %@", functionName, error);
    }
    return pipeline;
}

metal_kzg_handle metal_kzg_init(void) {
    @autoreleasepool {
        MetalKZGContext* ctx = new MetalKZGContext;
        
        // Get the default Metal device
        ctx->device = MTLCreateSystemDefaultDevice();
        if (!ctx->device) {
            NSLog(@"Metal is not supported on this device");
            delete ctx;
            return nullptr;
        }
        
        // Create command queue
        ctx->commandQueue = [ctx->device newCommandQueue];
        if (!ctx->commandQueue) {
            NSLog(@"Failed to create command queue");
            delete ctx;
            return nullptr;
        }
        
        // Load the Metal shader library
        NSError* error = nil;
        NSString* shaderPath = @"metal_poc/bls12_381_field.metal";
        NSString* source = [NSString stringWithContentsOfFile:shaderPath
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        if (!source) {
            // Try relative path
            shaderPath = @"bls12_381_field.metal";
            source = [NSString stringWithContentsOfFile:shaderPath
                                               encoding:NSUTF8StringEncoding
                                                  error:&error];
        }
        
        if (!source) {
            NSLog(@"Failed to load shader source: %@", error);
            delete ctx;
            return nullptr;
        }
        
        ctx->library = [ctx->device newLibraryWithSource:source options:nil error:&error];
        if (!ctx->library) {
            NSLog(@"Failed to compile Metal library: %@", error);
            delete ctx;
            return nullptr;
        }
        
        // Create compute pipelines
        ctx->fieldAddPipeline = createComputePipeline(ctx->device, ctx->library, @"field_add_batch");
        ctx->fieldMulPipeline = createComputePipeline(ctx->device, ctx->library, @"field_mul_batch");
        ctx->fftButterflyPipeline = createComputePipeline(ctx->device, ctx->library, @"fft_butterfly_batch");
        ctx->fftRadix2Pipeline = createComputePipeline(ctx->device, ctx->library, @"fft_radix2_layer");
        ctx->msmAccumulatePipeline = createComputePipeline(ctx->device, ctx->library, @"msm_accumulate");
        
        // Allocate scratch buffer (16 MB initially)
        ctx->scratchBufferSize = 16 * 1024 * 1024;
        ctx->scratchBuffer = [ctx->device newBufferWithLength:ctx->scratchBufferSize
                                                      options:MTLResourceStorageModeShared];
        
        return ctx;
    }
}

void metal_kzg_cleanup(metal_kzg_handle handle) {
    if (handle) {
        MetalKZGContext* ctx = (MetalKZGContext*)handle;
        delete ctx;
    }
}

int metal_field_add_batch(
    metal_kzg_handle handle,
    const gpu_field_element_t* a,
    const gpu_field_element_t* b,
    gpu_field_element_t* result,
    size_t count
) {
    @autoreleasepool {
        MetalKZGContext* ctx = (MetalKZGContext*)handle;
        if (!ctx || !ctx->fieldAddPipeline) return -1;
        
        size_t bufferSize = count * sizeof(gpu_field_element_t);
        
        // Create Metal buffers
        id<MTLBuffer> bufferA = [ctx->device newBufferWithBytes:a
                                                         length:bufferSize
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [ctx->device newBufferWithBytes:b
                                                         length:bufferSize
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferResult = [ctx->device newBufferWithLength:bufferSize
                                                              options:MTLResourceStorageModeShared];
        
        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:ctx->fieldAddPipeline];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferB offset:0 atIndex:1];
        [encoder setBuffer:bufferResult offset:0 atIndex:2];
        
        // Calculate thread groups
        NSUInteger threadGroupSize = ctx->fieldAddPipeline.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > count) threadGroupSize = count;
        
        MTLSize threadsPerThreadgroup = MTLSizeMake(threadGroupSize, 1, 1);
        MTLSize numThreadgroups = MTLSizeMake((count + threadGroupSize - 1) / threadGroupSize, 1, 1);
        
        [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Copy result back
        memcpy(result, bufferResult.contents, bufferSize);
        
        return 0;
    }
}

int metal_field_mul_batch(
    metal_kzg_handle handle,
    const gpu_field_element_t* a,
    const gpu_field_element_t* b,
    gpu_field_element_t* result,
    size_t count
) {
    @autoreleasepool {
        MetalKZGContext* ctx = (MetalKZGContext*)handle;
        if (!ctx || !ctx->fieldMulPipeline) return -1;
        
        size_t bufferSize = count * sizeof(gpu_field_element_t);
        
        // Create Metal buffers
        id<MTLBuffer> bufferA = [ctx->device newBufferWithBytes:a
                                                         length:bufferSize
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [ctx->device newBufferWithBytes:b
                                                         length:bufferSize
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferResult = [ctx->device newBufferWithLength:bufferSize
                                                              options:MTLResourceStorageModeShared];
        
        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:ctx->fieldMulPipeline];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferB offset:0 atIndex:1];
        [encoder setBuffer:bufferResult offset:0 atIndex:2];
        
        // Calculate thread groups
        NSUInteger threadGroupSize = ctx->fieldMulPipeline.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > count) threadGroupSize = count;
        
        MTLSize threadsPerThreadgroup = MTLSizeMake(threadGroupSize, 1, 1);
        MTLSize numThreadgroups = MTLSizeMake((count + threadGroupSize - 1) / threadGroupSize, 1, 1);
        
        [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Copy result back
        memcpy(result, bufferResult.contents, bufferSize);
        
        return 0;
    }
}

int metal_fft_fr(
    metal_kzg_handle handle,
    gpu_field_element_t* data,
    const gpu_field_element_t* roots,
    size_t n,
    bool inverse
) {
    @autoreleasepool {
        MetalKZGContext* ctx = (MetalKZGContext*)handle;
        if (!ctx || !ctx->fftRadix2Pipeline) return -1;
        
        size_t dataSize = n * sizeof(gpu_field_element_t);
        size_t rootsSize = n * sizeof(gpu_field_element_t);
        
        // Create Metal buffers
        id<MTLBuffer> bufferData = [ctx->device newBufferWithBytes:data
                                                           length:dataSize
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferRoots = [ctx->device newBufferWithBytes:roots
                                                            length:rootsSize
                                                           options:MTLResourceStorageModeShared];
        
        // FFT is performed in log2(n) layers
        uint32_t logN = 0;
        size_t temp = n;
        while (temp > 1) {
            temp >>= 1;
            logN++;
        }
        
        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [ctx->commandQueue commandBuffer];
        
        // Execute FFT layers
        for (uint32_t layer = 0; layer < logN; layer++) {
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            
            [encoder setComputePipelineState:ctx->fftRadix2Pipeline];
            [encoder setBuffer:bufferData offset:0 atIndex:0];
            [encoder setBuffer:bufferRoots offset:0 atIndex:1];
            [encoder setBytes:&layer length:sizeof(uint32_t) atIndex:2];
            [encoder setBytes:&n length:sizeof(uint32_t) atIndex:3];
            
            NSUInteger threadsNeeded = n / 2;
            NSUInteger threadGroupSize = ctx->fftRadix2Pipeline.maxTotalThreadsPerThreadgroup;
            if (threadGroupSize > threadsNeeded) threadGroupSize = threadsNeeded;
            
            MTLSize threadsPerThreadgroup = MTLSizeMake(threadGroupSize, 1, 1);
            MTLSize numThreadgroups = MTLSizeMake((threadsNeeded + threadGroupSize - 1) / threadGroupSize, 1, 1);
            
            [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Copy result back
        memcpy(data, bufferData.contents, dataSize);
        
        // If inverse FFT, divide by n
        if (inverse) {
            // This would need proper field division implementation
            // For now, just a placeholder
        }
        
        return 0;
    }
}

double metal_benchmark_fft(metal_kzg_handle handle, size_t size, int iterations) {
    MetalKZGContext* ctx = (MetalKZGContext*)handle;
    if (!ctx) return -1.0;
    
    // Allocate test data
    gpu_field_element_t* data = new gpu_field_element_t[size];
    gpu_field_element_t* roots = new gpu_field_element_t[size];
    
    // Initialize with dummy data
    for (size_t i = 0; i < size; i++) {
        for (int j = 0; j < 6; j++) {
            data[i].limbs[j] = i + j;
            roots[i].limbs[j] = (i * 7 + j * 13) % UINT64_MAX;
        }
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        metal_fft_fr(handle, data, roots, size, false);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    
    delete[] data;
    delete[] roots;
    
    return elapsed.count() / iterations;
}

double metal_benchmark_field_mul(metal_kzg_handle handle, size_t count, int iterations) {
    MetalKZGContext* ctx = (MetalKZGContext*)handle;
    if (!ctx) return -1.0;
    
    // Allocate test data
    gpu_field_element_t* a = new gpu_field_element_t[count];
    gpu_field_element_t* b = new gpu_field_element_t[count];
    gpu_field_element_t* result = new gpu_field_element_t[count];
    
    // Initialize with dummy data
    for (size_t i = 0; i < count; i++) {
        for (int j = 0; j < 6; j++) {
            a[i].limbs[j] = i + j;
            b[i].limbs[j] = (i * 7 + j * 13) % UINT64_MAX;
        }
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        metal_field_mul_batch(handle, a, b, result, count);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    
    delete[] a;
    delete[] b;
    delete[] result;
    
    return elapsed.count() / iterations;
}

// Conversion functions (simplified - would need proper implementation)
void fr_to_gpu_element(const void* fr, gpu_field_element_t* gpu_elem) {
    // Assuming fr_t is similar structure with 6 limbs
    memcpy(gpu_elem->limbs, fr, sizeof(gpu_elem->limbs));
}

void gpu_element_to_fr(const gpu_field_element_t* gpu_elem, void* fr) {
    memcpy(fr, gpu_elem->limbs, sizeof(gpu_elem->limbs));
}