#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <simd/simd.h>
#include "MetalKZGAccelerator.h"
#include <chrono>
#include <string.h>

// Embedded Metal shader source
static const char* metalShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

// Simple field element structure
struct FieldElement {
    ulong4 low;
    ulong2 high;
};

// Simple field addition
kernel void field_add_batch(
    device const FieldElement* a [[buffer(0)]],
    device const FieldElement* b [[buffer(1)]],
    device FieldElement* result [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    FieldElement sum;
    sum.low = a[index].low + b[index].low;
    sum.high = a[index].high + b[index].high;
    result[index] = sum;
}

// Simple field multiplication (placeholder)
kernel void field_mul_batch(
    device const FieldElement* a [[buffer(0)]],
    device const FieldElement* b [[buffer(1)]],
    device FieldElement* result [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    // Simplified multiplication - just for testing
    FieldElement prod;
    prod.low = a[index].low * b[index].low.x;
    prod.high = a[index].high * b[index].high.x;
    result[index] = prod;
}

// Simple FFT radix-2 layer
kernel void fft_radix2_layer(
    device FieldElement* data [[buffer(0)]],
    device const FieldElement* roots [[buffer(1)]],
    constant uint& layer [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint stride = 1 << layer;
    uint pair_idx = tid / stride;
    uint idx_in_pair = tid % stride;
    
    uint i = pair_idx * stride * 2 + idx_in_pair;
    uint j = i + stride;
    
    if (j < n) {
        FieldElement a = data[i];
        FieldElement b = data[j];
        
        // Simplified butterfly (no proper multiplication with twiddle)
        FieldElement sum, diff;
        sum.low = a.low + b.low;
        sum.high = a.high + b.high;
        diff.low = a.low - b.low;
        diff.high = a.high - b.high;
        
        data[i] = sum;
        data[j] = diff;
    }
}
)";

struct MetalKZGContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;
    
    // Compute pipelines
    id<MTLComputePipelineState> fieldAddPipeline;
    id<MTLComputePipelineState> fieldMulPipeline;
    id<MTLComputePipelineState> fftRadix2Pipeline;
};

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
        
        // Compile the embedded shader source
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:metalShaderSource];
        
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        ctx->library = [ctx->device newLibraryWithSource:source options:options error:&error];
        
        if (!ctx->library) {
            NSLog(@"Failed to compile Metal library: %@", error);
            delete ctx;
            return nullptr;
        }
        
        // Create compute pipelines
        id<MTLFunction> function;
        
        // Field add pipeline
        function = [ctx->library newFunctionWithName:@"field_add_batch"];
        if (function) {
            ctx->fieldAddPipeline = [ctx->device newComputePipelineStateWithFunction:function error:&error];
            if (!ctx->fieldAddPipeline) {
                NSLog(@"Failed to create field add pipeline: %@", error);
            }
        }
        
        // Field multiply pipeline
        function = [ctx->library newFunctionWithName:@"field_mul_batch"];
        if (function) {
            ctx->fieldMulPipeline = [ctx->device newComputePipelineStateWithFunction:function error:&error];
            if (!ctx->fieldMulPipeline) {
                NSLog(@"Failed to create field multiply pipeline: %@", error);
            }
        }
        
        // FFT pipeline
        function = [ctx->library newFunctionWithName:@"fft_radix2_layer"];
        if (function) {
            ctx->fftRadix2Pipeline = [ctx->device newComputePipelineStateWithFunction:function error:&error];
            if (!ctx->fftRadix2Pipeline) {
                NSLog(@"Failed to create FFT pipeline: %@", error);
            }
        }
        
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
        
        if (!bufferA || !bufferB || !bufferResult) return -1;
        
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
        
        if (!bufferA || !bufferB || !bufferResult) return -1;
        
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
        
        if (!bufferData || !bufferRoots) return -1;
        
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
            
            uint32_t n32 = (uint32_t)n;
            [encoder setBytes:&n32 length:sizeof(uint32_t) atIndex:3];
            
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

// Conversion functions (simplified)
void fr_to_gpu_element(const void* fr, gpu_field_element_t* gpu_elem) {
    memcpy(gpu_elem->limbs, fr, sizeof(gpu_elem->limbs));
}

void gpu_element_to_fr(const gpu_field_element_t* gpu_elem, void* fr) {
    memcpy(fr, gpu_elem->limbs, sizeof(gpu_elem->limbs));
}