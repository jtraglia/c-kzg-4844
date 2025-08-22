# Metal GPU Acceleration for c-kzg-4844 on Apple M1

This is a proof of concept demonstrating how to accelerate KZG computations using the Apple M1's GPU via Metal.

## Overview

The implementation focuses on accelerating the most compute-intensive operations in `ComputeCellsAndKZGProofs`:

1. **FFT operations** - Parallel butterfly computations on the GPU
2. **Field arithmetic** - Batch Montgomery multiplication in BLS12-381 field
3. **Multi-scalar multiplication (MSM)** - Parallel point-scalar multiplications

## Architecture

```
┌─────────────────────────────────────┐
│         c-kzg-4844 Library          │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│    kzg_metal_integration.c          │
│  (Integration layer with fallback)  │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│    MetalKZGAccelerator.mm           │
│   (Objective-C++ Metal wrapper)     │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│    bls12_381_field.metal            │
│      (GPU compute kernels)          │
└─────────────────────────────────────┘
```

## Building

```bash
# Build the benchmark
make all

# Or build and run immediately
make run
```

## Running the Benchmark

```bash
./benchmark_metal
```

This will:
1. Compare CPU vs GPU performance for `ComputeCellsAndKZGProofs`
2. Show FFT performance for various sizes
3. Benchmark field multiplication operations

## Expected Performance

On an M1 Mac Mini with precompute=8:
- **CPU baseline**: ~180ms
- **GPU target**: 50-100ms (2-3x speedup)
- **FFT operations**: 5-10x speedup for large sizes
- **Field multiplication**: 3-5x speedup with batching

## Implementation Notes

### What's Implemented
- Basic BLS12-381 field arithmetic (simplified Montgomery multiplication)
- Radix-2 FFT butterfly operations
- Batch field operations (add, multiply)
- Integration with c-kzg-4844's compute pipeline
- Benchmark suite

### What's Simplified (for POC)
- Montgomery reduction is simplified (full implementation needed for production)
- MSM uses placeholder elliptic curve operations
- Error handling is minimal
- Memory management could be optimized

## Optimization Opportunities

### 1. Complete Montgomery Implementation
The current Montgomery multiplication is simplified. A full implementation would:
- Use proper CIOS (Coarsely Integrated Operand Scanning) method
- Implement complete modular reduction
- Handle all edge cases

### 2. Elliptic Curve Operations
For production use, implement:
- Proper G1/G2 point addition and doubling
- Jacobian coordinate system for efficiency
- Window-based scalar multiplication

### 3. Memory Optimization
- Use Metal's shared memory for better cache utilization
- Implement memory pooling to reduce allocation overhead
- Use page-locked memory for faster CPU-GPU transfers

### 4. Advanced GPU Features
- Utilize M1's matrix multiplication units (AMX)
- Implement mixed-radix FFT for non-power-of-2 sizes
- Use Metal Performance Shaders where applicable

### 5. FK20 Algorithm Acceleration
The FK20 proof generation could be accelerated by:
- Parallelizing the MSM operations across columns
- Using precomputed tables in GPU constant memory
- Batching multiple polynomial evaluations

## Integration with Go

To integrate with the Go bindings:

```go
// #cgo CFLAGS: -I./metal_poc
// #cgo LDFLAGS: -L./metal_poc -lmetal_kzg -framework Metal -framework Foundation
// #include "MetalKZGAccelerator.h"
import "C"

func ComputeCellsAndProofsGPU(blob *Blob) (*[CellsPerExtBlob]Cell, *[CellsPerExtBlob]KZGProof, error) {
    // Initialize Metal acceleration
    handle := C.metal_kzg_init()
    defer C.metal_kzg_cleanup(handle)
    
    // Call GPU-accelerated function
    // ...
}
```

## Troubleshooting

### Metal Not Available
If you see "Metal is not supported on this device":
- Ensure you're running on an Apple Silicon Mac
- Check that Xcode command line tools are installed: `xcode-select --install`

### Performance Not Improved
If GPU is slower than CPU:
- The POC has overhead from data transfer and simplified algorithms
- Small workloads may not benefit from GPU acceleration
- Ensure you're using Release build (`-O3` optimization)

### Build Errors
- Make sure you're in the correct directory
- Verify all paths in the Makefile are correct
- Check that the blst library is built: `cd ../../../blst && ./build.sh`

## Next Steps

1. **Complete field arithmetic implementation** - Full Montgomery multiplication
2. **Implement proper EC operations** - For accurate MSM benchmarks
3. **Optimize memory transfers** - Use unified memory architecture effectively
4. **Profile with Instruments** - Identify remaining bottlenecks
5. **Production hardening** - Error handling, edge cases, testing

## References

- [Metal Programming Guide](https://developer.apple.com/metal/)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [BLS12-381 For The Rest Of Us](https://hackmd.io/@benjaminion/bls12-381)
- [FK20 Proofs](https://eprint.iacr.org/2023/033.pdf)
- [c-kzg-4844 Repository](https://github.com/ethereum/c-kzg-4844)