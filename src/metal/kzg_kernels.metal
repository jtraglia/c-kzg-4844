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

#include <metal_stdlib>
using namespace metal;

////////////////////////////////////////////////////////////////////////////////////////////////////
// BLS12-381 Field Constants
////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * BLS12-381 scalar field modulus (r):
 * r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
 *
 * In 4x64-bit limbs (little-endian):
 * r[0] = 0xffffffff00000001
 * r[1] = 0x53bda402fffe5bfe
 * r[2] = 0x3339d80809a1d805
 * r[3] = 0x73eda753299d7d48
 */

// BLS12-381 scalar field modulus in 4x64-bit limbs
constant ulong4 MODULUS = ulong4(
    0xffffffff00000001UL,
    0x53bda402fffe5bfeUL,
    0x3339d80809a1d805UL,
    0x73eda753299d7d48UL
);

// Montgomery R^2 mod p for converting to Montgomery form
// R = 2^256, R^2 mod r
constant ulong4 R2 = ulong4(
    0xc999e990f3f29c6dUL,
    0x2b6cedcb87925c23UL,
    0x05d314967254398fUL,
    0x0748d9d99f59ff11UL
);

// Montgomery constant: -r^(-1) mod 2^64
constant ulong INV = 0xfffffffeffffffffUL;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Field Element Type (256-bit in Montgomery form)
////////////////////////////////////////////////////////////////////////////////////////////////////

// A field element is represented as 4x64-bit limbs in little-endian order
// We use an array of 4 ulongs
struct Fr {
    ulong l[4];
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Low-level Arithmetic Helpers
////////////////////////////////////////////////////////////////////////////////////////////////////

// Add two 64-bit values with carry-in, return result and carry-out
inline ulong adc(ulong a, ulong b, thread ulong &carry) {
    // Use 128-bit arithmetic via __uint128_t equivalent
    ulong sum = a + b + carry;
    carry = ((sum < a) || (carry && sum == a)) ? 1UL : 0UL;
    return sum;
}

// Multiply two 64-bit values, return low 64 bits, store high 64 bits in hi
inline ulong mul64(ulong a, ulong b, thread ulong &hi) {
    // Metal doesn't have native 128-bit arithmetic, so we use 32-bit parts
    ulong a_lo = a & 0xFFFFFFFFUL;
    ulong a_hi = a >> 32;
    ulong b_lo = b & 0xFFFFFFFFUL;
    ulong b_hi = b >> 32;

    ulong p0 = a_lo * b_lo;
    ulong p1 = a_lo * b_hi;
    ulong p2 = a_hi * b_lo;
    ulong p3 = a_hi * b_hi;

    ulong mid = p1 + p2;
    ulong mid_carry = (mid < p1) ? 1UL : 0UL;

    ulong lo = p0 + (mid << 32);
    ulong lo_carry = (lo < p0) ? 1UL : 0UL;

    hi = p3 + (mid >> 32) + (mid_carry << 32) + lo_carry;

    return lo;
}

// Multiply-add: result = a * b + c, return low 64 bits, add high to hi
inline ulong mac(ulong a, ulong b, ulong c, thread ulong &hi) {
    ulong prod_hi;
    ulong prod_lo = mul64(a, b, prod_hi);

    ulong sum = prod_lo + c;
    ulong sum_carry = (sum < prod_lo) ? 1UL : 0UL;

    hi += prod_hi + sum_carry;
    return sum;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Field Arithmetic Operations
////////////////////////////////////////////////////////////////////////////////////////////////////

// Subtract modulus if result >= modulus (conditional subtraction)
inline void fr_reduce(thread Fr &a) {
    // Check if a >= MODULUS
    bool ge = false;
    if (a.l[3] > MODULUS.w) ge = true;
    else if (a.l[3] == MODULUS.w) {
        if (a.l[2] > MODULUS.z) ge = true;
        else if (a.l[2] == MODULUS.z) {
            if (a.l[1] > MODULUS.y) ge = true;
            else if (a.l[1] == MODULUS.y) {
                if (a.l[0] >= MODULUS.x) ge = true;
            }
        }
    }

    if (ge) {
        ulong borrow = 0;
        ulong diff;

        diff = a.l[0] - MODULUS.x - borrow;
        borrow = (a.l[0] < MODULUS.x + borrow) ? 1UL : 0UL;
        a.l[0] = diff;

        diff = a.l[1] - MODULUS.y - borrow;
        borrow = (a.l[1] < MODULUS.y + borrow) ? 1UL : 0UL;
        a.l[1] = diff;

        diff = a.l[2] - MODULUS.z - borrow;
        borrow = (a.l[2] < MODULUS.z + borrow) ? 1UL : 0UL;
        a.l[2] = diff;

        a.l[3] = a.l[3] - MODULUS.w - borrow;
    }
}

// Field addition: out = a + b mod p
inline Fr fr_add(Fr a, Fr b) {
    Fr result;
    ulong carry = 0;

    result.l[0] = adc(a.l[0], b.l[0], carry);
    result.l[1] = adc(a.l[1], b.l[1], carry);
    result.l[2] = adc(a.l[2], b.l[2], carry);
    result.l[3] = adc(a.l[3], b.l[3], carry);

    // Reduce if necessary
    fr_reduce(result);

    return result;
}

// Field subtraction: out = a - b mod p
inline Fr fr_sub(Fr a, Fr b) {
    Fr result;
    ulong borrow = 0;

    // Compute a - b
    ulong diff0 = a.l[0] - b.l[0];
    borrow = (a.l[0] < b.l[0]) ? 1UL : 0UL;

    ulong diff1 = a.l[1] - b.l[1] - borrow;
    borrow = (a.l[1] < b.l[1] + borrow) ? 1UL : 0UL;

    ulong diff2 = a.l[2] - b.l[2] - borrow;
    borrow = (a.l[2] < b.l[2] + borrow) ? 1UL : 0UL;

    ulong diff3 = a.l[3] - b.l[3] - borrow;
    borrow = (a.l[3] < b.l[3] + borrow) ? 1UL : 0UL;

    result.l[0] = diff0;
    result.l[1] = diff1;
    result.l[2] = diff2;
    result.l[3] = diff3;

    // If borrow occurred, add modulus back
    if (borrow) {
        ulong carry = 0;
        result.l[0] = adc(result.l[0], MODULUS.x, carry);
        result.l[1] = adc(result.l[1], MODULUS.y, carry);
        result.l[2] = adc(result.l[2], MODULUS.z, carry);
        result.l[3] = adc(result.l[3], MODULUS.w, carry);
    }

    return result;
}

// Montgomery multiplication: out = a * b * R^(-1) mod p
// Using CIOS (Coarsely Integrated Operand Scanning) method
inline Fr fr_mul(Fr a, Fr b) {
    // Temporary storage for intermediate results (8 limbs)
    ulong t[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // Multiply and accumulate
    for (int i = 0; i < 4; i++) {
        ulong carry = 0;

        // Multiply a * b[i] and add to t
        for (int j = 0; j < 4; j++) {
            ulong hi = 0;
            ulong lo = mac(a.l[j], b.l[i], t[i + j], hi);
            lo += carry;
            if (lo < carry) hi++;
            t[i + j] = lo;
            carry = hi;
        }
        t[i + 4] += carry;

        // Montgomery reduction step
        ulong m = t[i] * INV;
        carry = 0;

        // Add m * MODULUS to t, shifted by i limbs
        ulong hi = 0;
        ulong lo = mac(m, MODULUS.x, t[i], hi);
        carry = hi;

        hi = 0;
        lo = mac(m, MODULUS.y, t[i + 1], hi);
        lo += carry;
        if (lo < carry) hi++;
        t[i + 1] = lo;
        carry = hi;

        hi = 0;
        lo = mac(m, MODULUS.z, t[i + 2], hi);
        lo += carry;
        if (lo < carry) hi++;
        t[i + 2] = lo;
        carry = hi;

        hi = 0;
        lo = mac(m, MODULUS.w, t[i + 3], hi);
        lo += carry;
        if (lo < carry) hi++;
        t[i + 3] = lo;
        carry = hi;

        t[i + 4] += carry;
        if (t[i + 4] < carry && i + 5 < 8) {
            t[i + 5]++;
        }
    }

    // Result is in t[4..7]
    Fr result;
    result.l[0] = t[4];
    result.l[1] = t[5];
    result.l[2] = t[6];
    result.l[3] = t[7];

    // Final reduction
    fr_reduce(result);

    return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// FFT Kernels
////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Cooley-Tukey FFT butterfly operation.
 *
 * For each pair (a, b) at distance `half_n` apart:
 *   a' = a + w * b
 *   b' = a - w * b
 *
 * where w is the twiddle factor (root of unity).
 */
kernel void fft_butterfly(
    device Fr *data [[buffer(0)]],
    constant Fr *roots [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    constant uint &half_n [[buffer(3)]],
    constant uint &roots_stride [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    // Each thread handles one butterfly
    uint block_size = half_n * 2;
    uint block_idx = gid / half_n;
    uint idx_in_block = gid % half_n;

    if (block_idx * block_size + idx_in_block + half_n >= n) return;

    uint i = block_idx * block_size + idx_in_block;
    uint j = i + half_n;

    // Get twiddle factor
    uint root_idx = idx_in_block * roots_stride;
    Fr w = roots[root_idx];

    // Load values
    Fr a = data[i];
    Fr b = data[j];

    // Compute w * b
    Fr wb = fr_mul(w, b);

    // Compute butterfly
    data[i] = fr_add(a, wb);
    data[j] = fr_sub(a, wb);
}

/*
 * Bit-reversal permutation kernel.
 * Swaps elements at positions i and bit_reverse(i).
 */
kernel void bit_reversal(
    device Fr *data [[buffer(0)]],
    constant uint &n [[buffer(1)]],
    constant uint &log_n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    // Compute bit-reversed index
    uint rev = 0;
    uint val = gid;
    for (uint i = 0; i < log_n; i++) {
        rev = (rev << 1) | (val & 1);
        val >>= 1;
    }

    // Only swap if gid < rev to avoid double-swapping
    if (gid < rev && rev < n) {
        Fr temp = data[gid];
        data[gid] = data[rev];
        data[rev] = temp;
    }
}

/*
 * Scale all elements by a constant factor (for inverse FFT).
 */
kernel void scale_elements(
    device Fr *data [[buffer(0)]],
    constant Fr &scale [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = fr_mul(data[gid], scale);
}

/*
 * Combined FFT butterfly for multiple stages.
 * More efficient by reducing kernel launch overhead.
 */
kernel void fft_radix2_stage(
    device Fr *data [[buffer(0)]],
    constant Fr *roots [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    constant uint &stage [[buffer(3)]],
    constant uint &roots_stride_base [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint half_n = 1u << stage;
    uint block_size = half_n * 2;
    uint num_butterflies = n / 2;

    if (gid >= num_butterflies) return;

    uint block_idx = gid / half_n;
    uint idx_in_block = gid % half_n;

    uint i = block_idx * block_size + idx_in_block;
    uint j = i + half_n;

    // Calculate roots stride for this stage
    uint roots_stride = roots_stride_base >> stage;
    uint root_idx = idx_in_block * roots_stride;

    Fr w = roots[root_idx];
    Fr a = data[i];
    Fr b = data[j];

    Fr wb = fr_mul(w, b);

    data[i] = fr_add(a, wb);
    data[j] = fr_sub(a, wb);
}

/*
 * Copy kernel for data transfer.
 */
kernel void copy_elements(
    device Fr *dst [[buffer(0)]],
    constant Fr *src [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    dst[gid] = src[gid];
}
