/*
 * Copyright 2021 Benjamin Edgington
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

/**
 * @file c_kzg_4844.c
 *
 * Minimal implementation of the polynomial commitments API for EIP-4844.
 */
#include "c_kzg_4844.h"

#include <assert.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

///////////////////////////////////////////////////////////////////////////////
// Macros
///////////////////////////////////////////////////////////////////////////////

/** Returns C_KZG_BADARGS if the condition is not met. */
#define CHECK(cond) \
    if (!(cond)) return C_KZG_BADARGS

/** Returns number of elements in a statically defined array. */
#define NUM_ELEMENTS(a) (sizeof(a) / sizeof(a[0]))

/**
 * Helper macro to release memory allocated on the heap. Unlike free(),
 * c_kzg_free() macro sets the pointer value to NULL after freeing it.
 */
#define c_kzg_free(p) \
    do { \
        free(p); \
        (p) = NULL; \
    } while (0)

///////////////////////////////////////////////////////////////////////////////
// Types
///////////////////////////////////////////////////////////////////////////////

/** Internal representation of a polynomial. */
typedef struct {
    fr_t evals[FIELD_ELEMENTS_PER_BLOB];
} Polynomial;

///////////////////////////////////////////////////////////////////////////////
// Constants
///////////////////////////////////////////////////////////////////////////////

/** The domain separator for the Fiat-Shamir protocol. */
static const char *FIAT_SHAMIR_PROTOCOL_DOMAIN = "FSBLOBVERIFY_V1_";

/** The domain separator for a random challenge. */
static const char *RANDOM_CHALLENGE_KZG_BATCH_DOMAIN = "RCKZGBATCH___V1_";

/** Length of the domain strings above. */
#define DOMAIN_STR_LENGTH 16

/** The number of bytes in a g1 point. */
#define BYTES_PER_G1 48

/** The number of bytes in a g2 point. */
#define BYTES_PER_G2 96

/** The number of g1 points in a trusted setup. */
#define TRUSTED_SETUP_NUM_G1_POINTS FIELD_ELEMENTS_PER_BLOB

/** The number of g2 points in a trusted setup. */
#define TRUSTED_SETUP_NUM_G2_POINTS 65

// clang-format off

/** Deserialized form of the G1 identity/infinity point. */
static const g1_t G1_IDENTITY = {
    {0L, 0L, 0L, 0L, 0L, 0L},
    {0L, 0L, 0L, 0L, 0L, 0L},
    {0L, 0L, 0L, 0L, 0L, 0L}};

/** The G1 generator. */
static const g1_t G1_GENERATOR = {
    {0x5cb38790fd530c16L, 0x7817fc679976fff5L, 0x154f95c7143ba1c1L,
     0xf0ae6acdf3d0e747L, 0xedce6ecc21dbf440L, 0x120177419e0bfb75L},
    {0xbaac93d50ce72271L, 0x8c22631a7918fd8eL, 0xdd595f13570725ceL,
     0x51ac582950405194L, 0x0e1c8c3fad0059c0L, 0x0bbc3efc5008a26aL},
    {0x760900000002fffdL, 0xebf4000bc40c0002L, 0x5f48985753c758baL,
     0x77ce585370525745L, 0x5c071a97a256ec6dL, 0x15f65ec3fa80e493L}};

/** The G2 generator. */
static const g2_t G2_GENERATOR = {
    {{{0xf5f28fa202940a10L, 0xb3f5fb2687b4961aL, 0xa1a893b53e2ae580L,
       0x9894999d1a3caee9L, 0x6f67b7631863366bL, 0x058191924350bcd7L},
      {0xa5a9c0759e23f606L, 0xaaa0c59dbccd60c3L, 0x3bb17e18e2867806L,
       0x1b1ab6cc8541b367L, 0xc2b6ed0ef2158547L, 0x11922a097360edf3L}}},
    {{{0x4c730af860494c4aL, 0x597cfa1f5e369c5aL, 0xe7e6856caa0a635aL,
       0xbbefb5e96e0d495fL, 0x07d3a975f0ef25a2L, 0x0083fd8e7e80dae5L},
      {0xadc0fc92df64b05dL, 0x18aa270a2b1461dcL, 0x86adac6a3be4eba0L,
       0x79495c4ec93da33aL, 0xe7175850a43ccaedL, 0x0b2bc2a163de1bf2L}}},
    {{{0x760900000002fffdL, 0xebf4000bc40c0002L, 0x5f48985753c758baL,
       0x77ce585370525745L, 0x5c071a97a256ec6dL, 0x15f65ec3fa80e493L},
      {0x0000000000000000L, 0x0000000000000000L, 0x0000000000000000L,
       0x0000000000000000L, 0x0000000000000000L, 0x0000000000000000L}}}};

/** The zero field element. */
static const fr_t FR_ZERO = {0L, 0L, 0L, 0L};

/** This is 1 in Blst's `blst_fr` limb representation. Crazy but true. */
static const fr_t FR_ONE = {
    0x00000001fffffffeL, 0x5884b7fa00034802L,
    0x998c4fefecbc4ff5L, 0x1824b159acc5056fL};

// clang-format on

///////////////////////////////////////////////////////////////////////////////
// Memory Allocation Functions
///////////////////////////////////////////////////////////////////////////////

/**
 * Wrapped malloc() that reports failures to allocate.
 *
 * @remark Will return C_KZG_BADARGS if the requested size is zero.
 *
 * @param[out] out  Pointer to the allocated space
 * @param[in]  size The number of bytes to be allocated
 */
static C_KZG_RET c_kzg_malloc(void **out, size_t size) {
    *out = NULL;
    if (size == 0) return C_KZG_BADARGS;
    *out = malloc(size);
    return *out != NULL ? C_KZG_OK : C_KZG_MALLOC;
}

/**
 * Wrapped calloc() that reports failures to allocate.
 *
 * @remark Will return C_KZG_BADARGS if the requested size is zero.
 *
 * @param[out] out   Pointer to the allocated space
 * @param[in]  count The number of elements
 * @param[in]  size  The size of each element
 */
static C_KZG_RET c_kzg_calloc(void **out, size_t count, size_t size) {
    *out = NULL;
    if (count == 0 || size == 0) return C_KZG_BADARGS;
    *out = calloc(count, size);
    return *out != NULL ? C_KZG_OK : C_KZG_MALLOC;
}

/**
 * Allocate memory for an array of G1 group elements.
 *
 * @remark Free the space later using c_kzg_free().
 *
 * @param[out] x Pointer to the allocated space
 * @param[in]  n The number of G1 elements to be allocated
 */
static C_KZG_RET new_g1_array(g1_t **x, size_t n) {
    return c_kzg_calloc((void **)x, n, sizeof(g1_t));
}

/**
 * Allocate memory for an array of G2 group elements.
 *
 * @remark Free the space later using c_kzg_free().
 *
 * @param[out] x Pointer to the allocated space
 * @param[in]  n The number of G2 elements to be allocated
 */
static C_KZG_RET new_g2_array(g2_t **x, size_t n) {
    return c_kzg_calloc((void **)x, n, sizeof(g2_t));
}

/**
 * Allocate memory for an array of field elements.
 *
 * @remark Free the space later using c_kzg_free().
 *
 * @param[out] x Pointer to the allocated space
 * @param[in]  n The number of field elements to be allocated
 */
static C_KZG_RET new_fr_array(fr_t **x, size_t n) {
    return c_kzg_calloc((void **)x, n, sizeof(fr_t));
}

///////////////////////////////////////////////////////////////////////////////
// Helper Functions
///////////////////////////////////////////////////////////////////////////////

/**
 * Test whether the operand is zero in the finite field.
 *
 * @param[in] p The field element to be checked
 *
 * @retval true  The element is zero
 * @retval false The element is not zero
 */
static bool fr_is_zero(const fr_t *p) {
    uint64_t a[4];
    blst_uint64_from_fr(a, p);
    return a[0] == 0 && a[1] == 0 && a[2] == 0 && a[3] == 0;
}

/**
 * Test whether two field elements are equal.
 *
 * @param[in] aa The first element
 * @param[in] bb The second element
 *
 * @retval true  if @p aa and @p bb are equal
 * @retval false otherwise
 */
static bool fr_equal(const fr_t *aa, const fr_t *bb) {
    uint64_t a[4], b[4];
    blst_uint64_from_fr(a, aa);
    blst_uint64_from_fr(b, bb);
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
}

/**
 * Divide a field element by another.
 *
 * @remark The behaviour for @p b == 0 is unspecified.
 *
 * @remark This function does support in-place computation, i.e. @p out == @p a
 * or @p out == @p b work.
 *
 * @param[out] out @p a divided by @p b in the field
 * @param[in]  a   The dividend
 * @param[in]  b   The divisor
 */
static void fr_div(fr_t *out, const fr_t *a, const fr_t *b) {
    blst_fr tmp;
    blst_fr_eucl_inverse(&tmp, b);
    blst_fr_mul(out, a, &tmp);
}

/**
 * Exponentiation of a field element.
 *
 * Uses square and multiply for log(@p n) performance.
 *
 * @remark A 64-bit exponent is sufficient for our needs here.
 *
 * @remark This function does support in-place computation, i.e. @p a == @p out
 * works.
 *
 * @param[out] out @p a raised to the power of @p n
 * @param[in]  a   The field element to be exponentiated
 * @param[in]  n   The exponent
 */
static void fr_pow(fr_t *out, const fr_t *a, uint64_t n) {
    fr_t tmp = *a;
    *out = FR_ONE;

    while (true) {
        if (n & 1) {
            blst_fr_mul(out, out, &tmp);
        }
        if ((n >>= 1) == 0) break;
        blst_fr_sqr(&tmp, &tmp);
    }
}

/**
 * Create a field element from a single 64-bit unsigned integer.
 *
 * @remark This can only generate a tiny fraction of possible field elements,
 * and is mostly useful for testing.
 *
 * @param[out] out The field element equivalent of @p n
 * @param[in]  n   The 64-bit integer to be converted
 */
static void fr_from_uint64(fr_t *out, uint64_t n) {
    uint64_t vals[] = {n, 0, 0, 0};
    blst_fr_from_uint64(out, vals);
}

/**
 * Montgomery batch inversion in finite field.
 *
 * @remark Return C_KZG_BADARGS if a zero is found in the input. In this case,
 * the `out` output array has already been mutated.
 *
 * @remark This function does not support in-place computation (i.e. `a` MUST
 * NOT point to the same place as `out`)
 *
 * @remark This function only supports len > 0.
 *
 * @param[out] out The inverses of @p a, length @p len
 * @param[in]  a   A vector of field elements, length @p len
 * @param[in]  len The number of field elements
 */
static C_KZG_RET fr_batch_inv(fr_t *out, const fr_t *a, int len) {
    int i;

    assert(len > 0);
    assert(a != out);

    fr_t accumulator = FR_ONE;

    for (i = 0; i < len; i++) {
        out[i] = accumulator;
        blst_fr_mul(&accumulator, &accumulator, &a[i]);
    }

    /* Bail on any zero input */
    if (fr_is_zero(&accumulator)) {
        return C_KZG_BADARGS;
    }

    blst_fr_eucl_inverse(&accumulator, &accumulator);

    for (i = len - 1; i >= 0; i--) {
        blst_fr_mul(&out[i], &out[i], &accumulator);
        blst_fr_mul(&accumulator, &accumulator, &a[i]);
    }

    return C_KZG_OK;
}

/**
 * Multiply a G1 group element by a field element.
 *
 * @param[out] out @p a * @p b
 * @param[in]  a   The G1 group element
 * @param[in]  b   The multiplier
 */
static void g1_mul(g1_t *out, const g1_t *a, const fr_t *b) {
    blst_scalar s;
    blst_scalar_from_fr(&s, b);
    /* The last argument is the number of bits in the scalar */
    blst_p1_mult(out, a, s.b, 8 * sizeof(blst_scalar));
}

/**
 * Multiply a G2 group element by a field element.
 *
 * @param[out] out @p a * @p b
 * @param[in]  a   The G2 group element
 * @param[in]  b   The multiplier
 */
static void g2_mul(g2_t *out, const g2_t *a, const fr_t *b) {
    blst_scalar s;
    blst_scalar_from_fr(&s, b);
    /* The last argument is the number of bits in the scalar */
    blst_p2_mult(out, a, s.b, 8 * sizeof(blst_scalar));
}

/**
 * Subtraction of G1 group elements.
 *
 * @param[out] out @p a - @p b
 * @param[in]  a   A G1 group element
 * @param[in]  b   The G1 group element to be subtracted
 */
static void g1_sub(g1_t *out, const g1_t *a, const g1_t *b) {
    g1_t bneg = *b;
    blst_p1_cneg(&bneg, true);
    blst_p1_add_or_double(out, a, &bneg);
}

/**
 * Subtraction of G2 group elements.
 *
 * @param[out] out @p a - @p b
 * @param[in]  a   A G2 group element
 * @param[in]  b   The G2 group element to be subtracted
 */
static void g2_sub(g2_t *out, const g2_t *a, const g2_t *b) {
    g2_t bneg = *b;
    blst_p2_cneg(&bneg, true);
    blst_p2_add_or_double(out, a, &bneg);
}

/**
 * Perform pairings and test whether the outcomes are equal in G_T.
 *
 * Tests whether `e(a1, a2) == e(b1, b2)`.
 *
 * @param[in] a1 A G1 group point for the first pairing
 * @param[in] a2 A G2 group point for the first pairing
 * @param[in] b1 A G1 group point for the second pairing
 * @param[in] b2 A G2 group point for the second pairing
 *
 * @retval true  The pairings were equal
 * @retval false The pairings were not equal
 */
static bool pairings_verify(
    const g1_t *a1, const g2_t *a2, const g1_t *b1, const g2_t *b2
) {
    blst_fp12 loop0, loop1, gt_point;
    blst_p1_affine aa1, bb1;
    blst_p2_affine aa2, bb2;

    /*
     * As an optimisation, we want to invert one of the pairings,
     * so we negate one of the points.
     */
    g1_t a1neg = *a1;
    blst_p1_cneg(&a1neg, true);

    blst_p1_to_affine(&aa1, &a1neg);
    blst_p1_to_affine(&bb1, b1);
    blst_p2_to_affine(&aa2, a2);
    blst_p2_to_affine(&bb2, b2);

    blst_miller_loop(&loop0, &aa2, &aa1);
    blst_miller_loop(&loop1, &bb2, &bb1);

    blst_fp12_mul(&gt_point, &loop0, &loop1);
    blst_final_exp(&gt_point, &gt_point);

    return blst_fp12_is_one(&gt_point);
}

///////////////////////////////////////////////////////////////////////////////
// Bytes Conversion Helper Functions
///////////////////////////////////////////////////////////////////////////////

/**
 * Serialize a G1 group element into bytes.
 *
 * @param[out] out A 48-byte array to store the serialized G1 element
 * @param[in]  in  The G1 element to be serialized
 */
static void bytes_from_g1(Bytes48 *out, const g1_t *in) {
    blst_p1_compress(out->bytes, in);
}

/**
 * Serialize a BLS field element into bytes.
 *
 * @param[out] out A 32-byte array to store the serialized field element
 * @param[in] in The field element to be serialized
 */
static void bytes_from_bls_field(Bytes32 *out, const fr_t *in) {
    blst_scalar s;
    blst_scalar_from_fr(&s, in);
    blst_bendian_from_scalar(out->bytes, &s);
}

/**
 * Serialize a 64-bit unsigned integer into bytes.
 *
 * @remark The output format is big-endian.
 *
 * @param[out] out An 8-byte array to store the serialized integer
 * @param[in]  n   The integer to be serialized
 */
static void bytes_from_uint64(uint8_t out[8], uint64_t n) {
    for (int i = 7; i >= 0; i--) {
        out[i] = n & 0xFF;
        n >>= 8;
    }
}

///////////////////////////////////////////////////////////////////////////////
// BLS12-381 Helper Functions
///////////////////////////////////////////////////////////////////////////////

/**
 * Map bytes to a BLS field element.
 *
 * @param[out] out The field element to store the result
 * @param[in]  b   A 32-byte array containing the input
 */
static void hash_to_bls_field(fr_t *out, const Bytes32 *b) {
    blst_scalar tmp;
    blst_scalar_from_bendian(&tmp, b->bytes);
    blst_fr_from_scalar(out, &tmp);
}

/**
 * Convert untrusted bytes to a trusted and validated BLS scalar field
 * element.
 *
 * @param[out] out The field element to store the deserialized data
 * @param[in]  b   A 32-byte array containing the serialized field element
 */
static C_KZG_RET bytes_to_bls_field(fr_t *out, const Bytes32 *b) {
    blst_scalar tmp;
    blst_scalar_from_bendian(&tmp, b->bytes);
    if (!blst_scalar_fr_check(&tmp)) return C_KZG_BADARGS;
    blst_fr_from_scalar(out, &tmp);
    return C_KZG_OK;
}

/**
 * Perform BLS validation required by the types KZGProof and KZGCommitment.
 *
 * @remark This function deviates from the spec because it returns (via an
 *     output argument) the g1 point. This way is more efficient (faster)
 *     but the function name is a bit misleading.
 *
 * @param[out]  out The output g1 point
 * @param[in]   b   The proof/commitment bytes
 */
static C_KZG_RET validate_kzg_g1(g1_t *out, const Bytes48 *b) {
    blst_p1_affine p1_affine;

    /* Convert the bytes to a p1 point */
    /* The uncompress routine checks that the point is on the curve */
    if (blst_p1_uncompress(&p1_affine, b->bytes) != BLST_SUCCESS)
        return C_KZG_BADARGS;
    blst_p1_from_affine(out, &p1_affine);

    /* The point at infinity is accepted! */
    if (blst_p1_is_inf(out)) return C_KZG_OK;
    /* The point must be on the right subgroup */
    if (!blst_p1_in_g1(out)) return C_KZG_BADARGS;

    return C_KZG_OK;
}

/**
 * Convert untrusted bytes into a trusted and validated KZGCommitment.
 *
 * @param[out]  out The output commitment
 * @param[in]   b   The commitment bytes
 */
static C_KZG_RET bytes_to_kzg_commitment(g1_t *out, const Bytes48 *b) {
    return validate_kzg_g1(out, b);
}

/**
 * Convert untrusted bytes into a trusted and validated KZGProof.
 *
 * @param[out]  out The output proof
 * @param[in]   b   The proof bytes
 */
static C_KZG_RET bytes_to_kzg_proof(g1_t *out, const Bytes48 *b) {
    return validate_kzg_g1(out, b);
}

/**
 * Deserialize a Blob (array of bytes) into a Polynomial (array of field
 * elements).
 *
 * @param[out] p    The output polynomial (array of field elements)
 * @param[in]  blob The blob (an array of bytes)
 */
static C_KZG_RET blob_to_polynomial(Polynomial *p, const Blob *blob) {
    C_KZG_RET ret;
    for (size_t i = 0; i < FIELD_ELEMENTS_PER_BLOB; i++) {
        ret = bytes_to_bls_field(
            &p->evals[i], (Bytes32 *)&blob->bytes[i * BYTES_PER_FIELD_ELEMENT]
        );
        if (ret != C_KZG_OK) return ret;
    }
    return C_KZG_OK;
}

/* Input size to the Fiat-Shamir challenge computation. */
#define CHALLENGE_INPUT_SIZE \
    (DOMAIN_STR_LENGTH + 16 + BYTES_PER_BLOB + BYTES_PER_COMMITMENT)

/**
 * Return the Fiat-Shamir challenge required to verify `blob` and
 * `commitment`.
 *
 * @remark This function should compute challenges even if `n==0`.
 *
 * @param[out] eval_challenge_out The evaluation challenge
 * @param[in]  blob               A blob
 * @param[in]  commitment         A commitment
 */
static void compute_challenge(
    fr_t *eval_challenge_out, const Blob *blob, const g1_t *commitment
) {
    Bytes32 eval_challenge;
    uint8_t bytes[CHALLENGE_INPUT_SIZE];

    /* Pointer tracking `bytes` for writing on top of it */
    uint8_t *offset = bytes;

    /* Copy domain separator */
    memcpy(offset, FIAT_SHAMIR_PROTOCOL_DOMAIN, DOMAIN_STR_LENGTH);
    offset += DOMAIN_STR_LENGTH;

    /* Copy polynomial degree (16-bytes, big-endian) */
    bytes_from_uint64(offset, 0);
    offset += sizeof(uint64_t);
    bytes_from_uint64(offset, FIELD_ELEMENTS_PER_BLOB);
    offset += sizeof(uint64_t);

    /* Copy blob */
    memcpy(offset, blob->bytes, BYTES_PER_BLOB);
    offset += BYTES_PER_BLOB;

    /* Copy commitment */
    bytes_from_g1((Bytes48 *)offset, commitment);
    offset += BYTES_PER_COMMITMENT;

    /* Make sure we wrote the entire buffer */
    assert(offset == bytes + CHALLENGE_INPUT_SIZE);

    /* Now let's create the challenge! */
    blst_sha256(eval_challenge.bytes, bytes, CHALLENGE_INPUT_SIZE);
    hash_to_bls_field(eval_challenge_out, &eval_challenge);
}

/**
 * Calculate a linear combination of G1 group elements.
 *
 * Calculates `[coeffs_0]p_0 + [coeffs_1]p_1 + ... + [coeffs_n]p_n`
 * where `n` is `len - 1`.
 *
 * This function computes the result naively without using Pippenger's
 * algorithm.
 */
static void g1_lincomb_naive(
    g1_t *out, const g1_t *p, const fr_t *coeffs, uint64_t len
) {
    g1_t tmp;
    *out = G1_IDENTITY;
    for (uint64_t i = 0; i < len; i++) {
        g1_mul(&tmp, &p[i], &coeffs[i]);
        blst_p1_add_or_double(out, out, &tmp);
    }
}

/**
 * Calculate a linear combination of G1 group elements.
 *
 * Calculates `[coeffs_0]p_0 + [coeffs_1]p_1 + ... + [coeffs_n]p_n`
 * where `n` is `len - 1`.
 *
 * @remark This function MUST NOT be called with the point at infinity in `p`.
 *
 * @remark While this function is significantly faster than
 * `g1_lincomb_naive()`, we refrain from using it in security-critical places
 * (like verification) because the blst Pippenger code has not been
 * audited. In those critical places, we prefer using `g1_lincomb_naive()` which
 * is much simpler.
 *
 * @param[out] out    The resulting sum-product
 * @param[in]  p      Array of G1 group elements, length @p len
 * @param[in]  coeffs Array of field elements, length @p len
 * @param[in]  len    The number of group/field elements
 *
 * For the benefit of future generations (since Blst has no documentation to
 * speak of), there are two ways to pass the arrays of scalars and points
 * into blst_p1s_mult_pippenger().
 *
 * 1. Pass `points` as an array of pointers to the points, and pass
 *    `scalars` as an array of pointers to the scalars, each of length @p len.
 * 2. Pass an array where the first element is a pointer to the contiguous
 *    array of points and the second is null, and similarly for scalars.
 *
 * We do the second of these to save memory here.
 */
static C_KZG_RET g1_lincomb_fast(
    g1_t *out, const g1_t *p, const fr_t *coeffs, uint64_t len
) {
    C_KZG_RET ret;
    void *scratch = NULL;
    blst_p1_affine *p_affine = NULL;
    blst_scalar *scalars = NULL;

    /* Tunable parameter: must be at least 2 since blst fails for 0 or 1 */
    if (len < 8) {
        g1_lincomb_naive(out, p, coeffs, len);
    } else {
        /* blst's implementation of the Pippenger method */
        size_t scratch_size = blst_p1s_mult_pippenger_scratch_sizeof(len);
        ret = c_kzg_malloc(&scratch, scratch_size);
        if (ret != C_KZG_OK) goto out;
        ret = c_kzg_calloc((void **)&p_affine, len, sizeof(blst_p1_affine));
        if (ret != C_KZG_OK) goto out;
        ret = c_kzg_calloc((void **)&scalars, len, sizeof(blst_scalar));
        if (ret != C_KZG_OK) goto out;

        /* Transform the points to affine representation */
        const blst_p1 *p_arg[2] = {p, NULL};
        blst_p1s_to_affine(p_affine, p_arg, len);

        /* Transform the field elements to 256-bit scalars */
        for (uint64_t i = 0; i < len; i++) {
            blst_scalar_from_fr(&scalars[i], &coeffs[i]);
        }

        /* Call the Pippenger implementation */
        const byte *scalars_arg[2] = {(byte *)scalars, NULL};
        const blst_p1_affine *points_arg[2] = {p_affine, NULL};
        blst_p1s_mult_pippenger(
            out, points_arg, len, scalars_arg, 255, scratch
        );
    }

    ret = C_KZG_OK;

out:
    c_kzg_free(scratch);
    c_kzg_free(p_affine);
    c_kzg_free(scalars);
    return ret;
}

/**
 * Compute and return [ x^0, x^1, ..., x^{n-1} ].
 *
 * @remark `out` is left untouched if `n == 0`.
 *
 * @param[out] out The array to store the powers
 * @param[in]  x   The field element to raise to powers
 * @param[in]  n   The number of powers to compute
 */
static void compute_powers(fr_t *out, const fr_t *x, uint64_t n) {
    fr_t current_power = FR_ONE;
    for (uint64_t i = 0; i < n; i++) {
        out[i] = current_power;
        blst_fr_mul(&current_power, &current_power, x);
    }
}

///////////////////////////////////////////////////////////////////////////////
// Polynomials Functions
///////////////////////////////////////////////////////////////////////////////

/**
 * Evaluate a polynomial in evaluation form at a given point.
 *
 * @param[out] out The result of the evaluation
 * @param[in]  p   The polynomial in evaluation form
 * @param[in]  x   The point to evaluate the polynomial at
 * @param[in]  s   The trusted setup
 */
static C_KZG_RET evaluate_polynomial_in_evaluation_form(
    fr_t *out, const Polynomial *p, const fr_t *x, const KZGSettings *s
) {
    C_KZG_RET ret;
    fr_t tmp;
    fr_t *inverses_in = NULL;
    fr_t *inverses = NULL;
    uint64_t i;
    const fr_t *roots_of_unity = s->roots_of_unity;

    ret = new_fr_array(&inverses_in, FIELD_ELEMENTS_PER_BLOB);
    if (ret != C_KZG_OK) goto out;
    ret = new_fr_array(&inverses, FIELD_ELEMENTS_PER_BLOB);
    if (ret != C_KZG_OK) goto out;

    for (i = 0; i < FIELD_ELEMENTS_PER_BLOB; i++) {
        /*
         * If the point to evaluate at is one of the evaluation points by which
         * the polynomial is given, we can just return the result directly.
         * Note that special-casing this is necessary, as the formula below
         * would divide by zero otherwise.
         */
        if (fr_equal(x, &roots_of_unity[i])) {
            *out = p->evals[i];
            ret = C_KZG_OK;
            goto out;
        }
        blst_fr_sub(&inverses_in[i], x, &roots_of_unity[i]);
    }

    ret = fr_batch_inv(inverses, inverses_in, FIELD_ELEMENTS_PER_BLOB);
    if (ret != C_KZG_OK) goto out;

    *out = FR_ZERO;
    for (i = 0; i < FIELD_ELEMENTS_PER_BLOB; i++) {
        blst_fr_mul(&tmp, &inverses[i], &roots_of_unity[i]);
        blst_fr_mul(&tmp, &tmp, &p->evals[i]);
        blst_fr_add(out, out, &tmp);
    }
    fr_from_uint64(&tmp, FIELD_ELEMENTS_PER_BLOB);
    fr_div(out, out, &tmp);
    fr_pow(&tmp, x, FIELD_ELEMENTS_PER_BLOB);
    blst_fr_sub(&tmp, &tmp, &FR_ONE);
    blst_fr_mul(out, out, &tmp);

out:
    c_kzg_free(inverses_in);
    c_kzg_free(inverses);
    return ret;
}

///////////////////////////////////////////////////////////////////////////////
// KZG Functions
///////////////////////////////////////////////////////////////////////////////

/**
 * Compute a KZG commitment from a polynomial.
 *
 * @param[out] out The resulting commitment
 * @param[in]  p   The polynomial to commit to
 * @param[in]  s   The trusted setup
 */
static C_KZG_RET poly_to_kzg_commitment(
    g1_t *out, const Polynomial *p, const KZGSettings *s
) {
    return g1_lincomb_fast(
        out, s->g1_values, (const fr_t *)(&p->evals), FIELD_ELEMENTS_PER_BLOB
    );
}

/**
 * Convert a blob to a KZG commitment.
 *
 * @param[out] out  The resulting commitment
 * @param[in]  blob The blob representing the polynomial to be committed to
 * @param[in]  s    The trusted setup
 */
C_KZG_RET BLOB_TO_KZG_COMMITMENT(
    KZGCommitment *out, const Blob *blob, const KZGSettings *s
) {
    C_KZG_RET ret;
    Polynomial p;
    g1_t commitment;

    ret = blob_to_polynomial(&p, blob);
    if (ret != C_KZG_OK) return ret;
    ret = poly_to_kzg_commitment(&commitment, &p, s);
    if (ret != C_KZG_OK) return ret;
    bytes_from_g1(out, &commitment);
    return C_KZG_OK;
}

/* Forward function declaration */
static C_KZG_RET verify_kzg_proof_impl(
    bool *ok,
    const g1_t *commitment,
    const fr_t *z,
    const fr_t *y,
    const g1_t *proof,
    const KZGSettings *s
);

/**
 * Verify a KZG proof claiming that `p(z) == y`.
 *
 * @param[out] ok         True if the proofs are valid, otherwise false
 * @param[in]  commitment The KZG commitment corresponding to poly p(x)
 * @param[in]  z          The evaluation point
 * @param[in]  y          The claimed evaluation result
 * @param[in]  kzg_proof  The KZG proof
 * @param[in]  s          The trusted setup
 */
C_KZG_RET VERIFY_KZG_PROOF(
    bool *ok,
    const Bytes48 *commitment_bytes,
    const Bytes32 *z_bytes,
    const Bytes32 *y_bytes,
    const Bytes48 *proof_bytes,
    const KZGSettings *s
) {
    C_KZG_RET ret;
    fr_t z_fr, y_fr;
    g1_t commitment_g1, proof_g1;

    *ok = false;

    /* Convert untrusted inputs to trusted inputs */
    ret = bytes_to_kzg_commitment(&commitment_g1, commitment_bytes);
    if (ret != C_KZG_OK) return ret;
    ret = bytes_to_bls_field(&z_fr, z_bytes);
    if (ret != C_KZG_OK) return ret;
    ret = bytes_to_bls_field(&y_fr, y_bytes);
    if (ret != C_KZG_OK) return ret;
    ret = bytes_to_kzg_proof(&proof_g1, proof_bytes);
    if (ret != C_KZG_OK) return ret;

    /* Call helper to do pairings check */
    return verify_kzg_proof_impl(
        ok, &commitment_g1, &z_fr, &y_fr, &proof_g1, s
    );
}

/**
 * Helper function: Verify KZG proof claiming that `p(z) == y`.
 *
 * Given a @p commitment to a polynomial, a @p proof for @p z, and the
 * claimed value @p y at @p z, verify the claim.
 *
 * @param[out] out        `true` if the proof is valid, `false` if not
 * @param[in]  commitment The commitment to a polynomial
 * @param[in]  z          The point at which the proof is to be checked
 *                        (opened)
 * @param[in]  y          The claimed value of the polynomial at @p z
 * @param[in]  proof      A proof of the value of the polynomial at the
 *                        point @p z
 * @param[in]  s          The trusted setup
 */
static C_KZG_RET verify_kzg_proof_impl(
    bool *ok,
    const g1_t *commitment,
    const fr_t *z,
    const fr_t *y,
    const g1_t *proof,
    const KZGSettings *s
) {
    g2_t x_g2, X_minus_z;
    g1_t y_g1, P_minus_y;

    /* Calculate: X_minus_z */
    g2_mul(&x_g2, &G2_GENERATOR, z);
    g2_sub(&X_minus_z, &s->g2_values[1], &x_g2);

    /* Calculate: P_minus_y */
    g1_mul(&y_g1, &G1_GENERATOR, y);
    g1_sub(&P_minus_y, commitment, &y_g1);

    /* Verify: P - y = Q * (X - z) */
    *ok = pairings_verify(&P_minus_y, &G2_GENERATOR, proof, &X_minus_z);

    return C_KZG_OK;
}

/* Forward function declaration */
static C_KZG_RET compute_kzg_proof_impl(
    KZGProof *proof_out,
    fr_t *y_out,
    const Polynomial *polynomial,
    const fr_t *z,
    const KZGSettings *s
);

/**
 * Compute KZG proof for polynomial in Lagrange form at position z.
 *
 * @param[out] proof_out The combined proof as a single G1 element
 * @param[out] y_out     The evaluation of the polynomial at the evaluation
 *                       point z
 * @param[in]  blob      The blob (polynomial) to generate a proof for
 * @param[in]  z         The generator z-value for the evaluation points
 * @param[in]  s         The trusted setup
 */
C_KZG_RET COMPUTE_KZG_PROOF(
    KZGProof *proof_out,
    Bytes32 *y_out,
    const Blob *blob,
    const Bytes32 *z_bytes,
    const KZGSettings *s
) {
    C_KZG_RET ret;
    Polynomial polynomial;
    fr_t frz, fry;

    ret = blob_to_polynomial(&polynomial, blob);
    if (ret != C_KZG_OK) goto out;
    ret = bytes_to_bls_field(&frz, z_bytes);
    if (ret != C_KZG_OK) goto out;
    ret = compute_kzg_proof_impl(proof_out, &fry, &polynomial, &frz, s);
    if (ret != C_KZG_OK) goto out;
    bytes_from_bls_field(y_out, &fry);

out:
    return ret;
}

/**
 * Helper function for compute_kzg_proof() and
 * compute_blob_kzg_proof().
 *
 * @param[out] proof_out  The combined proof as a single G1 element
 * @param[out] y_out      The evaluation of the polynomial at the evaluation
 *                        point z
 * @param[in]  polynomial The polynomial in Lagrange form
 * @param[in]  z          The evaluation point
 * @param[in]  s          The trusted setup
 */
static C_KZG_RET compute_kzg_proof_impl(
    KZGProof *proof_out,
    fr_t *y_out,
    const Polynomial *polynomial,
    const fr_t *z,
    const KZGSettings *s
) {
    C_KZG_RET ret;
    fr_t *inverses_in = NULL;
    fr_t *inverses = NULL;

    ret = evaluate_polynomial_in_evaluation_form(y_out, polynomial, z, s);
    if (ret != C_KZG_OK) goto out;

    fr_t tmp;
    Polynomial q;
    const fr_t *roots_of_unity = s->roots_of_unity;
    uint64_t i;
    /* m != 0 indicates that the evaluation point z equals root_of_unity[m-1] */
    uint64_t m = 0;

    ret = new_fr_array(&inverses_in, FIELD_ELEMENTS_PER_BLOB);
    if (ret != C_KZG_OK) goto out;
    ret = new_fr_array(&inverses, FIELD_ELEMENTS_PER_BLOB);
    if (ret != C_KZG_OK) goto out;

    for (i = 0; i < FIELD_ELEMENTS_PER_BLOB; i++) {
        if (fr_equal(z, &roots_of_unity[i])) {
            /* We are asked to compute a KZG proof inside the domain */
            m = i + 1;
            inverses_in[i] = FR_ONE;
            continue;
        }
        // (p_i - y) / (ω_i - z)
        blst_fr_sub(&q.evals[i], &polynomial->evals[i], y_out);
        blst_fr_sub(&inverses_in[i], &roots_of_unity[i], z);
    }

    ret = fr_batch_inv(inverses, inverses_in, FIELD_ELEMENTS_PER_BLOB);
    if (ret != C_KZG_OK) goto out;

    for (i = 0; i < FIELD_ELEMENTS_PER_BLOB; i++) {
        blst_fr_mul(&q.evals[i], &q.evals[i], &inverses[i]);
    }

    if (m != 0) { /* ω_{m-1} == z */
        q.evals[--m] = FR_ZERO;
        for (i = 0; i < FIELD_ELEMENTS_PER_BLOB; i++) {
            if (i == m) continue;
            /* Build denominator: z * (z - ω_i) */
            blst_fr_sub(&tmp, z, &roots_of_unity[i]);
            blst_fr_mul(&inverses_in[i], &tmp, z);
        }

        ret = fr_batch_inv(inverses, inverses_in, FIELD_ELEMENTS_PER_BLOB);
        if (ret != C_KZG_OK) goto out;

        for (i = 0; i < FIELD_ELEMENTS_PER_BLOB; i++) {
            if (i == m) continue;
            /* Build numerator: ω_i * (p_i - y) */
            blst_fr_sub(&tmp, &polynomial->evals[i], y_out);
            blst_fr_mul(&tmp, &tmp, &roots_of_unity[i]);
            /* Do the division: (p_i - y) * ω_i / (z * (z - ω_i)) */
            blst_fr_mul(&tmp, &tmp, &inverses[i]);
            blst_fr_add(&q.evals[m], &q.evals[m], &tmp);
        }
    }

    g1_t out_g1;
    ret = g1_lincomb_fast(
        &out_g1, s->g1_values, (const fr_t *)(&q.evals), FIELD_ELEMENTS_PER_BLOB
    );
    if (ret != C_KZG_OK) goto out;

    bytes_from_g1(proof_out, &out_g1);

out:
    c_kzg_free(inverses_in);
    c_kzg_free(inverses);
    return ret;
}

/**
 * Given a blob and a commitment, return the KZG proof that is used to verify
 * it against the commitment. This function does not verify that the commitment
 * is correct with respect to the blob.
 *
 * @param[out] out              The resulting proof
 * @param[in]  blob             A blob
 * @param[in]  commitment_bytes Commitment to verify
 * @param[in]  s                The trusted setup
 */
C_KZG_RET COMPUTE_BLOB_KZG_PROOF(
    KZGProof *out,
    const Blob *blob,
    const Bytes48 *commitment_bytes,
    const KZGSettings *s
) {
    C_KZG_RET ret;
    Polynomial polynomial;
    g1_t commitment_g1;
    fr_t evaluation_challenge_fr;
    fr_t y;

    /* Do conversions first to fail fast, compute_challenge is expensive */
    ret = bytes_to_kzg_commitment(&commitment_g1, commitment_bytes);
    if (ret != C_KZG_OK) goto out;
    ret = blob_to_polynomial(&polynomial, blob);
    if (ret != C_KZG_OK) goto out;

    /* Compute the challenge for the given blob/commitment */
    compute_challenge(&evaluation_challenge_fr, blob, &commitment_g1);

    /* Call helper function to compute proof and y */
    ret = compute_kzg_proof_impl(
        out, &y, &polynomial, &evaluation_challenge_fr, s
    );
    if (ret != C_KZG_OK) goto out;

out:
    return ret;
}

/**
 * Given a blob and its proof, verify that it corresponds to the provided
 * commitment.
 *
 * @param[out] ok               True if the proofs are valid, otherwise false
 * @param[in]  blob             Blob to verify
 * @param[in]  commitment_bytes Commitment to verify
 * @param[in]  proof_bytes      Proof used for verification
 * @param[in]  s                The trusted setup
 */
C_KZG_RET VERIFY_BLOB_KZG_PROOF(
    bool *ok,
    const Blob *blob,
    const Bytes48 *commitment_bytes,
    const Bytes48 *proof_bytes,
    const KZGSettings *s
) {
    C_KZG_RET ret;
    Polynomial polynomial;
    fr_t evaluation_challenge_fr, y_fr;
    g1_t commitment_g1, proof_g1;

    *ok = false;

    /* Do conversions first to fail fast, compute_challenge is expensive */
    ret = bytes_to_kzg_commitment(&commitment_g1, commitment_bytes);
    if (ret != C_KZG_OK) return ret;
    ret = blob_to_polynomial(&polynomial, blob);
    if (ret != C_KZG_OK) return ret;
    ret = bytes_to_kzg_proof(&proof_g1, proof_bytes);
    if (ret != C_KZG_OK) return ret;

    /* Compute challenge for the blob/commitment */
    compute_challenge(&evaluation_challenge_fr, blob, &commitment_g1);

    /* Evaluate challenge to get y */
    ret = evaluate_polynomial_in_evaluation_form(
        &y_fr, &polynomial, &evaluation_challenge_fr, s
    );
    if (ret != C_KZG_OK) return ret;

    /* Call helper to do pairings check */
    return verify_kzg_proof_impl(
        ok, &commitment_g1, &evaluation_challenge_fr, &y_fr, &proof_g1, s
    );
}

/**
 * Compute random linear combination challenge scalars for batch verification.
 *
 * @param[out]  r_powers_out   The output challenges
 * @param[in]   commitments_g1 The input commitments
 * @param[in]   zs_fr          The input evaluation points
 * @param[in]   ys_fr          The input evaluation results
 * @param[in]   proofs_g1      The input proofs
 */
static C_KZG_RET compute_r_powers(
    fr_t *r_powers_out,
    const g1_t *commitments_g1,
    const fr_t *zs_fr,
    const fr_t *ys_fr,
    const g1_t *proofs_g1,
    size_t n
) {
    C_KZG_RET ret;
    uint8_t *bytes = NULL;
    Bytes32 r_bytes;
    fr_t r;

    size_t input_size = DOMAIN_STR_LENGTH + sizeof(uint64_t) +
                        sizeof(uint64_t) +
                        (n * (BYTES_PER_COMMITMENT +
                              2 * BYTES_PER_FIELD_ELEMENT + BYTES_PER_PROOF));
    ret = c_kzg_malloc((void **)&bytes, input_size);
    if (ret != C_KZG_OK) goto out;

    /* Pointer tracking `bytes` for writing on top of it */
    uint8_t *offset = bytes;

    /* Copy domain separator */
    memcpy(offset, RANDOM_CHALLENGE_KZG_BATCH_DOMAIN, DOMAIN_STR_LENGTH);
    offset += DOMAIN_STR_LENGTH;

    /* Copy degree of the polynomial */
    bytes_from_uint64(offset, FIELD_ELEMENTS_PER_BLOB);
    offset += sizeof(uint64_t);

    /* Copy number of commitments */
    bytes_from_uint64(offset, n);
    offset += sizeof(uint64_t);

    for (size_t i = 0; i < n; i++) {
        /* Copy commitment */
        bytes_from_g1((Bytes48 *)offset, &commitments_g1[i]);
        offset += BYTES_PER_COMMITMENT;

        /* Copy z */
        bytes_from_bls_field((Bytes32 *)offset, &zs_fr[i]);
        offset += BYTES_PER_FIELD_ELEMENT;

        /* Copy y */
        bytes_from_bls_field((Bytes32 *)offset, &ys_fr[i]);
        offset += BYTES_PER_FIELD_ELEMENT;

        /* Copy proof */
        bytes_from_g1((Bytes48 *)offset, &proofs_g1[i]);
        offset += BYTES_PER_PROOF;
    }

    /* Now let's create the challenge! */
    blst_sha256(r_bytes.bytes, bytes, input_size);
    hash_to_bls_field(&r, &r_bytes);

    compute_powers(r_powers_out, &r, n);

    /* Make sure we wrote the entire buffer */
    assert(offset == bytes + input_size);

out:
    c_kzg_free(bytes);
    return ret;
}

/**
 * Helper function for verify_blob_kzg_proof_batch(): actually perform the
 * verification.
 *
 * @remark This function assumes that `n` is trusted and that all input arrays
 *     contain `n` elements. `n` should be the actual size of the arrays and not
 *     read off a length field in the protocol.
 *
 * @remark This function only works for `n > 0`.
 *
 * @param[out] ok             True if the proofs are valid, otherwise false
 * @param[in]  commitments_g1 Array of commitments to verify
 * @param[in]  zs_fr          Array of evaluation points for the KZG proofs
 * @param[in]  ys_fr          Array of evaluation results for the KZG proofs
 * @param[in]  proofs_g1      Array of proofs used for verification
 * @param[in]  n              The number of blobs/commitments/proofs
 * @param[in]  s              The trusted setup
 */
static C_KZG_RET verify_kzg_proof_batch(
    bool *ok,
    const g1_t *commitments_g1,
    const fr_t *zs_fr,
    const fr_t *ys_fr,
    const g1_t *proofs_g1,
    size_t n,
    const KZGSettings *s
) {
    C_KZG_RET ret;
    g1_t proof_lincomb, proof_z_lincomb, C_minus_y_lincomb, rhs_g1;
    fr_t *r_powers = NULL;
    g1_t *C_minus_y = NULL;
    fr_t *r_times_z = NULL;

    assert(n > 0);

    *ok = false;

    /* First let's allocate our arrays */
    ret = new_fr_array(&r_powers, n);
    if (ret != C_KZG_OK) goto out;
    ret = new_g1_array(&C_minus_y, n);
    if (ret != C_KZG_OK) goto out;
    ret = new_fr_array(&r_times_z, n);
    if (ret != C_KZG_OK) goto out;

    /* Compute the random lincomb challenges */
    ret = compute_r_powers(
        r_powers, commitments_g1, zs_fr, ys_fr, proofs_g1, n
    );
    if (ret != C_KZG_OK) goto out;

    /* Compute \sum r^i * Proof_i */
    g1_lincomb_naive(&proof_lincomb, proofs_g1, r_powers, n);

    for (size_t i = 0; i < n; i++) {
        g1_t ys_encrypted;
        /* Get [y_i] */
        g1_mul(&ys_encrypted, &G1_GENERATOR, &ys_fr[i]);
        /* Get C_i - [y_i] */
        g1_sub(&C_minus_y[i], &commitments_g1[i], &ys_encrypted);
        /* Get r^i * z_i */
        blst_fr_mul(&r_times_z[i], &r_powers[i], &zs_fr[i]);
    }

    /* Get \sum r^i z_i Proof_i */
    g1_lincomb_naive(&proof_z_lincomb, proofs_g1, r_times_z, n);
    /* Get \sum r^i (C_i - [y_i]) */
    g1_lincomb_naive(&C_minus_y_lincomb, C_minus_y, r_powers, n);
    /* Get C_minus_y_lincomb + proof_z_lincomb */
    blst_p1_add_or_double(&rhs_g1, &C_minus_y_lincomb, &proof_z_lincomb);

    /* Do the pairing check! */
    *ok = pairings_verify(
        &proof_lincomb, &s->g2_values[1], &rhs_g1, &G2_GENERATOR
    );

out:
    c_kzg_free(r_powers);
    c_kzg_free(C_minus_y);
    c_kzg_free(r_times_z);
    return ret;
}

/**
 * Given a list of blobs and blob KZG proofs, verify that they correspond to the
 * provided commitments.
 *
 * @remark This function assumes that `n` is trusted and that all input arrays
 * contain `n` elements. `n` should be the actual size of the arrays and not
 * read off a length field in the protocol.
 *
 * @remark This function accepts if called with `n==0`.
 *
 * @param[out] ok                True if the proofs are valid, otherwise false
 * @param[in]  blobs             Array of blobs to verify
 * @param[in]  commitments_bytes Array of commitments to verify
 * @param[in]  proofs_bytes      Array of proofs used for verification
 * @param[in]  n                 The number of blobs/commitments/proofs
 * @param[in]  s                 The trusted setup
 */
C_KZG_RET VERIFY_BLOB_KZG_PROOF_BATCH(
    bool *ok,
    const Blob *blobs,
    const Bytes48 *commitments_bytes,
    const Bytes48 *proofs_bytes,
    size_t n,
    const KZGSettings *s
) {
    C_KZG_RET ret;
    g1_t *commitments_g1 = NULL;
    g1_t *proofs_g1 = NULL;
    fr_t *evaluation_challenges_fr = NULL;
    fr_t *ys_fr = NULL;

    /* Exit early if we are given zero blobs */
    if (n == 0) {
        *ok = true;
        return C_KZG_OK;
    }

    /* For a single blob, just do a regular single verification */
    if (n == 1) {
        return VERIFY_BLOB_KZG_PROOF(
            ok, &blobs[0], &commitments_bytes[0], &proofs_bytes[0], s
        );
    }

    /* We will need a bunch of arrays to store our objects... */
    ret = new_g1_array(&commitments_g1, n);
    if (ret != C_KZG_OK) goto out;
    ret = new_g1_array(&proofs_g1, n);
    if (ret != C_KZG_OK) goto out;
    ret = new_fr_array(&evaluation_challenges_fr, n);
    if (ret != C_KZG_OK) goto out;
    ret = new_fr_array(&ys_fr, n);
    if (ret != C_KZG_OK) goto out;

    for (size_t i = 0; i < n; i++) {
        Polynomial polynomial;

        /* Convert each commitment to a g1 point */
        ret = bytes_to_kzg_commitment(
            &commitments_g1[i], &commitments_bytes[i]
        );
        if (ret != C_KZG_OK) goto out;

        /* Convert each blob from bytes to a poly */
        ret = blob_to_polynomial(&polynomial, &blobs[i]);
        if (ret != C_KZG_OK) goto out;

        compute_challenge(
            &evaluation_challenges_fr[i], &blobs[i], &commitments_g1[i]
        );

        ret = evaluate_polynomial_in_evaluation_form(
            &ys_fr[i], &polynomial, &evaluation_challenges_fr[i], s
        );
        if (ret != C_KZG_OK) goto out;

        ret = bytes_to_kzg_proof(&proofs_g1[i], &proofs_bytes[i]);
        if (ret != C_KZG_OK) goto out;
    }

    ret = verify_kzg_proof_batch(
        ok, commitments_g1, evaluation_challenges_fr, ys_fr, proofs_g1, n, s
    );

out:
    c_kzg_free(commitments_g1);
    c_kzg_free(proofs_g1);
    c_kzg_free(evaluation_challenges_fr);
    c_kzg_free(ys_fr);
    return ret;
}

///////////////////////////////////////////////////////////////////////////////
// Trusted Setup Functions
///////////////////////////////////////////////////////////////////////////////

/**
 * Convert little-endian bytes to a uint16_t.
 *
 * @param[in] bytes A pointer to 2 bytes
 *
 * @return The uint16_t value
 */
static uint16_t parse_uint16_le(const byte *bytes) {
    return (uint32_t)(bytes[0]) | ((uint32_t)(bytes[1]) << 8);
}

/**
 * Convert little-endian bytes to a uint32_t.
 *
 * @param[in] bytes A pointer to 4 bytes
 *
 * @return The uint32_t value
 */
static uint32_t parse_uint32_le(const byte *bytes) {
    return (uint32_t)(bytes[0]) | ((uint32_t)(bytes[1]) << 8) |
           ((uint32_t)(bytes[2]) << 16) | ((uint32_t)(bytes[3]) << 24);
}

/**
 * Convert little-endian bytes to a uint64_t.
 *
 * @param[in] bytes A pointer to 8 bytes
 *
 * @return The uint64_t value
 */
static uint64_t parse_uint64_le(const byte *bytes) {
    return (uint64_t)(bytes[0]) | ((uint64_t)(bytes[1]) << 8) |
           ((uint64_t)(bytes[2]) << 16) | ((uint64_t)(bytes[3]) << 24) |
           ((uint64_t)(bytes[4]) << 32) | ((uint64_t)(bytes[5]) << 40) |
           ((uint64_t)(bytes[6]) << 48) | ((uint64_t)(bytes[7]) << 56);
}

/**
 * Convert little-endian bytes to a blst_p1_affine.
 *
 * @param[out] bytes A pointer to a blst_p1_affine variable.
 * @param[in] bytes A pointer to 96 bytes
 */
static void parse_blst_p1_affine_le(blst_p1_affine *out, const byte *bytes) {
    blst_p1_affine *in = (blst_p1_affine *)bytes;
    for (int i = 0; i < 6; i++) {
        out->x.l[i] = parse_uint64_le((byte *)&in->x.l[i]);
        out->y.l[i] = parse_uint64_le((byte *)&in->y.l[i]);
    }
}

/**
 * Convert little-endian bytes to a blst_p2_affine.
 *
 * @param[out] bytes A pointer to a blst_p2_affine variable.
 * @param[in] bytes A pointer to 192 bytes
 */
static void parse_blst_p2_affine_le(blst_p2_affine *out, const byte *bytes) {
    blst_p2_affine *in = (blst_p2_affine *)bytes;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 6; j++) {
            out->x.fp[i].l[j] = parse_uint64_le((byte *)&in->x.fp[i].l[j]);
            out->y.fp[i].l[j] = parse_uint64_le((byte *)&in->y.fp[i].l[j]);
        }
    }
}

/**
 * Convert little-endian bytes to a blst_p2_affine.
 *
 * @param[out] bytes A pointer to a blst_fr variable.
 * @param[in] bytes A pointer to 32 bytes
 */
static void parse_blst_fr_le(blst_fr *out, const byte *bytes) {
    blst_fr *in = (blst_fr *)bytes;
    for (int i = 0; i < 4; i++) {
        out->l[i] = parse_uint64_le((byte *)&in->l[i]);
    }
}

/**
 * Free a trusted setup (KZGSettings).
 *
 * @remark It's a NOP if `s` is NULL.
 *
 * @param[in] s The trusted setup to free
 */
void FREE_TRUSTED_SETUP(KZGSettings *s) {
    if (s == NULL) return;
    c_kzg_free(s->roots_of_unity);
    c_kzg_free(s->g1_values);
    c_kzg_free(s->g2_values);
}

/**
 * Read the trusted setup file contents.
 *
 * @remark The pointer must be freed afterwards.
 *
 * @param[out] out    A double-pointer to bytes
 * @param[out] length Length of the contents
 * @param[in]  in     File handle for input
 */
static C_KZG_RET get_trusted_setup_bytes(
    byte **out, size_t *length, FILE *file
) {
    C_KZG_RET ret;

    /* Get the length of the file */
    fseek(file, 0, SEEK_END);
    *length = ftell(file);
    fseek(file, 0, SEEK_SET);

    /* Allocate space for the file contents */
    ret = c_kzg_malloc((void **)out, *length);
    if (ret != C_KZG_OK) {
        goto out_error;
    }

    /* Read the file contents */
    size_t read = fread(*out, 1, *length, file);
    if (read != *length) {
        ret = C_KZG_BADARGS;
        goto out_error;
    }

    goto out_success;

out_error:
    c_kzg_free(*out);
out_success:
    return ret;
}

typedef struct __attribute__((__packed__)) {
    byte magic[12];
    byte version[4];
    char protocol[32];
    char curve[15];
    uint8_t fields;
} TsifHeader;

typedef struct __attribute__((__packed__)) {
    char description[15];
    byte type[2];
    byte order[3];
    byte element_size[4];
    byte num_elements[8];
} TsifSchema;

typedef enum {
    TSIF_SCHEMA_TYPE_G1 = 12647,
    TSIF_SCHEMA_TYPE_G2 = 12903,
    TSIF_SCHEMA_TYPE_FR = 29286
} TsifType;

/**
 * Load trusted setup from a file.
 *
 * @remark The input file will not be closed.
 * @remark Free after use with free_trusted_setup().
 *
 * @param[out] out Pointer to the loaded trusted setup data
 * @param[in]  in  File handle for input
 */
C_KZG_RET LOAD_TRUSTED_SETUP_FILE(KZGSettings *out, FILE *in) {
    C_KZG_RET ret;

    /* Initialize the fields */
    out->roots_of_unity = NULL;
    out->g1_values = NULL;
    out->g2_values = NULL;

    /* Read the trusted setup file */
    byte *tsif = NULL;
    size_t tsif_length = 0;
    ret = get_trusted_setup_bytes(&tsif, &tsif_length, in);
    if (ret != C_KZG_OK) {
        goto out_error;
    }

    /* Ensure the header can be read */
    if (tsif_length < sizeof(TsifHeader)) {
        ret = C_KZG_BADARGS;
        goto out_error;
    }

    TsifHeader *header = (TsifHeader *)tsif;

    /* Perform sanity checks */
    byte expected_magic[12] = {
        0xE2, 0x88, 0x83, 0xE2, 0x8B, 0x83, 0xE2, 0x88, 0x88, 0xE2, 0x88, 0x8E};
    if (memcmp(header->magic, expected_magic, 12) != 0) {
        ret = C_KZG_BADARGS;
        goto out_error;
    }
    byte expected_version[12] = {'v', 0x01, '.', 0x00};
    if (memcmp(header->version, expected_version, 4) != 0) {
        ret = C_KZG_BADARGS;
        goto out_error;
    }
    if (memcmp(header->protocol, "ethereum_deneb_kzg", 18) != 0) {
        ret = C_KZG_BADARGS;
        goto out_error;
    }
    if (memcmp(header->curve, "bls12_381", 9) != 0) {
        ret = C_KZG_BADARGS;
        goto out_error;
    }

    /* Calculate offset to the start of the data section */
    size_t schemas_length = header->fields * sizeof(TsifSchema);
    size_t data_offset = sizeof(TsifHeader) + schemas_length;

    /* Adjust for padding */
    if (data_offset % 64 != 0) {
        data_offset += 32;
    }

    /* Iterate over each schema & parse data */
    TsifSchema *schema = (TsifSchema *)(tsif + sizeof(TsifHeader));
    for (int i = 0; i < header->fields; i++) {
        byte *data = tsif + data_offset;

        /* Check the contents are long enough for reported schemas */
        if (tsif_length < data_offset) {
            ret = C_KZG_BADARGS;
            goto out_error;
        }

        /* Extract necessary information from the schema */
        TsifType type = parse_uint16_le(schema->type);
        uint64_t element_size = parse_uint32_le(schema->element_size);
        uint64_t num_elements = parse_uint64_le(schema->num_elements);

        /* Update offset to the next data section for the next iteration */
        data_offset += num_elements * element_size;

        switch (type) {
        case TSIF_SCHEMA_TYPE_G1:
            /* Perform sanity checks */
            if (num_elements != FIELD_ELEMENTS_PER_BLOB) {
                ret = C_KZG_BADARGS;
                goto out_error;
            }
            if (element_size != sizeof(blst_p1_affine)) {
                ret = C_KZG_BADARGS;
                goto out_error;
            }
            if (memcmp(schema->description, "srs_lagrange", 12) != 0) {
                ret = C_KZG_BADARGS;
                goto out_error;
            }
            if (memcmp(schema->order, "brp", 3) != 0) {
                ret = C_KZG_BADARGS;
                goto out_error;
            }

            /* Allocate memory for the values */
            ret = new_g1_array(&out->g1_values, num_elements);
            if (ret != C_KZG_OK) goto out_error;

            /* Copy points into the settings */
            blst_p1_affine g1_affine;
            for (uint16_t j = 0; j < num_elements; j++) {
                byte *ptr = data + j * element_size;
                parse_blst_p1_affine_le(&g1_affine, ptr);
                blst_p1_from_affine(&out->g1_values[j], &g1_affine);
            }
            break;

        case TSIF_SCHEMA_TYPE_G2:
            /* Perform sanity checks */
            if (num_elements != TRUSTED_SETUP_NUM_G2_POINTS) {
                ret = C_KZG_BADARGS;
                goto out_error;
            }
            if (element_size != sizeof(blst_p2_affine)) {
                ret = C_KZG_BADARGS;
                goto out_error;
            }
            if (memcmp(schema->description, "srs_monomial", 12) != 0) {
                ret = C_KZG_BADARGS;
                goto out_error;
            }
            if (memcmp(schema->order, "asc", 3) != 0) {
                ret = C_KZG_BADARGS;
                goto out_error;
            }

            /* Allocate memory for the values */
            ret = new_g2_array(&out->g2_values, num_elements);
            if (ret != C_KZG_OK) goto out_error;

            /* Copy points into the settings */
            blst_p2_affine g2_affine;
            for (uint16_t j = 0; j < num_elements; j++) {
                byte *ptr = data + j * element_size;
                parse_blst_p2_affine_le(&g2_affine, ptr);
                blst_p2_from_affine(&out->g2_values[j], &g2_affine);
            }
            break;

        case TSIF_SCHEMA_TYPE_FR:
            /* Perform sanity checks */
            if (num_elements != FIELD_ELEMENTS_PER_BLOB) {
                ret = C_KZG_BADARGS;
                goto out_error;
            }
            if (element_size != sizeof(fr_t)) {
                ret = C_KZG_BADARGS;
                goto out_error;
            }
            if (memcmp(schema->description, "roots_unity", 11) != 0) {
                ret = C_KZG_BADARGS;
                goto out_error;
            }
            if (memcmp(schema->order, "brp", 3) != 0) {
                ret = C_KZG_BADARGS;
                goto out_error;
            }

            /* Allocate memory for the values */
            ret = new_fr_array(&out->roots_of_unity, num_elements);
            if (ret != C_KZG_OK) goto out_error;

            /* Copy fields into the settings */
            for (uint16_t j = 0; j < num_elements; j++) {
                byte *ptr = data + j * element_size;
                parse_blst_fr_le(&out->roots_of_unity[j], ptr);
            }
            break;

        default:
            ret = C_KZG_BADARGS;
            goto out_error;
        }

        schema++;
    }

    /* Check that all of the contents were read */
    if (tsif_length != data_offset) {
        ret = C_KZG_BADARGS;
        goto out_error;
    }

    /* Check that the required fields were populated */
    if (out->g1_values == NULL || out->g2_values == NULL ||
        out->roots_of_unity == NULL) {
        ret = C_KZG_BADARGS;
        goto out_error;
    }

    goto out_success;

out_error:
    FREE_TRUSTED_SETUP(out);
out_success:
    c_kzg_free(tsif);
    return ret;
}
