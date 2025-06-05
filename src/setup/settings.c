/*
 * Copyright 2025 Benjamin Edgington
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

#include "setup/settings.h"
#include "setup/common.h"

#include <assert.h> /* For assert */
#include <stdlib.h> /* For NULL */
#include <string.h> /* For memcpy */

////////////////////////////////////////////////////////////////////////////////////////////////////
// MACROS
////////////////////////////////////////////////////////////////////////////////////////////////////

/** Represents a big endian platform. */
#define ENDIANNESS_BIG 1

/** Represents a little endian platform. */
#define ENDIANNESS_LITTLE 2

/** A helper constant to make things cleaner. */
#define FEPEB0 (FIELD_ELEMENTS_PER_EXT_BLOB + 0)

/** A helper constant to make things cleaner. */
#define FEPEB1 (FIELD_ELEMENTS_PER_EXT_BLOB + 1)

/** Do a memcpy then update offset. */
#define WRITE(dst, offset, src, size) \
    do { \
        memcpy((dst) + (offset), (src), (size)); \
        (offset) += (size); \
    } while (0)

////////////////////////////////////////////////////////////////////////////////////////////////////
// Types
////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct {
    uint8_t magic[4];
    uint8_t version;
    uint8_t endianness;
    uint8_t wordsize;
} header_t;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper Functions
////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Get the current platform's endianness.
 */
static uint8_t get_endianness(void) {
    uint32_t x = 1;
    return *((uint8_t *)&x) == 1 ? ENDIANNESS_LITTLE : ENDIANNESS_BIG;
}

/**
 * Get the current platform's wordsize (e.g., 32-bit or 64-bit).
 */
static uint8_t get_wordsize(void) {
    return (uint8_t)sizeof(void *);
}

/**
 * Get the size of settings if serialized.
 *
 * @param[in]   s   The settings to serialize
 *
 * @return The size (in bytes) of the serialized settings for this platform.
 */
static size_t compute_serialized_size(const KZGSettings *s) {
    size_t total_size = 0;

    /* header */
    total_size += sizeof(header_t);
    /* wbits */
    total_size += sizeof(size_t);
    /* scratch_size */
    total_size += sizeof(size_t);
    /* table_size */
    total_size += sizeof(size_t);
    /* roots_of_unity */
    total_size += FEPEB1 * sizeof(fr_t);
    /* brp_roots_of_unity */
    total_size += FEPEB0 * sizeof(fr_t);
    /* reverse_roots_of_unit */
    total_size += FEPEB1 * sizeof(fr_t);
    /* g1_values_monomial */
    total_size += NUM_G1_POINTS * sizeof(g1_t);
    /* g1_values_lagrange_brp */
    total_size += NUM_G1_POINTS * sizeof(g1_t);
    /* g2_values_monomial */
    total_size += NUM_G2_POINTS * sizeof(g2_t);
    /* x_ext_fft_columns */
    total_size += CELLS_PER_EXT_BLOB * FIELD_ELEMENTS_PER_CELL * sizeof(g1_t);
    /* tables */
    if (s->wbits != 0 && s->tables) {
        total_size += CELLS_PER_EXT_BLOB * s->table_size;
    }

    return total_size;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Initialization
////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Initialize all fields in KZGSettings to null/zero.
 *
 * @param[out]  out The KZGSettings to initialize.
 */
void init_settings(KZGSettings *out) {
    out->roots_of_unity = NULL;
    out->brp_roots_of_unity = NULL;
    out->reverse_roots_of_unity = NULL;
    out->g1_values_monomial = NULL;
    out->g1_values_lagrange_brp = NULL;
    out->g2_values_monomial = NULL;
    out->x_ext_fft_columns = NULL;
    out->tables = NULL;
    out->wbits = 0;
    out->scratch_size = 0;
    out->table_size = 0;
}

/**
 * Free all fields.
 *
 * @param[in]   s   The trusted setup to free
 *
 * @remark This does nothing if `s` is NULL.
 */
void free_settings(KZGSettings *s) {
    if (s == NULL) return;
    c_kzg_free(s->brp_roots_of_unity);
    c_kzg_free(s->roots_of_unity);
    c_kzg_free(s->reverse_roots_of_unity);
    c_kzg_free(s->g1_values_monomial);
    c_kzg_free(s->g1_values_lagrange_brp);
    c_kzg_free(s->g2_values_monomial);

    /*
     * If for whatever reason we accidentally call free_trusted_setup() on an uninitialized
     * structure, we don't want to deference these 2d arrays. Without these NULL checks, it's
     * possible for there to be a segmentation fault via null pointer dereference.
     */
    if (s->x_ext_fft_columns != NULL) {
        for (size_t i = 0; i < CELLS_PER_EXT_BLOB; i++) {
            c_kzg_free(s->x_ext_fft_columns[i]);
        }
    }
    if (s->tables != NULL) {
        for (size_t i = 0; i < CELLS_PER_EXT_BLOB; i++) {
            c_kzg_free(s->tables[i]);
        }
    }
    c_kzg_free(s->x_ext_fft_columns);
    c_kzg_free(s->tables);

    s->wbits = 0;
    s->scratch_size = 0;
    s->table_size = 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Serialization
////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Serialize the given KZG settings to bytes.
 *
 * @param[out]  out     A pointer to some uninitialized bytes
 * @param[out]  out_len The size of the output bytes
 * @param[in]   s       The settings to serialize
 *
 * @remark the output of this will only work on similar platforms.
 */
C_KZG_RET serialize_settings(uint8_t **out, size_t *out_len, const KZGSettings *s) {
    C_KZG_RET ret;

    /* Initialize the output length to zero */
    *out_len = 0;

    /* Allocate bytes for serialized output */
    size_t total_size = compute_serialized_size(s);
    ret = c_kzg_malloc((void **)out, total_size);
    if (ret != C_KZG_OK) goto out;

    /* Initialize header */
    header_t header = {
        .magic = {'K', 'Z', 'G', '\0'},
        .version = 1,
        .endianness = get_endianness(),
        .wordsize = get_wordsize()
    };

    /* Write all fields to the buffer */
    size_t offset = 0;
    WRITE(*out, offset, &header, sizeof(header_t));
    WRITE(*out, offset, &s->wbits, sizeof(size_t));
    WRITE(*out, offset, &s->scratch_size, sizeof(size_t));
    WRITE(*out, offset, &s->table_size, sizeof(size_t));
    WRITE(*out, offset, s->roots_of_unity, FEPEB1 * sizeof(fr_t));
    WRITE(*out, offset, s->brp_roots_of_unity, FEPEB0 * sizeof(fr_t));
    WRITE(*out, offset, s->reverse_roots_of_unity, FEPEB1 * sizeof(fr_t));
    WRITE(*out, offset, s->g1_values_monomial, NUM_G1_POINTS * sizeof(g1_t));
    WRITE(*out, offset, s->g1_values_lagrange_brp, NUM_G1_POINTS * sizeof(g1_t));
    WRITE(*out, offset, s->g2_values_monomial, NUM_G2_POINTS * sizeof(g2_t));
    for (size_t i = 0; i < CELLS_PER_EXT_BLOB; i++) {
        WRITE(*out, offset, s->x_ext_fft_columns[i], FIELD_ELEMENTS_PER_CELL * sizeof(g1_t));
    }
    if (s->wbits != 0 && s->tables) {
        for (size_t i = 0; i < CELLS_PER_EXT_BLOB; i++) {
            WRITE(*out, offset, s->tables[i], s->table_size);
        }
    }

    /* Set the output length */
    assert(total_size == offset);
    *out_len = total_size;

out:
    return ret;
}

/**
 * Deserialize some bytes to KZG settings.
 *
 * @param[out]  out     The deserialized settings
 * @param[out]  data    The serialized bytes
 * @param[in]   s       The size of the serialized bytes
 *
 * @remark the input must be from generated from a similar platform.
 */
C_KZG_RET deserialize_settings(KZGSettings *out, const uint8_t *data, size_t data_len) {
    C_KZG_RET ret;

    /*
     * Initialize all fields to null/zero so that if there's an error, we can can call
     * free_trusted_setup() without worrying about freeing a random pointer.
     */
    init_settings(out);

    /* Ensure the data is big enough to contain the header */
    if (data_len < sizeof(header_t)) return C_KZG_ERROR;

    size_t offset = 0;

    /* Read all fields from the buffer */
    header_t header;
    memcpy(&header, data + offset, sizeof(header_t));
    offset += sizeof(header_t);

    /* Ensure this data is compatible with the current platform */
    if (memcmp(header.magic, "KZG\0", 4) != 0) return C_KZG_BADARGS;
    if (header.version != 1) return C_KZG_BADARGS;
    if (header.endianness != get_endianness()) return C_KZG_BADARGS;
    if (header.wordsize != get_wordsize()) return C_KZG_BADARGS;

    /* wbits */
    memcpy(&out->wbits, data + offset, sizeof(size_t));
    offset += sizeof(size_t);
    /* scratch_size */
    memcpy(&out->scratch_size, data + offset, sizeof(size_t));
    offset += sizeof(size_t);
    /* table_size  */
    memcpy(&out->table_size, data + offset, sizeof(size_t));
    offset += sizeof(size_t);
    /* roots_of_unity  */
    ret = new_fr_array(&out->roots_of_unity, FEPEB1);
    if (ret != C_KZG_OK) goto out_error;
    memcpy(out->roots_of_unity, data + offset, FEPEB1 * sizeof(fr_t));
    offset += FEPEB1 * sizeof(fr_t);
    /* brp_roots_of_unity  */
    ret = new_fr_array(&out->brp_roots_of_unity, FEPEB0);
    if (ret != C_KZG_OK) goto out_error;
    memcpy(out->brp_roots_of_unity, data + offset, FEPEB0 * sizeof(fr_t));
    offset += FEPEB0 * sizeof(fr_t);
    /* reverse_roots_of_unity  */
    ret = new_fr_array(&out->reverse_roots_of_unity, FEPEB1);
    if (ret != C_KZG_OK) goto out_error;
    memcpy(out->reverse_roots_of_unity, data + offset, FEPEB1 * sizeof(fr_t));
    offset += FEPEB1 * sizeof(fr_t);
    /* g1_values_monomial  */
    ret = new_g1_array(&out->g1_values_monomial, NUM_G1_POINTS);
    if (ret != C_KZG_OK) goto out_error;
    memcpy(out->g1_values_monomial, data + offset, NUM_G1_POINTS * sizeof(g1_t));
    offset += NUM_G1_POINTS * sizeof(g1_t);
    /* g1_values_lagrange_brp  */
    ret = new_g1_array(&out->g1_values_lagrange_brp, NUM_G1_POINTS);
    if (ret != C_KZG_OK) goto out_error;
    memcpy(out->g1_values_lagrange_brp, data + offset, NUM_G1_POINTS * sizeof(g1_t));
    offset += NUM_G1_POINTS * sizeof(g1_t);
    /* g2_values_monomial  */
    ret = new_g2_array(&out->g2_values_monomial, NUM_G2_POINTS);
    if (ret != C_KZG_OK) goto out_error;
    memcpy(out->g2_values_monomial, data + offset, NUM_G2_POINTS * sizeof(g2_t));
    offset += NUM_G2_POINTS * sizeof(g2_t);
    /* x_ext_fft_columns  */
    ret = c_kzg_calloc((void **)&out->x_ext_fft_columns, CELLS_PER_EXT_BLOB, sizeof(g1_t *));
    if (ret != C_KZG_OK) goto out_error;
    for (size_t i = 0; i < CELLS_PER_EXT_BLOB; i++) {
        ret = new_g1_array(&out->x_ext_fft_columns[i], FIELD_ELEMENTS_PER_CELL);
        if (ret != C_KZG_OK) goto out_error;
        memcpy(out->x_ext_fft_columns[i], data + offset, FIELD_ELEMENTS_PER_CELL * sizeof(g1_t));
        offset += FIELD_ELEMENTS_PER_CELL * sizeof(g1_t);
    }
    /* tables  */
    if (out->wbits != 0) {
        ret = c_kzg_calloc((void **)&out->tables, CELLS_PER_EXT_BLOB, sizeof(uint8_t *));
        if (ret != C_KZG_OK) goto out_error;
        for (size_t i = 0; i < CELLS_PER_EXT_BLOB; i++) {
            ret = c_kzg_malloc((void **)&out->tables[i], out->table_size);
            if (ret != C_KZG_OK) goto out_error;
            memcpy(out->tables[i], data + offset, out->table_size);
            offset += out->table_size;
        }
    }

    goto out_success;

out_error:
    free_settings(out);

out_success:
    return ret;
}
