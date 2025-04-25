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

#include "eip4844/blob.h"
#include "setup/settings.h"

#include <inttypes.h> /* For uint8_t */

////////////////////////////////////////////////////////////////////////////////////////////////////
// Macros
////////////////////////////////////////////////////////////////////////////////////////////////////

/** The maximum number of field elements in a cell. */
#define MAX_FIELD_ELEMENTS_PER_CELL 64

////////////////////////////////////////////////////////////////////////////////////////////////////
// Types
////////////////////////////////////////////////////////////////////////////////////////////////////

/** A single cell for a blob. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wzero-length-array"
typedef struct {
    /*
     * This is a zero-length array because we cannot use flexible-array,
     * as it cannot be the only field in a structure. For these to be
     * stored as a flat, contiguous array, we must not include
     * other data in the structure.
     */
    uint8_t bytes[0];
} Cell;
#pragma GCC diagnostic pop

////////////////////////////////////////////////////////////////////////////////////////////////////
// Public Functions
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
extern "C" {
#endif

const Cell *cell_at(const Cell *cells, size_t index, const KZGSettings *s);
Cell *mut_cell_at(Cell *cells, size_t index, const KZGSettings *s);
void print_cell(const Cell *cell, const KZGSettings *s);

#ifdef __cplusplus
}
#endif
