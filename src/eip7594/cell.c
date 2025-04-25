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

#include "eip7594/cell.h"
#include "common/bytes.h"
#include "setup/settings.h"

#include <stdio.h> /* For printf */

/**
 * Get the cell at a specific index in an array of cells.
 *
 * @param[in]   cells   The array of cells to take from
 * @param[in]   index   The index into the array
 * @param[in]   s       The trusted setup
 */
const Cell *cell_at(const Cell *cells, size_t index, const KZGSettings *s) {
    return (const Cell *)((const uint8_t *)cells + index * s->bytes_per_cell);
}

/**
 * Get the cell at a specific index in an array of cells.
 *
 * @param[in]   cells   The array of cells to take from
 * @param[in]   index   The index into the array
 * @param[in]   s       The trusted setup
 *
 * @remark This version returns a mutable cell.
 */
Cell *mut_cell_at(Cell *cells, size_t index, const KZGSettings *s) {
    return (Cell *)((uint8_t *)cells + index * s->bytes_per_cell);
}

/**
 * Print Cell to the console.
 *
 * @param[in]   cell    The Cell to print
 */
void print_cell(const Cell *cell, const KZGSettings *s) {
    for (size_t i = 0; i < s->field_elements_per_cell; i++) {
        const Bytes32 *element_bytes = (const Bytes32 *)&cell[i * BYTES_PER_FIELD_ELEMENT];
        print_bytes32(element_bytes);
    }
}
