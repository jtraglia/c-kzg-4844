    # Valid: Large number of commitments, one point-at-infinity
    if True:
        for count in [2, 4, 8, 16, 32, 64, 128]:

            def get_inputs(count=count):
                commitments = []
                cell_indices = []
                cells = []
                proofs = []

                for i in range(count):
                    if i == 0:
                        # The all zeros blob
                        blob = VALID_BLOBS[0]
                    else:
                        blob = get_random_blob(i)
                    blob_commitment = cached_blob_to_kzg_commitment(blob)
                    commitments.extend([blob_commitment] * spec.CELLS_PER_EXT_BLOB)
                    cell_indices.extend(list(range(spec.CELLS_PER_EXT_BLOB)))
                    blob_cells, blob_proofs = cached_compute_cells_and_kzg_proofs(blob)
                    cells.extend(blob_cells)
                    proofs.extend(blob_proofs)

                return commitments, cell_indices, cells, proofs

            yield (
                f"verify_cell_kzg_proof_batch_case_valid_many_{count}",
                get_test_runner(get_inputs),
            )
