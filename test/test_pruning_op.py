# Owner(s): ["module: unknown"]
# [MODULE DOC] Tests for the torch._rowwise_prune operator: for random embedding weights of
# several dtypes and a random row mask, compares the operator against a pure-Python reference.
# Uses hypothesis for input generation; calls hu.assert_deadline_disabled at import time.

import hypothesis.strategies as st
from hypothesis import given
import numpy as np
import torch
from torch.testing._internal.common_utils import TestCase, run_tests, skipIfTorchDynamo
import torch.testing._internal.hypothesis_utils as hu
hu.assert_deadline_disabled()


class PruningOpTest(TestCase):
    # [CLASS DOC] Hypothesis-driven test case for torch._rowwise_prune with 32-bit and 64-bit index
    # outputs. Holds a mask generator and a shared check helper used by the two test methods.

    # Generate rowwise mask vector based on indicator and threshold value.
    # indicator is a vector that contains one value per weight row and it
    # represents the importance of a row.
    # We mask a row if its indicator value is less than the threshold.
    def _generate_rowwise_mask(self, embedding_rows):
        # [FUNCTION DOC] HELPER: Builds a random boolean row mask of length embedding_rows. Draws
        # one random float per row as an importance indicator and a random threshold; a row is kept
        # (True) when its indicator is at least the threshold. Returns the BoolTensor mask.
        indicator = torch.from_numpy((np.random.random_sample(embedding_rows)).astype(np.float32))
        threshold = float(np.random.random_sample())
        mask = torch.BoolTensor([val >= threshold for val in indicator])
        return mask

    def _test_rowwise_prune_op(self, embedding_rows, embedding_dims, indices_type, weights_dtype):
        # [FUNCTION DOC] HELPER: Runs torch._rowwise_prune on random weights of shape (rows, dims)
        # and compares with a reference. Integer dtypes get random ints in [0, 100); others get
        # uniform random values. Asserts the pruned weights are close, the compressed index map
        # equals the reference (kept rows numbered consecutively, pruned rows -1), and that its
        # dtype is indices_type.
        embedding_weights = None
        if weights_dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            embedding_weights = torch.randint(0, 100, (embedding_rows, embedding_dims), dtype=weights_dtype)
        else:
            embedding_weights = torch.rand((embedding_rows, embedding_dims), dtype=weights_dtype)
        mask = self._generate_rowwise_mask(embedding_rows)

        def get_pt_result(embedding_weights, mask, indices_type):
            # [FUNCTION DOC] HELPER: Calls torch._rowwise_prune on the weights and mask with the
            # requested index dtype and returns its pair (pruned weights, compressed index map).
            return torch._rowwise_prune(embedding_weights, mask, indices_type)

        # Reference implementation.
        def get_reference_result(embedding_weights, mask, indices_type):
            # [FUNCTION DOC] HELPER: Reference implementation. Selects the masked-in rows as the
            # pruned weights, and builds an index map of the given dtype holding the running count
            # of kept rows for kept positions and -1 for pruned ones. Returns a pair of the pruned
            # weights and the index map.
            num_embeddings = mask.size()[0]
            compressed_idx_out = torch.zeros(num_embeddings, dtype=indices_type)
            pruned_weights_out = embedding_weights[mask[:]]
            idx = 0
            for i in range(mask.size()[0]):
                if mask[i]:
                    compressed_idx_out[i] = idx
                    idx = idx + 1
                else:
                    compressed_idx_out[i] = -1
            return (pruned_weights_out, compressed_idx_out)

        pt_pruned_weights, pt_compressed_indices_map = get_pt_result(
            embedding_weights, mask, indices_type)
        ref_pruned_weights, ref_compressed_indices_map = get_reference_result(
            embedding_weights, mask, indices_type)

        torch.testing.assert_close(pt_pruned_weights, ref_pruned_weights)
        self.assertEqual(pt_compressed_indices_map, ref_compressed_indices_map)
        self.assertEqual(pt_compressed_indices_map.dtype, indices_type)


    @skipIfTorchDynamo()
    @given(
        embedding_rows=st.integers(1, 100),
        embedding_dims=st.integers(1, 100),
        weights_dtype=st.sampled_from([torch.float64, torch.float32,
                                       torch.float16, torch.int8,
                                       torch.int16, torch.int32, torch.int64])
    )
    def test_rowwise_prune_op_32bit_indices(self, embedding_rows, embedding_dims, weights_dtype):
        # [FUNCTION DOC] Hypothesis test: rowwise pruning with int32 index output across random
        # sizes (1 to 100 rows and dims) and several float and integer weight dtypes. Skipped under
        # Dynamo.
        self._test_rowwise_prune_op(embedding_rows, embedding_dims, torch.int, weights_dtype)


    @skipIfTorchDynamo()
    @given(
        embedding_rows=st.integers(1, 100),
        embedding_dims=st.integers(1, 100),
        weights_dtype=st.sampled_from([torch.float64, torch.float32,
                                       torch.float16, torch.int8,
                                       torch.int16, torch.int32, torch.int64])
    )
    def test_rowwise_prune_op_64bit_indices(self, embedding_rows, embedding_dims, weights_dtype):
        # [FUNCTION DOC] Hypothesis test: rowwise pruning with int64 index output across random
        # sizes (1 to 100 rows and dims) and several float and integer weight dtypes. Skipped under
        # Dynamo.
        self._test_rowwise_prune_op(embedding_rows, embedding_dims, torch.int64, weights_dtype)


if __name__ == '__main__':
    run_tests()
