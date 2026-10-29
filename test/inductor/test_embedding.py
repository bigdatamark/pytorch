# Owner(s): ["module: inductor"]
# [MODULE DOC] Inductor tests for embedding lowering: out-of-range index handling in compiled
# embedding, negative-index behavior of the dense backward, and shape broadcasting of block-size
# strings. Runs under the Inductor TestCase runner; no import-time side effects beyond imports.

import torch
from torch._inductor import config
from torch._inductor.shape_propagation import get_broadcasted_shape
from torch._inductor.test_case import run_tests, TestCase


class TestEmbedding(TestCase):
    # [CLASS DOC] Holds three small tests of embedding-related Inductor behavior; no state.
    @config.patch({"assert_indirect_indexing": True})
    def test_embedding_negative_index_errors(self):
        # [FUNCTION DOC] Checks that with the assert_indirect_indexing config enabled, a
        # negative index into an embedding raises in both modes. Eager execution must raise an
        # IndexError mentioning index out of range, while the Inductor-compiled function must
        # raise a RuntimeError mentioning index out of bounds.
        def fn(weight, indices):
            # [FUNCTION DOC] HELPER: Wrapper that calls the aten embedding op on a weight table
            # and an index tensor; used both eagerly and compiled.
            return torch.embedding(weight, indices)

        weight = torch.arange(8, dtype=torch.float32).reshape(8, 1)
        indices = torch.tensor([-1, 0])
        opt_fn = torch.compile(fn, backend="inductor")

        with self.assertRaisesRegex(IndexError, "index out of range"):
            fn(weight, indices)
        with self.assertRaisesRegex(RuntimeError, "index out of bounds"):
            opt_fn(weight, indices)

    def test_embedding_dense_backward_negative_index_no_wrap(self):
        # [FUNCTION DOC] Checks that the compiled dense backward of embedding does not wrap
        # negative indices. Uses a gradient of ones for 8 rows, indices from -4 to 3, 8 weight
        # rows and padding index -1. Compiled output must equal eager output and equal four rows
        # of ones followed by four rows of zeros.
        def fn(grad_output, indices):
            # [FUNCTION DOC] HELPER: Calls the aten embedding dense backward op with fixed
            # num_weights 8, padding index -1 and no frequency scaling.
            return torch.ops.aten.embedding_dense_backward.default(
                grad_output, indices, 8, -1, False
            )

        grad_output = torch.ones(8, 1)
        indices = torch.arange(8) - 4
        opt_fn = torch.compile(fn, backend="inductor")

        expected = fn(grad_output, indices)
        actual = opt_fn(grad_output, indices)

        self.assertEqual(expected, actual)
        self.assertEqual(actual, torch.cat((torch.ones(4, 1), torch.zeros(4, 1))))

    def test_embedding_block_shape_assert_accepts_list_shape(self):
        # [FUNCTION DOC] Checks that the shape-broadcast helper accepts a list-typed shape of
        # block-size strings against a tuple shape, returning the broadcast tuple of the
        # symbolic block names (size-one dims take the other side's name).
        self.assertEqual(
            get_broadcasted_shape(["XBLOCK", "1"], ("1", "RBLOCK")),
            ("XBLOCK", "RBLOCK"),
        )


if __name__ == "__main__":
    run_tests()
