# Owner(s): ["module: dsl-native-ops"]

from __future__ import annotations
# [MODULE DOC] Unit tests for the pure-Python inner-tree sum planning helpers in
# torch._native.ops.sum.inner_tree_plan: vec_size and compute_inner_tree_params, plus a check
# that importing that module does not pull in cutlass. Runs via run_tests when executed
# directly.

import sys

from torch._native.ops.sum.inner_tree_plan import (
    compute_inner_tree_params,
    InnerTreeParams,
    vec_size,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestInnerTreePlan(TestCase):
    # [CLASS DOC] Test case holding parametrized tests of the inner-tree plan helpers.
    # Parametrization is expanded by instantiate_parametrized_tests called right after the class
    # definition.
    @parametrize("itemsize, expected", [(2, 8), (4, 4), (8, 2)])
    def test_vec_size(self, itemsize: int, expected: int) -> None:
        # [FUNCTION DOC] Checks vec_size maps item size in bytes to elements per 16-byte vector
        # load: 2 gives 8, 4 gives 4, 8 gives 2 (parametrized).
        self.assertEqual(vec_size(itemsize), expected)

    @parametrize(
        "inputs_per_output, num_outputs, vector_size, expected",
        [
            (512, 32, 4, InnerTreeParams(4, 512, 1, 1, 2, 1)),
            (8192, 32, 4, InnerTreeParams(16, 8192, 1, 2, 1, 4)),
            (8193, 32, 4, InnerTreeParams(8, 8192, 2, 3, 1, 8)),
            (32769, 32, 4, InnerTreeParams(16, 8192, 5, 2, 1, 4)),
        ],
    )
    def test_compute_inner_tree_params(
        self,
        inputs_per_output: int,
        num_outputs: int,
        vector_size: int,
        expected: InnerTreeParams,
    ) -> None:
        # [FUNCTION DOC] Checks compute_inner_tree_params returns the expected InnerTreeParams
        # tuple (warps, batch elements, batches, depth, rows per block, effective loads) for
        # four parametrized sizes: 512, 8192, 8193 and 32769 inputs per output, with 32 outputs
        # and vector size 4 (batch counts 1, 1, 2, 5).
        self.assertEqual(
            compute_inner_tree_params(inputs_per_output, num_outputs, vector_size),
            expected,
        )

    def test_import_does_not_load_cutlass(self) -> None:
        # [FUNCTION DOC] Asserts that cutlass is not present among the already imported modules,
        # i.e. importing the planning module (done at the top of this test file) must not import
        # cutlass as a side effect.
        self.assertNotIn("cutlass", sys.modules)


instantiate_parametrized_tests(TestInnerTreePlan)


if __name__ == "__main__":
    run_tests()
