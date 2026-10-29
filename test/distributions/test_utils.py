# Owner(s): ["module: distributions"]
# [MODULE DOC] pytest test for tril_matrix_to_vec and vec_to_tril_matrix in
# torch.distributions.utils. Runs the common test runner when executed directly.

import pytest

import torch
from torch.distributions.utils import tril_matrix_to_vec, vec_to_tril_matrix
from torch.testing._internal.common_utils import run_tests


@pytest.mark.parametrize(
    "shape",
    [
        (2, 2),
        (3, 3),
        (2, 4, 4),
        (2, 2, 4, 4),
    ],
)
def test_tril_matrix_to_vec(shape):
    # [FUNCTION DOC] For random matrices of several shapes and each diagonal offset from minus n
    # to n minus 1, check that packing the lower triangle into a vector and unpacking it recovers
    # the tril-masked matrix. Raises AssertionError otherwise.
    mat = torch.randn(shape)
    n = mat.shape[-1]
    for diag in range(-n, n):
        actual = mat.tril(diag)
        vec = tril_matrix_to_vec(actual, diag)
        tril_mat = vec_to_tril_matrix(vec, diag)
        if not torch.allclose(tril_mat, actual):
            raise AssertionError("Expected tril_mat and actual to be close")


if __name__ == "__main__":
    run_tests()
