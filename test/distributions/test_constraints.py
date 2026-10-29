# Owner(s): ["module: distributions"]
# [MODULE DOC] pytest tests for torch.distributions.constraints. EXAMPLES lists matrix
# constraints with expected check results; CONSTRAINTS lists constraint objects/factories
# with their arguments. Tests check constraint.check, biject_to and transform_to on CPU and
# (when CUDA is present) GPU. No import-time side effects beyond building the lists.

import pytest

import torch
from torch.distributions import biject_to, constraints, transform_to
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests


EXAMPLES = [
    (constraints.symmetric, False, [[2.0, 0], [2.0, 2]]),
    (constraints.positive_semidefinite, False, [[2.0, 0], [2.0, 2]]),
    (constraints.positive_definite, False, [[2.0, 0], [2.0, 2]]),
    (constraints.symmetric, True, [[3.0, -5], [-5.0, 3]]),
    (constraints.positive_semidefinite, False, [[3.0, -5], [-5.0, 3]]),
    (constraints.positive_definite, False, [[3.0, -5], [-5.0, 3]]),
    (constraints.symmetric, True, [[1.0, 2], [2.0, 4]]),
    (constraints.positive_semidefinite, True, [[1.0, 2], [2.0, 4]]),
    (constraints.positive_definite, False, [[1.0, 2], [2.0, 4]]),
    (constraints.symmetric, True, [[[1.0, -2], [-2.0, 1]], [[2.0, 3], [3.0, 2]]]),
    (
        constraints.positive_semidefinite,
        False,
        [[[1.0, -2], [-2.0, 1]], [[2.0, 3], [3.0, 2]]],
    ),
    (
        constraints.positive_definite,
        False,
        [[[1.0, -2], [-2.0, 1]], [[2.0, 3], [3.0, 2]]],
    ),
    (constraints.symmetric, True, [[[1.0, -2], [-2.0, 4]], [[1.0, -1], [-1.0, 1]]]),
    (
        constraints.positive_semidefinite,
        True,
        [[[1.0, -2], [-2.0, 4]], [[1.0, -1], [-1.0, 1]]],
    ),
    (
        constraints.positive_definite,
        False,
        [[[1.0, -2], [-2.0, 4]], [[1.0, -1], [-1.0, 1]]],
    ),
    (constraints.symmetric, True, [[[4.0, 2], [2.0, 4]], [[3.0, -1], [-1.0, 3]]]),
    (
        constraints.positive_semidefinite,
        True,
        [[[4.0, 2], [2.0, 4]], [[3.0, -1], [-1.0, 3]]],
    ),
    (
        constraints.positive_definite,
        True,
        [[[4.0, 2], [2.0, 4]], [[3.0, -1], [-1.0, 3]]],
    ),
]

CONSTRAINTS = [
    (constraints.real,),
    (constraints.real_vector,),
    (constraints.positive,),
    (constraints.greater_than, [-10.0, -2, 0, 2, 10]),
    (constraints.greater_than, 0),
    (constraints.greater_than, 2),
    (constraints.greater_than, -2),
    (constraints.greater_than_eq, 0),
    (constraints.greater_than_eq, 2),
    (constraints.greater_than_eq, -2),
    (constraints.less_than, [-10.0, -2, 0, 2, 10]),
    (constraints.less_than, 0),
    (constraints.less_than, 2),
    (constraints.less_than, -2),
    (constraints.unit_interval,),
    (constraints.interval, [-4.0, -2, 0, 2, 4], [-3.0, 3, 1, 5, 5]),
    (constraints.interval, -2, -1),
    (constraints.interval, 1, 2),
    (constraints.half_open_interval, [-4.0, -2, 0, 2, 4], [-3.0, 3, 1, 5, 5]),
    (constraints.half_open_interval, -2, -1),
    (constraints.half_open_interval, 1, 2),
    (constraints.simplex,),
    (constraints.corr_cholesky,),
    (constraints.lower_cholesky,),
    (constraints.positive_definite,),
]


def build_constraint(constraint_fn, args, is_cuda=False):
    # [FUNCTION DOC] HELPER: Instantiates a constraint. With empty args the constraint_fn is
    # returned as is (it is already a constraint instance). Otherwise each list argument is
    # converted to a double tensor (CUDA if is_cuda) and non-list arguments pass through
    # unchanged, then constraint_fn is called with them and the result returned.
    if not args:
        return constraint_fn
    t = torch.cuda.DoubleTensor if is_cuda else torch.DoubleTensor
    return constraint_fn(*(t(x) if isinstance(x, list) else x for x in args))


@pytest.mark.parametrize(("constraint_fn", "result", "value"), EXAMPLES)
@pytest.mark.parametrize(
    "is_cuda",
    [
        False,
        pytest.param(
            True, marks=pytest.mark.skipif(not TEST_CUDA, reason="CUDA not found.")
        ),
    ],
)
def test_constraint(constraint_fn, result, value, is_cuda):
    # [FUNCTION DOC] Parametrized over EXAMPLES and CPU/CUDA. Converts value to a double
    # tensor and raises AssertionError if the all-reduced result of constraint_fn.check
    # differs from the expected boolean result.
    t = torch.cuda.DoubleTensor if is_cuda else torch.DoubleTensor
    if constraint_fn.check(t(value)).all() != result:
        raise AssertionError(
            f"Expected {result}, got {constraint_fn.check(t(value)).all()}"
        )


@pytest.mark.parametrize(
    ("constraint_fn", "args"), [(c[0], c[1:]) for c in CONSTRAINTS]
)
@pytest.mark.parametrize(
    "is_cuda",
    [
        False,
        pytest.param(
            True, marks=pytest.mark.skipif(not TEST_CUDA, reason="CUDA not found.")
        ),
    ],
)
def test_biject_to(constraint_fn, args, is_cuda):
    # [FUNCTION DOC] Parametrized over CONSTRAINTS and CPU/CUDA. Builds the constraint and
    # calls biject_to; skips the test if that raises NotImplementedError. Raises
    # AssertionError unless the transform is bijective, maps random double input (6x6 for
    # corr_cholesky, else 5x5) into the constraint, is inverted by t.inv (allclose), and
    # log_abs_det_jacobian has the batch shape of the input (event dims of the domain removed).
    constraint = build_constraint(constraint_fn, args, is_cuda=is_cuda)
    try:
        t = biject_to(constraint)
    except NotImplementedError:
        pytest.skip("`biject_to` not implemented.")
    if not t.bijective:
        raise AssertionError(f"biject_to({constraint}) is not bijective")
    if constraint_fn is constraints.corr_cholesky:
        # (D * (D-1)) / 2 (where D = 4) = 6 (size of last dim)
        x = torch.randn(6, 6, dtype=torch.double)
    else:
        x = torch.randn(5, 5, dtype=torch.double)
    if is_cuda:
        x = x.cuda()
    y = t(x)
    if not constraint.check(y).all():
        raise AssertionError(
            "\n".join(
                [
                    f"Failed to biject_to({constraint})",
                    f"x = {x}",
                    f"biject_to(...)(x) = {y}",
                ]
            )
        )
    x2 = t.inv(y)
    if not torch.allclose(x, x2):
        raise AssertionError(f"Error in biject_to({constraint}) inverse")

    j = t.log_abs_det_jacobian(x, y)
    if j.shape != x.shape[: x.dim() - t.domain.event_dim]:
        raise AssertionError(
            f"Expected shape {x.shape[: x.dim() - t.domain.event_dim]}, got {j.shape}"
        )


@pytest.mark.parametrize(
    ("constraint_fn", "args"), [(c[0], c[1:]) for c in CONSTRAINTS]
)
@pytest.mark.parametrize(
    "is_cuda",
    [
        False,
        pytest.param(
            True, marks=pytest.mark.skipif(not TEST_CUDA, reason="CUDA not found.")
        ),
    ],
)
def test_transform_to(constraint_fn, args, is_cuda):
    # [FUNCTION DOC] Parametrized over CONSTRAINTS and CPU/CUDA. Builds the constraint, gets
    # transform_to(constraint), applies it to random double input (6x6 for corr_cholesky,
    # else 5x5) and raises AssertionError if the output violates the constraint or if
    # applying the transform to the pseudo-inverse of the output does not reproduce it.
    constraint = build_constraint(constraint_fn, args, is_cuda=is_cuda)
    t = transform_to(constraint)
    if constraint_fn is constraints.corr_cholesky:
        # (D * (D-1)) / 2 (where D = 4) = 6 (size of last dim)
        x = torch.randn(6, 6, dtype=torch.double)
    else:
        x = torch.randn(5, 5, dtype=torch.double)
    if is_cuda:
        x = x.cuda()
    y = t(x)
    if not constraint.check(y).all():
        raise AssertionError(f"Failed to transform_to({constraint})")
    x2 = t.inv(y)
    y2 = t(x2)
    if not torch.allclose(y, y2):
        raise AssertionError(f"Error in transform_to({constraint}) pseudoinverse")


if __name__ == "__main__":
    run_tests()
