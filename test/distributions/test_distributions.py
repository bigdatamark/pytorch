# Owner(s): ["module: distributions"]
# [MODULE DOC] Unit tests for torch.distributions. Defines the registry of example and invalid-
# parameter configurations for every distribution (_get_examples and _get_bad_examples) and
# several test classes: TestDistributions (device-generic via instantiate_device_type_tests at
# the bottom, CUDA/XPU excluded), TestDistributionsGPU, TestRsample, TestDistributionShapes,
# TestKL, TestConstraints, TestNumericalStability, TestLazyLogitsInitialization,
# TestAgainstScipy, TestFunctors, TestValidation, TestJit. Many checks compare against
# scipy.stats and are skipped when NumPy/SciPy are missing (TEST_NUMPY). Import time: computes
# device_type from the current accelerator and tries to import numpy/scipy. Running as a script
# enables the default-dtype check and calls run_tests only when LAPACK is available.

"""
Note [Randomized statistical tests]
-----------------------------------

This note describes how to maintain tests in this file as random sources
change. This file contains two types of randomized tests:

1. The easier type of randomized test are tests that should always pass but are
   initialized with random data. If these fail something is wrong, but it's
   fine to use a fixed seed by inheriting from common.TestCase.

2. The trickier tests are statistical tests. These tests explicitly call
   set_rng_seed(n) and are marked "see Note [Randomized statistical tests]".
   These statistical tests have a known positive failure rate
   (we set failure_rate=1e-3 by default). We need to balance strength of these
   tests with annoyance of false alarms. One way that works is to specifically
   set seeds in each of the randomized tests. When a random generator
   occasionally changes (as in #4312 vectorizing the Box-Muller sampler), some
   of these statistical tests may (rarely) fail. If one fails in this case,
   it's fine to increment the seed of the failing test (but you shouldn't need
   to increment it more than once; otherwise something is probably actually
   wrong).

3. `test_geometric_sample`, `test_binomial_sample` and `test_poisson_sample`
   are validated against `scipy.stats.` which are not guaranteed to be identical
   across different versions of scipy (namely, they yield invalid results in 1.7+)
"""

import math
import numbers
import unittest
from collections import namedtuple
from itertools import product
from random import shuffle

from packaging import version

import torch
import torch.autograd.forward_ad as fwAD
from torch import inf, nan
from torch.autograd import grad
from torch.autograd.functional import jacobian
from torch.distributions import (
    Bernoulli,
    Beta,
    Binomial,
    Categorical,
    Cauchy,
    Chi2,
    constraints,
    ContinuousBernoulli,
    Dirichlet,
    Distribution,
    Exponential,
    ExponentialFamily,
    FisherSnedecor,
    Gamma,
    GeneralizedPareto,
    Geometric,
    Gumbel,
    HalfCauchy,
    HalfNormal,
    Independent,
    InverseGamma,
    kl_divergence,
    Kumaraswamy,
    Laplace,
    LKJCholesky,
    LogisticNormal,
    LogNormal,
    LowRankMultivariateNormal,
    MixtureSameFamily,
    Multinomial,
    MultivariateNormal,
    NegativeBinomial,
    Normal,
    OneHotCategorical,
    OneHotCategoricalStraightThrough,
    Pareto,
    Poisson,
    RelaxedBernoulli,
    RelaxedOneHotCategorical,
    StudentT,
    TransformedDistribution,
    Uniform,
    VonMises,
    Weibull,
    Wishart,
)
from torch.distributions.constraint_registry import transform_to
from torch.distributions.constraints import Constraint, is_dependent
from torch.distributions.dirichlet import _Dirichlet_backward
from torch.distributions.kl import _kl_expfamily_expfamily
from torch.distributions.transforms import (
    AffineTransform,
    CatTransform,
    ExpTransform,
    identity_transform,
    StackTransform,
)
from torch.distributions.utils import (
    lazy_property,
    probs_to_logits,
    tril_matrix_to_vec,
    vec_to_tril_matrix,
)
from torch.nn.functional import softmax
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCUDA,
    dtypesIfMPS,
    dtypesIfXPU,
    expectedFailureMPS,
    instantiate_device_type_tests,
    skipMPS,
)
from torch.testing._internal.common_utils import (
    gradcheck,
    load_tests,
    run_tests,
    set_default_dtype,
    set_default_dtype_if_supported,
    set_rng_seed,
    skipIfTorchDynamo,
    TEST_XPU,
    TestCase,
)


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

TEST_NUMPY = True
try:
    import numpy as np
    import scipy.special
    import scipy.stats
except ImportError:
    TEST_NUMPY = False


def pairwise(Dist, *params):
    # [FUNCTION DOC] HELPER: Builds two instances of Dist for pairwise comparison, e.g. for KL
    # tests. Each entry of params is a sequence; params1 has each sequence tiled len(p) times
    # as rows and params2 is its transpose, so the two distributions have batch shape (n, n)
    # and element (i, j) pairs parameter i of the first with parameter j of the second. Returns
    # the tuple.
    """
    Creates a pair of distributions `Dist` initialized to test each element of
    param with each other.
    """
    params1 = [torch.tensor([p] * len(p)) for p in params]
    params2 = [p.transpose(0, 1) for p in params1]
    return Dist(*params1), Dist(*params2)


def is_all_nan(tensor):
    # [FUNCTION DOC] HELPER: Returns a boolean scalar tensor that is true only when
    # every element of the tensor is NaN (uses self-inequality, so it is also true for
    # an empty tensor).
    """
    Checks if all entries of a tensor is nan.
    """
    return (tensor != tensor).all()


Example = namedtuple("Example", ["Dist", "params"])


# Register all distributions for generic tests by appending to this list.
# TODO: Add the ability to specify expected failures for specific devices. Every
# test that uses this generator expects all the examples to pass, but MPS does
# not yet fully support all of the distributions
def _get_examples():
    # [FUNCTION DOC] HELPER: Returns the registry of valid test configurations: a list of
    # Example named tuples, one per distribution class, whose params is a list of
    # keyword-argument dictionaries. Tensors are freshly random (randn-based) on each call, most
    # requiring grad, and cover scalar, vector, batched and extreme (1e-5 scale) parameters.
    # Generic tests iterate over this. A new distribution must be added here (test_has_examples
    # checks it). Every test using it expects all examples to pass on every device, so a
    # distribution unsupported on MPS cannot be listed per device (see the TODO above it).
    return [
        Example(
            Bernoulli,
            [
                {"probs": torch.tensor([0.7, 0.2, 0.4], requires_grad=True)},
                {"probs": torch.tensor([0.3], requires_grad=True)},
                {"probs": 0.3},
                {"logits": torch.tensor([0.0], requires_grad=True)},
            ],
        ),
        Example(
            Geometric,
            [
                {"probs": torch.tensor([0.7, 0.2, 0.4], requires_grad=True)},
                {"probs": torch.tensor([0.3], requires_grad=True)},
                {"probs": 0.3},
            ],
        ),
        Example(
            Beta,
            [
                {
                    "concentration1": torch.randn(2, 3).exp().requires_grad_(),
                    "concentration0": torch.randn(2, 3).exp().requires_grad_(),
                },
                {
                    "concentration1": torch.randn(4).exp().requires_grad_(),
                    "concentration0": torch.randn(4).exp().requires_grad_(),
                },
            ],
        ),
        Example(
            Categorical,
            [
                {
                    "probs": torch.tensor(
                        [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True
                    )
                },
                {"probs": torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)},
                {"logits": torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)},
            ],
        ),
        Example(
            Binomial,
            [
                {
                    "probs": torch.tensor(
                        [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True
                    ),
                    "total_count": 10,
                },
                {
                    "probs": torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True),
                    "total_count": 10,
                },
                {
                    "probs": torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True),
                    "total_count": torch.tensor([10]),
                },
                {
                    "probs": torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True),
                    "total_count": torch.tensor([10, 8]),
                },
                {
                    "probs": torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True),
                    "total_count": torch.tensor([[10.0, 8.0], [5.0, 3.0]]),
                },
                {
                    "probs": torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True),
                    "total_count": torch.tensor(0.0),
                },
            ],
        ),
        Example(
            NegativeBinomial,
            [
                {
                    "probs": torch.tensor(
                        [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True
                    ),
                    "total_count": 10,
                },
                {
                    "probs": torch.tensor([[0.9, 0.0], [0.0, 0.9]], requires_grad=True),
                    "total_count": 10,
                },
                {
                    "probs": torch.tensor([[0.9, 0.0], [0.0, 0.9]], requires_grad=True),
                    "total_count": torch.tensor([10]),
                },
                {
                    "probs": torch.tensor([[0.9, 0.0], [0.0, 0.9]], requires_grad=True),
                    "total_count": torch.tensor([10, 8]),
                },
                {
                    "probs": torch.tensor([[0.9, 0.0], [0.0, 0.9]], requires_grad=True),
                    "total_count": torch.tensor([[10.0, 8.0], [5.0, 3.0]]),
                },
                {
                    "probs": torch.tensor([[0.9, 0.0], [0.0, 0.9]], requires_grad=True),
                    "total_count": torch.tensor(0.0),
                },
            ],
        ),
        Example(
            Multinomial,
            [
                {
                    "probs": torch.tensor(
                        [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True
                    ),
                    "total_count": 10,
                },
                {
                    "probs": torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True),
                    "total_count": 10,
                },
            ],
        ),
        Example(
            Cauchy,
            [
                {"loc": 0.0, "scale": 1.0},
                {"loc": torch.tensor([0.0]), "scale": 1.0},
                {
                    "loc": torch.tensor([[0.0], [0.0]]),
                    "scale": torch.tensor([[1.0], [1.0]]),
                },
            ],
        ),
        Example(
            Chi2,
            [
                {"df": torch.randn(2, 3).exp().requires_grad_()},
                {"df": torch.randn(1).exp().requires_grad_()},
            ],
        ),
        Example(
            StudentT,
            [
                {"df": torch.randn(2, 3).exp().requires_grad_()},
                {"df": torch.randn(1).exp().requires_grad_()},
            ],
        ),
        Example(
            Dirichlet,
            [
                {"concentration": torch.randn(2, 3).exp().requires_grad_()},
                {"concentration": torch.randn(4).exp().requires_grad_()},
            ],
        ),
        Example(
            Exponential,
            [
                {"rate": torch.randn(5, 5).abs().requires_grad_()},
                {"rate": torch.randn(1).abs().requires_grad_()},
            ],
        ),
        Example(
            FisherSnedecor,
            [
                {
                    "df1": torch.randn(5, 5).abs().requires_grad_(),
                    "df2": torch.randn(5, 5).abs().requires_grad_(),
                },
                {
                    "df1": torch.randn(1).abs().requires_grad_(),
                    "df2": torch.randn(1).abs().requires_grad_(),
                },
                {
                    "df1": torch.tensor([1.0]),
                    "df2": 1.0,
                },
            ],
        ),
        Example(
            Gamma,
            [
                {
                    "concentration": torch.randn(2, 3).exp().requires_grad_(),
                    "rate": torch.randn(2, 3).exp().requires_grad_(),
                },
                {
                    "concentration": torch.randn(1).exp().requires_grad_(),
                    "rate": torch.randn(1).exp().requires_grad_(),
                },
            ],
        ),
        Example(
            Gumbel,
            [
                {
                    "loc": torch.randn(5, 5, requires_grad=True),
                    "scale": torch.randn(5, 5).abs().requires_grad_(),
                },
                {
                    "loc": torch.randn(1, requires_grad=True),
                    "scale": torch.randn(1).abs().requires_grad_(),
                },
            ],
        ),
        Example(HalfCauchy, [{"scale": 1.0}, {"scale": torch.tensor([[1.0], [1.0]])}]),
        Example(
            HalfNormal,
            [
                {"scale": torch.randn(5, 5).abs().requires_grad_()},
                {"scale": torch.randn(1).abs().requires_grad_()},
                {"scale": torch.tensor([1e-5, 1e-5], requires_grad=True)},
            ],
        ),
        Example(
            Independent,
            [
                {
                    "base_distribution": Normal(
                        torch.randn(2, 3, requires_grad=True),
                        torch.randn(2, 3).abs().requires_grad_(),
                    ),
                    "reinterpreted_batch_ndims": 0,
                },
                {
                    "base_distribution": Normal(
                        torch.randn(2, 3, requires_grad=True),
                        torch.randn(2, 3).abs().requires_grad_(),
                    ),
                    "reinterpreted_batch_ndims": 1,
                },
                {
                    "base_distribution": Normal(
                        torch.randn(2, 3, requires_grad=True),
                        torch.randn(2, 3).abs().requires_grad_(),
                    ),
                    "reinterpreted_batch_ndims": 2,
                },
                {
                    "base_distribution": Normal(
                        torch.randn(2, 3, 5, requires_grad=True),
                        torch.randn(2, 3, 5).abs().requires_grad_(),
                    ),
                    "reinterpreted_batch_ndims": 2,
                },
                {
                    "base_distribution": Normal(
                        torch.randn(2, 3, 5, requires_grad=True),
                        torch.randn(2, 3, 5).abs().requires_grad_(),
                    ),
                    "reinterpreted_batch_ndims": 3,
                },
            ],
        ),
        Example(
            Kumaraswamy,
            [
                {
                    "concentration1": torch.empty(2, 3).uniform_(1, 2).requires_grad_(),
                    "concentration0": torch.empty(2, 3).uniform_(1, 2).requires_grad_(),
                },
                {
                    "concentration1": torch.rand(4).uniform_(1, 2).requires_grad_(),
                    "concentration0": torch.rand(4).uniform_(1, 2).requires_grad_(),
                },
            ],
        ),
        Example(
            LKJCholesky,
            [
                {"dim": 2, "concentration": 0.5},
                {
                    "dim": 3,
                    "concentration": torch.tensor([0.5, 1.0, 2.0]),
                },
                {"dim": 100, "concentration": 4.0},
            ],
        ),
        Example(
            Laplace,
            [
                {
                    "loc": torch.randn(5, 5, requires_grad=True),
                    "scale": torch.randn(5, 5).abs().requires_grad_(),
                },
                {
                    "loc": torch.randn(1, requires_grad=True),
                    "scale": torch.randn(1).abs().requires_grad_(),
                },
                {
                    "loc": torch.tensor([1.0, 0.0], requires_grad=True),
                    "scale": torch.tensor([1e-5, 1e-5], requires_grad=True),
                },
            ],
        ),
        Example(
            LogNormal,
            [
                {
                    "loc": torch.randn(5, 5, requires_grad=True),
                    "scale": torch.randn(5, 5).abs().requires_grad_(),
                },
                {
                    "loc": torch.randn(1, requires_grad=True),
                    "scale": torch.randn(1).abs().requires_grad_(),
                },
                {
                    "loc": torch.tensor([1.0, 0.0], requires_grad=True),
                    "scale": torch.tensor([1e-5, 1e-5], requires_grad=True),
                },
            ],
        ),
        Example(
            LogisticNormal,
            [
                {
                    "loc": torch.randn(5, 5).requires_grad_(),
                    "scale": torch.randn(5, 5).abs().requires_grad_(),
                },
                {
                    "loc": torch.randn(1).requires_grad_(),
                    "scale": torch.randn(1).abs().requires_grad_(),
                },
                {
                    "loc": torch.tensor([1.0, 0.0], requires_grad=True),
                    "scale": torch.tensor([1e-5, 1e-5], requires_grad=True),
                },
            ],
        ),
        Example(
            LowRankMultivariateNormal,
            [
                {
                    "loc": torch.randn(5, 2, requires_grad=True),
                    "cov_factor": torch.randn(5, 2, 1, requires_grad=True),
                    "cov_diag": torch.tensor([2.0, 0.25], requires_grad=True),
                },
                {
                    "loc": torch.randn(4, 3, requires_grad=True),
                    "cov_factor": torch.randn(3, 2, requires_grad=True),
                    "cov_diag": torch.tensor([5.0, 1.5, 3.0], requires_grad=True),
                },
            ],
        ),
        Example(
            MultivariateNormal,
            [
                {
                    "loc": torch.randn(5, 2, requires_grad=True),
                    "covariance_matrix": torch.tensor(
                        [[2.0, 0.3], [0.3, 0.25]], requires_grad=True
                    ),
                },
                {
                    "loc": torch.randn(2, 3, requires_grad=True),
                    "precision_matrix": torch.tensor(
                        [[2.0, 0.1, 0.0], [0.1, 0.25, 0.0], [0.0, 0.0, 0.3]],
                        requires_grad=True,
                    ),
                },
                {
                    "loc": torch.randn(5, 3, 2, requires_grad=True),
                    "scale_tril": torch.tensor(
                        [
                            [[2.0, 0.0], [-0.5, 0.25]],
                            [[2.0, 0.0], [0.3, 0.25]],
                            [[5.0, 0.0], [-0.5, 1.5]],
                        ],
                        requires_grad=True,
                    ),
                },
                {
                    "loc": torch.tensor([1.0, -1.0]),
                    "covariance_matrix": torch.tensor([[5.0, -0.5], [-0.5, 1.5]]),
                },
            ],
        ),
        Example(
            Normal,
            [
                {
                    "loc": torch.randn(5, 5, requires_grad=True),
                    "scale": torch.randn(5, 5).abs().requires_grad_(),
                },
                {
                    "loc": torch.randn(1, requires_grad=True),
                    "scale": torch.randn(1).abs().requires_grad_(),
                },
                {
                    "loc": torch.tensor([1.0, 0.0], requires_grad=True),
                    "scale": torch.tensor([1e-5, 1e-5], requires_grad=True),
                },
            ],
        ),
        Example(
            OneHotCategorical,
            [
                {
                    "probs": torch.tensor(
                        [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True
                    )
                },
                {"probs": torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)},
                {"logits": torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)},
            ],
        ),
        Example(
            OneHotCategoricalStraightThrough,
            [
                {
                    "probs": torch.tensor(
                        [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True
                    )
                },
                {"probs": torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)},
                {"logits": torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)},
            ],
        ),
        Example(
            Pareto,
            [
                {"scale": 1.0, "alpha": 1.0},
                {
                    "scale": (torch.randn(5, 5).abs() + 0.1).requires_grad_(),
                    "alpha": (torch.randn(5, 5).abs() + 0.1).requires_grad_(),
                },
                {"scale": torch.tensor([1.0]), "alpha": 1.0},
            ],
        ),
        Example(
            Poisson,
            [
                {
                    "rate": torch.randn(5, 5).abs().requires_grad_(),
                },
                {
                    "rate": torch.randn(3).abs().requires_grad_(),
                },
                {
                    "rate": 0.2,
                },
                {
                    "rate": torch.tensor([0.0], requires_grad=True),
                },
                {
                    "rate": 0.0,
                },
            ],
        ),
        Example(
            RelaxedBernoulli,
            [
                {
                    "temperature": torch.tensor([0.5], requires_grad=True),
                    "probs": torch.tensor([0.7, 0.2, 0.4], requires_grad=True),
                },
                {
                    "temperature": torch.tensor([2.0]),
                    "probs": torch.tensor([0.3]),
                },
                {
                    "temperature": torch.tensor([7.2]),
                    "logits": torch.tensor([-2.0, 2.0, 1.0, 5.0]),
                },
            ],
        ),
        Example(
            RelaxedOneHotCategorical,
            [
                {
                    "temperature": torch.tensor([0.5], requires_grad=True),
                    "probs": torch.tensor(
                        [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]], requires_grad=True
                    ),
                },
                {
                    "temperature": torch.tensor([2.0]),
                    "probs": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                },
                {
                    "temperature": torch.tensor([7.2]),
                    "logits": torch.tensor([[-2.0, 2.0], [1.0, 5.0]]),
                },
            ],
        ),
        Example(
            TransformedDistribution,
            [
                {
                    "base_distribution": Normal(
                        torch.randn(2, 3, requires_grad=True),
                        torch.randn(2, 3).abs().requires_grad_(),
                    ),
                    "transforms": [],
                },
                {
                    "base_distribution": Normal(
                        torch.randn(2, 3, requires_grad=True),
                        torch.randn(2, 3).abs().requires_grad_(),
                    ),
                    "transforms": ExpTransform(),
                },
                {
                    "base_distribution": Normal(
                        torch.randn(2, 3, 5, requires_grad=True),
                        torch.randn(2, 3, 5).abs().requires_grad_(),
                    ),
                    "transforms": [
                        AffineTransform(torch.randn(3, 5), torch.randn(3, 5)),
                        ExpTransform(),
                    ],
                },
                {
                    "base_distribution": Normal(
                        torch.randn(2, 3, 5, requires_grad=True),
                        torch.randn(2, 3, 5).abs().requires_grad_(),
                    ),
                    "transforms": AffineTransform(1, 2),
                },
                {
                    "base_distribution": Uniform(
                        torch.tensor(1e8).log(), torch.tensor(1e10).log()
                    ),
                    "transforms": ExpTransform(),
                },
            ],
        ),
        Example(
            Uniform,
            [
                {
                    "low": torch.zeros(5, 5, requires_grad=True),
                    "high": torch.ones(5, 5, requires_grad=True),
                },
                {
                    "low": torch.zeros(1, requires_grad=True),
                    "high": torch.ones(1, requires_grad=True),
                },
                {
                    "low": torch.tensor([1.0, 1.0], requires_grad=True),
                    "high": torch.tensor([2.0, 3.0], requires_grad=True),
                },
            ],
        ),
        Example(
            Weibull,
            [
                {
                    "scale": torch.randn(5, 5).abs().requires_grad_(),
                    "concentration": torch.randn(1).abs().requires_grad_(),
                }
            ],
        ),
        Example(
            Wishart,
            [
                {
                    "covariance_matrix": torch.tensor(
                        [[2.0, 0.3], [0.3, 0.25]], requires_grad=True
                    ),
                    "df": torch.tensor([3.0], requires_grad=True),
                },
                {
                    "precision_matrix": torch.tensor(
                        [[2.0, 0.1, 0.0], [0.1, 0.25, 0.0], [0.0, 0.0, 0.3]],
                        requires_grad=True,
                    ),
                    "df": torch.tensor([5.0, 4], requires_grad=True),
                },
                {
                    "scale_tril": torch.tensor(
                        [
                            [[2.0, 0.0], [-0.5, 0.25]],
                            [[2.0, 0.0], [0.3, 0.25]],
                            [[5.0, 0.0], [-0.5, 1.5]],
                        ],
                        requires_grad=True,
                    ),
                    "df": torch.tensor([5.0, 3.5, 3], requires_grad=True),
                },
                {
                    "covariance_matrix": torch.tensor([[5.0, -0.5], [-0.5, 1.5]]),
                    "df": torch.tensor([3.0]),
                },
                {
                    "covariance_matrix": torch.tensor([[5.0, -0.5], [-0.5, 1.5]]),
                    "df": 3.0,
                },
            ],
        ),
        Example(
            MixtureSameFamily,
            [
                {
                    "mixture_distribution": Categorical(
                        torch.rand(5, requires_grad=True)
                    ),
                    "component_distribution": Normal(
                        torch.randn(5, requires_grad=True),
                        torch.rand(5, requires_grad=True),
                    ),
                },
                {
                    "mixture_distribution": Categorical(
                        torch.rand(5, requires_grad=True)
                    ),
                    "component_distribution": MultivariateNormal(
                        loc=torch.randn(5, 2, requires_grad=True),
                        covariance_matrix=torch.tensor(
                            [[2.0, 0.3], [0.3, 0.25]], requires_grad=True
                        ),
                    ),
                },
            ],
        ),
        Example(
            VonMises,
            [
                {
                    "loc": torch.tensor(1.0, requires_grad=True),
                    "concentration": torch.tensor(10.0, requires_grad=True),
                },
                {
                    "loc": torch.tensor([0.0, math.pi / 2], requires_grad=True),
                    "concentration": torch.tensor([1.0, 10.0], requires_grad=True),
                },
            ],
        ),
        Example(
            ContinuousBernoulli,
            [
                {"probs": torch.tensor([0.7, 0.2, 0.4], requires_grad=True)},
                {"probs": torch.tensor([0.3], requires_grad=True)},
                {"probs": 0.3},
                {"logits": torch.tensor([0.0], requires_grad=True)},
            ],
        ),
        Example(
            InverseGamma,
            [
                {
                    "concentration": torch.randn(2, 3).exp().requires_grad_(),
                    "rate": torch.randn(2, 3).exp().requires_grad_(),
                },
                {
                    "concentration": torch.randn(1).exp().requires_grad_(),
                    "rate": torch.randn(1).exp().requires_grad_(),
                },
            ],
        ),
        Example(
            GeneralizedPareto,
            [
                {
                    "loc": torch.randn(5, 5, requires_grad=True).mul(10),
                    "scale": torch.randn(5, 5).abs().requires_grad_(),
                    "concentration": torch.randn(5, 5).div(10).requires_grad_(),
                },
            ],
        ),
    ]


# Register all distributions for bad examples by appending to this list.
def _get_bad_examples():
    # [FUNCTION DOC] HELPER: Returns the registry of invalid configurations as Example tuples:
    # per distribution, keyword-argument dictionaries whose parameters violate the constraints
    # (negative or zero scale, probabilities outside [0, 1], non-positive-definite matrices,
    # non-callable transforms, etc.). Used by validation tests expecting ValueError/TypeError.
    return [
        Example(
            Bernoulli,
            [
                {"probs": torch.tensor([1.1, 0.2, 0.4], requires_grad=True)},
                {"probs": torch.tensor([-0.5], requires_grad=True)},
                {"probs": 1.00001},
            ],
        ),
        Example(
            Beta,
            [
                {
                    "concentration1": torch.tensor([0.0], requires_grad=True),
                    "concentration0": torch.tensor([0.0], requires_grad=True),
                },
                {
                    "concentration1": torch.tensor([-1.0], requires_grad=True),
                    "concentration0": torch.tensor([-2.0], requires_grad=True),
                },
            ],
        ),
        Example(
            Geometric,
            [
                {"probs": torch.tensor([1.1, 0.2, 0.4], requires_grad=True)},
                {"probs": torch.tensor([-0.3], requires_grad=True)},
                {"probs": 1.00000001},
            ],
        ),
        Example(
            Categorical,
            [
                {
                    "probs": torch.tensor(
                        [[-0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True
                    )
                },
                {
                    "probs": torch.tensor(
                        [[-1.0, 10.0], [0.0, -1.0]], requires_grad=True
                    )
                },
            ],
        ),
        Example(
            Binomial,
            [
                {
                    "probs": torch.tensor(
                        [[-0.0000001, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True
                    ),
                    "total_count": 10,
                },
                {
                    "probs": torch.tensor([[1.0, 0.0], [0.0, 2.0]], requires_grad=True),
                    "total_count": 10,
                },
            ],
        ),
        Example(
            NegativeBinomial,
            [
                {
                    "probs": torch.tensor(
                        [[-0.0000001, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True
                    ),
                    "total_count": 10,
                },
                {
                    "probs": torch.tensor([[1.0, 0.0], [0.0, 2.0]], requires_grad=True),
                    "total_count": 10,
                },
            ],
        ),
        Example(
            Cauchy,
            [
                {"loc": 0.0, "scale": -1.0},
                {"loc": torch.tensor([0.0]), "scale": 0.0},
                {
                    "loc": torch.tensor([[0.0], [-2.0]]),
                    "scale": torch.tensor([[-0.000001], [1.0]]),
                },
            ],
        ),
        Example(
            Chi2,
            [
                {"df": torch.tensor([0.0], requires_grad=True)},
                {"df": torch.tensor([-2.0], requires_grad=True)},
            ],
        ),
        Example(
            StudentT,
            [
                {"df": torch.tensor([0.0], requires_grad=True)},
                {"df": torch.tensor([-2.0], requires_grad=True)},
            ],
        ),
        Example(
            Dirichlet,
            [
                {"concentration": torch.tensor([0.0], requires_grad=True)},
                {"concentration": torch.tensor([-2.0], requires_grad=True)},
            ],
        ),
        Example(
            Exponential,
            [
                {"rate": torch.tensor([0.0, 0.0], requires_grad=True)},
                {"rate": torch.tensor([-2.0], requires_grad=True)},
            ],
        ),
        Example(
            FisherSnedecor,
            [
                {
                    "df1": torch.tensor([0.0, 0.0], requires_grad=True),
                    "df2": torch.tensor([-1.0, -100.0], requires_grad=True),
                },
                {
                    "df1": torch.tensor([1.0, 1.0], requires_grad=True),
                    "df2": torch.tensor([0.0, 0.0], requires_grad=True),
                },
            ],
        ),
        Example(
            Gamma,
            [
                {
                    "concentration": torch.tensor([0.0, 0.0], requires_grad=True),
                    "rate": torch.tensor([-1.0, -100.0], requires_grad=True),
                },
                {
                    "concentration": torch.tensor([1.0, 1.0], requires_grad=True),
                    "rate": torch.tensor([0.0, 0.0], requires_grad=True),
                },
            ],
        ),
        Example(
            Gumbel,
            [
                {
                    "loc": torch.tensor([1.0, 1.0], requires_grad=True),
                    "scale": torch.tensor([0.0, 1.0], requires_grad=True),
                },
                {
                    "loc": torch.tensor([1.0, 1.0], requires_grad=True),
                    "scale": torch.tensor([1.0, -1.0], requires_grad=True),
                },
            ],
        ),
        Example(
            HalfCauchy,
            [
                {"scale": -1.0},
                {"scale": 0.0},
                {"scale": torch.tensor([[-0.000001], [1.0]])},
            ],
        ),
        Example(
            HalfNormal,
            [
                {"scale": torch.tensor([0.0, 1.0], requires_grad=True)},
                {"scale": torch.tensor([1.0, -1.0], requires_grad=True)},
            ],
        ),
        Example(
            LKJCholesky,
            [
                {"dim": -2, "concentration": 0.1},
                {
                    "dim": 1,
                    "concentration": 2.0,
                },
                {
                    "dim": 2,
                    "concentration": 0.0,
                },
            ],
        ),
        Example(
            Laplace,
            [
                {
                    "loc": torch.tensor([1.0, 1.0], requires_grad=True),
                    "scale": torch.tensor([0.0, 1.0], requires_grad=True),
                },
                {
                    "loc": torch.tensor([1.0, 1.0], requires_grad=True),
                    "scale": torch.tensor([1.0, -1.0], requires_grad=True),
                },
            ],
        ),
        Example(
            LogNormal,
            [
                {
                    "loc": torch.tensor([1.0, 1.0], requires_grad=True),
                    "scale": torch.tensor([0.0, 1.0], requires_grad=True),
                },
                {
                    "loc": torch.tensor([1.0, 1.0], requires_grad=True),
                    "scale": torch.tensor([1.0, -1.0], requires_grad=True),
                },
            ],
        ),
        Example(
            MultivariateNormal,
            [
                {
                    "loc": torch.tensor([1.0, 1.0], requires_grad=True),
                    "covariance_matrix": torch.tensor(
                        [[1.0, 0.0], [0.0, -2.0]], requires_grad=True
                    ),
                },
            ],
        ),
        Example(
            Normal,
            [
                {
                    "loc": torch.tensor([1.0, 1.0], requires_grad=True),
                    "scale": torch.tensor([0.0, 1.0], requires_grad=True),
                },
                {
                    "loc": torch.tensor([1.0, 1.0], requires_grad=True),
                    "scale": torch.tensor([1.0, -1.0], requires_grad=True),
                },
                {
                    "loc": torch.tensor([1.0, 0.0], requires_grad=True),
                    "scale": torch.tensor([1e-5, -1e-5], requires_grad=True),
                },
            ],
        ),
        Example(
            OneHotCategorical,
            [
                {
                    "probs": torch.tensor(
                        [[0.1, 0.2, 0.3], [0.1, -10.0, 0.2]], requires_grad=True
                    )
                },
                {"probs": torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)},
            ],
        ),
        Example(
            OneHotCategoricalStraightThrough,
            [
                {
                    "probs": torch.tensor(
                        [[0.1, 0.2, 0.3], [0.1, -10.0, 0.2]], requires_grad=True
                    )
                },
                {"probs": torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)},
            ],
        ),
        Example(
            Pareto,
            [
                {"scale": 0.0, "alpha": 0.0},
                {
                    "scale": torch.tensor([0.0, 0.0], requires_grad=True),
                    "alpha": torch.tensor([-1e-5, 0.0], requires_grad=True),
                },
                {"scale": torch.tensor([1.0]), "alpha": -1.0},
            ],
        ),
        Example(
            Poisson,
            [
                {
                    "rate": torch.tensor([-0.1], requires_grad=True),
                },
                {
                    "rate": -1.0,
                },
            ],
        ),
        Example(
            RelaxedBernoulli,
            [
                {
                    "temperature": torch.tensor([1.5], requires_grad=True),
                    "probs": torch.tensor([1.7, 0.2, 0.4], requires_grad=True),
                },
                {
                    "temperature": torch.tensor([2.0]),
                    "probs": torch.tensor([-1.0]),
                },
            ],
        ),
        Example(
            RelaxedOneHotCategorical,
            [
                {
                    "temperature": torch.tensor([0.5], requires_grad=True),
                    "probs": torch.tensor(
                        [[-0.1, 0.2, 0.7], [0.5, 0.3, 0.2]], requires_grad=True
                    ),
                },
                {
                    "temperature": torch.tensor([2.0]),
                    "probs": torch.tensor([[-1.0, 0.0], [-1.0, 1.1]]),
                },
            ],
        ),
        Example(
            TransformedDistribution,
            [
                {
                    "base_distribution": Normal(0, 1),
                    "transforms": lambda x: x,
                },
                {
                    "base_distribution": Normal(0, 1),
                    "transforms": [lambda x: x],
                },
            ],
        ),
        Example(
            Uniform,
            [
                {
                    "low": torch.tensor([2.0], requires_grad=True),
                    "high": torch.tensor([2.0], requires_grad=True),
                },
                {
                    "low": torch.tensor([0.0], requires_grad=True),
                    "high": torch.tensor([0.0], requires_grad=True),
                },
                {
                    "low": torch.tensor([1.0], requires_grad=True),
                    "high": torch.tensor([0.0], requires_grad=True),
                },
            ],
        ),
        Example(
            Weibull,
            [
                {
                    "scale": torch.tensor([0.0], requires_grad=True),
                    "concentration": torch.tensor([0.0], requires_grad=True),
                },
                {
                    "scale": torch.tensor([1.0], requires_grad=True),
                    "concentration": torch.tensor([-1.0], requires_grad=True),
                },
            ],
        ),
        Example(
            Wishart,
            [
                {
                    "covariance_matrix": torch.tensor(
                        [[1.0, 0.0], [0.0, -2.0]], requires_grad=True
                    ),
                    "df": torch.tensor([1.5], requires_grad=True),
                },
                {
                    "covariance_matrix": torch.tensor(
                        [[1.0, 1.0], [1.0, -2.0]], requires_grad=True
                    ),
                    "df": torch.tensor([3.0], requires_grad=True),
                },
                {
                    "covariance_matrix": torch.tensor(
                        [[1.0, 1.0], [1.0, -2.0]], requires_grad=True
                    ),
                    "df": 3.0,
                },
            ],
        ),
        Example(
            ContinuousBernoulli,
            [
                {"probs": torch.tensor([1.1, 0.2, 0.4], requires_grad=True)},
                {"probs": torch.tensor([-0.5], requires_grad=True)},
                {"probs": 1.00001},
            ],
        ),
        Example(
            InverseGamma,
            [
                {
                    "concentration": torch.tensor([0.0, 0.0], requires_grad=True),
                    "rate": torch.tensor([-1.0, -100.0], requires_grad=True),
                },
                {
                    "concentration": torch.tensor([1.0, 1.0], requires_grad=True),
                    "rate": torch.tensor([0.0, 0.0], requires_grad=True),
                },
            ],
        ),
        Example(
            GeneralizedPareto,
            [
                {
                    "loc": torch.tensor([0.0, 0.0], requires_grad=True),
                    "scale": torch.tensor([-1.0, -100.0], requires_grad=True),
                    "concentration": torch.tensor([0.0, 0.0], requires_grad=True),
                },
                {
                    "loc": torch.tensor([1.0, 1.0], requires_grad=True),
                    "scale": torch.tensor([0.0, 0.0], requires_grad=True),
                    "concentration": torch.tensor([-1.0, -100.0], requires_grad=True),
                },
            ],
        ),
    ]


class DistributionsTestCase(TestCase):
    # [CLASS DOC] HELPER: Base class for the distribution tests in this file. Enables the CUDA
    # memory leak check and non-default-stream mode, and forces distribution argument validation
    # on in setUp. Provides shared statistical helpers: _check_sampler_discrete (chi-square test
    # against a reference pmf), _check_sampler_sampler (binned two-sample bias test against
    # reference random variates) and _check_log_prob (per-element log_prob comparison).
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    def setUp(self):
        # [FUNCTION DOC] HELPER: Enables default argument validation globally via
        # set_default_validate_args(True) (a global side effect that is not reset afterwards),
        # then runs the parent setUp.
        """The tests assume that the validation flag is set."""
        torch.distributions.Distribution.set_default_validate_args(True)
        super().setUp()

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def _check_sampler_discrete(
        self, torch_dist, ref_dist, message, num_samples=10000, failure_rate=1e-3
    ):
        # [FUNCTION DOC] HELPER: Chi-square goodness-of-fit check of a discrete torch
        # distribution against a scipy-like reference exposing pmf. Draws num_samples samples,
        # converts bfloat16 to float, counts unique values, renormalizes the reference pmf over
        # the observed values, and asserts the buckets with counts and expected counts above 5
        # hold more than 90 percent of the mass. Remaining low-count values are merged into one
        # remainder bucket. Asserts the chi-square p-value exceeds failure_rate, using message
        # on failure. Skipped without NumPy.
        """Runs a Chi2-test for the support, but ignores tail instead of combining"""
        torch_samples = torch_dist.sample((num_samples,)).squeeze()
        torch_samples = (
            torch_samples.float()
            if torch_samples.dtype == torch.bfloat16
            else torch_samples
        )
        torch_samples = torch_samples.cpu().numpy()
        unique, counts = np.unique(torch_samples, return_counts=True)
        pmf = ref_dist.pmf(unique)
        pmf = pmf / pmf.sum()  # renormalize to 1.0 for chisq test
        msk = (counts > 5) & ((pmf * num_samples) > 5)
        self.assertGreater(
            pmf[msk].sum(),
            0.9,
            "Distribution is too sparse for test; try increasing num_samples",
        )
        # Add a remainder bucket that combines counts for all values
        # below threshold, if such values exist (i.e. mask has False entries).
        if not msk.all():
            counts = np.concatenate([counts[msk], np.sum(counts[~msk], keepdims=True)])
            pmf = np.concatenate([pmf[msk], np.sum(pmf[~msk], keepdims=True)])
        _, p = scipy.stats.chisquare(counts, pmf * num_samples)
        self.assertGreater(p, failure_rate, message)

    def _check_log_prob(self, dist, asset_fn):
        # [FUNCTION DOC] HELPER: Draws one sample from dist, computes log_prob, flattens batch
        # and event data and calls asset_fn(index, sample_row_squeezed, log_prob_scalar) for
        # every batch element so the caller can compare against a reference. Assumes the event
        # data can be reshaped to one row per batch element.
        # checks that the log_prob matches a reference function
        s = dist.sample()
        log_probs = dist.log_prob(s)
        log_probs_data_flat = log_probs.view(-1)
        s_data_flat = s.view(len(log_probs_data_flat), -1)
        for i, (val, log_prob) in enumerate(zip(s_data_flat, log_probs_data_flat)):
            asset_fn(i, val.squeeze(), log_prob)

    def _check_sampler_sampler(
        self,
        torch_dist,
        ref_dist,
        message,
        multivariate=False,
        circular=False,
        num_samples=10000,
        failure_rate=1e-3,
    ):
        # [FUNCTION DOC] HELPER: Statistical check that torch_dist.sample matches a
        # reference sampler exposing rvs. Draws num_samples from each, optionally projects
        # multivariate samples onto a random axis, optionally maps circular values through
        # cosine, labels torch samples +1 and reference -1, shuffles and sorts by value,
        # splits into 10 bins whose means should be near zero, and asserts every bin mean
        # lies within an erfinv-derived threshold for failure_rate. Uses numpy global RNG
        # for the projection axis.
        # Checks that the .sample() method matches a reference function.
        torch_samples = torch_dist.sample((num_samples,)).squeeze()
        torch_samples = torch_samples.cpu().numpy()
        ref_samples = ref_dist.rvs(num_samples).astype(np.float64)
        if multivariate:
            # Project onto a random axis.
            axis = np.random.normal(size=(1,) + torch_samples.shape[1:])
            axis /= np.linalg.norm(axis)
            torch_samples = (axis * torch_samples).reshape(num_samples, -1).sum(-1)
            ref_samples = (axis * ref_samples).reshape(num_samples, -1).sum(-1)
        samples = [(x, +1) for x in torch_samples] + [(x, -1) for x in ref_samples]
        if circular:
            samples = [(np.cos(x), v) for (x, v) in samples]
        shuffle(
            samples
        )  # necessary to prevent stable sort from making uneven bins for discrete
        samples.sort(key=lambda x: x[0])
        samples = np.array(samples)[:, 1]

        # Aggregate into bins filled with roughly zero-mean unit-variance RVs.
        num_bins = 10
        samples_per_bin = len(samples) // num_bins
        bins = samples.reshape((num_bins, samples_per_bin)).mean(axis=1)
        stddev = samples_per_bin**-0.5
        threshold = stddev * scipy.special.erfinv(1 - 2 * failure_rate / num_bins)
        message = f"{message}.sample() is biased:\n{bins}"
        for bias in bins:
            self.assertLess(-threshold, bias, message)
            self.assertLess(bias, threshold, message)


@skipIfTorchDynamo("Not a TorchDynamo suitable test")
class TestDistributions(DistributionsTestCase):
    # [CLASS DOC] Device-generic test class (instantiated per device at the bottom of the file)
    # for the distributions API: sampling shapes, log_prob/entropy against scipy and analytic
    # values, gradients, enumerate_support, broadcasting, cdf/icdf, modes and similar. setUp and
    # tearDown set and clear the default device so tensors are created on the test device.
    def setUp(self):
        # [FUNCTION DOC] Runs the base setUp, then makes the test's primary device the global
        # default device.
        super().setUp()
        torch.set_default_device(self.get_primary_device())

    def tearDown(self):
        # [FUNCTION DOC] Resets the global default device to None, then runs the base tearDown.
        torch.set_default_device(None)
        super().tearDown()

    def test_default_device(self, device):
        # [FUNCTION DOC] Checks that the global default device type matches the device argument
        # and that a newly created random tensor lands on that device.
        device_type = torch.device(device).type
        self.assertEqual(torch.get_default_device().type, device_type)
        self.assertEqual(torch.randn(10).device.type, device_type)

    def _gradcheck_log_prob(self, dist_ctor, ctor_params):
        # [FUNCTION DOC] HELPER: Runs gradcheck on log_prob. Builds dist_ctor(*ctor_params),
        # takes one sample (detached with requires_grad for continuous support), asserts the
        # sample size equals batch_shape plus event_shape, then gradchecks log_prob with respect
        # to the sample and the constructor parameters; raises on failure.
        # performs gradient checks on log_prob
        distribution = dist_ctor(*ctor_params)
        s = distribution.sample()
        if not distribution.support.is_discrete:
            s = s.detach().requires_grad_()

        expected_shape = distribution.batch_shape + distribution.event_shape
        self.assertEqual(s.size(), expected_shape)

        def apply_fn(s, *params):
            # [FUNCTION DOC] HELPER: Function under gradcheck: builds a distribution from params
            # and returns the log_prob of the sample s.
            return dist_ctor(*params).log_prob(s)

        gradcheck(apply_fn, (s,) + tuple(ctor_params), raise_exception=True)

    def _check_forward_ad(self, fn):
        # [FUNCTION DOC] HELPER: Under a forward-AD dual level, applies fn to a dual scalar
        # tensor and asserts the tangent of the output has no non-zero entries, i.e. the
        # sampling op has zero derivative.
        with fwAD.dual_level():
            x = torch.tensor(1.0)
            t = torch.tensor(1.0)
            dual = fwAD.make_dual(x, t)
            dual_out = fn(dual)
            self.assertEqual(
                torch.count_nonzero(fwAD.unpack_dual(dual_out).tangent).item(), 0
            )

    def _check_enumerate_support(self, dist, examples):
        # [FUNCTION DOC] HELPER: For each (params, expected) pair, builds the distribution with
        # parameters converted to tensors and asserts enumerate_support(expand=False) equals
        # expected (cast to the result dtype) and enumerate_support(expand=True) equals expected
        # expanded over batch and event shape.
        for params, expected in examples:
            params = {k: torch.tensor(v) for k, v in params.items()}
            d = dist(**params)
            actual = d.enumerate_support(expand=False)
            expected = torch.tensor(expected, dtype=actual.dtype)
            self.assertEqual(actual, expected)
            actual = d.enumerate_support(expand=True)
            expected_with_expand = expected.expand(
                (-1,) + d.batch_shape + d.event_shape
            )
            self.assertEqual(actual, expected_with_expand)

    def test_repr(self):
        # [FUNCTION DOC] For every example distribution and parameter set, asserts repr starts
        # with the class name.
        for Dist, params in _get_examples():
            for param in params:
                dist = Dist(**param)
                self.assertTrue(repr(dist).startswith(dist.__class__.__name__))

    @expectedFailureMPS
    def test_sample_detached(self):
        # [FUNCTION DOC] For every example having a parameter that requires grad, asserts that
        # sample() returns a tensor that does not require grad. Expected to fail on MPS.
        for Dist, params in _get_examples():
            for i, param in enumerate(params):
                variable_params = [
                    p for p in param.values() if getattr(p, "requires_grad", False)
                ]
                if not variable_params:
                    continue
                dist = Dist(**param)
                sample = dist.sample()
                self.assertFalse(
                    sample.requires_grad,
                    msg=lambda msg: f"{msg}\n{Dist.__name__} example {i + 1}/{len(params)}, .sample() is not detached",
                )

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    def test_rsample_requires_grad(self):
        # [FUNCTION DOC] For every example with a grad-requiring parameter and a distribution
        # with has_rsample, asserts rsample() returns a tensor that requires grad.
        for Dist, params in _get_examples():
            for i, param in enumerate(params):
                if not any(getattr(p, "requires_grad", False) for p in param.values()):
                    continue
                dist = Dist(**param)
                if not dist.has_rsample:
                    continue
                sample = dist.rsample()
                self.assertTrue(
                    sample.requires_grad,
                    msg=lambda msg: f"{msg}\n{Dist.__name__} example {i + 1}/{len(params)}, .rsample() does not require grad",
                )

    @expectedFailureMPS
    def test_enumerate_support_type(self):
        # [FUNCTION DOC] For every example, asserts that sample() and enumerate_support() return
        # the same Python type; distributions whose enumerate_support raises NotImplementedError
        # are ignored.
        for Dist, params in _get_examples():
            for i, param in enumerate(params):
                dist = Dist(**param)
                try:
                    self.assertTrue(
                        type(dist.sample()) is type(dist.enumerate_support()),
                        msg=lambda msg: f"{msg}\n"
                        + (
                            (
                                "{} example {}/{}, return type mismatch between "
                                + "sample and enumerate_support."
                            ).format(Dist.__name__, i + 1, len(params))
                        ),
                    )
                except NotImplementedError:
                    pass

    def test_lazy_property_grad(self):
        # [FUNCTION DOC] Verifies lazy_property keeps autograd working: a dummy class property
        # depending on a tensor gives gradient one both with and without no_grad around the
        # first call, and a MultivariateNormal scale_tril first computed under no_grad still
        # yields a gradient to the covariance matrix when later accessed with grad enabled.
        x = torch.randn(1, requires_grad=True)

        class Dummy:
            # [CLASS DOC] HELPER: Local class with one lazy_property y returning x plus one.
            @lazy_property
            def y(self):
                # [FUNCTION DOC] HELPER: Lazily computed property returning the closed-over
                # tensor x plus one.
                return x + 1

        def test():
            # [FUNCTION DOC] HELPER: Clears x.grad, backpropagates through a fresh Dummy().y and
            # asserts x.grad is ones.
            x.grad = None
            Dummy().y.backward()
            self.assertEqual(x.grad, torch.ones(1))

        test()
        with torch.no_grad():
            test()

        mean = torch.randn(2)
        cov = torch.eye(2, requires_grad=True)
        distn = MultivariateNormal(mean, cov)
        with torch.no_grad():
            distn.scale_tril
        distn.scale_tril.sum().backward()
        self.assertIsNotNone(cov.grad)

    def test_has_examples(self):
        # [FUNCTION DOC] Asserts every Distribution subclass visible in this module's globals
        # (except Distribution and ExponentialFamily) has an entry in _get_examples; failure
        # message asks to add it.
        distributions_with_examples = {e.Dist for e in _get_examples()}
        for Dist in globals().values():
            if (
                isinstance(Dist, type)
                and issubclass(Dist, Distribution)
                and Dist is not Distribution
                and Dist is not ExponentialFamily
            ):
                self.assertIn(
                    Dist,
                    distributions_with_examples,
                    lambda msg: f"{msg}\nPlease add {Dist.__name__} to the _get_examples list in test_distributions.py",
                )

    def test_support_attributes(self):
        # [FUNCTION DOC] For each example checks that the instance support event_dim equals
        # the event_shape rank and that the class-level support gives the same event_dim
        # and is_discrete as the instance (class-level lookups raising NotImplementedError
        # are skipped).
        for Dist, params in _get_examples():
            for param in params:
                d = Dist(**param)
                event_dim = len(d.event_shape)
                self.assertEqual(d.support.event_dim, event_dim)
                try:
                    self.assertEqual(Dist.support.event_dim, event_dim)
                except NotImplementedError:
                    pass
                is_discrete = d.support.is_discrete
                try:
                    self.assertEqual(Dist.support.is_discrete, is_discrete)
                except NotImplementedError:
                    pass

    @expectedFailureMPS
    def test_distribution_expand(self):
        # [FUNCTION DOC] For each example and for expansion shapes (empty, (2,), (2, 1)):
        # expands the batch shape and asserts the class is preserved, sample and log_prob
        # shapes/values are consistent with the original, batch_shape is the expanded one,
        # and mean and variance equal the expanded originals when implemented. Expected to
        # fail on MPS.
        shapes = [torch.Size(), torch.Size((2,)), torch.Size((2, 1))]
        for Dist, params in _get_examples():
            for param in params:
                for shape in shapes:
                    d = Dist(**param)
                    expanded_shape = shape + d.batch_shape
                    original_shape = d.batch_shape + d.event_shape
                    expected_shape = shape + original_shape
                    expanded = d.expand(batch_shape=list(expanded_shape))
                    sample = expanded.sample()
                    actual_shape = expanded.sample().shape
                    self.assertEqual(expanded.__class__, d.__class__)
                    self.assertEqual(d.sample().shape, original_shape)
                    self.assertEqual(expanded.log_prob(sample), d.log_prob(sample))
                    self.assertEqual(actual_shape, expected_shape)
                    self.assertEqual(expanded.batch_shape, expanded_shape)
                    try:
                        self.assertEqual(
                            expanded.mean, d.mean.expand(expanded_shape + d.event_shape)
                        )
                        self.assertEqual(
                            expanded.variance,
                            d.variance.expand(expanded_shape + d.event_shape),
                        )
                    except NotImplementedError:
                        pass

    @expectedFailureMPS
    def test_distribution_subclass_expand(self):
        # [FUNCTION DOC] Like test_distribution_expand but on an empty subclass of each
        # distribution class: expand by a leading dimension of 2 must keep the subclass type,
        # produce samples of the expected shape and equal log_prob.
        expand_by = torch.Size((2,))
        for Dist, params in _get_examples():

            class SubClass(Dist):
                # [CLASS DOC] HELPER: Empty subclass of the distribution under test, to verify
                # expand preserves subclasses.
                pass

            for param in params:
                d = SubClass(**param)
                expanded_shape = expand_by + d.batch_shape
                original_shape = d.batch_shape + d.event_shape
                expected_shape = expand_by + original_shape
                expanded = d.expand(batch_shape=expanded_shape)
                sample = expanded.sample()
                actual_shape = expanded.sample().shape
                self.assertEqual(expanded.__class__, d.__class__)
                self.assertEqual(d.sample().shape, original_shape)
                self.assertEqual(expanded.log_prob(sample), d.log_prob(sample))
                self.assertEqual(actual_shape, expected_shape)

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_bernoulli(self):
        # [FUNCTION DOC] Bernoulli checks (double precision): sample shapes for tensor, 0-d
        # tensor and float probs, samples not requiring grad, gradcheck of log_prob, log_prob
        # against the analytic formula for probs and equivalent logits, rsample raising
        # NotImplementedError, entropy against known values (including probs 0), and forward AD
        # of bernoulli ops having zero tangent.
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True)
        r = torch.tensor(0.3, requires_grad=True)
        s = 0.3
        self.assertEqual(Bernoulli(p).sample((8,)).size(), (8, 3))
        self.assertFalse(Bernoulli(p).sample().requires_grad)
        self.assertEqual(Bernoulli(r).sample((8,)).size(), (8,))
        self.assertEqual(Bernoulli(r).sample().size(), ())
        self.assertEqual(
            Bernoulli(r).sample((3, 2)).size(),
            (
                3,
                2,
            ),
        )
        self.assertEqual(Bernoulli(s).sample().size(), ())
        self._gradcheck_log_prob(Bernoulli, (p,))

        def ref_log_prob(idx, val, log_prob):
            # [FUNCTION DOC] HELPER: Asserts log_prob equals the log of the success
            # probability if the sample is truthy, otherwise of the failure probability, for
            # batch element idx.
            prob = p[idx]
            self.assertEqual(log_prob, math.log(prob if val else 1 - prob))

        self._check_log_prob(Bernoulli(p), ref_log_prob)
        self._check_log_prob(Bernoulli(logits=p.log() - (-p).log1p()), ref_log_prob)
        self.assertRaises(NotImplementedError, Bernoulli(r).rsample)

        # check entropy computation
        self.assertEqual(
            Bernoulli(p).entropy(),
            torch.tensor([0.6108, 0.5004, 0.6730]),
            atol=1e-4,
            rtol=0,
        )
        self.assertEqual(Bernoulli(torch.tensor([0.0])).entropy(), torch.tensor([0.0]))
        self.assertEqual(
            Bernoulli(s).entropy(), torch.tensor(0.6108), atol=1e-4, rtol=0
        )

        self._check_forward_ad(torch.bernoulli)
        self._check_forward_ad(lambda x: x.bernoulli_())
        self._check_forward_ad(lambda x: x.bernoulli_(x.detach().clone()))
        self._check_forward_ad(lambda x: x.bernoulli_(x))

    def test_bernoulli_enumerate_support(self):
        # [FUNCTION DOC] Checks Bernoulli.enumerate_support returns [0, 1] with the expected
        # batch shapes.
        examples = [
            ({"probs": [0.1]}, [[0], [1]]),
            ({"probs": [0.1, 0.9]}, [[0], [1]]),
            ({"probs": [[0.1, 0.2], [0.3, 0.4]]}, [[[0]], [[1]]]),
        ]
        self._check_enumerate_support(Bernoulli, examples)

    def test_bernoulli_3d(self):
        # [FUNCTION DOC] Checks Bernoulli sample shapes for a (2, 3, 5) probability tensor with
        # several sample_shape values.
        p = torch.full((2, 3, 5), 0.5).requires_grad_()
        self.assertEqual(Bernoulli(p).sample().size(), (2, 3, 5))
        self.assertEqual(
            Bernoulli(p).sample(sample_shape=(2, 5)).size(), (2, 5, 2, 3, 5)
        )
        self.assertEqual(Bernoulli(p).sample((2,)).size(), (2, 2, 3, 5))

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_geometric(self):
        # [FUNCTION DOC] Geometric checks (double precision): sample shapes, probs 1 giving
        # sample 0 with log_prob 0 at 0 and minus infinity at 1, samples not requiring grad,
        # gradcheck of log_prob, ValueError for probs 0, rsample raising NotImplementedError and
        # forward AD of geometric_ having zero tangent.
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True)
        r = torch.tensor(0.3, requires_grad=True)
        s = 0.3
        self.assertEqual(Geometric(p).sample((8,)).size(), (8, 3))
        self.assertEqual(Geometric(1).sample(), 0)
        self.assertEqual(Geometric(1).log_prob(torch.tensor(1.0)), -inf)
        self.assertEqual(Geometric(1).log_prob(torch.tensor(0.0)), 0)
        self.assertFalse(Geometric(p).sample().requires_grad)
        self.assertEqual(Geometric(r).sample((8,)).size(), (8,))
        self.assertEqual(Geometric(r).sample().size(), ())
        self.assertEqual(Geometric(r).sample((3, 2)).size(), (3, 2))
        self.assertEqual(Geometric(s).sample().size(), ())
        self._gradcheck_log_prob(Geometric, (p,))
        self.assertRaises(ValueError, lambda: Geometric(0))
        self.assertRaises(NotImplementedError, Geometric(r).rsample)

        self._check_forward_ad(lambda x: x.geometric_(0.2))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_geometric_log_prob_and_entropy(self):
        # [FUNCTION DOC] Compares Geometric log_prob (probs and logits forms) and entropy with
        # scipy.stats.geom (loc minus one). Skipped without NumPy.
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True)
        s = 0.3

        def ref_log_prob(idx, val, log_prob):
            # [FUNCTION DOC] HELPER: Asserts log_prob equals the scipy geometric logpmf at the
            # sample for the batch element's probability.
            prob = p[idx].detach().cpu()
            self.assertEqual(log_prob, scipy.stats.geom(prob, loc=-1).logpmf(val.cpu()))

        self._check_log_prob(Geometric(p), ref_log_prob)
        self._check_log_prob(Geometric(logits=p.log() - (-p).log1p()), ref_log_prob)

        # check entropy computation
        self.assertEqual(
            Geometric(p).entropy(),
            scipy.stats.geom(p.detach().cpu().numpy(), loc=-1).entropy(),
            atol=1e-3,
            rtol=0,
        )
        self.assertEqual(
            float(Geometric(s).entropy()),
            scipy.stats.geom(s, loc=-1).entropy().item(),
            atol=1e-3,
            rtol=0,
        )

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_geometric_sample(self):
        # [FUNCTION DOC] Statistical test: Geometric samples for probs 0.01, 0.18 and 0.8 pass a
        # chi-square test against scipy geom, with seed 0.
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for prob in [0.01, 0.18, 0.8]:
            self._check_sampler_discrete(
                Geometric(prob),
                scipy.stats.geom(p=prob, loc=-1),
                f"Geometric(prob={prob})",
            )

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_binomial(self):
        # [FUNCTION DOC] Binomial gradcheck of log_prob with respect to probs and to logits for
        # total_count 1, 2 and 10, and rsample raising NotImplementedError. Also registered
        # below under half and bfloat16 default dtypes.
        p = torch.arange(0.05, 1, 0.1).requires_grad_()
        for total_count in [1, 2, 10]:
            self._gradcheck_log_prob(lambda p: Binomial(total_count, p), [p])
            self._gradcheck_log_prob(
                lambda p: Binomial(total_count, None, p.log()), [p]
            )
        self.assertRaises(NotImplementedError, Binomial(10, p).rsample)

    test_binomial_half = set_default_dtype(torch.float16)(test_binomial)
    test_binomial_bfloat16 = set_default_dtype(torch.bfloat16)(test_binomial)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_binomial_sample(self):
        # [FUNCTION DOC] Statistical chi-square test of Binomial samples against scipy binom for
        # a grid of probs and total_count values (seed 0).
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for prob in [0.01, 0.1, 0.5, 0.8, 0.9]:
            for count in [2, 10, 100, 500]:
                self._check_sampler_discrete(
                    Binomial(total_count=count, probs=prob),
                    scipy.stats.binom(count, prob),
                    f"Binomial(total_count={count}, probs={prob})",
                )

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_binomial_log_prob_and_entropy(self):
        # [FUNCTION DOC] Compares Binomial log_prob (from probs and from logits) and entropy
        # with scipy.stats.binom for several total_count values.
        probs = torch.arange(0.05, 1, 0.1)
        for total_count in [1, 2, 10]:

            def ref_log_prob(idx, x, log_prob):
                # [FUNCTION DOC] HELPER: Asserts log_prob equals the scipy binomial logpmf at x
                # for the batch element's probability.
                p = probs.view(-1)[idx].item()
                expected = scipy.stats.binom(total_count, p).logpmf(x.cpu())
                self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

            self._check_log_prob(Binomial(total_count, probs), ref_log_prob)
            logits = probs_to_logits(probs, is_binary=True)
            self._check_log_prob(Binomial(total_count, logits=logits), ref_log_prob)

            bin = Binomial(total_count, logits=logits)
            self.assertEqual(
                bin.entropy(),
                scipy.stats.binom(
                    total_count, bin.probs.detach().cpu().numpy(), loc=-1
                ).entropy(),
                atol=1e-3,
                rtol=0,
            )

    def test_binomial_stable(self):
        # [FUNCTION DOC] Checks Binomial log_prob is finite for logits of plus/minus 100 at
        # value 0, and that the gradient of log_prob with respect to zero logits at value 0
        # equals minus 0.5.
        logits = torch.tensor([-100.0, 100.0], dtype=torch.float)
        total_count = 1.0
        x = torch.tensor([0.0, 0.0], dtype=torch.float)
        log_prob = Binomial(total_count, logits=logits).log_prob(x)
        self.assertTrue(torch.isfinite(log_prob).all())

        # make sure that the grad at logits=0, value=0 is 0.5
        x = torch.tensor(0.0, requires_grad=True)
        y = Binomial(total_count, logits=x).log_prob(torch.tensor(0.0))
        self.assertEqual(grad(y, x)[0], torch.tensor(-0.5))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_binomial_log_prob_vectorized_count(self):
        # [FUNCTION DOC] Checks Binomial log_prob with vector-valued total_count (broadcast
        # against probs) matches scipy binom logpmf.
        probs = torch.tensor([0.2, 0.7, 0.9])
        for total_count, sample in [
            (torch.tensor([10]), torch.tensor([7.0, 3.0, 9.0])),
            (torch.tensor([1, 2, 10]), torch.tensor([0.0, 1.0, 9.0])),
        ]:
            log_prob = Binomial(total_count, probs).log_prob(sample)
            expected = scipy.stats.binom(
                total_count.cpu().numpy(), probs.cpu().numpy()
            ).logpmf(sample.cpu())
            self.assertEqual(log_prob, expected, atol=1e-4, rtol=0)

    def test_binomial_enumerate_support(self):
        # [FUNCTION DOC] Checks Binomial.enumerate_support yields 0..total_count with the right
        # batch shapes.
        examples = [
            ({"probs": [0.1], "total_count": 2}, [[0], [1], [2]]),
            ({"probs": [0.1, 0.9], "total_count": 2}, [[0], [1], [2]]),
            (
                {"probs": [[0.1, 0.2], [0.3, 0.4]], "total_count": 3},
                [[[0]], [[1]], [[2]], [[3]]],
            ),
        ]
        self._check_enumerate_support(Binomial, examples)

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_binomial_extreme_vals(self):
        # [FUNCTION DOC] Checks Binomial with probs 0 or 1: samples are exactly 0 or
        # total_count, log_prob is near 0 at the certain outcome and probability 0 elsewhere,
        # and with zero total_count the sample and log_prob are zero.
        total_count = 100
        bin0 = Binomial(total_count, 0)
        self.assertEqual(bin0.sample(), 0)
        self.assertEqual(bin0.log_prob(torch.tensor([0.0]))[0], 0, atol=1e-3, rtol=0)
        self.assertEqual(float(bin0.log_prob(torch.tensor([1.0])).exp()), 0)
        bin1 = Binomial(total_count, 1)
        self.assertEqual(bin1.sample(), total_count)
        self.assertEqual(
            bin1.log_prob(torch.tensor([float(total_count)]))[0], 0, atol=1e-3, rtol=0
        )
        self.assertEqual(
            float(bin1.log_prob(torch.tensor([float(total_count - 1)])).exp()), 0
        )
        zero_counts = torch.zeros(torch.Size((2, 2)))
        bin2 = Binomial(zero_counts, 1)
        self.assertEqual(bin2.sample(), zero_counts)
        self.assertEqual(bin2.log_prob(zero_counts), zero_counts)

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_binomial_vectorized_count(self):
        # [FUNCTION DOC] With seed 1, checks Binomial with a tensor total_count: probs 1 gives
        # total_count, and 100000 samples at probs 0.5 do not exceed total_count and match the
        # mean and variance.
        set_rng_seed(1)  # see Note [Randomized statistical tests]
        total_count = torch.tensor([[4, 7], [3, 8]], dtype=torch.float64)
        bin0 = Binomial(total_count, torch.tensor(1.0))
        self.assertEqual(bin0.sample(), total_count)
        bin1 = Binomial(total_count, torch.tensor(0.5))
        samples = bin1.sample(torch.Size((100000,)))
        self.assertTrue((samples <= total_count.type_as(samples)).all())
        self.assertEqual(samples.mean(dim=0), bin1.mean, atol=0.02, rtol=0)
        self.assertEqual(samples.var(dim=0), bin1.variance, atol=0.02, rtol=0)

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_negative_binomial(self):
        # [FUNCTION DOC] NegativeBinomial gradcheck of log_prob with respect to probs and
        # logits; rsample and entropy raise NotImplementedError.
        p = torch.arange(0.05, 1, 0.1).requires_grad_()
        for total_count in [1, 2, 10]:
            self._gradcheck_log_prob(lambda p: NegativeBinomial(total_count, p), [p])
            self._gradcheck_log_prob(
                lambda p: NegativeBinomial(total_count, None, p.log()), [p]
            )
        self.assertRaises(NotImplementedError, NegativeBinomial(10, p).rsample)
        self.assertRaises(NotImplementedError, NegativeBinomial(10, p).entropy)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_negative_binomial_log_prob(self):
        # [FUNCTION DOC] Compares NegativeBinomial log_prob (probs and logits forms) with scipy
        # nbinom logpmf, using failure probability one minus p. Skipped without NumPy.
        probs = torch.arange(0.05, 1, 0.1)
        for total_count in [1, 2, 10]:

            def ref_log_prob(idx, x, log_prob):
                # [FUNCTION DOC] HELPER: Asserts log_prob equals scipy nbinom logpmf at x for
                # the batch element.
                p = probs.view(-1)[idx].item()
                expected = scipy.stats.nbinom(total_count, 1 - p).logpmf(x.cpu())
                self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

            self._check_log_prob(NegativeBinomial(total_count, probs), ref_log_prob)
            logits = probs_to_logits(probs, is_binary=True)
            self._check_log_prob(
                NegativeBinomial(total_count, logits=logits), ref_log_prob
            )

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_negative_binomial_log_prob_vectorized_count(self):
        # [FUNCTION DOC] Checks NegativeBinomial log_prob with vector-valued total_count against
        # scipy nbinom.
        probs = torch.tensor([0.2, 0.7, 0.9])
        for total_count, sample in [
            (torch.tensor([10]), torch.tensor([7.0, 3.0, 9.0])),
            (torch.tensor([1, 2, 10]), torch.tensor([0.0, 1.0, 9.0])),
        ]:
            log_prob = NegativeBinomial(total_count, probs).log_prob(sample)
            expected = scipy.stats.nbinom(
                total_count.cpu().numpy(), 1 - probs.cpu().numpy()
            ).logpmf(sample.cpu().numpy())
            self.assertEqual(log_prob, expected, atol=1e-4, rtol=0)

    def test_zero_excluded_binomial(self):
        # [FUNCTION DOC] Draws 1e8 samples from Binomial(total_count 1, probs 0.9 and 0.1) and
        # checks values stay within [0, 1], plus 10000 samples at 0.5 each yield more than 4000
        # zeros and ones (raising AssertionError otherwise). Large memory use.
        vals = Binomial(
            total_count=torch.tensor(1.0),
            probs=torch.tensor(0.9),
        ).sample(torch.Size((100000000,)))
        self.assertTrue((vals >= 0).all())
        vals = Binomial(
            total_count=torch.tensor(1.0),
            probs=torch.tensor(0.1),
        ).sample(torch.Size((100000000,)))
        self.assertTrue((vals < 2).all())
        vals = Binomial(
            total_count=torch.tensor(1.0),
            probs=torch.tensor(0.5),
        ).sample(torch.Size((10000,)))
        # vals should be roughly half zeroes, half ones
        zeros_count = (vals == 0.0).sum()
        ones_count = (vals == 1.0).sum()
        if zeros_count <= 4000:
            raise AssertionError(
                f"Expected (vals == 0.0).sum() > 4000, got {zeros_count}"
            )
        if ones_count <= 4000:
            raise AssertionError(
                f"Expected (vals == 1.0).sum() > 4000, got {ones_count}"
            )

    def test_torch_binomial_dtype_errors(self):
        # [FUNCTION DOC] Checks torch.binomial raises ValueError with messages about
        # floating-point dtypes when count or prob has an integer dtype (int, long, short).
        dtypes = [torch.int, torch.long, torch.short]

        for count_dtype in dtypes:
            total_count = torch.tensor([10, 10], dtype=count_dtype)
            total_prob = torch.tensor([0.5, 0.5], dtype=torch.float)

            with self.assertRaisesRegex(
                ValueError,
                "binomial only supports floating-point dtypes for count.*",
            ):
                torch.binomial(total_count, total_prob)

        for prob_dtype in dtypes:
            total_count = torch.tensor([10, 10], dtype=torch.float)
            total_prob = torch.tensor([0.5, 0.5], dtype=prob_dtype)

            with self.assertRaisesRegex(
                ValueError,
                "binomial only supports floating-point dtypes for prob.*",
            ):
                torch.binomial(total_count, total_prob)

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_multinomial_1d(self):
        # [FUNCTION DOC] Multinomial checks: sample shapes for 1-d probs, gradcheck of log_prob
        # for probs and logits, and rsample raising NotImplementedError.
        total_count = 10
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        self.assertEqual(Multinomial(total_count, p).sample().size(), (3,))
        self.assertEqual(Multinomial(total_count, p).sample((2, 2)).size(), (2, 2, 3))
        self.assertEqual(Multinomial(total_count, p).sample((1,)).size(), (1, 3))
        self._gradcheck_log_prob(lambda p: Multinomial(total_count, p), [p])
        self._gradcheck_log_prob(lambda p: Multinomial(total_count, None, p.log()), [p])
        self.assertRaises(NotImplementedError, Multinomial(10, p).rsample)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @set_default_dtype_if_supported(torch.double)
    def test_multinomial_1d_log_prob_and_entropy(self):
        # [FUNCTION DOC] Compares Multinomial log_prob (probs and logits forms) and entropy with
        # scipy.stats.multinomial. Skipped without NumPy.
        total_count = 10
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        dist = Multinomial(total_count, probs=p)
        x = dist.sample()
        log_prob = dist.log_prob(x)
        expected = torch.tensor(
            scipy.stats.multinomial.logpmf(
                x.cpu().numpy(), n=total_count, p=dist.probs.detach().cpu().numpy()
            ),
            dtype=torch.get_default_dtype(),
        )
        self.assertEqual(log_prob, expected)

        dist = Multinomial(total_count, logits=p.log())
        x = dist.sample()
        log_prob = dist.log_prob(x)
        expected = torch.tensor(
            scipy.stats.multinomial.logpmf(
                x.cpu().numpy(), n=total_count, p=dist.probs.detach().cpu().numpy()
            ),
            dtype=torch.get_default_dtype(),
        )
        self.assertEqual(log_prob, expected)

        expected = torch.tensor(
            scipy.stats.multinomial.entropy(
                total_count, dist.probs.detach().cpu().numpy()
            ),
            dtype=torch.get_default_dtype(),
        )
        self.assertEqual(dist.entropy(), expected, atol=1e-3, rtol=0)

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_multinomial_2d(self):
        # [FUNCTION DOC] Multinomial with 2-d probs: sample shapes for several sample_shape
        # values, gradchecks, and extreme probs (identity matrix) giving samples that put all
        # counts on one category.
        total_count = 10
        probabilities = [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]]
        probabilities_1 = [[1.0, 0.0], [0.0, 1.0]]
        p = torch.tensor(probabilities, requires_grad=True)
        s = torch.tensor(probabilities_1, requires_grad=True)
        self.assertEqual(Multinomial(total_count, p).sample().size(), (2, 3))
        self.assertEqual(
            Multinomial(total_count, p).sample(sample_shape=(3, 4)).size(), (3, 4, 2, 3)
        )
        self.assertEqual(Multinomial(total_count, p).sample((6,)).size(), (6, 2, 3))
        set_rng_seed(0)
        self._gradcheck_log_prob(lambda p: Multinomial(total_count, p), [p])
        self._gradcheck_log_prob(lambda p: Multinomial(total_count, None, p.log()), [p])

        # sample check for extreme value of probs
        self.assertEqual(
            Multinomial(total_count, s).sample(),
            torch.tensor([[total_count, 0], [0, total_count]], dtype=torch.float64),
        )

    @skipMPS  # very long runtime on MPS
    def test_multinomial_sequential_draw(self):
        # [FUNCTION DOC] Draws 300000 permutations from torch.multinomial without replacement
        # over 26 equal weights and asserts fewer than 10 duplicates, regression for gh-132395.
        # Skipped on MPS.
        # Adapted after script mentioned in https://github.com/pytorch/pytorch/issues/132395
        torch.manual_seed(0xDE0B6B3A764007E8)
        prob = torch.ones(26)
        dups_mult = 0
        perm_counts_mult = {}
        for _ in range(300_000):
            p = tuple(torch.multinomial(prob, prob.numel(), replacement=False).tolist())
            if p in perm_counts_mult:
                dups_mult += 1
                perm_counts_mult[p] += 1
            else:
                perm_counts_mult[p] = 1
        self.assertLess(dups_mult, 10)

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_categorical_1d(self):
        # [FUNCTION DOC] Categorical with 1-d probs: mean and variance are all NaN, sample shape
        # and no grad, gradcheck of log_prob and rsample raising NotImplementedError.
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        self.assertTrue(is_all_nan(Categorical(p).mean))
        self.assertTrue(is_all_nan(Categorical(p).variance))
        self.assertEqual(Categorical(p).sample().size(), ())
        self.assertFalse(Categorical(p).sample().requires_grad)
        self.assertEqual(Categorical(p).sample((2, 2)).size(), (2, 2))
        self.assertEqual(Categorical(p).sample((1,)).size(), (1,))
        self._gradcheck_log_prob(Categorical, (p,))
        self.assertRaises(NotImplementedError, Categorical(p).rsample)

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_categorical_2d(self):
        # [FUNCTION DOC] Categorical with 2-d probs: mean/variance shape and NaN values, sample
        # shapes, gradcheck, extreme probs sample check, log_prob against analytic probabilities
        # for probs and logits, entropy known values and -inf logits entropy (gh-40553).
        probabilities = [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]]
        probabilities_1 = [[1.0, 0.0], [0.0, 1.0]]
        p = torch.tensor(probabilities, requires_grad=True)
        s = torch.tensor(probabilities_1, requires_grad=True)
        self.assertEqual(Categorical(p).mean.size(), (2,))
        self.assertEqual(Categorical(p).variance.size(), (2,))
        self.assertTrue(is_all_nan(Categorical(p).mean))
        self.assertTrue(is_all_nan(Categorical(p).variance))
        self.assertEqual(Categorical(p).sample().size(), (2,))
        self.assertEqual(Categorical(p).sample(sample_shape=(3, 4)).size(), (3, 4, 2))
        self.assertEqual(Categorical(p).sample((6,)).size(), (6, 2))
        self._gradcheck_log_prob(Categorical, (p,))

        # sample check for extreme value of probs
        set_rng_seed(0)
        self.assertEqual(
            Categorical(s).sample(sample_shape=(2,)), torch.tensor([[0, 1], [0, 1]])
        )

        def ref_log_prob(idx, val, log_prob):
            # [FUNCTION DOC] HELPER: Asserts log_prob equals the log of the normalized
            # probability of the sampled category for batch element idx.
            sample_prob = p[idx][val] / p[idx].sum()
            self.assertEqual(log_prob, math.log(sample_prob))

        self._check_log_prob(Categorical(p), ref_log_prob)
        self._check_log_prob(Categorical(logits=p.log()), ref_log_prob)

        # check entropy computation
        self.assertEqual(
            Categorical(p).entropy(), torch.tensor([1.0114, 1.0297]), atol=1e-4, rtol=0
        )
        self.assertEqual(Categorical(s).entropy(), torch.tensor([0.0, 0.0]))
        # issue gh-40553
        logits = p.log()
        logits[1, 1] = logits[0, 2] = float("-inf")
        e = Categorical(logits=logits).entropy()
        self.assertEqual(e, torch.tensor([0.6365, 0.5983]), atol=1e-4, rtol=0)

    def test_categorical_enumerate_support(self):
        # [FUNCTION DOC] Checks Categorical.enumerate_support returns the category indices with
        # the right shapes.
        examples = [
            ({"probs": [0.1, 0.2, 0.7]}, [0, 1, 2]),
            ({"probs": [[0.1, 0.9], [0.3, 0.7]]}, [[0], [1]]),
        ]
        self._check_enumerate_support(Categorical, examples)

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_one_hot_categorical_1d(self):
        # [FUNCTION DOC] OneHotCategorical with 1-d probs: sample shapes, samples not requiring
        # grad, gradcheck and rsample raising NotImplementedError.
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        self.assertEqual(OneHotCategorical(p).sample().size(), (3,))
        self.assertFalse(OneHotCategorical(p).sample().requires_grad)
        self.assertEqual(OneHotCategorical(p).sample((2, 2)).size(), (2, 2, 3))
        self.assertEqual(OneHotCategorical(p).sample((1,)).size(), (1, 3))
        self._gradcheck_log_prob(OneHotCategorical, (p,))
        self.assertRaises(NotImplementedError, OneHotCategorical(p).rsample)

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_one_hot_categorical_2d(self):
        # [FUNCTION DOC] OneHotCategorical with 2-d probs: sample shapes, gradcheck, and
        # log_prob equal to the Categorical log_prob of the argmax index.
        probabilities = [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]]
        p = torch.tensor(probabilities, requires_grad=True)
        self.assertEqual(OneHotCategorical(p).sample().size(), (2, 3))
        self.assertEqual(
            OneHotCategorical(p).sample(sample_shape=(3, 4)).size(), (3, 4, 2, 3)
        )
        self.assertEqual(OneHotCategorical(p).sample((6,)).size(), (6, 2, 3))
        self._gradcheck_log_prob(OneHotCategorical, (p,))

        dist = OneHotCategorical(p)
        x = dist.sample()
        self.assertEqual(dist.log_prob(x), Categorical(p).log_prob(x.max(-1)[1]))

    def test_one_hot_categorical_enumerate_support(self):
        # [FUNCTION DOC] Checks OneHotCategorical.enumerate_support yields identity rows with
        # the right shapes.
        examples = [
            ({"probs": [0.1, 0.2, 0.7]}, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            ({"probs": [[0.1, 0.9], [0.3, 0.7]]}, [[[1, 0]], [[0, 1]]]),
        ]
        self._check_enumerate_support(OneHotCategorical, examples)

    def test_poisson_forward_ad(self):
        # [FUNCTION DOC] Checks forward AD of torch.poisson gives zero tangent.
        self._check_forward_ad(torch.poisson)

    def test_poisson_shape(self):
        # [FUNCTION DOC] Checks Poisson sample shapes for 2-d, 1-element and float rates.
        rate = torch.randn(2, 3).abs().requires_grad_()
        rate_1d = torch.randn(1).abs().requires_grad_()
        self.assertEqual(Poisson(rate).sample().size(), (2, 3))
        self.assertEqual(Poisson(rate).sample((7,)).size(), (7, 2, 3))
        self.assertEqual(Poisson(rate_1d).sample().size(), (1,))
        self.assertEqual(Poisson(rate_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Poisson(2.0).sample((2,)).size(), (2,))

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_poisson_log_prob(self):
        # [FUNCTION DOC] Compares Poisson log_prob to scipy poisson logpmf (including zero
        # rate), gradchecks it, and checks that at rate zero and value one the gradient wrt rate
        # is infinity.
        rate = torch.randn(2, 3).abs().requires_grad_()
        rate_1d = torch.randn(1).abs().requires_grad_()
        rate_zero = torch.zeros([], requires_grad=True)

        def ref_log_prob(ref_rate, idx, x, log_prob):
            # [FUNCTION DOC] HELPER: Asserts log_prob equals scipy poisson logpmf using the
            # flattened rate element.
            l = ref_rate.view(-1)[idx].detach()
            expected = scipy.stats.poisson.logpmf(x.cpu(), l.cpu())
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        set_rng_seed(0)
        self._check_log_prob(Poisson(rate), lambda *args: ref_log_prob(rate, *args))
        self._check_log_prob(
            Poisson(rate_zero), lambda *args: ref_log_prob(rate_zero, *args)
        )
        self._gradcheck_log_prob(Poisson, (rate,))
        self._gradcheck_log_prob(Poisson, (rate_1d,))

        # We cannot check gradients automatically for zero rates because the finite difference
        # approximation enters the forbidden parameter space. We instead compare with the
        # theoretical results.
        dist = Poisson(rate_zero)
        dist.log_prob(torch.ones_like(rate_zero)).backward()
        self.assertEqual(rate_zero.grad, torch.inf)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_poisson_sample(self, device):
        # [FUNCTION DOC] Statistical test for Poisson samples with seed 1 under default dtypes
        # float, bfloat16, half (and double except on MPS), rates 0.1, 1 and 5; restores the
        # default dtype in a finally block. Takes the test device.
        set_rng_seed(1)  # see Note [Randomized statistical tests]
        saved_dtype = torch.get_default_dtype()
        try:
            dtypes = [torch.float, torch.bfloat16, torch.half]
            if torch.device(device).type != "mps":
                dtypes += [torch.double]
            for dtype in dtypes:
                torch.set_default_dtype(dtype)
                for rate in [0.1, 1.0, 5.0]:
                    self._check_sampler_discrete(
                        Poisson(rate),
                        scipy.stats.poisson(rate),
                        f"Poisson(lambda={rate})",
                        failure_rate=1e-3,
                    )
        finally:
            torch.set_default_dtype(saved_dtype)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_poisson_tensor_sample(self):
        # [FUNCTION DOC] Statistical chi-square test of Poisson with a 1-element tensor rate
        # (seed 1).
        set_rng_seed(1)
        for rate in [0.12, 0.9, 4.0]:
            self._check_sampler_discrete(
                Poisson(torch.tensor([rate])),
                scipy.stats.poisson(rate),
                f"Poisson(lambda={rate})",
                failure_rate=1e-3,
            )

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_relaxed_bernoulli(self):
        # [FUNCTION DOC] RelaxedBernoulli: sample shapes for tensor, 0-d and float probs,
        # samples not requiring grad, gradchecks for log_prob, and rsample supporting backward.
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True)
        r = torch.tensor(0.3, requires_grad=True)
        s = 0.3
        temp = torch.tensor(0.67, requires_grad=True)
        self.assertEqual(RelaxedBernoulli(temp, p).sample((8,)).size(), (8, 3))
        self.assertFalse(RelaxedBernoulli(temp, p).sample().requires_grad)
        self.assertEqual(RelaxedBernoulli(temp, r).sample((8,)).size(), (8,))
        self.assertEqual(RelaxedBernoulli(temp, r).sample().size(), ())
        self.assertEqual(
            RelaxedBernoulli(temp, r).sample((3, 2)).size(),
            (
                3,
                2,
            ),
        )
        self.assertEqual(RelaxedBernoulli(temp, s).sample().size(), ())
        self._gradcheck_log_prob(RelaxedBernoulli, (temp, p))
        self._gradcheck_log_prob(RelaxedBernoulli, (temp, r))

        # test that rsample doesn't fail
        s = RelaxedBernoulli(temp, p).rsample()
        s.backward(torch.ones_like(s))

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_rounded_relaxed_bernoulli(self):
        # [FUNCTION DOC] Statistical test: rounding RelaxedBernoulli samples matches Bernoulli
        # across probs and temperatures, and a huge temperature yields rsample of exactly 0.5.
        set_rng_seed(0)  # see Note [Randomized statistical tests]

        class Rounded:
            # [CLASS DOC] HELPER: Wrapper exposing sample() that rounds samples of the
            # wrapped distribution.
            def __init__(self, dist):
                # [FUNCTION DOC] HELPER: Stores the wrapped distribution.
                self.dist = dist

            def sample(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Returns the rounded samples of the
                # wrapped distribution.
                return torch.round(self.dist.sample(*args, **kwargs))

        for probs, temp in product([0.1, 0.2, 0.8], [0.1, 1.0, 10.0]):
            self._check_sampler_discrete(
                Rounded(RelaxedBernoulli(temp, probs)),
                scipy.stats.bernoulli(probs),
                f"Rounded(RelaxedBernoulli(temp={temp}, probs={probs}))",
                failure_rate=1e-3,
            )

        for probs in [0.001, 0.2, 0.999]:
            equal_probs = torch.tensor(0.5)
            dist = RelaxedBernoulli(1e10, probs)
            s = dist.rsample()
            self.assertEqual(equal_probs, s)

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_relaxed_one_hot_categorical_1d(self):
        # [FUNCTION DOC] RelaxedOneHotCategorical with 1-d probs: sample shapes, samples not
        # requiring grad and gradcheck of log_prob (with validation off).
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        temp = torch.tensor(0.67, requires_grad=True)
        self.assertEqual(
            RelaxedOneHotCategorical(probs=p, temperature=temp).sample().size(), (3,)
        )
        self.assertFalse(
            RelaxedOneHotCategorical(probs=p, temperature=temp).sample().requires_grad
        )
        self.assertEqual(
            RelaxedOneHotCategorical(probs=p, temperature=temp).sample((2, 2)).size(),
            (2, 2, 3),
        )
        self.assertEqual(
            RelaxedOneHotCategorical(probs=p, temperature=temp).sample((1,)).size(),
            (1, 3),
        )
        self._gradcheck_log_prob(
            lambda t, p: RelaxedOneHotCategorical(t, p, validate_args=False), (temp, p)
        )

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_relaxed_one_hot_categorical_2d(self):
        # [FUNCTION DOC] RelaxedOneHotCategorical with 2-d probs: sample shapes and
        # gradchecks at temperatures 3 and 0.25 (lower temperatures are numerically unstable
        # for gradcheck).
        probabilities = [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]]
        temp = torch.tensor([3.0], requires_grad=True)
        # The lower the temperature, the more unstable the log_prob gradcheck is
        # w.r.t. the sample. Values below 0.25 empirically fail the default tol.
        temp_2 = torch.tensor([0.25], requires_grad=True)
        p = torch.tensor(probabilities, requires_grad=True)
        self.assertEqual(RelaxedOneHotCategorical(temp, p).sample().size(), (2, 3))
        self.assertEqual(
            RelaxedOneHotCategorical(temp, p).sample(sample_shape=(3, 4)).size(),
            (3, 4, 2, 3),
        )
        self.assertEqual(
            RelaxedOneHotCategorical(temp, p).sample((6,)).size(), (6, 2, 3)
        )
        self._gradcheck_log_prob(
            lambda t, p: RelaxedOneHotCategorical(t, p, validate_args=False), (temp, p)
        )
        self._gradcheck_log_prob(
            lambda t, p: RelaxedOneHotCategorical(t, p, validate_args=False),
            (temp_2, p),
        )

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_argmax_relaxed_categorical(self):
        # [FUNCTION DOC] Statistical test: argmax of RelaxedOneHotCategorical samples matches
        # scipy multinomial for several probs/temperatures, and very high temperature gives
        # uniform rsample.
        set_rng_seed(0)  # see Note [Randomized statistical tests]

        class ArgMax:
            # [CLASS DOC] HELPER: Wrapper whose sample returns the argmax index over the last
            # dim of the wrapped distribution's samples.
            def __init__(self, dist):
                # [FUNCTION DOC] HELPER: Stores the wrapped distribution.
                self.dist = dist

            def sample(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Returns the argmax index of samples from the
                # wrapped distribution.
                s = self.dist.sample(*args, **kwargs)
                _, idx = torch.max(s, -1)
                return idx

        class ScipyCategorical:
            # [CLASS DOC] HELPER: Adapter turning scipy multinomial pmf over one-hot vectors
            # into a pmf over category indices.
            def __init__(self, dist):
                # [FUNCTION DOC] HELPER: Stores the wrapped scipy distribution.
                self.dist = dist

            def pmf(self, samples):
                # [FUNCTION DOC] HELPER: Converts integer category samples to one-hot rows and
                # returns the scipy pmf.
                new_samples = np.zeros(samples.shape + self.dist.p.shape)
                new_samples[np.arange(samples.shape[0]), samples] = 1
                return self.dist.pmf(new_samples)

        for probs, temp in product(
            [torch.tensor([0.1, 0.9]), torch.tensor([0.2, 0.2, 0.6])], [0.1, 1.0, 10.0]
        ):
            self._check_sampler_discrete(
                ArgMax(RelaxedOneHotCategorical(temp, probs)),
                ScipyCategorical(scipy.stats.multinomial(1, [p.cpu() for p in probs])),
                f"Rounded(RelaxedOneHotCategorical(temp={temp}, probs={probs}))",
                failure_rate=1e-3,
            )

        for probs in [torch.tensor([0.1, 0.9]), torch.tensor([0.2, 0.2, 0.6])]:
            equal_probs = torch.ones(probs.size()) / probs.size()[0]
            dist = RelaxedOneHotCategorical(1e10, probs)
            s = dist.rsample()
            self.assertEqual(equal_probs, s)

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_uniform(self):
        # [FUNCTION DOC] Uniform checks: sample shapes, log_prob of minus infinity and cdf 0/1
        # outside the range (validation off), gradchecks, rsample gradients equal one minus the
        # base random draw for low and the draw for high, and forward AD of uniform_.
        low = torch.zeros(5, 5, requires_grad=True)
        high = (torch.ones(5, 5) * 3).requires_grad_()
        low_1d = torch.zeros(1, requires_grad=True)
        high_1d = (torch.ones(1) * 3).requires_grad_()
        self.assertEqual(Uniform(low, high).sample().size(), (5, 5))
        self.assertEqual(Uniform(low, high).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Uniform(low_1d, high_1d).sample().size(), (1,))
        self.assertEqual(Uniform(low_1d, high_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Uniform(0.0, 1.0).sample((1,)).size(), (1,))

        # Check log_prob computation when value outside range
        uniform = Uniform(low_1d, high_1d, validate_args=False)
        above_high = torch.tensor([4.0])
        below_low = torch.tensor([-1.0])
        self.assertEqual(uniform.log_prob(above_high).item(), -inf)
        self.assertEqual(uniform.log_prob(below_low).item(), -inf)

        # check cdf computation when value outside range
        self.assertEqual(uniform.cdf(below_low).item(), 0)
        self.assertEqual(uniform.cdf(above_high).item(), 1)

        set_rng_seed(1)
        self._gradcheck_log_prob(Uniform, (low, high))
        self._gradcheck_log_prob(Uniform, (low, 1.0))
        self._gradcheck_log_prob(Uniform, (0.0, high))

        state = torch.get_rng_state()
        rand = low.new(low.size()).uniform_()
        torch.set_rng_state(state)
        u = Uniform(low, high).rsample()
        u.backward(torch.ones_like(u))
        self.assertEqual(low.grad, 1 - rand)
        self.assertEqual(high.grad, rand)
        low.grad.zero_()
        high.grad.zero_()

        self._check_forward_ad(lambda x: x.uniform_())

    @expectedFailureMPS
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_vonmises_sample(self):
        # [FUNCTION DOC] Statistical test of VonMises samples against scipy vonmises over locs
        # and concentrations using cosine of circular samples. Skipped without NumPy and
        # expected failure on MPS.
        for loc in [0.0, math.pi / 2.0]:
            for concentration in [0.03, 0.3, 1.0, 10.0, 100.0]:
                self._check_sampler_sampler(
                    VonMises(loc, concentration),
                    scipy.stats.vonmises(loc=loc, kappa=concentration),
                    f"VonMises(loc={loc}, concentration={concentration})",
                    num_samples=int(1e5),
                    circular=True,
                )

    def test_vonmises_logprob(self):
        # [FUNCTION DOC] Checks that exp of VonMises log_prob integrates to about 1 over [0, 2
        # pi) for a range of concentrations (tolerance 1e-3).
        concentrations = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
        for concentration in concentrations:
            grid = torch.arange(0.0, 2 * math.pi, 1e-4)
            prob = VonMises(0.0, concentration).log_prob(grid).exp()
            norm = prob.mean().item() * 2 * math.pi
            self.assertLess(abs(norm - 1), 1e-3)

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_cauchy(self):
        # [FUNCTION DOC] Cauchy checks: mean is NaN and variance infinite, sample shapes,
        # gradchecks, rsample gradients (loc one, scale equal to the standard Cauchy draw) and
        # forward AD of cauchy_.
        loc = torch.zeros(5, 5, requires_grad=True)
        scale = torch.ones(5, 5, requires_grad=True)
        loc_1d = torch.zeros(1, requires_grad=True)
        scale_1d = torch.ones(1, requires_grad=True)
        self.assertTrue(is_all_nan(Cauchy(loc_1d, scale_1d).mean))
        self.assertEqual(Cauchy(loc_1d, scale_1d).variance, inf)
        self.assertEqual(Cauchy(loc, scale).sample().size(), (5, 5))
        self.assertEqual(Cauchy(loc, scale).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Cauchy(loc_1d, scale_1d).sample().size(), (1,))
        self.assertEqual(Cauchy(loc_1d, scale_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Cauchy(0.0, 1.0).sample((1,)).size(), (1,))

        set_rng_seed(1)
        self._gradcheck_log_prob(Cauchy, (loc, scale))
        self._gradcheck_log_prob(Cauchy, (loc, 1.0))
        self._gradcheck_log_prob(Cauchy, (0.0, scale))

        state = torch.get_rng_state()
        eps = loc.new(loc.size()).cauchy_()
        torch.set_rng_state(state)
        c = Cauchy(loc, scale).rsample()
        c.backward(torch.ones_like(c))
        self.assertEqual(loc.grad, torch.ones_like(scale))
        self.assertEqual(scale.grad, eps)
        loc.grad.zero_()
        scale.grad.zero_()

        self._check_forward_ad(lambda x: x.cauchy_())

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_halfcauchy(self):
        # [FUNCTION DOC] HalfCauchy checks: infinite mean and variance, sample shapes,
        # gradchecks and rsample gradient of scale equal to the absolute standard Cauchy draw.
        scale = torch.ones(5, 5, requires_grad=True)
        scale_1d = torch.ones(1, requires_grad=True)
        self.assertTrue(torch.isinf(HalfCauchy(scale_1d).mean).all())
        self.assertEqual(HalfCauchy(scale_1d).variance, inf)
        self.assertEqual(HalfCauchy(scale).sample().size(), (5, 5))
        self.assertEqual(HalfCauchy(scale).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(HalfCauchy(scale_1d).sample().size(), (1,))
        self.assertEqual(HalfCauchy(scale_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(HalfCauchy(1.0).sample((1,)).size(), (1,))

        set_rng_seed(1)
        self._gradcheck_log_prob(HalfCauchy, (scale,))
        self._gradcheck_log_prob(HalfCauchy, (1.0,))

        state = torch.get_rng_state()
        eps = scale.new(scale.size()).cauchy_().abs_()
        torch.set_rng_state(state)
        c = HalfCauchy(scale).rsample()
        c.backward(torch.ones_like(c))
        self.assertEqual(scale.grad, eps)
        scale.grad.zero_()

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_halfnormal(self):
        # [FUNCTION DOC] HalfNormal checks: sample shapes, near-zero samples for tiny scale,
        # gradchecks, and log_prob broadcasting of a (3, 1) value against a (2, 1, 4) scale to
        # shape (2, 3, 4).
        std = torch.randn(5, 5).abs().requires_grad_()
        std_1d = torch.randn(1).abs().requires_grad_()
        std_delta = torch.tensor([1e-5, 1e-5])
        self.assertEqual(HalfNormal(std).sample().size(), (5, 5))
        self.assertEqual(HalfNormal(std).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(HalfNormal(std_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(HalfNormal(std_1d).sample().size(), (1,))
        self.assertEqual(HalfNormal(0.6).sample((1,)).size(), (1,))
        self.assertEqual(HalfNormal(50.0).sample((1,)).size(), (1,))

        # sample check for extreme value of std
        set_rng_seed(1)
        self.assertEqual(
            HalfNormal(std_delta).sample(sample_shape=(1, 2)),
            torch.tensor([[[0.0, 0.0], [0.0, 0.0]]]),
            atol=1e-4,
            rtol=0,
        )

        self._gradcheck_log_prob(HalfNormal, (std,))
        self._gradcheck_log_prob(HalfNormal, (1.0,))

        # check .log_prob() can broadcast.
        dist = HalfNormal(torch.ones(2, 1, 4))
        log_prob = dist.log_prob(torch.ones(3, 1))
        self.assertEqual(log_prob.shape, (2, 3, 4))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_halfnormal_logprob(self):
        # [FUNCTION DOC] Compares HalfNormal log_prob with scipy halfnorm logpdf.
        std = torch.randn(5, 1).abs().requires_grad_()

        def ref_log_prob(idx, x, log_prob):
            # [FUNCTION DOC] HELPER: Asserts log_prob equals scipy halfnorm logpdf at x for the
            # batch element's scale.
            s = std.view(-1)[idx].detach().cpu()
            expected = scipy.stats.halfnorm(scale=s).logpdf(x.cpu())
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(HalfNormal(std), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_halfnormal_sample(self):
        # [FUNCTION DOC] Statistical test of HalfNormal samples against scipy halfnorm for
        # scales 0.1, 1, 10.
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for std in [0.1, 1.0, 10.0]:
            self._check_sampler_sampler(
                HalfNormal(std),
                scipy.stats.halfnorm(scale=std),
                f"HalfNormal(scale={std})",
            )

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_inversegamma(self):
        # [FUNCTION DOC] InverseGamma checks: sample shapes, gradcheck of log_prob and
        # broadcasting of log_prob to shape (2, 3, 4).
        alpha = torch.randn(2, 3).exp().requires_grad_()
        beta = torch.randn(2, 3).exp().requires_grad_()
        alpha_1d = torch.randn(1).exp().requires_grad_()
        beta_1d = torch.randn(1).exp().requires_grad_()
        self.assertEqual(InverseGamma(alpha, beta).sample().size(), (2, 3))
        self.assertEqual(InverseGamma(alpha, beta).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(InverseGamma(alpha_1d, beta_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(InverseGamma(alpha_1d, beta_1d).sample().size(), (1,))
        self.assertEqual(InverseGamma(0.5, 0.5).sample().size(), ())
        self.assertEqual(InverseGamma(0.5, 0.5).sample((1,)).size(), (1,))

        self._gradcheck_log_prob(InverseGamma, (alpha, beta))

        dist = InverseGamma(torch.ones(4), torch.ones(2, 1, 1))
        log_prob = dist.log_prob(torch.ones(3, 1))
        self.assertEqual(log_prob.shape, (2, 3, 4))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_inversegamma_sample(self):
        # [FUNCTION DOC] Statistical test of InverseGamma samples against scipy invgamma over
        # concentrations and rates.
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for concentration, rate in product([2, 5], [0.1, 1.0, 10.0]):
            self._check_sampler_sampler(
                InverseGamma(concentration, rate),
                scipy.stats.invgamma(concentration, scale=rate),
                "InverseGamma()",
            )

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_lognormal(self):
        # [FUNCTION DOC] LogNormal checks: sample shapes, near-deterministic samples for tiny
        # std, gradchecks, log_prob broadcasting and forward AD of log_normal_.
        mean = torch.randn(5, 5, requires_grad=True)
        std = torch.randn(5, 5).abs().requires_grad_()
        mean_1d = torch.randn(1, requires_grad=True)
        std_1d = torch.randn(1).abs().requires_grad_()
        mean_delta = torch.tensor([1.0, 0.0])
        std_delta = torch.tensor([1e-5, 1e-5])
        self.assertEqual(LogNormal(mean, std).sample().size(), (5, 5))
        self.assertEqual(LogNormal(mean, std).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(LogNormal(mean_1d, std_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(LogNormal(mean_1d, std_1d).sample().size(), (1,))
        self.assertEqual(LogNormal(0.2, 0.6).sample((1,)).size(), (1,))
        self.assertEqual(LogNormal(-0.7, 50.0).sample((1,)).size(), (1,))

        # sample check for extreme value of mean, std
        set_rng_seed(1)
        self.assertEqual(
            LogNormal(mean_delta, std_delta).sample(sample_shape=(1, 2)),
            torch.tensor([[[math.exp(1), 1.0], [math.exp(1), 1.0]]]),
            atol=1e-4,
            rtol=0,
        )

        self._gradcheck_log_prob(LogNormal, (mean, std))
        self._gradcheck_log_prob(LogNormal, (mean, 1.0))
        self._gradcheck_log_prob(LogNormal, (0.0, std))

        # check .log_prob() can broadcast.
        dist = LogNormal(torch.zeros(4), torch.ones(2, 1, 1))
        log_prob = dist.log_prob(torch.ones(3, 1))
        self.assertEqual(log_prob.shape, (2, 3, 4))

        self._check_forward_ad(lambda x: x.log_normal_())

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_lognormal_logprob(self):
        # [FUNCTION DOC] Compares LogNormal log_prob with scipy lognorm logpdf.
        mean = torch.randn(5, 1, requires_grad=True)
        std = torch.randn(5, 1).abs().requires_grad_()

        def ref_log_prob(idx, x, log_prob):
            # [FUNCTION DOC] HELPER: Asserts log_prob equals scipy lognorm logpdf at x for the
            # batch element.
            m = mean.view(-1)[idx].detach().cpu()
            s = std.view(-1)[idx].detach().cpu()
            expected = scipy.stats.lognorm(s=s, scale=math.exp(m)).logpdf(x.cpu())
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(LogNormal(mean, std), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_lognormal_sample(self):
        # [FUNCTION DOC] Statistical test of LogNormal samples against scipy lognorm over means
        # and stds.
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for mean, std in product([-1.0, 0.0, 1.0], [0.1, 1.0, 10.0]):
            self._check_sampler_sampler(
                LogNormal(mean, std),
                scipy.stats.lognorm(scale=math.exp(mean), s=std),
                f"LogNormal(loc={mean}, scale={std})",
            )

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_logisticnormal(self):
        # [FUNCTION DOC] LogisticNormal checks: samples have one more event dim than the
        # parameters, tiny-std samples match the closed-form stick-breaking values, and
        # gradchecks of log_prob with validation off (gradcheck can perturb samples slightly off
        # the simplex).
        set_rng_seed(1)  # see Note [Randomized statistical tests]
        mean = torch.randn(5, 5).requires_grad_()
        std = torch.randn(5, 5).abs().requires_grad_()
        mean_1d = torch.randn(1).requires_grad_()
        std_1d = torch.randn(1).abs().requires_grad_()
        mean_delta = torch.tensor([1.0, 0.0])
        std_delta = torch.tensor([1e-5, 1e-5])
        self.assertEqual(LogisticNormal(mean, std).sample().size(), (5, 6))
        self.assertEqual(LogisticNormal(mean, std).sample((7,)).size(), (7, 5, 6))
        self.assertEqual(LogisticNormal(mean_1d, std_1d).sample((1,)).size(), (1, 2))
        self.assertEqual(LogisticNormal(mean_1d, std_1d).sample().size(), (2,))
        self.assertEqual(LogisticNormal(0.2, 0.6).sample().size(), (2,))
        self.assertEqual(LogisticNormal(-0.7, 50.0).sample().size(), (2,))

        # sample check for extreme value of mean, std
        set_rng_seed(1)
        self.assertEqual(
            LogisticNormal(mean_delta, std_delta).sample(),
            torch.tensor(
                [
                    math.exp(1) / (1.0 + 1.0 + math.exp(1)),
                    1.0 / (1.0 + 1.0 + math.exp(1)),
                    1.0 / (1.0 + 1.0 + math.exp(1)),
                ]
            ),
            atol=1e-4,
            rtol=0,
        )

        # TODO: gradcheck seems to mutate the sample values so that the simplex
        # constraint fails by a very small margin.
        self._gradcheck_log_prob(
            lambda m, s: LogisticNormal(m, s, validate_args=False), (mean, std)
        )
        self._gradcheck_log_prob(
            lambda m, s: LogisticNormal(m, s, validate_args=False), (mean, 1.0)
        )
        self._gradcheck_log_prob(
            lambda m, s: LogisticNormal(m, s, validate_args=False), (0.0, std)
        )

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_logisticnormal_logprob(self):
        # [FUNCTION DOC] Smoke test: LogisticNormal log_prob of a sample has the batch shape
        # (5,). The TODO says proper value checks are missing.
        mean = torch.randn(5, 7).requires_grad_()
        std = torch.randn(5, 7).abs().requires_grad_()

        # Smoke test for now
        # TODO: Once _check_log_prob works with multidimensional distributions,
        #       add proper testing of the log probabilities.
        dist = LogisticNormal(mean, std)
        shape = dist.log_prob(dist.sample()).detach().cpu().numpy().shape
        if shape != (5,):
            raise AssertionError(f"Expected log_prob shape (5,), got {shape}")

    def _get_logistic_normal_ref_sampler(self, base_dist):
        # [FUNCTION DOC] HELPER: Returns a function that maps a multivariate normal base
        # distribution's samples to the stick-breaking (logistic normal) simplex with numpy,
        # used as a reference sampler.
        def _sampler(num_samples):
            # [FUNCTION DOC] HELPER: Draws num_samples from the base distribution and applies
            # the stick-breaking transform (sigmoid with offset, cumulative product, padding) to
            # return simplex samples.
            x = base_dist.rvs(num_samples)
            offset = np.log((x.shape[-1] + 1) - np.ones_like(x).cumsum(-1))
            z = 1.0 / (1.0 + np.exp(offset - x))
            z_cumprod = np.cumprod(1 - z, axis=-1)
            y1 = np.pad(z, ((0, 0), (0, 1)), mode="constant", constant_values=1.0)
            y2 = np.pad(
                z_cumprod, ((0, 0), (1, 0)), mode="constant", constant_values=1.0
            )
            return y1 * y2

        return _sampler

    @expectedFailureMPS
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_logisticnormal_sample(self):
        # [FUNCTION DOC] Statistical test of LogisticNormal samples against the numpy reference
        # sampler for several means and diagonal covariances (multivariate projection).
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        means = map(np.asarray, [(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)])
        covs = map(np.diag, [(0.1, 0.1), (1.0, 1.0), (10.0, 10.0)])
        for mean, cov in product(means, covs):
            base_dist = scipy.stats.multivariate_normal(mean=mean, cov=cov)
            ref_dist = scipy.stats.multivariate_normal(mean=mean, cov=cov)
            ref_dist.rvs = self._get_logistic_normal_ref_sampler(base_dist)
            mean_th = torch.tensor(mean)
            std_th = torch.tensor(np.sqrt(np.diag(cov)))
            self._check_sampler_sampler(
                LogisticNormal(mean_th, std_th),
                ref_dist,
                f"LogisticNormal(loc={mean_th}, scale={std_th})",
                multivariate=True,
            )

    def test_mixture_same_family_shape(self):
        # [FUNCTION DOC] Checks MixtureSameFamily sample shapes for scalar, batched and
        # multi-batched mixtures of univariate Normal and of Independent-Normal (event dim
        # 1) components.
        normal_case_1d = MixtureSameFamily(
            Categorical(torch.rand(5)), Normal(torch.randn(5), torch.rand(5))
        )
        normal_case_1d_batch = MixtureSameFamily(
            Categorical(torch.rand(3, 5)), Normal(torch.randn(3, 5), torch.rand(3, 5))
        )
        normal_case_1d_multi_batch = MixtureSameFamily(
            Categorical(torch.rand(4, 3, 5)),
            Normal(torch.randn(4, 3, 5), torch.rand(4, 3, 5)),
        )
        normal_case_2d = MixtureSameFamily(
            Categorical(torch.rand(5)),
            Independent(Normal(torch.randn(5, 2), torch.rand(5, 2)), 1),
        )
        normal_case_2d_batch = MixtureSameFamily(
            Categorical(torch.rand(3, 5)),
            Independent(Normal(torch.randn(3, 5, 2), torch.rand(3, 5, 2)), 1),
        )
        normal_case_2d_multi_batch = MixtureSameFamily(
            Categorical(torch.rand(4, 3, 5)),
            Independent(Normal(torch.randn(4, 3, 5, 2), torch.rand(4, 3, 5, 2)), 1),
        )

        self.assertEqual(normal_case_1d.sample().size(), ())
        self.assertEqual(normal_case_1d.sample((2,)).size(), (2,))
        self.assertEqual(normal_case_1d.sample((2, 7)).size(), (2, 7))
        self.assertEqual(normal_case_1d_batch.sample().size(), (3,))
        self.assertEqual(normal_case_1d_batch.sample((2,)).size(), (2, 3))
        self.assertEqual(normal_case_1d_batch.sample((2, 7)).size(), (2, 7, 3))
        self.assertEqual(normal_case_1d_multi_batch.sample().size(), (4, 3))
        self.assertEqual(normal_case_1d_multi_batch.sample((2,)).size(), (2, 4, 3))
        self.assertEqual(normal_case_1d_multi_batch.sample((2, 7)).size(), (2, 7, 4, 3))

        self.assertEqual(normal_case_2d.sample().size(), (2,))
        self.assertEqual(normal_case_2d.sample((2,)).size(), (2, 2))
        self.assertEqual(normal_case_2d.sample((2, 7)).size(), (2, 7, 2))
        self.assertEqual(normal_case_2d_batch.sample().size(), (3, 2))
        self.assertEqual(normal_case_2d_batch.sample((2,)).size(), (2, 3, 2))
        self.assertEqual(normal_case_2d_batch.sample((2, 7)).size(), (2, 7, 3, 2))
        self.assertEqual(normal_case_2d_multi_batch.sample().size(), (4, 3, 2))
        self.assertEqual(normal_case_2d_multi_batch.sample((2,)).size(), (2, 4, 3, 2))
        self.assertEqual(
            normal_case_2d_multi_batch.sample((2, 7)).size(), (2, 7, 4, 3, 2)
        )

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_mixture_same_family_normal_log_prob(self):
        # [FUNCTION DOC] Compares MixtureSameFamily(Categorical, Normal) log_prob with a scipy
        # logsumexp mixture.
        probs = torch.rand(10, 5).softmax(dim=-1)
        loc = torch.randn(10, 5)
        scale = torch.rand(10, 5)

        def ref_log_prob(idx, x, log_prob):
            # [FUNCTION DOC] HELPER: Asserts log_prob equals the log-sum-exp of component logpdf
            # plus log weights.
            p = probs[idx].cpu().numpy()
            m = loc[idx].cpu().numpy()
            s = scale[idx].cpu().numpy()
            mix = scipy.stats.multinomial(1, p)
            comp = scipy.stats.norm(m, s)
            expected = scipy.special.logsumexp(comp.logpdf(x.cpu()) + np.log(mix.p))
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(
            MixtureSameFamily(Categorical(probs=probs), Normal(loc, scale)),
            ref_log_prob,
        )

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_mixture_same_family_binomial_log_prob(self):
        # [FUNCTION DOC] Compares MixtureSameFamily(Categorical, Binomial) log_prob with a scipy
        # logsumexp mixture.
        max_count = 20
        probs = torch.rand(10, 5).softmax(dim=-1)
        binom_probs = torch.rand(10, 5)

        def ref_log_prob(idx, x, log_prob):
            # [FUNCTION DOC] HELPER: Asserts log_prob equals the log-sum-exp of component
            # binomial logpmf plus log weights.
            p = probs[idx].cpu().numpy()
            binom_p = binom_probs[idx].cpu().numpy()
            mix = scipy.stats.multinomial(1, p)
            comp = scipy.stats.binom(max_count, binom_p)
            expected = scipy.special.logsumexp(comp.logpmf(x.cpu()) + np.log(mix.p))
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(
            MixtureSameFamily(
                Categorical(probs=probs), Binomial(max_count, binom_probs)
            ),
            ref_log_prob,
        )

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_mixture_same_family_sample(self):
        # [FUNCTION DOC] Statistical test of MixtureSameFamily(Categorical, Normal) samples
        # against a scipy-based reference mixture sampler.
        probs = torch.rand(5).softmax(dim=-1)
        loc = torch.randn(5)
        scale = torch.rand(5)

        class ScipyMixtureNormal:
            # [CLASS DOC] HELPER: Reference mixture of normals sampled with scipy.
            def __init__(self, probs, mu, std):
                # [FUNCTION DOC] HELPER: Stores mixture probabilities, means and
                # standard deviations.
                self.probs = probs
                self.mu = mu
                self.std = std

            def rvs(self, n_sample):
                # [FUNCTION DOC] HELPER: Draws n_sample values by drawing every component and a
                # one-hot component selection, returning the selected component sample per draw
                # as an array.
                comp_samples = [
                    scipy.stats.norm(m, s).rvs(n_sample)
                    for m, s in zip(self.mu, self.std)
                ]
                mix_samples = scipy.stats.multinomial(1, self.probs).rvs(n_sample)
                samples = []
                for i in range(n_sample):
                    samples.append(comp_samples[mix_samples[i].argmax()][i])
                return np.asarray(samples)

        self._check_sampler_sampler(
            MixtureSameFamily(Categorical(probs=probs), Normal(loc, scale)),
            ScipyMixtureNormal(
                probs.cpu().numpy(), loc.cpu().numpy(), scale.cpu().numpy()
            ),
            f"""MixtureSameFamily(Categorical(probs={probs}),
            Normal(loc={loc}, scale={scale}))""",
        )

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_normal(self):
        # [FUNCTION DOC] Normal checks: sample shapes, extreme-parameter samples, gradchecks,
        # rsample gradients (loc ones, scale equal to the standard normal draw), log_prob
        # against the analytic density, and forward AD of normal ops with zero tangent.
        loc = torch.randn(5, 5, requires_grad=True)
        scale = torch.randn(5, 5).abs().requires_grad_()
        loc_1d = torch.randn(1, requires_grad=True)
        scale_1d = torch.randn(1).abs().requires_grad_()
        loc_delta = torch.tensor([1.0, 0.0])
        scale_delta = torch.tensor([1e-5, 1e-5])
        self.assertEqual(Normal(loc, scale).sample().size(), (5, 5))
        self.assertEqual(Normal(loc, scale).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Normal(loc_1d, scale_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Normal(loc_1d, scale_1d).sample().size(), (1,))
        self.assertEqual(Normal(0.2, 0.6).sample((1,)).size(), (1,))
        self.assertEqual(Normal(-0.7, 50.0).sample((1,)).size(), (1,))

        # sample check for extreme value of mean, std
        set_rng_seed(1)
        self.assertEqual(
            Normal(loc_delta, scale_delta).sample(sample_shape=(1, 2)),
            torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
            atol=1e-4,
            rtol=0,
        )

        self._gradcheck_log_prob(Normal, (loc, scale))
        self._gradcheck_log_prob(Normal, (loc, 1.0))
        self._gradcheck_log_prob(Normal, (0.0, scale))

        state = torch.get_rng_state()
        eps = torch.normal(torch.zeros_like(loc), torch.ones_like(scale))
        torch.set_rng_state(state)
        z = Normal(loc, scale).rsample()
        z.backward(torch.ones_like(z))
        self.assertEqual(loc.grad, torch.ones_like(loc))
        self.assertEqual(scale.grad, eps)
        loc.grad.zero_()
        scale.grad.zero_()
        self.assertEqual(z.size(), (5, 5))

        def ref_log_prob(idx, x, log_prob):
            # [FUNCTION DOC] HELPER: Asserts log_prob equals the analytic normal log-density for
            # the batch element.
            m = loc.view(-1)[idx]
            s = scale.view(-1)[idx]
            expected = math.exp(-((x - m) ** 2) / (2 * s**2)) / math.sqrt(
                2 * math.pi * s**2
            )
            self.assertEqual(log_prob, math.log(expected), atol=1e-3, rtol=0)

        self._check_log_prob(Normal(loc, scale), ref_log_prob)
        self._check_forward_ad(torch.normal)
        self._check_forward_ad(lambda x: torch.normal(x, 0.5))
        self._check_forward_ad(lambda x: torch.normal(0.2, x))
        self._check_forward_ad(lambda x: torch.normal(x, x))
        self._check_forward_ad(lambda x: x.normal_())

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_normal_sample(self):
        # [FUNCTION DOC] Statistical test of Normal samples against scipy norm over locs
        # and scales.
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for loc, scale in product([-1.0, 0.0, 1.0], [0.1, 1.0, 10.0]):
            self._check_sampler_sampler(
                Normal(loc, scale),
                scipy.stats.norm(loc=loc, scale=scale),
                f"Normal(mean={loc}, std={scale})",
            )

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_lowrank_multivariate_normal_shape(self):
        # [FUNCTION DOC] LowRankMultivariateNormal shape and gradient checks (double precision):
        # sample sizes for unbatched, batched and multi-batched means with unbatched and batched
        # cov_factor/cov_diag and sample shapes (), (2,) and (2, 7), then gradchecks log_prob
        # for three batch combinations.
        mean = torch.randn(5, 3, requires_grad=True)
        mean_no_batch = torch.randn(3, requires_grad=True)
        mean_multi_batch = torch.randn(6, 5, 3, requires_grad=True)

        # construct PSD covariance
        cov_factor = torch.randn(3, 1, requires_grad=True)
        cov_diag = torch.randn(3).abs().requires_grad_()

        # construct batch of PSD covariances
        cov_factor_batched = torch.randn(6, 5, 3, 2, requires_grad=True)
        cov_diag_batched = torch.randn(6, 5, 3).abs().requires_grad_()

        # ensure that sample, batch, event shapes all handled correctly
        self.assertEqual(
            LowRankMultivariateNormal(mean, cov_factor, cov_diag).sample().size(),
            (5, 3),
        )
        self.assertEqual(
            LowRankMultivariateNormal(mean_no_batch, cov_factor, cov_diag)
            .sample()
            .size(),
            (3,),
        )
        self.assertEqual(
            LowRankMultivariateNormal(mean_multi_batch, cov_factor, cov_diag)
            .sample()
            .size(),
            (6, 5, 3),
        )
        self.assertEqual(
            LowRankMultivariateNormal(mean, cov_factor, cov_diag).sample((2,)).size(),
            (2, 5, 3),
        )
        self.assertEqual(
            LowRankMultivariateNormal(mean_no_batch, cov_factor, cov_diag)
            .sample((2,))
            .size(),
            (2, 3),
        )
        self.assertEqual(
            LowRankMultivariateNormal(mean_multi_batch, cov_factor, cov_diag)
            .sample((2,))
            .size(),
            (2, 6, 5, 3),
        )
        self.assertEqual(
            LowRankMultivariateNormal(mean, cov_factor, cov_diag).sample((2, 7)).size(),
            (2, 7, 5, 3),
        )
        self.assertEqual(
            LowRankMultivariateNormal(mean_no_batch, cov_factor, cov_diag)
            .sample((2, 7))
            .size(),
            (2, 7, 3),
        )
        self.assertEqual(
            LowRankMultivariateNormal(mean_multi_batch, cov_factor, cov_diag)
            .sample((2, 7))
            .size(),
            (2, 7, 6, 5, 3),
        )
        self.assertEqual(
            LowRankMultivariateNormal(mean, cov_factor_batched, cov_diag_batched)
            .sample((2, 7))
            .size(),
            (2, 7, 6, 5, 3),
        )
        self.assertEqual(
            LowRankMultivariateNormal(
                mean_no_batch, cov_factor_batched, cov_diag_batched
            )
            .sample((2, 7))
            .size(),
            (2, 7, 6, 5, 3),
        )
        self.assertEqual(
            LowRankMultivariateNormal(
                mean_multi_batch, cov_factor_batched, cov_diag_batched
            )
            .sample((2, 7))
            .size(),
            (2, 7, 6, 5, 3),
        )

        # check gradients
        self._gradcheck_log_prob(
            LowRankMultivariateNormal, (mean, cov_factor, cov_diag)
        )
        self._gradcheck_log_prob(
            LowRankMultivariateNormal, (mean_multi_batch, cov_factor, cov_diag)
        )
        self._gradcheck_log_prob(
            LowRankMultivariateNormal,
            (mean_multi_batch, cov_factor_batched, cov_diag_batched),
        )

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_lowrank_multivariate_normal_log_prob(self):
        # [FUNCTION DOC] Compares LowRankMultivariateNormal log_prob with scipy
        # multivariate_normal logpdf (mean squared error below 1e-3) using the covariance
        # cov_factor times its transpose plus diag(cov_diag), then checks batched log_prob
        # equals a stack of per-element unbatched distributions. Skipped without NumPy. The
        # comment mentions scale_tril equivalence but only the factor/diag parametrization
        # is exercised.
        mean = torch.randn(3, requires_grad=True)
        cov_factor = torch.randn(3, 1, requires_grad=True)
        cov_diag = torch.randn(3).abs().requires_grad_()
        cov = cov_factor.matmul(cov_factor.t()) + cov_diag.diag()

        # check that logprob values match scipy logpdf,
        # and that covariance and scale_tril parameters are equivalent
        dist1 = LowRankMultivariateNormal(mean, cov_factor, cov_diag)
        ref_dist = scipy.stats.multivariate_normal(
            mean.detach().cpu().numpy(), cov.detach().cpu().numpy()
        )

        x = dist1.sample((10,))
        expected = ref_dist.logpdf(x.cpu().numpy())

        self.assertEqual(
            0.0,
            np.mean((dist1.log_prob(x).detach().cpu().numpy() - expected) ** 2),
            atol=1e-3,
            rtol=0,
        )

        # Double-check that batched versions behave the same as unbatched
        mean = torch.randn(5, 3, requires_grad=True)
        cov_factor = torch.randn(5, 3, 2, requires_grad=True)
        cov_diag = torch.randn(5, 3).abs().requires_grad_()

        dist_batched = LowRankMultivariateNormal(mean, cov_factor, cov_diag)
        dist_unbatched = [
            LowRankMultivariateNormal(mean[i], cov_factor[i], cov_diag[i])
            for i in range(mean.size(0))
        ]

        x = dist_batched.sample((10,))
        batched_prob = dist_batched.log_prob(x)
        unbatched_prob = torch.stack(
            [dist_unbatched[i].log_prob(x[:, i]) for i in range(5)]
        ).t()

        self.assertEqual(batched_prob.shape, unbatched_prob.shape)
        self.assertEqual(batched_prob, unbatched_prob, atol=1e-3, rtol=0)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_lowrank_multivariate_normal_sample(self):
        # [FUNCTION DOC] Statistical test (seed 0): LowRankMultivariateNormal samples match
        # scipy multivariate_normal with the equivalent dense covariance.
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        mean = torch.randn(5, requires_grad=True)
        cov_factor = torch.randn(5, 1, requires_grad=True)
        cov_diag = torch.randn(5).abs().requires_grad_()
        cov = cov_factor.matmul(cov_factor.t()) + cov_diag.diag()

        self._check_sampler_sampler(
            LowRankMultivariateNormal(mean, cov_factor, cov_diag),
            scipy.stats.multivariate_normal(
                mean.detach().cpu().numpy(), cov.detach().cpu().numpy()
            ),
            f"LowRankMultivariateNormal(loc={mean}, cov_factor={cov_factor}, cov_diag={cov_diag})",
            multivariate=True,
        )

    def test_lowrank_multivariate_normal_properties(self):
        # [FUNCTION DOC] Checks LowRankMultivariateNormal properties (mean, variance,
        # covariance_matrix, scale_tril, precision_matrix, entropy) equal those of a
        # MultivariateNormal built from the equivalent dense covariance.
        loc = torch.randn(5)
        cov_factor = torch.randn(5, 2)
        cov_diag = torch.randn(5).abs()
        cov = cov_factor.matmul(cov_factor.t()) + cov_diag.diag()
        m1 = LowRankMultivariateNormal(loc, cov_factor, cov_diag)
        m2 = MultivariateNormal(loc=loc, covariance_matrix=cov)
        self.assertEqual(m1.mean, m2.mean)
        self.assertEqual(m1.variance, m2.variance)
        self.assertEqual(m1.covariance_matrix, m2.covariance_matrix)
        self.assertEqual(m1.scale_tril, m2.scale_tril)
        self.assertEqual(m1.precision_matrix, m2.precision_matrix)
        self.assertEqual(m1.entropy(), m2.entropy())

    @expectedFailureMPS
    def test_lowrank_multivariate_normal_moments(self):
        # [FUNCTION DOC] With seed 0 draws 100000 rsamples from a LowRankMultivariateNormal and
        # checks the empirical mean and variance match the distribution's mean and variance
        # within tolerance.
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        mean = torch.randn(5)
        cov_factor = torch.randn(5, 2)
        cov_diag = torch.randn(5).abs()
        d = LowRankMultivariateNormal(mean, cov_factor, cov_diag)
        samples = d.rsample((100000,))
        empirical_mean = samples.mean(0)
        self.assertEqual(d.mean, empirical_mean, atol=0.01, rtol=0)
        empirical_var = samples.var(0)
        self.assertEqual(d.variance, empirical_var, atol=0.02, rtol=0)

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_multivariate_normal_shape(self):
        # [FUNCTION DOC] MultivariateNormal shape and gradient checks: sample sizes for
        # unbatched, batched and multi-batched means with covariance, precision_matrix and
        # scale_tril parametrizations and sample shapes (), (2,), (2, 7), then gradchecks
        # log_prob for each parametrization using a custom gradcheck wrapper.
        mean = torch.randn(5, 3, requires_grad=True)
        mean_no_batch = torch.randn(3, requires_grad=True)
        mean_multi_batch = torch.randn(6, 5, 3, requires_grad=True)

        # construct PSD covariance
        tmp = torch.randn(3, 10)
        cov = (torch.matmul(tmp, tmp.t()) / tmp.size(-1)).requires_grad_()
        prec = cov.inverse().requires_grad_()
        scale_tril = torch.linalg.cholesky(cov).requires_grad_()

        # construct batch of PSD covariances
        tmp = torch.randn(6, 5, 3, 10)
        cov_batched = (tmp.unsqueeze(-2) * tmp.unsqueeze(-3)).mean(-1).requires_grad_()
        prec_batched = cov_batched.inverse()
        scale_tril_batched = torch.linalg.cholesky(cov_batched)

        # ensure that sample, batch, event shapes all handled correctly
        self.assertEqual(MultivariateNormal(mean, cov).sample().size(), (5, 3))
        self.assertEqual(MultivariateNormal(mean_no_batch, cov).sample().size(), (3,))
        self.assertEqual(
            MultivariateNormal(mean_multi_batch, cov).sample().size(), (6, 5, 3)
        )
        self.assertEqual(MultivariateNormal(mean, cov).sample((2,)).size(), (2, 5, 3))
        self.assertEqual(
            MultivariateNormal(mean_no_batch, cov).sample((2,)).size(), (2, 3)
        )
        self.assertEqual(
            MultivariateNormal(mean_multi_batch, cov).sample((2,)).size(), (2, 6, 5, 3)
        )
        self.assertEqual(
            MultivariateNormal(mean, cov).sample((2, 7)).size(), (2, 7, 5, 3)
        )
        self.assertEqual(
            MultivariateNormal(mean_no_batch, cov).sample((2, 7)).size(), (2, 7, 3)
        )
        self.assertEqual(
            MultivariateNormal(mean_multi_batch, cov).sample((2, 7)).size(),
            (2, 7, 6, 5, 3),
        )
        self.assertEqual(
            MultivariateNormal(mean, cov_batched).sample((2, 7)).size(), (2, 7, 6, 5, 3)
        )
        self.assertEqual(
            MultivariateNormal(mean_no_batch, cov_batched).sample((2, 7)).size(),
            (2, 7, 6, 5, 3),
        )
        self.assertEqual(
            MultivariateNormal(mean_multi_batch, cov_batched).sample((2, 7)).size(),
            (2, 7, 6, 5, 3),
        )
        self.assertEqual(
            MultivariateNormal(mean, precision_matrix=prec).sample((2, 7)).size(),
            (2, 7, 5, 3),
        )
        self.assertEqual(
            MultivariateNormal(mean, precision_matrix=prec_batched)
            .sample((2, 7))
            .size(),
            (2, 7, 6, 5, 3),
        )
        self.assertEqual(
            MultivariateNormal(mean, scale_tril=scale_tril).sample((2, 7)).size(),
            (2, 7, 5, 3),
        )
        self.assertEqual(
            MultivariateNormal(mean, scale_tril=scale_tril_batched)
            .sample((2, 7))
            .size(),
            (2, 7, 6, 5, 3),
        )

        # check gradients
        # We write a custom gradcheck function to maintain the symmetry
        # of the perturbed covariances and their inverses (precision)
        def multivariate_normal_log_prob_gradcheck(
            mean, covariance=None, precision=None, scale_tril=None
        ):
            # [FUNCTION DOC] HELPER: Draws a sample from MultivariateNormal built from mean and
            # exactly one of covariance, precision, scale_tril, then runs gradcheck of log_prob
            # with respect to the sample and all given parameters.
            mvn_samples = (
                MultivariateNormal(mean, covariance, precision, scale_tril)
                .sample()
                .requires_grad_()
            )

            def gradcheck_func(samples, mu, sigma, prec, scale_tril):
                # [FUNCTION DOC] HELPER: Function under gradcheck. Symmetrizes the covariance
                # and precision (average with the transpose) and takes the lower triangle of
                # scale_tril so that perturbations stay valid, then returns the
                # MultivariateNormal log_prob of samples.
                if sigma is not None:
                    sigma = 0.5 * (sigma + sigma.mT)  # Ensure symmetry of covariance
                if prec is not None:
                    prec = 0.5 * (prec + prec.mT)  # Ensure symmetry of precision
                if scale_tril is not None:
                    scale_tril = scale_tril.tril()
                return MultivariateNormal(mu, sigma, prec, scale_tril).log_prob(samples)

            gradcheck(
                gradcheck_func,
                (mvn_samples, mean, covariance, precision, scale_tril),
                raise_exception=True,
            )

        multivariate_normal_log_prob_gradcheck(mean, cov)
        multivariate_normal_log_prob_gradcheck(mean_multi_batch, cov)
        multivariate_normal_log_prob_gradcheck(mean_multi_batch, cov_batched)
        multivariate_normal_log_prob_gradcheck(mean, None, prec)
        multivariate_normal_log_prob_gradcheck(mean_no_batch, None, prec_batched)
        multivariate_normal_log_prob_gradcheck(mean, None, None, scale_tril)
        multivariate_normal_log_prob_gradcheck(
            mean_no_batch, None, None, scale_tril_batched
        )

    @set_default_dtype_if_supported(torch.double)
    def test_multivariate_normal_stable_with_precision_matrix(self):
        # [FUNCTION DOC] Regression check that constructing a MultivariateNormal from an
        # RBF-kernel precision matrix succeeds (no exception); no values are asserted.
        x = torch.randn(10)
        P = torch.exp(-((x - x.unsqueeze(-1)) ** 2))  # RBF kernel
        MultivariateNormal(x.new_zeros(10), precision_matrix=P)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_multivariate_normal_log_prob(self):
        # [FUNCTION DOC] Compares MultivariateNormal log_prob from covariance, precision and
        # scale_tril parametrizations with scipy multivariate_normal logpdf (mean squared error
        # below 1e-3), then verifies batched log_prob matches a stack of unbatched
        # distributions. Skipped without NumPy.
        mean = torch.randn(3, requires_grad=True)
        tmp = torch.randn(3, 10)
        cov = (torch.matmul(tmp, tmp.t()) / tmp.size(-1)).requires_grad_()
        prec = cov.inverse().requires_grad_()
        scale_tril = torch.linalg.cholesky(cov).requires_grad_()

        # check that logprob values match scipy logpdf,
        # and that covariance and scale_tril parameters are equivalent
        dist1 = MultivariateNormal(mean, cov)
        dist2 = MultivariateNormal(mean, precision_matrix=prec)
        dist3 = MultivariateNormal(mean, scale_tril=scale_tril)
        ref_dist = scipy.stats.multivariate_normal(
            mean.detach().cpu().numpy(), cov.detach().cpu().numpy()
        )

        x = dist1.sample((10,))
        expected = ref_dist.logpdf(x.cpu().numpy())

        self.assertEqual(
            0.0,
            np.mean((dist1.log_prob(x).detach().cpu().numpy() - expected) ** 2),
            atol=1e-3,
            rtol=0,
        )
        self.assertEqual(
            0.0,
            np.mean((dist2.log_prob(x).detach().cpu().numpy() - expected) ** 2),
            atol=1e-3,
            rtol=0,
        )
        self.assertEqual(
            0.0,
            np.mean((dist3.log_prob(x).detach().cpu().numpy() - expected) ** 2),
            atol=1e-3,
            rtol=0,
        )

        # Double-check that batched versions behave the same as unbatched
        mean = torch.randn(5, 3, requires_grad=True)
        tmp = torch.randn(5, 3, 10)
        cov = (tmp.unsqueeze(-2) * tmp.unsqueeze(-3)).mean(-1).requires_grad_()

        dist_batched = MultivariateNormal(mean, cov)
        dist_unbatched = [
            MultivariateNormal(mean[i], cov[i]) for i in range(mean.size(0))
        ]

        x = dist_batched.sample((10,))
        batched_prob = dist_batched.log_prob(x)
        unbatched_prob = torch.stack(
            [dist_unbatched[i].log_prob(x[:, i]) for i in range(5)]
        ).t()

        self.assertEqual(batched_prob.shape, unbatched_prob.shape)
        self.assertEqual(batched_prob, unbatched_prob, atol=1e-3, rtol=0)

    @expectedFailureMPS
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_multivariate_normal_sample(self):
        # [FUNCTION DOC] Statistical test (seed 0) that MultivariateNormal samples match scipy
        # multivariate_normal for covariance, precision and scale_tril parametrizations.
        # KNOWN ISSUE: the failure message for the precision case labels the matrix as atol.
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        mean = torch.randn(3, requires_grad=True)
        tmp = torch.randn(3, 10)
        cov = (torch.matmul(tmp, tmp.t()) / tmp.size(-1)).requires_grad_()
        prec = cov.inverse().requires_grad_()
        scale_tril = torch.linalg.cholesky(cov).requires_grad_()

        self._check_sampler_sampler(
            MultivariateNormal(mean, cov),
            scipy.stats.multivariate_normal(
                mean.detach().cpu().numpy(), cov.detach().cpu().numpy()
            ),
            f"MultivariateNormal(loc={mean}, cov={cov})",
            multivariate=True,
        )
        self._check_sampler_sampler(
            MultivariateNormal(mean, precision_matrix=prec),
            scipy.stats.multivariate_normal(
                mean.detach().cpu().numpy(), cov.detach().cpu().numpy()
            ),
            f"MultivariateNormal(loc={mean}, atol={prec})",
            multivariate=True,
        )
        self._check_sampler_sampler(
            MultivariateNormal(mean, scale_tril=scale_tril),
            scipy.stats.multivariate_normal(
                mean.detach().cpu().numpy(), cov.detach().cpu().numpy()
            ),
            f"MultivariateNormal(loc={mean}, scale_tril={scale_tril})",
            multivariate=True,
        )

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_multivariate_normal_properties(self):
        # [FUNCTION DOC] Checks MultivariateNormal built from scale_tril: covariance_matrix
        # equals scale_tril times its transpose, covariance times precision is the identity, and
        # scale_tril equals the Cholesky factor of the covariance.
        loc = torch.randn(5)
        scale_tril = transform_to(constraints.lower_cholesky)(torch.randn(5, 5))
        m = MultivariateNormal(loc=loc, scale_tril=scale_tril)
        self.assertEqual(m.covariance_matrix, m.scale_tril.mm(m.scale_tril.t()))
        self.assertEqual(
            m.covariance_matrix.mm(m.precision_matrix), torch.eye(m.event_shape[0])
        )
        self.assertEqual(m.scale_tril, torch.linalg.cholesky(m.covariance_matrix))

    @set_default_dtype_if_supported(torch.double)
    def test_multivariate_normal_moments(self):
        # [FUNCTION DOC] With seed 0 draws 100000 rsamples and checks empirical mean and
        # variance against the distribution's mean and variance.
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        mean = torch.randn(5)
        scale_tril = transform_to(constraints.lower_cholesky)(torch.randn(5, 5))
        d = MultivariateNormal(mean, scale_tril=scale_tril)
        samples = d.rsample((100000,))
        empirical_mean = samples.mean(0)
        self.assertEqual(d.mean, empirical_mean, atol=0.01, rtol=0)
        empirical_var = samples.var(0)
        self.assertEqual(d.variance, empirical_var, atol=0.05, rtol=0)

    # We applied same tests in Multivariate Normal distribution for Wishart distribution
    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_wishart_shape(self):
        # [FUNCTION DOC] Wishart shape and gradient checks: sample sizes (event shape of two
        # matrix dims) for unbatched, batched and multi-batched df with covariance, precision
        # and scale_tril parametrizations and several sample shapes, then gradchecks log_prob
        # for each. Mirrors the MultivariateNormal tests.
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        ndim = 3

        df = torch.rand(5, requires_grad=True) + ndim
        df_no_batch = torch.rand([], requires_grad=True) + ndim
        df_multi_batch = torch.rand(6, 5, requires_grad=True) + ndim

        # construct PSD covariance
        tmp = torch.randn(ndim, 10)
        cov = (torch.matmul(tmp, tmp.t()) / tmp.size(-1)).requires_grad_()
        prec = cov.inverse().requires_grad_()
        scale_tril = torch.linalg.cholesky(cov).requires_grad_()

        # construct batch of PSD covariances
        tmp = torch.randn(6, 5, ndim, 10)
        cov_batched = (tmp.unsqueeze(-2) * tmp.unsqueeze(-3)).mean(-1).requires_grad_()
        prec_batched = cov_batched.inverse()
        scale_tril_batched = torch.linalg.cholesky(cov_batched)

        # ensure that sample, batch, event shapes all handled correctly
        self.assertEqual(Wishart(df, cov).sample().size(), (5, ndim, ndim))
        self.assertEqual(Wishart(df_no_batch, cov).sample().size(), (ndim, ndim))
        self.assertEqual(
            Wishart(df_multi_batch, cov).sample().size(), (6, 5, ndim, ndim)
        )
        self.assertEqual(Wishart(df, cov).sample((2,)).size(), (2, 5, ndim, ndim))
        self.assertEqual(Wishart(df_no_batch, cov).sample((2,)).size(), (2, ndim, ndim))
        self.assertEqual(
            Wishart(df_multi_batch, cov).sample((2,)).size(), (2, 6, 5, ndim, ndim)
        )
        self.assertEqual(Wishart(df, cov).sample((2, 7)).size(), (2, 7, 5, ndim, ndim))
        self.assertEqual(
            Wishart(df_no_batch, cov).sample((2, 7)).size(), (2, 7, ndim, ndim)
        )
        self.assertEqual(
            Wishart(df_multi_batch, cov).sample((2, 7)).size(), (2, 7, 6, 5, ndim, ndim)
        )
        self.assertEqual(
            Wishart(df, cov_batched).sample((2, 7)).size(), (2, 7, 6, 5, ndim, ndim)
        )
        self.assertEqual(
            Wishart(df_no_batch, cov_batched).sample((2, 7)).size(),
            (2, 7, 6, 5, ndim, ndim),
        )
        self.assertEqual(
            Wishart(df_multi_batch, cov_batched).sample((2, 7)).size(),
            (2, 7, 6, 5, ndim, ndim),
        )
        self.assertEqual(
            Wishart(df, precision_matrix=prec).sample((2, 7)).size(),
            (2, 7, 5, ndim, ndim),
        )
        self.assertEqual(
            Wishart(df, precision_matrix=prec_batched).sample((2, 7)).size(),
            (2, 7, 6, 5, ndim, ndim),
        )
        self.assertEqual(
            Wishart(df, scale_tril=scale_tril).sample((2, 7)).size(),
            (2, 7, 5, ndim, ndim),
        )
        self.assertEqual(
            Wishart(df, scale_tril=scale_tril_batched).sample((2, 7)).size(),
            (2, 7, 6, 5, ndim, ndim),
        )

        # check gradients
        # Modified and applied the same tests for multivariate_normal
        def wishart_log_prob_gradcheck(
            df=None, covariance=None, precision=None, scale_tril=None
        ):
            # [FUNCTION DOC] HELPER: Draws a Wishart sample from df and exactly one of
            # covariance, precision, scale_tril and runs gradcheck of log_prob with respect to
            # the sample and given parameters.
            wishart_samples = (
                Wishart(df, covariance, precision, scale_tril).sample().requires_grad_()
            )

            def gradcheck_func(samples, nu, sigma, prec, scale_tril):
                # [FUNCTION DOC] HELPER: Function under gradcheck. Symmetrizes covariance and
                # precision, takes the lower triangle of scale_tril, and returns the Wishart
                # log_prob of samples.
                if sigma is not None:
                    sigma = 0.5 * (sigma + sigma.mT)  # Ensure symmetry of covariance
                if prec is not None:
                    prec = 0.5 * (prec + prec.mT)  # Ensure symmetry of precision
                if scale_tril is not None:
                    scale_tril = scale_tril.tril()
                return Wishart(nu, sigma, prec, scale_tril).log_prob(samples)

            gradcheck(
                gradcheck_func,
                (wishart_samples, df, covariance, precision, scale_tril),
                raise_exception=True,
            )

        wishart_log_prob_gradcheck(df, cov)
        wishart_log_prob_gradcheck(df_multi_batch, cov)
        wishart_log_prob_gradcheck(df_multi_batch, cov_batched)
        wishart_log_prob_gradcheck(df, None, prec)
        wishart_log_prob_gradcheck(df_no_batch, None, prec_batched)
        wishart_log_prob_gradcheck(df, None, None, scale_tril)
        wishart_log_prob_gradcheck(df_no_batch, None, None, scale_tril_batched)

    @skipMPS  # flaky failure
    def test_wishart_stable_with_precision_matrix(self):
        # [FUNCTION DOC] Regression check (seed 0) that constructing a Wishart from an
        # RBF-kernel precision matrix succeeds; no values are asserted. Skipped on MPS because
        # of flakiness.
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        ndim = 10
        x = torch.randn(ndim)
        P = torch.exp(-((x - x.unsqueeze(-1)) ** 2))  # RBF kernel
        Wishart(torch.tensor(ndim), precision_matrix=P)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    @set_default_dtype_if_supported(torch.double)
    def test_wishart_log_prob(self):
        # [FUNCTION DOC] Compares Wishart log_prob from covariance, precision and scale_tril
        # with scipy wishart logpdf (df shifted by one for scipy older than 1.7.0), then
        # verifies batched log_prob equals a stack of unbatched distributions. Skipped
        # without NumPy.
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        ndim = 3
        df = torch.rand([], requires_grad=True) + ndim - 1
        # SciPy allowed ndim -1 < df < ndim for Wishar distribution after version 1.7.0
        if version.parse(scipy.__version__) < version.parse("1.7.0"):
            df += 1.0
        tmp = torch.randn(ndim, 10)
        cov = (torch.matmul(tmp, tmp.t()) / tmp.size(-1)).requires_grad_()
        prec = cov.inverse().requires_grad_()
        scale_tril = torch.linalg.cholesky(cov).requires_grad_()

        # check that logprob values match scipy logpdf,
        # and that covariance and scale_tril parameters are equivalent
        dist1 = Wishart(df, cov)
        dist2 = Wishart(df, precision_matrix=prec)
        dist3 = Wishart(df, scale_tril=scale_tril)
        ref_dist = scipy.stats.wishart(df.item(), cov.detach().cpu().numpy())

        x = dist1.sample((1000,))
        expected = ref_dist.logpdf(x.transpose(0, 2).cpu().numpy())

        self.assertEqual(
            0.0,
            np.mean((dist1.log_prob(x).detach().cpu().numpy() - expected) ** 2),
            atol=1e-3,
            rtol=0,
        )
        self.assertEqual(
            0.0,
            np.mean((dist2.log_prob(x).detach().cpu().numpy() - expected) ** 2),
            atol=1e-3,
            rtol=0,
        )
        self.assertEqual(
            0.0,
            np.mean((dist3.log_prob(x).detach().cpu().numpy() - expected) ** 2),
            atol=1e-3,
            rtol=0,
        )

        # Double-check that batched versions behave the same as unbatched
        df = torch.rand(5, requires_grad=True) + ndim - 1
        # SciPy allowed ndim -1 < df < ndim for Wishar distribution after version 1.7.0
        if version.parse(scipy.__version__) < version.parse("1.7.0"):
            df += 1.0
        tmp = torch.randn(5, ndim, 10)
        cov = (tmp.unsqueeze(-2) * tmp.unsqueeze(-3)).mean(-1).requires_grad_()

        dist_batched = Wishart(df, cov)
        dist_unbatched = [Wishart(df[i], cov[i]) for i in range(df.size(0))]

        x = dist_batched.sample((1000,))
        batched_prob = dist_batched.log_prob(x)
        unbatched_prob = torch.stack(
            [dist_unbatched[i].log_prob(x[:, i]) for i in range(5)]
        ).t()

        self.assertEqual(batched_prob.shape, unbatched_prob.shape)
        self.assertEqual(batched_prob, unbatched_prob, atol=1e-3, rtol=0)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @skipMPS  # flaky failure
    @set_default_dtype_if_supported(torch.double)
    def test_wishart_sample(self):
        # [FUNCTION DOC] Statistical test (seed 0) that Wishart samples from covariance,
        # precision and scale_tril parametrizations match scipy wishart (df adjusted for old
        # scipy). Skipped on MPS.
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        ndim = 3
        df = torch.rand([], requires_grad=True) + ndim - 1
        # SciPy allowed ndim -1 < df < ndim for Wishar distribution after version 1.7.0
        if version.parse(scipy.__version__) < version.parse("1.7.0"):
            df += 1.0
        tmp = torch.randn(ndim, 10)
        cov = (torch.matmul(tmp, tmp.t()) / tmp.size(-1)).requires_grad_()
        prec = cov.inverse().requires_grad_()
        scale_tril = torch.linalg.cholesky(cov).requires_grad_()

        ref_dist = scipy.stats.wishart(df.item(), cov.detach().cpu().numpy())

        self._check_sampler_sampler(
            Wishart(df, cov),
            ref_dist,
            f"Wishart(df={df}, covariance_matrix={cov})",
            multivariate=True,
        )
        self._check_sampler_sampler(
            Wishart(df, precision_matrix=prec),
            ref_dist,
            f"Wishart(df={df}, precision_matrix={prec})",
            multivariate=True,
        )
        self._check_sampler_sampler(
            Wishart(df, scale_tril=scale_tril),
            ref_dist,
            f"Wishart(df={df}, scale_tril={scale_tril})",
            multivariate=True,
        )

    def test_wishart_properties(self):
        # [FUNCTION DOC] Checks Wishart built from scale_tril: covariance_matrix equals
        # scale_tril times its transpose, covariance times precision is the identity, and
        # scale_tril equals the Cholesky factor of the covariance.
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        ndim = 5
        df = torch.rand([]) + ndim - 1
        scale_tril = transform_to(constraints.lower_cholesky)(torch.randn(ndim, ndim))
        m = Wishart(df=df, scale_tril=scale_tril)
        self.assertEqual(m.covariance_matrix, m.scale_tril.mm(m.scale_tril.t()))
        self.assertEqual(
            m.covariance_matrix.mm(m.precision_matrix), torch.eye(m.event_shape[0])
        )
        self.assertEqual(m.scale_tril, torch.linalg.cholesky(m.covariance_matrix))

    def test_wishart_moments(self):
        # [FUNCTION DOC] With seed 0 draws 900000 (ndim squared times 100000) rsamples from a 3
        # by 3 Wishart and checks empirical mean and variance against the distribution's within
        # absolute tolerance 0.5.
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        ndim = 3
        df = torch.rand([]) + ndim - 1
        scale_tril = transform_to(constraints.lower_cholesky)(torch.randn(ndim, ndim))
        d = Wishart(df=df, scale_tril=scale_tril)
        samples = d.rsample((ndim * ndim * 100000,))
        empirical_mean = samples.mean(0)
        self.assertEqual(d.mean, empirical_mean, atol=0.5, rtol=0)
        empirical_var = samples.var(0)
        self.assertEqual(d.variance, empirical_var, atol=0.5, rtol=0)

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_exponential(self):
        # [FUNCTION DOC] Exponential checks: sample shapes, gradcheck, rsample gradient equal to
        # minus the standard exponential draw over rate squared, log_prob against log rate minus
        # rate times x, forward AD of exponential_, and mean/variance of exponential_ for
        # several dtypes and rates.
        rate = torch.randn(5, 5).abs().requires_grad_()
        rate_1d = torch.randn(1).abs().requires_grad_()
        self.assertEqual(Exponential(rate).sample().size(), (5, 5))
        self.assertEqual(Exponential(rate).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Exponential(rate_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Exponential(rate_1d).sample().size(), (1,))
        self.assertEqual(Exponential(0.2).sample((1,)).size(), (1,))
        self.assertEqual(Exponential(50.0).sample((1,)).size(), (1,))

        self._gradcheck_log_prob(Exponential, (rate,))
        state = torch.get_rng_state()
        eps = rate.new(rate.size()).exponential_()
        torch.set_rng_state(state)
        z = Exponential(rate).rsample()
        z.backward(torch.ones_like(z))
        self.assertEqual(rate.grad, -eps / rate**2)
        rate.grad.zero_()
        self.assertEqual(z.size(), (5, 5))

        def ref_log_prob(idx, x, log_prob):
            # [FUNCTION DOC] HELPER: Asserts log_prob equals log rate minus rate times x for the
            # batch element.
            m = rate.view(-1)[idx]
            expected = math.log(m) - m * x
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(Exponential(rate), ref_log_prob)
        self._check_forward_ad(lambda x: x.exponential_())

        def mean_var(lambd, sample):
            # [FUNCTION DOC] HELPER: Fills sample in place with exponential_ draws for rate
            # lambd and asserts the empirical mean is about 1 over lambd and variance about 1
            # over lambd squared.
            sample.exponential_(lambd)
            mean = sample.float().mean()
            var = sample.float().var()
            self.assertEqual((1.0 / lambd), mean, atol=2e-2, rtol=2e-2)
            self.assertEqual((1.0 / lambd) ** 2, var, atol=2e-2, rtol=2e-2)

        for dtype in [torch.float, torch.double, torch.bfloat16, torch.float16]:
            for lambd in [0.2, 0.5, 1.0, 1.5, 2.0, 5.0]:
                sample_len = 50000
                mean_var(lambd, torch.rand(sample_len, dtype=dtype))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_exponential_sample(self):
        # [FUNCTION DOC] Statistical test (seed 1) of Exponential samples against scipy expon
        # for several rates.
        set_rng_seed(1)  # see Note [Randomized statistical tests]
        for rate in [1e-5, 1.0, 10.0]:
            self._check_sampler_sampler(
                Exponential(rate),
                scipy.stats.expon(scale=1.0 / rate),
                f"Exponential(rate={rate})",
            )

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_laplace(self):
        # [FUNCTION DOC] Laplace checks: sample shapes, near-deterministic samples for tiny
        # scale, gradchecks, rsample gradients (loc one, scale from the uniform draw) and
        # log_prob against the analytic density.
        loc = torch.randn(5, 5, requires_grad=True)
        scale = torch.randn(5, 5).abs().requires_grad_()
        loc_1d = torch.randn(1, requires_grad=True)
        scale_1d = torch.randn(1).abs().requires_grad_()
        loc_delta = torch.tensor([1.0, 0.0])
        scale_delta = torch.tensor([1e-5, 1e-5])
        self.assertEqual(Laplace(loc, scale).sample().size(), (5, 5))
        self.assertEqual(Laplace(loc, scale).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Laplace(loc_1d, scale_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Laplace(loc_1d, scale_1d).sample().size(), (1,))
        self.assertEqual(Laplace(0.2, 0.6).sample((1,)).size(), (1,))
        self.assertEqual(Laplace(-0.7, 50.0).sample((1,)).size(), (1,))

        # sample check for extreme value of mean, std
        set_rng_seed(0)
        self.assertEqual(
            Laplace(loc_delta, scale_delta).sample(sample_shape=(1, 2)),
            torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
            atol=1e-4,
            rtol=0,
        )

        self._gradcheck_log_prob(Laplace, (loc, scale))
        self._gradcheck_log_prob(Laplace, (loc, 1.0))
        self._gradcheck_log_prob(Laplace, (0.0, scale))

        state = torch.get_rng_state()
        eps = torch.ones_like(loc).uniform_(-0.5, 0.5)
        torch.set_rng_state(state)
        z = Laplace(loc, scale).rsample()
        z.backward(torch.ones_like(z))
        self.assertEqual(loc.grad, torch.ones_like(loc))
        self.assertEqual(scale.grad, -eps.sign() * torch.log1p(-2 * eps.abs()))
        loc.grad.zero_()
        scale.grad.zero_()
        self.assertEqual(z.size(), (5, 5))

        def ref_log_prob(idx, x, log_prob):
            # [FUNCTION DOC] HELPER: Asserts log_prob equals minus log of twice the scale minus
            # |x - loc| over scale.
            m = loc.view(-1)[idx]
            s = scale.view(-1)[idx]
            expected = -math.log(2 * s) - abs(x - m) / s
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(Laplace(loc, scale), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @set_default_dtype_if_supported(torch.double)
    def test_laplace_sample(self):
        # [FUNCTION DOC] Statistical test (seed 1) of Laplace samples against scipy laplace over
        # locs and scales.
        set_rng_seed(1)  # see Note [Randomized statistical tests]
        for loc, scale in product([-1.0, 0.0, 1.0], [0.1, 1.0, 10.0]):
            self._check_sampler_sampler(
                Laplace(loc, scale),
                scipy.stats.laplace(loc=loc, scale=scale),
                f"Laplace(loc={loc}, scale={scale})",
            )

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_gamma_shape(self):
        # [FUNCTION DOC] Gamma sample shapes for tensor and float parameters and log_prob
        # against scipy gamma logpdf. Skipped without NumPy.
        alpha = torch.randn(2, 3).exp().requires_grad_()
        beta = torch.randn(2, 3).exp().requires_grad_()
        alpha_1d = torch.randn(1).exp().requires_grad_()
        beta_1d = torch.randn(1).exp().requires_grad_()
        self.assertEqual(Gamma(alpha, beta).sample().size(), (2, 3))
        self.assertEqual(Gamma(alpha, beta).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(Gamma(alpha_1d, beta_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Gamma(alpha_1d, beta_1d).sample().size(), (1,))
        self.assertEqual(Gamma(0.5, 0.5).sample().size(), ())
        self.assertEqual(Gamma(0.5, 0.5).sample((1,)).size(), (1,))

        def ref_log_prob(idx, x, log_prob):
            # [FUNCTION DOC] HELPER: Asserts log_prob equals the scipy gamma logpdf for the
            # batch element's concentration and rate.
            a = alpha.view(-1)[idx].detach().cpu()
            b = beta.view(-1)[idx].detach().cpu()
            expected = scipy.stats.gamma.logpdf(x.cpu(), a, scale=1 / b)
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(Gamma(alpha, beta), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_gamma_sample(self):
        # [FUNCTION DOC] Statistical test (seed 0) of Gamma samples against scipy gamma.
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for alpha, beta in product([0.1, 1.0, 5.0], [0.1, 1.0, 10.0]):
            self._check_sampler_sampler(
                Gamma(alpha, beta),
                scipy.stats.gamma(alpha, scale=1.0 / beta),
                f"Gamma(concentration={alpha}, rate={beta})",
            )

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_gamma_tensor_sample(self):
        # [FUNCTION DOC] Statistical test of Gamma with 1-element tensor parameters against
        # scipy gamma, using a stricter failure_rate of 1e-4.
        set_rng_seed(0)
        for alpha, beta in product([0.1, 1.0, 5.0], [0.1, 1.0, 10.0]):
            a, b = (
                torch.tensor([alpha]),
                torch.tensor([beta]),
            )
            self._check_sampler_sampler(
                Gamma(a, b),
                scipy.stats.gamma(alpha, scale=1.0 / beta),
                f"Gamma(alpha={alpha}, beta={beta})",
                failure_rate=1e-4,
            )

    def test_gamma_sample_generator(self):
        # [FUNCTION DOC] Checks Gamma.sample accepts a generator argument: requested shapes are
        # honored, sampling without generator works, equal seeds give equal samples and
        # different seeds give different samples.
        gamma = Gamma(torch.tensor(2.0), torch.tensor(1.0))
        device = gamma.concentration.device
        # sampling with a generator honors the requested shape
        gen = torch.Generator(device=device).manual_seed(42)
        self.assertEqual(gamma.sample((5,), generator=gen).size(), (5,))
        self.assertEqual(gamma.sample((5, 3), generator=gen).size(), (5, 3))
        # sampling without a generator still works
        self.assertEqual(gamma.sample((5,)).size(), (5,))
        # same seed produces identical samples
        gen1 = torch.Generator(device=device).manual_seed(42)
        gen2 = torch.Generator(device=device).manual_seed(42)
        self.assertEqual(
            gamma.sample((5,), generator=gen1), gamma.sample((5,), generator=gen2)
        )
        # different seeds produce different samples
        gen1 = torch.Generator(device=device).manual_seed(42)
        gen2 = torch.Generator(device=device).manual_seed(99)
        self.assertFalse(
            torch.allclose(
                gamma.sample((5,), generator=gen1),
                gamma.sample((5,), generator=gen2),
            )
        )

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_pareto(self):
        # [FUNCTION DOC] Pareto checks: infinite mean and variance for alpha 0.5, sample shapes,
        # and log_prob against scipy pareto logpdf.
        scale = torch.randn(2, 3).abs().requires_grad_()
        alpha = torch.randn(2, 3).abs().requires_grad_()
        scale_1d = torch.randn(1).abs().requires_grad_()
        alpha_1d = torch.randn(1).abs().requires_grad_()
        self.assertEqual(Pareto(scale_1d, 0.5).mean, inf)
        self.assertEqual(Pareto(scale_1d, 0.5).variance, inf)
        self.assertEqual(Pareto(scale, alpha).sample().size(), (2, 3))
        self.assertEqual(Pareto(scale, alpha).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(Pareto(scale_1d, alpha_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Pareto(scale_1d, alpha_1d).sample().size(), (1,))
        self.assertEqual(Pareto(1.0, 1.0).sample().size(), ())
        self.assertEqual(Pareto(1.0, 1.0).sample((1,)).size(), (1,))

        def ref_log_prob(idx, x, log_prob):
            # [FUNCTION DOC] HELPER: Asserts log_prob equals the scipy pareto logpdf for the
            # batch element.
            s = scale.view(-1)[idx].detach().cpu()
            a = alpha.view(-1)[idx].detach().cpu()
            expected = scipy.stats.pareto.logpdf(x.cpu(), a, scale=s)
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(Pareto(scale, alpha), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_pareto_sample(self):
        # [FUNCTION DOC] Statistical test (seed 1) of Pareto samples against scipy pareto.
        set_rng_seed(1)  # see Note [Randomized statistical tests]
        for scale, alpha in product([0.1, 1.0, 5.0], [0.1, 1.0, 10.0]):
            self._check_sampler_sampler(
                Pareto(scale, alpha),
                scipy.stats.pareto(alpha, scale=scale),
                f"Pareto(scale={scale}, alpha={alpha})",
            )

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_generalized_pareto(self):
        # [FUNCTION DOC] GeneralizedPareto sample shapes and log_prob against scipy
        # genpareto logpdf.
        loc = torch.randn(2, 3).requires_grad_()
        scale = torch.randn(2, 3).abs().requires_grad_()
        concentration = torch.randn(2, 3).requires_grad_()
        loc_1d = torch.randn(1).requires_grad_()
        scale_1d = torch.randn(1).abs().requires_grad_()
        concentration_1d = torch.randn(1).requires_grad_()
        self.assertEqual(
            GeneralizedPareto(loc, scale, concentration).sample().size(), (2, 3)
        )
        self.assertEqual(
            GeneralizedPareto(loc, scale, concentration).sample((5,)).size(), (5, 2, 3)
        )
        self.assertEqual(
            GeneralizedPareto(loc_1d, scale_1d, concentration_1d).sample((1,)).size(),
            (1, 1),
        )
        self.assertEqual(
            GeneralizedPareto(loc_1d, scale_1d, concentration_1d).sample().size(), (1,)
        )
        self.assertEqual(GeneralizedPareto(1.0, 1.0, 1.0).sample().size(), ())
        self.assertEqual(GeneralizedPareto(1.0, 1.0, 1.0).sample((1,)).size(), (1,))

        def ref_log_prob(idx, x, log_prob):
            # [FUNCTION DOC] HELPER: Asserts log_prob equals the scipy genpareto logpdf for the
            # batch element.
            l = loc.view(-1)[idx].detach().cpu()
            s = scale.view(-1)[idx].detach().cpu()
            c = concentration.view(-1)[idx].detach().cpu()
            expected = scipy.stats.genpareto.logpdf(x.cpu(), c, loc=l, scale=s)
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(GeneralizedPareto(loc, scale, concentration), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_generalized_pareto_sample(self):
        # [FUNCTION DOC] Statistical test (seed 1) of GeneralizedPareto samples against scipy
        # genpareto over loc, scale and concentration grids, with failure_rate 7e-4.
        set_rng_seed(1)  # see note [Randomized statistical tests]
        for loc, scale, concentration in product(
            [-1.0, 0.0, 1.0], [0.1, 1.0, 10.0], [-0.5, 0.0, 0.5]
        ):
            self._check_sampler_sampler(
                GeneralizedPareto(loc, scale, concentration),
                scipy.stats.genpareto(c=concentration, loc=loc, scale=scale),
                f"GeneralizedPareto(loc={loc}, scale={scale}, concentration={concentration})",
                failure_rate=7e-4,
            )

    def test_gumbel(self, device):
        # [FUNCTION DOC] Gumbel checks: sample shapes, saturating cdf values in float32 (and
        # float64 except on MPS) far in each tail with validation off, and log_prob against
        # scipy gumbel_r logpdf. Takes the test device.
        loc = torch.randn(2, 3, requires_grad=True)
        scale = torch.randn(2, 3).abs().requires_grad_()
        loc_1d = torch.randn(1, requires_grad=True)
        scale_1d = torch.randn(1).abs().requires_grad_()
        self.assertEqual(Gumbel(loc, scale).sample().size(), (2, 3))
        self.assertEqual(Gumbel(loc, scale).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(Gumbel(loc_1d, scale_1d).sample().size(), (1,))
        self.assertEqual(Gumbel(loc_1d, scale_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Gumbel(1.0, 1.0).sample().size(), ())
        self.assertEqual(Gumbel(1.0, 1.0).sample((1,)).size(), (1,))
        self.assertEqual(
            Gumbel(
                torch.tensor(0.0, dtype=torch.float32),
                torch.tensor(1.0, dtype=torch.float32),
                validate_args=False,
            ).cdf(20.0),
            1.0,
            atol=1e-4,
            rtol=0,
        )
        self.assertEqual(
            Gumbel(
                torch.tensor(0.0, dtype=torch.float32),
                torch.tensor(1.0, dtype=torch.float32),
                validate_args=False,
            ).cdf(-5.0),
            0.0,
            atol=1e-4,
            rtol=0,
        )
        if torch.device(device).type != "mps":
            self.assertEqual(
                Gumbel(
                    torch.tensor(0.0, dtype=torch.float64),
                    torch.tensor(1.0, dtype=torch.float64),
                    validate_args=False,
                ).cdf(50.0),
                1.0,
                atol=1e-4,
                rtol=0,
            )
            self.assertEqual(
                Gumbel(
                    torch.tensor(0.0, dtype=torch.float64),
                    torch.tensor(1.0, dtype=torch.float64),
                    validate_args=False,
                ).cdf(-10.0),
                0.0,
                atol=1e-8,
                rtol=0,
            )

        def ref_log_prob(idx, x, log_prob):
            # [FUNCTION DOC] HELPER: Asserts log_prob equals the scipy gumbel_r logpdf for the
            # batch element.
            l = loc.view(-1)[idx].detach().cpu()
            s = scale.view(-1)[idx].detach().cpu()
            expected = scipy.stats.gumbel_r.logpdf(x.cpu(), loc=l, scale=s)
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(Gumbel(loc, scale), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @set_default_dtype_if_supported(torch.double)
    def test_gumbel_sample(self):
        # [FUNCTION DOC] Statistical test (seed 1) of Gumbel samples against scipy gumbel_r over
        # locs and scales.
        set_rng_seed(1)  # see note [Randomized statistical tests]
        for loc, scale in product([-5.0, -1.0, -0.1, 0.1, 1.0, 5.0], [0.1, 1.0, 10.0]):
            self._check_sampler_sampler(
                Gumbel(loc, scale),
                scipy.stats.gumbel_r(loc=loc, scale=scale),
                f"Gumbel(loc={loc}, scale={scale})",
            )

    def test_kumaraswamy_shape(self):
        # [FUNCTION DOC] Checks Kumaraswamy sample shapes for tensor and float concentrations.
        concentration1 = torch.randn(2, 3).abs().requires_grad_()
        concentration0 = torch.randn(2, 3).abs().requires_grad_()
        concentration1_1d = torch.randn(1).abs().requires_grad_()
        concentration0_1d = torch.randn(1).abs().requires_grad_()
        self.assertEqual(
            Kumaraswamy(concentration1, concentration0).sample().size(), (2, 3)
        )
        self.assertEqual(
            Kumaraswamy(concentration1, concentration0).sample((5,)).size(), (5, 2, 3)
        )
        self.assertEqual(
            Kumaraswamy(concentration1_1d, concentration0_1d).sample().size(), (1,)
        )
        self.assertEqual(
            Kumaraswamy(concentration1_1d, concentration0_1d).sample((1,)).size(),
            (1, 1),
        )
        self.assertEqual(Kumaraswamy(1.0, 1.0).sample().size(), ())
        self.assertEqual(Kumaraswamy(1.0, 1.0).sample((1,)).size(), (1,))

    # Kumaraswamy distribution is not implemented in SciPy
    # Hence these tests are explicit
    def test_kumaraswamy_mean_variance(self):
        # [FUNCTION DOC] Compares Kumaraswamy mean and variance with empirical values from
        # 60000 samples (maximum absolute error below 0.01, ignoring NaN), since SciPy has
        # no Kumaraswamy.
        c1_1 = torch.randn(2, 3).abs().requires_grad_()
        c0_1 = torch.randn(2, 3).abs().requires_grad_()
        c1_2 = torch.randn(4).abs().requires_grad_()
        c0_2 = torch.randn(4).abs().requires_grad_()
        cases = [(c1_1, c0_1), (c1_2, c0_2)]
        for i, (a, b) in enumerate(cases):
            m = Kumaraswamy(a, b)
            samples = m.sample((60000,))
            expected = samples.mean(0)
            actual = m.mean
            error = (expected - actual).abs()
            max_error = max(error[error == error])
            self.assertLess(
                max_error,
                0.01,
                lambda msg: f"{msg}\nKumaraswamy example {i + 1}/{len(cases)}, incorrect .mean",
            )
            expected = samples.var(0)
            actual = m.variance
            error = (expected - actual).abs()
            max_error = max(error[error == error])
            self.assertLess(
                max_error,
                0.01,
                lambda msg: f"{msg}\nKumaraswamy example {i + 1}/{len(cases)}, incorrect .variance",
            )

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_fishersnedecor(self):
        # [FUNCTION DOC] FisherSnedecor checks: NaN mean/variance for small df, sample shapes
        # and log_prob against scipy f logpdf. Skipped without NumPy.
        df1 = torch.randn(2, 3).abs().requires_grad_()
        df2 = torch.randn(2, 3).abs().requires_grad_()
        df1_1d = torch.randn(1).abs()
        df2_1d = torch.randn(1).abs()
        self.assertTrue(is_all_nan(FisherSnedecor(1, 2).mean))
        self.assertTrue(is_all_nan(FisherSnedecor(1, 4).variance))
        self.assertEqual(FisherSnedecor(df1, df2).sample().size(), (2, 3))
        self.assertEqual(FisherSnedecor(df1, df2).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(FisherSnedecor(df1_1d, df2_1d).sample().size(), (1,))
        self.assertEqual(FisherSnedecor(df1_1d, df2_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(FisherSnedecor(1.0, 1.0).sample().size(), ())
        self.assertEqual(FisherSnedecor(1.0, 1.0).sample((1,)).size(), (1,))

        def ref_log_prob(idx, x, log_prob):
            # [FUNCTION DOC] HELPER: Asserts log_prob equals the scipy f logpdf for the
            # batch element.
            f1 = df1.view(-1)[idx].detach()
            f2 = df2.view(-1)[idx].detach()
            expected = scipy.stats.f.logpdf(x.cpu(), f1.cpu(), f2.cpu())
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(FisherSnedecor(df1, df2), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @expectedFailureMPS
    def test_fishersnedecor_sample(self):
        # [FUNCTION DOC] Statistical test (seed 1) of FisherSnedecor samples against scipy f
        # over a grid of df values. Expected failure on MPS.
        set_rng_seed(1)  # see note [Randomized statistical tests]
        for df1, df2 in product([0.1, 0.5, 1.0, 5.0, 10.0], [0.1, 0.5, 1.0, 5.0, 10.0]):
            self._check_sampler_sampler(
                FisherSnedecor(df1, df2),
                scipy.stats.f(df1, df2),
                f"FisherSnedecor(loc={df1}, scale={df2})",
            )

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_chi2_shape(self):
        # [FUNCTION DOC] Chi2 sample shapes and log_prob against scipy chi2 logpdf.
        df = torch.randn(2, 3).exp().requires_grad_()
        df_1d = torch.randn(1).exp().requires_grad_()
        self.assertEqual(Chi2(df).sample().size(), (2, 3))
        self.assertEqual(Chi2(df).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(Chi2(df_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Chi2(df_1d).sample().size(), (1,))
        self.assertEqual(
            Chi2(torch.tensor(0.5, requires_grad=True)).sample().size(), ()
        )
        self.assertEqual(Chi2(0.5).sample().size(), ())
        self.assertEqual(Chi2(0.5).sample((1,)).size(), (1,))

        def ref_log_prob(idx, x, log_prob):
            # [FUNCTION DOC] HELPER: Asserts log_prob equals the scipy chi2 logpdf for the
            # batch element.
            d = df.view(-1)[idx].detach()
            expected = scipy.stats.chi2.logpdf(x.cpu(), d.cpu())
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(Chi2(df), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_chi2_sample(self):
        # [FUNCTION DOC] Statistical test (seed 0) of Chi2 samples against scipy chi2.
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for df in [0.1, 1.0, 5.0]:
            self._check_sampler_sampler(
                Chi2(df), scipy.stats.chi2(df), f"Chi2(df={df})"
            )

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_studentT(self):
        # [FUNCTION DOC] StudentT checks: NaN mean/variance for df 1, infinite variance for df
        # 2, sample shapes and log_prob against scipy t logpdf.
        df = torch.randn(2, 3).exp().requires_grad_()
        df_1d = torch.randn(1).exp().requires_grad_()
        self.assertTrue(is_all_nan(StudentT(1).mean))
        self.assertTrue(is_all_nan(StudentT(1).variance))
        self.assertEqual(StudentT(2).variance, inf)
        self.assertEqual(StudentT(df).sample().size(), (2, 3))
        self.assertEqual(StudentT(df).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(StudentT(df_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(StudentT(df_1d).sample().size(), (1,))
        self.assertEqual(
            StudentT(torch.tensor(0.5, requires_grad=True)).sample().size(), ()
        )
        self.assertEqual(StudentT(0.5).sample().size(), ())
        self.assertEqual(StudentT(0.5).sample((1,)).size(), (1,))

        def ref_log_prob(idx, x, log_prob):
            # [FUNCTION DOC] HELPER: Asserts log_prob equals the scipy t logpdf for the
            # batch element.
            d = df.view(-1)[idx].detach().cpu()
            expected = scipy.stats.t.logpdf(x.cpu(), d)
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(StudentT(df), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_studentT_sample(self):
        # [FUNCTION DOC] Statistical test (seed 11) of StudentT samples against scipy t over df,
        # loc and scale grids.
        set_rng_seed(11)  # see Note [Randomized statistical tests]
        for df, loc, scale in product(
            [0.1, 1.0, 5.0, 10.0], [-1.0, 0.0, 1.0], [0.1, 1.0, 10.0]
        ):
            self._check_sampler_sampler(
                StudentT(df=df, loc=loc, scale=scale),
                scipy.stats.t(df=df, loc=loc, scale=scale),
                f"StudentT(df={df}, loc={loc}, scale={scale})",
            )

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    @expectedFailureMPS
    def test_studentT_log_prob(self):
        # [FUNCTION DOC] Compares StudentT log_prob of 10 samples per df/loc/scale combination
        # with scipy t logpdf within 1e-3.
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        num_samples = 10
        for df, loc, scale in product(
            [0.1, 1.0, 5.0, 10.0], [-1.0, 0.0, 1.0], [0.1, 1.0, 10.0]
        ):
            dist = StudentT(df=df, loc=loc, scale=scale)
            x = dist.sample((num_samples,))
            actual_log_prob = dist.log_prob(x)
            for i in range(num_samples):
                expected_log_prob = scipy.stats.t.logpdf(
                    x[i].cpu(), df=df, loc=loc, scale=scale
                )
                self.assertEqual(
                    float(actual_log_prob[i]),
                    float(expected_log_prob),
                    atol=1e-3,
                    rtol=0,
                )

    def test_dirichlet_shape(self):
        # [FUNCTION DOC] Checks Dirichlet sample shapes for 2-d and 1-d concentrations.
        alpha = torch.randn(2, 3).exp().requires_grad_()
        alpha_1d = torch.randn(4).exp().requires_grad_()
        self.assertEqual(Dirichlet(alpha).sample().size(), (2, 3))
        self.assertEqual(Dirichlet(alpha).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(Dirichlet(alpha_1d).sample().size(), (4,))
        self.assertEqual(Dirichlet(alpha_1d).sample((1,)).size(), (1, 4))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @set_default_dtype_if_supported(torch.double)
    def test_dirichlet_log_prob(self, device):
        # [FUNCTION DOC] Compares Dirichlet log_prob of 10 samples with scipy dirichlet logpdf;
        # on MPS samples are converted to double and renormalized because float32 sums deviate
        # from 1 too much for scipy.
        num_samples = 10
        alpha = torch.exp(torch.randn(5))
        dist = Dirichlet(alpha)
        x = dist.sample((num_samples,))
        actual_log_prob = dist.log_prob(x)
        for i in range(num_samples):
            xi_cpu = x[i].cpu()

            # Since MPS doesn't support doubles, the sum of `x[i]` can be
            # slightly more or less than 1, with relative error around 1e-7,
            # which is larger than scipy allows. So convert to double and divide
            # by the sum to bring it much closer to 1.
            if torch.device(device).type == "mps":
                xi_cpu = xi_cpu.double()
                xi_cpu = xi_cpu / xi_cpu.sum()

            expected_log_prob = scipy.stats.dirichlet.logpdf(
                xi_cpu, alpha.cpu().numpy()
            )
            self.assertEqual(actual_log_prob[i], expected_log_prob, atol=1e-3, rtol=0)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_dirichlet_log_prob_zero(self):
        # [FUNCTION DOC] Checks Dirichlet log_prob at a boundary point (x of 0 where alpha is 1)
        # matches scipy and is not NaN.
        # Specifically test the special case where x=0 and alpha=1.  The PDF is
        # proportional to x**(alpha-1), which in this case works out to 0**0=1.
        # The log PDF of this term should therefore be 0.  However, it's easy
        # to accidentally introduce NaNs by calculating log(x) without regard
        # for the value of alpha-1.
        alpha = torch.tensor([1, 2])
        dist = Dirichlet(alpha)
        x = torch.tensor([0, 1])
        actual_log_prob = dist.log_prob(x)
        expected_log_prob = scipy.stats.dirichlet.logpdf(
            x.cpu().numpy(), alpha.cpu().numpy()
        )
        self.assertEqual(actual_log_prob, expected_log_prob, atol=1e-3, rtol=0)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_dirichlet_sample(self):
        # [FUNCTION DOC] Statistical test (seed 0) of Dirichlet samples against scipy dirichlet
        # (multivariate projection).
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        alpha = torch.exp(torch.randn(3))
        self._check_sampler_sampler(
            Dirichlet(alpha),
            scipy.stats.dirichlet(alpha.cpu().numpy()),
            f"Dirichlet(alpha={list(alpha)})",
            multivariate=True,
        )

    def test_dirichlet_mode(self):
        # [FUNCTION DOC] Checks Dirichlet.mode on edge cases, including a zero entry, a single
        # peak and the all-ones case which gives NaN.
        # Test a few edge cases for the Dirichlet distribution mode. This also covers beta distributions.
        concentrations_and_modes = [
            ([2, 2, 1], [0.5, 0.5, 0.0]),
            ([3, 2, 1], [2 / 3, 1 / 3, 0]),
            ([0.5, 0.2, 0.2], [1.0, 0.0, 0.0]),
            ([1, 1, 1], [nan, nan, nan]),
        ]
        for concentration, mode in concentrations_and_modes:
            dist = Dirichlet(torch.tensor(concentration))
            self.assertEqual(dist.mode, torch.tensor(mode))

    def test_beta_shape(self):
        # [FUNCTION DOC] Checks Beta sample shapes for tensor and float concentrations.
        con1 = torch.randn(2, 3).exp().requires_grad_()
        con0 = torch.randn(2, 3).exp().requires_grad_()
        con1_1d = torch.randn(4).exp().requires_grad_()
        con0_1d = torch.randn(4).exp().requires_grad_()
        self.assertEqual(Beta(con1, con0).sample().size(), (2, 3))
        self.assertEqual(Beta(con1, con0).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(Beta(con1_1d, con0_1d).sample().size(), (4,))
        self.assertEqual(Beta(con1_1d, con0_1d).sample((1,)).size(), (1, 4))
        self.assertEqual(Beta(0.1, 0.3).sample().size(), ())
        self.assertEqual(Beta(0.1, 0.3).sample((5,)).size(), (5,))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_beta_log_prob(self):
        # [FUNCTION DOC] Draws 100 scalar Beta distributions with random log-normal
        # concentrations, samples once from each and compares the summed log_prob with scipy
        # beta logpdf within 1e-3. Uses the numpy global RNG (not seeded here). Skipped
        # without NumPy.
        for _ in range(100):
            con1 = np.exp(np.random.normal())
            con0 = np.exp(np.random.normal())
            dist = Beta(con1, con0)
            x = dist.sample()
            actual_log_prob = dist.log_prob(x).sum()
            expected_log_prob = scipy.stats.beta.logpdf(x.cpu(), con1, con0)
            self.assertEqual(
                float(actual_log_prob), float(expected_log_prob), atol=1e-3, rtol=0
            )

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_beta_sample(self):
        # [FUNCTION DOC] Statistical test (seed 1) of Beta samples against scipy beta over a
        # concentration grid, plus a check that very small concentrations (1e-6) in float and
        # double produce a finite positive sample.
        set_rng_seed(1)  # see Note [Randomized statistical tests]
        for con1, con0 in product([0.1, 1.0, 10.0], [0.1, 1.0, 10.0]):
            self._check_sampler_sampler(
                Beta(con1, con0),
                scipy.stats.beta(con1, con0),
                f"Beta(alpha={con1}, beta={con0})",
            )
        # Check that small alphas do not cause NANs.
        for Tensor in [torch.FloatTensor, torch.DoubleTensor]:
            x = Beta(Tensor([1e-6]), Tensor([1e-6])).sample()[0]
            self.assertTrue(
                np.isfinite(x) and x > 0,
                lambda msg: f"{msg}\nInvalid Beta.sample(): {x}",
            )

    @dtypes(torch.float, torch.double)
    @dtypesIfMPS(torch.float)
    @dtypesIfCUDA(torch.double)
    @dtypesIfXPU(torch.double)
    def test_beta_underflow(self, dtype, device):
        # [FUNCTION DOC] Checks Beta with tiny concentrations (0.01) for the given dtype: 50000
        # samples contain no exact 0 or 1 (no gamma underflow), and about half of the samples
        # fall below 0.1 and half above 0.9 within a tolerance that is looser on cuda, xpu and
        # mps. Takes dtype and device.
        # For low values of (alpha, beta), the gamma samples can underflow
        # with float32 and result in a spurious mode at 0.5. To prevent this,
        # torch._sample_dirichlet works with double precision for intermediate
        # calculations.
        set_rng_seed(1)
        num_samples = 50000
        conc = torch.tensor(1e-2, dtype=dtype)
        beta_samples = Beta(conc, conc).sample([num_samples])
        self.assertEqual((beta_samples == 0).sum(), 0)
        self.assertEqual((beta_samples == 1).sum(), 0)
        # assert support is concentrated around 0 and 1
        frac_zeros = float((beta_samples < 0.1).sum()) / num_samples
        frac_ones = float((beta_samples > 0.9).sum()) / num_samples
        # TODO: increase precision once imbalance on GPU is fixed.
        atol = 0.12 if torch.device(device).type in ["cuda", "xpu", "mps"] else 0.05
        self.assertEqual(frac_zeros, 0.5, atol=atol, rtol=0)
        self.assertEqual(frac_ones, 0.5, atol=atol, rtol=0)

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_continuous_bernoulli(self):
        # [FUNCTION DOC] ContinuousBernoulli checks: sample shapes, samples not requiring grad,
        # gradcheck, log_prob against the analytic density with a Taylor-expanded normalizer
        # near probs 0.5 (for probs and logits forms), and entropy against known values
        # including the clamped probs 0 case.
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True)
        r = torch.tensor(0.3, requires_grad=True)
        s = 0.3
        self.assertEqual(ContinuousBernoulli(p).sample((8,)).size(), (8, 3))
        self.assertFalse(ContinuousBernoulli(p).sample().requires_grad)
        self.assertEqual(ContinuousBernoulli(r).sample((8,)).size(), (8,))
        self.assertEqual(ContinuousBernoulli(r).sample().size(), ())
        self.assertEqual(
            ContinuousBernoulli(r).sample((3, 2)).size(),
            (
                3,
                2,
            ),
        )
        self.assertEqual(ContinuousBernoulli(s).sample().size(), ())
        self._gradcheck_log_prob(ContinuousBernoulli, (p,))

        def ref_log_prob(idx, val, log_prob):
            # [FUNCTION DOC] HELPER: Asserts log_prob equals the analytic continuous Bernoulli
            # log-density: val log p plus (1 - val) log(1 - p) plus log normalizing constant
            # (series expansion within 0.001 of 0.5, otherwise the atanh closed form).
            prob = p[idx]
            if prob > 0.499 and prob < 0.501:  # using default value of lim here
                log_norm_const = (
                    math.log(2.0)
                    + 4.0 / 3.0 * math.pow(prob - 0.5, 2)
                    + 104.0 / 45.0 * math.pow(prob - 0.5, 4)
                )
            else:
                log_norm_const = math.log(
                    2.0 * math.atanh(1.0 - 2.0 * prob) / (1.0 - 2.0 * prob)
                )
            res = (
                val * math.log(prob) + (1.0 - val) * math.log1p(-prob) + log_norm_const
            )
            self.assertEqual(log_prob, res)

        self._check_log_prob(ContinuousBernoulli(p), ref_log_prob)
        self._check_log_prob(
            ContinuousBernoulli(logits=p.log() - (-p).log1p()), ref_log_prob
        )

        # check entropy computation
        self.assertEqual(
            ContinuousBernoulli(p).entropy(),
            torch.tensor([-0.02938, -0.07641, -0.00682]),
            atol=1e-4,
            rtol=0,
        )
        # entropy below corresponds to the clamped value of prob when using float 64
        # the value for float32 should be -1.76898
        self.assertEqual(
            ContinuousBernoulli(torch.tensor([0.0])).entropy(),
            torch.tensor([-2.58473]),
            atol=1e-5,
            rtol=0,
        )
        self.assertEqual(
            ContinuousBernoulli(s).entropy(), torch.tensor(-0.02938), atol=1e-4, rtol=0
        )

    def test_continuous_bernoulli_3d(self):
        # [FUNCTION DOC] Checks ContinuousBernoulli sample shapes for a (2, 3, 5)
        # probability tensor.
        p = torch.full((2, 3, 5), 0.5).requires_grad_()
        self.assertEqual(ContinuousBernoulli(p).sample().size(), (2, 3, 5))
        self.assertEqual(
            ContinuousBernoulli(p).sample(sample_shape=(2, 5)).size(), (2, 5, 2, 3, 5)
        )
        self.assertEqual(ContinuousBernoulli(p).sample((2,)).size(), (2, 2, 3, 5))

    def test_lkj_cholesky_log_prob(self):
        # [FUNCTION DOC] Checks LKJCholesky log_prob for dims 2 to 4 with concentration 1: after
        # subtracting the log absolute Jacobian determinant of the map from Cholesky to
        # correlation entries, the density is the same for two samples (and 0.5 for dim 2), and
        # log_prob of a sample with an extra row raises ValueError.
        def tril_cholesky_to_tril_corr(x):
            # [FUNCTION DOC] HELPER: Converts a vector of strictly-lower-triangular Cholesky
            # entries to the vector of strictly-lower-triangular correlation matrix entries
            # (rebuilds the diagonal so rows have unit norm, multiplies by the transpose).
            x = vec_to_tril_matrix(x, -1)
            diag = (1 - (x * x).sum(-1)).sqrt().diag_embed()
            x = x + diag
            return tril_matrix_to_vec(x @ x.T, -1)

        for dim in range(2, 5):
            log_probs = []
            lkj = LKJCholesky(dim, concentration=1.0, validate_args=True)
            for _ in range(2):
                sample = lkj.sample()
                sample_tril = tril_matrix_to_vec(sample, diag=-1)
                log_prob = lkj.log_prob(sample)
                log_abs_det_jacobian = torch.slogdet(
                    jacobian(tril_cholesky_to_tril_corr, sample_tril)
                ).logabsdet
                log_probs.append(log_prob - log_abs_det_jacobian)
            # for concentration=1., the density is uniform over the space of all
            # correlation matrices.
            if dim == 2:
                # for dim=2, pdf = 0.5 (jacobian adjustment factor is 0.)
                self.assertTrue(
                    all(
                        torch.allclose(x, torch.tensor(0.5).log(), atol=1e-10)
                        for x in log_probs
                    )
                )
            self.assertEqual(log_probs[0], log_probs[1])
            invalid_sample = torch.cat([sample, sample.new_ones(1, dim)], dim=0)
            self.assertRaises(ValueError, lambda: lkj.log_prob(invalid_sample))

    @expectedFailureMPS
    def test_independent_shape(self):
        # [FUNCTION DOC] For each example and each reinterpreted_batch_ndims, wraps the
        # distribution in Independent and checks log_prob, sample, has_rsample,
        # enumerate_support, mean, variance and entropy shapes (where implemented) against the
        # base distribution with batch dims absorbed.
        for Dist, params in _get_examples():
            for param in params:
                base_dist = Dist(**param)
                x = base_dist.sample()
                base_log_prob_shape = base_dist.log_prob(x).shape
                for reinterpreted_batch_ndims in range(len(base_dist.batch_shape) + 1):
                    indep_dist = Independent(base_dist, reinterpreted_batch_ndims)
                    indep_log_prob_shape = base_log_prob_shape[
                        : len(base_log_prob_shape) - reinterpreted_batch_ndims
                    ]
                    self.assertEqual(indep_dist.log_prob(x).shape, indep_log_prob_shape)
                    self.assertEqual(
                        indep_dist.sample().shape, base_dist.sample().shape
                    )
                    self.assertEqual(indep_dist.has_rsample, base_dist.has_rsample)
                    if indep_dist.has_rsample:
                        self.assertEqual(
                            indep_dist.sample().shape, base_dist.sample().shape
                        )
                    try:
                        self.assertEqual(
                            indep_dist.enumerate_support().shape,
                            base_dist.enumerate_support().shape,
                        )
                        self.assertEqual(indep_dist.mean.shape, base_dist.mean.shape)
                    except NotImplementedError:
                        pass
                    try:
                        self.assertEqual(
                            indep_dist.variance.shape, base_dist.variance.shape
                        )
                    except NotImplementedError:
                        pass
                    try:
                        self.assertEqual(
                            indep_dist.entropy().shape, indep_log_prob_shape
                        )
                    except NotImplementedError:
                        pass

    @expectedFailureMPS
    def test_independent_expand(self):
        # [FUNCTION DOC] For each example, reinterpreted ndims and expansion shapes, expands an
        # Independent distribution and checks sample shape, log_prob equality with the
        # unexpanded version, and event_shape and batch_shape of the result.
        for Dist, params in _get_examples():
            for param in params:
                base_dist = Dist(**param)
                for reinterpreted_batch_ndims in range(len(base_dist.batch_shape) + 1):
                    for s in [torch.Size(), torch.Size((2,)), torch.Size((2, 3))]:
                        indep_dist = Independent(base_dist, reinterpreted_batch_ndims)
                        expanded_shape = s + indep_dist.batch_shape
                        expanded = indep_dist.expand(expanded_shape)
                        expanded_sample = expanded.sample()
                        expected_shape = expanded_shape + indep_dist.event_shape
                        self.assertEqual(expanded_sample.shape, expected_shape)
                        self.assertEqual(
                            expanded.log_prob(expanded_sample),
                            indep_dist.log_prob(expanded_sample),
                        )
                        self.assertEqual(expanded.event_shape, indep_dist.event_shape)
                        self.assertEqual(expanded.batch_shape, expanded_shape)

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_cdf_icdf_inverse(self):
        # [FUNCTION DOC] For every example with both cdf and icdf implemented, checks
        # icdf(cdf(x)) recovers 20 samples with relative error below 1e-4 (double precision).
        # Tests the invertibility property on the distributions
        for Dist, params in _get_examples():
            for i, param in enumerate(params):
                dist = Dist(**param)
                samples = dist.sample(sample_shape=(20,))
                try:
                    cdf = dist.cdf(samples)
                    actual = dist.icdf(cdf)
                except NotImplementedError:
                    continue
                rel_error = torch.abs(actual - samples) / (1e-10 + torch.abs(samples))
                self.assertLess(
                    rel_error.max(),
                    1e-4,
                    msg="\n".join(
                        [
                            f"{Dist.__name__} example {i + 1}/{len(params)}, icdf(cdf(x)) != x",
                            f"x = {samples}",
                            f"cdf(x) = {cdf}",
                            f"icdf(cdf(x)) = {actual}",
                        ]
                    ),
                )

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_gamma_log_prob_at_boundary(self):
        # [FUNCTION DOC] Checks Gamma log_prob at zero for concentration 0.5, 1 and 2 (infinity,
        # 0, minus infinity) against both the expected values and scipy gamma logpdf.
        for concentration, log_prob in [(0.5, inf), (1, 0), (2, -inf)]:
            dist = Gamma(concentration, 1)
            scipy_dist = scipy.stats.gamma(concentration)
            self.assertAlmostEqual(dist.log_prob(0), log_prob)
            self.assertAlmostEqual(dist.log_prob(0), scipy_dist.logpdf(0))

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_cdf_log_prob(self):
        # [FUNCTION DOC] For every example with cdf and log_prob implemented, checks the
        # autograd derivative of the cdf at a sample equals the pdf (exp of log_prob).
        # Parameters are detached first.
        # Tests if the differentiation of the CDF gives the PDF at a given value
        for Dist, params in _get_examples():
            for i, param in enumerate(params):
                # We do not need grads wrt params here, e.g. shape of gamma distribution.
                param = {
                    key: value.detach() if isinstance(value, torch.Tensor) else value
                    for key, value in param.items()
                }
                dist = Dist(**param)
                samples = dist.sample()
                if not dist.support.is_discrete:
                    samples.requires_grad_()
                try:
                    cdfs = dist.cdf(samples)
                    pdfs = dist.log_prob(samples).exp()
                except NotImplementedError:
                    continue
                cdfs_derivative = grad(cdfs.sum(), [samples])[
                    0
                ]  # this should not be wrapped in torch.abs()
                self.assertEqual(
                    cdfs_derivative,
                    pdfs,
                    msg="\n".join(
                        [
                            f"{Dist.__name__} example {i + 1}/{len(params)}, d(cdf)/dx != pdf(x)",
                            f"x = {samples}",
                            f"cdf = {cdfs}",
                            f"pdf = {pdfs}",
                            f"grad(cdf) = {cdfs_derivative}",
                        ]
                    ),
                )

    def test_valid_parameter_broadcasting(self):
        # [FUNCTION DOC] Checks parameter broadcasting for Normal, FisherSnedecor, Gamma,
        # Gumbel, Kumaraswamy, Laplace, Pareto and StudentT: the sample size must equal the
        # expected broadcast shape, with and without an extra leading sample_shape of (2,).
        # Test correct broadcasting of parameter sizes for distributions that have multiple
        # parameters.
        # example type (distribution instance, expected sample shape)
        valid_examples = [
            (Normal(loc=torch.tensor([0.0, 0.0]), scale=1), (2,)),
            (Normal(loc=0, scale=torch.tensor([1.0, 1.0])), (2,)),
            (Normal(loc=torch.tensor([0.0, 0.0]), scale=torch.tensor([1.0])), (2,)),
            (
                Normal(
                    loc=torch.tensor([0.0, 0.0]), scale=torch.tensor([[1.0], [1.0]])
                ),
                (2, 2),
            ),
            (Normal(loc=torch.tensor([0.0, 0.0]), scale=torch.tensor([[1.0]])), (1, 2)),
            (Normal(loc=torch.tensor([0.0]), scale=torch.tensor([[1.0]])), (1, 1)),
            (FisherSnedecor(df1=torch.tensor([1.0, 1.0]), df2=1), (2,)),
            (FisherSnedecor(df1=1, df2=torch.tensor([1.0, 1.0])), (2,)),
            (
                FisherSnedecor(df1=torch.tensor([1.0, 1.0]), df2=torch.tensor([1.0])),
                (2,),
            ),
            (
                FisherSnedecor(
                    df1=torch.tensor([1.0, 1.0]), df2=torch.tensor([[1.0], [1.0]])
                ),
                (2, 2),
            ),
            (
                FisherSnedecor(df1=torch.tensor([1.0, 1.0]), df2=torch.tensor([[1.0]])),
                (1, 2),
            ),
            (
                FisherSnedecor(df1=torch.tensor([1.0]), df2=torch.tensor([[1.0]])),
                (1, 1),
            ),
            (Gamma(concentration=torch.tensor([1.0, 1.0]), rate=1), (2,)),
            (Gamma(concentration=1, rate=torch.tensor([1.0, 1.0])), (2,)),
            (
                Gamma(
                    concentration=torch.tensor([1.0, 1.0]),
                    rate=torch.tensor([[1.0], [1.0], [1.0]]),
                ),
                (3, 2),
            ),
            (
                Gamma(
                    concentration=torch.tensor([1.0, 1.0]),
                    rate=torch.tensor([[1.0], [1.0]]),
                ),
                (2, 2),
            ),
            (
                Gamma(
                    concentration=torch.tensor([1.0, 1.0]), rate=torch.tensor([[1.0]])
                ),
                (1, 2),
            ),
            (
                Gamma(concentration=torch.tensor([1.0]), rate=torch.tensor([[1.0]])),
                (1, 1),
            ),
            (Gumbel(loc=torch.tensor([0.0, 0.0]), scale=1), (2,)),
            (Gumbel(loc=0, scale=torch.tensor([1.0, 1.0])), (2,)),
            (Gumbel(loc=torch.tensor([0.0, 0.0]), scale=torch.tensor([1.0])), (2,)),
            (
                Gumbel(
                    loc=torch.tensor([0.0, 0.0]), scale=torch.tensor([[1.0], [1.0]])
                ),
                (2, 2),
            ),
            (Gumbel(loc=torch.tensor([0.0, 0.0]), scale=torch.tensor([[1.0]])), (1, 2)),
            (Gumbel(loc=torch.tensor([0.0]), scale=torch.tensor([[1.0]])), (1, 1)),
            (
                Kumaraswamy(
                    concentration1=torch.tensor([1.0, 1.0]), concentration0=1.0
                ),
                (2,),
            ),
            (
                Kumaraswamy(concentration1=1, concentration0=torch.tensor([1.0, 1.0])),
                (2,),
            ),
            (
                Kumaraswamy(
                    concentration1=torch.tensor([1.0, 1.0]),
                    concentration0=torch.tensor([1.0]),
                ),
                (2,),
            ),
            (
                Kumaraswamy(
                    concentration1=torch.tensor([1.0, 1.0]),
                    concentration0=torch.tensor([[1.0], [1.0]]),
                ),
                (2, 2),
            ),
            (
                Kumaraswamy(
                    concentration1=torch.tensor([1.0, 1.0]),
                    concentration0=torch.tensor([[1.0]]),
                ),
                (1, 2),
            ),
            (
                Kumaraswamy(
                    concentration1=torch.tensor([1.0]),
                    concentration0=torch.tensor([[1.0]]),
                ),
                (1, 1),
            ),
            (Laplace(loc=torch.tensor([0.0, 0.0]), scale=1), (2,)),
            (Laplace(loc=0, scale=torch.tensor([1.0, 1.0])), (2,)),
            (Laplace(loc=torch.tensor([0.0, 0.0]), scale=torch.tensor([1.0])), (2,)),
            (
                Laplace(
                    loc=torch.tensor([0.0, 0.0]), scale=torch.tensor([[1.0], [1.0]])
                ),
                (2, 2),
            ),
            (
                Laplace(loc=torch.tensor([0.0, 0.0]), scale=torch.tensor([[1.0]])),
                (1, 2),
            ),
            (Laplace(loc=torch.tensor([0.0]), scale=torch.tensor([[1.0]])), (1, 1)),
            (Pareto(scale=torch.tensor([1.0, 1.0]), alpha=1), (2,)),
            (Pareto(scale=1, alpha=torch.tensor([1.0, 1.0])), (2,)),
            (Pareto(scale=torch.tensor([1.0, 1.0]), alpha=torch.tensor([1.0])), (2,)),
            (
                Pareto(
                    scale=torch.tensor([1.0, 1.0]), alpha=torch.tensor([[1.0], [1.0]])
                ),
                (2, 2),
            ),
            (
                Pareto(scale=torch.tensor([1.0, 1.0]), alpha=torch.tensor([[1.0]])),
                (1, 2),
            ),
            (Pareto(scale=torch.tensor([1.0]), alpha=torch.tensor([[1.0]])), (1, 1)),
            (StudentT(df=torch.tensor([1.0, 1.0]), loc=1), (2,)),
            (StudentT(df=1, scale=torch.tensor([1.0, 1.0])), (2,)),
            (StudentT(df=torch.tensor([1.0, 1.0]), loc=torch.tensor([1.0])), (2,)),
            (
                StudentT(
                    df=torch.tensor([1.0, 1.0]), scale=torch.tensor([[1.0], [1.0]])
                ),
                (2, 2),
            ),
            (StudentT(df=torch.tensor([1.0, 1.0]), loc=torch.tensor([[1.0]])), (1, 2)),
            (StudentT(df=torch.tensor([1.0]), scale=torch.tensor([[1.0]])), (1, 1)),
            (StudentT(df=1.0, loc=torch.zeros(5, 1), scale=torch.ones(3)), (5, 3)),
        ]

        for dist, expected_size in valid_examples:
            actual_size = dist.sample().size()
            self.assertEqual(
                actual_size,
                expected_size,
                msg=lambda msg: f"{msg}\n{dist} actual size: {actual_size} != expected size: {expected_size}",
            )

            sample_shape = torch.Size((2,))
            expected_size = sample_shape + expected_size
            actual_size = dist.sample(sample_shape).size()
            self.assertEqual(
                actual_size,
                expected_size,
                msg=lambda msg: f"{msg}\n{dist} actual size: {actual_size} != expected size: {expected_size}",
            )

    def test_invalid_parameter_broadcasting(self):
        # [FUNCTION DOC] Checks that constructing Normal, FisherSnedecor, Gumbel, Gamma,
        # Kumaraswamy, Laplace, Pareto or StudentT from non-broadcastable parameter shapes
        # raises RuntimeError.
        # invalid broadcasting cases; should throw error
        # example type (distribution class, distribution params)
        invalid_examples = [
            (
                Normal,
                {"loc": torch.tensor([[0, 0]]), "scale": torch.tensor([1, 1, 1, 1])},
            ),
            (
                Normal,
                {
                    "loc": torch.tensor([[[0, 0, 0], [0, 0, 0]]]),
                    "scale": torch.tensor([1, 1]),
                },
            ),
            (
                FisherSnedecor,
                {
                    "df1": torch.tensor([1, 1]),
                    "df2": torch.tensor([1, 1, 1]),
                },
            ),
            (
                Gumbel,
                {"loc": torch.tensor([[0, 0]]), "scale": torch.tensor([1, 1, 1, 1])},
            ),
            (
                Gumbel,
                {
                    "loc": torch.tensor([[[0, 0, 0], [0, 0, 0]]]),
                    "scale": torch.tensor([1, 1]),
                },
            ),
            (
                Gamma,
                {
                    "concentration": torch.tensor([0, 0]),
                    "rate": torch.tensor([1, 1, 1]),
                },
            ),
            (
                Kumaraswamy,
                {
                    "concentration1": torch.tensor([[1, 1]]),
                    "concentration0": torch.tensor([1, 1, 1, 1]),
                },
            ),
            (
                Kumaraswamy,
                {
                    "concentration1": torch.tensor([[[1, 1, 1], [1, 1, 1]]]),
                    "concentration0": torch.tensor([1, 1]),
                },
            ),
            (Laplace, {"loc": torch.tensor([0, 0]), "scale": torch.tensor([1, 1, 1])}),
            (Pareto, {"scale": torch.tensor([1, 1]), "alpha": torch.tensor([1, 1, 1])}),
            (
                StudentT,
                {
                    "df": torch.tensor([1.0, 1.0]),
                    "scale": torch.tensor([1.0, 1.0, 1.0]),
                },
            ),
            (
                StudentT,
                {"df": torch.tensor([1.0, 1.0]), "loc": torch.tensor([1.0, 1.0, 1.0])},
            ),
        ]

        for dist, kwargs in invalid_examples:
            self.assertRaises(RuntimeError, dist, **kwargs)

    def _test_discrete_distribution_mode(self, dist, sanitized_mode, batch_isfinite):
        # [FUNCTION DOC] HELPER: Checks mode optimality for a discrete distribution: for
        # neighbours of the mode (plus or minus one, or the next one-hot class for
        # OneHotCategorical) that are in the support and finite, log_prob at the mode must not
        # be smaller than at the neighbour (1e-12 tolerance). Entries without a valid neighbour
        # are replaced by fresh samples and masked out.
        # We cannot easily check the mode for discrete distributions, but we can look left and right
        # to ensure the log probability is smaller than at the mode.
        for step in [-1, 1]:
            log_prob_mode = dist.log_prob(sanitized_mode)
            if isinstance(dist, OneHotCategorical):
                idx = (dist._categorical.mode + 1) % dist.probs.shape[-1]
                other = torch.nn.functional.one_hot(
                    idx, num_classes=dist.probs.shape[-1]
                ).to(dist.mode)
            else:
                other = dist.mode + step
            mask = batch_isfinite & dist.support.check(other)
            self.assertTrue(mask.any() or dist.mode.unique().numel() == 1)
            # Add a dimension to the right if the event shape is not a scalar, e.g. OneHotCategorical.
            other = torch.where(
                mask[..., None] if mask.ndim < other.ndim else mask,
                other,
                dist.sample(),
            )
            log_prob_other = dist.log_prob(other)
            delta = log_prob_mode - log_prob_other
            self.assertTrue(
                (-1e-12 < delta[mask].detach()).all()
            )  # Allow up to 1e-12 rounding error.

    def _test_continuous_distribution_mode(self, dist, sanitized_mode, batch_isfinite):
        # [FUNCTION DOC] HELPER: Checks mode optimality for a continuous distribution: perturbs
        # the (sanitized) mode by 1e-5 uniform noise in the unconstrained space of the support
        # (10 points), maps back and asserts log_prob at the mode is not lower, treating equal
        # infinities as ties, for every batch element with a finite mode.
        # We perturb the mode in the unconstrained space and expect the log probability to decrease.
        num_points = 10
        transform = transform_to(dist.support)
        unconstrained_mode = transform.inv(sanitized_mode)
        perturbation = 1e-5 * (
            torch.rand((num_points,) + unconstrained_mode.shape) - 0.5
        )
        perturbed_mode = transform(perturbation + unconstrained_mode)
        log_prob_mode = dist.log_prob(sanitized_mode)
        log_prob_other = dist.log_prob(perturbed_mode)
        delta = log_prob_mode - log_prob_other

        # We pass the test with a small tolerance to allow for rounding and manually set the
        # difference to zero if both log probs are infinite with the same sign.
        both_infinite_with_same_sign = (log_prob_mode == log_prob_other) & (
            log_prob_mode.abs() == inf
        )
        delta[both_infinite_with_same_sign] = 0.0
        ordering = (delta > -1e-12).all(axis=0)
        self.assertTrue(ordering[batch_isfinite].all())

    @expectedFailureMPS
    @set_default_dtype_if_supported(torch.double)
    def test_mode(self):
        # [FUNCTION DOC] For every example distribution checks Distribution.mode: classes
        # without a mode and TransformedDistribution raise NotImplementedError; otherwise modes
        # are either fully finite or fully non-finite per event, undefined modes are replaced by
        # samples, the discrete or continuous optimality helper is run, and log_prob at the mode
        # is never NaN.
        discrete_distributions = (
            Bernoulli,
            Binomial,
            Categorical,
            Geometric,
            NegativeBinomial,
            OneHotCategorical,
            Poisson,
        )
        no_mode_available = (
            ContinuousBernoulli,
            LKJCholesky,
            LogisticNormal,
            MixtureSameFamily,
            Multinomial,
            RelaxedBernoulli,
            RelaxedOneHotCategorical,
        )

        for dist_cls, params in _get_examples():
            for param in params:
                dist = dist_cls(**param)
                if (
                    isinstance(dist, no_mode_available)
                    or type(dist) is TransformedDistribution
                ):
                    with self.assertRaises(NotImplementedError):
                        dist.mode
                    continue

                # Check that either all or no elements in the event shape are nan: the mode cannot be
                # defined for part of an event.
                isfinite = dist.mode.isfinite().reshape(
                    dist.batch_shape + (dist.event_shape.numel(),)
                )
                batch_isfinite = isfinite.all(axis=-1)
                self.assertTrue((batch_isfinite | ~isfinite.any(axis=-1)).all())

                # We sanitize undefined modes by sampling from the distribution.
                sanitized_mode = torch.where(
                    ~dist.mode.isnan(), dist.mode, dist.sample()
                )
                if isinstance(dist, discrete_distributions):
                    self._test_discrete_distribution_mode(
                        dist, sanitized_mode, batch_isfinite
                    )
                else:
                    self._test_continuous_distribution_mode(
                        dist, sanitized_mode, batch_isfinite
                    )

                self.assertFalse(dist.log_prob(sanitized_mode).isnan().any())


# TODO: Enable CUDA/XPU in `instantiate_device_type_tests` and remove the following class
@skipIfTorchDynamo("Not a TorchDynamo suitable test")
class TestDistributionsGPU(DistributionsTestCase):
    # [CLASS DOC] Tests that need an accelerator (CUDA or XPU) for Binomial/Poisson/Gamma/Beta
    # sampling. Kept separate from TestDistributions because those device types are excluded
    # from the device-generic instantiation (see the TODO above). Tensors are placed on the
    # module-level device_type.
    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "CUDA and XPU not found")
    def test_zero_excluded_binomial(self):
        # [FUNCTION DOC] Accelerator version of the zero-excluded Binomial check: 1e8 samples at
        # probs 0.9 and 0.1 stay within [0, 1] and 10000 samples at 0.5 contain more than 4000
        # of each value. Raises AssertionError otherwise. Large memory use; skipped without CUDA
        # or XPU.
        vals = Binomial(
            total_count=torch.tensor(1.0).to(device_type),
            probs=torch.tensor(0.9).to(device_type),
        ).sample(torch.Size((100000000,)))
        self.assertTrue((vals >= 0).all())
        vals = Binomial(
            total_count=torch.tensor(1.0).to(device_type),
            probs=torch.tensor(0.1).to(device_type),
        ).sample(torch.Size((100000000,)))
        self.assertTrue((vals < 2).all())
        vals = Binomial(
            total_count=torch.tensor(1.0).to(device_type),
            probs=torch.tensor(0.5).to(device_type),
        ).sample(torch.Size((10000,)))
        # vals should be roughly half zeroes, half ones
        zeros_count = (vals == 0.0).sum()
        ones_count = (vals == 1.0).sum()
        if zeros_count <= 4000:
            raise AssertionError(
                f"Expected (vals == 0.0).sum() > 4000, got {zeros_count}"
            )
        if ones_count <= 4000:
            raise AssertionError(
                f"Expected (vals == 1.0).sum() > 4000, got {ones_count}"
            )

    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "CUDA and XPU not found")
    def test_torch_binomial_dtype_errors(self):
        # [FUNCTION DOC] Accelerator version of the torch.binomial dtype check: integer count or
        # prob dtypes raise ValueError with the floating-point message.
        dtypes = [torch.int, torch.long, torch.short]
        device = device_type

        for count_dtype in dtypes:
            total_count = torch.tensor([10, 10], dtype=count_dtype, device=device)
            total_prob = torch.tensor([0.5, 0.5], dtype=torch.float, device=device)

            with self.assertRaisesRegex(
                ValueError,
                "binomial only supports floating-point dtypes for count.*",
            ):
                torch.binomial(total_count, total_prob)

        for prob_dtype in dtypes:
            total_count = torch.tensor([10, 10], dtype=torch.float, device=device)
            total_prob = torch.tensor([0.5, 0.5], dtype=prob_dtype, device=device)

            with self.assertRaisesRegex(
                ValueError,
                "binomial only supports floating-point dtypes for prob.*",
            ):
                torch.binomial(total_count, total_prob)

    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "CUDA and XPU not found")
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_poisson_gpu_sample(self):
        # [FUNCTION DOC] Chi-square test of Poisson samples drawn on the accelerator (seed 1)
        # for rates 0.12, 0.9 and 4. Skipped without CUDA/XPU or NumPy.
        set_rng_seed(1)
        for rate in [0.12, 0.9, 4.0]:
            self._check_sampler_discrete(
                Poisson(torch.tensor([rate]).to(device_type)),
                scipy.stats.poisson(rate),
                f"Poisson(lambda={rate}, {device_type})",
                failure_rate=1e-3,
            )

    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "CUDA and XPU not found")
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_gamma_gpu_shape(self):
        # [FUNCTION DOC] Accelerator version of the Gamma shape and log_prob check against
        # scipy gamma.
        alpha = torch.randn(2, 3).to(device_type).exp().requires_grad_()
        beta = torch.randn(2, 3).to(device_type).exp().requires_grad_()
        alpha_1d = torch.randn(1).to(device_type).exp().requires_grad_()
        beta_1d = torch.randn(1).to(device_type).exp().requires_grad_()
        self.assertEqual(Gamma(alpha, beta).sample().size(), (2, 3))
        self.assertEqual(Gamma(alpha, beta).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(Gamma(alpha_1d, beta_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Gamma(alpha_1d, beta_1d).sample().size(), (1,))
        self.assertEqual(Gamma(0.5, 0.5).sample().size(), ())
        self.assertEqual(Gamma(0.5, 0.5).sample((1,)).size(), (1,))

        def ref_log_prob(idx, x, log_prob):
            # [FUNCTION DOC] HELPER: Asserts log_prob equals scipy gamma logpdf for the
            # batch element.
            a = alpha.view(-1)[idx].detach().cpu()
            b = beta.view(-1)[idx].detach().cpu()
            expected = scipy.stats.gamma.logpdf(x.cpu(), a, scale=1 / b)
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(Gamma(alpha, beta), ref_log_prob)

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_gamma_gpu_sample(self):
        # [FUNCTION DOC] Statistical test (seed 0) of Gamma samples drawn on CUDA with 1-element
        # tensor parameters against scipy gamma (failure_rate 1e-4). Skipped without CUDA (XPU
        # is not covered).
        set_rng_seed(0)
        for alpha, beta in product([0.1, 1.0, 5.0], [0.1, 1.0, 10.0]):
            a, b = (
                torch.tensor([alpha]).to(device_type),
                torch.tensor([beta]).to(device_type),
            )
            self._check_sampler_sampler(
                Gamma(a, b),
                scipy.stats.gamma(alpha, scale=1.0 / beta),
                f"Gamma(alpha={alpha}, beta={beta})",
                failure_rate=1e-4,
            )

    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "CUDA and XPU not found")
    def test_beta_underflow_gpu(self):
        # [FUNCTION DOC] Accelerator version of the Beta underflow check using float64
        # concentration 0.01: no exact 0 or 1 samples and about half the mass below 0.1 and
        # above 0.9 within 0.12.
        set_rng_seed(1)
        num_samples = 50000
        conc = torch.tensor(1e-2, dtype=torch.float64).to(device_type)
        beta_samples = Beta(conc, conc).sample([num_samples])
        self.assertEqual((beta_samples == 0).sum(), 0)
        self.assertEqual((beta_samples == 1).sum(), 0)
        # assert support is concentrated around 0 and 1
        frac_zeros = float((beta_samples < 0.1).sum()) / num_samples
        frac_ones = float((beta_samples > 0.9).sum()) / num_samples
        # TODO: increase precision once imbalance on GPU is fixed.
        self.assertEqual(frac_zeros, 0.5, atol=0.12, rtol=0)
        self.assertEqual(frac_ones, 0.5, atol=0.12, rtol=0)


# These tests are only needed for a few distributions that implement custom
# reparameterized gradients. Most .rsample() implementations simply rely on
# the reparameterization trick and do not need to be tested for accuracy.
@skipIfTorchDynamo("Not a TorchDynamo suitable test")
class TestRsample(DistributionsTestCase):
    # [CLASS DOC] Tests the accuracy of custom reparameterized gradients (rsample) for Gamma,
    # Chi2, Dirichlet and Beta by comparing autograd gradients to finite differences of the
    # scipy cdf along constant-cdf curves, and a Dirichlet tangent-field consistency check.
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_gamma(self):
        # [FUNCTION DOC] Checks the Gamma rsample gradient wrt concentration for alphas 0.01 to
        # 10000: 100 samples per alpha, gradient compared with minus d cdf/d alpha over pdf
        # (finite differences with scipy) with maximum relative error below 5e-4. Skipped
        # without NumPy.
        num_samples = 100
        for alpha in [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]:
            alphas = torch.tensor(
                [alpha] * num_samples, dtype=torch.float, requires_grad=True
            )
            betas = alphas.new_ones(num_samples)
            x = Gamma(alphas, betas).rsample()
            x.sum().backward()
            x, ind = x.sort()
            x = x.detach().numpy()
            actual_grad = alphas.grad[ind].numpy()
            # Compare with expected gradient dx/dalpha along constant cdf(x,alpha).
            cdf = scipy.stats.gamma.cdf
            pdf = scipy.stats.gamma.pdf
            eps = 0.01 * alpha / (1.0 + alpha**0.5)
            cdf_alpha = (cdf(x, alpha + eps) - cdf(x, alpha - eps)) / (2 * eps)
            cdf_x = pdf(x, alpha)
            expected_grad = -cdf_alpha / cdf_x
            rel_error = np.abs(actual_grad - expected_grad) / (expected_grad + 1e-30)
            self.assertLess(
                np.max(rel_error),
                0.0005,
                "\n".join(
                    [
                        f"Bad gradient dx/alpha for x ~ Gamma({alpha}, 1)",
                        f"x {x}",
                        f"expected {expected_grad}",
                        f"actual {actual_grad}",
                        f"rel error {rel_error}",
                        f"max error {rel_error.max()}",
                        f"at alpha={alpha}, x={x[rel_error.argmax()]}",
                    ]
                ),
            )

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_chi2(self):
        # [FUNCTION DOC] Same as test_gamma for the Chi2 gradient wrt df (relative error
        # below 1e-3).
        num_samples = 100
        for df in [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]:
            dfs = torch.tensor(
                [df] * num_samples, dtype=torch.float, requires_grad=True
            )
            x = Chi2(dfs).rsample()
            x.sum().backward()
            x, ind = x.sort()
            x = x.detach().numpy()
            actual_grad = dfs.grad[ind].numpy()
            # Compare with expected gradient dx/ddf along constant cdf(x,df).
            cdf = scipy.stats.chi2.cdf
            pdf = scipy.stats.chi2.pdf
            eps = 0.01 * df / (1.0 + df**0.5)
            cdf_df = (cdf(x, df + eps) - cdf(x, df - eps)) / (2 * eps)
            cdf_x = pdf(x, df)
            expected_grad = -cdf_df / cdf_x
            rel_error = np.abs(actual_grad - expected_grad) / (expected_grad + 1e-30)
            self.assertLess(
                np.max(rel_error),
                0.001,
                "\n".join(
                    [
                        f"Bad gradient dx/ddf for x ~ Chi2({df})",
                        f"x {x}",
                        f"expected {expected_grad}",
                        f"actual {actual_grad}",
                        f"rel error {rel_error}",
                        f"max error {rel_error.max()}",
                    ]
                ),
            )

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_dirichlet_on_diagonal(self):
        # [FUNCTION DOC] Checks the gradient of the first component of a Dirichlet rsample with
        # respect to the first concentration against the Beta(alpha0, alpha1 plus alpha2) cdf
        # finite difference, over a 3-value grid, with relative error below 1e-3.
        num_samples = 20
        grid = [1e-1, 1e0, 1e1]
        for a0, a1, a2 in product(grid, grid, grid):
            alphas = torch.tensor(
                [[a0, a1, a2]] * num_samples, dtype=torch.float, requires_grad=True
            )
            x = Dirichlet(alphas).rsample()[:, 0]
            x.sum().backward()
            x, ind = x.sort()
            x = x.detach().numpy()
            actual_grad = alphas.grad[ind].numpy()[:, 0]
            # Compare with expected gradient dx/dalpha0 along constant cdf(x,alpha).
            # This reduces to a distribution Beta(alpha[0], alpha[1] + alpha[2]).
            cdf = scipy.stats.beta.cdf
            pdf = scipy.stats.beta.pdf
            alpha, beta = a0, a1 + a2
            eps = 0.01 * alpha / (1.0 + np.sqrt(alpha))
            cdf_alpha = (cdf(x, alpha + eps, beta) - cdf(x, alpha - eps, beta)) / (
                2 * eps
            )
            cdf_x = pdf(x, alpha, beta)
            expected_grad = -cdf_alpha / cdf_x
            rel_error = np.abs(actual_grad - expected_grad) / (expected_grad + 1e-30)
            self.assertLess(
                np.max(rel_error),
                0.001,
                "\n".join(
                    [
                        f"Bad gradient dx[0]/dalpha[0] for Dirichlet([{a0}, {a1}, {a2}])",
                        f"x {x}",
                        f"expected {expected_grad}",
                        f"actual {actual_grad}",
                        f"rel error {rel_error}",
                        f"max error {rel_error.max()}",
                        f"at x={x[rel_error.argmax()]}",
                    ]
                ),
            )

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_beta_wrt_alpha(self):
        # [FUNCTION DOC] Checks the Beta rsample gradient with respect to concentration1 against
        # finite differences of the scipy beta cdf over a grid (relative error below 5e-3).
        num_samples = 20
        grid = [1e-2, 1e-1, 1e0, 1e1, 1e2]
        for con1, con0 in product(grid, grid):
            con1s = torch.tensor(
                [con1] * num_samples, dtype=torch.float, requires_grad=True
            )
            con0s = con1s.new_tensor([con0] * num_samples)
            x = Beta(con1s, con0s).rsample()
            x.sum().backward()
            x, ind = x.sort()
            x = x.detach().numpy()
            actual_grad = con1s.grad[ind].numpy()
            # Compare with expected gradient dx/dcon1 along constant cdf(x,con1,con0).
            cdf = scipy.stats.beta.cdf
            pdf = scipy.stats.beta.pdf
            eps = 0.01 * con1 / (1.0 + np.sqrt(con1))
            cdf_alpha = (cdf(x, con1 + eps, con0) - cdf(x, con1 - eps, con0)) / (
                2 * eps
            )
            cdf_x = pdf(x, con1, con0)
            expected_grad = -cdf_alpha / cdf_x
            rel_error = np.abs(actual_grad - expected_grad) / (expected_grad + 1e-30)
            self.assertLess(
                np.max(rel_error),
                0.005,
                "\n".join(
                    [
                        f"Bad gradient dx/dcon1 for x ~ Beta({con1}, {con0})",
                        f"x {x}",
                        f"expected {expected_grad}",
                        f"actual {actual_grad}",
                        f"rel error {rel_error}",
                        f"max error {rel_error.max()}",
                        f"at x = {x[rel_error.argmax()]}",
                    ]
                ),
            )

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_beta_wrt_beta(self):
        # [FUNCTION DOC] Same as test_beta_wrt_alpha but for the gradient with respect
        # to concentration0.
        num_samples = 20
        grid = [1e-2, 1e-1, 1e0, 1e1, 1e2]
        for con1, con0 in product(grid, grid):
            con0s = torch.tensor(
                [con0] * num_samples, dtype=torch.float, requires_grad=True
            )
            con1s = con0s.new_tensor([con1] * num_samples)
            x = Beta(con1s, con0s).rsample()
            x.sum().backward()
            x, ind = x.sort()
            x = x.detach().numpy()
            actual_grad = con0s.grad[ind].numpy()
            # Compare with expected gradient dx/dcon0 along constant cdf(x,con1,con0).
            cdf = scipy.stats.beta.cdf
            pdf = scipy.stats.beta.pdf
            eps = 0.01 * con0 / (1.0 + np.sqrt(con0))
            cdf_beta = (cdf(x, con1, con0 + eps) - cdf(x, con1, con0 - eps)) / (2 * eps)
            cdf_x = pdf(x, con1, con0)
            expected_grad = -cdf_beta / cdf_x
            rel_error = np.abs(actual_grad - expected_grad) / (expected_grad + 1e-30)
            self.assertLess(
                np.max(rel_error),
                0.005,
                "\n".join(
                    [
                        f"Bad gradient dx/dcon0 for x ~ Beta({con1}, {con0})",
                        f"x {x}",
                        f"expected {expected_grad}",
                        f"actual {actual_grad}",
                        f"rel error {rel_error}",
                        f"max error {rel_error.max()}",
                        f"at x = {x[rel_error.argmax()]!r}",
                    ]
                ),
            )

    def test_dirichlet_multivariate(self):
        # [FUNCTION DOC] Checks the Dirichlet rsample gradient with a loss on the third
        # component around the critical alpha (0.25 times sqrt(5) minus one) against a
        # hand-derived expected gradient within 0.002, using 100000 samples per shift.
        alpha_crit = 0.25 * (5.0**0.5 - 1.0)
        num_samples = 100000
        for shift in [-0.1, -0.05, -0.01, 0.0, 0.01, 0.05, 0.10]:
            alpha = alpha_crit + shift
            alpha = torch.tensor([alpha], dtype=torch.float, requires_grad=True)
            alpha_vec = torch.cat([alpha, alpha, alpha.new([1])])
            z = Dirichlet(alpha_vec.expand(num_samples, 3)).rsample()
            mean_z3 = 1.0 / (2.0 * alpha + 1.0)
            loss = torch.pow(z[:, 2] - mean_z3, 2.0).mean()
            actual_grad = grad(loss, [alpha])[0]
            # Compute expected gradient by hand.
            num = 1.0 - 2.0 * alpha - 4.0 * alpha**2
            den = (1.0 + alpha) ** 2 * (1.0 + 2.0 * alpha) ** 3
            expected_grad = num / den
            self.assertEqual(
                actual_grad,
                expected_grad,
                atol=0.002,
                rtol=0,
                msg="\n".join(
                    [
                        "alpha = alpha_c + %.2g" % shift,  # noqa: UP031
                        "expected_grad: %.5g" % expected_grad,  # noqa: UP031
                        "actual_grad: %.5g" % actual_grad,  # noqa: UP031
                        "error = %.2g"  # noqa: UP031
                        % torch.abs(expected_grad - actual_grad).max(),
                    ]
                ),
            )

    @set_default_dtype(torch.double)
    def test_dirichlet_tangent_field(self):
        # [FUNCTION DOC] Checks that the Dirichlet reparameterization gradient field satisfies
        # the continuity equation (log_prob derivative terms plus divergence of the tangent
        # field, estimated with finite differences along the simplex) with error below 0.005,
        # over a grid of alphas, in double precision. Also verifies the compute_v helper equals
        # autograd gradients.
        num_samples = 20
        alpha_grid = [0.5, 1.0, 2.0]

        # v = dx/dalpha[0] is the reparameterized gradient aka tangent field.
        def compute_v(x, alpha):
            # [FUNCTION DOC] HELPER: Builds the tangent field v (derivative of each coordinate
            # of x wrt the first alpha) by calling _Dirichlet_backward with each unit vector of
            # the three coordinates.
            return torch.stack(
                [
                    _Dirichlet_backward(x, alpha, torch.eye(3, 3)[i].expand_as(x))[:, 0]
                    for i in range(3)
                ],
                dim=-1,
            )

        for a1, a2, a3 in product(alpha_grid, alpha_grid, alpha_grid):
            alpha = torch.tensor([a1, a2, a3], requires_grad=True).expand(
                num_samples, 3
            )
            x = Dirichlet(alpha).rsample()
            dlogp_da = grad(
                [Dirichlet(alpha).log_prob(x.detach()).sum()],
                [alpha],
                retain_graph=True,
            )[0][:, 0]
            dlogp_dx = grad(
                [Dirichlet(alpha.detach()).log_prob(x).sum()], [x], retain_graph=True
            )[0]
            v = torch.stack(
                [
                    grad([x[:, i].sum()], [alpha], retain_graph=True)[0][:, 0]
                    for i in range(3)
                ],
                dim=-1,
            )
            # Compute ramaining properties by finite difference.
            self.assertEqual(compute_v(x, alpha), v, msg="Bug in compute_v() helper")
            # dx is an arbitrary orthonormal basis tangent to the simplex.
            dx = torch.tensor([[2.0, -1.0, -1.0], [0.0, 1.0, -1.0]])
            dx /= dx.norm(2, -1, True)
            eps = 1e-2 * x.min(-1, True)[0]  # avoid boundary
            dv0 = (
                compute_v(x + eps * dx[0], alpha) - compute_v(x - eps * dx[0], alpha)
            ) / (2 * eps)
            dv1 = (
                compute_v(x + eps * dx[1], alpha) - compute_v(x - eps * dx[1], alpha)
            ) / (2 * eps)
            div_v = (dv0 * dx[0] + dv1 * dx[1]).sum(-1)
            # This is a modification of the standard continuity equation, using the product rule to allow
            # expression in terms of log_prob rather than the less numerically stable log_prob.exp().
            error = dlogp_da + (dlogp_dx * v).sum(-1) + div_v
            self.assertLess(
                torch.abs(error).max(),
                0.005,
                "\n".join(
                    [
                        f"Dirichlet([{a1}, {a2}, {a3}]) gradient violates continuity equation:",
                        f"error = {error}",
                    ]
                ),
            )


class TestDistributionShapes(DistributionsTestCase):
    # [CLASS DOC] Tests batch_shape, event_shape, sample, log_prob and entropy shapes for many
    # distributions with scalar and tensor parameters. setUp defines scalar and tensor sample
    # values (shapes (3, 2) and (3, 2, 3)) used to verify that log_prob accepts broadcastable
    # values and raises ValueError for incompatible shapes.
    def setUp(self):
        # [FUNCTION DOC] Runs the base setUp and stores scalar_sample (the integer 1),
        # tensor_sample_1 (ones of shape 3 by 2) and tensor_sample_2 (ones of shape 3 by 2 by 3)
        # for the shape tests.
        super().setUp()
        self.scalar_sample = 1
        self.tensor_sample_1 = torch.ones(3, 2)
        self.tensor_sample_2 = torch.ones(3, 2, 3)

    def test_entropy_shape(self):
        # [FUNCTION DOC] For each example builds the distribution with validation off and checks
        # entropy shape equals the batch shape (empty if scalar); distributions without entropy
        # are skipped.
        for Dist, params in _get_examples():
            for i, param in enumerate(params):
                dist = Dist(validate_args=False, **param)
                try:
                    actual_shape = dist.entropy().size()
                    expected_shape = (
                        dist.batch_shape if dist.batch_shape else torch.Size()
                    )
                    message = f"{Dist.__name__} example {i + 1}/{len(params)}, shape mismatch. expected {expected_shape}, actual {actual_shape}"
                    self.assertEqual(actual_shape, expected_shape, msg=message)
                except NotImplementedError:
                    continue

    def test_bernoulli_shape_scalar_params(self):
        # [FUNCTION DOC] Checks Bernoulli with scalar probs: empty batch/event shape, sample
        # sizes, ValueError for a scalar log_prob argument, and log_prob sizes for the two
        # tensor samples.
        bernoulli = Bernoulli(0.3)
        self.assertEqual(bernoulli._batch_shape, torch.Size())
        self.assertEqual(bernoulli._event_shape, torch.Size())
        self.assertEqual(bernoulli.sample().size(), torch.Size())
        self.assertEqual(bernoulli.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, bernoulli.log_prob, self.scalar_sample)
        self.assertEqual(
            bernoulli.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertEqual(
            bernoulli.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    def test_bernoulli_shape_tensor_params(self):
        # [FUNCTION DOC] Checks Bernoulli with a (3, 2) probs tensor: shapes, sample shape,
        # log_prob sizes, ValueError for the incompatible (3, 2, 3) value and broadcasting for a
        # (3, 1, 1) value.
        bernoulli = Bernoulli(torch.tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]]))
        self.assertEqual(bernoulli._batch_shape, torch.Size((3, 2)))
        self.assertEqual(bernoulli._event_shape, torch.Size(()))
        self.assertEqual(bernoulli.sample().size(), torch.Size((3, 2)))
        self.assertEqual(bernoulli.sample((3, 2)).size(), torch.Size((3, 2, 3, 2)))
        self.assertEqual(
            bernoulli.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertRaises(ValueError, bernoulli.log_prob, self.tensor_sample_2)
        self.assertEqual(
            bernoulli.log_prob(torch.ones(3, 1, 1)).size(), torch.Size((3, 3, 2))
        )

    def test_geometric_shape_scalar_params(self):
        # [FUNCTION DOC] Same as test_bernoulli_shape_scalar_params for Geometric.
        geometric = Geometric(0.3)
        self.assertEqual(geometric._batch_shape, torch.Size())
        self.assertEqual(geometric._event_shape, torch.Size())
        self.assertEqual(geometric.sample().size(), torch.Size())
        self.assertEqual(geometric.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, geometric.log_prob, self.scalar_sample)
        self.assertEqual(
            geometric.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertEqual(
            geometric.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    def test_geometric_shape_tensor_params(self):
        # [FUNCTION DOC] Same as test_bernoulli_shape_tensor_params for Geometric.
        geometric = Geometric(torch.tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]]))
        self.assertEqual(geometric._batch_shape, torch.Size((3, 2)))
        self.assertEqual(geometric._event_shape, torch.Size(()))
        self.assertEqual(geometric.sample().size(), torch.Size((3, 2)))
        self.assertEqual(geometric.sample((3, 2)).size(), torch.Size((3, 2, 3, 2)))
        self.assertEqual(
            geometric.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertRaises(ValueError, geometric.log_prob, self.tensor_sample_2)
        self.assertEqual(
            geometric.log_prob(torch.ones(3, 1, 1)).size(), torch.Size((3, 3, 2))
        )

    def test_beta_shape_scalar_params(self):
        # [FUNCTION DOC] Same as test_bernoulli_shape_scalar_params for Beta with
        # scalar concentrations.
        dist = Beta(0.1, 0.1)
        self.assertEqual(dist._batch_shape, torch.Size())
        self.assertEqual(dist._event_shape, torch.Size())
        self.assertEqual(dist.sample().size(), torch.Size())
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, dist.log_prob, self.scalar_sample)
        self.assertEqual(dist.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(
            dist.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    def test_beta_shape_tensor_params(self):
        # [FUNCTION DOC] Same as test_bernoulli_shape_tensor_params for Beta with (3,
        # 2) concentrations.
        dist = Beta(
            torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        )
        self.assertEqual(dist._batch_shape, torch.Size((3, 2)))
        self.assertEqual(dist._event_shape, torch.Size(()))
        self.assertEqual(dist.sample().size(), torch.Size((3, 2)))
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2, 3, 2)))
        self.assertEqual(dist.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_2)
        self.assertEqual(
            dist.log_prob(torch.ones(3, 1, 1)).size(), torch.Size((3, 3, 2))
        )

    def test_binomial_shape(self):
        # [FUNCTION DOC] Checks Binomial with scalar total_count and probs of shape (2,):
        # shapes, sample sizes and ValueError for an incompatible value.
        dist = Binomial(10, torch.tensor([0.6, 0.3]))
        self.assertEqual(dist._batch_shape, torch.Size((2,)))
        self.assertEqual(dist._event_shape, torch.Size(()))
        self.assertEqual(dist.sample().size(), torch.Size((2,)))
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        self.assertEqual(dist.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_2)

    def test_binomial_shape_vectorized_n(self):
        # [FUNCTION DOC] Checks Binomial with tensor total_count (2, 3) and probs (3,): batch
        # shape, sample sizes, log_prob sizes and ValueError for an incompatible value.
        dist = Binomial(
            torch.tensor([[10, 3, 1], [4, 8, 4]]), torch.tensor([0.6, 0.3, 0.1])
        )
        self.assertEqual(dist._batch_shape, torch.Size((2, 3)))
        self.assertEqual(dist._event_shape, torch.Size(()))
        self.assertEqual(dist.sample().size(), torch.Size((2, 3)))
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2, 2, 3)))
        self.assertEqual(
            dist.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_1)

    def test_multinomial_shape(self):
        # [FUNCTION DOC] Checks Multinomial batch shape (3,) and event shape (2,): sample sizes,
        # log_prob sizes, ValueError for an incompatible value and broadcasting.
        dist = Multinomial(10, torch.tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]]))
        self.assertEqual(dist._batch_shape, torch.Size((3,)))
        self.assertEqual(dist._event_shape, torch.Size((2,)))
        self.assertEqual(dist.sample().size(), torch.Size((3, 2)))
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2, 3, 2)))
        self.assertEqual(dist.log_prob(self.tensor_sample_1).size(), torch.Size((3,)))
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_2)
        self.assertEqual(dist.log_prob(torch.ones(3, 1, 2)).size(), torch.Size((3, 3)))

    def test_categorical_shape(self):
        # [FUNCTION DOC] Checks Categorical shapes for unbatched, 1-batch and 2-d batched probs,
        # including sample sizes, ValueError for scalar or incompatible log_prob values and
        # broadcast results.
        # unbatched
        dist = Categorical(torch.tensor([0.6, 0.3, 0.1]))
        self.assertEqual(dist._batch_shape, torch.Size(()))
        self.assertEqual(dist._event_shape, torch.Size(()))
        self.assertEqual(dist.sample().size(), torch.Size())
        self.assertEqual(
            dist.sample((3, 2)).size(),
            torch.Size(
                (
                    3,
                    2,
                )
            ),
        )
        self.assertEqual(dist.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(
            dist.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )
        self.assertEqual(dist.log_prob(torch.ones(3, 1)).size(), torch.Size((3, 1)))
        # batched
        dist = Categorical(torch.tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]]))
        self.assertEqual(dist._batch_shape, torch.Size((3,)))
        self.assertEqual(dist._event_shape, torch.Size(()))
        self.assertEqual(dist.sample().size(), torch.Size((3,)))
        self.assertEqual(
            dist.sample((3, 2)).size(),
            torch.Size(
                (
                    3,
                    2,
                    3,
                )
            ),
        )
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_1)
        self.assertEqual(
            dist.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )
        self.assertEqual(dist.log_prob(torch.ones(3, 1)).size(), torch.Size((3, 3)))

    def test_one_hot_categorical_shape(self):
        # [FUNCTION DOC] Checks OneHotCategorical shapes for unbatched (event shape (3,))
        # and batched probs: sample sizes, ValueError for non-one-hot or mismatched values,
        # and log_prob sizes for one-hot samples, enumerate_support output and
        # identity-matrix values.
        # unbatched
        dist = OneHotCategorical(torch.tensor([0.6, 0.3, 0.1]))
        self.assertEqual(dist._batch_shape, torch.Size(()))
        self.assertEqual(dist._event_shape, torch.Size((3,)))
        self.assertEqual(dist.sample().size(), torch.Size((3,)))
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2, 3)))
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_1)
        sample = torch.tensor([0.0, 1.0, 0.0]).expand(3, 2, 3)
        self.assertEqual(
            dist.log_prob(sample).size(),
            torch.Size(
                (
                    3,
                    2,
                )
            ),
        )
        self.assertEqual(
            dist.log_prob(dist.enumerate_support()).size(), torch.Size((3,))
        )
        sample = torch.eye(3)
        self.assertEqual(dist.log_prob(sample).size(), torch.Size((3,)))
        # batched
        dist = OneHotCategorical(torch.tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]]))
        self.assertEqual(dist._batch_shape, torch.Size((3,)))
        self.assertEqual(dist._event_shape, torch.Size((2,)))
        self.assertEqual(dist.sample().size(), torch.Size((3, 2)))
        self.assertEqual(dist.sample((3, 2)).size(), torch.Size((3, 2, 3, 2)))
        sample = torch.tensor([0.0, 1.0])
        self.assertEqual(dist.log_prob(sample).size(), torch.Size((3,)))
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_2)
        self.assertEqual(
            dist.log_prob(dist.enumerate_support()).size(), torch.Size((2, 3))
        )
        sample = torch.tensor([0.0, 1.0]).expand(3, 1, 2)
        self.assertEqual(dist.log_prob(sample).size(), torch.Size((3, 3)))

    def test_cauchy_shape_scalar_params(self):
        # [FUNCTION DOC] Checks Cauchy with scalar parameters: empty batch/event shape,
        # sample sizes, ValueError for a scalar log_prob argument and log_prob sizes for the
        # tensor samples.
        cauchy = Cauchy(0, 1)
        self.assertEqual(cauchy._batch_shape, torch.Size())
        self.assertEqual(cauchy._event_shape, torch.Size())
        self.assertEqual(cauchy.sample().size(), torch.Size())
        self.assertEqual(cauchy.sample(torch.Size((3, 2))).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, cauchy.log_prob, self.scalar_sample)
        self.assertEqual(
            cauchy.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertEqual(
            cauchy.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    def test_cauchy_shape_tensor_params(self):
        # [FUNCTION DOC] Checks Cauchy with parameters of shape (2,): shapes, sample sizes,
        # ValueError for the incompatible (3, 2, 3) value and broadcasting of a (2, 1) value to
        # (2, 2).
        cauchy = Cauchy(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
        self.assertEqual(cauchy._batch_shape, torch.Size((2,)))
        self.assertEqual(cauchy._event_shape, torch.Size(()))
        self.assertEqual(cauchy.sample().size(), torch.Size((2,)))
        self.assertEqual(
            cauchy.sample(torch.Size((3, 2))).size(), torch.Size((3, 2, 2))
        )
        self.assertEqual(
            cauchy.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertRaises(ValueError, cauchy.log_prob, self.tensor_sample_2)
        self.assertEqual(cauchy.log_prob(torch.ones(2, 1)).size(), torch.Size((2, 2)))

    def test_halfcauchy_shape_scalar_params(self):
        # [FUNCTION DOC] Same as test_cauchy_shape_scalar_params for HalfCauchy.
        halfcauchy = HalfCauchy(1)
        self.assertEqual(halfcauchy._batch_shape, torch.Size())
        self.assertEqual(halfcauchy._event_shape, torch.Size())
        self.assertEqual(halfcauchy.sample().size(), torch.Size())
        self.assertEqual(
            halfcauchy.sample(torch.Size((3, 2))).size(), torch.Size((3, 2))
        )
        self.assertRaises(ValueError, halfcauchy.log_prob, self.scalar_sample)
        self.assertEqual(
            halfcauchy.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertEqual(
            halfcauchy.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    def test_halfcauchy_shape_tensor_params(self):
        # [FUNCTION DOC] Same as test_cauchy_shape_tensor_params for HalfCauchy.
        halfcauchy = HalfCauchy(torch.tensor([1.0, 1.0]))
        self.assertEqual(halfcauchy._batch_shape, torch.Size((2,)))
        self.assertEqual(halfcauchy._event_shape, torch.Size(()))
        self.assertEqual(halfcauchy.sample().size(), torch.Size((2,)))
        self.assertEqual(
            halfcauchy.sample(torch.Size((3, 2))).size(), torch.Size((3, 2, 2))
        )
        self.assertEqual(
            halfcauchy.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertRaises(ValueError, halfcauchy.log_prob, self.tensor_sample_2)
        self.assertEqual(
            halfcauchy.log_prob(torch.ones(2, 1)).size(), torch.Size((2, 2))
        )

    def test_dirichlet_shape(self):
        # [FUNCTION DOC] Checks Dirichlet with a (3, 2) concentration: batch shape (3,), event
        # shape (2,), sample sizes, log_prob on normalized simplex values, ValueError for an
        # incompatible value and broadcasting of a (3, 1, 2) simplex value to (3, 3).
        dist = Dirichlet(torch.tensor([[0.6, 0.3], [1.6, 1.3], [2.6, 2.3]]))
        self.assertEqual(dist._batch_shape, torch.Size((3,)))
        self.assertEqual(dist._event_shape, torch.Size((2,)))
        self.assertEqual(dist.sample().size(), torch.Size((3, 2)))
        self.assertEqual(dist.sample((5, 4)).size(), torch.Size((5, 4, 3, 2)))
        simplex_sample = self.tensor_sample_1 / self.tensor_sample_1.sum(
            -1, keepdim=True
        )
        self.assertEqual(dist.log_prob(simplex_sample).size(), torch.Size((3,)))
        self.assertRaises(ValueError, dist.log_prob, self.tensor_sample_2)
        simplex_sample = torch.ones(3, 1, 2)
        simplex_sample = simplex_sample / simplex_sample.sum(-1).unsqueeze(-1)
        self.assertEqual(dist.log_prob(simplex_sample).size(), torch.Size((3, 3)))

    def test_mixture_same_family_shape(self):
        # [FUNCTION DOC] Checks MixtureSameFamily of a Categorical and Normal: scalar batch and
        # event shape, sample sizes and log_prob sizes for the two tensor samples.
        dist = MixtureSameFamily(
            Categorical(torch.rand(5)), Normal(torch.randn(5), torch.rand(5))
        )
        self.assertEqual(dist._batch_shape, torch.Size())
        self.assertEqual(dist._event_shape, torch.Size())
        self.assertEqual(dist.sample().size(), torch.Size())
        self.assertEqual(dist.sample((5, 4)).size(), torch.Size((5, 4)))
        self.assertEqual(dist.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(
            dist.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    def test_gamma_shape_scalar_params(self):
        # [FUNCTION DOC] Checks Gamma with scalar parameters: shapes, sample sizes and log_prob
        # sizes (a scalar value is accepted and gives an empty result).
        gamma = Gamma(1, 1)
        self.assertEqual(gamma._batch_shape, torch.Size())
        self.assertEqual(gamma._event_shape, torch.Size())
        self.assertEqual(gamma.sample().size(), torch.Size())
        self.assertEqual(gamma.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertEqual(gamma.log_prob(self.scalar_sample).size(), torch.Size())
        self.assertEqual(
            gamma.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertEqual(
            gamma.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    def test_gamma_shape_tensor_params(self):
        # [FUNCTION DOC] Checks Gamma with parameters of shape (2,): shapes, sample sizes,
        # log_prob sizes, ValueError for the incompatible value and broadcasting.
        gamma = Gamma(torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0]))
        self.assertEqual(gamma._batch_shape, torch.Size((2,)))
        self.assertEqual(gamma._event_shape, torch.Size(()))
        self.assertEqual(gamma.sample().size(), torch.Size((2,)))
        self.assertEqual(gamma.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        self.assertEqual(
            gamma.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertRaises(ValueError, gamma.log_prob, self.tensor_sample_2)
        self.assertEqual(gamma.log_prob(torch.ones(2, 1)).size(), torch.Size((2, 2)))

    def test_chi2_shape_scalar_params(self):
        # [FUNCTION DOC] Same as test_gamma_shape_scalar_params for Chi2.
        chi2 = Chi2(1)
        self.assertEqual(chi2._batch_shape, torch.Size())
        self.assertEqual(chi2._event_shape, torch.Size())
        self.assertEqual(chi2.sample().size(), torch.Size())
        self.assertEqual(chi2.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertEqual(chi2.log_prob(self.scalar_sample).size(), torch.Size())
        self.assertEqual(chi2.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(
            chi2.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    def test_chi2_shape_tensor_params(self):
        # [FUNCTION DOC] Same as test_gamma_shape_tensor_params for Chi2.
        chi2 = Chi2(torch.tensor([1.0, 1.0]))
        self.assertEqual(chi2._batch_shape, torch.Size((2,)))
        self.assertEqual(chi2._event_shape, torch.Size(()))
        self.assertEqual(chi2.sample().size(), torch.Size((2,)))
        self.assertEqual(chi2.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        self.assertEqual(chi2.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, chi2.log_prob, self.tensor_sample_2)
        self.assertEqual(chi2.log_prob(torch.ones(2, 1)).size(), torch.Size((2, 2)))

    def test_studentT_shape_scalar_params(self):
        # [FUNCTION DOC] Checks StudentT with a scalar df: shapes, sample sizes, ValueError for
        # a scalar log_prob argument and log_prob sizes for the tensor samples.
        st = StudentT(1)
        self.assertEqual(st._batch_shape, torch.Size())
        self.assertEqual(st._event_shape, torch.Size())
        self.assertEqual(st.sample().size(), torch.Size())
        self.assertEqual(st.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, st.log_prob, self.scalar_sample)
        self.assertEqual(st.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertEqual(
            st.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    def test_studentT_shape_tensor_params(self):
        # [FUNCTION DOC] Checks StudentT with df of shape (2,): shapes, sample sizes, log_prob
        # sizes, ValueError for the incompatible value and broadcasting.
        st = StudentT(torch.tensor([1.0, 1.0]))
        self.assertEqual(st._batch_shape, torch.Size((2,)))
        self.assertEqual(st._event_shape, torch.Size(()))
        self.assertEqual(st.sample().size(), torch.Size((2,)))
        self.assertEqual(st.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        self.assertEqual(st.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, st.log_prob, self.tensor_sample_2)
        self.assertEqual(st.log_prob(torch.ones(2, 1)).size(), torch.Size((2, 2)))

    def test_pareto_shape_scalar_params(self):
        # [FUNCTION DOC] Checks Pareto with scalar parameters: shapes, sample sizes and log_prob
        # sizes (values are shifted by one to lie inside the support).
        pareto = Pareto(1, 1)
        self.assertEqual(pareto._batch_shape, torch.Size())
        self.assertEqual(pareto._event_shape, torch.Size())
        self.assertEqual(pareto.sample().size(), torch.Size())
        self.assertEqual(pareto.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertEqual(
            pareto.log_prob(self.tensor_sample_1 + 1).size(), torch.Size((3, 2))
        )
        self.assertEqual(
            pareto.log_prob(self.tensor_sample_2 + 1).size(), torch.Size((3, 2, 3))
        )

    def test_gumbel_shape_scalar_params(self):
        # [FUNCTION DOC] Checks Gumbel with scalar parameters: shapes, sample sizes and
        # log_prob sizes.
        gumbel = Gumbel(1, 1)
        self.assertEqual(gumbel._batch_shape, torch.Size())
        self.assertEqual(gumbel._event_shape, torch.Size())
        self.assertEqual(gumbel.sample().size(), torch.Size())
        self.assertEqual(gumbel.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertEqual(
            gumbel.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertEqual(
            gumbel.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    def test_kumaraswamy_shape_scalar_params(self):
        # [FUNCTION DOC] Checks Kumaraswamy with scalar parameters: shapes, sample sizes and
        # log_prob sizes.
        kumaraswamy = Kumaraswamy(1, 1)
        self.assertEqual(kumaraswamy._batch_shape, torch.Size())
        self.assertEqual(kumaraswamy._event_shape, torch.Size())
        self.assertEqual(kumaraswamy.sample().size(), torch.Size())
        self.assertEqual(kumaraswamy.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertEqual(
            kumaraswamy.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertEqual(
            kumaraswamy.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    def test_vonmises_shape_tensor_params(self):
        # [FUNCTION DOC] Checks VonMises with parameters of shape (2,): shapes, sample sizes and
        # log_prob sizes including broadcasting of a (2, 1) value.
        von_mises = VonMises(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
        self.assertEqual(von_mises._batch_shape, torch.Size((2,)))
        self.assertEqual(von_mises._event_shape, torch.Size(()))
        self.assertEqual(von_mises.sample().size(), torch.Size((2,)))
        self.assertEqual(
            von_mises.sample(torch.Size((3, 2))).size(), torch.Size((3, 2, 2))
        )
        self.assertEqual(
            von_mises.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertEqual(
            von_mises.log_prob(torch.ones(2, 1)).size(), torch.Size((2, 2))
        )

    def test_vonmises_shape_scalar_params(self):
        # [FUNCTION DOC] Checks VonMises with scalar parameters: shapes, sample sizes and
        # log_prob sizes.
        von_mises = VonMises(0.0, 1.0)
        self.assertEqual(von_mises._batch_shape, torch.Size())
        self.assertEqual(von_mises._event_shape, torch.Size())
        self.assertEqual(von_mises.sample().size(), torch.Size())
        self.assertEqual(
            von_mises.sample(torch.Size((3, 2))).size(), torch.Size((3, 2))
        )
        self.assertEqual(
            von_mises.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertEqual(
            von_mises.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    def test_weibull_scale_scalar_params(self):
        # [FUNCTION DOC] Checks Weibull (scale and concentration scalars): shapes, sample sizes
        # and log_prob sizes.
        weibull = Weibull(1, 1)
        self.assertEqual(weibull._batch_shape, torch.Size())
        self.assertEqual(weibull._event_shape, torch.Size())
        self.assertEqual(weibull.sample().size(), torch.Size())
        self.assertEqual(weibull.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertEqual(
            weibull.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertEqual(
            weibull.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    def test_wishart_shape_scalar_params(self):
        # [FUNCTION DOC] Checks Wishart with scalar df and a 1 by 1 scale matrix: empty batch
        # shape, event shape (1, 1), sample sizes and ValueError for a scalar log_prob argument.
        wishart = Wishart(torch.tensor(1), torch.tensor([[1.0]]))
        self.assertEqual(wishart._batch_shape, torch.Size())
        self.assertEqual(wishart._event_shape, torch.Size((1, 1)))
        self.assertEqual(wishart.sample().size(), torch.Size((1, 1)))
        self.assertEqual(wishart.sample((3, 2)).size(), torch.Size((3, 2, 1, 1)))
        self.assertRaises(ValueError, wishart.log_prob, self.scalar_sample)

    def test_wishart_shape_tensor_params(self):
        # [FUNCTION DOC] Checks Wishart with df of shape (2,) and a batch of 1 by 1 matrices:
        # batch shape (2,), event shape (1, 1), sample sizes, ValueError for an incompatible
        # value and a (2, 1, 1) value giving a (2,) log_prob.
        wishart = Wishart(torch.tensor([1.0, 1.0]), torch.tensor([[[1.0]], [[1.0]]]))
        self.assertEqual(wishart._batch_shape, torch.Size((2,)))
        self.assertEqual(wishart._event_shape, torch.Size((1, 1)))
        self.assertEqual(wishart.sample().size(), torch.Size((2, 1, 1)))
        self.assertEqual(wishart.sample((3, 2)).size(), torch.Size((3, 2, 2, 1, 1)))
        self.assertRaises(ValueError, wishart.log_prob, self.tensor_sample_2)
        self.assertEqual(wishart.log_prob(torch.ones(2, 1, 1)).size(), torch.Size((2,)))

    def test_normal_shape_scalar_params(self):
        # [FUNCTION DOC] Checks Normal with scalar parameters: shapes, sample sizes, ValueError
        # for a scalar log_prob argument and log_prob sizes for the tensor samples.
        normal = Normal(0, 1)
        self.assertEqual(normal._batch_shape, torch.Size())
        self.assertEqual(normal._event_shape, torch.Size())
        self.assertEqual(normal.sample().size(), torch.Size())
        self.assertEqual(normal.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, normal.log_prob, self.scalar_sample)
        self.assertEqual(
            normal.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertEqual(
            normal.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    def test_normal_shape_tensor_params(self):
        # [FUNCTION DOC] Checks Normal with parameters of shape (2,): shapes, sample sizes,
        # log_prob sizes, ValueError for the incompatible value and broadcasting.
        normal = Normal(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
        self.assertEqual(normal._batch_shape, torch.Size((2,)))
        self.assertEqual(normal._event_shape, torch.Size(()))
        self.assertEqual(normal.sample().size(), torch.Size((2,)))
        self.assertEqual(normal.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        self.assertEqual(
            normal.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertRaises(ValueError, normal.log_prob, self.tensor_sample_2)
        self.assertEqual(normal.log_prob(torch.ones(2, 1)).size(), torch.Size((2, 2)))

    def test_uniform_shape_scalar_params(self):
        # [FUNCTION DOC] Checks Uniform with scalar bounds: shapes, sample sizes, ValueError for
        # a scalar log_prob argument and log_prob sizes for the tensor samples.
        uniform = Uniform(0, 1)
        self.assertEqual(uniform._batch_shape, torch.Size())
        self.assertEqual(uniform._event_shape, torch.Size())
        self.assertEqual(uniform.sample().size(), torch.Size())
        self.assertEqual(uniform.sample(torch.Size((3, 2))).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, uniform.log_prob, self.scalar_sample)
        self.assertEqual(
            uniform.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertEqual(
            uniform.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    def test_uniform_shape_tensor_params(self):
        # [FUNCTION DOC] Checks Uniform with bounds of shape (2,): shapes, sample sizes,
        # log_prob sizes, ValueError for the incompatible value and broadcasting.
        uniform = Uniform(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
        self.assertEqual(uniform._batch_shape, torch.Size((2,)))
        self.assertEqual(uniform._event_shape, torch.Size(()))
        self.assertEqual(uniform.sample().size(), torch.Size((2,)))
        self.assertEqual(
            uniform.sample(torch.Size((3, 2))).size(), torch.Size((3, 2, 2))
        )
        self.assertEqual(
            uniform.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertRaises(ValueError, uniform.log_prob, self.tensor_sample_2)
        self.assertEqual(uniform.log_prob(torch.ones(2, 1)).size(), torch.Size((2, 2)))

    def test_exponential_shape_scalar_param(self):
        # [FUNCTION DOC] Checks Exponential with a scalar rate: shapes, sample sizes, ValueError
        # for a scalar log_prob argument and log_prob sizes for the tensor samples.
        expon = Exponential(1.0)
        self.assertEqual(expon._batch_shape, torch.Size())
        self.assertEqual(expon._event_shape, torch.Size())
        self.assertEqual(expon.sample().size(), torch.Size())
        self.assertEqual(expon.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, expon.log_prob, self.scalar_sample)
        self.assertEqual(
            expon.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertEqual(
            expon.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    def test_exponential_shape_tensor_param(self):
        # [FUNCTION DOC] Checks Exponential with rate of shape (2,): shapes, sample sizes,
        # log_prob sizes, ValueError for the incompatible value and a (2, 2) value.
        expon = Exponential(torch.tensor([1.0, 1.0]))
        self.assertEqual(expon._batch_shape, torch.Size((2,)))
        self.assertEqual(expon._event_shape, torch.Size(()))
        self.assertEqual(expon.sample().size(), torch.Size((2,)))
        self.assertEqual(expon.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        self.assertEqual(
            expon.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertRaises(ValueError, expon.log_prob, self.tensor_sample_2)
        self.assertEqual(expon.log_prob(torch.ones(2, 2)).size(), torch.Size((2, 2)))

    def test_laplace_shape_scalar_params(self):
        # [FUNCTION DOC] Checks Laplace with scalar parameters: shapes, sample sizes, ValueError
        # for a scalar log_prob argument and log_prob sizes for the tensor samples.
        laplace = Laplace(0, 1)
        self.assertEqual(laplace._batch_shape, torch.Size())
        self.assertEqual(laplace._event_shape, torch.Size())
        self.assertEqual(laplace.sample().size(), torch.Size())
        self.assertEqual(laplace.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, laplace.log_prob, self.scalar_sample)
        self.assertEqual(
            laplace.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertEqual(
            laplace.log_prob(self.tensor_sample_2).size(), torch.Size((3, 2, 3))
        )

    def test_laplace_shape_tensor_params(self):
        # [FUNCTION DOC] Checks Laplace with parameters of shape (2,): shapes, sample sizes,
        # log_prob sizes, ValueError for the incompatible value and broadcasting.
        laplace = Laplace(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
        self.assertEqual(laplace._batch_shape, torch.Size((2,)))
        self.assertEqual(laplace._event_shape, torch.Size(()))
        self.assertEqual(laplace.sample().size(), torch.Size((2,)))
        self.assertEqual(laplace.sample((3, 2)).size(), torch.Size((3, 2, 2)))
        self.assertEqual(
            laplace.log_prob(self.tensor_sample_1).size(), torch.Size((3, 2))
        )
        self.assertRaises(ValueError, laplace.log_prob, self.tensor_sample_2)
        self.assertEqual(laplace.log_prob(torch.ones(2, 1)).size(), torch.Size((2, 2)))

    def test_continuous_bernoulli_shape_scalar_params(self):
        # [FUNCTION DOC] Checks ContinuousBernoulli with scalar probs: shapes, sample sizes,
        # ValueError for a scalar log_prob argument and log_prob sizes for the tensor samples.
        continuous_bernoulli = ContinuousBernoulli(0.3)
        self.assertEqual(continuous_bernoulli._batch_shape, torch.Size())
        self.assertEqual(continuous_bernoulli._event_shape, torch.Size())
        self.assertEqual(continuous_bernoulli.sample().size(), torch.Size())
        self.assertEqual(continuous_bernoulli.sample((3, 2)).size(), torch.Size((3, 2)))
        self.assertRaises(ValueError, continuous_bernoulli.log_prob, self.scalar_sample)
        self.assertEqual(
            continuous_bernoulli.log_prob(self.tensor_sample_1).size(),
            torch.Size((3, 2)),
        )
        self.assertEqual(
            continuous_bernoulli.log_prob(self.tensor_sample_2).size(),
            torch.Size((3, 2, 3)),
        )

    def test_continuous_bernoulli_shape_tensor_params(self):
        # [FUNCTION DOC] Checks ContinuousBernoulli with a (3, 2) probs tensor: shapes, sample
        # sizes, log_prob sizes, ValueError for the incompatible value and broadcasting of a (3,
        # 1, 1) value.
        continuous_bernoulli = ContinuousBernoulli(
            torch.tensor([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]])
        )
        self.assertEqual(continuous_bernoulli._batch_shape, torch.Size((3, 2)))
        self.assertEqual(continuous_bernoulli._event_shape, torch.Size(()))
        self.assertEqual(continuous_bernoulli.sample().size(), torch.Size((3, 2)))
        self.assertEqual(
            continuous_bernoulli.sample((3, 2)).size(), torch.Size((3, 2, 3, 2))
        )
        self.assertEqual(
            continuous_bernoulli.log_prob(self.tensor_sample_1).size(),
            torch.Size((3, 2)),
        )
        self.assertRaises(
            ValueError, continuous_bernoulli.log_prob, self.tensor_sample_2
        )
        self.assertEqual(
            continuous_bernoulli.log_prob(torch.ones(3, 1, 1)).size(),
            torch.Size((3, 3, 2)),
        )

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    def test_mixture_same_family_mean_shape(self):
        # [FUNCTION DOC] Checks that the mean of a MixtureSameFamily whose mixture has batch
        # shape (3, 1) and components have batch shape (3, 3) has two dimensions (the mixture
        # dimension is reduced over the component dimension).
        mix_distribution = Categorical(torch.ones([3, 1, 3]))
        component_distribution = Normal(torch.zeros([3, 3, 3]), torch.ones([3, 3, 3]))
        gmm = MixtureSameFamily(mix_distribution, component_distribution)
        self.assertEqual(len(gmm.mean.shape), 2)


@skipIfTorchDynamo("Not a TorchDynamo suitable test")
class TestKL(DistributionsTestCase):
    # [CLASS DOC] Tests torch.distributions.kl_divergence. setUp builds pairs of distributions
    # (via pairwise, so every parameter combination is covered) into finite_examples (KL must
    # match a Monte Carlo estimate or exponential-family formula) and infinite_examples (KL must
    # be infinity because of support mismatch), plus Monte Carlo settings: precision 0.1, up to
    # 1e7 samples in batches of 1e4.
    def setUp(self):
        # [FUNCTION DOC] Builds the KL test fixtures. Defines a local Binomial subclass with
        # total_count 30, creates pairwise distributions for many families (Bernoulli, Beta,
        # Categorical, Cauchy, Chi2, Dirichlet, Exponential, Gamma, Gumbel, Laplace, LogNormal,
        # Normal, Pareto, Poisson, Uniform variants, ContinuousBernoulli and others), and
        # stores self.precision, self.max_samples, self.samples_per_batch, self.finite_examples
        # (list of pair-of-pairs) and self.infinite_examples (list of single distribution
        # pairs). The Pareto versus Normal pair is intentionally excluded because of the Monte
        # Carlo variance.
        super().setUp()

        class Binomial30(Binomial):
            # [CLASS DOC] HELPER: Binomial subclass with fixed total_count 30 so pairwise can
            # build it from probs only.
            def __init__(self, probs):
                # [FUNCTION DOC] HELPER: Initializes Binomial with total_count 30 and the
                # given probs.
                super().__init__(30, probs)

        # These are pairs of distributions with 4 x 4 parameters as specified.
        # The first of the pair e.g. bernoulli[0] varies column-wise and the second
        # e.g. bernoulli[1] varies row-wise; that way we test all param pairs.
        bernoulli = pairwise(Bernoulli, [0.1, 0.2, 0.6, 0.9])
        binomial30 = pairwise(Binomial30, [0.1, 0.2, 0.6, 0.9])
        binomial_vectorized_count = (
            Binomial(torch.tensor([3, 4]), torch.tensor([0.4, 0.6])),
            Binomial(torch.tensor([3, 4]), torch.tensor([0.5, 0.8])),
        )
        beta = pairwise(Beta, [1.0, 2.5, 1.0, 2.5], [1.5, 1.5, 3.5, 3.5])
        categorical = pairwise(
            Categorical,
            [[0.4, 0.3, 0.3], [0.2, 0.7, 0.1], [0.33, 0.33, 0.34], [0.2, 0.2, 0.6]],
        )
        cauchy = pairwise(Cauchy, [-2.0, 2.0, -3.0, 3.0], [1.0, 2.0, 1.0, 2.0])
        chi2 = pairwise(Chi2, [1.0, 2.0, 2.5, 5.0])
        dirichlet = pairwise(
            Dirichlet,
            [[0.1, 0.2, 0.7], [0.5, 0.4, 0.1], [0.33, 0.33, 0.34], [0.2, 0.2, 0.4]],
        )
        exponential = pairwise(Exponential, [1.0, 2.5, 5.0, 10.0])
        gamma = pairwise(Gamma, [1.0, 2.5, 1.0, 2.5], [1.5, 1.5, 3.5, 3.5])
        gumbel = pairwise(Gumbel, [-2.0, 4.0, -3.0, 6.0], [1.0, 2.5, 1.0, 2.5])
        halfnormal = pairwise(HalfNormal, [1.0, 2.0, 1.0, 2.0])
        inversegamma = pairwise(
            InverseGamma, [1.0, 2.5, 1.0, 2.5], [1.5, 1.5, 3.5, 3.5]
        )
        laplace = pairwise(Laplace, [-2.0, 4.0, -3.0, 6.0], [1.0, 2.5, 1.0, 2.5])
        lognormal = pairwise(LogNormal, [-2.0, 2.0, -3.0, 3.0], [1.0, 2.0, 1.0, 2.0])
        normal = pairwise(Normal, [-2.0, 2.0, -3.0, 3.0], [1.0, 2.0, 1.0, 2.0])
        independent = (Independent(normal[0], 1), Independent(normal[1], 1))
        onehotcategorical = pairwise(
            OneHotCategorical,
            [[0.4, 0.3, 0.3], [0.2, 0.7, 0.1], [0.33, 0.33, 0.34], [0.2, 0.2, 0.6]],
        )
        pareto = (
            Pareto(
                torch.tensor([2.5, 4.0, 2.5, 4.0]).expand(4, 4),
                torch.tensor([2.25, 3.75, 2.25, 3.75]).expand(4, 4),
            ),
            Pareto(
                torch.tensor([2.25, 3.75, 2.25, 3.8]).expand(4, 4),
                torch.tensor([2.25, 3.75, 2.25, 3.75]).expand(4, 4),
            ),
        )
        poisson = pairwise(Poisson, [0.3, 1.0, 5.0, 10.0])
        uniform_within_unit = pairwise(
            Uniform, [0.1, 0.9, 0.2, 0.75], [0.15, 0.95, 0.25, 0.8]
        )
        uniform_positive = pairwise(Uniform, [1, 1.5, 2, 4], [1.2, 2.0, 3, 7])
        uniform_real = pairwise(Uniform, [-2.0, -1, 0, 2], [-1.0, 1, 1, 4])
        uniform_pareto = pairwise(Uniform, [6.5, 7.5, 6.5, 8.5], [7.5, 8.5, 9.5, 9.5])
        continuous_bernoulli = pairwise(ContinuousBernoulli, [0.1, 0.2, 0.5, 0.9])

        # These tests should pass with precision = 0.01, but that makes tests very expensive.
        # Instead, we test with precision = 0.1 and only test with higher precision locally
        # when adding a new KL implementation.
        # The following pairs are not tested due to very high variance of the monte carlo
        # estimator; their implementations have been reviewed with extra care:
        # - (pareto, normal)
        self.precision = 0.1  # Set this to 0.01 when testing a new KL implementation.
        self.max_samples = int(1e07)  # Increase this when testing at smaller precision.
        self.samples_per_batch = int(1e04)
        self.finite_examples = [
            (bernoulli, bernoulli),
            (bernoulli, poisson),
            (beta, beta),
            (beta, chi2),
            (beta, exponential),
            (beta, gamma),
            (beta, normal),
            (binomial30, binomial30),
            (binomial_vectorized_count, binomial_vectorized_count),
            (categorical, categorical),
            (cauchy, cauchy),
            (chi2, chi2),
            (chi2, exponential),
            (chi2, gamma),
            (chi2, normal),
            (dirichlet, dirichlet),
            (exponential, chi2),
            (exponential, exponential),
            (exponential, gamma),
            (exponential, gumbel),
            (exponential, normal),
            (gamma, chi2),
            (gamma, exponential),
            (gamma, gamma),
            (gamma, gumbel),
            (gamma, normal),
            (gumbel, gumbel),
            (gumbel, normal),
            (halfnormal, halfnormal),
            (independent, independent),
            (inversegamma, inversegamma),
            (laplace, laplace),
            (lognormal, lognormal),
            (laplace, normal),
            (normal, gumbel),
            (normal, laplace),
            (normal, normal),
            (onehotcategorical, onehotcategorical),
            (pareto, chi2),
            (pareto, pareto),
            (pareto, exponential),
            (pareto, gamma),
            (poisson, poisson),
            (uniform_within_unit, beta),
            (uniform_positive, chi2),
            (uniform_positive, exponential),
            (uniform_positive, gamma),
            (uniform_real, gumbel),
            (uniform_real, normal),
            (uniform_pareto, pareto),
            (continuous_bernoulli, continuous_bernoulli),
            (continuous_bernoulli, exponential),
            (continuous_bernoulli, normal),
            (beta, continuous_bernoulli),
        ]

        self.infinite_examples = [
            (Bernoulli(0), Bernoulli(1)),
            (Bernoulli(1), Bernoulli(0)),
            (
                Categorical(torch.tensor([0.9, 0.1])),
                Categorical(torch.tensor([1.0, 0.0])),
            ),
            (
                Categorical(torch.tensor([[0.9, 0.1], [0.9, 0.1]])),
                Categorical(torch.tensor([1.0, 0.0])),
            ),
            (Beta(1, 2), Uniform(0.25, 1)),
            (Beta(1, 2), Uniform(0, 0.75)),
            (Beta(1, 2), Uniform(0.25, 0.75)),
            (Beta(1, 2), Pareto(1, 2)),
            (Binomial(31, 0.7), Binomial(30, 0.3)),
            (
                Binomial(torch.tensor([3, 4]), torch.tensor([0.4, 0.6])),
                Binomial(torch.tensor([2, 3]), torch.tensor([0.5, 0.8])),
            ),
            (Chi2(1), Beta(2, 3)),
            (Chi2(1), Pareto(2, 3)),
            (Chi2(1), Uniform(-2, 3)),
            (Exponential(1), Beta(2, 3)),
            (Exponential(1), Pareto(2, 3)),
            (Exponential(1), Uniform(-2, 3)),
            (Gamma(1, 2), Beta(3, 4)),
            (Gamma(1, 2), Pareto(3, 4)),
            (Gamma(1, 2), Uniform(-3, 4)),
            (Gumbel(-1, 2), Beta(3, 4)),
            (Gumbel(-1, 2), Chi2(3)),
            (Gumbel(-1, 2), Exponential(3)),
            (Gumbel(-1, 2), Gamma(3, 4)),
            (Gumbel(-1, 2), Pareto(3, 4)),
            (Gumbel(-1, 2), Uniform(-3, 4)),
            (Laplace(-1, 2), Beta(3, 4)),
            (Laplace(-1, 2), Chi2(3)),
            (Laplace(-1, 2), Exponential(3)),
            (Laplace(-1, 2), Gamma(3, 4)),
            (Laplace(-1, 2), Pareto(3, 4)),
            (Laplace(-1, 2), Uniform(-3, 4)),
            (Normal(-1, 2), Beta(3, 4)),
            (Normal(-1, 2), Chi2(3)),
            (Normal(-1, 2), Exponential(3)),
            (Normal(-1, 2), Gamma(3, 4)),
            (Normal(-1, 2), Pareto(3, 4)),
            (Normal(-1, 2), Uniform(-3, 4)),
            (Pareto(2, 1), Chi2(3)),
            (Pareto(2, 1), Exponential(3)),
            (Pareto(2, 1), Gamma(3, 4)),
            (Pareto(1, 2), Normal(-3, 4)),
            (Pareto(1, 2), Pareto(3, 4)),
            (Poisson(2), Bernoulli(0.5)),
            (Poisson(2.3), Binomial(10, 0.2)),
            (Uniform(-1, 1), Beta(2, 2)),
            (Uniform(0, 2), Beta(3, 4)),
            (Uniform(-1, 2), Beta(3, 4)),
            (Uniform(-1, 2), Chi2(3)),
            (Uniform(-1, 2), Exponential(3)),
            (Uniform(-1, 2), Gamma(3, 4)),
            (Uniform(-1, 2), Pareto(3, 4)),
            (ContinuousBernoulli(0.25), Uniform(0.25, 1)),
            (ContinuousBernoulli(0.25), Uniform(0, 0.75)),
            (ContinuousBernoulli(0.25), Uniform(0.25, 0.75)),
            (ContinuousBernoulli(0.25), Pareto(1, 2)),
            (Exponential(1), ContinuousBernoulli(0.75)),
            (Gamma(1, 2), ContinuousBernoulli(0.75)),
            (Gumbel(-1, 2), ContinuousBernoulli(0.75)),
            (Laplace(-1, 2), ContinuousBernoulli(0.75)),
            (Normal(-1, 2), ContinuousBernoulli(0.75)),
            (Uniform(-1, 1), ContinuousBernoulli(0.75)),
            (Uniform(0, 2), ContinuousBernoulli(0.75)),
            (Uniform(-1, 2), ContinuousBernoulli(0.75)),
        ]

    def test_kl_monte_carlo(self):
        # [FUNCTION DOC] For each finite example pair, compares the analytic kl_divergence with
        # a Monte Carlo estimate (mean of log_prob difference over samples from p, drawing
        # batches until the relative error ignoring NaN is below self.precision or max_samples
        # is reached) and asserts the final error is below self.precision. Seed 0.
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for (p, _), (_, q) in self.finite_examples:
            actual = kl_divergence(p, q)
            numerator = 0
            denominator = 0
            while denominator < self.max_samples:
                x = p.sample(sample_shape=(self.samples_per_batch,))
                numerator += (p.log_prob(x) - q.log_prob(x)).sum(0)
                denominator += x.size(0)
                expected = numerator / denominator
                error = torch.abs(expected - actual) / (1 + expected)
                if error[error == error].max() < self.precision:
                    break
            self.assertLess(
                error[error == error].max(),
                self.precision,
                "\n".join(
                    [
                        f"Incorrect KL({type(p).__name__}, {type(q).__name__}).",
                        f"Expected ({denominator} Monte Carlo samples): {expected}",
                        f"Actual (analytic): {actual}",
                    ]
                ),
            )

    # Multivariate normal has a separate Monte Carlo based test due to the requirement of random generation of
    # positive (semi) definite matrices. n is set to 5, but can be increased during testing.
    def test_kl_multivariate_normal(self):
        # [FUNCTION DOC] Monte Carlo check of kl_divergence between 5 random pairs of 4-d
        # MultivariateNormal distributions (random lower-triangular scales) against the
        # estimator, with the same batched stopping rule and precision as test_kl_monte_carlo.
        # Seed 0.
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        n = 5  # Number of tests for multivariate_normal
        for i in range(n):
            loc = [torch.randn(4) for _ in range(2)]
            scale_tril = [
                transform_to(constraints.lower_cholesky)(torch.randn(4, 4))
                for _ in range(2)
            ]
            p = MultivariateNormal(loc=loc[0], scale_tril=scale_tril[0])
            q = MultivariateNormal(loc=loc[1], scale_tril=scale_tril[1])
            actual = kl_divergence(p, q)
            numerator = 0
            denominator = 0
            while denominator < self.max_samples:
                x = p.sample(sample_shape=(self.samples_per_batch,))
                numerator += (p.log_prob(x) - q.log_prob(x)).sum(0)
                denominator += x.size(0)
                expected = numerator / denominator
                error = torch.abs(expected - actual) / (1 + expected)
                if error[error == error].max() < self.precision:
                    break
            self.assertLess(
                error[error == error].max(),
                self.precision,
                "\n".join(
                    [
                        f"Incorrect KL(MultivariateNormal, MultivariateNormal) instance {i + 1}/{n}",
                        f"Expected ({denominator} Monte Carlo sample): {expected}",
                        f"Actual (analytic): {actual}",
                    ]
                ),
            )

    def test_kl_multivariate_normal_batched(self):
        # [FUNCTION DOC] Checks batched MultivariateNormal KL (7 batches of dim 3) equals the
        # stack of per-batch unbatched KL values.
        b = 7  # Number of batches
        loc = [torch.randn(b, 3) for _ in range(2)]
        scale_tril = [
            transform_to(constraints.lower_cholesky)(torch.randn(b, 3, 3))
            for _ in range(2)
        ]
        expected_kl = torch.stack(
            [
                kl_divergence(
                    MultivariateNormal(loc[0][i], scale_tril=scale_tril[0][i]),
                    MultivariateNormal(loc[1][i], scale_tril=scale_tril[1][i]),
                )
                for i in range(b)
            ]
        )
        actual_kl = kl_divergence(
            MultivariateNormal(loc[0], scale_tril=scale_tril[0]),
            MultivariateNormal(loc[1], scale_tril=scale_tril[1]),
        )
        self.assertEqual(expected_kl, actual_kl)

    def test_kl_multivariate_normal_batched_broadcasted(self):
        # [FUNCTION DOC] Like the batched KL test but q has an unbatched scale_tril that must
        # broadcast against the batched p; compares with the stacked per-batch result.
        b = 7  # Number of batches
        loc = [torch.randn(b, 3) for _ in range(2)]
        scale_tril = [
            transform_to(constraints.lower_cholesky)(torch.randn(b, 3, 3)),
            transform_to(constraints.lower_cholesky)(torch.randn(3, 3)),
        ]
        expected_kl = torch.stack(
            [
                kl_divergence(
                    MultivariateNormal(loc[0][i], scale_tril=scale_tril[0][i]),
                    MultivariateNormal(loc[1][i], scale_tril=scale_tril[1]),
                )
                for i in range(b)
            ]
        )
        actual_kl = kl_divergence(
            MultivariateNormal(loc[0], scale_tril=scale_tril[0]),
            MultivariateNormal(loc[1], scale_tril=scale_tril[1]),
        )
        self.assertEqual(expected_kl, actual_kl)

    def test_kl_lowrank_multivariate_normal(self):
        # [FUNCTION DOC] Checks KL between LowRankMultivariateNormal pairs, and mixed
        # low-rank/full pairs in both orders, against KL between the equivalent dense
        # MultivariateNormal distributions (5 random instances, seed 0, maximum absolute error
        # below self.precision).
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        n = 5  # Number of tests for lowrank_multivariate_normal
        for i in range(n):
            loc = [torch.randn(4) for _ in range(2)]
            cov_factor = [torch.randn(4, 3) for _ in range(2)]
            cov_diag = [
                transform_to(constraints.positive)(torch.randn(4)) for _ in range(2)
            ]
            covariance_matrix = [
                cov_factor[i].matmul(cov_factor[i].t()) + cov_diag[i].diag()
                for i in range(2)
            ]
            p = LowRankMultivariateNormal(loc[0], cov_factor[0], cov_diag[0])
            q = LowRankMultivariateNormal(loc[1], cov_factor[1], cov_diag[1])
            p_full = MultivariateNormal(loc[0], covariance_matrix[0])
            q_full = MultivariateNormal(loc[1], covariance_matrix[1])
            expected = kl_divergence(p_full, q_full)

            actual_lowrank_lowrank = kl_divergence(p, q)
            actual_lowrank_full = kl_divergence(p, q_full)
            actual_full_lowrank = kl_divergence(p_full, q)

            error_lowrank_lowrank = torch.abs(actual_lowrank_lowrank - expected).max()
            self.assertLess(
                error_lowrank_lowrank,
                self.precision,
                "\n".join(
                    [
                        f"Incorrect KL(LowRankMultivariateNormal, LowRankMultivariateNormal) instance {i + 1}/{n}",
                        f"Expected (from KL MultivariateNormal): {expected}",
                        f"Actual (analytic): {actual_lowrank_lowrank}",
                    ]
                ),
            )

            error_lowrank_full = torch.abs(actual_lowrank_full - expected).max()
            self.assertLess(
                error_lowrank_full,
                self.precision,
                "\n".join(
                    [
                        f"Incorrect KL(LowRankMultivariateNormal, MultivariateNormal) instance {i + 1}/{n}",
                        f"Expected (from KL MultivariateNormal): {expected}",
                        f"Actual (analytic): {actual_lowrank_full}",
                    ]
                ),
            )

            error_full_lowrank = torch.abs(actual_full_lowrank - expected).max()
            self.assertLess(
                error_full_lowrank,
                self.precision,
                "\n".join(
                    [
                        f"Incorrect KL(MultivariateNormal, LowRankMultivariateNormal) instance {i + 1}/{n}",
                        f"Expected (from KL MultivariateNormal): {expected}",
                        f"Actual (analytic): {actual_full_lowrank}",
                    ]
                ),
            )

    def test_kl_lowrank_multivariate_normal_batched(self):
        # [FUNCTION DOC] Checks batched LowRankMultivariateNormal KL (7 batches) equals the
        # stack of per-batch KL values.
        b = 7  # Number of batches
        loc = [torch.randn(b, 3) for _ in range(2)]
        cov_factor = [torch.randn(b, 3, 2) for _ in range(2)]
        cov_diag = [
            transform_to(constraints.positive)(torch.randn(b, 3)) for _ in range(2)
        ]
        expected_kl = torch.stack(
            [
                kl_divergence(
                    LowRankMultivariateNormal(
                        loc[0][i], cov_factor[0][i], cov_diag[0][i]
                    ),
                    LowRankMultivariateNormal(
                        loc[1][i], cov_factor[1][i], cov_diag[1][i]
                    ),
                )
                for i in range(b)
            ]
        )
        actual_kl = kl_divergence(
            LowRankMultivariateNormal(loc[0], cov_factor[0], cov_diag[0]),
            LowRankMultivariateNormal(loc[1], cov_factor[1], cov_diag[1]),
        )
        self.assertEqual(expected_kl, actual_kl)

    def test_kl_exponential_family(self):
        # [FUNCTION DOC] For same-type ExponentialFamily pairs from
        # finite_examples checks kl_divergence equals the generic Bregman
        # divergence implementation _kl_expfamily_expfamily.
        for (p, _), (_, q) in self.finite_examples:
            if type(p) is type(q) and issubclass(type(p), ExponentialFamily):
                actual = kl_divergence(p, q)
                expected = _kl_expfamily_expfamily(p, q)
                self.assertEqual(
                    actual,
                    expected,
                    msg="\n".join(
                        [
                            f"Incorrect KL({type(p).__name__}, {type(q).__name__}).",
                            f"Expected (using Bregman Divergence) {expected}",
                            f"Actual (analytic) {actual}",
                            f"max error = {torch.abs(actual - expected).max()}",
                        ]
                    ),
                )

    def test_kl_infinite(self):
        # [FUNCTION DOC] Asserts kl_divergence is infinity for every pair in infinite_examples
        # (supports not contained or incompatible parameters).
        for p, q in self.infinite_examples:
            self.assertTrue(
                (kl_divergence(p, q) == inf).all(),
                lambda msg: f"{msg}\nIncorrect KL({type(p).__name__}, {type(q).__name__})",
            )

    def test_kl_edgecases(self):
        # [FUNCTION DOC] Checks KL edge cases equal 0: identical degenerate Bernoulli (probs 0
        # and 1), identical Categorical with a zero probability, and Uniform(0, 1) versus
        # Beta(1, 1).
        self.assertEqual(kl_divergence(Bernoulli(0), Bernoulli(0)), 0)
        self.assertEqual(kl_divergence(Bernoulli(1), Bernoulli(1)), 0)
        self.assertEqual(
            kl_divergence(
                Categorical(torch.tensor([0.0, 1.0])),
                Categorical(torch.tensor([0.0, 1.0])),
            ),
            0,
        )
        self.assertEqual(kl_divergence(Uniform(0, 1), Beta(1, 1)), 0)

    def test_kl_shape(self):
        # [FUNCTION DOC] For every example whose kl_divergence(dist, dist) is implemented,
        # asserts the result shape equals the distribution batch shape.
        for Dist, params in _get_examples():
            for i, param in enumerate(params):
                dist = Dist(**param)
                try:
                    kl = kl_divergence(dist, dist)
                except NotImplementedError:
                    continue
                expected_shape = dist.batch_shape if dist.batch_shape else torch.Size()
                self.assertEqual(
                    kl.shape,
                    expected_shape,
                    msg="\n".join(
                        [
                            f"{Dist.__name__} example {i + 1}/{len(params)}",
                            f"Expected {expected_shape}",
                            f"Actual {kl.shape}",
                        ]
                    ),
                )

    def test_kl_transformed(self):
        # [FUNCTION DOC] Regression test for gh-34859: KL between an Independent Normal and
        # itself, and between an affine TransformedDistribution of it and itself, has the batch
        # shape (2,).
        # Regression test for https://github.com/pytorch/pytorch/issues/34859
        scale = torch.ones(2, 3)
        loc = torch.zeros(2, 3)
        normal = Normal(loc=loc, scale=scale)
        diag_normal = Independent(normal, reinterpreted_batch_ndims=1)
        trans_dist = TransformedDistribution(
            diag_normal, AffineTransform(loc=0.0, scale=2.0)
        )
        self.assertEqual(kl_divergence(diag_normal, diag_normal).shape, (2,))
        self.assertEqual(kl_divergence(trans_dist, trans_dist).shape, (2,))

    @set_default_dtype(torch.double)
    def test_entropy_monte_carlo(self):
        # [FUNCTION DOC] For every example with an implemented entropy, compares the analytic
        # entropy with the negative mean log_prob over 60000 samples (atol 0.2; infinite Monte
        # Carlo entries are replaced by the analytic value). Seed 0, double precision.
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for Dist, params in _get_examples():
            for i, param in enumerate(params):
                dist = Dist(**param)
                try:
                    actual = dist.entropy()
                except NotImplementedError:
                    continue
                x = dist.sample(sample_shape=(60000,))
                expected = -dist.log_prob(x).mean(0)
                ignore = (expected == inf) | (expected == -inf)
                expected[ignore] = actual[ignore]
                self.assertEqual(
                    actual,
                    expected,
                    atol=0.2,
                    rtol=0,
                    msg="\n".join(
                        [
                            f"{Dist.__name__} example {i + 1}/{len(params)}, incorrect .entropy().",
                            f"Expected (monte carlo) {expected}",
                            f"Actual (analytic) {actual}",
                            f"max error = {torch.abs(actual - expected).max()}",
                        ]
                    ),
                )

    @set_default_dtype(torch.double)
    def test_entropy_exponential_family(self):
        # [FUNCTION DOC] For ExponentialFamily examples, checks the distribution's entropy
        # equals the generic ExponentialFamily.entropy (Bregman divergence) implementation;
        # either side raising NotImplementedError skips the case.
        for Dist, params in _get_examples():
            if not issubclass(Dist, ExponentialFamily):
                continue
            for i, param in enumerate(params):
                dist = Dist(**param)
                try:
                    actual = dist.entropy()
                except NotImplementedError:
                    continue
                try:
                    expected = ExponentialFamily.entropy(dist)
                except NotImplementedError:
                    continue
                self.assertEqual(
                    actual,
                    expected,
                    msg="\n".join(
                        [
                            f"{Dist.__name__} example {i + 1}/{len(params)}, incorrect .entropy().",
                            f"Expected (Bregman Divergence) {expected}",
                            f"Actual (analytic) {actual}",
                            f"max error = {torch.abs(actual - expected).max()}",
                        ]
                    ),
                )


class TestConstraints(DistributionsTestCase):
    # [CLASS DOC] Tests that distribution argument constraints and supports are consistent with
    # the example parameters and samples.
    def test_params_constraints(self):
        # [FUNCTION DOC] For each example and each parameter in arg_constraints: asserts the
        # value has at least the constraint's event_dim dimensions, that its batch shape
        # broadcasts with the distribution batch_shape, and (for non-dependent constraints) that
        # the value satisfies the constraint. Scalars are wrapped in tensors, and probs of
        # categorical-type distributions are normalized. Parameters without a constraint entry
        # are ignored.
        normalize_probs_dists = (
            Categorical,
            Multinomial,
            OneHotCategorical,
            OneHotCategoricalStraightThrough,
            RelaxedOneHotCategorical,
        )

        for Dist, params in _get_examples():
            for i, param in enumerate(params):
                dist = Dist(**param)
                for name, value in param.items():
                    if isinstance(value, numbers.Number):
                        value = torch.tensor([value])
                    if Dist in normalize_probs_dists and name == "probs":
                        # These distributions accept positive probs, but elsewhere we
                        # use a stricter constraint to the simplex.
                        value = value / value.sum(-1, True)
                    try:
                        constraint = dist.arg_constraints[name]
                    except KeyError:
                        continue  # ignore optional parameters

                    # Check param shape is compatible with distribution shape.
                    self.assertGreaterEqual(value.dim(), constraint.event_dim)
                    value_batch_shape = value.shape[
                        : value.dim() - constraint.event_dim
                    ]
                    torch.broadcast_shapes(dist.batch_shape, value_batch_shape)

                    if is_dependent(constraint):
                        continue

                    message = f"{Dist.__name__} example {i + 1}/{len(params)} parameter {name} = {value}"
                    self.assertTrue(constraint.check(value).all(), msg=message)

    def test_support_constraints(self):
        # [FUNCTION DOC] For each example asserts the class support is a Constraint, and that a
        # sample satisfies the instance support with the constraint event_dim equal to the event
        # rank and a check result shape equal to batch_shape.
        for Dist, params in _get_examples():
            self.assertIsInstance(Dist.support, Constraint)
            for i, param in enumerate(params):
                dist = Dist(**param)
                value = dist.sample()
                constraint = dist.support
                message = (
                    f"{Dist.__name__} example {i + 1}/{len(params)} sample = {value}"
                )
                self.assertEqual(
                    constraint.event_dim, len(dist.event_shape), msg=message
                )
                ok = constraint.check(value)
                self.assertEqual(ok.shape, dist.batch_shape, msg=message)
                self.assertTrue(ok.all(), msg=message)


@skipIfTorchDynamo("Not a TorchDynamo suitable test")
class TestNumericalStability(DistributionsTestCase):
    # [CLASS DOC] Tests log_prob values and gradients at numerical extremes
    # (zero/one probabilities, huge logits) for Bernoulli, Categorical, Multinomial
    # and ContinuousBernoulli.
    def _test_pdf_score(
        self,
        dist_class,
        x,
        expected_value,
        probs=None,
        logits=None,
        expected_gradient=None,
        atol=1e-5,
    ):
        # [FUNCTION DOC] HELPER: Builds dist_class from probs or logits (detached, requiring
        # grad), evaluates log_prob at x, backpropagates the sum, and asserts the log_prob
        # equals expected_value within atol and, if given, the parameter gradient equals
        # expected_gradient within atol.
        if probs is not None:
            p = probs.detach().requires_grad_()
            dist = dist_class(p)
        else:
            p = logits.detach().requires_grad_()
            dist = dist_class(logits=p)
        log_pdf = dist.log_prob(x)
        log_pdf.sum().backward()
        self.assertEqual(
            log_pdf,
            expected_value,
            atol=atol,
            rtol=0,
            msg=lambda msg: f"{msg}\nIncorrect value for tensor type: {type(x)}. Expected = {expected_value}, Actual = {log_pdf}",
        )
        if expected_gradient is not None:
            self.assertEqual(
                p.grad,
                expected_gradient,
                atol=atol,
                rtol=0,
                msg=lambda msg: f"{msg}\nIncorrect gradient for tensor type: {type(x)}. Expected = {expected_gradient}, Actual = {p.grad}",
            )

    def test_bernoulli_gradient(self):
        # [FUNCTION DOC] Checks Bernoulli log_prob and gradient for float and double tensors at
        # probs 0, 1e-4 and 1 - 1e-4 and logits log(9999), including the clamped log(eps) value
        # for probability 0.
        for tensor_type in [torch.FloatTensor, torch.DoubleTensor]:
            self._test_pdf_score(
                dist_class=Bernoulli,
                probs=tensor_type([0]),
                x=tensor_type([0]),
                expected_value=tensor_type([0]),
                expected_gradient=tensor_type([0]),
            )

            self._test_pdf_score(
                dist_class=Bernoulli,
                probs=tensor_type([0]),
                x=tensor_type([1]),
                expected_value=tensor_type(
                    [torch.finfo(tensor_type([]).dtype).eps]
                ).log(),
                expected_gradient=tensor_type([0]),
            )

            self._test_pdf_score(
                dist_class=Bernoulli,
                probs=tensor_type([1e-4]),
                x=tensor_type([1]),
                expected_value=tensor_type([math.log(1e-4)]),
                expected_gradient=tensor_type([10000]),
            )

            # Lower precision due to:
            # >>> 1 / (1 - torch.FloatTensor([0.9999]))
            # 9998.3408
            # [torch.FloatTensor of size 1]
            self._test_pdf_score(
                dist_class=Bernoulli,
                probs=tensor_type([1 - 1e-4]),
                x=tensor_type([0]),
                expected_value=tensor_type([math.log(1e-4)]),
                expected_gradient=tensor_type([-10000]),
                atol=2,
            )

            self._test_pdf_score(
                dist_class=Bernoulli,
                logits=tensor_type([math.log(9999)]),
                x=tensor_type([0]),
                expected_value=tensor_type([math.log(1e-4)]),
                expected_gradient=tensor_type([-1]),
                atol=1e-3,
            )

    def test_bernoulli_with_logits_underflow(self):
        # [FUNCTION DOC] Checks Bernoulli with very negative logits (about -1e38 or -1e308)
        # gives log_prob 0 and gradient 0 at value 0 (no underflow problems).
        for tensor_type, lim in [
            (torch.FloatTensor, -1e38),
            (torch.DoubleTensor, -1e308),
        ]:
            self._test_pdf_score(
                dist_class=Bernoulli,
                logits=tensor_type([lim]),
                x=tensor_type([0]),
                expected_value=tensor_type([0]),
                expected_gradient=tensor_type([0]),
            )

    def test_bernoulli_with_logits_overflow(self):
        # [FUNCTION DOC] Checks Bernoulli with very large logits (about 1e38 or 1e308) gives
        # log_prob 0 and gradient 0 at value 1 (no overflow problems).
        for tensor_type, lim in [
            (torch.FloatTensor, 1e38),
            (torch.DoubleTensor, 1e308),
        ]:
            self._test_pdf_score(
                dist_class=Bernoulli,
                logits=tensor_type([lim]),
                x=tensor_type([1]),
                expected_value=tensor_type([0]),
                expected_gradient=tensor_type([0]),
            )

    def test_categorical_log_prob(self):
        # [FUNCTION DOC] Checks OneHotCategorical with probs [0, 1] gives log_prob 0 for the
        # one-hot sample [0, 1], in float and double.
        for dtype in [torch.float, torch.double]:
            p = torch.tensor([0, 1], dtype=dtype, requires_grad=True)
            categorical = OneHotCategorical(p)
            log_pdf = categorical.log_prob(torch.tensor([0, 1], dtype=dtype))
            self.assertEqual(log_pdf.item(), 0)

    def test_categorical_log_prob_with_logits(self):
        # [FUNCTION DOC] Checks OneHotCategorical with logits [-inf, 0]: log_prob is 0 for the
        # likely sample and minus infinity for the impossible one, in float and double.
        for dtype in [torch.float, torch.double]:
            p = torch.tensor([-inf, 0], dtype=dtype, requires_grad=True)
            categorical = OneHotCategorical(logits=p)
            log_pdf_prob_1 = categorical.log_prob(torch.tensor([0, 1], dtype=dtype))
            self.assertEqual(log_pdf_prob_1.item(), 0)
            log_pdf_prob_0 = categorical.log_prob(torch.tensor([1, 0], dtype=dtype))
            self.assertEqual(log_pdf_prob_0.item(), -inf)

    def test_multinomial_log_prob(self):
        # [FUNCTION DOC] Checks Multinomial with probs [0, 1] and total_count 10 gives log_prob
        # 0 for the sample [0, 10].
        for dtype in [torch.float, torch.double]:
            p = torch.tensor([0, 1], dtype=dtype, requires_grad=True)
            s = torch.tensor([0, 10], dtype=dtype)
            multinomial = Multinomial(10, p)
            log_pdf = multinomial.log_prob(s)
            self.assertEqual(log_pdf.item(), 0)

    def test_multinomial_log_prob_with_logits(self):
        # [FUNCTION DOC] Checks Multinomial with logits [-inf, 0]: log_prob is 0 for [0, 10] and
        # minus infinity for [10, 0].
        for dtype in [torch.float, torch.double]:
            p = torch.tensor([-inf, 0], dtype=dtype, requires_grad=True)
            multinomial = Multinomial(10, logits=p)
            log_pdf_prob_1 = multinomial.log_prob(torch.tensor([0, 10], dtype=dtype))
            self.assertEqual(log_pdf_prob_1.item(), 0)
            log_pdf_prob_0 = multinomial.log_prob(torch.tensor([10, 0], dtype=dtype))
            self.assertEqual(log_pdf_prob_0.item(), -inf)

    def test_continuous_bernoulli_gradient(self):
        # [FUNCTION DOC] Checks ContinuousBernoulli log_prob values and gradients against
        # analytic references for probs near 0.1, 0.4999 (Taylor-expansion region), 1e-4, 1 -
        # 1e-4 and for logits, in float and double. The reference functions expec_val and
        # expec_grad implement the closed forms.
        def expec_val(x, probs=None, logits=None):
            # [FUNCTION DOC] HELPER: Reference ContinuousBernoulli log-likelihood at x for
            # given probs or logits (logits converted to probs); raises AssertionError if
            # neither is given. Uses the atanh normalizer outside (0.499, 0.501) and a series
            # expansion inside.
            if probs is None and logits is None:
                raise AssertionError("At least one of probs or logits must be provided")
            if logits is not None:
                probs = 1.0 / (1.0 + math.exp(-logits))
            bern_log_lik = x * math.log(probs) + (1.0 - x) * math.log1p(-probs)
            if probs < 0.499 or probs > 0.501:  # using default values of lims here
                log_norm_const = (
                    math.log(math.fabs(math.atanh(1.0 - 2.0 * probs)))
                    - math.log(math.fabs(1.0 - 2.0 * probs))
                    + math.log(2.0)
                )
            else:
                aux = math.pow(probs - 0.5, 2)
                log_norm_const = math.log(2.0) + (4.0 / 3.0 + 104.0 / 45.0 * aux) * aux
            log_lik = bern_log_lik + log_norm_const
            return log_lik

        def expec_grad(x, probs=None, logits=None):
            # [FUNCTION DOC] HELPER: Reference derivative of the ContinuousBernoulli
            # log-likelihood with respect to probs, or with respect to logits via the chain rule
            # when logits are given; raises AssertionError if neither is provided.
            if probs is None and logits is None:
                raise AssertionError("At least one of probs or logits must be provided")
            if logits is not None:
                probs = 1.0 / (1.0 + math.exp(-logits))
            grad_bern_log_lik = x / probs - (1.0 - x) / (1.0 - probs)
            if probs < 0.499 or probs > 0.501:  # using default values of lims here
                grad_log_c = (
                    2.0 * probs
                    - 4.0 * (probs - 1.0) * probs * math.atanh(1.0 - 2.0 * probs)
                    - 1.0
                )
                grad_log_c /= (
                    2.0
                    * (probs - 1.0)
                    * probs
                    * (2.0 * probs - 1.0)
                    * math.atanh(1.0 - 2.0 * probs)
                )
            else:
                grad_log_c = 8.0 / 3.0 * (probs - 0.5) + 416.0 / 45.0 * math.pow(
                    probs - 0.5, 3
                )
            grad = grad_bern_log_lik + grad_log_c
            if logits is not None:
                grad *= 1.0 / (1.0 + math.exp(logits)) - 1.0 / math.pow(
                    1.0 + math.exp(logits), 2
                )
            return grad

        for tensor_type in [torch.FloatTensor, torch.DoubleTensor]:
            self._test_pdf_score(
                dist_class=ContinuousBernoulli,
                probs=tensor_type([0.1]),
                x=tensor_type([0.1]),
                expected_value=tensor_type([expec_val(0.1, probs=0.1)]),
                expected_gradient=tensor_type([expec_grad(0.1, probs=0.1)]),
            )

            self._test_pdf_score(
                dist_class=ContinuousBernoulli,
                probs=tensor_type([0.1]),
                x=tensor_type([1.0]),
                expected_value=tensor_type([expec_val(1.0, probs=0.1)]),
                expected_gradient=tensor_type([expec_grad(1.0, probs=0.1)]),
            )

            self._test_pdf_score(
                dist_class=ContinuousBernoulli,
                probs=tensor_type([0.4999]),
                x=tensor_type([0.9]),
                expected_value=tensor_type([expec_val(0.9, probs=0.4999)]),
                expected_gradient=tensor_type([expec_grad(0.9, probs=0.4999)]),
            )

            self._test_pdf_score(
                dist_class=ContinuousBernoulli,
                probs=tensor_type([1e-4]),
                x=tensor_type([1]),
                expected_value=tensor_type([expec_val(1, probs=1e-4)]),
                expected_gradient=tensor_type(tensor_type([expec_grad(1, probs=1e-4)])),
                atol=1e-3,
            )

            self._test_pdf_score(
                dist_class=ContinuousBernoulli,
                probs=tensor_type([1 - 1e-4]),
                x=tensor_type([0.1]),
                expected_value=tensor_type([expec_val(0.1, probs=1 - 1e-4)]),
                expected_gradient=tensor_type([expec_grad(0.1, probs=1 - 1e-4)]),
                atol=2,
            )

            self._test_pdf_score(
                dist_class=ContinuousBernoulli,
                logits=tensor_type([math.log(9999)]),
                x=tensor_type([0]),
                expected_value=tensor_type([expec_val(0, logits=math.log(9999))]),
                expected_gradient=tensor_type([expec_grad(0, logits=math.log(9999))]),
                atol=1e-3,
            )

            self._test_pdf_score(
                dist_class=ContinuousBernoulli,
                logits=tensor_type([0.001]),
                x=tensor_type([0.5]),
                expected_value=tensor_type([expec_val(0.5, logits=0.001)]),
                expected_gradient=tensor_type([expec_grad(0.5, logits=0.001)]),
            )

    def test_continuous_bernoulli_with_logits_underflow(self):
        # [FUNCTION DOC] Checks ContinuousBernoulli with extremely negative logits (-1e38 float,
        # -1e308 double): log_prob at 0 equals the clamped-probability constant (2.76898 or
        # 3.58473) and gradient 0.
        for tensor_type, lim, expected in [
            (torch.FloatTensor, -1e38, 2.76898),
            (torch.DoubleTensor, -1e308, 3.58473),
        ]:
            self._test_pdf_score(
                dist_class=ContinuousBernoulli,
                logits=tensor_type([lim]),
                x=tensor_type([0]),
                expected_value=tensor_type([expected]),
                expected_gradient=tensor_type([0.0]),
            )

    def test_continuous_bernoulli_with_logits_overflow(self):
        # [FUNCTION DOC] Checks ContinuousBernoulli with extremely positive logits: log_prob at
        # 1 equals the clamped-probability constant (2.76898 or 3.58473) and gradient 0.
        for tensor_type, lim, expected in [
            (torch.FloatTensor, 1e38, 2.76898),
            (torch.DoubleTensor, 1e308, 3.58473),
        ]:
            self._test_pdf_score(
                dist_class=ContinuousBernoulli,
                logits=tensor_type([lim]),
                x=tensor_type([1]),
                expected_value=tensor_type([expected]),
                expected_gradient=tensor_type([0.0]),
            )


# TODO: make this a pytest parameterized test
class TestLazyLogitsInitialization(DistributionsTestCase):
    # [CLASS DOC] Checks that distributions defined by either probs or logits (Categorical,
    # OneHotCategorical, Bernoulli, Binomial, Multinomial) compute the other form lazily, i.e.
    # the unused parameter is not stored in the instance dictionary after sampling, log_prob,
    # enumerate_support and shape queries.
    def setUp(self):
        # [FUNCTION DOC] Runs the base setUp and stores in self.examples the example entries of
        # the five tested distribution classes (ContinuousBernoulli is excluded because its
        # log_prob needs both forms).
        super().setUp()
        # ContinuousBernoulli is not tested because log_prob is not computed simply
        # from 'logits', but 'probs' is also needed
        self.examples = [
            e
            for e in _get_examples()
            if e.Dist
            in (Categorical, OneHotCategorical, Bernoulli, Binomial, Multinomial)
        ]

    def test_lazy_logits_initialization(self):
        # [FUNCTION DOC] For the first probs-based example of each class, rebuilds the
        # distribution from logits and checks the probs attribute is never
        # materialized by log_prob of a sample, enumerate_support (NotImplementedError
        # ignored) or batch_shape/event_shape.
        for Dist, params in self.examples:
            param = params[0].copy()
            if "probs" not in param:
                continue
            probs = param.pop("probs")
            param["logits"] = probs_to_logits(probs)
            dist = Dist(**param)
            # Create new instance to generate a valid sample
            dist.log_prob(Dist(**param).sample())
            message = f"Failed for {Dist.__name__} example 0/{len(params)}"
            self.assertNotIn("probs", dist.__dict__, msg=message)
            try:
                dist.enumerate_support()
            except NotImplementedError:
                pass
            self.assertNotIn("probs", dist.__dict__, msg=message)
            _ = (dist.batch_shape, dist.event_shape)
            self.assertNotIn("probs", dist.__dict__, msg=message)

    def test_lazy_probs_initialization(self):
        # [FUNCTION DOC] For the first example of each class built from probs, checks logits is
        # never materialized by sample, enumerate_support or shape queries.
        for Dist, params in self.examples:
            param = params[0].copy()
            if "probs" not in param:
                continue
            dist = Dist(**param)
            dist.sample()
            message = f"Failed for {Dist.__name__} example 0/{len(params)}"
            self.assertNotIn("logits", dist.__dict__, msg=message)
            try:
                dist.enumerate_support()
            except NotImplementedError:
                pass
            self.assertNotIn("logits", dist.__dict__, msg=message)
            _ = (dist.batch_shape, dist.event_shape)
            self.assertNotIn("logits", dist.__dict__, msg=message)


@unittest.skipIf(not TEST_NUMPY, "NumPy not found")
@skipIfTorchDynamo("FIXME: Tries to trace through SciPy and fails")
class TestAgainstScipy(DistributionsTestCase):
    # [CLASS DOC] Compares mean, variance, stddev, cdf and icdf of torch distributions with
    # equivalent scipy frozen distributions built from the same random double-precision
    # parameters. The class is skipped without NumPy and under TorchDynamo.
    def setUp(self):
        # [FUNCTION DOC] Builds self.distribution_pairs: a list of (torch distribution, scipy
        # distribution) pairs for Bernoulli, Beta, Binomial, Cauchy, Dirichlet, Exponential,
        # FisherSnedecor, Gamma, Geometric, Gumbel, GeneralizedPareto, HalfCauchy, HalfNormal,
        # InverseGamma, Laplace, LogNormal, LowRankMultivariateNormal, Multinomial,
        # MultivariateNormal, Normal, OneHotCategorical, Pareto, Poisson, StudentT, Uniform,
        # VonMises, Weibull and Wishart, using random positive, real, simplex and covariance
        # parameters of size 20 (unseeded).
        super().setUp()
        positive_var = torch.randn(20, dtype=torch.double).exp()
        positive_var2 = torch.randn(20, dtype=torch.double).exp()
        random_var = torch.randn(20, dtype=torch.double)
        simplex_tensor = softmax(torch.randn(20, dtype=torch.double), dim=-1)
        cov_tensor = torch.randn(20, 20, dtype=torch.double)
        cov_tensor = cov_tensor @ cov_tensor.mT
        self.distribution_pairs = [
            (Bernoulli(simplex_tensor), scipy.stats.bernoulli(simplex_tensor)),
            (
                Beta(positive_var, positive_var2),
                scipy.stats.beta(positive_var, positive_var2),
            ),
            (
                Binomial(10, simplex_tensor),
                scipy.stats.binom(
                    10 * np.ones(simplex_tensor.shape), simplex_tensor.numpy()
                ),
            ),
            (
                Cauchy(random_var, positive_var),
                scipy.stats.cauchy(loc=random_var, scale=positive_var),
            ),
            (Dirichlet(positive_var), scipy.stats.dirichlet(positive_var)),
            (
                Exponential(positive_var),
                scipy.stats.expon(scale=positive_var.reciprocal()),
            ),
            (
                FisherSnedecor(
                    positive_var, 4 + positive_var2
                ),  # var for df2<=4 is undefined
                scipy.stats.f(positive_var, 4 + positive_var2),
            ),
            (
                Gamma(positive_var, positive_var2),
                scipy.stats.gamma(positive_var, scale=positive_var2.reciprocal()),
            ),
            (Geometric(simplex_tensor), scipy.stats.geom(simplex_tensor, loc=-1)),
            (
                Gumbel(random_var, positive_var2),
                scipy.stats.gumbel_r(random_var, positive_var2),
            ),
            (
                GeneralizedPareto(
                    loc=random_var, scale=positive_var, concentration=random_var / 10
                ),
                scipy.stats.genpareto(
                    c=random_var / 10, loc=random_var, scale=positive_var
                ),
            ),
            (HalfCauchy(positive_var), scipy.stats.halfcauchy(scale=positive_var)),
            (HalfNormal(positive_var2), scipy.stats.halfnorm(scale=positive_var2)),
            (
                InverseGamma(positive_var, positive_var2),
                scipy.stats.invgamma(positive_var, scale=positive_var2),
            ),
            (
                Laplace(random_var, positive_var2),
                scipy.stats.laplace(random_var, positive_var2),
            ),
            (
                # Tests fail 1e-5 threshold if scale > 3
                LogNormal(random_var, positive_var.clamp(max=3)),
                scipy.stats.lognorm(
                    s=positive_var.clamp(max=3), scale=random_var.exp()
                ),
            ),
            (
                LowRankMultivariateNormal(
                    random_var, torch.zeros(20, 1, dtype=torch.double), positive_var2
                ),
                scipy.stats.multivariate_normal(random_var, torch.diag(positive_var2)),
            ),
            (
                Multinomial(10, simplex_tensor),
                scipy.stats.multinomial(10, simplex_tensor),
            ),
            (
                MultivariateNormal(random_var, torch.diag(positive_var2)),
                scipy.stats.multivariate_normal(random_var, torch.diag(positive_var2)),
            ),
            (
                MultivariateNormal(random_var, cov_tensor),
                scipy.stats.multivariate_normal(random_var, cov_tensor),
            ),
            (
                Normal(random_var, positive_var2),
                scipy.stats.norm(random_var, positive_var2),
            ),
            (
                OneHotCategorical(simplex_tensor),
                scipy.stats.multinomial(1, simplex_tensor),
            ),
            (
                Pareto(positive_var, 2 + positive_var2),
                scipy.stats.pareto(2 + positive_var2, scale=positive_var),
            ),
            (Poisson(positive_var), scipy.stats.poisson(positive_var)),
            (
                StudentT(2 + positive_var, random_var, positive_var2),
                scipy.stats.t(2 + positive_var, random_var, positive_var2),
            ),
            (
                Uniform(random_var, random_var + positive_var),
                scipy.stats.uniform(random_var, positive_var),
            ),
            (
                VonMises(random_var, positive_var),
                scipy.stats.vonmises(positive_var, loc=random_var),
            ),
            (
                Weibull(
                    positive_var[0], positive_var2[0]
                ),  # scipy var for Weibull only supports scalars
                scipy.stats.weibull_min(c=positive_var2[0], scale=positive_var[0]),
            ),
            (
                # scipy var for Wishart only supports scalars
                # SciPy allowed ndim -1 < df < ndim for Wishar distribution after version 1.7.0
                Wishart(
                    (
                        20
                        if version.parse(scipy.__version__) < version.parse("1.7.0")
                        else 19
                    )
                    + positive_var[0],
                    cov_tensor,
                ),
                scipy.stats.wishart(
                    (
                        20
                        if version.parse(scipy.__version__) < version.parse("1.7.0")
                        else 19
                    )
                    + positive_var[0].item(),
                    cov_tensor,
                ),
            ),
        ]

    def test_mean(self):
        # [FUNCTION DOC] Asserts the torch mean equals the scipy mean for every pair, skipping
        # Cauchy and HalfCauchy (NaN mean); multivariate normals compare against the scipy
        # mean attribute.
        for pytorch_dist, scipy_dist in self.distribution_pairs:
            if isinstance(pytorch_dist, (Cauchy, HalfCauchy)):
                # Cauchy, HalfCauchy distributions' mean is nan, skipping check
                continue
            elif isinstance(
                pytorch_dist, (LowRankMultivariateNormal, MultivariateNormal)
            ):
                self.assertEqual(pytorch_dist.mean, scipy_dist.mean, msg=pytorch_dist)
            else:
                self.assertEqual(pytorch_dist.mean, scipy_dist.mean(), msg=pytorch_dist)

    def test_variance_stddev(self):
        # [FUNCTION DOC] Asserts torch variance and stddev equal scipy values for every pair.
        # Cauchy, HalfCauchy and VonMises are skipped; Multinomial and OneHotCategorical use the
        # diagonal of the scipy covariance; multivariate normals use the diagonal of the scipy
        # cov attribute.
        for pytorch_dist, scipy_dist in self.distribution_pairs:
            if isinstance(pytorch_dist, (Cauchy, HalfCauchy, VonMises)):
                # Cauchy, HalfCauchy distributions' standard deviation is nan, skipping check
                # VonMises variance is circular and scipy doesn't produce a correct result
                continue
            elif isinstance(pytorch_dist, (Multinomial, OneHotCategorical)):
                self.assertEqual(
                    pytorch_dist.variance, np.diag(scipy_dist.cov()), msg=pytorch_dist
                )
                self.assertEqual(
                    pytorch_dist.stddev,
                    np.diag(scipy_dist.cov()) ** 0.5,
                    msg=pytorch_dist,
                )
            elif isinstance(
                pytorch_dist, (LowRankMultivariateNormal, MultivariateNormal)
            ):
                self.assertEqual(
                    pytorch_dist.variance, np.diag(scipy_dist.cov), msg=pytorch_dist
                )
                self.assertEqual(
                    pytorch_dist.stddev,
                    np.diag(scipy_dist.cov) ** 0.5,
                    msg=pytorch_dist,
                )
            else:
                self.assertEqual(
                    pytorch_dist.variance, scipy_dist.var(), msg=pytorch_dist
                )
                self.assertEqual(
                    pytorch_dist.stddev, scipy_dist.var() ** 0.5, msg=pytorch_dist
                )

    @set_default_dtype(torch.double)
    def test_cdf(self):
        # [FUNCTION DOC] For each pair draws 5 samples (double default dtype) and asserts torch
        # cdf equals scipy cdf; pairs without cdf are skipped.
        for pytorch_dist, scipy_dist in self.distribution_pairs:
            samples = pytorch_dist.sample((5,))
            try:
                cdf = pytorch_dist.cdf(samples)
            except NotImplementedError:
                continue
            self.assertEqual(cdf, scipy_dist.cdf(samples), msg=pytorch_dist)

    def test_icdf(self):
        # [FUNCTION DOC] For each pair evaluates icdf at uniform random probabilities (5 by
        # batch shape) and asserts it equals scipy ppf; pairs without icdf are skipped.
        for pytorch_dist, scipy_dist in self.distribution_pairs:
            samples = torch.rand((5,) + pytorch_dist.batch_shape, dtype=torch.double)
            try:
                icdf = pytorch_dist.icdf(samples)
            except NotImplementedError:
                continue
            self.assertEqual(icdf, scipy_dist.ppf(samples), msg=pytorch_dist)


class TestFunctors(DistributionsTestCase):
    # [CLASS DOC] Tests the CatTransform and StackTransform containers: domain/codomain checks,
    # forward and inverse maps and log_abs_det_jacobian must equal the per-component results
    # combined with concatenation or stacking.
    def test_cat_transform(self):
        # [FUNCTION DOC] Checks CatTransform over three components (ExpTransform,
        # AffineTransform, identity) along dim 0 with equal lengths: domain check, forward
        # value, codomain check, inverse and log_abs_det_jacobian match the concatenation of the
        # component results.
        x1 = -1 * torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        x2 = (torch.arange(1, 101, dtype=torch.float).view(-1, 100) - 1) / 100
        x3 = torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        t1, t2, t3 = ExpTransform(), AffineTransform(1, 100), identity_transform
        dim = 0
        x = torch.cat([x1, x2, x3], dim=dim)
        t = CatTransform([t1, t2, t3], dim=dim)
        actual_dom_check = t.domain.check(x)
        expected_dom_check = torch.cat(
            [t1.domain.check(x1), t2.domain.check(x2), t3.domain.check(x3)], dim=dim
        )
        self.assertEqual(expected_dom_check, actual_dom_check)
        actual = t(x)
        expected = torch.cat([t1(x1), t2(x2), t3(x3)], dim=dim)
        self.assertEqual(expected, actual)
        y1 = torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        y2 = torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        y3 = torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        y = torch.cat([y1, y2, y3], dim=dim)
        actual_cod_check = t.codomain.check(y)
        expected_cod_check = torch.cat(
            [t1.codomain.check(y1), t2.codomain.check(y2), t3.codomain.check(y3)],
            dim=dim,
        )
        self.assertEqual(actual_cod_check, expected_cod_check)
        actual_inv = t.inv(y)
        expected_inv = torch.cat([t1.inv(y1), t2.inv(y2), t3.inv(y3)], dim=dim)
        self.assertEqual(expected_inv, actual_inv)
        actual_jac = t.log_abs_det_jacobian(x, y)
        expected_jac = torch.cat(
            [
                t1.log_abs_det_jacobian(x1, y1),
                t2.log_abs_det_jacobian(x2, y2),
                t3.log_abs_det_jacobian(x3, y3),
            ],
            dim=dim,
        )
        self.assertEqual(actual_jac, expected_jac)

    def test_cat_transform_non_uniform(self):
        # [FUNCTION DOC] Like test_cat_transform but with unequal lengths [1, 2] and a nested
        # CatTransform as the second component.
        x1 = -1 * torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        x2 = torch.cat(
            [
                (torch.arange(1, 101, dtype=torch.float).view(-1, 100) - 1) / 100,
                torch.arange(1, 101, dtype=torch.float).view(-1, 100),
            ]
        )
        t1 = ExpTransform()
        t2 = CatTransform([AffineTransform(1, 100), identity_transform], dim=0)
        dim = 0
        x = torch.cat([x1, x2], dim=dim)
        t = CatTransform([t1, t2], dim=dim, lengths=[1, 2])
        actual_dom_check = t.domain.check(x)
        expected_dom_check = torch.cat(
            [t1.domain.check(x1), t2.domain.check(x2)], dim=dim
        )
        self.assertEqual(expected_dom_check, actual_dom_check)
        actual = t(x)
        expected = torch.cat([t1(x1), t2(x2)], dim=dim)
        self.assertEqual(expected, actual)
        y1 = torch.arange(1, 101, dtype=torch.float).view(-1, 100)
        y2 = torch.cat(
            [
                torch.arange(1, 101, dtype=torch.float).view(-1, 100),
                torch.arange(1, 101, dtype=torch.float).view(-1, 100),
            ]
        )
        y = torch.cat([y1, y2], dim=dim)
        actual_cod_check = t.codomain.check(y)
        expected_cod_check = torch.cat(
            [t1.codomain.check(y1), t2.codomain.check(y2)], dim=dim
        )
        self.assertEqual(actual_cod_check, expected_cod_check)
        actual_inv = t.inv(y)
        expected_inv = torch.cat([t1.inv(y1), t2.inv(y2)], dim=dim)
        self.assertEqual(expected_inv, actual_inv)
        actual_jac = t.log_abs_det_jacobian(x, y)
        expected_jac = torch.cat(
            [t1.log_abs_det_jacobian(x1, y1), t2.log_abs_det_jacobian(x2, y2)], dim=dim
        )
        self.assertEqual(actual_jac, expected_jac)

    def test_cat_event_dim(self):
        # [FUNCTION DOC] Checks that CatTransform of two event_dim=1 affine transforms sums the
        # component log_abs_det_jacobian values of the chunks (lengths [2, 2] along dim 1).
        t1 = AffineTransform(0, 2 * torch.ones(2), event_dim=1)
        t2 = AffineTransform(0, 2 * torch.ones(2), event_dim=1)
        dim = 1
        bs = 16
        x1 = torch.randn(bs, 2)
        x2 = torch.randn(bs, 2)
        x = torch.cat([x1, x2], dim=1)
        t = CatTransform([t1, t2], dim=dim, lengths=[2, 2])
        y1 = t1(x1)
        y2 = t2(x2)
        y = t(x)
        actual_jac = t.log_abs_det_jacobian(x, y)
        expected_jac = sum(
            [t1.log_abs_det_jacobian(x1, y1), t2.log_abs_det_jacobian(x2, y2)]
        )
        self.assertEqual(actual_jac, expected_jac)

    def test_stack_transform(self):
        # [FUNCTION DOC] Checks StackTransform over three components along dim 0: domain and
        # codomain checks, forward map, inverse and log_abs_det_jacobian match the stack of the
        # component results.
        # KNOWN ISSUE: the inverse is applied to x (not y) on both the actual and expected side,
        # so it checks consistency of the stacking only, not that it inverts the forward map.
        x1 = -1 * torch.arange(1, 101, dtype=torch.float)
        x2 = (torch.arange(1, 101, dtype=torch.float) - 1) / 100
        x3 = torch.arange(1, 101, dtype=torch.float)
        t1, t2, t3 = ExpTransform(), AffineTransform(1, 100), identity_transform
        dim = 0
        x = torch.stack([x1, x2, x3], dim=dim)
        t = StackTransform([t1, t2, t3], dim=dim)
        actual_dom_check = t.domain.check(x)
        expected_dom_check = torch.stack(
            [t1.domain.check(x1), t2.domain.check(x2), t3.domain.check(x3)], dim=dim
        )
        self.assertEqual(expected_dom_check, actual_dom_check)
        actual = t(x)
        expected = torch.stack([t1(x1), t2(x2), t3(x3)], dim=dim)
        self.assertEqual(expected, actual)
        y1 = torch.arange(1, 101, dtype=torch.float)
        y2 = torch.arange(1, 101, dtype=torch.float)
        y3 = torch.arange(1, 101, dtype=torch.float)
        y = torch.stack([y1, y2, y3], dim=dim)
        actual_cod_check = t.codomain.check(y)
        expected_cod_check = torch.stack(
            [t1.codomain.check(y1), t2.codomain.check(y2), t3.codomain.check(y3)],
            dim=dim,
        )
        self.assertEqual(actual_cod_check, expected_cod_check)
        actual_inv = t.inv(x)
        expected_inv = torch.stack([t1.inv(x1), t2.inv(x2), t3.inv(x3)], dim=dim)
        self.assertEqual(expected_inv, actual_inv)
        actual_jac = t.log_abs_det_jacobian(x, y)
        expected_jac = torch.stack(
            [
                t1.log_abs_det_jacobian(x1, y1),
                t2.log_abs_det_jacobian(x2, y2),
                t3.log_abs_det_jacobian(x3, y3),
            ],
            dim=dim,
        )
        self.assertEqual(actual_jac, expected_jac)


class TestValidation(DistributionsTestCase):
    # [CLASS DOC] Tests distribution argument validation (validate_args) on valid and
    # invalid inputs.
    def test_valid(self):
        # [FUNCTION DOC] Checks every example in _get_examples constructs without error when
        # validate_args is true.
        for Dist, params in _get_examples():
            for param in params:
                Dist(validate_args=True, **param)

    @set_default_dtype(torch.double)
    def test_invalid_log_probs_arg(self):
        # [FUNCTION DOC] For every example except TransformedDistribution: with validation on,
        # log_prob of values with wrong shape raises only ValueError, values outside the
        # support raise the "must be within the support" ValueError (the unvalidated
        # distribution may then raise RuntimeError), a valid sample is accepted, and an all-NaN
        # value raises ValueError matching the support message (AssertionError with the
        # distribution name if not).
        # Check that validation errors are indeed disabled,
        # but they might raise another error
        for Dist, params in _get_examples():
            if Dist == TransformedDistribution:
                # TransformedDistribution has a distribution instance
                # as the argument, so we cannot do much about that
                continue
            for i, param in enumerate(params):
                d_nonval = Dist(validate_args=False, **param)
                d_val = Dist(validate_args=True, **param)
                for v in torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]):
                    # samples with incorrect shape must throw ValueError only
                    try:
                        d_val.log_prob(v)
                    except ValueError:
                        pass
                    # get sample of correct shape
                    val = torch.full(d_val.batch_shape + d_val.event_shape, v)
                    # check samples with incorrect support
                    try:
                        d_val.log_prob(val)
                    except ValueError as e:
                        if e.args and "must be within the support" in e.args[0]:
                            try:
                                d_nonval.log_prob(val)
                            except RuntimeError:
                                pass

                # check correct samples are ok
                valid_value = d_val.sample()
                d_val.log_prob(valid_value)
                # check invalid values raise ValueError
                if valid_value.dtype == torch.long:
                    valid_value = valid_value.float()
                invalid_value = torch.full_like(valid_value, math.nan)
                try:
                    with self.assertRaisesRegex(
                        ValueError,
                        "Expected value argument .* to be within the support .*",
                    ):
                        d_val.log_prob(invalid_value)
                except AssertionError as e:
                    fail_string = "Support ValueError not raised for {} example {}/{}"
                    raise AssertionError(
                        fail_string.format(Dist.__name__, i + 1, len(params))
                    ) from e

    @set_default_dtype(torch.double)
    def test_invalid(self):
        # [FUNCTION DOC] For every invalid example in _get_bad_examples asserts that
        # constructing with validate_args true raises ValueError; otherwise raises
        # AssertionError naming the distribution and example.
        for Dist, params in _get_bad_examples():
            for i, param in enumerate(params):
                try:
                    with self.assertRaises(ValueError):
                        Dist(validate_args=True, **param)
                except AssertionError as e:
                    fail_string = "ValueError not raised for {} example {}/{}"
                    raise AssertionError(
                        fail_string.format(Dist.__name__, i + 1, len(params))
                    ) from e

    def test_warning_unimplemented_constraints(self):
        # [FUNCTION DOC] Checks a minimal Distribution subclass without declared constraints
        # emits UserWarning both at construction and when log_prob validates a sample.
        class Delta(Distribution):
            # [CLASS DOC] HELPER: Minimal Distribution that is a point mass at zero, with no
            # arg_constraints or support defined, used to trigger the
            # unimplemented-constraint warnings.
            def __init__(self, validate_args=True):
                # [FUNCTION DOC] HELPER: Forwards validate_args to the base constructor.
                super().__init__(validate_args=validate_args)

            def sample(self, sample_shape=torch.Size()):
                # [FUNCTION DOC] HELPER: Returns a scalar zero tensor expanded to the
                # sample shape.
                return torch.tensor(0.0).expand(sample_shape)

            def log_prob(self, value):
                # [FUNCTION DOC] HELPER: Validates the value if validation is enabled, then
                # overwrites the input tensor in place with 0 where it was 0 and minus infinity
                # elsewhere, and returns it (mutates the argument).
                if self._validate_args:
                    self._validate_sample(value)
                value[value != 0.0] = -float("inf")
                value[value == 0.0] = 0.0
                return value

        with self.assertWarns(UserWarning):
            d = Delta()
        sample = d.sample((2,))
        with self.assertWarns(UserWarning):
            d.log_prob(sample)


class TestJit(DistributionsTestCase):
    # [CLASS DOC] Tests that distribution methods can be traced with torch.jit.trace and give
    # the same results on new inputs. Helper methods generate examples and perturb parameters
    # within constraints.
    def _examples(self):
        # [FUNCTION DOC] HELPER: Generator over all examples whose parameters are all tensors,
        # yielding the distribution class, the parameter names, the tuple of parameter values
        # and a sample.
        for Dist, params in _get_examples():
            for param in params:
                keys = param.keys()
                values = tuple(param[key] for key in keys)
                if not all(isinstance(x, torch.Tensor) for x in values):
                    continue
                sample = Dist(**param).sample()
                yield Dist, keys, values, sample

    def _perturb_tensor(self, value, constraint):
        # [FUNCTION DOC] HELPER: Returns a changed copy of a parameter tensor that still
        # satisfies the constraint: integer-greater-than adds 1, less-than subtracts a uniform
        # value, greater-than(-or-equal) adds a uniform value, positive-(semi)definite adds the
        # identity; otherwise floating tensors are perturbed by normal noise in the
        # unconstrained space of the constraint transform, and long tensors have 0 and 1
        # swapped. Raises NotImplementedError for other dtypes.
        if isinstance(constraint, constraints._IntegerGreaterThan):
            return value + 1
        if isinstance(constraint, constraints._LessThan):
            return value - torch.rand(value.shape)
        if isinstance(
            constraint, (constraints._GreaterThan, constraints._GreaterThanEq)
        ):
            return value + torch.rand(value.shape)
        if isinstance(
            constraint,
            (constraints._PositiveDefinite, constraints._PositiveSemidefinite),
        ):
            return value + torch.eye(value.shape[-1])
        if value.dtype in [torch.float, torch.double]:
            transform = transform_to(constraint)
            delta = value.new(value.shape).normal_()
            return transform(transform.inv(value) + delta)
        if value.dtype == torch.long:
            result = value.clone()
            result[value == 0] = 1
            result[value == 1] = 0
            return result
        raise NotImplementedError

    def _perturb(self, Dist, keys, values, sample):
        # [FUNCTION DOC] HELPER: Under no_grad perturbs every parameter value according to the
        # distribution's arg_constraints (looked up on the class, or on an instance when the
        # constraints are parameter-dependent; missing keys default to real), then draws a new
        # sample from the perturbed distribution. Returns the new values list and sample.
        with torch.no_grad():
            if isinstance(Dist.arg_constraints, dict):
                values = [
                    self._perturb_tensor(
                        value, Dist.arg_constraints.get(key, constraints.real)
                    )
                    for key, value in zip(keys, values)
                ]
            else:
                # arg_constraints is parameter-dependent
                dist = Dist(**dict(zip(keys, values)))
                values = [
                    self._perturb_tensor(
                        value, dist.arg_constraints.get(key, constraints.real)
                    )
                    for key, value in zip(keys, values)
                ]
            param = dict(zip(keys, values))
            sample = Dist(**param).sample()
            return values, sample

    @set_default_dtype(torch.double)
    def test_sample(self):
        # [FUNCTION DOC] Traces a function that builds the distribution and calls sample with
        # torch.jit.trace for each all-tensor example (double default dtype), skipping Cauchy,
        # HalfCauchy and VonMises, and checks the traced result equals the eager result under
        # a forked RNG and that the trace contains a nondeterministic node (except for Beta
        # and Dirichlet).
        for Dist, keys, values, sample in self._examples():

            def f(*values):
                # [FUNCTION DOC] HELPER: Builds the distribution from the parameters and returns
                # one sample.
                param = dict(zip(keys, values))
                dist = Dist(**param)
                return dist.sample()

            traced_f = torch.jit.trace(f, values, check_trace=False)

            # FIXME Schema not found for node
            xfail = [
                Cauchy,  # aten::cauchy(Double(2,1), float, float, Generator)
                HalfCauchy,  # aten::cauchy(Double(2, 1), float, float, Generator)
                VonMises,  # Variance is not Euclidean
            ]
            if Dist in xfail:
                continue

            with torch.random.fork_rng():
                sample = f(*values)
            traced_sample = traced_f(*values)
            self.assertEqual(sample, traced_sample)

            # FIXME no nondeterministic nodes found in trace
            xfail = [Beta, Dirichlet]
            if Dist not in xfail:
                self.assertTrue(
                    any(n.isNondeterministic() for n in traced_f.graph.nodes())
                )

    def test_rsample(self):
        # [FUNCTION DOC] Same as test_sample but for rsample, only for distributions with
        # has_rsample and skipping Cauchy and HalfCauchy.
        for Dist, keys, values, sample in self._examples():
            if not Dist.has_rsample:
                continue

            def f(*values):
                # [FUNCTION DOC] HELPER: Builds the distribution from the parameters and returns
                # one reparameterized sample.
                param = dict(zip(keys, values))
                dist = Dist(**param)
                return dist.rsample()

            traced_f = torch.jit.trace(f, values, check_trace=False)

            # FIXME Schema not found for node
            xfail = [
                Cauchy,  # aten::cauchy(Double(2,1), float, float, Generator)
                HalfCauchy,  # aten::cauchy(Double(2, 1), float, float, Generator)
            ]
            if Dist in xfail:
                continue

            with torch.random.fork_rng():
                sample = f(*values)
            traced_sample = traced_f(*values)
            self.assertEqual(sample, traced_sample)

            # FIXME no nondeterministic nodes found in trace
            xfail = [Beta, Dirichlet]
            if Dist not in xfail:
                self.assertTrue(
                    any(n.isNondeterministic() for n in traced_f.graph.nodes())
                )

    @set_default_dtype(torch.double)
    def test_log_prob(self):
        # [FUNCTION DOC] Traces log_prob and checks the traced function equals the eager one on
        # perturbed parameters and sample (double dtype); LowRankMultivariateNormal and
        # MultivariateNormal are skipped because traced results are wrong.
        for Dist, keys, values, sample in self._examples():
            # FIXME traced functions produce incorrect results
            xfail = [LowRankMultivariateNormal, MultivariateNormal]
            if Dist in xfail:
                continue

            def f(sample, *values):
                # [FUNCTION DOC] HELPER: Builds the distribution from the parameters and returns
                # log_prob of the sample.
                param = dict(zip(keys, values))
                dist = Dist(**param)
                return dist.log_prob(sample)

            traced_f = torch.jit.trace(f, (sample,) + values)

            # check on different data
            values, sample = self._perturb(Dist, keys, values, sample)
            expected = f(sample, *values)
            actual = traced_f(sample, *values)
            self.assertEqual(
                expected,
                actual,
                msg=lambda msg: f"{msg}\n{Dist.__name__}\nExpected:\n{expected}\nActual:\n{actual}",
            )

    def test_enumerate_support(self):
        # [FUNCTION DOC] Traces enumerate_support and compares to eager on perturbed
        # parameters; Binomial is skipped and distributions raising NotImplementedError at
        # trace time are ignored.
        for Dist, keys, values, sample in self._examples():
            # FIXME traced functions produce incorrect results
            xfail = [Binomial]
            if Dist in xfail:
                continue

            def f(*values):
                # [FUNCTION DOC] HELPER: Builds the distribution and returns
                # its enumerate_support.
                param = dict(zip(keys, values))
                dist = Dist(**param)
                return dist.enumerate_support()

            try:
                traced_f = torch.jit.trace(f, values)
            except NotImplementedError:
                continue

            # check on different data
            values, sample = self._perturb(Dist, keys, values, sample)
            expected = f(*values)
            actual = traced_f(*values)
            self.assertEqual(
                expected,
                actual,
                msg=lambda msg: f"{msg}\n{Dist.__name__}\nExpected:\n{expected}\nActual:\n{actual}",
            )

    def test_mean(self):
        # [FUNCTION DOC] Traces the mean property and compares with eager on perturbed
        # parameters, treating infinite entries as zero on both sides; NotImplementedError at
        # trace time is ignored.
        for Dist, keys, values, sample in self._examples():

            def f(*values):
                # [FUNCTION DOC] HELPER: Builds the distribution and returns its mean.
                param = dict(zip(keys, values))
                dist = Dist(**param)
                return dist.mean

            try:
                traced_f = torch.jit.trace(f, values)
            except NotImplementedError:
                continue

            # check on different data
            values, sample = self._perturb(Dist, keys, values, sample)
            expected = f(*values)
            actual = traced_f(*values)
            expected[expected == float("inf")] = 0.0
            actual[actual == float("inf")] = 0.0
            self.assertEqual(
                expected,
                actual,
                msg=lambda msg: f"{msg}\n{Dist.__name__}\nExpected:\n{expected}\nActual:\n{actual}",
            )

    def test_variance(self):
        # [FUNCTION DOC] Same as test_mean for variance (cloning results before editing
        # infinities), skipping Cauchy and HalfCauchy whose variance is infinite.
        for Dist, keys, values, sample in self._examples():
            if Dist in [Cauchy, HalfCauchy]:
                continue  # infinite variance

            def f(*values):
                # [FUNCTION DOC] HELPER: Builds the distribution and returns its variance.
                param = dict(zip(keys, values))
                dist = Dist(**param)
                return dist.variance

            try:
                traced_f = torch.jit.trace(f, values)
            except NotImplementedError:
                continue

            # check on different data
            values, sample = self._perturb(Dist, keys, values, sample)
            expected = f(*values).clone()
            actual = traced_f(*values).clone()
            expected[expected == float("inf")] = 0.0
            actual[actual == float("inf")] = 0.0
            self.assertEqual(
                expected,
                actual,
                msg=lambda msg: f"{msg}\n{Dist.__name__}\nExpected:\n{expected}\nActual:\n{actual}",
            )

    @set_default_dtype(torch.double)
    def test_entropy(self):
        # [FUNCTION DOC] Traces entropy and compares with eager on perturbed parameters (double
        # dtype); LowRankMultivariateNormal and MultivariateNormal are skipped because traced
        # results are wrong.
        for Dist, keys, values, sample in self._examples():
            # FIXME traced functions produce incorrect results
            xfail = [LowRankMultivariateNormal, MultivariateNormal]
            if Dist in xfail:
                continue

            def f(*values):
                # [FUNCTION DOC] HELPER: Builds the distribution and returns its entropy.
                param = dict(zip(keys, values))
                dist = Dist(**param)
                return dist.entropy()

            try:
                traced_f = torch.jit.trace(f, values)
            except NotImplementedError:
                continue

            # check on different data
            values, sample = self._perturb(Dist, keys, values, sample)
            expected = f(*values)
            actual = traced_f(*values)
            self.assertEqual(
                expected,
                actual,
                msg=lambda msg: f"{msg}\n{Dist.__name__}\nExpected:\n{expected}\nActual:\n{actual}",
            )

    @set_default_dtype(torch.double)
    def test_cdf(self):
        # [FUNCTION DOC] Traces a function that returns icdf of cdf of a sample and compares
        # with eager on perturbed parameters and sample (double dtype); distributions raising
        # NotImplementedError are ignored.
        for Dist, keys, values, sample in self._examples():

            def f(sample, *values):
                # [FUNCTION DOC] HELPER: Builds the distribution and returns the icdf applied to
                # the cdf of the sample.
                param = dict(zip(keys, values))
                dist = Dist(**param)
                cdf = dist.cdf(sample)
                return dist.icdf(cdf)

            try:
                traced_f = torch.jit.trace(f, (sample,) + values)
            except NotImplementedError:
                continue

            # check on different data
            values, sample = self._perturb(Dist, keys, values, sample)
            expected = f(sample, *values)
            actual = traced_f(sample, *values)
            self.assertEqual(
                expected,
                actual,
                msg=lambda msg: f"{msg}\n{Dist.__name__}\nExpected:\n{expected}\nActual:\n{actual}",
            )


instantiate_device_type_tests(
    TestDistributions,
    globals(),
    allow_mps=True,
    except_for=(
        "cuda",
        "xpu",
    ),
)

if __name__ == "__main__" and torch._C.has_lapack:
    TestCase._default_dtype_check_enabled = True
    run_tests()
