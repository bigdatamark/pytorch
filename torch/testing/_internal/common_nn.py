# mypy: ignore-errors
# [MODULE DOC] Shared infrastructure for torch.nn tests. Contains: (1) fixture factories, one
# per loss configuration with no reduction, each returning a test-spec mapping with fullname,
# constructor (via wrap_functional), C++ parity call and variable map, reference function and
# flags; (2) module_tests and get_new_module_tests, which build the long list of
# module/functional specs, plus no-batch-dimension and padding-mode variants; (3) reference
# implementations of many losses collected in loss_reference_fns; (4) criterion_tests, filled at
# import time with no-batch-dimension criterion specs; (5) NNTestCase, TestBase, ModuleTest,
# NewModuleTest, CriterionTest and InputVariableMixin which run forward, Jacobian, gradcheck,
# pickle, non-contiguous, dtype and CPU/GPU comparisons; (6) small helpers for bfloat16 and
# empty-input checks and a basic net factory. Import-time side effects: defines module_tests,
# loss_reference_fns and criterion_tests; does not seed the RNG (get_new_module_tests does).
# KNOWN ISSUE: the constructor lambdas in the no-batch criterion loops read the loop variable
# reduction when called later rather than when created, so every generated spec ends up using
# the last reduction (sum) instead of its own; the loss_reference_fns annotation also uses the
# string literal str as the key type.

from abc import abstractmethod
import tempfile
import unittest

from copy import deepcopy
from functools import reduce, partial
from itertools import product
from operator import mul


import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import _reduction as _Reduction
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import TestCase, to_gpu, freeze_rng_state, is_iterable, \
    gradcheck, gradgradcheck, set_default_dtype, skipIfTorchDynamo, TEST_WITH_ROCM
from torch.testing._internal.common_cuda import TEST_CUDA, SM90OrLater
from torch.autograd.gradcheck import _get_numerical_jacobian, _iter_tensors
from torch.autograd import Variable
from torch.types import _TensorOrTensors
import torch.backends.cudnn

from typing import Any
from collections.abc import Callable
from collections.abc import Sequence

TemporaryFile = tempfile.TemporaryFile
PRECISION = 1e-5


def get_reduction(m):
    # [FUNCTION DOC] Returns the reduction string of a loss module: its reduction attribute if
    # set, otherwise the string derived from the legacy size_average setting. Raises an error if
    # neither yields a value.
    result = getattr(m, 'reduction', None)
    if result is None:
        result = _Reduction.legacy_get_string(getattr(m, 'sizeAverage', None), True, emit_warning=False)
    if result is None:
        raise AssertionError("Expected result to not be None")
    return result


def get_weight(m):
    # [FUNCTION DOC] Returns the weight tensor of a module: the weight attribute if present and
    # not None, otherwise the weights attribute (or None if absent).
    result = getattr(m, 'weight', None)
    if result is not None:
        return result
    return getattr(m, 'weights', None)

# NOTE [How to check NN module / functional API parity between Python and C++ frontends]
#
# The way to check API parity is to add parity tests for the NN module / functional of interest.
# Here are the detailed steps:
#
# For NN module:
# 1. Make sure you already have a test dict with the module configuration you want to test.
# 2. Add `cpp_constructor_args` entry to the test dict, with its value exactly matching
#    the Python module constructor arguments. For example, if in the test dict we pass
#    `(10, 8)` to `torch.nn.Linear` constructor, then we should pass `torch::nn::LinearOptions(10, 8)`
#    as the corresponding C++ constructor argument to `torch::nn::Linear`.
# 3. If in the process of performing the above step you referenced any variables
#    in the `cpp_constructor_args` entry, you must add `cpp_var_map` entry
#    to the test dict to make sure that those variables are populated with the right Python values.
#    For example, if the Python constructor call is
#    `torch.nn.FractionalMaxPool2d(2, output_ratio=0.5, _random_samples=random_samples)`,
#    the corresponding C++ constructor argument is
#    `torch::nn::FractionalMaxPool2dOptions(2).output_ratio(0.5)._random_samples(random_samples)`,
#    and the `cpp_var_map` entry must be
#    `{'random_samples': random_samples}` in order to populate the C++ variable `random_samples`
#    used in the C++ constructor argument with the Python tensor value `random_samples`.
#
# For NN functional:
# 1. Make sure you already have a test dict with the functional configuration you want to test.
# 2. If the test dict's `constructor` entry looks like `wrap_functional(F.some_functional_name, ...)`,
#    then you must add `cpp_options_args` entry to the test dict, with its value exactly matching the Python
#    functional optional arguments. For example, if the test dict's `constructor` entry is
#    `wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest')`,
#    then the `cpp_options_args` entry should be
#    "F::InterpolateFuncOptions().size(std::vector<int64_t>({12})).scale_factor(std::nullopt).mode(torch::kNearest)".
# 3. Otherwise, if the test dict's `constructor` entry looks like
#    `wrap_functional(lambda i: F.some_functional_name(...))`,
#    then you must add `cpp_function_call` entry to the test dict, with its value exactly matching the Python
#    functional function call. For example, if the test dict's `constructor` entry is
#    `wrap_functional(lambda i: F.poisson_nll_loss(i, t.type_as(i), reduction='none'))`,
#    then the `cpp_function_call` entry should be
#    "F::poisson_nll_loss(i, t.to(i.options()), F::PoissonNLLLossFuncOptions().reduction(torch::kNone))".
# 4. If in the process of performing the above two steps you referenced any variables
#    in the `cpp_options_args` or `cpp_function_call` entry, you must
#    add `cpp_var_map` entry to the test dict to make sure that those variables
#    are populated with the right Python values. For example, if the test dict's `constructor` entry is
#    `wrap_functional(lambda i: F.poisson_nll_loss(i, t.type_as(i), reduction='none'))`,
#    then the `cpp_function_call` entry should be
#    "F::poisson_nll_loss(i, t.to(i.options()), F::PoissonNLLLossFuncOptions().reduction(torch::kNone))".
#    Notice that there are two variables `i` and `t` that need to have their values provided,
#    and the way to do so is to add a `cpp_var_map` entry: `cpp_var_map={'i': '_get_input()', 't': t}`.
#    (Note that for `i`, since we want it to take the Python input value, we pass '_get_input()' string as value
#    and the C++ parity test mechanism will populate `i` with the Python input value correctly.)
#
# There are also a few optional flags in the test dict to control the C++ parity test behavior:
#
# - `test_cpp_api_parity`: if `False`, skips the C++ parity test for this test dict. Default: True.
# - `has_parity`: if `False`, expects this test dict to fail the C++ parity test. Default: True.


module_tests = [
    dict(
        module_name='Linear',
        constructor_args=(10, 8),
        cpp_constructor_args='torch::nn::LinearOptions(10, 8)',
        input_size=(4, 10),
        reference_fn=lambda i, p, _: torch.mm(i, p[0].t()) + p[1].view(1, -1).expand(4, 8),
        with_tf32=True,
        # See no_bias variant below for rationale; same K=10 shape, seed-
        # dependent realization near the AMD XF32 envelope (issue #155216).
        tf32_precision=0.01 if TEST_WITH_ROCM else 0.005,
        default_dtype=torch.double,
    ),
    dict(
        module_name='Linear',
        constructor_args=(10, 8, False),
        cpp_constructor_args='torch::nn::LinearOptions(10, 8).bias(false)',
        input_size=(4, 10),
        desc='no_bias',
        reference_fn=lambda i, p, _: torch.mm(i, p[0].t()),
        with_tf32=True,
        # AMD XF32 at K=10 sits right at the 0.005 envelope (ideal 5.2e-3;
        # ideal NV-TF32 2.3e-3). ROCm tolerance relaxed to cover the
        # seed-dependent realization; see https://github.com/jeffdaily/tf32_analysis.
        tf32_precision=0.01 if TEST_WITH_ROCM else 0.005,
        default_dtype=torch.double,
    ),
    dict(
        module_name='RReLU',
        input_size=(1, 2, 2),
        test_cuda=False,
        default_dtype=torch.double,
    ),
    dict(
        module_name='RReLU',
        constructor_args=(0.1, 0.9),
        cpp_constructor_args='torch::nn::RReLUOptions().lower(0.1).upper(0.9)',
        input_size=(4, 4, 5),
        desc='with_up_down',
        test_cuda=False,
        default_dtype=torch.double,
    ),
    dict(
        module_name='Flatten',
        input_size=(2, 3, 4, 5),
        reference_fn=lambda i, *_: torch.flatten(i, 1),
        default_dtype=torch.double,
    ),
    # TODO: reference function
    dict(
        module_name='CrossMapLRN2d',
        constructor_args=(5, 5e-3, 1e-3, 2),
        cpp_constructor_args='torch::nn::CrossMapLRN2dOptions(5).alpha(5e-3).beta(1e-3).k(2)',
        input_size=(2, 3, 6, 6),
        check_gradgrad=False,
        # TODO(#50743): Figure out the error. "RuntimeError: Unrecognized tensor type ID: Batched"
        check_batched_grad=False,
        default_dtype=torch.double,
    ),
]


# Generates rand tensor with non-equal values. This ensures that duplicate
# values won't be causing test failure for modules like MaxPooling.
# size should be small, otherwise randperm fails / long overflows.
def _rand_tensor_non_equal(*size):
    # [FUNCTION DOC] HELPER: Returns a double tensor of the given shape holding a random
    # permutation of 0..N-1, so all values are distinct (avoids ties in pooling tests). Size
    # must be small since a full permutation is generated.
    total = reduce(mul, size, 1)
    return torch.randperm(total).view(*size).double()


def wrap_functional(fn, **kwargs):
    # [FUNCTION DOC] Wraps a functional callable into a module class: returns a new nn.Module
    # subclass whose forward applies the callable to the positional arguments plus the fixed
    # keyword arguments given here. Returns the class, not an instance.
    class FunctionalModule(nn.Module):
        # [CLASS DOC] HELPER: Throwaway module type created per call of wrap_functional. Holds
        # no state; the wrapped callable and its keyword arguments are captured by closure.
        def forward(self, *args):
            # [FUNCTION DOC] Calls the captured functional with the given positional arguments
            # and the captured keyword arguments and returns its result.
            return fn(*args, **kwargs)
    return FunctionalModule


def poissonnllloss_no_reduce_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for the Poisson
    # negative log likelihood loss with no reduction. A fixed random normal target is created
    # once per call; inputs are uniform random 10 by 10; the reference computes the
    # exponential of the input minus target times input. Pickling is skipped and the default
    # dtype is double.
    t = torch.randn(10, 10)
    return dict(
        fullname='PoissonNLLLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.poisson_nll_loss(i, t.type_as(i), reduction='none')),
        cpp_function_call='F::poisson_nll_loss('
                          'i, t.to(i.options()), F::PoissonNLLLossFuncOptions().reduction(torch::kNone))',
        input_fn=lambda: torch.rand(10, 10),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_: i.exp() - t.mul(i),
        pickle=False,
        default_dtype=torch.double)


def bceloss_no_reduce_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for binary cross
    # entropy with no reduction. A fixed random binary target of shape 15 by 10 is created once;
    # inputs are uniform values clamped away from 0 and 1; the reference applies the elementwise
    # log formula. Precision is relaxed to 7e-4; pickling is skipped; double dtype.
    t = Variable(torch.randn(15, 10).gt(0).to(torch.double))
    return dict(
        fullname='BCELoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy(i, t.type_as(i), reduction='none')),
        cpp_function_call='F::binary_cross_entropy('
                          'i, t.to(i.options()), F::BinaryCrossEntropyFuncOptions().reduction(torch::kNone))',
        input_fn=lambda: torch.rand(15, 10).clamp_(2.8e-2, 1 - 2.8e-2),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_: -(t * i.log() + (1 - t) * (1 - i).log()),
        pickle=False,
        precision=7e-4,
        default_dtype=torch.double)


def bceloss_no_reduce_scalar_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for binary cross
    # entropy with no reduction on 0-dim (scalar) input and target.
    t = torch.randn(()).gt(0).to(torch.double)
    return dict(
        fullname='BCELoss_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy(i, t.type_as(i), reduction='none')),
        cpp_function_call='F::binary_cross_entropy('
                          'i, t.to(i.options()), F::BinaryCrossEntropyFuncOptions().reduction(torch::kNone))',
        input_fn=lambda: torch.rand(()).clamp_(2.8e-2, 1 - 2.8e-2),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_: -(t * i.log() + (1 - t) * (1 - i).log()),
        pickle=False,
        default_dtype=torch.double)


def bceloss_weights_no_reduce_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for weighted binary
    # cross entropy with no reduction. A random per-class weight of length 10 is applied; the
    # target is a fixed random binary 15 by 10 tensor; precision 3e-4; pickling skipped.
    t = Variable(torch.randn(15, 10, dtype=torch.double).gt(0).to(torch.double))
    weights = torch.rand(10, dtype=torch.double)
    return dict(
        fullname='BCELoss_weights_no_reduce',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy(i, t.type_as(i),
                                             weight=weights.type_as(i), reduction='none')),
        cpp_function_call='F::binary_cross_entropy('
                          'i, t.to(i.options()), '
                          'F::BinaryCrossEntropyFuncOptions().weight(weights.to(i.options())).reduction(torch::kNone))',
        input_fn=lambda: torch.rand(15, 10).clamp_(2.8e-2, 1 - 2.8e-2),
        cpp_var_map={'i': '_get_input()', 't': t, 'weights': weights},
        reference_fn=lambda i, p, m: -(t * i.log() + (1 - t) * (1 - i).log()) * weights,
        pickle=False,
        precision=3e-4,
        default_dtype=torch.double,
    )


def bceloss_weights_no_reduce_scalar_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for weighted binary
    # cross entropy with no reduction on 0-dim input, target and weight.
    t = torch.randn(()).gt(0).to(torch.double)
    weights = torch.rand((), dtype=torch.double)
    return dict(
        fullname='BCELoss_weights_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy(i, t.type_as(i),
                                             weight=weights.type_as(i), reduction='none')),
        cpp_function_call='''F::binary_cross_entropy(
            i, t.to(i.options()),
            F::BinaryCrossEntropyFuncOptions().weight(weights.to(i.options())).reduction(torch::kNone))''',
        cpp_var_map={'i': '_get_input()', 't': t, 'weights': weights},
        input_fn=lambda: torch.rand(()).clamp_(2.8e-2, 1 - 2.8e-2),
        reference_fn=lambda i, *_: -(t * i.log() + (1 - t) * (1 - i).log()) * weights,
        pickle=False,
        default_dtype=torch.double,
    )


def bce_with_logistic_legacy_enum_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for binary cross
    # entropy with logits using the deprecated boolean reduce=False argument instead of a
    # reduction string. The reference applies sigmoid then the log formula; the double-backward
    # check is disabled.
    t = Variable(torch.randn(15, 10).gt(0).to(torch.double))
    sigmoid = nn.Sigmoid()
    return dict(
        fullname='BCEWithLogitsLoss_legacy_enum',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy_with_logits(i, t.type_as(i), reduce=False)),
        cpp_function_call='''F::binary_cross_entropy_with_logits(
            i, t.to(i.options()), F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(15, 10).clamp_(2.8e-2, 1 - 2.8e-2),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_: -(t * sigmoid(i).log() + (1 - t) * (1 - sigmoid(i)).log()),
        check_gradgrad=False,
        pickle=False,
        default_dtype=torch.double,
    )


def bce_with_logistic_no_reduce_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for binary cross
    # entropy with logits and no reduction on 15 by 10 input with a fixed binary target.
    # Double-backward check disabled.
    t = Variable(torch.randn(15, 10).gt(0).to(torch.double))
    sigmoid = nn.Sigmoid()
    return dict(
        fullname='BCEWithLogitsLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy_with_logits(i, t.type_as(i), reduction='none')),
        cpp_function_call='''F::binary_cross_entropy_with_logits(
            i, t.to(i.options()), F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(15, 10).clamp_(2.8e-2, 1 - 2.8e-2),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_: -(t * sigmoid(i).log() + (1 - t) * (1 - sigmoid(i)).log()),
        check_gradgrad=False,
        pickle=False,
        default_dtype=torch.double,
    )


def bce_with_logistic_no_reduce_scalar_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for binary
    # cross entropy with logits and no reduction on 0-dim input and target.
    # Double-backward check disabled.
    t = torch.randn(()).gt(0).to(torch.double)
    sigmoid = nn.Sigmoid()
    return dict(
        fullname='BCEWithLogitsLoss_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy_with_logits(i, t.type_as(i), reduction='none')),
        cpp_function_call='''F::binary_cross_entropy_with_logits(
            i, t.to(i.options()), F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(()).clamp_(2.8e-2, 1 - 2.8e-2),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_: -(t * sigmoid(i).log() + (1 - t) * (1 - sigmoid(i)).log()),
        check_gradgrad=False,
        pickle=False,
        default_dtype=torch.double,
    )


def kldivloss_with_target_no_reduce_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for KL divergence
    # loss with no reduction. The target is a fixed uniform 10 by 10 double tensor and the input
    # is the log of uniform values; the reference is loss_reference_fns for KLDivLoss;
    # forward-mode AD is declared supported.
    # KNOWN ISSUE: body is identical to kldivloss_no_reduce_test apart from the fullname, so it
    # is a duplicate test configuration.
    t = torch.rand(10, 10, dtype=torch.double)
    return dict(
        fullname='KLDivLoss_with_target_no_reduce',
        constructor=wrap_functional(
            lambda i: F.kl_div(i, t.type_as(i), reduction='none')),
        cpp_function_call='F::kl_div(i, t.to(i.options()), F::KLDivFuncOptions().reduction(torch::kNone))',
        input_fn=lambda: torch.rand(10, 10).log(),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['KLDivLoss'](i, t.type_as(i), reduction='none'),
        supports_forward_ad=True,
        pickle=False,
        default_dtype=torch.double)


def kldivloss_no_reduce_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for KL divergence
    # loss with no reduction (same configuration as kldivloss_with_target_no_reduce_test under a
    # different fullname).
    t = torch.rand(10, 10, dtype=torch.double)
    return dict(
        fullname='KLDivLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.kl_div(i, t.type_as(i), reduction='none')),
        cpp_function_call='F::kl_div(i, t.to(i.options()), F::KLDivFuncOptions().reduction(torch::kNone))',
        input_fn=lambda: torch.rand(10, 10).log(),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['KLDivLoss'](i, t.type_as(i), reduction='none'),
        supports_forward_ad=True,
        pickle=False,
        default_dtype=torch.double,
    )


def kldivloss_no_reduce_scalar_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for KL divergence
    # loss with no reduction on 0-dim input and target.
    t = torch.rand((), dtype=torch.double)
    return dict(
        fullname='KLDivLoss_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.kl_div(i, t.type_as(i), reduction='none')),
        cpp_function_call='F::kl_div(i, t.to(i.options()), F::KLDivFuncOptions().reduction(torch::kNone))',
        input_fn=lambda: torch.rand(()).log(),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['KLDivLoss'](i, t.type_as(i), reduction='none'),
        supports_forward_ad=True,
        pickle=False,
        default_dtype=torch.double)


def kldivloss_with_log_target_no_reduce_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for KL divergence
    # loss with no reduction and log_target enabled. The target is the log of uniform values and
    # the reference is the log-target variant of the KL reference.
    # KNOWN ISSUE: body is identical to kldivloss_no_reduce_log_target_test apart from the
    # fullname (duplicate test).
    t = torch.rand(10, 10, dtype=torch.double).log()
    return dict(
        fullname='KLDivLoss_with_log_target_no_reduce',
        constructor=wrap_functional(
            lambda i: F.kl_div(i, t.type_as(i), reduction='none', log_target=True)),
        cpp_function_call='F::kl_div(i, t.to(i.options()), F::KLDivFuncOptions().reduction(torch::kNone).log_target(true))',
        input_fn=lambda: torch.rand(10, 10).log(),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['KLDivLoss_log_target'](i, t.type_as(i), reduction='none'),
        supports_forward_ad=True,
        pickle=False,
        default_dtype=torch.double)


def kldivloss_no_reduce_log_target_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for KL divergence
    # loss with no reduction and log_target enabled (duplicate of
    # kldivloss_with_log_target_no_reduce_test under another fullname).
    t = torch.rand(10, 10, dtype=torch.double).log()
    return dict(
        fullname='KLDivLoss_no_reduce_log_target',
        constructor=wrap_functional(
            lambda i: F.kl_div(i, t.type_as(i), reduction='none', log_target=True)),
        cpp_function_call='F::kl_div(i, t.to(i.options()), F::KLDivFuncOptions().reduction(torch::kNone).log_target(true))',
        input_fn=lambda: torch.rand(10, 10).log(),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['KLDivLoss_log_target'](i, t.type_as(i), reduction='none'),
        supports_forward_ad=True,
        pickle=False,
        default_dtype=torch.double,
    )


def kldivloss_no_reduce_scalar_log_target_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for KL divergence
    # loss with no reduction and log_target enabled on 0-dim input and target.
    t = torch.rand((), dtype=torch.double).log()
    return dict(
        fullname='KLDivLoss_no_reduce_scalar_log_target',
        constructor=wrap_functional(
            lambda i: F.kl_div(i, t.type_as(i), reduction='none', log_target=True)),
        cpp_function_call='F::kl_div(i, t.to(i.options()), F::KLDivFuncOptions().reduction(torch::kNone).log_target(true))',
        input_fn=lambda: torch.rand(()).log(),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['KLDivLoss_log_target'](i, t.type_as(i), reduction='none'),
        supports_forward_ad=True,
        pickle=False,
        default_dtype=torch.double)


def l1loss_no_reduce_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for L1 loss
    # with no reduction on 2 by 3 by 4 tensors. Reference is the absolute difference;
    # forward AD supported.
    t = torch.randn(2, 3, 4, dtype=torch.double)
    return dict(
        fullname='L1Loss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.l1_loss(i, t.type_as(i), reduction='none')),
        cpp_function_call='F::l1_loss(i, t.to(i.options()), F::L1LossFuncOptions().reduction(torch::kNone))',
        input_fn=lambda: torch.randn(2, 3, 4),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_: (i - t.type_as(i)).abs(),
        supports_forward_ad=True,
        pickle=False,
        default_dtype=torch.double)


def l1loss_no_reduce_complex_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for L1 loss with
    # no reduction on complex double 2 by 3 by 4 tensors. Reference is the absolute value of
    # the difference.
    # KNOWN ISSUE: unlike its siblings it does not set default_dtype to double.
    t = torch.randn(2, 3, 4, dtype=torch.cdouble)
    return dict(
        fullname='L1Loss_no_reduce_complex',
        constructor=wrap_functional(
            lambda i: F.l1_loss(i, t.type_as(i), reduction='none')),
        cpp_function_call='F::l1_loss(i, t.to(i.options()), F::L1LossFuncOptions().reduction(torch::kNone))',
        input_fn=lambda: torch.randn(2, 3, 4, dtype=torch.cdouble),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_: (i - t.type_as(i)).abs(),
        supports_forward_ad=True,
        pickle=False)


def l1loss_no_reduce_scalar_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for L1 loss with no
    # reduction on 0-dim input and target.
    t = torch.randn((), dtype=torch.double)
    return dict(
        fullname='L1Loss_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.l1_loss(i, t.type_as(i), reduction='none')),
        cpp_function_call='F::l1_loss(i, t.to(i.options()), F::L1LossFuncOptions().reduction(torch::kNone))',
        input_fn=lambda: torch.randn(()),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_: (i - t.type_as(i)).abs(),
        supports_forward_ad=True,
        pickle=False,
        default_dtype=torch.double)


def mseloss_no_reduce_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for mean squared
    # error loss with no reduction. Input shape is 2 by 3 by 4 by 5 with a fixed random double
    # target; reference is the squared difference.
    input_size = (2, 3, 4, 5)
    target = torch.randn(*input_size, dtype=torch.double)
    return dict(
        fullname='MSELoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.mse_loss(i, target.type_as(i), reduction='none')),
        cpp_function_call='F::mse_loss(i, target.to(i.options()), F::MSELossFuncOptions().reduction(torch::kNone))',
        input_size=input_size,
        cpp_var_map={'i': '_get_input()', 'target': target},
        reference_fn=lambda i, *_: (i - target).pow(2),
        supports_forward_ad=True,
        pickle=False,
        default_dtype=torch.double)


def mseloss_no_reduce_scalar_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for mean squared
    # error loss with no reduction on 0-dim input and target.
    input_size = ()
    target = torch.randn(input_size, dtype=torch.double)
    return dict(
        fullname='MSELoss_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.mse_loss(i, target.type_as(i), reduction='none')),
        cpp_function_call='F::mse_loss(i, target.to(i.options()), F::MSELossFuncOptions().reduction(torch::kNone))',
        input_size=input_size,
        cpp_var_map={'i': '_get_input()', 'target': target},
        reference_fn=lambda i, *_: (i - target).pow(2),
        supports_forward_ad=True,
        pickle=False,
        default_dtype=torch.double)


def nllloss_no_reduce_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for negative
    # log likelihood loss with no reduction. Targets are 15 random class indices in 0..9
    # created once; inputs are logs of uniform 15 by 10 values; the reference is the
    # NLLLoss entry of loss_reference_fns.
    t = Variable(torch.empty(15).uniform_().mul(10).floor().long())
    kwargs = {'reduction': 'none'}
    return dict(
        fullname='NLLLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), reduction=kwargs['reduction'])),
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong), F::NLLLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(15, 10).log(),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLoss'](i, t.type_as(i).long(), **kwargs),
        pickle=False,
        default_dtype=torch.double)


def nllloss_no_reduce_ignore_index_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for negative log
    # likelihood loss with no reduction and ignore_index 2. Same data as nllloss_no_reduce_test;
    # targets equal to the ignored index must contribute zero.
    t = Variable(torch.empty(15).uniform_().mul(10).floor().long())
    kwargs: dict[str, int | str] = {'ignore_index': 2, 'reduction': 'none'}
    return dict(
        fullname='NLLLoss_no_reduce_ignore_index',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), ignore_index=int(kwargs['ignore_index']),
                                 reduction=str(kwargs['reduction']))),
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong), F::NLLLossFuncOptions().ignore_index(2).reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(15, 10).log(),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLoss'](i, t.type_as(i).long(), **kwargs),
        pickle=False,
        default_dtype=torch.double)


def nllloss_no_reduce_weights_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for negative log
    # likelihood loss with no reduction and per-class weights. A random weight vector of length
    # 10 is cast to the input dtype at call time.
    t = Variable(torch.empty(15).uniform_().mul(10).floor().long())
    weight = torch.rand(10)

    def kwargs(i):
        # [FUNCTION DOC] HELPER: Returns the options mapping (weight cast to the dtype of the
        # given tensor, reduction none) used by the constructor and the reference.
        return {'weight': weight.type_as(i), 'reduction': 'none'}

    return dict(
        fullname='NLLLoss_no_reduce_weights',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs(i))),
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong),
            F::NLLLossFuncOptions().weight(weight.to(i.options())).reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(15, 10).add(1e-2).log(),
        cpp_var_map={'i': '_get_input()', 't': t, 'weight': weight},
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLoss'](i, t.type_as(i).long(), **kwargs(i)),
        pickle=False,
        default_dtype=torch.double)


def nllloss_no_reduce_weights_ignore_index_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for negative log
    # likelihood loss with no reduction, class weights and ignore_index 2.
    # KNOWN ISSUE: the constructor passes the data attribute of the input to the options builder
    # while the reference passes the input itself; harmless since only the dtype is used.
    t = Variable(torch.empty(15).uniform_().mul(10).floor().long())
    weight = torch.rand(10)

    def kwargs(i):
        # [FUNCTION DOC] HELPER: Returns the options mapping (weight cast to the dtype of the
        # given tensor, reduction none, ignore_index 2).
        return {'weight': weight.type_as(i), 'reduction': 'none',
                'ignore_index': 2}

    return dict(
        fullname='NLLLoss_no_reduce_weights_ignore_index',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs(i.data))),
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong),
            F::NLLLossFuncOptions().weight(weight.to(i.options())).reduction(torch::kNone).ignore_index(2))''',
        input_fn=lambda: torch.rand(15, 10).add(1e-2).log(),
        cpp_var_map={'i': '_get_input()', 't': t, 'weight': weight},
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLoss'](i, t.type_as(i).long(), **kwargs(i)),
        pickle=False,
        default_dtype=torch.double)


def nllloss_no_reduce_weights_ignore_index_neg_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for negative log
    # likelihood loss with no reduction, class weights and negative ignore_index -1. Uses a
    # fixed double input tensor rather than an input generator.
    t = Variable(torch.empty(15).uniform_().mul(10).floor().long())
    weight = torch.rand(10)

    def kwargs(i):
        # [FUNCTION DOC] HELPER: Returns the options mapping (weight cast to the dtype of the
        # given tensor, reduction none, ignore_index -1).
        return {'weight': weight.type_as(i), 'reduction': 'none',
                'ignore_index': -1}

    return dict(
        fullname='NLLLoss_no_reduce_weights_ignore_index_neg',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs(i))),
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong),
            F::NLLLossFuncOptions().weight(weight.to(i.options())).reduction(torch::kNone).ignore_index(-1))''',
        input=torch.rand(15, 10, dtype=torch.double).add(1e-2).log(),
        cpp_var_map={'i': '_get_input()', 't': t, 'weight': weight},
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLoss'](i, t.type_as(i).long(), **kwargs(i)),
        pickle=False,
        default_dtype=torch.double)


def nllloss2d_no_reduce_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for 2-d negative
    # log likelihood loss with no reduction. Input is the log of uniform values of shape 2 by
    # 3 by 5 by 5 and targets have shape 2 by 5 by 5 with classes 0..2; reference is the N-d
    # NLL reference.
    t = Variable(torch.rand(2, 5, 5).mul(3).floor().long())
    kwargs = {'reduction': 'none'}
    return dict(
        fullname='NLLLoss2d_no_reduce',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), reduction=kwargs['reduction'])),
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong), F::NLLLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(2, 3, 5, 5).log(),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs),
        pickle=False,
        default_dtype=torch.double)


def nllloss2d_no_reduce_ignore_index_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for 2-d negative log
    # likelihood loss with no reduction and ignore_index 1.
    t = Variable(torch.rand(2, 5, 5).mul(3).floor().long())
    kwargs: dict[str, int | str] = {'ignore_index': 1, 'reduction': 'none'}
    return dict(
        fullname='NLLLoss2d_no_reduce_ignore_index',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), ignore_index=int(kwargs['ignore_index']),
                                 reduction=str(kwargs['reduction']))),
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong), F::NLLLossFuncOptions().ignore_index(1).reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(2, 3, 5, 5).log(),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs),
        pickle=False,
        default_dtype=torch.double)


def nllloss2d_no_reduce_weights_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for 2-d negative log
    # likelihood loss with no reduction and per-class weights of length 3.
    t = Variable(torch.rand(2, 5, 5).mul(3).floor().long())
    weight = torch.rand(3)

    def kwargs(i):
        # [FUNCTION DOC] HELPER: Returns the options mapping (weight cast to the dtype of the
        # given tensor, reduction none).
        return {'weight': weight.type_as(i), 'reduction': 'none'}

    return dict(
        fullname='NLLLoss2d_no_reduce_weights',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs(i))),
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong),
            F::NLLLossFuncOptions().weight(weight.to(i.options())).reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(2, 3, 5, 5).log(),
        cpp_var_map={'i': '_get_input()', 't': t, 'weight': weight},
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs(i)),
        pickle=False,
        default_dtype=torch.double)


def nlllossNd_no_reduce_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for N-d negative log
    # likelihood loss with no reduction. Input has shape 2 by 3 by 5 by 5 by 2 by 2 and targets
    # have shape 2 by 5 by 5 by 2 by 2.
    t = Variable(torch.rand(2, 5, 5, 2, 2).mul(3).floor().long())
    kwargs = {'reduction': 'none'}
    return dict(
        fullname='NLLLossNd_no_reduce',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), reduction=kwargs['reduction'])),
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong), F::NLLLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(2, 3, 5, 5, 2, 2).log(),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs),
        pickle=False,
        default_dtype=torch.double)


def nlllossNd_no_reduce_ignore_index_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for N-d negative log
    # likelihood loss with no reduction and ignore_index 1.
    t = Variable(torch.rand(2, 5, 5, 2, 2).mul(3).floor().long())
    kwargs: dict[str, int | str] = {'ignore_index': 1, 'reduction': 'none'}
    return dict(
        fullname='NLLLossNd_no_reduce_ignore_index',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), ignore_index=int(kwargs['ignore_index']),
                                 reduction=str(kwargs['reduction']))),
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong), F::NLLLossFuncOptions().ignore_index(1).reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(2, 3, 5, 5, 2, 2).log(),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs),
        pickle=False,
        default_dtype=torch.double)


def nlllossNd_no_reduce_weights_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for N-d negative log
    # likelihood loss with no reduction and per-class weights of length 3.
    t = Variable(torch.rand(2, 5, 5, 2, 2).mul(3).floor().long())
    weight = torch.rand(3)

    def kwargs(i):
        # [FUNCTION DOC] HELPER: Returns the options mapping (weight cast to the dtype of the
        # given tensor, reduction none).
        return {'weight': weight.type_as(i), 'reduction': 'none'}

    return dict(
        fullname='NLLLossNd_no_reduce_weights',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs(i))),
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong),
            F::NLLLossFuncOptions().weight(weight.to(i.options())).reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(2, 3, 5, 5, 2, 2).log(),
        cpp_var_map={'i': '_get_input()', 't': t, 'weight': weight},
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs(i)),
        pickle=False,
        default_dtype=torch.double)


def smoothl1loss_no_reduce_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for smooth L1 loss
    # with no reduction on 2 by 3 by 4 tensors.
    t = torch.randn(2, 3, 4, dtype=torch.double)
    return dict(
        fullname='SmoothL1Loss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.smooth_l1_loss(i, t.type_as(i), reduction='none')),
        cpp_function_call='''F::smooth_l1_loss(
            i, t.to(i.options()), F::SmoothL1LossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(2, 3, 4),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['SmoothL1Loss'](i, t.type_as(i), reduction='none'),
        supports_forward_ad=True,
        pickle=False,
        default_dtype=torch.double)


def smoothl1loss_no_reduce_scalar_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for smooth L1 loss
    # with no reduction on 0-dim input and target.
    t = torch.randn((), dtype=torch.double)
    return dict(
        fullname='SmoothL1Loss_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.smooth_l1_loss(i, t.type_as(i), reduction='none')),
        cpp_function_call='''F::smooth_l1_loss(
            i, t.to(i.options()), F::SmoothL1LossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(()),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['SmoothL1Loss'](i, t.type_as(i), reduction='none'),
        supports_forward_ad=True,
        pickle=False,
        default_dtype=torch.double)


def smoothl1loss_beta_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for smooth L1 loss
    # with no reduction and beta 0.5.
    t = torch.randn(2, 3, 4, dtype=torch.double)
    return dict(
        fullname='SmoothL1Loss_beta',
        constructor=wrap_functional(
            lambda i: F.smooth_l1_loss(i, t.type_as(i), reduction='none', beta=0.5)),
        cpp_function_call='''F::smooth_l1_loss(
            i, t.to(i.options()), F::SmoothL1LossFuncOptions().reduction(torch::kNone), 0.5)''',
        input_fn=lambda: torch.randn(2, 3, 4),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['SmoothL1Loss'](i, t.type_as(i), reduction='none', beta=0.5),
        supports_forward_ad=True,
        pickle=False,
        default_dtype=torch.double)


def smoothl1loss_zero_beta_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for smooth L1 loss
    # with no reduction and beta 0, where the loss degenerates to L1.
    t = torch.randn(2, 3, 4, dtype=torch.double)
    return dict(
        fullname='SmoothL1Loss_zero_beta',
        constructor=wrap_functional(
            lambda i: F.smooth_l1_loss(i, t.type_as(i), reduction='none', beta=0)),
        cpp_function_call='''F::smooth_l1_loss(
            i, t.to(i.options()), F::SmoothL1LossFuncOptions().reduction(torch::kNone), 0)''',
        input_fn=lambda: torch.randn(2, 3, 4),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['SmoothL1Loss'](i, t.type_as(i), reduction='none', beta=0),
        supports_forward_ad=True,
        pickle=False,
        default_dtype=torch.double)


def huberloss_delta_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for Huber loss
    # with no reduction and delta 0.5. The target is generated in the default dtype rather
    # than double.
    t = torch.randn(2, 3, 4)
    return dict(
        fullname='HuberLoss_delta',
        constructor=wrap_functional(
            lambda i: F.huber_loss(i, t.type_as(i), reduction='none', delta=0.5)),
        cpp_function_call='''F::huber_loss(
            i, t.to(i.options()), F::HuberLossFuncOptions().reduction(torch::kNone).delta(0.5))''',
        input_fn=lambda: torch.randn(2, 3, 4),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['HuberLoss'](i, t.type_as(i), reduction='none', delta=0.5),
        supports_forward_ad=True,
        pickle=False,
        default_dtype=torch.double)


def multilabelmarginloss_0d_no_reduce_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for multilabel margin
    # loss with no reduction on 0-dim input and a zero 0-dim long target. Sum-reduction checking
    # is enabled and double-backward is disabled.
    # KNOWN ISSUE: does not set default_dtype to double, unlike most siblings.
    t = torch.zeros(()).long()
    return dict(
        fullname='MultiLabelMarginLoss_0d_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multilabel_margin_loss(i, t.type_as(i).long(), reduction='none')),
        cpp_function_call='''F::multilabel_margin_loss(
            i, t.to(i.options()).to(torch::kLong), F::MultilabelMarginLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(()),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiLabelMarginLoss'](i, t.data.type_as(i).long(), reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False)


def multilabelmarginloss_1d_no_reduce_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for multilabel margin
    # loss with no reduction on a 1-d input of length 10. Target is random class indices 0..9
    # (no negative terminator); sum-reduction check on, double-backward off.
    t = Variable(torch.rand(10).mul(10).floor().long())
    return dict(
        fullname='MultiLabelMarginLoss_1d_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multilabel_margin_loss(i, t.type_as(i).long(), reduction='none')),
        cpp_function_call='''F::multilabel_margin_loss(
            i, t.to(i.options()).to(torch::kLong), F::MultilabelMarginLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(10),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiLabelMarginLoss'](i, t.data.type_as(i).long(), reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False,
        default_dtype=torch.double)


def multilabelmarginloss_index_neg_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for multilabel margin
    # loss with no reduction where targets may contain negative values. Targets are random
    # integers clamped from below at -1 so rows end early at the first negative entry.
    t = Variable(torch.clamp(torch.rand(5, 10).add(-.5).mul(20).floor().long(), min=-1))
    return dict(
        fullname='MultiLabelMarginLoss_index_neg',
        constructor=wrap_functional(
            lambda i: F.multilabel_margin_loss(i, t.type_as(i).long(), reduction='none')),
        cpp_function_call='''F::multilabel_margin_loss(
            i, t.to(i.options()).to(torch::kLong), F::MultilabelMarginLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(5, 10),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiLabelMarginLoss'](i, t.data.type_as(i).long(), reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False,
        default_dtype=torch.double)


def multilabelmarginloss_no_reduce_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for multilabel margin
    # loss with no reduction on 5 by 10 input and random class-index targets.
    t = Variable(torch.rand(5, 10).mul(10).floor().long())
    return dict(
        fullname='MultiLabelMarginLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multilabel_margin_loss(i, t.type_as(i).long(), reduction='none')),
        cpp_function_call='''F::multilabel_margin_loss(
            i, t.to(i.options()).to(torch::kLong), F::MultilabelMarginLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(5, 10),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiLabelMarginLoss'](i, t.data.type_as(i).long(), reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False,
        default_dtype=torch.double)


def hingeembeddingloss_no_reduce_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for hinge embedding
    # loss with no reduction. Targets are random plus or minus one values of length 10.
    t = Variable(torch.randn(10).gt(0).to(torch.double).mul_(2).sub(1))
    return dict(
        fullname='HingeEmbeddingLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.hinge_embedding_loss(i, t.type_as(i), reduction='none')),
        cpp_function_call='''F::hinge_embedding_loss(
            i, t.to(i.options()), F::HingeEmbeddingLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(10),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['HingeEmbeddingLoss'](i, t.type_as(i), reduction='none'),
        check_sum_reduction=True,
        pickle=False,
        default_dtype=torch.double)


def hingeembeddingloss_margin_no_reduce_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for hinge embedding
    # loss with no reduction and margin 0.5.
    t = Variable(torch.randn(10).gt(0).to(torch.double).mul_(2).sub(1))
    return dict(
        fullname='HingeEmbeddingLoss_margin_no_reduce',
        constructor=wrap_functional(
            lambda i: F.hinge_embedding_loss(i, t.type_as(i), margin=0.5, reduction='none')),
        cpp_function_call='''F::hinge_embedding_loss(
            i, t.to(i.options()), F::HingeEmbeddingLossFuncOptions().margin(0.5).reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(10),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['HingeEmbeddingLoss'](i, t.type_as(i), margin=0.5, reduction='none'),
        check_sum_reduction=True,
        pickle=False,
        default_dtype=torch.double)


def softmarginloss_no_reduce_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for soft margin loss
    # with no reduction on 5 by 5 tensors with a random normal target.
    t = torch.randn(5, 5, dtype=torch.double)
    return dict(
        fullname='SoftMarginLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.soft_margin_loss(i, t.type_as(i), reduction='none')),
        cpp_function_call='''F::soft_margin_loss(
            i, t.to(i.options()), F::SoftMarginLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(5, 5),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['SoftMarginLoss'](i, t.type_as(i), reduction='none'),
        supports_forward_ad=True,
        pickle=False,
        default_dtype=torch.double)


def multilabelsoftmarginloss_no_reduce_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for multilabel soft
    # margin loss with no reduction. Targets are random zero/one values of shape 5 by 10; the
    # reference is the negative log-sigmoid expression summed over classes and divided by the
    # class count; double-backward off.
    t = torch.rand(5, 10).mul(2).floor()
    return dict(
        fullname='MultiLabelSoftMarginLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multilabel_soft_margin_loss(i, t.type_as(i), reduction='none')),
        cpp_function_call='''F::multilabel_soft_margin_loss(
            i, t.to(i.options()), F::MultilabelSoftMarginLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(5, 10),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            (-(t * i.sigmoid().log() + (1 - t) * (-i).sigmoid().log())).sum(dim=1) / i.size(1),
        check_gradgrad=False,
        pickle=False,
        default_dtype=torch.double)


def multilabelsoftmarginloss_weights_no_reduce_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for multilabel soft
    # margin loss with no reduction and per-class weights of length 10.
    t = torch.rand(5, 10).mul(2).floor()
    weights = torch.rand(10)
    return dict(
        fullname='MultiLabelSoftMarginLoss_weights_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multilabel_soft_margin_loss(i, t.type_as(i),
                                                    weight=weights.type_as(i), reduction='none')),
        cpp_function_call='''F::multilabel_soft_margin_loss(
            i, t.to(i.options()),
            F::MultilabelSoftMarginLossFuncOptions().weight(weights.to(i.options())).reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(5, 10),
        cpp_var_map={'i': '_get_input()', 't': t, 'weights': weights},
        reference_fn=lambda i, *_:
            (-(t * i.sigmoid().log() + (1 - t) * (-i).sigmoid().log()) * weights).sum(dim=1) / i.size(1),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False,
        default_dtype=torch.double)


def multimarginloss_no_reduce_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for multi margin loss
    # with no reduction. Targets are 5 random class indices in 0..7 with 10-class inputs of
    # shape 5 by 10; sum-reduction check on, double-backward off.
    t = torch.rand(5).mul(8).floor().long()
    return dict(
        fullname='MultiMarginLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), reduction='none')),
        cpp_function_call='''F::multi_margin_loss(
            i, t.to(i.options()).to(torch::kLong), F::MultiMarginLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(5, 10),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(), reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False,
        default_dtype=torch.double)


def multimarginloss_1d_no_reduce_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for multi margin loss
    # with no reduction on a 1-d input of length 10 with a length-1 target.
    t = torch.rand(1).mul(8).floor().long()
    return dict(
        fullname='MultiMarginLoss_1d_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), reduction='none')),
        cpp_function_call='''F::multi_margin_loss(
            i, t.to(i.options()).to(torch::kLong), F::MultiMarginLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(10),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(), reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False,
        default_dtype=torch.double)


def multimarginloss_1d_input_0d_target_no_reduce_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for multi margin loss
    # with no reduction on a 1-d input with a 0-dim target.
    t = torch.rand(()).mul(8).floor().long()
    return dict(
        fullname='multimarginloss_1d_input_0d_target_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), reduction='none')),
        cpp_function_call='''F::multi_margin_loss(
            i, t.to(i.options()).to(torch::kLong), F::MultiMarginLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(10),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(), reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False,
        default_dtype=torch.double)


def multimarginloss_p_no_reduce_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for multi margin loss
    # with no reduction and p equal to 2. Inputs are clamped to the open unit interval.
    t = torch.rand(5).mul(8).floor().long()
    return dict(
        fullname='MultiMarginLoss_p_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), p=2, reduction='none')),
        cpp_function_call='''F::multi_margin_loss(
            i, t.to(i.options()).to(torch::kLong), F::MultiMarginLossFuncOptions().p(2).reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(5, 10).clamp_(1e-2, 1 - 1e-2),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(), p=2, reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False,
        default_dtype=torch.double)


def multimarginloss_margin_no_reduce_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for multi margin loss
    # with no reduction and margin 0.5.
    t = torch.rand(5).mul(8).floor().long()
    return dict(
        fullname='MultiMarginLoss_margin_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), margin=0.5, reduction='none')),
        cpp_function_call='''F::multi_margin_loss(
            i, t.to(i.options()).to(torch::kLong),
            F::MultiMarginLossFuncOptions().margin(0.5).reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(5, 10),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(),
                                                  margin=0.5, reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False,
        default_dtype=torch.double)


def multimarginloss_weights_no_reduce_test():
    # [FUNCTION DOC] HELPER: Builds and returns one test-spec mapping (keys fullname,
    # constructor, cpp_function_call, cpp_var_map, reference_fn and flags) for multi margin loss
    # with no reduction and per-class weights of length 10.
    t = torch.rand(5).mul(8).floor().long()
    weights = torch.rand(10, dtype=torch.double)
    return dict(
        fullname='MultiMarginLoss_weights_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), weight=weights.type_as(i),
                                          reduction='none')),
        cpp_function_call='''F::multi_margin_loss(
            i, t.to(i.options()).to(torch::kLong),
            F::MultiMarginLossFuncOptions().weight(weights.to(i.options())).reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(5, 10),
        cpp_var_map={'i': '_get_input()', 't': t, 'weights': weights},
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(),
                                                  weight=weights, reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False,
        default_dtype=torch.double)


def single_batch_reference_fn(input, parameters, module):
    # [FUNCTION DOC] Reference function for module tests without a batch dimension: adds a
    # leading batch dimension of size one to the input (or each input in a list or tuple), runs
    # the module with the random number generator state frozen, and removes the batch dimension
    # from the output. Args: input, parameters (unused), module. Returns the squeezed output.
    """Reference function for modules supporting no batch dimensions.

    The module is passed the input and target in batched form with a single item.
    The output is squeezed to compare with the no-batch input.
    """
    def unsqueeze_inp(inp):
        # [FUNCTION DOC] HELPER: Adds a leading dimension of size one to a tensor, or to every
        # tensor of a list or tuple (returned as a list).
        if isinstance(inp, (list, tuple)):
            return [t.unsqueeze(0) for t in inp]
        return inp.unsqueeze(0)

    single_batch_input = unsqueeze_inp(input)
    single_batch_input = [single_batch_input] if isinstance(single_batch_input, torch.Tensor) else single_batch_input
    with freeze_rng_state():
        return module(*single_batch_input).squeeze(0)


def get_new_module_tests():
    # [FUNCTION DOC] Builds and returns the full list of test-spec mappings for nn modules and
    # functionals. Seeds the global random number generator first, then collects the loss
    # fixture mappings, a long literal list of per-module mappings (keys such as module_name,
    # constructor_args, cpp_constructor_args, input_size, desc, reference_fn, cudnn, tf32
    # settings, default_dtype), convolution padding-mode variants for 1-d, 2-d and 3-d (3-d
    # reflect skipped), and no-batch-dimension variants of non-linear activations. Mutates
    # global RNG state; consumers must call it once and share the result.
    common_utils.set_rng_seed()
    new_module_tests = [
        poissonnllloss_no_reduce_test(),
        bceloss_no_reduce_test(),
        bceloss_weights_no_reduce_test(),
        bce_with_logistic_legacy_enum_test(),
        bce_with_logistic_no_reduce_test(),
        bceloss_no_reduce_scalar_test(),
        bceloss_weights_no_reduce_scalar_test(),
        bce_with_logistic_no_reduce_scalar_test(),
        kldivloss_with_target_no_reduce_test(),
        kldivloss_no_reduce_test(),
        kldivloss_no_reduce_scalar_test(),
        kldivloss_with_log_target_no_reduce_test(),
        kldivloss_no_reduce_log_target_test(),
        kldivloss_no_reduce_scalar_log_target_test(),
        l1loss_no_reduce_test(),
        l1loss_no_reduce_complex_test(),
        l1loss_no_reduce_scalar_test(),
        mseloss_no_reduce_test(),
        mseloss_no_reduce_scalar_test(),
        nllloss_no_reduce_test(),
        nllloss_no_reduce_ignore_index_test(),
        nllloss_no_reduce_weights_test(),
        nllloss_no_reduce_weights_ignore_index_test(),
        nllloss_no_reduce_weights_ignore_index_neg_test(),
        nllloss2d_no_reduce_test(),
        nllloss2d_no_reduce_weights_test(),
        nllloss2d_no_reduce_ignore_index_test(),
        nlllossNd_no_reduce_test(),
        nlllossNd_no_reduce_weights_test(),
        nlllossNd_no_reduce_ignore_index_test(),
        smoothl1loss_no_reduce_test(),
        smoothl1loss_no_reduce_scalar_test(),
        smoothl1loss_beta_test(),
        smoothl1loss_zero_beta_test(),
        huberloss_delta_test(),
        multilabelmarginloss_0d_no_reduce_test(),
        multilabelmarginloss_1d_no_reduce_test(),
        multilabelmarginloss_index_neg_test(),
        multilabelmarginloss_no_reduce_test(),
        hingeembeddingloss_no_reduce_test(),
        hingeembeddingloss_margin_no_reduce_test(),
        softmarginloss_no_reduce_test(),
        multilabelsoftmarginloss_no_reduce_test(),
        multilabelsoftmarginloss_weights_no_reduce_test(),
        multimarginloss_no_reduce_test(),
        multimarginloss_1d_no_reduce_test(),
        multimarginloss_1d_input_0d_target_no_reduce_test(),
        multimarginloss_p_no_reduce_test(),
        multimarginloss_margin_no_reduce_test(),
        multimarginloss_weights_no_reduce_test(),
        dict(
            module_name='Conv1d',
            constructor_args=(4, 5, 3),
            cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3)',
            input_size=(2, 4, 10),
            cudnn=True,
            with_tf32=True,
            tf32_precision=0.005,
            default_dtype=torch.double,
        ),
        dict(
            module_name='Conv1d',
            constructor_args=(4, 5, 3, 2),
            cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3).stride(2)',
            input_size=(2, 4, 10),
            cudnn=True,
            desc='stride',
            with_tf32=True,
            tf32_precision=0.005,
            default_dtype=torch.double,
        ),
        dict(
            module_name='Conv1d',
            constructor_args=(4, 5, 3, 1, 1),
            cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3).stride(1).padding(1)',
            input_size=(2, 4, 10),
            cudnn=True,
            desc='pad1',
            with_tf32=True,
            tf32_precision=0.01,
            default_dtype=torch.double,
        ),
        dict(
            module_name='Conv1d',
            constructor_args=(4, 5, 5, 1, 2),
            cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 5).stride(1).padding(2)',
            input_size=(2, 4, 10),
            cudnn=True,
            desc='pad2',
            with_tf32=True,
            tf32_precision=0.005,
            default_dtype=torch.double,
        ),
        dict(
            module_name='Conv1d',
            constructor_args=(4, 4, 3, 1, 1),
            cpp_constructor_args='torch::nn::Conv1dOptions(4, 4, 3).stride(1).padding(1)',
            input_size=(1, 4, 1),
            cudnn=True,
            desc='pad1size1',
            with_tf32=True,
            tf32_precision=0.005,
            default_dtype=torch.double,
        ),
        dict(
            module_name='Conv1d',
            constructor_args=(4, 4, 5, 1, 2),
            cpp_constructor_args='torch::nn::Conv1dOptions(4, 4, 5).stride(1).padding(2)',
            input_size=(1, 4, 1),
            cudnn=True,
            desc='pad2size1',
            with_tf32=True,
            tf32_precision=0.005,
            default_dtype=torch.double,
        ),
        dict(
            module_name='Conv1d',
            constructor_args=(4, 5, 3),
            cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3)',
            input_size=(0, 4, 10),
            cudnn=True,
            desc='zero_batch',
            with_tf32=True,
            tf32_precision=0.005,
        ),
        dict(
            fullname='Conv1d_dilated',
            constructor=lambda: nn.Conv1d(4, 5, kernel_size=3, dilation=2),
            cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3).dilation(2)',
            input_size=(2, 4, 10),
            with_tf32=True,
            tf32_precision=0.005,
            default_dtype=torch.double,
        ),
        dict(
            fullname='Conv1d_groups',
            constructor=lambda: nn.Conv1d(4, 6, kernel_size=3, groups=2),
            cpp_constructor_args='torch::nn::Conv1dOptions(4, 6, 3).groups(2)',
            input_size=(2, 4, 6),
            cudnn=True,
            with_tf32=True,
            tf32_precision=0.005,
            default_dtype=torch.double,
        ),
        dict(
            fullname='Conv1d_pad_valid',
            constructor=lambda: nn.Conv1d(4, 5, 3, padding="valid"),
            cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3).padding(torch::kValid)',
            input_size=(2, 4, 10),
            cudnn=True,
            with_tf32=True,
            tf32_precision=0.005,
            default_dtype=torch.double,
        ),
        dict(
            fullname='Conv1d_pad_same',
            constructor=lambda: nn.Conv1d(4, 5, 3, padding="same"),
            cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3).padding(torch::kSame)',
            input_size=(2, 4, 10),
            cudnn=True,
            with_tf32=True,
            tf32_precision=0.005,
            default_dtype=torch.double,
        ),
        dict(
            fullname='Conv1d_pad_same2',
            constructor=lambda: nn.Conv1d(4, 5, 4, padding="same"),
            cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 4).padding(torch::kSame)',
            input_size=(2, 4, 10),
            cudnn=True,
            with_tf32=True,
            tf32_precision=0.005,
            default_dtype=torch.double,
        ),
        dict(
            fullname='Conv1d_pad_same_dilated',
            constructor=lambda: nn.Conv1d(4, 5, 4, padding="same", dilation=2),
            cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3).padding(torch::kSame).dilation(2)',
            input_size=(2, 4, 10),
            cudnn=True,
            with_tf32=True,
            tf32_precision=0.005,
            default_dtype=torch.double,
        ),
        dict(
            fullname='ConvTranspose1d',
            constructor=lambda: nn.ConvTranspose1d(3, 4, kernel_size=3, stride=(3,), padding=1, output_padding=(1,)),
            cpp_constructor_args='torch::nn::ConvTranspose1dOptions(3, 4, 3).stride(3).padding(1).output_padding(1)',
            cudnn=True,
            input_size=(1, 3, 7),
            with_tf32=True,
            tf32_precision=0.005,
            default_dtype=torch.double,
        ),
        dict(
            module_name='ConvTranspose1d',
            constructor_args=(3, 4, 3, 2, 1, 1, 1, False),
            cpp_constructor_args='''torch::nn::ConvTranspose1dOptions(3, 4, 3)
                                    .stride(2).padding(1).output_padding(1).groups(1).bias(false)''',
            input_size=(1, 3, 6),
            cudnn=True,
            desc='no_bias',
            with_tf32=True,
            tf32_precision=0.005,
            default_dtype=torch.double,
        ),
        dict(
            module_name='ConvTranspose1d',
            constructor_args=(3, 4, 3, 2, 1, 1, 1, True, 2),
            cpp_constructor_args='''torch::nn::ConvTranspose1dOptions(3, 4, 3)
                                    .stride(2).padding(1).output_padding(1).groups(1).bias(true).dilation(2)''',
            input_size=(1, 3, 6),
            cudnn=True,
            desc='dilated',
            with_tf32=True,
            tf32_precision=0.005,
            default_dtype=torch.double,
        ),
        dict(
            fullname='ConvTranspose1d_groups',
            constructor=lambda: nn.ConvTranspose1d(4, 6, 3, stride=(3,), padding=1, output_padding=(1,), groups=2),
            cpp_constructor_args='''torch::nn::ConvTranspose1dOptions(4, 6, 3)
                                    .stride(3).padding(1).output_padding(1).groups(2)''',
            cudnn=True,
            input_size=(2, 4, 7),
            with_tf32=True,
            tf32_precision=0.005,
            default_dtype=torch.double,
        ),
        dict(
            module_name='Conv2d',
            constructor_args=(3, 4, (3, 2)),
            cpp_constructor_args='torch::nn::Conv2dOptions(3, 4, {3, 2})',
            input_size=(2, 3, 7, 5),
            cudnn=True,
            check_with_long_tensor=True,
            with_tf32=True,
            tf32_precision=0.005,
            default_dtype=torch.double,
        ),
        dict(
            module_name='Conv2d',
            constructor_args=(3, 4, (3, 3), (2, 2)),
            cpp_constructor_args='torch::nn::Conv2dOptions(3, 4, {3, 3}).stride({2, 2})',
            input_size=(2, 3, 6, 6),
            cudnn=True,
            desc='strided',
            check_with_long_tensor=True,
            with_tf32=True,
            tf32_precision=0.005,
            default_dtype=torch.double,
        ),
        dict(
            module_name='Conv2d',
            constructor_args=(3, 4, (3, 3), (2, 2), (1, 1)),
            cpp_constructor_args='torch::nn::Conv2dOptions(3, 4, {3, 3}).stride({2, 2}).padding({1, 1})',
            input_size=(2, 3, 6, 6),
            cudnn=True,
            desc='padding',
            check_with_long_tensor=True,
            with_tf32=True,
            tf32_precision=0.005,
            default_dtype=torch.double,
        ),
        dict(
            module_name='Conv2d',
            constructor_args=(3, 2, (3, 3), (2, 2), (1, 1), (2, 2)),
            cpp_constructor_args='torch::nn::Conv2dOptions(3, 2, {3, 3}).stride({2, 2}).padding({1, 1}).dilation({2, 2})',
            input_size=(2, 3, 8, 8),
            cudnn=True,
            desc='dilated',
            check_with_long_tensor=True,
            with_tf32=True,
            tf32_precision=0.005,
            default_dtype=torch.double,
        ),
        dict(
            module_name='Conv2d',
            constructor_args=(3, 4, (3, 2), 1, 0, 1, 1, False),
            cpp_constructor_args='''torch::nn::Conv2dOptions(3, 4, {3, 2})
                                    .stride(1).padding(0).dilation(1).groups(1).bias(false)''',
            input_size=(2, 3, 6, 5),
            cudnn=True,
            desc='no_bias',
            check_with_long_tensor=True,
            with_tf32=True,
            tf32_precision=0.015,
            default_dtype=torch.double,
        ),
        dict(
            module_name='Conv2d',
            constructor_args=(3, 4, (3, 2)),
            cpp_constructor_args='torch::nn::Conv2dOptions(3, 4, {3, 2})',
            input_size=(0, 3, 7, 5),
            cudnn=True,
            desc='zero_batch',
            check_with_long_tensor=True,
            with_tf32=True,
        ),
        dict(
            fullname='Conv2d_groups',
            constructor=lambda: nn.Conv2d(4, 6, (3, 2), groups=2),
            cpp_constructor_args='torch::nn::Conv2dOptions(4, 6, {3, 2}).groups(2)',
            input_size=(2, 4, 6, 5),
            cudnn=True,
            check_with_long_tensor=True,
            with_tf32=True,
            tf32_precision=0.015,
            default_dtype=torch.double,
        ),
        dict(
            fullname='Conv2d_groups_thnn',
            constructor=lambda: nn.Conv2d(4, 6, (3, 2), groups=2),
            cpp_constructor_args='torch::nn::Conv2dOptions(4, 6, {3, 2}).groups(2)',
            input_size=(2, 4, 6, 5),
            check_with_long_tensor=True,
            with_tf32=True,
            tf32_precision=0.015,
            default_dtype=torch.double,
        ),
        dict(
            fullname='Conv2d_pad_valid',
            constructor=lambda: nn.Conv2d(2, 4, (3, 4), padding="valid"),
            cpp_constructor_args='torch::nn::Conv2dOptions(2, 4, {3, 4}).padding(torch::kValid)',
            input_size=(2, 2, 6, 5),
            cudnn=True,
            with_tf32=True,
            tf32_precision=0.005,
            default_dtype=torch.double,
        ),
        dict(
            fullname='Conv2d_pad_same',
            constructor=lambda: nn.Conv2d(2, 4, (3, 4), padding="same"),
            cpp_constructor_args='torch::nn::Conv2dOptions(2, 4, {3, 4}).padding(torch::kSame)',
            input_size=(2, 2, 6, 5),
            cudnn=True,
            with_tf32=True,
            tf32_precision=0.01,
            default_dtype=torch.double,
        ),
        dict(
            fullname='Conv2d_pad_same_dilated',
            constructor=lambda: nn.Conv2d(2, 4, (3, 4), padding="same", dilation=2),
            cpp_constructor_args='torch::nn::Conv2dOptions(2, 4, {3, 4}).padding(torch::kSame).dilation(2)',
            input_size=(2, 2, 6, 5),
            cudnn=True,
            with_tf32=True,
            tf32_precision=0.01,
            default_dtype=torch.double,
        ),
        dict(
            module_name='ConvTranspose2d',
            constructor_args=(3, 4, 3, (3, 2), 1, (1, 1)),
            cpp_constructor_args='''torch::nn::ConvTranspose2dOptions(3, 4, 3)
                                    .stride({3, 2}).padding(1).output_padding({1, 1})''',
            cudnn=True,
            input_size=(1, 3, 7, 6),
            check_with_long_tensor=True,
            with_tf32=True,
            tf32_precision=0.01,
            default_dtype=torch.double,
        ),
        dict(
            module_name='ConvTranspose2d',
            constructor_args=(3, 4, 3, (2, 3), 1, (1, 1), 1, False, (2, 2)),
            cpp_constructor_args='''torch::nn::ConvTranspose2dOptions(3, 4, 3)
                                    .stride({2, 3})
                                    .padding(1)
                                    .output_padding({1, 1})
                                    .groups(1)
                                    .bias(false)
                                    .dilation({2, 2})''',
            input_size=(1, 3, 6, 7),
            cudnn=True,
            desc='dilated',
            check_with_long_tensor=True,
            with_tf32=True,
            tf32_precision=0.01,
            default_dtype=torch.double,
        ),
        dict(
            module_name='ConvTranspose2d',
            constructor_args=(3, 4, 3, (2, 3), 1, (1, 1), 1, False),
            cpp_constructor_args='''torch::nn::ConvTranspose2dOptions(3, 4, 3)
                                    .stride({2, 3}).padding(1).output_padding({1, 1}).groups(1).bias(false)''',
            input_size=(1, 3, 6, 7),
            cudnn=True,
            desc='no_bias',
            check_with_long_tensor=True,
            with_tf32=True,
            tf32_precision=0.01,
            default_dtype=torch.double,
        ),
        dict(
            fullname='ConvTranspose2d_groups',
            constructor=lambda: nn.ConvTranspose2d(2, 4, (2, 3), groups=2),
            cpp_constructor_args='torch::nn::ConvTranspose2dOptions(2, 4, {2, 3}).groups(2)',
            input_size=(1, 2, 4, 5),
            cudnn=True,
            check_with_long_tensor=True,
            with_tf32=True,
            tf32_precision=0.01,
            default_dtype=torch.double,
        ),
        dict(
            fullname='Conv2d_depthwise',
            constructor=lambda: nn.Conv2d(4, 4, (3, 3), groups=4),
            cpp_constructor_args='torch::nn::Conv2dOptions(4, 4, {3, 3}).groups(4)',
            input_size=(2, 4, 6, 6),
            with_tf32=True,
            tf32_precision=0.005,
            default_dtype=torch.double,
        ),
        dict(
            fullname='Conv2d_depthwise_with_multiplier',
            constructor=lambda: nn.Conv2d(4, 8, (3, 3), groups=4),
            cpp_constructor_args='torch::nn::Conv2dOptions(4, 8, {3, 3}).groups(4)',
            input_size=(2, 4, 6, 6),
            with_tf32=True,
            tf32_precision=0.005,
            default_dtype=torch.double,
        ),
        dict(
            fullname='Conv2d_depthwise_strided',
            constructor=lambda: nn.Conv2d(4, 4, (3, 3), stride=(2, 2), groups=4),
            cpp_constructor_args='torch::nn::Conv2dOptions(4, 4, {3, 3}).stride({2, 2}).groups(4)',
            input_size=(2, 4, 6, 6),
            with_tf32=True,
            tf32_precision=0.005,
            default_dtype=torch.double,
        ),
        dict(
            fullname='Conv2d_depthwise_padded',
            constructor=lambda: nn.Conv2d(4, 4, (3, 3), padding=(1, 1), groups=4),
            cpp_constructor_args='torch::nn::Conv2dOptions(4, 4, {3, 3}).padding({1, 1}).groups(4)',
            input_size=(2, 4, 6, 6),
            with_tf32=True,
            tf32_precision=0.005,
            default_dtype=torch.double,
        ),
        dict(
            fullname='Conv2d_depthwise_dilated',
            constructor=lambda: nn.Conv2d(4, 4, (2, 2), dilation=(2, 2), groups=4),
            cpp_constructor_args='torch::nn::Conv2dOptions(4, 4, {2, 2}).dilation({2, 2}).groups(4)',
            input_size=(2, 4, 5, 5),
            with_tf32=True,
            tf32_precision=0.005,
            default_dtype=torch.double,
        ),
        dict(
            module_name='Conv3d',
            constructor_args=(2, 3, (2, 3, 2)),
            cpp_constructor_args='torch::nn::Conv3dOptions(2, 3, {2, 3, 2})',
            input_size=(1, 2, 4, 5, 4),
            cudnn=True,
            check_with_long_tensor=True,
            with_tf32=True,
            tf32_precision=0.05,
            default_dtype=torch.double,
        ),
        dict(
            module_name='Conv3d',
            constructor_args=(2, 3, (2, 3, 4), 1, 0, 1, 1, False),
            cpp_constructor_args='''torch::nn::Conv3dOptions(2, 3, {2, 3, 4})
                                    .stride(1).padding(0).dilation(1).groups(1).bias(false)''',
            input_size=(1, 2, 3, 4, 5),
            cudnn=True,
            desc='no_bias',
            check_with_long_tensor=True,
            with_tf32=True,
            tf32_precision=0.05,
            default_dtype=torch.double,
        ),
        dict(
            module_name='Conv3d',
            constructor_args=(2, 3, (1, 1, 1), 1, 0, 1, 1, False),
            cpp_constructor_args='''torch::nn::Conv3dOptions(2, 3, {2, 3, 4})
                                    .stride(1).padding(0).dilation(1).groups(1).bias(false)''',
            input_size=(1, 2, 3, 4, 5),
            cudnn=True,
            desc='1x1x1_no_bias',
            check_with_long_tensor=False,
            with_tf32=True,
            tf32_precision=0.05,
            default_dtype=torch.double,
        ),
        dict(
            module_name='Conv3d',
            constructor_args=(3, 4, 2, 2),
            cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, 2).stride(2)',
            input_size=(2, 3, 5, 5, 5),
            cudnn=True,
            desc='stride',
            check_with_long_tensor=True,
            with_tf32=True,
            tf32_precision=0.05,
            default_dtype=torch.double,
        ),
        dict(
            module_name='Conv3d',
            constructor_args=(3, 4, 2, 2, 1),
            cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, 2).stride(2).padding(1)',
            input_size=(2, 3, 5, 5, 5),
            cudnn=True,
            desc='stride_padding',
            check_with_long_tensor=True,
            with_tf32=True,
            tf32_precision=0.05,
            default_dtype=torch.double,
        ),
        dict(
            module_name='Conv3d',
            constructor_args=(3, 4, (2, 3, 4)),
            cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, {2, 3, 4})',
            input_size=(0, 3, 3, 4, 5),
            cudnn=True,
            check_with_long_tensor=True,
            desc='zero_batch',
            with_tf32=True,
        ),
        dict(
            fullname='Conv3d_groups',
            constructor=lambda: nn.Conv3d(2, 4, kernel_size=3, groups=2),
            cpp_constructor_args='torch::nn::Conv3dOptions(2, 4, 3).groups(2)',
            input_size=(1, 2, 4, 5, 4),
            cudnn=True,
            check_with_long_tensor=True,
            with_tf32=True,
            tf32_precision=0.005,
            default_dtype=torch.double,
        ),
        dict(
            fullname='Conv3d_dilated',
            constructor=lambda: nn.Conv3d(3, 4, kernel_size=2, dilation=2),
            cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, 2).dilation(2)',
            input_size=(2, 3, 5, 5, 5),
            with_tf32=True,
            tf32_precision=0.05,
            default_dtype=torch.double,
        ),
        dict(
            fullname='Conv3d_dilated_strided',
            constructor=lambda: nn.Conv3d(3, 4, kernel_size=2, dilation=2, stride=2),
            cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, 2).dilation(2).stride(2)',
            input_size=(2, 3, 5, 5, 5),
            with_tf32=True,
            tf32_precision=0.05,
            default_dtype=torch.double,
        ),
        dict(
            fullname='Conv3d_pad_valid',
            constructor=lambda: nn.Conv3d(3, 4, (2, 3, 4), padding="valid"),
            cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, {2, 3, 4}).padding(torch::kValid)',
            input_size=(2, 3, 6, 5, 4),
            cudnn=True,
            with_tf32=True,
            tf32_precision=0.05,
            default_dtype=torch.double,
        ),
        dict(
            fullname='Conv3d_pad_same',
            constructor=lambda: nn.Conv3d(3, 4, (2, 3, 4), padding="same"),
            cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, {2, 3, 4}).padding(torch::kSame)',
            input_size=(2, 3, 6, 5, 4),
            cudnn=True,
            with_tf32=True,
            tf32_precision=0.05,
            default_dtype=torch.double,
        ),
        dict(
            fullname='Conv3d_pad_same_dilated',
            constructor=lambda: nn.Conv3d(3, 4, (2, 3, 4), padding="same", dilation=2),
            cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, {2, 3, 4}).padding(torch::kSame).dilation(2)',
            input_size=(2, 3, 6, 5, 4),
            cudnn=True,
            with_tf32=True,
            tf32_precision=0.05,
            default_dtype=torch.double,
        ),
        dict(
            module_name='ConvTranspose3d',
            constructor_args=(2, 3, (2, 3, 2)),
            cpp_constructor_args='torch::nn::ConvTranspose3dOptions(2, 3, {2, 3, 2})',
            cudnn=True,
            input_size=(1, 2, 4, 5, 4),
            with_tf32=True,
            tf32_precision=0.05,
            default_dtype=torch.double,
        ),
        dict(
            module_name='ConvTranspose3d',
            constructor_args=(2, 3, (2, 3, 2), 1, 0, 0, 1, True, (2, 2, 2)),
            cpp_constructor_args='''torch::nn::ConvTranspose3dOptions(2, 3, {2, 3, 2})
                                    .stride(1).padding(0).output_padding(0).groups(1).bias(true).dilation({2, 2, 2})''',
            cudnn=True,
            input_size=(1, 2, 4, 5, 4),
            desc='dilated',
            with_tf32=True,
            tf32_precision=0.05,
            default_dtype=torch.double,
        ),
        dict(
            module_name='ReplicationPad3d',
            constructor_args=((1, 2, 3, 3, 2, 1),),
            cpp_constructor_args='torch::nn::ReplicationPad3dOptions({1, 2, 3, 3, 2, 1})',
            input_size=(2, 3, 2, 2, 2),
            default_dtype=torch.double,
        ),
        dict(
            module_name='ReplicationPad3d',
            constructor_args=((1, 2, 3, 3, 2, 1),),
            cpp_constructor_args='torch::nn::ReplicationPad3dOptions({1, 2, 3, 3, 2, 1})',
            input_size=(3, 2, 2, 2),
            reference_fn=single_batch_reference_fn,
            desc='no_batch_dim',
            default_dtype=torch.double,
        ),
        dict(
            module_name='ReplicationPad3d',
            constructor_args=((1, 2, 3, 3, 2, 1),),
            cpp_constructor_args='torch::nn::ReplicationPad3dOptions({1, 2, 3, 3, 2, 1})',
            input_fn=lambda: torch.rand(2, 3, 2, 2, 2, dtype=torch.complex128, requires_grad=True),
            skip_half=True,
            desc='complex'
        ),
        dict(
            module_name='Embedding',
            constructor_args=(4, 3),
            cpp_constructor_args='torch::nn::EmbeddingOptions(4, 3)',
            input_fn=lambda: torch.empty(2, 3, dtype=torch.long).random_(4),
            check_gradgrad=False,
            default_dtype=torch.double,
            decorator=skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/117971")
        ),
        dict(
            module_name='Embedding',
            constructor_args=(4, 3),
            cpp_constructor_args='torch::nn::EmbeddingOptions(4, 3)',
            input_fn=lambda: torch.empty(1, 512, dtype=torch.long).random_(4).expand(7, 512),
            check_gradgrad=False,
            desc='discontiguous',
            default_dtype=torch.double,
            decorator=skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/117971")
        ),
        dict(
            module_name='EmbeddingBag',
            constructor_args=(4, 3),
            cpp_constructor_args='torch::nn::EmbeddingBagOptions(4, 3)',
            input_fn=lambda: torch.empty(2, 3, dtype=torch.long).random_(4),
            check_gradgrad=False,
            desc='mean',
            default_dtype=torch.double,
        ),
        dict(
            module_name='EmbeddingBag',
            constructor_args=(4, 3),
            cpp_constructor_args='torch::nn::EmbeddingBagOptions(4, 3)',
            input_fn=lambda: torch.empty(1, 512, dtype=torch.long).random_(4).expand(7, 512),
            check_gradgrad=False,
            desc='discontiguous',
            default_dtype=torch.double,
        ),
        dict(
            module_name='EmbeddingBag',
            constructor_args=(4, 3, None, 2., False, 'sum'),
            cpp_constructor_args='''torch::nn::EmbeddingBagOptions(4, 3)
                                    .max_norm(std::nullopt).norm_type(2.).scale_grad_by_freq(false).mode(torch::kSum)''',
            input_fn=lambda: torch.empty(2, 3, dtype=torch.long).random_(4),
            check_gradgrad=False,
            desc='sum',
            default_dtype=torch.double,
        ),
        dict(
            module_name='EmbeddingBag',
            constructor_args=(4, 3, None, 2., False, 'max'),
            cpp_constructor_args='''torch::nn::EmbeddingBagOptions(4, 3)
                                    .max_norm(std::nullopt).norm_type(2.).scale_grad_by_freq(false).mode(torch::kMax)''',
            input_fn=lambda: torch.empty(2, 3, dtype=torch.long).random_(4),
            check_gradgrad=False,
            desc='max',
            default_dtype=torch.double,
        ),
        dict(
            fullname='EmbeddingBag_mean_padding_idx',
            constructor=lambda: nn.EmbeddingBag(4, 3, padding_idx=1),
            cpp_constructor_args='torch::nn::EmbeddingBagOptions(4, 3).padding_idx(1)',
            input_fn=lambda: torch.stack([torch.randperm(3), torch.randperm(3)]),
            check_gradgrad=False,
            default_dtype=torch.double,
        ),
        dict(
            fullname='EmbeddingBag_sum_padding_idx',
            constructor=lambda: nn.EmbeddingBag(4, 3, None, 2., False, 'sum', padding_idx=1),
            cpp_constructor_args='''torch::nn::EmbeddingBagOptions(4, 3)
                                    .max_norm(std::nullopt).norm_type(2.).scale_grad_by_freq(false).mode(torch::kSum).padding_idx(1)''',
            input_fn=lambda: torch.stack([torch.randperm(3), torch.randperm(3)]),
            check_gradgrad=False,
            default_dtype=torch.double,
        ),
        dict(
            fullname='EmbeddingBag_max_padding_idx',
            constructor=lambda: nn.EmbeddingBag(4, 3, None, 2., False, 'max', padding_idx=1),
            cpp_constructor_args='''torch::nn::EmbeddingBagOptions(4, 3)
                                    .max_norm(std::nullopt).norm_type(2.).scale_grad_by_freq(false).mode(torch::kMax).padding_idx(1)''',
            input_fn=lambda: torch.stack([torch.randperm(3), torch.randperm(3)]),
            check_gradgrad=False,
            default_dtype=torch.double,
        ),
        dict(
            fullname='EmbeddingBag_sparse',
            constructor=lambda: nn.EmbeddingBag(4, 3, sparse=True, dtype=torch.double),
            cpp_constructor_args='''torch::nn::EmbeddingBagOptions(4, 3)
                                    .sparse(true)._weight(torch::rand({4, 3}).to(torch::kFloat64))''',
            input_fn=lambda: torch.randperm(2).repeat(1, 2),
            check_gradgrad=False,
            has_sparse_gradients=True,
        ),
        dict(
            constructor=lambda: nn.Embedding(4, 3, dtype=torch.double, sparse=True),
            cpp_constructor_args='torch::nn::EmbeddingOptions(4, 3).sparse(true)._weight(torch::rand({4, 3}).to(torch::kFloat64))',
            input_fn=lambda: torch.randperm(2).repeat(1, 2),
            fullname='Embedding_sparse',
            check_gradgrad=False,
            has_sparse_gradients=True,
        ),
        dict(
            module_name='PixelShuffle',
            constructor_args=(3,),
            cpp_constructor_args='torch::nn::PixelShuffleOptions(3)',
            input_size=(1, 9, 4, 4),
            default_dtype=torch.double,
        ),
        dict(
            module_name='PixelUnshuffle',
            constructor_args=(3,),
            cpp_constructor_args='torch::nn::PixelUnshuffleOptions(3)',
            input_size=(1, 1, 12, 12),
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest'),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({12})).scale_factor(std::nullopt).mode(torch::kNearest)''',
            input_size=(1, 2, 4),
            fullname='interpolate_nearest_1d',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest'),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({12})).scale_factor(std::nullopt).mode(torch::kNearest)''',
            input_size=(0, 2, 4),
            fullname='interpolate_nearest_1d_zero_dim',
            pickle=False,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=(12, ), scale_factor=None, mode='nearest'),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({12})).scale_factor(std::nullopt).mode(torch::kNearest)''',
            input_size=(1, 2, 3),
            fullname='interpolate_nearest_tuple_1d',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=None, scale_factor=4., mode='nearest'),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::nullopt).scale_factor(std::vector<double>({4.})).mode(torch::kNearest)''',
            input_size=(1, 2, 4),
            fullname='interpolate_nearest_scale_1d',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='linear', align_corners=False),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({12}))
                                .scale_factor(std::nullopt)
                                .mode(torch::kLinear)
                                .align_corners(false)''',
            input_size=(1, 2, 4),
            fullname='interpolate_linear_1d',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=(4, ), scale_factor=None, mode='linear', align_corners=False),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({4}))
                                .scale_factor(std::nullopt)
                                .mode(torch::kLinear)
                                .align_corners(false)''',
            input_size=(1, 2, 3),
            fullname='interpolate_linear_tuple_1d',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=None, scale_factor=4., mode='linear', align_corners=False),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::nullopt)
                                .scale_factor(std::vector<double>({4.}))
                                .mode(torch::kLinear)
                                .align_corners(false)''',
            input_size=(1, 2, 4),
            fullname='interpolate_linear_scale_1d',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='linear', align_corners=False),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({12}))
                                .scale_factor(std::nullopt)
                                .mode(torch::kLinear)
                                .align_corners(false)''',
            input_size=(0, 2, 4),
            fullname='interpolate_linear_1d_zero_dim',
            pickle=False,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='linear', align_corners=True),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({12}))
                                .scale_factor(std::nullopt)
                                .mode(torch::kLinear)
                                .align_corners(true)''',
            input_size=(1, 2, 4),
            fullname='interpolate_linear_1d_align_corners',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=None, scale_factor=4., mode='linear', align_corners=True),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::nullopt)
                                .scale_factor(std::vector<double>({4.}))
                                .mode(torch::kLinear)
                                .align_corners(true)''',
            input_size=(1, 2, 4),
            fullname='interpolate_linear_scale_1d_align_corners',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=2, scale_factor=None, mode='nearest'),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({2, 2}))
                                .scale_factor(std::nullopt)
                                .mode(torch::kNearest)''',
            input_size=(1, 128, 1, 1),
            fullname='interpolate_nearest_2d_launch_configs',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest'),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({12, 12}))
                                .scale_factor(std::nullopt)
                                .mode(torch::kNearest)''',
            input_size=(1, 2, 4, 4),
            fullname='interpolate_nearest_2d',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=(12, 16), scale_factor=None, mode='nearest'),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({12, 16}))
                                .scale_factor(std::nullopt)
                                .mode(torch::kNearest)''',
            input_size=(1, 2, 3, 4),
            fullname='interpolate_nearest_tuple_2d',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=None, scale_factor=4., mode='nearest'),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::nullopt)
                                .scale_factor(std::vector<double>({4., 4.}))
                                .mode(torch::kNearest)''',
            input_size=(1, 2, 4, 4),
            fullname='interpolate_nearest_scale_2d',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest'),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({12, 12}))
                                .scale_factor(std::nullopt)
                                .mode(torch::kNearest)''',
            input_size=(0, 2, 4, 4),
            fullname='interpolate_nearest_2d_zero_dim',
            pickle=False,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='bilinear', align_corners=False),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({12, 12}))
                                .scale_factor(std::nullopt)
                                .mode(torch::kBilinear)
                                .align_corners(false)''',
            input_size=(1, 2, 4, 4),
            fullname='interpolate_bilinear_2d',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='bilinear', align_corners=False),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({12, 12}))
                                .scale_factor(std::nullopt)
                                .mode(torch::kBilinear)
                                .align_corners(false)''',
            input_size=(0, 2, 4, 4),
            fullname='interpolate_bilinear_2d_zero_dim',
            pickle=False,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=(4, 6), scale_factor=None,
                                        mode='bilinear', align_corners=False),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({4, 6}))
                                .scale_factor(std::nullopt)
                                .mode(torch::kBilinear)
                                .align_corners(false)''',
            input_size=(1, 2, 2, 3),
            fullname='interpolate_bilinear_tuple_2d',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=None, scale_factor=4.,
                                        mode='bilinear', align_corners=False),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::nullopt)
                                .scale_factor(std::vector<double>({4., 4.}))
                                .mode(torch::kBilinear)
                                .align_corners(false)''',
            input_size=(1, 2, 4, 4),
            fullname='interpolate_bilinear_scale_2d',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2., 2.),
                                        mode='bilinear', align_corners=False),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::nullopt)
                                .scale_factor(std::vector<double>({2., 2.}))
                                .mode(torch::kBilinear)
                                .align_corners(false)''',
            input_size=(1, 2, 4, 4),
            fullname='interpolate_bilinear_scale_tuple_shared_2d',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2., 1.),
                                        mode='bilinear', align_corners=False),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::nullopt)
                                .scale_factor(std::vector<double>({2., 1.}))
                                .mode(torch::kBilinear)
                                .align_corners(false)''',
            input_size=(1, 2, 4, 4),
            fullname='interpolate_bilinear_scale_tuple_skewed_2d',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=(4, 6), scale_factor=None, mode='bilinear', align_corners=True),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({4, 6}))
                                .scale_factor(std::nullopt)
                                .mode(torch::kBilinear)
                                .align_corners(true)''',
            input_size=(1, 2, 4, 4),
            fullname='interpolate_bilinear_tuple_2d_align_corners',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2., 1.),
                                        mode='bilinear', align_corners=True),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::nullopt)
                                .scale_factor(std::vector<double>({2., 1.}))
                                .mode(torch::kBilinear)
                                .align_corners(true)''',
            input_size=(1, 2, 4, 4),
            fullname='interpolate_bilinear_scale_tuple_skewed_2d_align_corners',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='bicubic', align_corners=False),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({12, 12}))
                                .scale_factor(std::nullopt)
                                .mode(torch::kBicubic)
                                .align_corners(false)''',
            input_size=(1, 2, 4, 4),
            fullname='interpolate_bicubic_2d',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='bicubic', align_corners=False),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({12, 12}))
                                .scale_factor(std::nullopt)
                                .mode(torch::kBicubic)
                                .align_corners(false)''',
            input_size=(0, 2, 4, 4),
            fullname='interpolate_bicubic_2d_zero_dim',
            pickle=False,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=(4, 6), scale_factor=None,
                                        mode='bicubic', align_corners=False),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({4, 6}))
                                .scale_factor(std::nullopt)
                                .mode(torch::kBicubic)
                                .align_corners(false)''',
            input_size=(1, 2, 2, 3),
            fullname='interpolate_bicubic_tuple_2d',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=None, scale_factor=4., mode='bicubic', align_corners=False),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::nullopt)
                                .scale_factor(std::vector<double>({4., 4.}))
                                .mode(torch::kBicubic)
                                .align_corners(false)''',
            input_size=(1, 2, 4, 4),
            fullname='interpolate_bicubic_scale_2d',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2., 2.),
                                        mode='bicubic', align_corners=False),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::nullopt)
                                .scale_factor(std::vector<double>({2., 2.}))
                                .mode(torch::kBicubic)
                                .align_corners(false)''',
            input_size=(1, 2, 4, 4),
            fullname='interpolate_bicubic_scale_tuple_shared_2d',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2., 1.),
                                        mode='bicubic', align_corners=False),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::nullopt)
                                .scale_factor(std::vector<double>({2., 1.}))
                                .mode(torch::kBicubic)
                                .align_corners(false)''',
            input_size=(1, 2, 4, 4),
            fullname='interpolate_bicubic_scale_tuple_skewed_2d',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=(4, 6), scale_factor=None, mode='bicubic', align_corners=True),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({4, 6}))
                                .scale_factor(std::nullopt)
                                .mode(torch::kBicubic)
                                .align_corners(true)''',
            input_size=(1, 2, 4, 4),
            fullname='interpolate_bicubic_tuple_2d_align_corners',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2., 1.),
                                        mode='bicubic', align_corners=True),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::nullopt)
                                .scale_factor(std::vector<double>({2., 1.}))
                                .mode(torch::kBicubic)
                                .align_corners(true)''',
            input_size=(1, 2, 4, 4),
            fullname='interpolate_bicubic_scale_tuple_skewed_2d_align_corners',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest'),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({12, 12, 12}))
                                .scale_factor(std::nullopt)
                                .mode(torch::kNearest)''',
            input_size=(1, 2, 4, 4, 4),
            fullname='interpolate_nearest_3d',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest'),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({12, 12, 12}))
                                .scale_factor(std::nullopt)
                                .mode(torch::kNearest)''',
            input_size=(0, 2, 4, 4, 4),
            fullname='interpolate_nearest_3d_zero_dim',
            pickle=False,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=(12, 16, 16), scale_factor=None, mode='nearest'),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({12, 16, 16}))
                                .scale_factor(std::nullopt)
                                .mode(torch::kNearest)''',
            input_size=(1, 2, 3, 4, 4),
            fullname='interpolate_nearest_tuple_3d',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=None, scale_factor=4., mode='nearest'),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::nullopt)
                                .scale_factor(std::vector<double>({4., 4., 4.}))
                                .mode(torch::kNearest)''',
            input_size=(1, 2, 4, 4, 4),
            fullname='interpolate_nearest_scale_3d',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='trilinear', align_corners=False),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({12, 12, 12}))
                                .scale_factor(std::nullopt)
                                .mode(torch::kTrilinear)
                                .align_corners(false)''',
            input_size=(1, 2, 4, 4, 4),
            fullname='interpolate_trilinear_3d',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='trilinear', align_corners=False),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({12, 12, 12}))
                                .scale_factor(std::nullopt)
                                .mode(torch::kTrilinear)
                                .align_corners(false)''',
            input_size=(0, 2, 4, 4, 4),
            fullname='interpolate_trilinear_3d_zero_dim',
            pickle=False,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=(4, 6, 6),
                                        scale_factor=None, mode='trilinear', align_corners=False),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({4, 6, 6}))
                                .scale_factor(std::nullopt)
                                .mode(torch::kTrilinear)
                                .align_corners(false)''',
            input_size=(1, 2, 2, 3, 3),
            fullname='interpolate_trilinear_tuple_3d',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=None, scale_factor=3., mode='trilinear', align_corners=False),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::nullopt)
                                .scale_factor(std::vector<double>({3., 3., 3.}))
                                .mode(torch::kTrilinear)
                                .align_corners(false)''',
            input_size=(1, 2, 3, 4, 5),
            fullname='interpolate_trilinear_scale_3d',
            # See https://github.com/pytorch/pytorch/issues/5006
            precision=3e-4,
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=(4, 6, 6), scale_factor=None,
                                        mode='trilinear', align_corners=True),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({4, 6, 6}))
                                .scale_factor(std::nullopt)
                                .mode(torch::kTrilinear)
                                .align_corners(true)''',
            input_size=(1, 2, 2, 3, 3),
            fullname='interpolate_trilinear_tuple_3d_align_corners',
            pickle=False,
            default_dtype=torch.double
        ),
        dict(
            constructor=wrap_functional(F.interpolate, size=None, scale_factor=3., mode='trilinear', align_corners=True),
            cpp_options_args='''F::InterpolateFuncOptions()
                                .size(std::nullopt)
                                .scale_factor(std::vector<double>({3., 3., 3.}))
                                .mode(torch::kTrilinear)
                                .align_corners(true)''',
            input_size=(1, 2, 3, 4, 4),
            fullname='interpolate_trilinear_scale_3d_align_corners',
            # See https://github.com/pytorch/pytorch/issues/5006
            precision=3e-4,
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.softmax, dim=-1),
            cpp_options_args='F::SoftmaxFuncOptions(-1)',
            input_size=(2, 128),  # trigger the last-dim algo in CUDA
            fullname='softmax_lastdim',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.softmax, dim=1, dtype=torch.float64),
            cpp_options_args='F::SoftmaxFuncOptions(1).dtype(torch::kFloat64)',
            input_size=(2, 128),
            fullname='softmax_lastdim_dtype',
            pickle=False,
            test_cuda=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.softmax, dim=1),
            cpp_options_args='F::SoftmaxFuncOptions(1)',
            input_size=(2, 128, 2, 2),  # trigger special case of spatial CUDA algo
            fullname='softmax_spatial_special',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.softmax, dim=1),
            cpp_options_args='F::SoftmaxFuncOptions(1)',
            input_size=(2, 2, 4, 4),  # regular spatial algorithm
            fullname='softmax_spatial',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.softmax, dim=1, dtype=torch.float64),
            cpp_options_args='F::SoftmaxFuncOptions(1).dtype(torch::kFloat64)',
            input_size=(2, 2, 4, 4),  # regular spatial algorithm
            fullname='softmax_spatial_dtype',
            pickle=False,
            test_cuda=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.softmax, dim=0),
            cpp_options_args='F::SoftmaxFuncOptions(0)',
            input_size=(2, 3, 4, 5),
            fullname='softmax_functional_dim0',
            test_cuda=False,
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.softmax, dim=3),
            cpp_options_args='F::SoftmaxFuncOptions(3)',
            input_size=(2, 3, 4, 5),
            fullname='softmax_functional_dim3',
            test_cuda=False,
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.softmax, dim=-1),
            cpp_options_args='F::SoftmaxFuncOptions(-1)',
            input_size=(),
            fullname='softmax_functional_scalar',
            test_cuda=False,
            pickle=False,
        ),
        dict(
            constructor=wrap_functional(F.log_softmax, dim=-1),
            cpp_options_args='F::LogSoftmaxFuncOptions(-1)',
            input_size=(2, 128),  # trigger the last-dim algo in CUDA
            fullname='log_softmax_lastdim',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.log_softmax, dim=1),
            cpp_options_args='F::LogSoftmaxFuncOptions(1)',
            input_size=(2, 128, 2, 2),  # trigger special case of spatial CUDA algo
            fullname='log_softmax_spatial_special',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.log_softmax, dim=1),
            cpp_options_args='F::LogSoftmaxFuncOptions(1)',
            input_size=(2, 2, 4, 4),  # regular spatial algorithm
            fullname='log_softmax_spatial',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.log_softmax, dim=0),
            cpp_options_args='F::LogSoftmaxFuncOptions(0)',
            input_size=(2, 3, 4, 5),
            fullname='log_softmax_dim0',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.log_softmax, dim=3),
            cpp_options_args='F::LogSoftmaxFuncOptions(3)',
            input_size=(2, 3, 4, 5),
            fullname='log_softmax_dim3',
            pickle=False,
            default_dtype=torch.double,
        ),
        dict(
            constructor=wrap_functional(F.log_softmax, dim=0),
            cpp_options_args='F::LogSoftmaxFuncOptions(0)',
            input_size=(),
            fullname='log_softmax_scalar',
            pickle=False,
        ),
        dict(
            fullname='Unfold',
            constructor=lambda: nn.Unfold((2, 2), (1, 1), (0, 0), (1, 1)),
            cpp_constructor_args='torch::nn::UnfoldOptions({2, 2}).dilation({1, 1}).padding({0, 0}).stride({1, 1})',
            input_size=(2, 4, 3, 3),
            check_gradgrad=False,
            test_cuda=True,
            default_dtype=torch.double,
        ),
        dict(
            fullname='Fold',
            constructor=lambda: nn.Fold((3, 3), (2, 2), (1, 1), (0, 0), (1, 1)),
            cpp_constructor_args='torch::nn::FoldOptions({3, 3}, {2, 2}).dilation({1, 1}).padding({0, 0}).stride({1, 1})',
            input_size=(2, 16, 4),
            check_gradgrad=False,
            test_cuda=True,
            default_dtype=torch.double,
        ),
        dict(
            fullname='Fold_no_batch_dim_input',
            constructor=lambda: nn.Fold((3, 3), (2, 2), (1, 1), (0, 0), (1, 1)),
            cpp_constructor_args='torch::nn::FoldOptions({3, 3}, {2, 2}).dilation({1, 1}).padding({0, 0}).stride({1, 1})',
            input_size=(16, 4),
            check_gradgrad=False,
            ref=single_batch_reference_fn,
            test_cuda=True,
            default_dtype=torch.double,
        ),
        dict(
            fullname='Unfold_int_input',
            constructor=lambda: nn.Unfold(2, 1, 0, 1),
            cpp_constructor_args='torch::nn::UnfoldOptions(2).dilation(1).padding(0).stride(1)',
            input_size=(2, 4, 3, 3),
            check_gradgrad=False,
            test_cuda=True,
            default_dtype=torch.double,
        ),
        dict(
            fullname='Fold_int_input',
            constructor=lambda: nn.Fold(3, 2, 1, 0, 1),
            cpp_constructor_args='torch::nn::FoldOptions(3, 2).dilation(1).padding(0).stride(1)',
            input_size=(2, 16, 4),
            check_gradgrad=False,
            test_cuda=True,
            default_dtype=torch.double,
        ),
        dict(
            fullname='Fold_no_batch_dim_int_input',
            constructor=lambda: nn.Fold(3, 2, 1, 0, 1),
            cpp_constructor_args='torch::nn::FoldOptions(3, 2).dilation(1).padding(0).stride(1)',
            input_size=(16, 4),
            ref=single_batch_reference_fn,
            check_gradgrad=False,
            test_cuda=True,
            default_dtype=torch.double,
        ),
        dict(
            module_name='RReLU',
            constructor_args=(0.1, 0.9),
            cpp_constructor_args='torch::nn::RReLUOptions().lower(0.1).upper(0.9)',
            input_size=(),
            desc='with_up_down_scalar',
            test_cuda=False,
            default_dtype=torch.double,
        ),
        dict(
            module_name='PairwiseDistance',
            input_fn=lambda: (torch.randn(10, 8), torch.randn(10, 8)),
            default_dtype=torch.double,
        ),
        dict(
            module_name='PairwiseDistance',
            input_fn=lambda: (torch.randn(10, 1), torch.randn(10, 8)),
            desc='broadcast_lhs',
            default_dtype=torch.double,
        ),
        dict(
            module_name='PairwiseDistance',
            input_fn=lambda: (torch.randn(10, 8), torch.randn(1, 8)),
            desc='broadcast_rhs',
            default_dtype=torch.double,
        ),
        dict(
            module_name='PairwiseDistance',
            constructor_args=(1.5, 1e-05, True),
            cpp_constructor_args='torch::nn::PairwiseDistanceOptions().p(1.5).eps(1e-05).keepdim(true)',
            input_fn=lambda: (torch.randn(10, 8), torch.randn(10, 8)),
            desc='with_non_default_args',
            default_dtype=torch.double,
        ),
        dict(
            module_name='PairwiseDistance',
            input_fn=lambda: (torch.randn(8), torch.randn(8)),
            reference_fn=single_batch_reference_fn,
            desc='no_batch_dim',
            default_dtype=torch.double,
        ),
        dict(
            module_name='TransformerEncoderLayer',
            constructor_args=(4, 2, 16, 0.0),
            cpp_constructor_args='''torch::nn::TransformerEncoderLayerOptions(4, 2)
                                    .dim_feedforward(16)
                                    .dropout(0.0)''',
            input_size=(2, 3, 4),
            desc='relu_activation',
            with_tf32=True,
            tf32_precision=0.1,
            # TODO(#50743): figure out the error
            # RuntimeError: The size of tensor a (6) must match the size of tensor b (4)
            # at non-singleton dimension 2
            check_batched_grad=False,
            check_gradgrad=False,
            default_dtype=torch.double,
        ),
        dict(
            module_name='TransformerEncoderLayer',
            constructor_args=(4, 2, 8, 0.0, F.gelu),
            cpp_constructor_args='''torch::nn::TransformerEncoderLayerOptions(4, 2)
                                    .dim_feedforward(8)
                                    .dropout(0.0)
                                    .activation(torch::kGELU)''',
            input_size=(2, 3, 4),
            check_gradgrad=False,
            desc='gelu_activation',
            with_tf32=True,
            tf32_precision=0.08 if SM90OrLater else 0.05,
            default_dtype=torch.double,
        ),
        dict(
            module_name='TransformerDecoderLayer',
            constructor_args=(4, 2, 8, 0.0),
            cpp_constructor_args='''torch::nn::TransformerDecoderLayerOptions(4, 2)
                                    .dim_feedforward(8)
                                    .dropout(0.0)''',
            input_fn=lambda: (torch.rand(3, 3, 4), torch.rand(2, 3, 4)),
            check_gradgrad=False,
            desc='relu_activation',
            with_tf32=True,
            tf32_precision=0.05,
            default_dtype=torch.double,
        ),
        dict(
            module_name='TransformerDecoderLayer',
            constructor_args=(4, 2, 8, 0.0, F.gelu),
            cpp_constructor_args='''torch::nn::TransformerDecoderLayerOptions(4, 2)
                                    .dim_feedforward(8)
                                    .dropout(0.0)
                                    .activation(torch::kGELU)''',
            input_fn=lambda: (torch.rand(3, 3, 4), torch.rand(2, 3, 4)),
            check_gradgrad=False,
            desc='gelu_activation',
            with_tf32=True,
            tf32_precision=0.05,
            default_dtype=torch.double,
        ),
        dict(
            module_name='Transformer',
            constructor_args=(4, 2, 2, 2, 8, 0.0, F.relu),
            cpp_constructor_args='''torch::nn::TransformerOptions()
                                    .d_model(4)
                                    .nhead(2)
                                    .num_encoder_layers(2)
                                    .num_decoder_layers(2)
                                    .dim_feedforward(8)
                                    .dropout(0.0)
                                    .activation(torch::kReLU)''',
            input_fn=lambda: (torch.rand(3, 3, 4), torch.rand(2, 3, 4), torch.rand(3, 3)),
            check_gradgrad=False,
            desc='multilayer_coder',
            with_tf32=True,
            tf32_precision=0.05 if SM90OrLater else 0.03,
            default_dtype=torch.double,
        ),
        dict(
            module_name='Linear',
            constructor_args=(3, 5),
            cpp_constructor_args='torch::nn::LinearOptions(3, 5)',
            input_fn=lambda: torch.rand(3),
            reference_fn=lambda i, p, _: torch.mm(i.view(1, -1), p[0].t()).view(-1) + p[1],
            desc="no_batch_dim",
            with_tf32=True,
            tf32_precision=0.005,
            default_dtype=torch.double,
        ),
        dict(
            module_name='Flatten',
            cpp_constructor_args='torch::nn::FlattenOptions().start_dim(-3).end_dim(-1)',
            constructor_args=(-3, -1),
            input_size=(3, 4, 5),
            reference_fn=single_batch_reference_fn,
            desc="no_batch_dim",
            default_dtype=torch.double,
        ),
        dict(
            module_name='Unflatten',
            cpp_constructor_args='torch::nn::UnflattenOptions(-2, {2, 2})',
            constructor_args=(-2, torch.Size([2, 2])),
            input_size=(3, 4, 5),
            reference_fn=single_batch_reference_fn,
            desc="no_batch_dim",
            default_dtype=torch.double,
        ),
        dict(
            module_name='LayerNorm',
            constructor_args=([56, 56, 56], 1e-5, False),
            cpp_constructor_args='torch::nn::LayerNormOptions({56, 56, 56}).eps(1e-5).elementwise_affine(false)',
            input_size=(4, 56, 56, 56),
            cudnn=True,
            check_eval=True,
            gradcheck_fast_mode=True,
            check_half=True,
            desc='3d_no_affine_large_feature',
        ),
    ]

    # add conv padding mode tests:
    for padding_mode, cpp_padding_mode in zip(
            ['reflect', 'circular', 'replicate', 'zeros'],
            ['torch::kReflect', 'torch::kCircular', 'torch::kReplicate', 'torch::kZeros'], strict=True):
        # conv signature:
        #     in_channels, out_channels, kernel_size, stride=1,
        #     padding=0, dilation=1, groups=1,
        #     bias=True, padding_mode='zeros'
        for d in (1, 2, 3):
            if d == 3 and padding_mode == 'reflect':
                # FIXME: remove after implementing reflection pad 3d
                #        https://github.com/pytorch/pytorch/issues/27655
                continue
            padding = tuple(range(1, d + 1))
            cpp_padding = '{' + ', '.join(map(str, padding)) + '}'
            input_size = (2, 2) + (4,) * d
            output_size = (2, 3) + tuple(p + 1 for p in padding)  # simplified from `(4 + 2 * p - 3) // 2 + 1`
            new_module_tests.append(
                dict(
                    module_name=f'Conv{d}d',
                    constructor_args=(2, 3, 3, 2, padding, 1, 1, True, padding_mode),
                    cpp_constructor_args=f'''torch::nn::Conv{d}dOptions(2, 3, 3)
                                            .stride(2)
                                            .padding({cpp_padding})
                                            .dilation(1)
                                            .groups(1)
                                            .bias(true)
                                            .padding_mode({cpp_padding_mode})''',
                    input_size=input_size,
                    output_size=output_size,
                    cudnn=True,
                    desc=f'{padding_mode}_stride2_pad2',
                    with_tf32=True,
                    tf32_precision=0.05,
                    default_dtype=torch.double,
                ),
            )

    # Check that non linear activations work with no batch dimensions
    non_linear_activations_no_batch = [
        'ELU', 'Hardshrink', 'Hardsigmoid', 'Hardtanh', 'Hardswish', 'LeakyReLU',
        'LogSigmoid', 'PReLU', 'ReLU', 'ReLU6', 'RReLU', 'SELU', 'CELU', 'GELU', 'GLU',
        'Sigmoid', 'SiLU', 'Mish', 'Softplus', 'Softshrink', 'Softsign', 'Tanh',
        'Tanhshrink', 'Threshold'
    ]
    non_linear_activations_extra_info: dict[str, dict] = {
        'CELU': {'constructor_args': (2.,), 'default_dtype': torch.double},
        'Threshold': {'constructor_args': (2., 1.)},
        'Hardsigmoid': {'check_gradgrad': False, 'check_jit': False, 'default_dtype': torch.double},
        'Hardswish': {'check_gradgrad': False, 'check_jit': False, 'default_dtype': torch.double},
        # For RRelu, test that compare CPU and GPU results fail because RNG
        # is different between CPU and GPU
        'RReLU': {'test_cuda': False, 'default_dtype': torch.double},
        'ELU': {'default_dtype': torch.double},
        'GELU': {'default_dtype': torch.double},
        'GLU': {'default_dtype': torch.double},
        'Hardshrink': {'default_dtype': torch.double},
        'Hardtanh': {'default_dtype': torch.double},
        'LeakyReLU': {'default_dtype': torch.double},
        'LogSigmoid': {'default_dtype': torch.double},
        'Mish': {'default_dtype': torch.double},
        'PReLU': {'default_dtype': torch.double},
        'ReLU6': {'default_dtype': torch.double},
        'ReLU': {'default_dtype': torch.double},
        'SELU': {'default_dtype': torch.double},
        'SiLU': {'default_dtype': torch.double},
        'Sigmoid': {'default_dtype': torch.double},
        'Softplus': {'default_dtype': torch.double},
        'Softshrink': {'default_dtype': torch.double},
        'Softsign': {'default_dtype': torch.double},
        'Tanh': {'default_dtype': torch.double},
        'Tanhshrink': {'default_dtype': torch.double},
    }
    for non_linear_activation in non_linear_activations_no_batch:
        activation_test_info = dict(
            module_name=non_linear_activation,
            input_size=(4,),
            reference_fn=single_batch_reference_fn,
            desc='no_batch_dim',
            test_cpp_api_parity=False,
        )
        extra_info = non_linear_activations_extra_info.get(non_linear_activation, {})
        activation_test_info.update(extra_info)
        new_module_tests.append(activation_test_info)


    return new_module_tests


def kldivloss_reference(input, target, reduction='mean', log_target=False):
    # [FUNCTION DOC] Reference KL divergence between prediction (log probabilities) and target,
    # elementwise: with log_target the target is exponentiated times the difference of target
    # and input, otherwise target times the difference of log target and input. Reduction mean,
    # sum or batchmean (sum over batch size, only for non-scalar results); anything else returns
    # the elementwise tensor.
    if log_target:
        result = torch.exp(target) * (target - input)
    else:
        result = target * (target.log() - input)
    if reduction == 'mean':
        return result.mean()
    elif reduction == 'sum':
        return result.sum()
    elif reduction == 'batchmean' and result.dim() != 0:
        return result.sum() / result.size(0)
    return result


def nlllossNd_reference(input, target, weight=None, ignore_index=-100,
                        reduction='mean'):
    # [FUNCTION DOC] Reference N-d negative log likelihood loss written with explicit loops over
    # every output position. Input must have at least 3 dimensions (batch, class, spatial...).
    # Per position picks the input value at the target class, multiplies by the class weight
    # (zero when the target equals ignore_index) and negates. Reduction mean divides the sum by
    # the total weight, sum sums, otherwise returns the per-position tensor. Raises an error if
    # input has fewer than 3 dimensions.
    if input.dim() < 3:
        raise AssertionError(f"Expected input.dim() >= 3, got {input.dim()}")
    N = input.size(0)
    C = input.size(1)
    out_size = (N,) + input.size()[2:]
    output = torch.zeros(out_size).type_as(input)

    if weight is None:
        weight = torch.ones(C).type_as(input)
    total_weight = 0
    for tup in product(*[range(size) for size in out_size]):
        t_nx = target[tup]
        norm = 0. if ignore_index == t_nx else weight[t_nx].item()
        input_index = list(tup)
        input_index.insert(1, t_nx)
        output[tup] = -input[tuple(input_index)] * norm
        total_weight += norm

    if reduction == 'mean':
        return output.sum() / total_weight
    elif reduction == 'sum':
        return output.sum()
    return output


def cross_entropy_loss_prob_target_reference(input, target, weight=None, reduction='mean',
                                             label_smoothing=0.0):
    # [FUNCTION DOC] Reference cross entropy for class-probability targets: log-softmax over
    # dimension 1, optional label smoothing mixing the target with a uniform distribution,
    # per-class weights, sum over classes, then mean/sum/none reduction. Raises an error if
    # input has fewer than 2 dimensions or label_smoothing exceeds 1.
    if input.dim() < 2:
        raise AssertionError(f"Expected input.dim() >= 2, got {input.dim()}")

    input = torch.log_softmax(input, 1)
    C = input.size(1)
    if weight is None:
        weight = torch.ones(C).type_as(input)
    weight = weight.view(1, C, *(1 for _ in input.shape[2:]))

    if label_smoothing > 0.0:
        if label_smoothing > 1.0:
            raise AssertionError(f"Expected label_smoothing <= 1.0, got {label_smoothing}")
        target = (target * (1 - label_smoothing) + label_smoothing / C)

    output = -(input * target * weight).sum(dim=1)
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output


def cross_entropy_loss_indices_target_reference(input, target, weight=None, ignore_index=-100,
                                                reduction='mean', label_smoothing=0.0):
    # [FUNCTION DOC] Reference cross entropy for class-index targets: log-softmax over dimension
    # 1 followed by nll_loss with the given weight, ignore_index and reduction. If label
    # smoothing is zero returns that value; otherwise adds a smoothing term (negative sum of
    # weighted log-probabilities over classes with ignored positions zeroed, reduced the same
    # way) and returns the mix of the two scaled by one minus smoothing and smoothing over class
    # count. Raises an error unless smoothing is in the half-open range above 0 up to 1.
    log_softmax_input = torch.log_softmax(input, 1)
    nllloss = F.nll_loss(
        log_softmax_input,
        target,
        weight,
        ignore_index=ignore_index,
        reduction=reduction)

    if label_smoothing == 0.0:
        return nllloss

    if not (0.0 < label_smoothing <= 1.0):
        raise AssertionError(f"Expected 0.0 < label_smoothing <= 1.0, got {label_smoothing}")

    input = torch.log_softmax(input, 1)
    C = input.size(1)
    if weight is not None:
        input = input * weight.view(1, C, *(1 for _ in input.shape[2:]))

    smooth_loss = -torch.sum(input, 1)

    ignore_mask = target == ignore_index
    smooth_loss.masked_fill_(ignore_mask, 0.0)

    if reduction == 'mean':
        if weight is not None:
            # TODO: This code can path can be removed if #61309 is resolved
            # loss is normalized by the weights to be consistent with nll_loss_nd
            ret = torch.sum(smooth_loss) / weight.gather(0, target.masked_select(ignore_mask.logical_not()).flatten()).sum()
        else:
            ret = torch.mean(smooth_loss.masked_select(ignore_mask.logical_not()))
    elif reduction == 'sum':
        ret = torch.sum(smooth_loss)
    else:
        ret = smooth_loss

    return (1 - label_smoothing) * nllloss + ret * (label_smoothing / C)


def cross_entropy_loss_reference(input, target, weight=None, ignore_index=-100, reduction='mean',
                                 label_smoothing=0.0):
    # [FUNCTION DOC] Dispatches to the probability-target reference when input and target have
    # the same shape, otherwise to the class-index reference; forwards weight, reduction, label
    # smoothing and (index case only) ignore_index.
    if input.shape == target.shape:
        return cross_entropy_loss_prob_target_reference(
            input,
            target,
            weight=weight,
            reduction=reduction,
            label_smoothing=label_smoothing)
    else:
        return cross_entropy_loss_indices_target_reference(
            input, target, weight=weight, reduction=reduction,
            ignore_index=ignore_index, label_smoothing=label_smoothing
        )


def linear_cross_entropy_loss_reference(input, linear_weight, target,
                                        linear_bias=None,
                                        weight=None,
                                        ignore_index=None,
                                        reduction='mean',
                                        label_smoothing=0.0):
    # [FUNCTION DOC] Reference for fused linear plus cross entropy: applies a linear map (weight
    # flattened to 2-d, optional flattened bias) to the input, reshapes the result to batch
    # dimensions, class dimension and any extra trailing output dimensions, then calls the
    # standard cross entropy with the given weight, reduction and label smoothing. An
    # ignore_index of None means the default -100.
    num_classes = linear_weight.shape[0]
    out_features = linear_weight.shape[1:-1]
    in_features = linear_weight.shape[-1]
    num_batches = input.shape[:-1]
    logits = F.linear(
        input,
        linear_weight.reshape((-1, in_features)),
        linear_bias.reshape(-1) if linear_bias is not None else None,
    ).reshape((*num_batches, num_classes, *out_features))
    ignore_index = ignore_index if ignore_index is not None else -100
    return F.cross_entropy(
        logits, target, weight=weight,
        reduction=reduction, ignore_index=ignore_index,
        label_smoothing=label_smoothing
    )


def nllloss_reference(input, target, weight=None, ignore_index=-100,
                      reduction='mean'):
    # [FUNCTION DOC] Reference negative log likelihood loss for 2-d input (samples by classes)
    # with 1-d targets, computed per sample with a Python loop. Samples whose target equals
    # ignore_index contribute zero loss and zero weight. Reduction mean divides summed loss by
    # summed weight, sum sums, otherwise returns the per-sample tensor.

    def nll_loss_helper(input, target, weight, ignore_index):
        # [FUNCTION DOC] HELPER: Returns a pair (loss, weight) for one sample: (0, 0) if the
        # target is ignored, otherwise the negated input value at the target class times its
        # weight and that weight.
        if target == ignore_index:
            return (0, 0)
        norm = 1 if weight is None else weight[target]
        result = -input[target] * norm
        return (result, norm)

    losses_and_weights = [nll_loss_helper(i, t, weight, ignore_index)
                          for i, t in zip(input, target, strict=True)]
    losses, weights = zip(*losses_and_weights, strict=True)
    losses_tensor = input.new_tensor(losses)
    if reduction == 'mean':
        return sum(losses_tensor) / sum(weights)
    elif reduction == 'sum':
        return sum(losses_tensor)
    else:
        return losses_tensor


def smoothl1loss_reference(input, target, reduction='mean', beta=1.0):
    # [FUNCTION DOC] Reference smooth L1 loss: for absolute difference below beta uses half the
    # squared difference over beta, otherwise absolute difference minus half beta; beta equal to
    # zero degenerates to absolute difference. Reduction mean, sum or none.
    abs_diff = (input - target).abs()
    ge_beta_mask = (abs_diff >= beta).type_as(abs_diff)
    lt_beta_mask = (abs_diff < beta).type_as(abs_diff)
    # when beta <= 0 we should just use l1_loss
    if beta == 0:
        output = abs_diff
    else:
        output = ge_beta_mask * (abs_diff - 0.5 * beta) + lt_beta_mask * 0.5 * (abs_diff ** 2) / beta
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output


def huberloss_reference(input, target, reduction='mean', delta=1.0):
    # [FUNCTION DOC] Reference Huber loss: half the squared difference where the absolute
    # difference is below delta, otherwise delta times (absolute difference minus half delta).
    # Reduction mean, sum or none.
    abs_diff = (input - target).abs()
    ge_delta_mask = (abs_diff >= delta)
    lt_delta_mask = (abs_diff < delta)
    output = ge_delta_mask * delta * (abs_diff - 0.5 * delta) + lt_delta_mask * 0.5 * (abs_diff ** 2)
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output


def _multilabelmarginloss_reference(input, target):
    # [FUNCTION DOC] HELPER: Reference multilabel margin loss for a single sample: collects
    # target indices up to the first negative entry, then sums the hinge terms (one minus
    # target-class score plus other-class score) over every pair of target class and non-target
    # class. Returns the unnormalized sum.
    targets = []
    for target_index in target:
        if target_index < 0:
            break
        targets.append(target_index)

    sum = 0
    for target_index in targets:
        for i in range(len(input)):
            if i not in targets:
                sum += max(0, 1 - input[target_index] + input[i])

    return sum


def multilabelmarginloss_reference(input, target, reduction='mean'):
    # [FUNCTION DOC] Reference multilabel margin loss supporting 0-d/1-d inputs by promoting to
    # 2-d, summing the per-sample loss, dividing by the class count, and applying mean, sum or
    # none reduction (squeezing back for low-dimensional input with no reduction). Raises an
    # error if a low-dimensional input has a target with 2 or more dimensions.
    # make everything 2-dimensional
    input_dim = input.dim()
    if input.dim() < 2:
        if target.dim() >= 2:
            raise AssertionError(f"Expected target.dim() < 2, got {target.dim()}")
        input = input.unsqueeze(0) if input.dim() == 1 else input.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0) if target.dim() == 1 else target.unsqueeze(0).unsqueeze(0)

    n = input.size(0)
    dim = input.size(1)
    output = input.new(n).zero_()
    for i in range(n):
        output[i] = _multilabelmarginloss_reference(input[i], target[i])

    if reduction == 'mean':
        return output.mean() / dim
    elif reduction == 'sum':
        return output.sum() / dim
    elif input_dim < 2:
        # we know we have (1, C) X (1, C) -> (1,), so squeeze will get us
        # back to correct dimensionality
        return output.squeeze() / dim
    else:
        return output / dim


def hingeembeddingloss_reference(input, target, margin=1.0, reduction='mean'):
    # [FUNCTION DOC] Reference hinge embedding loss: input where target is 1, otherwise the
    # margin minus input clamped at zero. Reduction mean, sum or none.
    margin_clamp = (margin - input).clamp(min=0).type_as(input)
    output = torch.where(target == 1, input, margin_clamp)

    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output


def softmarginloss_reference(input, target, reduction='mean'):
    # [FUNCTION DOC] Reference soft margin loss: log of one plus exp of negative input times
    # target. Reduction mean, sum or none.
    output = (1 + (-input * target).exp()).log()

    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output


def _multimarginloss_reference(input, target_idx, p, margin, weight):
    # [FUNCTION DOC] HELPER: Reference multi margin loss for one sample: sums, over classes
    # other than the target index, the target-class weight times the margin-hinge raised to
    # power p. Uses unit weights if no weight is given.
    if weight is None:
        weight = input.new(len(input)).fill_(1)

    output = 0
    for i in range(len(input)):
        if i != target_idx:
            output += weight[target_idx] * (max(0, (margin - input[target_idx] + input[i])) ** p)
    return output


def multimarginloss_reference(input, target, p=1, margin=1, weight=None, reduction='mean'):
    # [FUNCTION DOC] Reference multi margin loss: promotes low-dimensional input to 2-d and a
    # 0-d target to 1-d, computes the per-sample loss, divides by the class count, and applies
    # mean, sum or none reduction (squeezing the result back when the target was 0-d).
    if input.dim() < 2:
        input = input.unsqueeze(0) if input.dim() == 1 else input.unsqueeze(0).unsqueeze(0)

    target_dim = target.dim()
    if target.dim() == 0:
        target = target.unsqueeze(0)

    n = input.size(0)
    dim = input.size(1)
    output = input.new(n)
    for x in range(n):
        output[x] = _multimarginloss_reference(input[x], target[x], p, margin, weight)

    if reduction == 'mean':
        return output.mean() / dim
    elif reduction == 'sum':
        return output.sum() / dim
    elif target_dim == 0:
        return output.squeeze(0) / dim
    return output / dim


def cosineembeddingloss_reference(input1, input2, target, margin=0, reduction='mean'):
    # [FUNCTION DOC] Reference cosine embedding loss: one minus cosine similarity where target
    # is 1, otherwise cosine similarity minus margin clamped at zero. Cosine is computed per row
    # with a small epsilon added to the norms. Reduction mean, sum or none.
    def _cos(a, b):
        # [FUNCTION DOC] HELPER: Computes row-wise cosine similarity of two 2-d tensors with a
        # loop over rows; returns a 1-d tensor.
        cos = a.new(a.size(0))
        for i in range(a.size(0)):
            cos[i] = (a[i] * b[i]).sum() / ((((a[i] * a[i]).sum() + 1e-12) * ((b[i] * b[i]).sum() + 1e-12)) ** 0.5)
        return cos

    output = torch.where(target == 1, 1 - _cos(input1, input2), (_cos(input1, input2) - margin).clamp(min=0))

    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output


def tripletmarginloss_reference(anchor, positive, negative, margin=1.0, p=2, eps=1e-6, swap=False,
                                reduction='mean'):
    # [FUNCTION DOC] Reference triplet margin loss: pairwise p-norm distances anchor-positive
    # and anchor-negative (with swap, the negative distance becomes the smaller of that and the
    # positive-negative distance), then margin plus positive distance minus negative distance
    # clamped at zero. Reduction mean, sum or none.
    d_p = torch.pairwise_distance(anchor, positive, p, eps)
    d_n = torch.pairwise_distance(anchor, negative, p, eps)
    if swap:
        d_s = torch.pairwise_distance(positive, negative, p, eps)
        d_n = torch.min(d_n, d_s)

    output = torch.clamp(margin + d_p - d_n, min=0.0)
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output


def marginrankingloss_reference(input1, input2, target, margin=0, reduction='mean'):
    # [FUNCTION DOC] Reference margin ranking loss: negative target times (input1 minus input2)
    # plus margin, clamped at zero. Reduction mean, sum or none.
    output = (-target * (input1 - input2) + margin).clamp(min=0)
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output


# this directly follows Graves et al.'s paper, in contrast to the production implementation, it does not use log-space
def ctcloss_reference(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean'):
    # [FUNCTION DOC] Reference CTC loss following the original paper with the forward (alpha)
    # recursion in probability space, computed in double precision per batch element. Targets
    # may be padded 2-d or concatenated 1-d; input and target lengths are converted to long
    # tensors. Reduction mean divides each loss by its target length before averaging, sum sums,
    # none returns the vector. Result is cast back to the original dtype of log_probs.
    input_lengths = torch.as_tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.as_tensor(target_lengths, dtype=torch.long)
    dt = log_probs.dtype
    log_probs = log_probs.double()  # we need the accuracy as we are not in logspace
    targets = targets.long()
    cum_target_lengths = target_lengths.cumsum(0)
    losses = []
    for i in range(log_probs.size(1)):
        input_length = input_lengths[i].item()
        target_length = target_lengths[i].item()
        cum_target_length = cum_target_lengths[i].item()
        targets_prime = targets.new_full((2 * target_length + 1,), blank)
        if targets.dim() == 2:
            targets_prime[1::2] = targets[i, :target_length]
        else:
            targets_prime[1::2] = targets[cum_target_length - target_length:cum_target_length]
        probs = log_probs[:input_length, i].exp()
        alpha = log_probs.new_zeros((target_length * 2 + 1,))
        alpha[0] = probs[0, blank]
        alpha[1] = probs[0, targets_prime[1]]
        mask_third = (targets_prime[:-2] != targets_prime[2:])
        for t in range(1, input_length):
            alpha_next = alpha.clone()
            alpha_next[1:] += alpha[:-1]
            alpha_next[2:] += torch.where(mask_third, alpha[:-2], alpha.new_zeros(1))
            alpha = probs[t, targets_prime] * alpha_next
        losses.append(-alpha[-2:].sum().log()[None])
    output = torch.cat(losses, 0)
    if reduction == 'mean':
        output = (output / target_lengths.to(dtype=output.dtype, device=output.device)).mean()
    elif reduction == 'sum':
        output = output.sum()
    output = output.to(dt)
    return output


loss_reference_fns: dict['str', Callable] = {
    'KLDivLoss': kldivloss_reference,
    'KLDivLoss_log_target': partial(kldivloss_reference, log_target=True),
    'NLLLoss': nllloss_reference,
    'NLLLossNd': nlllossNd_reference,
    'SmoothL1Loss': smoothl1loss_reference,
    'HuberLoss': huberloss_reference,
    'MultiLabelMarginLoss': multilabelmarginloss_reference,
    'HingeEmbeddingLoss': hingeembeddingloss_reference,
    'SoftMarginLoss': softmarginloss_reference,
    'MultiMarginLoss': multimarginloss_reference,
    'CosineEmbeddingLoss': cosineembeddingloss_reference,
    'TripletMarginLoss': tripletmarginloss_reference,
    'MarginRankingLoss': marginrankingloss_reference,
    'CTCLoss': ctcloss_reference,
    'CrossEntropyLoss': cross_entropy_loss_reference,
    'LinearCrossEntropyLoss': linear_cross_entropy_loss_reference,
}


criterion_tests = []


def single_batch_reference_criterion_fn(*args):
    # [FUNCTION DOC] Reference function for criterion tests without a batch dimension. The last
    # positional argument is the criterion itself and the others are inputs/targets (possibly
    # nested lists or tuples); each is given a leading batch dimension of size one and flattened
    # into the criterion call. For reduction none the batch dimension is removed from the
    # output, otherwise the scalar is returned.
    """Reference function for criterion supporting no batch dimensions.

    The criterion is passed the input and target in batched form with a single item.
    The output is squeezed to compare with the no-batch input.
    """
    criterion = args[-1]

    def unsqueeze_inp(inp):
        # [FUNCTION DOC] HELPER: Adds a leading dimension of size one to a tensor, or to every
        # tensor of a list or tuple (returned as a list).
        if isinstance(inp, (list, tuple)):
            return [t.unsqueeze(0) for t in inp]
        return inp.unsqueeze(0)

    def flatten(xs):
        # [FUNCTION DOC] HELPER: Recursively flattens nested lists and tuples into one flat list
        # of leaves.
        result = []
        if isinstance(xs, (list, tuple)):
            for x in xs:
                result.extend(flatten(x))
        else:
            result.append(xs)
        return result

    single_batch_input_args = flatten([unsqueeze_inp(input) for input in args[:-1]])

    output = criterion(*single_batch_input_args)
    reduction = get_reduction(criterion)

    if reduction == 'none':
        return output.squeeze(0)
    # reduction is 'sum' or 'mean' which results in a scalar
    return output


# Check that regression criterion work with no batch dimensions
regression_criterion_no_batch = [
    'L1Loss', 'MSELoss', 'PoissonNLLLoss', 'HuberLoss', 'SmoothL1Loss'
]
reductions = ['none', 'mean', 'sum']
for name, reduction in product(regression_criterion_no_batch, reductions):
    regression_test_info = dict(
        fullname=f"{name}_no_batch_dim_{reduction}",
        constructor=lambda *args, name=name: getattr(nn, name)(reduction=reduction),
        input_size=(3, ),
        target_size=(3, ),
        reference_fn=single_batch_reference_criterion_fn,
        test_cpp_api_parity=False,
        default_dtype=torch.double,
    )
    criterion_tests.append(regression_test_info)


for reduction in reductions:
    regression_test_info = dict(
        fullname=f"KLDivLoss_no_batch_dim_{reduction}",
        constructor=lambda: nn.KLDivLoss(reduction=reduction),
        input_fn=lambda: torch.rand((3,)).log(),
        target_fn=lambda: torch.rand((3,)),
        reference_fn=single_batch_reference_criterion_fn,
        test_cpp_api_parity=False,
        default_dtype=torch.double,
    )
    criterion_tests.append(regression_test_info)


# Check that classification criterion work with no batch dimensions
# List of tuples of (name, input_fn, target_fn)
classification_criterion_no_batch = [
    (
        'BCELoss',
        lambda: torch.sigmoid(torch.randn(9, dtype=torch.double)),
        lambda: torch.randn(9, dtype=torch.double).gt(0).to(torch.double)
    ),
    ('BCEWithLogitsLoss', lambda: torch.randn(9, dtype=torch.double), lambda: torch.randn(9, dtype=torch.double)),
    ('HingeEmbeddingLoss', lambda: torch.randn(9, dtype=torch.double), lambda: torch.tensor([-1, 1, 1] * 3)),
    ('MultiLabelMarginLoss', lambda: torch.randn(4, dtype=torch.double), lambda: torch.tensor([3, 0, -1, 1])),
    ('SoftMarginLoss', lambda: torch.randn(9, dtype=torch.double), lambda: torch.tensor([-1, 1, 1] * 3)),
    ('NLLLoss', lambda: F.log_softmax(torch.randn(3, dtype=torch.double), dim=0), lambda: torch.tensor(1)),
    (
        'CosineEmbeddingLoss',
        lambda: (torch.randn(9, dtype=torch.double), torch.randn(9, dtype=torch.double)),
        lambda: torch.tensor(1, dtype=torch.double)
    ),
    # For MarginRankingLoss, input_fn : (x1, x2) and target_fn : target
    ('MarginRankingLoss', lambda: (torch.randn(()), torch.randn(())), lambda: torch.randn(()).sign()),
    # For TripletMarginLoss, input_fn : (anchor, positive) and target_fn : negative
    (
        'TripletMarginLoss',
        lambda: (torch.randn(9, dtype=torch.double), torch.randn(9, dtype=torch.double)),
        lambda: torch.randn(9, dtype=torch.double)
    ),
    ('MultiLabelSoftMarginLoss', lambda: torch.randn(9, dtype=torch.double), lambda: torch.randn(9)),
]
classification_criterion_no_batch_extra_info: dict[str, dict] = {
    'MultiLabelMarginLoss': {'check_gradgrad': False},
}
# TODO : Fix these discrepancies
classification_cpp_parity = {
    'BCELoss': False,
    'BCEWithLogitsLoss': False,
    'HingeEmbeddingLoss': False,
    'NLLLoss': False,
    'SoftMarginLoss': False,
}
reductions = ['none', 'mean', 'sum']
for (name, input_fn, target_fn), reduction in product(classification_criterion_no_batch,
                                                      reductions):
    classification_test_info = dict(
        fullname=f"{name}_no_batch_dim_{reduction}",
        constructor=lambda *args, name=name: getattr(nn, name)(reduction=reduction),
        input_fn=lambda f=input_fn: f(),
        target_fn=lambda f=target_fn: f(),
        reference_fn=single_batch_reference_criterion_fn,
        test_cpp_api_parity=True,
        has_parity=classification_cpp_parity.get(name, True)
    )
    extra_info = classification_criterion_no_batch_extra_info.get(name, {})
    classification_test_info.update(extra_info)
    criterion_tests.append(classification_test_info)


class NNTestCase(TestCase):
    # [CLASS DOC] Base test case for nn tests (extends the common TestCase). Subclasses supply
    # the abstract hooks _forward, _get_parameters, _zero_grad_parameters and _backward; the
    # base class builds on them to compute analytical and numerical Jacobians with respect to
    # inputs and parameters and compare them (check_jacobian) using the module-level PRECISION
    # tolerance. Holds no state of its own.
    # KNOWN ISSUE: abstractmethod is used but the base class does not use an abstract
    # metaclass, so missing overrides are not detected at instantiation; the base methods
    # just raise NotImplementedError.

    # _forward is defined in classes inheriting from NNTestCase
    @abstractmethod
    def _forward(self, *args, **kwargs):
        # [FUNCTION DOC] Abstract hook: runs the module forward on the given input. Base version
        # raises NotImplementedError.
        raise NotImplementedError

    @abstractmethod
    def _get_parameters(self, module: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        # [FUNCTION DOC] Abstract hook: returns a pair of lists (parameters, gradients of
        # parameters) for the module. Base version raises NotImplementedError.
        raise NotImplementedError

    @abstractmethod
    def _zero_grad_parameters(self, module: nn.Module) -> None:
        # [FUNCTION DOC] Abstract hook: resets the parameter gradients of the module. Base
        # version raises NotImplementedError.
        raise NotImplementedError

    @abstractmethod
    def _backward(self, module: nn.Module,
                  input: _TensorOrTensors, output: torch.Tensor,
                  grad_output: torch.Tensor | Sequence[torch.Tensor],
                  create_graph: bool = False):
        # [FUNCTION DOC] Abstract hook: runs backward from grad_output (optionally
        # creating a graph) and returns the gradient with respect to input. Base version
        # raises NotImplementedError.
        raise NotImplementedError

    def _jacobian(self, input, num_out):
        # [FUNCTION DOC] Returns zero-filled Jacobian containers with shape (number of elements
        # of each input tensor, num_out), mirroring nested tuple/list structure of the input.
        if isinstance(input, tuple):
            return tuple(self._jacobian(elem, num_out) for elem in input)
        elif isinstance(input, list):
            return [self._jacobian(elem, num_out) for elem in input]
        else:
            return torch.zeros(input.nelement(), num_out)

    def _flatten_tensors(self, x):
        # [FUNCTION DOC] Flattens a tensor (sparse tensors are densified first) to 1-d, or
        # recursively returns a tuple of flattened tensors for a nested sequence.
        if isinstance(x, torch.Tensor):
            if x.is_sparse:
                return x.to_dense().view(-1)
            else:
                return x.view(-1)
        else:
            return tuple(self._flatten_tensors(a) for a in x)

    def _zero_grad_input(self, input):
        # [FUNCTION DOC] Zeroes and detaches the gradient of an input tensor if it requires grad
        # and has one; recurses over sequences. Mutates the gradients in place.
        if isinstance(input, torch.Tensor):
            if input.requires_grad and input.grad is not None:
                input.grad.zero_()
                input.grad.detach_()
        else:
            for i in input:
                self._zero_grad_input(i)

    def _analytical_jacobian(self, module, input: _TensorOrTensors, jacobian_input=True, jacobian_parameters=True):
        # [FUNCTION DOC] Computes the analytical Jacobian by running forward once and then one
        # backward per output element with a one-hot output gradient. Returns a tuple containing
        # the input Jacobian (if jacobian_input) and the parameter Jacobian (if
        # jacobian_parameters), each of shape (inputs or parameters by output elements). Side
        # effects: zeroes parameter and input gradients repeatedly and calls the backward hook.
        output = self._forward(module, input)
        output_size = output.nelement()

        if jacobian_input:
            jacobian_inp = self._jacobian(input, output_size)
            flat_jacobian_input = list(_iter_tensors(jacobian_inp))

        if jacobian_parameters:
            num_param = sum(p.numel() for p in self._get_parameters(module)[0])
            jacobian_param = torch.zeros(num_param, output_size)

        for i in range(output_size):
            param, d_param = self._get_parameters(module)
            # make non grad zeros
            d_param = [torch.zeros_like(p) if d is None else d for (p, d) in zip(param, d_param, strict=True)]

            d_out = torch.zeros_like(output)
            flat_d_out = d_out.view(-1)
            flat_d_out[i] = 1

            if jacobian_parameters:
                self._zero_grad_parameters(module)
            # Tensors will accumulate gradient from multiple steps
            if jacobian_input:
                self._zero_grad_input(input)
            d_input = self._backward(module, input, output, d_out)

            if jacobian_input:
                for jacobian_x, d_x in zip(flat_jacobian_input, _iter_tensors(d_input), strict=True):
                    jacobian_x[:, i] = d_x.reshape(-1)
            if jacobian_parameters:
                jacobian_param[:, i] = torch.cat(self._flatten_tensors(d_param), 0)

        res: tuple[torch.Tensor, ...] = ()
        if jacobian_input:
            res += jacobian_inp,
        if jacobian_parameters:
            res += jacobian_param,

        return res

    def _numerical_jacobian(self, module, input: _TensorOrTensors, jacobian_input=True, jacobian_parameters=True):
        # [FUNCTION DOC] Computes the numerical Jacobian by finite differences (epsilon 1e-6) of
        # the detached forward result, with respect to the input (if jacobian_input) and the
        # concatenation of all parameters (if jacobian_parameters). Returns a tuple in the same
        # order as the analytical version.
        def fw(*input):
            # [FUNCTION DOC] HELPER: Forward closure used by the numerical Jacobian: runs the
            # module and returns the detached output.
            return self._forward(module, input).detach()

        res: tuple[torch.Tensor, ...] = ()
        if jacobian_input:
            res += _get_numerical_jacobian(fw, input, eps=1e-6),
        if jacobian_parameters:
            param, _ = self._get_parameters(module)
            to_cat = []
            for p in param:
                jacobian = _get_numerical_jacobian(fw, input, target=p, eps=1e-6)
                # get_numerical_jacobian returns a list of tuples but we require a tensor
                to_cat.append(jacobian[0][0])
            res += (torch.cat(to_cat, 0),)
        return res

    def check_jacobian(self, module, input: _TensorOrTensors, jacobian_input=True):
        # [FUNCTION DOC] Checks analytical versus numerical Jacobians for a module. Parameter
        # Jacobians are included if the module has parameters. Asserts the largest absolute
        # difference across all non-empty pairs is at most PRECISION. Fails the test via
        # assertLessEqual. Does not verify the shapes of the Jacobians.
        jacobian_parameters = bool(self._get_parameters(module)[0])
        analytical = self._analytical_jacobian(module, input, jacobian_input, jacobian_parameters)
        numerical = self._numerical_jacobian(module, input, jacobian_input, jacobian_parameters)
        analytical_t = list(_iter_tensors(analytical))
        numerical_t = list(_iter_tensors(numerical))

        differences = []
        for a, n in zip(analytical_t, numerical_t, strict=True):
            if a.numel() != 0:
                differences.append(a.add(n, alpha=-1).abs().max())
            # TODO: compare structure (ensure analytic jacobian has correct shape)
        if len(differences) > 0:
            self.assertLessEqual(max(differences), PRECISION)  # type: ignore[type-var]


class TestBase:
    # [CLASS DOC] Base class for declarative test specs. Constructed from a constructor (module
    # class or factory), an optional description, optional reference function, optional
    # fullname, and keyword arguments that supply the required arguments constructor_args, input
    # and extra_args either as a value, a generating function (suffix _fn) or a size (suffix
    # _size). State: desc, fullname, constructor, reference_fn, the extra keyword arguments and
    # a per-instance cache of materialised arguments. Subclasses define how a spec is run when
    # called with a test case.

    _required_arg_names = {'constructor_args', 'input', 'extra_args'}

    def __init__(self, constructor, desc='', reference_fn=None, fullname=None, **kwargs):
        # [FUNCTION DOC] Stores the constructor, description, reference function, fullname and
        # extra keyword arguments, and creates the argument cache. For each required argument
        # name, defaults constructor_args and extra_args to an empty tuple when none of the
        # value/_fn/_size forms is given; raises ValueError if input is missing in all three
        # forms. Mutates the passed-in keyword mapping.
        self.desc = desc
        self.fullname = fullname
        self.constructor = constructor
        self.reference_fn = reference_fn
        for name in self._required_arg_names:
            if name not in kwargs and name + '_fn' not in kwargs and name + '_size' not in kwargs:
                if name in {'constructor_args', 'extra_args'}:
                    kwargs[name] = ()
                else:
                    raise ValueError(f"{self.get_name()}: Specify {name} by a value, a function to generate it, or its size!")
        self._extra_kwargs = kwargs
        self._arg_cache = {}

    def get_name(self):
        # [FUNCTION DOC] Returns the test name: test_ plus fullname if set, otherwise test_ plus
        # the constructor name with the description appended after an underscore if non-empty.
        if self.fullname is not None:
            return 'test_' + self.fullname

        test_name = 'test_' + self.constructor.__name__
        if self.desc:
            test_name += '_' + self.desc
        return test_name

    def _unpack(self, value):
        # [FUNCTION DOC] HELPER: Returns tensors and non-iterable values unchanged;
        # rebuilds any other iterable container as the same container type with each
        # element unpacked recursively.
        if isinstance(value, torch.Tensor):
            return value
        elif is_iterable(value):
            return type(value)(self._unpack(v) for v in value)
        else:
            return value

    @property
    def constructor_args(self):
        # [FUNCTION DOC] Property: the (cached) constructor arguments, with containers rebuilt
        # by the unpack step.
        return self._get_arg('constructor_args', True)

    @property
    def extra_args(self):
        # [FUNCTION DOC] Property: the (cached) extra arguments, with containers rebuilt by the
        # unpack step.
        return self._get_arg('extra_args', True)

    def _get_arg(self, name, unpack):
        # [FUNCTION DOC] HELPER: Returns the cached argument for a required argument name,
        # creating it on first use from the value, the generating function, or random normal
        # tensors of the given size (nested lists of sizes allowed; existing tensors become
        # double). Optionally rebuilds containers via the unpack step. Raises an error if the
        # name is not a required argument or no source exists.
        if name not in self._required_arg_names:
            raise AssertionError(f"Expected name '{name}' to be in required arg names")

        if name not in self._arg_cache:
            fn_name = name + '_fn'
            size_name = name + '_size'

            if name in self._extra_kwargs:
                self._arg_cache[name] = self._extra_kwargs[name]
            elif fn_name in self._extra_kwargs:
                self._arg_cache[name] = self._extra_kwargs[fn_name]()
            else:
                if size_name not in self._extra_kwargs:
                    raise AssertionError(
                        f"Missing `{name}`, `{size_name}` or `{fn_name}` for {self.get_name()}"
                    )

                def map_tensor_sizes(sizes):
                    # [FUNCTION DOC] HELPER: Maps a size, a tensor or nested lists of those to
                    # random normal tensors; existing tensors are converted to double.
                    if isinstance(sizes, list):
                        return [map_tensor_sizes(s) for s in sizes]
                    elif isinstance(sizes, torch.Tensor):
                        return sizes.double()
                    else:
                        return torch.randn(sizes)

                self._arg_cache[name] = map_tensor_sizes(self._extra_kwargs[size_name])

        return self._unpack(self._arg_cache[name]) if unpack else self._arg_cache[name]

    def _get_input(self, unpack=True):
        # [FUNCTION DOC] HELPER: Returns the cached input argument, optionally unpacked.
        return self._get_arg('input', unpack)

    def __call__(self, test_case):
        # [FUNCTION DOC] Abstract entry point: runs the test against a test case. Base version
        # raises NotImplementedError.
        raise NotImplementedError


class ModuleTest(TestBase):
    # [CLASS DOC] Test spec for nn modules, running forward against an optional reference
    # function, a non-contiguous input/gradient check, a pickling round trip and a
    # subclass-defined extra check (_do_test). State from keyword arguments: jacobian_input,
    # test_cuda, pickle, check_gradgrad, FIXME_no_cuda_gradgrad_comparison, precision (default
    # 2e-4), check_forward_only and default_dtype (falls back to the current default dtype).
    # Also provides a CUDA-versus-CPU comparison (test_cuda).

    @abstractmethod
    def _do_test(self, test_case: Any, module: nn.Module, input: Any) -> Any:
        # [FUNCTION DOC] Abstract hook: subclass-specific checks on the constructed module and
        # input. Base version raises NotImplementedError.
        raise NotImplementedError

    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] Initializes the base spec and reads the flags and tolerances listed in
        # the class description from the keyword arguments.
        super().__init__(*args, **kwargs)
        self.jacobian_input = kwargs.get('jacobian_input', True)
        self.should_test_cuda = kwargs.get('test_cuda', True)
        self.should_test_pickle = kwargs.get('pickle', True)
        self.check_gradgrad = kwargs.get('check_gradgrad', True)
        self.FIXME_no_cuda_gradgrad_comparison = \
            kwargs.get('FIXME_no_cuda_gradgrad_comparison', False)
        self.precision = kwargs.get('precision', 2e-4)
        self.check_forward_only = kwargs.get('check_forward_only', False)
        self.default_dtype = kwargs.get('default_dtype')
        if self.default_dtype is None:
            self.default_dtype = torch.get_default_dtype()

    def __call__(self, test_case):
        # [FUNCTION DOC] Runs the spec under the configured default dtype: constructs the module
        # and input, compares forward output with the reference function (if given, dtype not
        # strictly checked), returns early if check_forward_only, runs the non-contiguous test,
        # optionally saves and reloads the module through a temporary file (full unpickling
        # allowed) comparing outputs, then runs the subclass _do_test. Failures are reported
        # through the test case assertions.
        with set_default_dtype(self.default_dtype):
            module = self.constructor(*self.constructor_args)
            input = self._get_input()

            if self.reference_fn is not None:
                out = test_case._forward(module, input)
                ref_input = deepcopy(input)
                ref_module = deepcopy(module)
                expected_out = self.reference_fn(ref_input, test_case._get_parameters(module)[0], ref_module)
                test_case.assertEqual(out, expected_out, exact_dtype=False)
            if self.check_forward_only:
                return
            self.test_noncontig(test_case, module, input)

            if self.should_test_pickle:
                # TODO: do this with in-memory files as soon as torch.save will support it
                with tempfile.TemporaryFile() as f:
                    test_case._forward(module, input)
                    torch.save(module, f)
                    f.seek(0)
                    # weights_only=False as this is legacy code that saves the model
                    module_copy = torch.load(f, weights_only=False)
                    test_case.assertEqual(test_case._forward(module, input), test_case._forward(module_copy, input))

            self._do_test(test_case, module, input)

    def noncontiguize(self, obj):
        # [FUNCTION DOC] Returns a non-contiguous copy of a tensor (recursing over lists and
        # tuples): stacks the tensor with an empty tensor along the dimension just after the
        # first dimension of size above one and selects the tensor slice, detaching it and
        # keeping requires_grad. Raises an error if the result is unexpectedly contiguous when
        # it has more than one element.
        if isinstance(obj, list):
            return [self.noncontiguize(o) for o in obj]
        elif isinstance(obj, tuple):
            return tuple(self.noncontiguize(o) for o in obj)
        tensor = obj
        ndim = tensor.dim()
        # Always making only the last dimension noncontiguous is easy to hide
        # bugs because .view(-1) will still work. So try to find a dim with size
        # > 1 and make that non-contiguous, i.e., stack + select on the
        # dimension directly after that.
        dim = ndim
        for d in range(ndim):
            if tensor.size(d) > 1:
                dim = d + 1
                break
        noncontig = torch.stack([torch.empty_like(tensor), tensor], dim).select(dim, 1).detach()
        if not (noncontig.numel() == 1 or noncontig.numel() == 0 or not noncontig.is_contiguous()):
            raise AssertionError(
                f"Expected noncontig to be non-contiguous or have numel <= 1, got numel={noncontig.numel()}"
            )
        noncontig.requires_grad = tensor.requires_grad
        return noncontig

    def test_noncontig(self, test_case, module, input):
        # [FUNCTION DOC] Checks that forward and backward results do not depend on memory
        # layout: skips scalars; computes reference output, input gradient and parameter
        # gradients with contiguous data, then repeats for every combination of contiguous or
        # non-contiguous input and grad_output (with the RNG state frozen) and asserts equality.
        # Mutates gradients of the module and input.
        # check no scalars, can't make non-contig
        if isinstance(input, torch.Tensor) and input.dim() == 0:
            return
        if any(i.dim() == 0 for i in input if isinstance(i, torch.Tensor)):
            return

        test_case._zero_grad_parameters(module)
        test_case._zero_grad_input(input)
        with freeze_rng_state():
            output = test_case._forward(module, input)
            if getattr(module, "return_indices", False):
                output = output[0]
            grad_output = output.new(output.shape).normal_()
            output = output.clone()
            d_input = deepcopy(test_case._backward(module, input, output, grad_output))
            d_param = deepcopy(test_case._get_parameters(module)[1])

        nc_input = self.noncontiguize(input)
        nc_grad_output = self.noncontiguize(grad_output)
        for contig_i, contig_g in product((True, False), repeat=2):
            i = input if contig_i else nc_input
            # Some ops, e.g., nn.Flatten, return gradient that shares
            # storage with the grad_output. Hence we copy here.
            go = deepcopy(grad_output if contig_g else nc_grad_output)
            test_case._zero_grad_parameters(module)
            test_case._zero_grad_input(i)
            with freeze_rng_state():
                out = test_case._forward(module, i)
                if getattr(module, "return_indices", False):
                    out = out[0]
                grad = test_case._backward(module, i, out, go)

                test_case.assertEqual(out, output)
                test_case.assertEqual(grad, d_input, atol=1e-4, rtol=0)
                test_case.assertEqual(test_case._get_parameters(module)[1], d_param)

    def test_cuda(self, test_case):
        # [FUNCTION DOC] Compares the module on CPU and GPU: skips (raises SkipTest) if CUDA is
        # unavailable or disabled. Copies CPU parameters to a float GPU copy, compares outputs,
        # then five rounds of random backward gradients (inputs and parameters), then optional
        # double-backward comparisons, and finally the non-contiguous check on the GPU module.
        # Tolerance comes from the precision attribute.
        # KNOWN ISSUE: in the double-backward section, the comparison of the last first-order
        # input gradients is repeated although they were already compared in the loop above; it
        # does not use the gradient results computed just before it.
        if not TEST_CUDA or not self.should_test_cuda:
            raise unittest.SkipTest('Excluded from CUDA tests')

        with set_default_dtype(self.default_dtype):
            cpu_input = self._get_input()

            type_map = {torch.double: torch.float}
            cpu_input_tuple = cpu_input if isinstance(cpu_input, tuple) else (cpu_input,)

            is_any_input_complex = any(isinstance(t, torch.Tensor) and t.dtype.is_complex for t in cpu_input_tuple)

            gpu_input_tuple = to_gpu(cpu_input_tuple, type_map=type_map)

            cpu_module = self.constructor(*self.constructor_args)
            gpu_module = self.constructor(*self.constructor_args).float().cuda()
            cpu_param = test_case._get_parameters(cpu_module)
            gpu_param = test_case._get_parameters(gpu_module)
            for cpu_p, gpu_p in zip(cpu_param[0], gpu_param[0], strict=True):
                gpu_p.data.copy_(cpu_p)

            test_case._zero_grad_input(cpu_input_tuple)
            test_case._zero_grad_input(gpu_input_tuple)
            test_case._zero_grad_parameters(cpu_module)
            test_case._zero_grad_parameters(gpu_module)
            cpu_output = test_case._forward(cpu_module, cpu_input_tuple)
            gpu_output = test_case._forward(gpu_module, gpu_input_tuple)
            if getattr(cpu_module, "return_indices", False):
                cpu_output = cpu_output[0]
                gpu_output = gpu_output[0]
            test_case.assertEqual(cpu_output, gpu_output, atol=self.precision, rtol=0, exact_dtype=False)

            # Run backwards on CPU and GPU and compare results
            for _ in range(5):
                cpu_gradOutput = cpu_output.clone().normal_()
                gpu_gradOutput = cpu_gradOutput.type_as(gpu_output)
                cpu_gradInput = test_case._backward(cpu_module, cpu_input_tuple, cpu_output, cpu_gradOutput)
                gpu_gradInput = test_case._backward(gpu_module, gpu_input_tuple, gpu_output, gpu_gradOutput)
                test_case.assertEqual(cpu_gradInput, gpu_gradInput, atol=self.precision, rtol=0, exact_dtype=False)
                for cpu_d_p, gpu_d_p in zip(cpu_param[1], gpu_param[1], strict=True):
                    test_case.assertEqual(cpu_d_p, gpu_d_p, atol=self.precision, rtol=0)

            # Run double-backwards on CPU and GPU and compare results
            if self.check_gradgrad and not self.FIXME_no_cuda_gradgrad_comparison:
                cpu_output = cpu_module(*cpu_input_tuple)
                gpu_output = gpu_module(*gpu_input_tuple)
                if getattr(cpu_module, "return_indices", False):
                    cpu_output = cpu_output[0]
                    gpu_output = gpu_output[0]

                cpu_gradOutput = torch.randn_like(cpu_output, requires_grad=True)
                gpu_gradOutput = cpu_gradOutput.type_as(gpu_output).detach()
                gpu_gradOutput.requires_grad = True

                cpu_gradInputs = torch.autograd.grad(
                    cpu_output,
                    cpu_input_tuple + tuple(cpu_module.parameters()),
                    cpu_gradOutput,
                    create_graph=True)
                gpu_gradInputs = torch.autograd.grad(
                    gpu_output,
                    gpu_input_tuple + tuple(gpu_module.parameters()),
                    gpu_gradOutput,
                    create_graph=True)

                for cpu_d_i, gpu_d_i in zip(cpu_gradInputs, gpu_gradInputs, strict=True):
                    test_case.assertEqual(cpu_d_i, gpu_d_i, atol=self.precision, rtol=0, exact_dtype=False)

                # We mix output into the second backwards computation so that
                # torch.autograd.grad doesn't complain that some inputs
                # are unreachable (which can happen if you differentiate
                # only on the gradient.
                if is_any_input_complex:
                    outputs_cpu = cpu_output.sum().abs() + sum(x.sum().abs() for x in cpu_gradInputs)
                    outputs_gpu = gpu_output.sum().abs() + sum(x.sum().abs() for x in gpu_gradInputs)
                else:
                    outputs_cpu = cpu_output.sum() + sum(x.sum() for x in cpu_gradInputs)
                    outputs_gpu = gpu_output.sum() + sum(x.sum() for x in gpu_gradInputs)

                cpu_gg = torch.autograd.grad(
                    outputs_cpu,
                    cpu_input_tuple + (cpu_gradOutput,) + tuple(cpu_module.parameters()),
                    retain_graph=True)
                gpu_gg = torch.autograd.grad(
                    outputs_gpu,
                    gpu_input_tuple + (gpu_gradOutput,) + tuple(gpu_module.parameters()),
                    retain_graph=True)
                test_case.assertEqual(cpu_gradInput, gpu_gradInput, atol=self.precision, rtol=0, exact_dtype=False)
                for cpu_d_p, gpu_d_p in zip(cpu_gg, gpu_gg, strict=True):
                    test_case.assertEqual(cpu_d_p, gpu_d_p, atol=self.precision, rtol=0, exact_dtype=False)

            self.test_noncontig(test_case, gpu_module, gpu_input_tuple)


class InputVariableMixin:
    # [CLASS DOC] Mixin for specs whose inputs must require gradients; overrides _get_input to
    # mark floating-point and complex tensors (recursively in containers) as requiring grad.
    # Holds no state of its own.
    def _get_input(self):
        # [FUNCTION DOC] Returns the input argument (unpacked form disabled) with every floating
        # or complex tensor, including those nested in containers, set to require grad. Mutates
        # the cached input tensors.
        input = TestBase._get_input(self, False)  # type: ignore[arg-type]

        def map_variables(i):
            # [FUNCTION DOC] HELPER: Sets requires_grad on floating or complex tensors and
            # rebuilds containers recursively.
            if isinstance(i, torch.Tensor):
                if i.is_floating_point() or i.is_complex():
                    i.requires_grad = True
                return i
            else:
                return type(i)(map_variables(elem) for elem in i)

        return map_variables(input)


class NewModuleTest(InputVariableMixin, ModuleTest):  # type: ignore[misc]
    # [CLASS DOC] Full-featured module test: adds gradient checking, in-place variant
    # comparison, dtype/device movement checks and printing to ModuleTest. Extra state from
    # keyword arguments: cudnn, check_inplace, check_gradgrad, skip_double, skip_half,
    # with_tf32, tf32_precision, test_cpu, has_sparse_gradients, check_batched_grad,
    # gradcheck_fast_mode, supports_forward_ad, supports_fwgrad_bwgrad. Inputs require grad via
    # the mixin.
    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] Initializes the base test and reads the extra flags and tolerances
        # listed in the class description from the keyword arguments.
        super().__init__(*args, **kwargs)
        self.cudnn = kwargs.get('cudnn', False)
        self.check_inplace = kwargs.get('check_inplace', False)
        self.check_gradgrad = kwargs.get('check_gradgrad', True)
        self.skip_double = kwargs.get('skip_double', False)
        self.skip_half = kwargs.get('skip_half', False)
        self.with_tf32 = kwargs.get('with_tf32', False)
        self.tf32_precision = kwargs.get('tf32_precision', 0.001)
        self.test_cpu = kwargs.get('test_cpu', True)
        self.has_sparse_gradients = kwargs.get('has_sparse_gradients', False)
        self.check_batched_grad = kwargs.get('check_batched_grad', True)
        self.gradcheck_fast_mode = kwargs.get('gradcheck_fast_mode')
        self.supports_forward_ad = kwargs.get('supports_forward_ad', False)
        self.supports_fwgrad_bwgrad = kwargs.get('supports_fwgrad_bwgrad', False)

    def _check_gradients(self, test_case, module, input_tuple):
        # [FUNCTION DOC] HELPER: Runs gradient checks on module parameters and inputs: for
        # modules with sparse gradients uses the Jacobian comparison on the single input,
        # otherwise runs gradcheck (with batched grad and forward AD options); additionally runs
        # gradgradcheck unless disabled. Failures are test assertions. Raises an error if kwargs
        # are passed to the checked function or the sparse path has more than one input.
        params = tuple(x for x in module.parameters())
        num_inputs = len(input_tuple)

        def fn_to_gradcheck(*inputs_and_params, **kwargs):
            # [FUNCTION DOC] HELPER: Closure for gradcheck: runs the module forward on the
            # leading inputs (parameters are included as extra arguments only so that they are
            # differentiated). Raises an error if keyword arguments are given.
            if kwargs:
                raise AssertionError(f"Expected no kwargs, got {kwargs}")
            return test_case._forward(module, inputs_and_params[:num_inputs])

        # gradcheck doesn't support operators that take in dense inputs but
        # return sparse parameters. This only happens in the case of nn.Embedding
        # and nn.EmbeddingBag. Instead, we call `self.check_jacobian`, which
        # is a slightly different version of gradcheck that can handle this.
        if self.has_sparse_gradients:
            if num_inputs != 1:
                raise AssertionError(f"Expected num_inputs == 1, got {num_inputs}")
            test_input_jacobian = torch.is_floating_point(input_tuple[0])
            test_case.check_jacobian(module, input_tuple[0], test_input_jacobian)
        else:
            test_case.assertTrue(gradcheck(fn_to_gradcheck, input_tuple + params,
                                           check_batched_grad=self.check_batched_grad,
                                           fast_mode=self.gradcheck_fast_mode,
                                           check_forward_ad=self.supports_forward_ad))

        if self.check_gradgrad:
            test_case.assertTrue(gradgradcheck(fn_to_gradcheck, input_tuple + params,
                                               check_batched_grad=self.check_batched_grad,
                                               fast_mode=self.gradcheck_fast_mode,
                                               check_fwd_over_rev=self.supports_fwgrad_bwgrad))

    def _do_test(self, test_case, module, input):
        # [FUNCTION DOC] Subclass check run after the common checks: temporarily sets thread
        # count to 1, runs the gradient checks, prints the module repr, optionally verifies the
        # in-place variant gives the same output and input gradient without changing the
        # original input version, then checks that casting to float and double, and (if CUDA)
        # moving to GPU 0, back to CPU, GPU 1, double and half keep parameters at the expected
        # tensor type and device. Restores the thread count at the end.
        # KNOWN ISSUE: the thread count is not restored if any step raises, since there is
        # no try/finally.
        num_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        input_tuple = input if isinstance(input, tuple) else (input,)

        self._check_gradients(test_case, module, input_tuple)

        # check if module can be printed
        module.__repr__()

        if self.check_inplace:
            # check if the inplace variant of the module gives the same result
            # as the out-of-place

            # check_inplace doesn't support multiple input tensors, since we don't have any modules
            # that modify the inputs in-place and that accept more than one input
            if len(input_tuple) != 1:
                raise AssertionError(f"Expected len(input_tuple) == 1, got {len(input_tuple)}")
            input = input_tuple[0]

            module_ip = self.constructor(*self.constructor_args, inplace=True)

            input_version = input._version
            with freeze_rng_state():
                output = module(input)
            test_case.assertEqual(input._version, input_version)

            input_ip = deepcopy(input)
            input_ip_clone = input_ip.clone()
            with freeze_rng_state():
                output_ip = module_ip(input_ip_clone)
            test_case.assertNotEqual(input_ip_clone._version, input_version)
            test_case.assertEqual(output, output_ip)
            grad = output.data.clone().normal_()
            if input.grad is not None:
                with torch.no_grad():
                    input.grad.zero_()
            if input_ip.grad is not None:
                with torch.no_grad():
                    input_ip.grad.zero_()
            output.backward(grad)
            output_ip.backward(grad)
            test_case.assertEqual(input.grad, input_ip.grad)

        def assert_module_parameters_are(tensor_type, device_id=None):
            # [FUNCTION DOC] HELPER: Asserts that every parameter of the module has the given
            # tensor type and, if device_id is given, lives on that device.
            for p in module.parameters():
                test_case.assertIsInstance(p, tensor_type)
                if device_id is not None:
                    test_case.assertEqual(p.get_device(), device_id)

        if all(isinstance(t, torch.LongTensor) for t in input_tuple) and TEST_CUDA:
            # check that cuda() moves module parameters to correct GPU device,
            # and that float() casts parameters correctly
            input_tuple = tuple(t.cuda() for t in input_tuple)
            module.float().cuda()
            module(*input_tuple)
            assert_module_parameters_are(torch.cuda.FloatTensor, 0)  # type: ignore[attr-defined]

            if torch.cuda.device_count() > 1:
                input_tuple = tuple(t.cuda(1) for t in input_tuple)
                module.cuda(1)
                with torch.cuda.device(1):
                    module(*input_tuple)
                assert_module_parameters_are(torch.cuda.FloatTensor, 1)  # type: ignore[attr-defined]
        else:
            # check that float()/double() casters work correctly
            def to_type(tensor, real, complex):
                # [FUNCTION DOC] HELPER: Converts a tensor to the given real dtype if floating,
                # the given complex dtype if complex, else returns it unchanged.
                if tensor.is_complex():
                    return tensor.to(complex)
                elif tensor.is_floating_point():
                    return tensor.to(real)
                else:
                    return tensor

            def to_half(x):
                # [FUNCTION DOC] HELPER: Converts floating tensors to half precision; complex
                # tensors are not supported.
                # TODO: torch.complex32 when properly supported
                return to_type(x, torch.float16, None)

            def to_single(x):
                # [FUNCTION DOC] HELPER: Converts floating tensors to float32 and complex
                # tensors to complex64.
                return to_type(x, torch.float32, torch.complex64)

            def to_double(x):
                # [FUNCTION DOC] HELPER: Converts floating tensors to float64 and complex
                # tensors to complex128.
                return to_type(x, torch.float64, torch.complex128)

            # to float
            input_tuple = tuple(to_single(t) for t in input_tuple)
            module.float()
            module(*input_tuple)
            assert_module_parameters_are(torch.FloatTensor)

            # and back to double
            input_tuple = tuple(to_double(t) for t in input_tuple)
            module.double()
            module(*input_tuple)
            assert_module_parameters_are(torch.DoubleTensor)

            if TEST_CUDA and self.should_test_cuda:
                # check that cuda() moves module parameters to correct GPU device,
                # and that float() casts parameters correctly

                # to GPU0
                input_tuple = tuple(to_single(t).cuda() for t in input_tuple)
                module.float().cuda()
                module(*input_tuple)
                assert_module_parameters_are(torch.cuda.FloatTensor, 0)  # type: ignore[attr-defined]

                # to CPU
                input_tuple = tuple(t.cpu() for t in input_tuple)
                module.cpu()
                module(*input_tuple)
                assert_module_parameters_are(torch.FloatTensor)

                # back to GPU0
                input_tuple = tuple(t.cuda() for t in input_tuple)
                module.cuda()
                module(*input_tuple)
                assert_module_parameters_are(torch.cuda.FloatTensor, 0)  # type: ignore[attr-defined]

                # test that forwards of module runs correctly without cuDNN
                if self.cudnn:
                    with torch.backends.cudnn.flags(enabled=False):
                        module(*input_tuple)
                        assert_module_parameters_are(torch.cuda.FloatTensor, 0)  # type: ignore[attr-defined]

                if torch.cuda.device_count() >= 2:
                    # test cross-GPU transfer works
                    # to GPU1
                    input_tuple = tuple(t.cuda(1) for t in input_tuple)
                    module.cuda(1)
                    with torch.cuda.device(1):
                        module(*input_tuple)
                    assert_module_parameters_are(torch.cuda.FloatTensor, 1)  # type: ignore[attr-defined]

                if not self.skip_double:
                    # test double()
                    input_tuple = tuple(to_double(t).cuda() for t in input_tuple)
                    module.double().cuda()
                    module(*input_tuple)
                    assert_module_parameters_are(torch.cuda.DoubleTensor, 0)  # type: ignore[attr-defined]

                # test half()
                if not self.skip_half:
                    input_tuple = tuple(to_half(t).cuda() for t in input_tuple)
                    module.half().cuda()
                    module(*input_tuple)
                    assert_module_parameters_are(torch.cuda.HalfTensor, 0)  # type: ignore[attr-defined]
        torch.set_num_threads(num_threads)

    def _get_target(self):
        # [FUNCTION DOC] HELPER: Returns the target argument without unpacking.
        return self._get_arg('target', False)

    @property
    def constructor_args(self):
        # [FUNCTION DOC] Property: the constructor arguments without unpacking containers.
        return self._get_arg('constructor_args', False)


class CriterionTest(InputVariableMixin, TestBase):  # type: ignore[misc]
    # [CLASS DOC] Test spec for loss modules (criterions). Required arguments are the base ones
    # plus a target. State from keyword arguments: test_cuda, check_forward_only,
    # check_gradgrad, check_half, check_bfloat16, check_complex, test_cpu, with_tf32 (default
    # True), tf32_precision, check_batched_grad and default_dtype. Running it checks the forward
    # result against the reference function and runs gradcheck/gradgradcheck; test_cuda compares
    # CPU and GPU results for a given dtype. Test cases are expected to provide
    # _forward_criterion and _backward_criterion.
    # TODO: check that criterions don't ignore grad_output

    _required_arg_names = TestBase._required_arg_names.union({'target'})

    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] Initializes the base spec and reads the flags and tolerances listed in
        # the class description from the keyword arguments; default dtype falls back to the
        # current global default.
        super().__init__(*args, **kwargs)
        self.should_test_cuda = kwargs.get('test_cuda', True)
        self.check_forward_only = kwargs.get('check_forward_only', False)
        self.check_gradgrad = kwargs.get('check_gradgrad', True)
        self.check_half = kwargs.get('check_half', True)
        self.check_bfloat16 = kwargs.get('check_bfloat16', False)
        self.check_complex = kwargs.get('check_complex', False)
        self.test_cpu = kwargs.get('test_cpu', True)
        self.with_tf32 = kwargs.get('with_tf32', True)
        self.tf32_precision = kwargs.get('tf32_precision', 0.001)
        self.check_batched_grad = kwargs.get('check_batched_grad', True)
        self.default_dtype = kwargs.get('default_dtype')
        if self.default_dtype is None:
            self.default_dtype = torch.get_default_dtype()

    def __call__(self, test_case):
        # [FUNCTION DOC] Runs the spec under the configured default dtype: builds module,
        # input and target, checks that repr and str work, compares forward output with the
        # reference function when given, returns early if check_forward_only, then runs
        # gradcheck over inputs, parameters and target, and gradgradcheck if enabled.
        # Failures are test assertions.
        with set_default_dtype(self.default_dtype):
            module = self.constructor(*self.constructor_args)
            input = self._get_input()

            # Check that these methods don't raise errors
            module.__repr__()
            str(module)

            target = self._get_target()

            if self.reference_fn is not None:
                out = test_case._forward_criterion(module, input, target, extra_args=self.extra_args)
                ref_args = (deepcopy(input), deepcopy(target)) + self.extra_args + (module,)
                expected_out = self.reference_fn(*ref_args)
                test_case.assertEqual(out, expected_out)

            if self.check_forward_only:
                return

            params = tuple(x for x in module.parameters())
            if not isinstance(input, tuple):
                inputs = (input,) + params + (target,)

                def apply_fn(input, target, *params):
                    # [FUNCTION DOC] HELPER: Wraps a one-input criterion call for gradcheck;
                    # parameters are accepted only to be differentiated.
                    return module(input, target)
            else:
                inputs = input + params + (target,)

                def apply_fn(input1, input2, target, *params):  # type: ignore[misc]
                    # [FUNCTION DOC] HELPER: Wraps a two-input criterion call for gradcheck;
                    # parameters are accepted only to be differentiated.
                    return module(input1, input2, target)

            gradcheck(apply_fn, inputs, check_batched_grad=self.check_batched_grad)

            if self.check_gradgrad:
                gradgradcheck(apply_fn, inputs, check_batched_grad=self.check_batched_grad)

    def test_cuda(self, test_case, dtype, extra_args=None):
        # [FUNCTION DOC] Compares CPU and GPU criterion output and input gradients for the given
        # dtype: skips (SkipTest) if CUDA is unavailable or disabled; converts input, target and
        # module to dtype, moves GPU copies to CUDA, and for half or bfloat16 recomputes the CPU
        # side in the default dtype. Tolerance is 1e-1 for half and bfloat16 and 4e-4 otherwise.
        # Optional extra_args are forwarded to the test case helpers.
        def convert_dtype(obj, dtype, requires_grad=False):
            # [FUNCTION DOC] HELPER: Detaches tensors (recursing over tuples), converts them to
            # the dtype and sets requires_grad; non-tensors are returned unchanged.
            if isinstance(obj, torch.Tensor):
                return obj.detach().to(dtype=dtype).requires_grad_(requires_grad)
            elif isinstance(obj, tuple):
                return tuple(convert_dtype(o, dtype, requires_grad) for o in obj)
            else:
                return obj

        if not TEST_CUDA or not self.should_test_cuda:
            raise unittest.SkipTest('Excluded from CUDA tests')

        with set_default_dtype(self.default_dtype):
            cpu_input = self._get_input()
            cpu_target = self._get_target()
            cpu_module = self.constructor(*self.constructor_args)
            gpu_module = self.constructor(*self.constructor_args)

            # Convert input, target and module parameters to dtype
            cpu_input = convert_dtype(cpu_input, dtype, True)
            if cpu_target.is_floating_point() or cpu_target.is_complex():
                cpu_target = convert_dtype(cpu_target, dtype)
            cpu_module.type(dtype)
            gpu_module.type(dtype)

            # GPU setup
            gpu_input = to_gpu(cpu_input)
            gpu_target = to_gpu(cpu_target)
            gpu_module.cuda()

            # torch.HalfTensor doesn't support most operations, converting back to default
            if dtype in {torch.half, torch.bfloat16}:
                cpu_input = self._get_input()
                cpu_target = self._get_target()
                # Loss modules with weights require consistent input/module weight types
                cpu_module = self.constructor(*self.constructor_args)

            cpu_output = test_case._forward_criterion(cpu_module, cpu_input, cpu_target, extra_args=extra_args)
            gpu_output = test_case._forward_criterion(gpu_module, gpu_input, gpu_target, extra_args=extra_args)
            # dtype used to be able to be None, so set precision in this way instead of a precision map
            test_case.assertEqual(cpu_output, gpu_output,
                                  atol=1e-1 if dtype in {torch.half, torch.bfloat16} else 4e-4, rtol=0, exact_dtype=False)

            cpu_gradInput = test_case._backward_criterion(
                cpu_module, cpu_input, cpu_output, cpu_target, extra_args=extra_args)
            gpu_gradInput = test_case._backward_criterion(
                gpu_module, gpu_input, gpu_output, gpu_target, extra_args=extra_args)
            # dtype used to be able to be None, so set precision in this way instead of a precision map
            test_case.assertEqual(cpu_gradInput, gpu_gradInput,
                                  atol=1e-1 if dtype in {torch.half, torch.bfloat16} else 4e-4, rtol=0, exact_dtype=False)

    def _get_target(self):
        # [FUNCTION DOC] HELPER: Returns the target argument without unpacking.
        return self._get_arg('target', False)

    @property
    def constructor_args(self):
        # [FUNCTION DOC] Property: the constructor arguments without unpacking containers.
        return self._get_arg('constructor_args', False)

    @property
    def extra_args(self):
        # [FUNCTION DOC] Property: the extra arguments without unpacking containers.
        return self._get_arg('extra_args', False)


def _test_bfloat16_ops(test_case, op, device, inp_dims=(), prec=1e-2, scale_factor=None):
    # [FUNCTION DOC] HELPER: Compares an op in float32 against the same op converted to
    # bfloat16 for forward output and input gradient within tolerance prec. Input has the
    # given dimensions on the device; if scale_factor is given the input is regenerated as
    # scaled uniform values rounded through bfloat16. Mutates the op in place by converting
    # it to bfloat16.
    # KNOWN ISSUE: when scale_factor is given the first random input is generated and
    # then discarded.
    # fp32 compute
    input1 = torch.randn(inp_dims, dtype=torch.float32, device=device, requires_grad=True)
    if scale_factor is not None:
        input1 = (torch.rand(inp_dims, dtype=torch.bfloat16, device=device) * scale_factor).float().requires_grad_()
    out1 = op(input1)
    grad_input1 = torch.randn_like(out1, device=device)
    out1.backward(grad_input1)

    # bfloat16 compute
    op_bfp16 = op.bfloat16()
    input2 = input1.detach().bfloat16().requires_grad_()
    grad_input2 = grad_input1.bfloat16()
    out2 = op_bfp16(input2)
    out2.backward(grad_input2)

    test_case.assertEqual(out1, out2, atol=prec, rtol=prec, exact_dtype=False)
    test_case.assertEqual(input1.grad.data, input2.grad.data, atol=prec, rtol=prec, exact_dtype=False)

def _test_module_empty_input(test_case, module, inp, check_size=True, inference=False):
    # [FUNCTION DOC] HELPER: Runs a module on an input with an empty dimension and checks
    # results: forward (unless inference only the input is set to require grad), backward with
    # random gradient, optionally that output size equals input size, and that all parameter
    # gradients and the input gradient are zero.
    if not inference:
        inp.requires_grad_(True)
    out = module(inp)
    if not inference:
        gO = torch.rand_like(out)
        out.backward(gO)
    if check_size:
        test_case.assertEqual(out.size(), inp.size())
    if not inference:
        for p in module.parameters():
            if p.requires_grad:
                test_case.assertEqual(p.grad, torch.zeros_like(p.grad))
        test_case.assertEqual(inp.grad, torch.zeros_like(inp))


def _create_basic_net():
    # [FUNCTION DOC] HELPER: Builds and returns a triple of test modules: a Layer with a
    # parameter and a buffer, a Net containing a Layer plus its own parameter and buffer, and a
    # Sequential containing the same Net twice (shared module).
    class Layer(nn.Module):
        # [CLASS DOC] HELPER: Tiny module with one parameter (shape 3 by 5, uninitialized) and
        # one buffer (zeros of shape 1 by 3 by 3 by 7); no forward.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Registers the dummy parameter and buffer.
            super().__init__()
            self.layer_dummy_param = nn.Parameter(torch.empty(3, 5))
            self.layer_dummy_buf = nn.Buffer(torch.zeros(1, 3, 3, 7))

    class Net(nn.Module):
        # [CLASS DOC] HELPER: Tiny module containing a Layer child, an uninitialized 3 by 5
        # parameter and a zeros buffer of shape 7 by 3 by 3 by 1; no forward.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Registers the child layer, dummy parameter and buffer.
            super().__init__()
            self.l1 = Layer()
            self.dummy_param = nn.Parameter(torch.empty(3, 5))
            self.dummy_buf = nn.Buffer(torch.zeros(7, 3, 3, 1))

    l = Layer()
    n = Net()
    s = nn.Sequential(n, n)

    return l, n, s
