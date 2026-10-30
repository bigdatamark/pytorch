# Owner(s): ["module: tests"]
# [MODULE DOC] Tests for reduction operators (sum, mean, var/std, min/max/argmin/argmax,
# mode, median, quantile, all/any, bincount, bucketize/searchsorted, histc, histogram,
# histogramdd and friends). The device-generic class TestReductions mixes OpInfo-driven
# checks over reduction_ops (dim/keepdim shape rules, identity, nan policy, result
# dtype, noncontiguous inputs, numpy reference comparison) with many legacy hand-written
# tests; TestReductionsOnCPU holds CPU-only tests. Import time: the device-generic class
# is expanded per device type by instantiate_device_type_tests near the bottom of the
# module, and run_tests is invoked when the file is executed as a script.

import contextlib
import torch
import numpy as np

import math
from collections.abc import Sequence
import random
from functools import partial
from itertools import product, combinations, permutations
import warnings

from torch import inf, nan
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import (
    all_types_and_complex_and, get_all_math_dtypes, highest_precision_float,
    integral_types, complex_types, floating_types_and,
    integral_types_and, floating_and_complex_types_and, all_types_and, all_types,
)
from torch.testing._internal.common_utils import (
    TestCase, run_tests, skipIfNoSciPy, slowTest, torch_to_numpy_dtype_dict,
    parametrize, serialTest, subtest,
    gradcheck, gradgradcheck,
    skipIfMPS,
    skipIfTorchDynamo,
    IS_WINDOWS)
from torch.testing._internal.common_device_type import (
    OpDTypes, onlyCPU, onlyNativeDeviceTypes, expectedFailureMeta, instantiate_device_type_tests, dtypes, dtypesIfCUDA,
    dtypesIfCPU, dtypesIfMPS, dtypesIfXPU, onlyAccelerator, largeMPSBufferTest, largeTensorTest, ops,
    precisionOverride)
from torch.testing._internal.common_methods_invocations import (
    ReductionOpInfo, ReductionPythonRefInfo, reduction_ops, reference_masked_ops)




# TODO: replace with make_tensor
def _generate_input(shape, dtype, device, with_extremal):
    # [FUNCTION DOC] HELPER: Build a random test tensor of the given shape, dtype and
    # device. An empty shape gives an empty one-dimensional tensor. Floating and
    # complex dtypes get scaled normal random values with roughly half of them zeroed;
    # when with_extremal is set, random positions are overwritten with nan, inf and
    # -inf (floating or complex). Bool dtype gets random True entries; other integer
    # dtypes get random integers in a small positive range. bfloat16 is generated in
    # float32 and converted. Uses the global python and torch random state.
    if shape == ():
        x = torch.tensor((), dtype=dtype, device=device)
    else:
        if dtype.is_floating_point or dtype.is_complex:
            # work around torch.randn not being implemented for bfloat16
            if dtype == torch.bfloat16:
                x = torch.randn(*shape, device=device) * random.randint(30, 100)
                x = x.to(torch.bfloat16)
            else:
                x = torch.randn(*shape, dtype=dtype, device=device) * random.randint(30, 100)
            x[torch.randn(*shape) > 0.5] = 0
            if with_extremal and dtype.is_floating_point:
                # Use extremal values
                x[torch.randn(*shape) > 0.5] = float('nan')
                x[torch.randn(*shape) > 0.5] = float('inf')
                x[torch.randn(*shape) > 0.5] = float('-inf')
            elif with_extremal and dtype.is_complex:
                x[torch.randn(*shape) > 0.5] = complex('nan')
                x[torch.randn(*shape) > 0.5] = complex('inf')
                x[torch.randn(*shape) > 0.5] = complex('-inf')
        elif dtype == torch.bool:
            x = torch.zeros(shape, dtype=dtype, device=device)
            x[torch.randn(*shape) > 0.5] = True
        else:
            x = torch.randint(15, 100, shape, dtype=dtype, device=device)

    return x

# TODO: replace with make_tensor
def _rand_shape(dim, min_size, max_size):
    # [FUNCTION DOC] HELPER: Return a tuple of dim random integers,
    # each drawn uniformly (inclusive) between min_size and max_size,
    # to be used as a tensor shape. Uses the python random module.
    shape = []
    for _ in range(dim):
        shape.append(random.randint(min_size, max_size))
    return tuple(shape)

def _reduced_shape(shape, empty_dim_as_none=False, dim=None, keepdim=False):
    # [FUNCTION DOC] HELPER: Compute the expected shape (as a list) of a reduction result. When
    # dim is None, or when empty_dim_as_none is set and dim is an empty list, all dimensions are
    # reduced: the result is all ones of the input rank if keepdim else an empty list. Otherwise
    # dim may be an int or a sequence; negative entries are wrapped and duplicates collapse via
    # a set. Reduced dimensions are dropped, or kept with size 1 when keepdim is true.
    """Computes the expected reduced shape given dim and keepdim

    Args:
        shape: The shape to reduce
        dim : The dimensions to reduce
        keepdim: If true, reduced dimensions have size 1 in the reduced shape,
            otherwise they are removed from the reduced shape.

    Returns:
        The reduced shape
    """
    if dim is None or (empty_dim_as_none and dim == []):
        return [1] * len(shape) if keepdim else []

    # Wrap negative dims
    dim = dim if isinstance(dim, Sequence) else [dim]
    dim = {i if i >= 0 else len(shape) + i for i in dim}

    result = []
    for i, size in enumerate(shape):
        if i not in dim:
            result.append(size)
        elif keepdim:
            result.append(1)

    return result

class TestReductions(TestCase):
    # [CLASS DOC] Device-generic reduction tests (instantiated per device type). The first
    # section contains OpInfo-based checks that run over reduction_ops; the second section
    # contains legacy tests that call specific torch reduction functions directly and
    # compare against numpy, sorted references or other torch ops. Holds no per-instance
    # state beyond what TestCase provides; private underscore methods are shared helpers.
    ###########################################################################
    # ReductionOpInfo unit tests
    ###########################################################################

    def _test_dim_keepdim(self, op: ReductionOpInfo, device, *, ndim, **dim_keepdim):
        # [FUNCTION DOC] HELPER: Run the reduction op on a random float tensor of rank
        # ndim (each size randomly 2 to 4) using the first generated args/kwargs sample
        # plus the given dim/keepdim keyword arguments, and assert the result shape equals
        # the shape computed by _reduced_shape. For the linalg vector_norm ops (plain and
        # ref) an empty dim list is treated like None when computing the expected shape.
        """Tests output shape for input with ndim and dim and keepdim kwargs"""
        shape = torch.randint(2, 5, (ndim,)).tolist()
        t = make_tensor(shape, dtype=torch.float, device=device)
        args, kwargs = next(op.generate_args_kwargs(t, **dim_keepdim))
        result = op(t, *args, **dim_keepdim, **kwargs)
        empty_dim_as_none = (op.name == "linalg.vector_norm" or op.name == "_refs.linalg.vector_norm")
        expected_shape = _reduced_shape(shape, empty_dim_as_none, **dim_keepdim)
        self.assertEqual(result.shape, expected_shape, lambda msg: f"""{msg}\n
        expected output shape to be {expected_shape} but got {list(result.shape)}
        for input shape {shape} and {dim_keepdim}
        """)

    # TODO(@heitorschueroff) combine cases with and without keepdim once
    # there's support for a @parametrize decorator.

    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_dim_default(self, device, op: ReductionOpInfo):
        # [FUNCTION DOC] Checks that with default arguments the
        # op reduces every dimension, for input ranks 0 to 2.
        """Tests that the default dim reduces all dimensions."""
        for ndim in range(3):
            self._test_dim_keepdim(op, device, ndim=ndim)

    @skipIfMPS
    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_dim_default_keepdim(self, device, op: ReductionOpInfo):
        # [FUNCTION DOC] Checks that with default dim and keepdim true the op
        # reduces all dimensions to size 1, for ranks 0 to 2. Skipped on MPS.
        """Tests that the default dim, when keepdim=True, reduces all dimensions to size 1."""
        for ndim in range(3):
            self._test_dim_keepdim(op, device, ndim=ndim, keepdim=True)

    @skipIfMPS
    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_dim_none(self, device, op: ReductionOpInfo):
        # [FUNCTION DOC] Checks that an explicit dim of None
        # reduces all dimensions, for ranks 0 to 2. Skipped on MPS.
        """Tests that dim=None reduces all dimensions."""
        for ndim in range(3):
            self._test_dim_keepdim(op, device, ndim=ndim, dim=None)

    @skipIfMPS
    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_dim_none_keepdim(self, device, op: ReductionOpInfo):
        # [FUNCTION DOC] Checks that dim of None with keepdim true reduces
        # all dimensions to size 1, for ranks 0 to 2. Skipped on MPS.
        """Tests that dim=None, when keepdim=True, reduces all dimensions to size 1."""
        for ndim in range(3):
            self._test_dim_keepdim(op, device, ndim=ndim, dim=None, keepdim=True)

    @skipIfMPS
    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_dim_single(self, device, op: ReductionOpInfo):
        # [FUNCTION DOC] Checks that a single integer dim (including a negative
        # one) reduces exactly that dimension, for ranks 0 to 3. Skipped on MPS.
        """Tests that dim=i reduces dimension i."""
        self._test_dim_keepdim(op, device, ndim=0, dim=0)
        self._test_dim_keepdim(op, device, ndim=1, dim=0)
        self._test_dim_keepdim(op, device, ndim=2, dim=-1)
        self._test_dim_keepdim(op, device, ndim=3, dim=1)

    @skipIfMPS
    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_dim_single_keepdim(self, device, op: ReductionOpInfo):
        # [FUNCTION DOC] Checks that a single integer dim with keepdim true
        # reduces that dimension to size 1, for ranks 0 to 3. Skipped on MPS.
        """Tests that dim=i, when keepdim=True, reduces dimension i to size 1."""
        self._test_dim_keepdim(op, device, ndim=0, dim=0, keepdim=True)
        self._test_dim_keepdim(op, device, ndim=1, dim=0, keepdim=True)
        self._test_dim_keepdim(op, device, ndim=2, dim=-1, keepdim=True)
        self._test_dim_keepdim(op, device, ndim=3, dim=1, keepdim=True)

    @ops(filter(lambda op: op.supports_multiple_dims, reduction_ops), dtypes=OpDTypes.none)
    def test_dim_empty(self, device, op: ReductionOpInfo):
        # [FUNCTION DOC] For ops that support multiple dims, checks that an empty
        # dim list gives the expected shape (as computed by the helper, which
        # treats it as reducing nothing except for vector_norm), for ranks 0 and 2.
        """Tests that dim=[] is a no-op"""
        self._test_dim_keepdim(op, device, ndim=0, dim=[])
        self._test_dim_keepdim(op, device, ndim=2, dim=[])

    @ops(filter(lambda op: op.supports_multiple_dims, reduction_ops), dtypes=OpDTypes.none)
    def test_dim_empty_keepdim(self, device, op: ReductionOpInfo):
        # [FUNCTION DOC] Same as the empty dim list test but with keepdim
        # true, for ranks 0 and 2; only ops that support multiple dims.
        """Tests that dim=[], when keepdim=True, is a no-op"""
        self._test_dim_keepdim(op, device, ndim=0, dim=[], keepdim=True)
        self._test_dim_keepdim(op, device, ndim=2, dim=[], keepdim=True)

    @skipIfMPS
    @ops(filter(lambda op: op.supports_multiple_dims, reduction_ops), dtypes=OpDTypes.none)
    def test_dim_multi(self, device, op: ReductionOpInfo):
        # [FUNCTION DOC] For ops supporting multiple dims, checks that a list of dims reduces
        # those dimensions (rank 1 with one dim, rank 3 with two dims). Skipped on MPS.
        """Tests that dim=[i, j, ...] reduces dimensions i, j, ...."""
        self._test_dim_keepdim(op, device, ndim=1, dim=[0])
        self._test_dim_keepdim(op, device, ndim=3, dim=[0, 2])

    @skipIfMPS
    @ops(filter(lambda op: op.supports_multiple_dims, reduction_ops), dtypes=OpDTypes.none)
    def test_dim_multi_keepdim(self, device, op: ReductionOpInfo):
        # [FUNCTION DOC] Same as the multi-dim test but with keepdim
        # true, so the reduced dimensions become size 1. Skipped on MPS.
        """Tests that dim=[i, j, ...], when keepdim=True, reduces dimensions i, j, .... to size 1."""
        self._test_dim_keepdim(op, device, ndim=1, dim=[0], keepdim=True)
        self._test_dim_keepdim(op, device, ndim=3, dim=[0, 2], keepdim=True)

    @skipIfMPS
    @ops(filter(lambda op: op.supports_multiple_dims, reduction_ops), dtypes=OpDTypes.none)
    def test_dim_multi_unsorted(self, device, op: ReductionOpInfo):
        # [FUNCTION DOC] For ops supporting multiple dims, checks that an unsorted
        # dim list on a rank 4 input gives the right shape. Skipped on MPS.
        """Tests that operator correctly handles unsorted dim list."""
        self._test_dim_keepdim(op, device, ndim=4, dim=[3, 0, 2])

    @skipIfMPS
    @ops(filter(lambda op: op.supports_multiple_dims, reduction_ops), dtypes=OpDTypes.none)
    def test_dim_multi_unsorted_keepdim(self, device, op: ReductionOpInfo):
        # [FUNCTION DOC] Same as the unsorted dim test but with keepdim true. Skipped on MPS.
        """Tests that operator correctly handles unsorted dim list when keepdim=True."""
        self._test_dim_keepdim(op, device, ndim=4, dim=[3, 0, 2], keepdim=True)

    @skipIfMPS
    @ops(filter(lambda op: op.supports_multiple_dims, reduction_ops), dtypes=OpDTypes.none)
    def test_dim_multi_duplicate(self, device, op: ReductionOpInfo):
        # [FUNCTION DOC] For ops supporting multiple dims, checks that a dim
        # list containing a repeated entry raises RuntimeError. Skipped on MPS.
        """Tests that an error is raised if dim has duplicate entries."""
        with self.assertRaises(RuntimeError):
            self._test_dim_keepdim(op, device, ndim=3, dim=[0, 1, 1, 2])

    @ops(filter(lambda op: not op.supports_multiple_dims, reduction_ops), dtypes=OpDTypes.none)
    def test_dim_multi_unsupported(self, device, op: ReductionOpInfo):
        # [FUNCTION DOC] For ops that declare they do not support multiple
        # dims, checks that passing a list of dims raises TypeError.
        """Tests that ops claiming to not support multi dim actually don't."""
        with self.assertRaises(TypeError):
            self._test_dim_keepdim(op, device, ndim=3, dim=[0, 2])

    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_dim_offbounds(self, device, op: ReductionOpInfo):
        # [FUNCTION DOC] Checks that a dim equal to the input rank
        # (out of bounds) raises IndexError for every reduction op.
        """Tests that passing an off-bounds dim throws"""
        with self.assertRaises(IndexError):
            self._test_dim_keepdim(op, device, ndim=2, dim=2)

    @skipIfMPS
    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_dim_ndim_limit(self, device, op: ReductionOpInfo):
        # [FUNCTION DOC] Checks that reducing along a specific dim of a 65-dimensional float
        # tensor raises RuntimeError mentioning the 64-dimension limit. Skipped on MPS.
        """Tests that an exception is raised when reducing a tensor with more
        than 64 dims along some specific dimensions. dim=None is ok"""
        t = make_tensor([1] * 65, dtype=torch.float, device=device)
        with self.assertRaisesRegex(RuntimeError, "only tensors with up to 64 dims are supported"):
            op(t, dim=0)

    @ops(filter(lambda op: op.identity is not None, reduction_ops), dtypes=OpDTypes.supported)
    def test_identity(self, device, dtype, op: ReductionOpInfo):
        # [FUNCTION DOC] For ops with a defined identity value, builds a tensor whose
        # odd positions are filled with the identity and asserts reducing the full
        # tensor equals reducing only the even positions. Uses each supported dtype.
        """Tests that the identity value is an identity for the operator"""
        t = make_tensor((10,), dtype=dtype, device=device)
        t[1::2] = op.identity
        args, kwargs = next(op.generate_args_kwargs(t))
        result = op(t[::2], *args, **kwargs)
        result_with_identity = op(t, *args, **kwargs)
        self.assertEqual(result, result_with_identity, """
        Adding identity value to the input tensor should not change the result.
        """)

    # TODO(@heitorschueroff) Update these to use the nan_policy kwarg once
    # it is added to reduction operators.

    @ops(filter(lambda op: op.nan_policy == 'propagate', reduction_ops), dtypes=OpDTypes.supported,
         allowed_dtypes=floating_and_complex_types_and(torch.bfloat16, torch.float16))
    def test_nan_policy_propagate(self, device, dtype, op: ReductionOpInfo):
        # [FUNCTION DOC] For ops whose nan policy is propagate, checks
        # that an input containing a single nan yields a nan result.
        # Floating and complex dtypes only (including half and bfloat16).
        """Tests that nan is propagated to the output by default"""
        t = make_tensor((5,), dtype=dtype, device=device)
        t[2] = torch.nan
        args, kwargs = next(op.generate_args_kwargs(t))
        result = op(t, *args, **kwargs)
        self.assertTrue(result.isnan())

    @ops(filter(lambda op: op.nan_policy == 'omit', reduction_ops), dtypes=OpDTypes.supported,
         allowed_dtypes=floating_and_complex_types_and(torch.bfloat16, torch.float16))
    def test_nan_policy_omit(self, device, dtype, op: ReductionOpInfo):
        # [FUNCTION DOC] For ops whose nan policy is omit, checks that filling
        # every odd position with nan gives the same result as reducing only
        # the even (non-nan) positions. Floating and complex dtypes only.
        """Tests that NaN values do not affect the result."""
        t = make_tensor((10,), dtype=dtype, device=device)
        t[1::2] = torch.nan
        args, kwargs = next(op.generate_args_kwargs(t))
        result = op(t[::2], *args, **kwargs)
        result_with_nan = op(t, *args, **kwargs)
        self.assertEqual(result, result_with_nan)

    @skipIfMPS
    @ops(reduction_ops, dtypes=OpDTypes.supported)
    def test_result_dtype(self, device, dtype, op: ReductionOpInfo):
        # [FUNCTION DOC] Checks the result dtype of each op on all supported dtypes.
        # Integral inputs: promotion to float or to int64 when the op declares so; otherwise
        # the op's declared result dtype if any; complex-to-real ops map complex to the
        # matching real dtype; else the result dtype equals the input dtype. Skipped on MPS.
        """Tests that the result has the correct dtype"""
        t = make_tensor((5,), dtype=dtype, device=device)
        args, kwargs = next(op.generate_args_kwargs(t))
        result: torch.Tensor = op(t, *args, **kwargs)
        is_integral = dtype in integral_types_and(torch.bool)
        if op.promotes_int_to_float and is_integral:
            self.assertTrue(torch.is_floating_point(result))
        elif op.promotes_int_to_int64 and is_integral:
            self.assertEqual(result.dtype, torch.int64)
        elif op.result_dtype is not None:
            self.assertEqual(result.dtype, op.result_dtype)
        elif op.complex_to_real:
            _complex_to_real_dtype_map = {
                torch.complex128: torch.float64,
                torch.complex64: torch.float32,
                torch.complex32: torch.float16,
            }
            self.assertEqual(result.dtype, _complex_to_real_dtype_map.get(dtype, dtype))
        else:
            self.assertEqual(result.dtype, dtype)

    @skipIfMPS
    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_empty_tensor_empty_slice(self, device, op: ReductionOpInfo):
        # [FUNCTION DOC] Checks reduction over an empty slice of a tensor with shape
        # zero by two by three, along dim 0 (and a dim list when multiple dims are
        # supported). Expected: ops with an identity return it; ops that promote int
        # to float return nan; otherwise an error is expected, RuntimeError for
        # python reference ops and IndexError for the others. Skipped on MPS.
        # KNOWN ISSUE: the loop iterable is a conditional expression over the whole list
        # concatenation, so the whole list is conditional and the loop body never runs for
        # ops without multi-dim support (the plain dim 0 case is not tested for them).
        """Tests for consistent behavior when reducing over an empty slice.

        The rules for reducing over an empty slice are as follows:
            - Return the identity value if the operator has one
            - Otherwise, return NaN if the operator promotes integral dtype to
              floating point dtypes.
            - Otherwise, raise an error

        See discussion here https://github.com/pytorch/pytorch/issues/61901
        """
        t = make_tensor((0, 2, 3), dtype=torch.float, device=device)
        for dim in [0] + [[0, 2]] if op.supports_multiple_dims else []:
            args, kwargs = next(op.generate_args_kwargs(t, dim=dim))
            if op.identity is not None:
                # Reducing along empty slice should return identity
                result = op(t, *args, dim=dim, **kwargs)
                self.assertEqual(result, torch.full_like(result, op.identity))
            elif op.promotes_int_to_float:
                # Reducing along empty slice should return NaN
                result = op(t, *args, dim=dim, **kwargs)
                self.assertEqual(result, torch.full_like(result, torch.nan))
            else:
                # Reducing along empty slice should raise an error
                if isinstance(op, ReductionPythonRefInfo):
                    # ref reductions throw RuntimeError for this
                    with self.assertRaises(RuntimeError):
                        op(t, *args, dim=dim, **kwargs)
                else:
                    with self.assertRaises(IndexError):
                        op(t, *args, dim=dim, **kwargs)

    @skipIfMPS
    @ops(reduction_ops, dtypes=OpDTypes.none)
    def test_empty_tensor_nonempty_slice(self, device, op: ReductionOpInfo):
        # [FUNCTION DOC] Checks that reducing along a non-empty dimension of a zero-by-two-by-three
        # tensor returns a tensor whose shape matches the expected reduced shape. Skipped on MPS.
        # KNOWN ISSUE: same precedence problem as the empty-slice test:
        # the dims iterable is a conditional over the whole list
        # expression, so ops without multi-dim support run no iterations.
        """Tests that reducing a nonempty slice of an empty tensor returns an
        empty tensor with the dimensions reduced."""
        t = make_tensor((0, 2, 3), dtype=torch.float, device=device)
        for dim in [1] + [[1, 2]] if op.supports_multiple_dims else []:
            args, kwargs = next(op.generate_args_kwargs(t, dim=dim))
            result = op(t, *args, dim=dim, **kwargs)
            self.assertEqual(result.shape, _reduced_shape(t.shape, dim=dim))

    def _test_noncontiguous(self, op: ReductionOpInfo, t: torch.Tensor, **reduction_kwargs):
        # [FUNCTION DOC] HELPER: Given a noncontiguous tensor, assert (raising
        # AssertionError otherwise) that it is not contiguous, then for each generated
        # args/kwargs sample compare op applied to it with op applied to its contiguous
        # copy. Extra reduction kwargs are merged into each sample's kwargs.
        """Helper method to test noncontiguous input tensors."""
        if t.is_contiguous():
            raise AssertionError("expected tensor to be non-contiguous")

        t_contig = t.contiguous()
        for args, kwargs in op.generate_args_kwargs(t_contig, **reduction_kwargs):
            kwargs.update(reduction_kwargs)
            result = op(t, *args, **kwargs)
            expected = op(t_contig, *args, **kwargs)
            self.assertEqual(result, expected)

    @skipIfMPS
    @ops(reduction_ops)
    def test_noncontiguous_innermost(self, device, dtype, op: ReductionOpInfo):
        # [FUNCTION DOC] Checks that reducing dim 1 of a strided slice (every second
        # column) of a ten by ten tensor matches the contiguous result. Skipped on MPS.
        """Tests reducing along noncontiguous innermost dimension."""
        t = make_tensor((10, 10), dtype=dtype, device=device, low=-1, high=1)
        self._test_noncontiguous(op, t[:, ::2], dim=1)

    @skipIfMPS
    @ops(reduction_ops)
    def test_noncontiguous_outermost(self, device, dtype, op: ReductionOpInfo):
        # [FUNCTION DOC] Checks that reducing dim 0 of a strided slice (every second
        # row) of a ten by ten tensor matches the contiguous result. Skipped on MPS.
        """Tests reducing along noncontiguous outermost dimension."""
        t = make_tensor((10, 10), dtype=dtype, device=device, low=-1, high=1)
        self._test_noncontiguous(op, t[::2, :], dim=0)

    @skipIfMPS
    @ops(reduction_ops)
    def test_noncontiguous_all(self, device, dtype, op: ReductionOpInfo):
        # [FUNCTION DOC] Checks that reducing all dimensions of a rank 3
        # strided/sliced tensor matches the contiguous result. Skipped on MPS.
        """Tests reducing all dimensions of a noncontiguous tensor."""
        t = make_tensor((5, 5, 5), dtype=dtype, device=device, low=-1, high=1)
        self._test_noncontiguous(op, t[::2, ::3, 1:-1:2])

    @skipIfMPS
    @ops(reduction_ops)
    def test_noncontiguous_transposed(self, device, dtype, op: ReductionOpInfo):
        # [FUNCTION DOC] Checks that reducing all dimensions of a transposed
        # five by five tensor matches the contiguous result. Skipped on MPS.
        """Tests reducing a transposed tensor."""
        t = make_tensor((5, 5), dtype=dtype, device=device, low=-1, high=1)
        self._test_noncontiguous(op, t.T)

    @skipIfMPS
    @ops(reduction_ops)
    def test_noncontiguous_expanded(self, device, dtype, op: ReductionOpInfo):
        # [FUNCTION DOC] Checks that reducing a tensor made by unsqueezing and expanding
        # (stride 0) a singleton dimension matches the contiguous result. Skipped on MPS.
        """Tests reducing a tensor with expanded singleton dimensions."""
        t = make_tensor((2, 3), dtype=dtype, device=device, low=-1, high=1)
        self._test_noncontiguous(op, t.unsqueeze(1).expand(-1, 5, -1))

    # NumPy does not support BFloat16 so we don't test that against reference
    # implementations. We also don't compare dtypes or test for different
    # keepdim because we already have other tests covering those.
    # The test_reference_testing in test_ops.py only uses the samples from
    # sample_inputs_func which do not test as exhaustively as these tests.

    def _test_ref(self, op: ReductionOpInfo, t: torch.Tensor, **reduction_kwargs):
        # [FUNCTION DOC] HELPER: For each generated args/kwargs sample (merged with the
        # extra reduction kwargs), compare op applied to tensor t with op.ref (the numpy
        # reference) applied to a detached CPU numpy copy; dtype exactness is not checked.
        """Compares op against op.ref for the given input and reduction kwargs"""
        for args, kwargs in op.generate_args_kwargs(t, **reduction_kwargs):
            kwargs.update(reduction_kwargs)
            result = op(t, *args, **kwargs)
            expected = op.ref(t.detach().cpu().numpy(), *args, **kwargs)
            self.assertEqual(result, expected, exact_dtype=False)

    @skipIfMPS
    @ops(filter(lambda op: op.ref is not None, reduction_ops),
         allowed_dtypes=all_types_and_complex_and(torch.half, torch.bool))
    def test_ref_scalar_input(self, device, dtype, op: ReductionOpInfo):
        # [FUNCTION DOC] Compares ops that have a numpy reference against it
        # for a 0-dimensional input of each allowed dtype. Skipped on MPS.
        """Compares op against reference for scalar input tensors"""
        self._test_ref(op, make_tensor([], dtype=dtype, device=device))

    @skipIfMPS
    @ops(filter(lambda op: op.ref is not None, reduction_ops),
         allowed_dtypes=all_types_and_complex_and(torch.half, torch.bool))
    def test_ref_small_input(self, device, dtype, op: ReductionOpInfo):
        # [FUNCTION DOC] Compares ops that have a numpy reference against it on a small
        # rank 4 tensor with nonzero values in a small range, for no dim and dims 0, 1
        # and 3, plus dim lists when multiple dims are supported. Skipped on MPS.
        """Compares op against reference for small input tensors"""
        t = make_tensor((5, 3, 4, 2), dtype=dtype, device=device, low=-2, high=2, exclude_zero=True)
        self._test_ref(op, t)
        for dim in [0, 1, 3] + ([[0, 2], [1, 3]] if op.supports_multiple_dims else []):
            self._test_ref(op, t, dim=dim)

    @ops(filter(lambda op: op.ref is not None, reduction_ops),
         allowed_dtypes=[torch.float64])
    def test_ref_large_input_1D(self, device, dtype, op: ReductionOpInfo):
        # [FUNCTION DOC] Compares ops with a numpy reference against it on a float64 one-
        # dimensional tensor with about one million elements, to check numerical stability.
        """Compares op against reference for a large 1D input tensor to check stability"""
        self._test_ref(op, make_tensor((2 ** 20,), dtype=dtype, device=device, low=-1, high=1, exclude_zero=True))

    @ops(filter(lambda op: op.ref is not None, reduction_ops),
         allowed_dtypes=[torch.float64])
    def test_ref_large_input_2D(self, device, dtype, op: ReductionOpInfo):
        # [FUNCTION DOC] Compares ops with a numpy reference against it on a float64
        # tensor of shape 32 by 65536 reduced along dim 1, to exercise parallel reduction.
        """Compares op against reference for a large 2D input tensor to test parallelism"""
        t = make_tensor((32, 2 ** 16), dtype=dtype, device=device, low=-1, high=1, exclude_zero=True)
        self._test_ref(op, t, dim=1)

    @largeTensorTest("8gb")
    @ops(filter(lambda op: op.ref is not None, reduction_ops),
         allowed_dtypes=[torch.float64])
    def test_ref_large_input_64bit_indexing(self, device, dtype, op: ReductionOpInfo):
        # [FUNCTION DOC] Compares ops with a numpy reference against it
        # on a float64 tensor with 275 million elements, which needs
        # 64-bit indexing. Requires about 8GB (large tensor test).
        """Compares op against reference for a very large input tensor that requires 64 bit indexing"""
        self._test_ref(op, make_tensor((275000000,), dtype=dtype, device=device, low=-1, high=1, exclude_zero=True))

    @skipIfMPS
    @ops(filter(lambda op: op.ref is not None, reduction_ops),
         allowed_dtypes=all_types_and_complex_and(torch.half, torch.bool))
    def test_ref_duplicate_values(self, device, dtype, op: ReductionOpInfo):
        # [FUNCTION DOC] Compares ops with a numpy reference against it on a four by four tensor in
        # which the even-row/even-column entries are overwritten with copies of the odd-row/odd-
        # column entries to create duplicate values, for no dim, dim 0 and dim 1. Skipped on MPS.
        """Compares op against reference for input tensors with duplicate values"""
        t = make_tensor((4, 4), dtype=dtype, device=device, low=-2, high=2, exclude_zero=True)
        t[::2, ::2] = t[1::2, 1::2]
        self._test_ref(op, t)
        self._test_ref(op, t, dim=0)
        self._test_ref(op, t, dim=1)

    @skipIfMPS
    @ops(filter(lambda op: op.ref is not None, reduction_ops),
         allowed_dtypes=[torch.float32, torch.complex64])
    def test_ref_extremal_values(self, device, dtype, op: ReductionOpInfo):
        # [FUNCTION DOC] Compares ops with a numpy reference against it
        # on a five element float32/complex64 tensor in which position 2
        # is set in turn to zero, one, nan, inf and -inf. Skipped on MPS.
        """Compares op against reference for input tensors with extremal values"""
        t = make_tensor((5,), dtype=dtype, device=device, exclude_zero=True)
        extremals = [0, 1, nan, inf, -inf]
        for extremal in extremals:
            t[2] = extremal
            self._test_ref(op, t)

    ###########################################################################
    # TODO: Legacy tests - port to ReductionOpInfo
    ###########################################################################

    def test_var_unbiased(self, device):
        # [FUNCTION DOC] Legacy test: checks that var and std with default arguments, an
        # explicit unbiased true or false, and dim 0 agree with each other, and that known
        # small inputs give the expected unbiased (0.5, 1.0) and biased (0.25, 2/3) variances.
        tensor = torch.randn(100, device=device)
        self.assertEqual(tensor.var(0), tensor.var(0, unbiased=True))
        self.assertEqual(tensor.var(), tensor.var(unbiased=True))
        self.assertEqual(tensor.var(unbiased=False), tensor.var(0, unbiased=False))

        tensor = torch.tensor([1.0, 2.0], device=device)
        self.assertEqual(tensor.var(unbiased=True), 0.5)
        self.assertEqual(tensor.var(unbiased=False), 0.25)

        tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
        self.assertEqual(tensor.var(unbiased=True), 1.0)
        self.assertEqual(tensor.var(unbiased=False), 2.0 / 3.0)

        tensor = torch.randn(100, device=device)
        self.assertEqual(tensor.std(0), tensor.std(0, unbiased=True))
        self.assertEqual(tensor.std(), tensor.std(unbiased=True))
        self.assertEqual(tensor.std(unbiased=False), tensor.std(0, unbiased=False))

    def test_var_stability(self, device):
        # [FUNCTION DOC] Legacy test: checks var is numerically stable for two nearby
        # large values (expected 0.03125), both along dim 0 and over the whole tensor.
        tensor = torch.tensor([2281.5, 2281.25], device=device)
        self.assertEqual(tensor.var(dim=0), 0.03125)
        self.assertEqual(tensor.var(), 0.03125)

    def test_sum_dim_reduction_uint8_overflow(self, device):
        # [FUNCTION DOC] Legacy test: sums a uint8 tensor containing a negative literal
        # (wrapped) with uint8 accumulation dtype over all dims, dim 0 and dim 1, checking
        # wrapped-around expected values, and that the out variant writes the same result.
        example = [[-1, 2, 1], [5, 3, 6]]
        x = torch.tensor(example, dtype=torch.uint8, device=device)
        self.assertEqual(x.sum(dtype=torch.uint8).item(), 16)
        self.assertEqual(x.sum(0, dtype=torch.uint8), torch.tensor([4, 5, 7], dtype=torch.uint8, device=device))
        self.assertEqual(x.sum(1, dtype=torch.uint8), torch.tensor([2, 14], dtype=torch.uint8, device=device))
        y = torch.tensor(example, dtype=torch.uint8, device=device)
        torch.sum(x, 0, out=y)
        self.assertEqual(x.sum(0, dtype=torch.uint8), y)

    @skipIfMPS
    def test_dim_reduction_less_than_64(self, device):
        # [FUNCTION DOC] Checks that mean, sum, nansum, std, logsumexp, var
        # and norm each raise RuntimeError about the 64-dimension limit when
        # given a 65-dimensional tensor and dim 64 or -1. Skipped on MPS.
        sizes = [1] * 65
        x = torch.randn(sizes, device=device)
        ops = [torch.mean, torch.sum, torch.nansum, torch.std, torch.logsumexp, torch.std, torch.var,
               torch.norm]
        for op in ops:
            with self.assertRaisesRegex(RuntimeError, "only tensors with up to 64 dims are supported"):
                op(x, dim=64)
            with self.assertRaisesRegex(RuntimeError, "only tensors with up to 64 dims are supported"):
                op(x, dim=-1)

    @dtypes(torch.float, torch.bfloat16)
    @skipIfMPS
    def test_dim_reduction_lastdim(self, device, dtype):
        # [FUNCTION DOC] Checks that norm, argmax and argmin over the last dim
        # give the same result for a strided (every second element) tensor and
        # its contiguous copy, for float and bfloat16. Skipped on MPS.
        x = torch.randn(3, 5, 40, device=device, dtype=dtype)
        x = x[:, :, 0:40:2]
        x2 = x.contiguous()
        ops = [torch.norm, torch.argmax, torch.argmin]
        for op in ops:
            y = op(x, dim=-1)
            y2 = op(x2, dim=-1)
            self.assertEqual(y, y2)

    @skipIfNoSciPy
    @dtypes(torch.float32, torch.double, torch.complex64, torch.complex128)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_logsumexp(self, device, dtype):
        # [FUNCTION DOC] Compares logsumexp over dim 1 with scipy for real and
        # complex dtypes, with a row of -inf (and an inf entry except on CPU,
        # where complex exp of inf differs). Also checks that the out variant
        # writes through to a strided view of a larger tensor. Requires scipy.
        from scipy.special import logsumexp
        a = torch.randn(5, 4, device=device, dtype=dtype)
        # torch.exp(complex(inf, 0)) yields inf+nan*j instead of inf+0*j on CPU which disagrees with CUDA, C++ std::exp,
        # numpy and scipy. Skip inf testing on CPU. Related to https://github.com/pytorch/pytorch/issues/95740
        if torch.device(device) != torch.device('cpu'):
            a[0, 0] = inf
        a[1, :] = -inf
        actual = a.logsumexp(1)
        expected = logsumexp(a.cpu().numpy(), 1)
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(expected, actual)

        # check that out is actually inplace
        b = torch.zeros(5, 2, device=device, dtype=dtype)
        c = b[:, 0]
        torch.logsumexp(a, 1, out=c)
        self.assertEqual(expected, b[:, 0])

    @skipIfNoSciPy
    @skipIfMPS  # promotes to float64, unsupported on MPS
    def test_logsumexp_integral_promotion(self, device):
        # [FUNCTION DOC] Checks that logsumexp of an integer tensor is promoted
        # to floating point and matches scipy. Requires scipy; skipped on MPS.
        from scipy.special import logsumexp
        # check integral inputs is promoted to floating point
        e = torch.randint(-100, 100, [5, 4], device=device)
        actual = e.logsumexp(1).to(torch.float64)
        expected = logsumexp(e.cpu().numpy(), 1)
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(expected, actual)

    @skipIfNoSciPy
    @dtypes(torch.complex64, torch.complex128)
    @dtypesIfMPS(torch.complex64)
    def test_logcumsumexp_complex(self, device, dtype):
        # [FUNCTION DOC] Compares logcumsumexp for complex dtypes along every dim against
        # a slow reference built from scipy logsumexp on growing prefixes (random data
        # and non-normal values), and against hand-written expectations for inf and nan
        # inputs (not run on Windows). Imaginary parts are normalised modulo two pi and
        # set to zero where the real part is -inf before comparing. Requires scipy.
        # logcumsumexp is a more precise way to compute than ``log(cumsum(exp(a)))``
        # and faster than ``[log(sum(exp(a[:i]))) for i in range(a.shape[0])]``
        # the for-loop above should produce similar precision as logcumsumexp (it's just slower),
        # so it can be used as the expected values to check our computation

        # using logsumexp from scipy because by the time of writing this test code,
        # torch.logsumexp has not been implemented for complex numbers
        from scipy.special import logsumexp

        def zero_out_neg_inf(t):
            # [FUNCTION DOC] HELPER: Return a clone of t in which elements that are
            # non-finite and have a negative real part get their imaginary part dropped
            # (real part cast back to the dtype), since that part is indeterminate.
            t = t.clone()
            idx = torch.logical_and(~(torch.isfinite(t)), torch.real(t) < 0)
            t[idx] = torch.real(t[idx]).to(t.dtype)
            return t

        def standardize_phase(t):
            # [FUNCTION DOC] HELPER: Return the tensor with its imaginary
            # part reduced modulo two pi so phases are comparable.
            t = torch.real(t) + 1j * (torch.imag(t) % (2 * np.pi))
            return t

        def logcumsumexp_slow(a, dim):
            # [FUNCTION DOC] HELPER: Reference logcumsumexp along dim: for each
            # prefix length it takes scipy logsumexp of the prefix (keepdims)
            # and concatenates the results along dim, returning a torch tensor.
            res_lst = []
            for i in range(a.size(dim)):
                index = [slice(None, None, None) for _ in range(a.ndim)]
                index[dim] = slice(None, i + 1, None)
                a_inp = a[tuple(index)]
                res_lst.append(logsumexp(a_inp.cpu().numpy(), axis=dim, keepdims=True))
            res = np.concatenate(res_lst, axis=dim)
            return torch.as_tensor(res)

        def compare_logcumsumexp(a, expected=None):
            # [FUNCTION DOC] HELPER: For each dimension of a, compute
            # torch.logcumsumexp and compare with the expected tensor (or the slow
            # scipy reference if expected is None) after phase standardisation and
            # zeroing of -inf imaginary parts; asserts equal shapes and values.
            for i in range(a.ndim):
                actual = torch.logcumsumexp(a, dim=i)
                # if the expected is not given, then revert to scipy's logsumexp
                if expected is None:
                    expected2 = logcumsumexp_slow(a, dim=i)
                else:
                    expected2 = expected

                # move the imaginary values to (0, 2 * pi)
                actual = standardize_phase(actual)
                expected2 = standardize_phase(expected2)

                # zeroing the imaginary part of the element if the real part is -inf
                # as the imaginary part cannot be determined exactly and it does not
                # really matter if we take the exp of the output
                actual = zero_out_neg_inf(actual)
                expected2 = zero_out_neg_inf(expected2)
                self.assertEqual(expected2.shape, actual.shape)
                self.assertEqual(expected2, actual)

        # randomly specified values
        # in this case, scipy.logsumexp should be enough
        a1 = torch.randn((5, 10), dtype=dtype, device=device)
        compare_logcumsumexp(a1)

        # test with some non-normal values
        a2 = torch.tensor([1e3 + 0j, 1e-18 + 1e4j, 1e2 + 1e-8j], dtype=dtype, device=device)
        compare_logcumsumexp(a2)

        # handle special case involving infinites and nans
        # here we don't use scipy.logsumexp as it gives confusing answer on
        # some inf cases
        # see here:
        inf = float('inf')
        nan = float('nan')
        a3_input = torch.tensor([
            -inf + 4j,
            -inf + 1j,
            1.2 + 2.1j,
            1e10 + 1e20j,
            inf + 0j,
            inf + 1j,
            inf + 3j,
            nan + 2j,
        ])
        a3_expected = torch.tensor([
            -inf + 0j,
            -inf + 0j,
            1.2 + 2.1j,
            1e10 + 1e20j,
            inf + 0j,  # scipy's logsumexp gives (inf + 0.7853982j) here, unclear why
            inf + (np.pi / 4) * 1j,  # the imaginary part thanks to some weird behaviour of log(inf + infj)
            complex(inf, nan),
            complex(nan, nan),
        ])
        # windows give strange results on the second-to-last results where it gives inf + pi/4 j
        # instead of inf + nan j
        if not IS_WINDOWS:
            compare_logcumsumexp(a3_input, a3_expected)

        a4_input = torch.tensor([
            complex(-inf, inf),
            complex(-inf, inf),
            -inf + 1j,
            1.2 + 2.1j,
            complex(2.4, inf),
        ])
        a4_expected = torch.tensor([
            -inf + 0j,
            -inf + 0j,
            -inf + 0j,
            1.2 + 2.1j,
            complex(nan, nan),
        ])
        if not IS_WINDOWS:
            compare_logcumsumexp(a4_input, a4_expected)

    @skipIfMPS
    def test_sum_parallel(self, device):
        # [FUNCTION DOC] Compares torch sum with numpy sum on large 0/1 tensors (more
        # elements than the parallel grain size, checked by assertion) over every single
        # dim and the full reduction, to exercise parallel code paths. Skipped on MPS.
        # To use parallel branches we'll need to compare on tensors
        # that are relatively large. Even if this is run on a single
        # core machine these tests will still give you signal on
        # the correctness

        def _run_test(size):
            # [FUNCTION DOC] HELPER: For a list of sizes, for each dim and the full
            # reduction, build a random 0/1 numpy array, assert it has more than 32768
            # elements, and check the torch sum on the device equals the numpy sum exactly.
            for dim in range(len(size) + 1):
                nv = np.round(np.random.rand(*size))  # 0s and 1s
                tv = torch.from_numpy(nv).to(device)
                # Parallelisim is only used if numel is
                # larger than grainsize defined in Parallel.h
                self.assertTrue(tv.numel() > 32768)
                if dim == len(size):
                    nvs = nv.sum()
                    tvs = tv.sum()
                else:
                    nvs = nv.sum(dim)
                    tvs = tv.sum(dim)
                diff = np.abs(nvs - tvs.cpu().numpy()).sum()
                self.assertEqual(diff, 0)

        _run_test([2, 3, 3, 3, 3, 2, 2, 3, 2, 3, 2, 3, 3])
        _run_test([4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        _run_test([1, 32 * 8 * 32 * 8])
        _run_test([1, 32770])

    def test_all_any(self, device):
        # [FUNCTION DOC] Checks all and any on byte and bool tensors of shape ten and five by
        # five: all ones, one zero element, all zeros, and all twos (nonzero counts as true).
        def test(size):
            # [FUNCTION DOC] HELPER: Runs the all/any checks described by the parent
            # test on a ones tensor of the given size, first as bytes and then as
            # bool, mutating one element or the whole tensor between assertions.
            x = torch.ones(*size, device=device).byte()
            self.assertTrue(x.all())
            self.assertTrue(x.any())

            x[3] = 0
            self.assertFalse(x.all())
            self.assertTrue(x.any())

            x.zero_()
            self.assertFalse(x.all())
            self.assertFalse(x.any())

            x.fill_(2)
            self.assertTrue(x.all())
            self.assertTrue(x.any())

            x = torch.ones(*size, device=device).bool()
            self.assertTrue(x.all())
            self.assertTrue(x.any())

            x[3] = False
            self.assertFalse(x.all())
            self.assertTrue(x.any())

        test((10,))
        test((5, 5))

    def test_all_any_with_dim(self, device):
        # [FUNCTION DOC] Checks that all with dim 0 equals the product reduction cast to
        # byte, and any with dim 1 and keepdim equals a clamped sum, on a small uint8 matrix.
        def test(x):
            # [FUNCTION DOC] HELPER: Compares all(dim 0) with prod(dim 0) as bytes and any(dim
            # 1, keepdim) with the clamped sum as bytes, asserting equal shapes and values.
            r1 = x.prod(dim=0, keepdim=False).byte()
            r2 = x.all(dim=0, keepdim=False)
            self.assertEqual(r1.shape, r2.shape)
            self.assertTrue((r1 == r2).all())

            r3 = x.sum(dim=1, keepdim=True).clamp(0, 1).byte()
            r4 = x.any(dim=1, keepdim=True)
            self.assertEqual(r3.shape, r4.shape)
            self.assertTrue((r3 == r4).all())

        test(torch.tensor([[0, 0, 0],
                           [0, 0, 1],
                           [0, 1, 1],
                           [1, 1, 1]], device=device, dtype=torch.uint8))

    def test_numpy_named_args(self, device):
        # [FUNCTION DOC] Checks numpy-style keyword aliases: add with x1/x2 names equals add
        # with input/other, and sum with axis and keepdims equals sum with dim and keepdim.
        x1 = torch.randn(10, device=device)
        x2 = torch.randn(10, device=device)
        res1 = torch.add(input=x1, other=x2)
        res2 = torch.add(x1=x1, x2=x2)
        self.assertEqual(res1, res2)

        x1 = torch.randn(10, 10, 10, device=device)
        res1 = x1.sum(dim=(0, 2), keepdim=True)
        res2 = x1.sum(axis=(0, 2), keepdims=True)
        self.assertEqual(res1, res2)

    # TODO: kill this and replace with common creation ops
    def _make_tensors(self, shape, val_range=(-100, 100), use_floating=True, use_integral=True,
                      use_complex=False) -> dict[str, list[torch.Tensor]]:
        # [FUNCTION DOC] HELPER: Create test tensors of the given shape in three categories
        # returned as a mapping from category name ("cont", "noncont", "slice") to a list of
        # tensors, one per dtype: doubles and floats if use_floating, int64/32/16 if
        # use_integral, complex64/128 if use_complex. Float values follow a clamped scaled
        # normal in val_range; integer values are random in val_range. Asserts the
        # noncontiguous tensors really are noncontiguous and the slice source is contiguous.
        float_types = [torch.double,
                       torch.float]
        int_types = [torch.int64,
                     torch.int32,
                     torch.int16]

        complex_types = [torch.complex64,
                         torch.complex128]

        def make_contiguous(shape, dtype) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Create a contiguous tensor of the given
            # shape and dtype with values in val_range (clamped scaled
            # normal for floats, random integers via apply for others).
            if dtype in float_types:
                val = torch.randn(shape, dtype=dtype)
                val = val * ((val_range[1] - val_range[0]) / (math.pi * 2.0))
                val = val + ((val_range[1] - val_range[0]) / 2.0)
                val = torch.clamp(val, min=val_range[0], max=val_range[1])
                return val
            result = torch.zeros(shape, dtype=dtype)
            result.apply_(lambda x: random.randint(val_range[0], val_range[1]))
            return result

        def make_non_contiguous(shape, dtype) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Create a noncontiguous tensor by copying contiguous
            # data into a strided view of a larger tensor, asserting it is noncontiguous.
            contig = make_contiguous(shape, dtype)
            non_contig = torch.empty(shape + (2, 2), dtype=dtype)[..., 0]
            non_contig = non_contig.select(-1, -1)
            non_contig.copy_(contig)
            self.assertFalse(non_contig.is_contiguous())
            return non_contig

        def make_contiguous_slice(size, dtype) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Create a contiguous (1, size) tensor and
            # assert that its inner slice is contiguous; returns the full tensor.
            contig = make_contiguous((1, size), dtype)
            non_contig = contig[:1, 1:size - 1]
            self.assertTrue(non_contig.is_contiguous())
            return contig

        types = []
        if use_floating:
            types += float_types
        if use_integral:
            types += int_types
        if use_complex:
            types += complex_types
        tensors: dict[str, list[torch.Tensor]] = {"cont": [], "noncont": [], "slice": []}
        for dtype in types:
            tensors["cont"].append(make_contiguous(shape, dtype))
            tensors["noncont"].append(make_non_contiguous(shape, dtype))
            tensors["slice"].append(make_contiguous_slice(sum(list(shape)), dtype))

        return tensors

    # TODO: refactor this to use comparators from common_utils
    def _assert_matches_numpy(self, t, n):
        # [FUNCTION DOC] HELPER: Assert that torch tensor t matches numpy array
        # n in shape and value (equal nan); float32 uses relaxed rtol and atol.
        self.assertEqual(n.shape, t.shape)
        if t.dtype == torch.float:
            self.assertEqual(n, t, rtol=1e-03, atol=1e-05, equal_nan=True)
        else:
            self.assertEqual(n, t, equal_nan=True)

    # TODO: update this and tests that use it to use the device argument properly
    def _test_dim_ops(self, pytorch_op, numpy_op, device,
                      use_floating=True, use_integral=True, use_complex=False):
        # [FUNCTION DOC] HELPER: Compare a torch dim reduction against a numpy one over
        # generated tensors of many shapes and several dim arguments (including dim tuples and
        # a very wide 5 by 400000 shape), using contiguous, noncontiguous and slice tensors.
        # Each result is checked both on the original tensor and on the copy moved to device;
        # numpy warnings are suppressed. Arguments choose which dtype families are generated.
        def do_one(tensors_dict, dim):
            # [FUNCTION DOC] HELPER: For each category and tensor in the
            # mapping, compute the numpy result on the CPU array and compare
            # with the torch result; the slice category always uses dim 0.
            for category, tensors in tensors_dict.items():
                if category == "slice":
                    dim = 0
                for tensor in tensors:
                    # we have no control over NumPy warnings...
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        expected = numpy_op(tensor.cpu().numpy(), dim)
                    actual = pytorch_op(tensor, dim)
                    self._assert_matches_numpy(actual, expected)
                    self._assert_matches_numpy(pytorch_op(tensor.to(device), dim).cpu(), expected)
        do_one(self._make_tensors((5, 400000), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 1)
        do_one(self._make_tensors((3, 5, 7), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 0)
        do_one(self._make_tensors((3, 5, 7), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 1)
        do_one(self._make_tensors((3, 5, 7), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 2)
        do_one(self._make_tensors((100000, ), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), -1)
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 0)
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 1)
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 2)
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), (1, 2))
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), (1, -1))
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), (0, 2))
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), (0, 2, 1))

    @slowTest
    @skipIfMPS
    def test_sum_dim(self, device):
        # [FUNCTION DOC] Slow test: compares torch sum over dims with
        # numpy for float, integral and complex tensors. Skipped on MPS.
        self._test_dim_ops(
            lambda t, d: t.sum(d),
            lambda n, d: n.sum(d), device,
            use_floating=True, use_integral=True, use_complex=True)

    @skipIfMPS
    def test_mean_dim(self, device):
        # [FUNCTION DOC] Compares torch mean over dims with numpy for
        # float and complex tensors (no integers). Skipped on MPS.
        self._test_dim_ops(
            lambda t, d: t.mean(d),
            lambda n, d: n.mean(d),
            device,
            use_integral=False,
            use_complex=True)

    @skipIfMPS
    def test_std_dim(self, device):
        # [FUNCTION DOC] Compares torch std (biased and unbiased) over dims with
        # numpy using the matching ddof, for float tensors. Skipped on MPS.
        for unbiased in [False, True]:
            self._test_dim_ops(
                lambda t, d: t.std(d, unbiased=unbiased),
                lambda n, d: n.std(d, ddof=1 if unbiased else 0),
                device,
                use_integral=False)

    @skipIfMPS
    def test_var_dim(self, device):
        # [FUNCTION DOC] Compares torch var (biased and unbiased) over dims with
        # numpy using the matching ddof, for float tensors. Skipped on MPS.
        for unbiased in [False, True]:
            self._test_dim_ops(
                lambda t, d: t.var(d, unbiased=unbiased),
                lambda n, d: n.var(d, ddof=1 if unbiased else 0),
                device,
                use_integral=False)

    @skipIfNoSciPy
    @skipIfMPS
    def test_logsumexp_dim(self, device):
        # [FUNCTION DOC] Compares torch logsumexp over dims with scipy
        # logsumexp for float tensors. Requires scipy; skipped on MPS.
        from scipy.special import logsumexp
        self._test_dim_ops(
            lambda t, d: t.logsumexp(d),
            lambda n, d: logsumexp(n, d),
            device,
            use_integral=False)

    @skipIfMPS
    def test_mean_int_with_optdtype(self, device):
        # [FUNCTION DOC] Checks that mean of an int64 tensor with an explicit
        # float32 dtype equals mean of the tensor cast to float32. Skipped on MPS.
        a = make_tensor((3, 4, 5), dtype=torch.int64, device=device)

        # If the optional desired output type is given, the input
        # is internally cast.
        a_float = a.to(torch.float32)
        self.assertEqual(a_float.mean(), a.mean(dtype=torch.float32))

    @dtypes(torch.half, torch.bfloat16, torch.float, torch.double)
    @dtypesIfMPS(torch.half, torch.bfloat16, torch.float)
    def test_mean_out_is_alias_of_return(self, dtype, device):
        # [FUNCTION DOC] Checks that mean with an out tensor returns a tensor that
        # aliases the out argument and that its values equal the expected means.
        a = torch.tensor([[[1.0, 1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0, 2.0]], [[3.0, 3.0, 3.0, 3.0]]],
                         dtype=dtype, device=device)
        out = torch.empty((1, 1, 4), dtype=dtype, device=device)

        return_out = torch.mean(a, dim=0, keepdim=True, out=out)
        target = torch.tensor([[[2.0, 2.0, 2.0, 2.0]]], dtype=dtype, device=device)
        self.assertTrue(torch._C._is_alias_of(out, return_out))
        self.assertTrue(torch.allclose(out, target))

    @dtypes(*all_types())
    @skipIfMPS
    def test_mode(self, device, dtype):
        # [FUNCTION DOC] Checks torch.mode on a ten by ten tensor with repeated ones
        # in the first two rows and columns: values are all one and indices are the
        # last occurrence (nine for the first two rows, else one); also checks the out
        # variant, explicit dim, and that the input is unmodified. Skipped on MPS.
        SIZE = 10
        x = torch.arange(1., SIZE * SIZE + 1, device=device, dtype=dtype).clone().resize_(SIZE, SIZE)
        x[:2] = 1
        x[:, :2] = 1
        x0 = x.clone()

        # Pre-calculated results.
        res1val = torch.ones(SIZE, device=device, dtype=dtype)
        # The indices are the position of the last appearance of the mode element.
        res1ind = torch.ones(SIZE, device=device, dtype=torch.long)
        res1ind[0] = SIZE - 1
        res1ind[1] = SIZE - 1

        res2val, res2ind = torch.mode(x, keepdim=False)
        self.assertEqual(res1val, res2val, atol=0, rtol=0)
        self.assertEqual(res1ind, res2ind, atol=0, rtol=0)

        # Test use of result tensor
        res2val = torch.tensor((), device=device, dtype=dtype)
        res2ind = torch.tensor((), device=device, dtype=torch.long)
        torch.mode(x, keepdim=False, out=(res2val, res2ind))
        self.assertEqual(res1val, res2val, atol=0, rtol=0)
        self.assertEqual(res1ind, res2ind, atol=0, rtol=0)

        # Test non-default dim
        res2val, res2ind = torch.mode(x, 0, False)
        self.assertEqual(res1val, res2val, atol=0, rtol=0)
        self.assertEqual(res1ind, res2ind, atol=0, rtol=0)

        # input unchanged
        self.assertEqual(x, x0, atol=0, rtol=0)

    def _test_mode_intervals(self, shape, intervals, device, dtype, v=1):
        # [FUNCTION DOC] HELPER: Build a tensor in which each row is an arange
        # expanded to the shape, set the given intervals to the mode value v, run
        # mode over the last dim, and assert the returned indices point at values
        # equal to the returned values and that all returned values equal v.
        x = torch.arange(0, shape[1], device=device, dtype=dtype).expand(shape)
        x = x.contiguous()
        x[:, v] = intervals[0][0]

        # Set the value of each interval to the mode "v"
        for (beg, end) in intervals:
            x[:, beg:end] = v

        values, indices = torch.mode(x, -1, False)

        # Check whether the returned indices correspond to the returned values
        self.assertTrue((x.gather(1, indices.unsqueeze(1)).t() == values).all())
        # Check whether the returned values are the mode
        self.assertTrue((values == v).all().item())

    @onlyAccelerator
    @skipIfMPS
    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    def test_mode_large(self, device, dtype):
        # [FUNCTION DOC] Accelerator-only test of mode with large slices, covering
        # many rows (more than one block row), the maximum slice size and larger-
        # than-maximum slices that take the naive kernel path. Skipped on MPS.
        # i should be less than (d - 2) / 2
        def testset_for_shape(shape, i):
            # [FUNCTION DOC] HELPER: For a shape and offset, check the mode with
            # a single middle interval and with three discontiguous intervals.
            d = shape[-1]
            # Mode only in the middle.
            self._test_mode_intervals(shape, [(i, d - i)], device, dtype)
            # Mode in discontiguous parts of the input.
            self._test_mode_intervals(shape, [(0, i), (i + 1, d - i - 1), (d - i, d)], device, dtype)

        # More than one line of (65535) thread blocks
        testset_for_shape((65536, 10), 3)

        # Max slice size (2048)
        testset_for_shape((10, 2048), 10)

        # Naive kernel for big slice sizes (> 2048)
        testset_for_shape((10, 4096), 10)

    @skipIfMPS
    def test_mode_boolean(self, device):
        # [FUNCTION DOC] Checks mode of boolean tensors of several shapes: with the true
        # half dominating the mode is true, and with the false half dominating it is false,
        # and that the returned indices point at values equal to the mode. Skipped on MPS.
        # KNOWN ISSUE: the test leaves a stray debug print of the indices tensor in its body.
        shapes = [
            (10, 10),
            (4, 2048),
            (1, 4096),
        ]

        for shape in shapes:
            a = torch.zeros(shape, device=device, dtype=torch.bool)

            a[:, (shape[1] - 1) // 2:] = True
            values, indices = a.mode(-1)
            self.assertEqual(values, torch.ones(shape[0], dtype=torch.bool))
            indexed = a.gather(1, indices.unsqueeze(1)).squeeze(1)
            self.assertEqual(values, indexed)

            a.fill_(False)
            a[:, shape[1] // 2 + 1:] = True
            values, indices = a.mode(-1)
            print(indices)
            self.assertEqual(values, torch.zeros(shape[0], dtype=torch.bool))
            indexed = a.gather(1, indices.unsqueeze(1)).squeeze(1)
            self.assertEqual(values, indexed)


    @expectedFailureMeta  # mode only supports CPU and CUDA device type
    @onlyNativeDeviceTypes
    @skipIfMPS
    def test_mode_wrong_dtype(self, device):
        # [FUNCTION DOC] Marked as an expected failure on the meta device. Checks that torch.mode
        # with out tensors of the wrong value dtype or index dtype raises RuntimeError mentioning
        # the expected scalar type. Native device types only; skipped on MPS.
        def test_for_dtypes(x_ty, v_ty, i_ty, message):
            # [FUNCTION DOC] HELPER: Allocate input, values and indices
            # tensors with the given dtypes and assert that mode with
            # these out tensors raises RuntimeError matching the message.
            x = torch.ones(10, device=device, dtype=x_ty)
            v = torch.ones(10, device=device, dtype=v_ty)
            i = torch.ones(10, device=device, dtype=i_ty)

            with self.assertRaisesRegex(RuntimeError, message):
                torch.mode(x, -1, True, out=(v, i))

        err_msg = "expected scalar type .* but got .* for "
        values_err = err_msg + "values"
        indices_err = err_msg + "indices"

        test_for_dtypes(torch.uint8, torch.int8, torch.long, values_err)
        test_for_dtypes(torch.int8, torch.int16, torch.long, values_err)
        test_for_dtypes(torch.int32, torch.float32, torch.long, values_err)
        test_for_dtypes(torch.float32, torch.float64, torch.long, values_err)

        test_for_dtypes(torch.uint8, torch.uint8, torch.int8, indices_err)
        test_for_dtypes(torch.int8, torch.int8, torch.int16, indices_err)
        test_for_dtypes(torch.int32, torch.int32, torch.float32, indices_err)
        test_for_dtypes(torch.float32, torch.float32, torch.float64, indices_err)

    @onlyAccelerator
    @skipIfMPS
    def test_mode_wrong_device(self, device):
        # [FUNCTION DOC] Accelerator-only: checks that mode on a CPU input with
        # out tensors placed on the accelerator raises RuntimeError complaining
        # about the device of values and of indices. Skipped on MPS.
        # CPU Input Tensor
        x = torch.ones(2)

        with self.assertRaisesRegex(RuntimeError,
                                    "expected device .* but got .* for values"):
            values = torch.tensor([], device=device)
            torch.mode(x, -1, True, out=(values, torch.tensor([], dtype=torch.long)))

        with self.assertRaisesRegex(RuntimeError,
                                    "expected device .* but got .* for indices"):
            indices = torch.tensor([], device=device)
            torch.mode(x, -1, True, out=(torch.tensor([]), indices))

    def test_var_mean_some_dims(self, device):
        # [FUNCTION DOC] Checks that var_mean over every combination of two to four dims of a
        # rank 5 tensor, with biased/unbiased and keepdim settings, equals separate var and mean.
        sizes = (4, 6, 7, 5, 3)
        dims = len(sizes)

        x = torch.rand(sizes, device=device)
        for num_of_dims in range(2, dims):
            dim_list = list(combinations(list(range(dims)), r=num_of_dims))
            for dim in dim_list:
                for unbiased in [False, True]:
                    for keepdim in [False, True]:
                        var1, mean1 = torch.var_mean(x, dim=dim, unbiased=unbiased, keepdim=keepdim)
                        var2 = x.var(dim=dim, unbiased=unbiased, keepdim=keepdim)
                        mean2 = x.mean(dim=dim, keepdim=keepdim)
                        self.assertEqual(var1, var2)
                        self.assertEqual(mean1, mean2)

    # TODO: this should be a generic opinfo test
    def test_all_any_empty(self, device):
        # [FUNCTION DOC] Checks all and any on empty byte
        # and bool tensors: all is true and any is false.
        x = torch.ByteTensor().to(device)
        self.assertTrue(x.all())
        self.assertFalse(x.any())

        x = torch.BoolTensor().to(device)
        self.assertTrue(x.all())
        self.assertFalse(x.any())

    def test_all_issue117215(self, device):
        # [FUNCTION DOC] Regression test for issue 117215: all over dim 0 of
        # a random uint8 tensor equals all over the bool-converted tensor.
        info = torch.iinfo(torch.uint8)
        a = torch.randint(info.min, info.max, (73, 11, 3, 17), dtype=torch.uint8)
        b = torch.all(a, dim=0)
        c = a.to(torch.bool).all(dim=0)
        self.assertEqual(torch.ne(b, c).sum(), 0)

    @dtypesIfCUDA(torch.half, torch.bfloat16, torch.float, torch.double)
    @dtypesIfXPU(torch.half, torch.bfloat16, torch.float, torch.double)
    @dtypes(torch.half, torch.bfloat16, torch.float, torch.double)
    @skipIfMPS
    def test_max_with_inf(self, device, dtype):
        # [FUNCTION DOC] Checks max and amax on a tensor containing both -inf
        # and inf: row-wise results and global results are inf. Skipped on MPS.
        a = torch.tensor([[-inf, -inf, inf, 3], [inf, inf, -inf, -1]], dtype=dtype, device=device)
        self.assertTrue(torch.all(torch.max(a, dim=1).values == inf).item())
        self.assertTrue(torch.all(torch.amax(a, dim=1) == inf).item())
        self.assertTrue(torch.max(a).item() == inf)
        self.assertTrue(torch.amax(a).item() == inf)

    @dtypesIfCUDA(torch.half, torch.bfloat16, torch.float, torch.double)
    @dtypesIfXPU(torch.half, torch.bfloat16, torch.float, torch.double)
    @dtypes(torch.half, torch.float, torch.bfloat16, torch.double)
    @skipIfMPS
    def test_min_with_inf(self, device, dtype):
        # [FUNCTION DOC] Checks min and amin on a tensor containing both -inf
        # and inf: row-wise results and global results are -inf. Skipped on MPS.
        a = torch.tensor([[-inf, -inf, inf, 3], [inf, inf, -inf, -1]], dtype=dtype, device=device)
        self.assertTrue(torch.all(torch.min(a, dim=1).values == (-inf)).item())
        self.assertTrue(torch.all(torch.amin(a, dim=1) == (-inf)).item())
        self.assertTrue(torch.min(a).item() == -inf)
        self.assertTrue(torch.amin(a).item() == -inf)

    def _test_minmax_helper(self, torchfn, reffn, device, dtype, skip_indices=False):
        # [FUNCTION DOC] HELPER: Shared checker for min/max style functions. Compares
        # torchfn against reffn using compare_with_numpy on a random 100 by 100 tensor and
        # on a noncontiguous slice. Unless skip_indices is set, also tests dim 0 and 1 of a
        # square tensor and its transpose against numpy, and when the torch result is a
        # tuple checks the returned indices gather the returned values. For floating dtypes
        # puts a nan at several positions and checks that the value (and index) is nan.
        def create_input(shape, device, dtype):
            # [FUNCTION DOC] HELPER: Create a random input of the shape:
            # normal values for floating dtypes, otherwise random integers
            # (zero or one for bool, within a symmetric thousand range else).
            if dtype.is_floating_point:
                return torch.randn(*shape, device=device, dtype=dtype)
            else:
                low = 0 if dtype == torch.bool else -1000
                high = 2 if dtype == torch.bool else 1000
                return torch.randint(low, high, shape, device=device, dtype=dtype)
        x = create_input((100, 100), device, dtype)
        self.compare_with_numpy(torchfn, reffn, x)
        # non contiguous
        x = create_input((10, 10, 10), device, dtype)
        x = x[:, 4]
        self.compare_with_numpy(torchfn, reffn, x)

        def get_values(x):
            # [FUNCTION DOC] HELPER: Return the first element of x if it
            # is a tuple (values of a values/indices pair), else x itself.
            if isinstance(x, tuple):
                return x[0]
            return x

        # indices
        if not skip_indices:
            size = 5
            x = create_input((size, size), device, dtype)
            inputs = (x, x.t())
            dims = (0, 1)
            for xinp, d in product(inputs, dims):
                self.compare_with_numpy(lambda x: get_values(torchfn(x, d, False)), lambda x: reffn(x, d, keepdims=False), xinp)
                result = torchfn(xinp, d, False)
                if isinstance(result, tuple):
                    v, i = result
                    if d == 1:
                        self.assertEqual(xinp[torch.arange(size), i], v, atol=0, rtol=0)
                    else:
                        self.assertEqual(xinp[i, torch.arange(size)], v, atol=0, rtol=0)
        # nan
        if dtype.is_floating_point:
            for index in (0, 4, 99):
                x = create_input((100,), device, dtype)
                x[index] = nan
                if not skip_indices:
                    result = torchfn(x, 0)
                    v = get_values(result)
                    self.assertEqual(v, nan)
                    if isinstance(result, tuple):
                        i = result[1]
                        self.assertEqual(i, index)
                self.assertEqual(torchfn(x), nan)

    @dtypesIfCPU(torch.float, torch.double, torch.long, torch.bool, torch.half)
    @dtypesIfCUDA(torch.half, torch.float, torch.long, torch.bool)
    @dtypesIfXPU(torch.half, torch.float, torch.long, torch.bool)
    @dtypes(torch.half, torch.float, torch.double)
    @skipIfMPS
    def test_max(self, device, dtype):
        # [FUNCTION DOC] Runs the shared min/max checker for torch.max against
        # numpy amax for the dtypes registered for this device. Skipped on MPS.
        self._test_minmax_helper(torch.max, np.amax, device, dtype)

    @dtypesIfCPU(torch.float, torch.double, torch.long, torch.bool, torch.half)
    @dtypesIfCUDA(torch.half, torch.float, torch.long, torch.bool)
    @dtypesIfXPU(torch.half, torch.float, torch.long, torch.bool)
    @dtypes(torch.half, torch.float, torch.double)
    @skipIfMPS
    def test_min(self, device, dtype):
        # [FUNCTION DOC] Runs the shared min/max checker for torch.min against
        # numpy amin for the dtypes registered for this device. Skipped on MPS.
        self._test_minmax_helper(torch.min, np.amin, device, dtype)

    @dtypesIfCPU(torch.half, torch.float, torch.double, torch.int, torch.long, torch.bool)
    @dtypesIfCUDA(torch.half, torch.float, torch.int, torch.long, torch.bool)
    @dtypesIfXPU(torch.half, torch.float, torch.int, torch.long, torch.bool)
    @dtypes(torch.half, torch.float, torch.double)
    @skipIfMPS
    def test_amin(self, device, dtype):
        # [FUNCTION DOC] Runs the shared min/max checker for
        # torch.amin against numpy amin. Skipped on MPS.
        self._test_minmax_helper(torch.amin, np.amin, device, dtype)

    @dtypesIfCPU(torch.half, torch.float, torch.double, torch.int, torch.long, torch.bool)
    @dtypesIfCUDA(torch.half, torch.float, torch.int, torch.long, torch.bool)
    @dtypesIfXPU(torch.half, torch.float, torch.int, torch.long, torch.bool)
    @dtypes(torch.float, torch.double)
    @skipIfMPS
    def test_amax(self, device, dtype):
        # [FUNCTION DOC] Runs the shared min/max checker for
        # torch.amax against numpy amax. Skipped on MPS.
        self._test_minmax_helper(torch.amax, np.amax, device, dtype)

    @onlyNativeDeviceTypes
    @dtypes(torch.float, torch.double, torch.bfloat16, torch.half)
    @dtypesIfCUDA(torch.half, torch.float, torch.bfloat16)
    @dtypesIfXPU(torch.half, torch.float, torch.bfloat16)
    @dtypesIfMPS(torch.half, torch.float, torch.bfloat16)
    def test_aminmax(self, device, dtype):
        # [FUNCTION DOC] Runs the shared min/max checker for the min output and the max output of
        # torch.aminmax, using wrappers, against numpy amin and amax. Native device types only.

        def _amin_wrapper(x, dim=None, keepdims=False):
            # [FUNCTION DOC] HELPER: Wrapper returning only the min
            # component of aminmax for the given dim and keepdims.
            return torch.aminmax(x, dim=dim, keepdim=keepdims)[0]

        def _amax_wrapper(x, dim=None, keepdims=False):
            # [FUNCTION DOC] HELPER: Wrapper returning only the max
            # component of aminmax for the given dim and keepdims.
            return torch.aminmax(x, dim=dim, keepdim=keepdims)[1]

        self._test_minmax_helper(_amin_wrapper, np.amin, device, dtype)
        self._test_minmax_helper(_amax_wrapper, np.amax, device, dtype)

    @onlyNativeDeviceTypes
    @dtypes(*complex_types())
    @dtypesIfMPS(torch.complex64)
    def test_invalid_0dim_aminmax(self, device, dtype):
        # [FUNCTION DOC] Checks that aminmax with a dim on a zero-dimensional complex tensor
        # raises RuntimeError saying it is not implemented. Native device types only.
        with self.assertRaisesRegex(RuntimeError, 'not implemented'):
            torch.aminmax(torch.tensor(1., dtype=dtype, device=device), dim=0)

    # TODO: bincount isn't a classic reduction -- maybe this test suite is
    #   reductions and summary ops?
    @skipIfMPS
    def test_bincount(self, device):
        # [FUNCTION DOC] Exercises torch.bincount: error cases (negative input, multi-
        # dimensional input, floating input, negative minlength, multi-dimensional or mismatched
        # weights), empty input with and without minlength, the tensor method, no overflow for
        # uint8 input, minlength, float and int weights, noncontiguous inputs and weights,
        # noncontiguous slices, a very large number of bins and a large input. Skipped on MPS.
        # negative input throws
        with self.assertRaisesRegex(RuntimeError, '1-d non-negative integral'):
            torch.bincount(torch.tensor([1, -1], device=device))
        # n-d input, with n > 1 throws
        with self.assertRaisesRegex(RuntimeError, '1-d non-negative integral'):
            torch.bincount(torch.tensor([[1, 2], [3, 4]], device=device))
        # floating input type throws
        with self.assertRaisesRegex(RuntimeError, 'not implemented'):
            torch.bincount(torch.tensor([1., 0.3], device=device))
        # minlength < 0 throws
        with self.assertRaisesRegex(RuntimeError, 'minlength should be >= 0'):
            torch.bincount(torch.tensor([1, 3], device=device),
                           torch.tensor([.2, .2], device=device),
                           minlength=-1)
        # n-d weights, with n > 1 throws
        with self.assertRaisesRegex(RuntimeError, '1-d'):
            torch.bincount(torch.tensor([1, 0], device=device),
                           torch.tensor([[1., 0.3], [1., 0.3]], device=device))
        # input and weights dim mismatch
        with self.assertRaisesRegex(RuntimeError, 'same length'):
            torch.bincount(torch.tensor([1, 0], device=device),
                           torch.tensor([1., 0.3, 0.5], device=device))
        # 1-d input with no elements and default minlength
        self.assertEqual(torch.bincount(torch.tensor([], device=device, dtype=torch.long)),
                         torch.zeros(0, dtype=torch.long, device=device))
        # 1-d input with no elements and specified minlength
        self.assertEqual(torch.bincount(torch.tensor([], device=device, dtype=torch.long), minlength=10),
                         torch.zeros(10, dtype=torch.long, device=device))

        # test tensor method without weights
        long_counts = torch.tensor(
            [0, 3, 2, 1, 3], dtype=torch.uint8, device=device).bincount()
        self.assertEqual(
            torch.tensor([1, 1, 1, 2], dtype=torch.int64, device=device),
            long_counts)
        # test avoiding overflow for uint8 (#76979)
        count_uint8 = torch.tensor([0, 1, 2, 3, 255], dtype=torch.uint8, device=device).bincount()
        count_int16 = torch.tensor([0, 1, 2, 3, 255], dtype=torch.int16, device=device).bincount()
        self.assertEqual(count_uint8, count_int16)
        # test minlength functionality
        int_counts = torch.bincount(
            torch.tensor([1, 1, 1, 1], device=device), minlength=5)
        self.assertEqual(
            torch.tensor([0, 4, 0, 0, 0], dtype=torch.int64, device=device),
            int_counts)
        # test weights
        byte_counts = torch.bincount(
            torch.tensor([0, 1, 1, 1, 4], device=device),
            torch.tensor([.1, .2, .3, .4, .5], device=device))
        self.assertEqual(
            torch.tensor([0.1, 0.9, 0, 0, 0.5], device=device), byte_counts)
        byte_counts = torch.bincount(
            torch.tensor([0, 1, 1, 1, 4], device=device),
            torch.tensor([1, 2, 3, 4, 5], dtype=torch.int8, device=device))
        self.assertEqual(
            torch.tensor([1, 9, 0, 0, 5], device=device, dtype=torch.float64), byte_counts)
        # test non-contiguous inputs and weights
        inputs = torch.tensor([[0, 0], [3, 1], [2, 1], [1, 1], [3, 4]], device=device)
        weights = torch.tensor([[.1, 1], [.2, 2], [.3, 3], [.4, 4], [.5, 5]], device=device)
        for i in [0, 1]:
            if inputs[:, i].is_contiguous():
                raise AssertionError("Inputs are supposed to be non-contiguous")
            if weights[:, i].is_contiguous():
                raise AssertionError("Weights are supposed to be non-contiguous")
        # inputs are non-contiguous but weights are contiguous
        self.assertEqual(inputs[:, 0].bincount(), torch.tensor([1, 1, 1, 2]))
        # inputs and weights are non-contiguous
        self.assertEqual(
            inputs[:, 1].bincount(weights[:, 1]),
            torch.tensor([1, 9, 0, 0, 5], dtype=torch.float32))
        # weights are non-contiguous but inputs are contiguous
        self.assertEqual(inputs[:, 1].contiguous().bincount(weights[:, 1]),
                         torch.tensor([1, 9, 0, 0, 5], dtype=torch.float32))

        # test bincount on non-contiguous slices
        all0s = torch.zeros((32, 2), dtype=torch.int64, device=device)
        self.assertEqual(all0s[:, 0].bincount(), torch.tensor([32]))

        all1s = torch.ones((32, 2), dtype=torch.int64, device=device)
        self.assertEqual(all1s[:, 0].bincount(), torch.tensor([0, 32]))

        # test large number of bins - global memory use
        big_exp = torch.zeros(10000000, device=device)
        big_exp[-1] = 50.0
        big_w = torch.tensor([.5] * 100, device=device)
        big_out = torch.tensor([9999999] * 100, device=device).bincount(big_w)
        self.assertEqual(big_exp, big_out)
        # test large input size
        big_exp = torch.zeros(2, device=device, dtype=torch.int64)
        big_exp[1] = 1000000
        big_out = torch.ones(1000000, dtype=torch.int8, device=device).bincount()
        self.assertEqual(big_exp, big_out)

    # TODO: how many var stability tests are there?
    def test_var_stability2(self, device):
        # [FUNCTION DOC] Checks var stability for two nearby large values along the inner dim,
        # over all elements, and along the outer dim after unsqueeze (expected 0.03125 each).
        tensor = torch.FloatTensor([2281.5, 2281.25]).to(device)

        # Stability for inner dim
        self.assertEqual(tensor.var(0), 0.03125)

        # General stability
        self.assertEqual(tensor.var(), 0.03125)

        # Stability for outer dimensions
        tensor = tensor.unsqueeze(1)
        self.assertEqual(tensor.var(0), 0.03125)

    @dtypes(torch.bfloat16, torch.float16)
    @skipIfMPS
    def test_sum_noncontig_lowp(self, device, dtype) -> None:
        # [FUNCTION DOC] For bfloat16 and float16, sums noncontiguous (strided)
        # tensors of ranks 2 to 5 permuted in every way, over every combination of
        # dims, and compares against the float32 sum cast back. Skipped on MPS.
        dim_sequences = {
            2: [0, 1],
            3: [0, 1, 2],
            4: [0, 1, 2, 3],
            5: [0, 1, 2, 3, 4],
        }

        def create_noncontig_inputs(x, ndim):
            # [FUNCTION DOC] HELPER: Slice every second element in each
            # dimension of x for rank 2 to 5 (returns None for other ranks).
            if ndim == 2:
                return x[::2, ::2]
            elif ndim == 3:
                return x[::2, ::2, ::2]
            elif ndim == 4:
                return x[::2, ::2, ::2, ::2]
            elif ndim == 5:
                return x[::2, ::2, ::2, ::2, ::2]

        def helper(self, shape, reduce_dims, device, dtype):
            # [FUNCTION DOC] HELPER: For every permutation of the dims, build a ones tensor of
            # the shape, make it noncontiguous and permuted, sum over reduce_dims and compare
            # to the float32 reference cast to dtype. The first parameter is the test case.
            for permute_list in list(permutations(dim_sequences[len(shape)], len(shape))):
                x = torch.ones(shape, device=device, dtype=dtype)
                x = create_noncontig_inputs(x, len(shape))
                x_trans = x.permute(permute_list)
                x_sum = torch.sum(x_trans, reduce_dims)
                x_trans_ref = x_trans.float()
                x_sum_ref = torch.sum(x_trans_ref, reduce_dims)
                self.assertEqual(x_sum, x_sum_ref.to(dtype=dtype))

        shapes = [
            (50, 50),
            (50, 50, 50),
            (10, 50, 30, 30),
            (10, 5, 10, 50, 7),
        ]

        for shape in shapes:
            for i in range(1, len(shape) + 1):
                reduce_dims = list(combinations(dim_sequences[len(shape)], i))
                for reduce_dim in reduce_dims:
                    helper(self, shape, reduce_dim, device, dtype)


    @dtypes(torch.bool, torch.double)
    @skipIfMPS
    def test_sum_all(self, device, dtype) -> None:
        # [FUNCTION DOC] Checks that the full sum equals python sum of the list of values
        # for double tensors (small, large, strided) and a bool tensor. Skipped on MPS.
        def check_sum_all(tensor: torch.Tensor) -> None:
            # [FUNCTION DOC] HELPER: Assert that the tensor's sum
            # equals the python sum of its flattened list of values.
            pylist = tensor.reshape(-1).tolist()
            self.assertEqual(tensor.sum(), sum(pylist))

        if dtype != torch.bool:
            check_sum_all(torch.tensor([1, 2, 3, 4, 5], dtype=dtype, device=device))
            check_sum_all(torch.randn(200000, dtype=dtype, device=device))
            check_sum_all(torch.randn(2000, 2, dtype=dtype, device=device)[:, 0])
        else:
            check_sum_all(torch.tensor([True, False, True], dtype=torch.bool, device=device))

    def _test_memory_format_transformations(self, device, input_generator_fn, transformation_fn,
                                            memory_format, compare_data=True, default_is_preserve=False):
        # [FUNCTION DOC] HELPER: Check a memory-format-sensitive transformation (for example
        # clone or to). Asserts memory_format is channels_last or channels_last_3d
        # (AssertionError otherwise). Using a non-dense strided slice of the generated input,
        # checks that the transformation with preserve_format keeps channels-last strides,
        # with contiguous_format gives a contiguous result, and with default arguments
        # preserves or not depending on default_is_preserve; optionally compares data.
        # Finally permutes a random rank 7 tensor ten times checking strides are preserved.

        if memory_format not in (torch.channels_last, torch.channels_last_3d):
            raise AssertionError(f"unexpected memory_format: {memory_format}")

        # xc is a channels last tensor
        xc = input_generator_fn(device)
        # xc is not memory dense, but looks like channels last
        if memory_format == torch.channels_last:
            xc = xc[..., ::2, ::2]
        else:
            xc = xc[..., ::2, ::2, ::2]

        clone = transformation_fn(xc, memory_format=torch.preserve_format)
        self.assertFalse(clone.is_contiguous())
        self.assertTrue(clone.is_contiguous(memory_format=memory_format))
        self.assertFalse(xc.is_contiguous())
        self.assertFalse(xc.is_contiguous(memory_format=memory_format))
        if compare_data:
            self.assertEqual(xc, clone.to(xc))

        xc = input_generator_fn(device)
        clone = transformation_fn(xc, memory_format=torch.contiguous_format)
        self.assertTrue(clone.is_contiguous())
        self.assertFalse(clone.is_contiguous(memory_format=memory_format))
        if compare_data:
            self.assertEqual(xc, clone.to(xc))

        xc = input_generator_fn(device)
        clone = transformation_fn(xc)

        if default_is_preserve:
            self.assertFalse(clone.is_contiguous())
            self.assertTrue(clone.is_contiguous(memory_format=memory_format))
        else:
            self.assertTrue(clone.is_contiguous())
            self.assertFalse(clone.is_contiguous(memory_format=memory_format))
        if compare_data:
            self.assertEqual(xc, clone.to(xc))

        x = torch.randn((3, 4, 5, 6, 7, 8, 9), device=device)
        for _ in range(10):
            permutation = list(range(len(x.shape)))
            random.shuffle(permutation)
            x = x.permute(permutation)
            self.assertEqual(x.stride(), transformation_fn(x, memory_format=torch.preserve_format).stride())

    @dtypes(torch.double)
    @skipIfMPS
    def test_sum_out(self, device, dtype: torch.dtype) -> None:
        # [FUNCTION DOC] Checks that torch.sum with an out argument, over one dim and
        # over two dims, matches the non-out result for double tensors. Skipped on MPS.
        x = torch.rand(100, 100, dtype=dtype, device=device)
        res1 = torch.sum(x, 1)
        res2 = torch.tensor((), dtype=dtype, device=device)
        torch.sum(x, 1, out=res2)
        self.assertEqual(res1, res2)
        x = torch.rand(100, 100, 100, dtype=dtype, device=device)
        res1 = x.sum(2).sum(1)
        res2 = torch.tensor((), dtype=dtype, device=device)
        torch.sum(x, (2, 1), out=res2)
        self.assertEqual(res1, res2)

    @onlyAccelerator
    @skipIfMPS
    @dtypes(torch.float16, torch.float32)
    def test_prod_gpu(self, device, dtype):
        # [FUNCTION DOC] Accelerator-only: checks that prod of fp16 or fp32
        # inputs with fp16 or fp32 output dtype gives 2592 for a fixed five
        # element tensor, both as a function and a method. Skipped on MPS.
        x = torch.tensor([2, 3, 6, 9, 8], dtype=dtype, device=device)

        # Check all combinations: fp16 input - fp16 output, fp16 input - fp32
        # output, fp32 input - fp16 output, fp32 input - fp32 output
        for dtype_output in [torch.float16, torch.float32]:
            result_expected = torch.tensor(2592, dtype=dtype_output, device=device)
            output = torch.prod(x, dtype=dtype_output)
            self.assertEqual(output, result_expected)

            output = x.prod(dtype=dtype_output)
            self.assertEqual(output, result_expected)

    @dtypes(torch.float)
    @skipIfMPS
    def test_prod(self, device, dtype):
        # [FUNCTION DOC] Checks prod over dim 1 of a random float
        # matrix equals the out-variant result. Skipped on MPS.
        x = torch.rand(100, 100, dtype=dtype, device=device)
        res1 = torch.prod(x, 1)
        res2 = torch.tensor((), dtype=dtype, device=device)
        torch.prod(x, 1, out=res2)
        self.assertEqual(res1, res2)

    @dtypes(torch.float16, torch.bfloat16)
    @skipIfMPS
    def test_prod_lowp(self, device, dtype):
        # [FUNCTION DOC] Checks prod over dims 0 and 1 for fp16 and bfloat16
        # equals the float32 result cast to the dtype. Skipped on MPS.
        x = torch.rand(100, 100, dtype=dtype, device=device)
        x_ref = x.float()
        res1 = torch.prod(x, 1)
        res2 = torch.prod(x_ref, 1)
        self.assertEqual(res1, res2.to(dtype=dtype))
        res1 = torch.prod(x, 0)
        res2 = torch.prod(x_ref, 0)
        self.assertEqual(res1, res2.to(dtype=dtype))

    def test_prod_bool(self, device):
        # [FUNCTION DOC] Checks prod with bool dtype for several bool lists
        # (including an empty one and a 256 element all-false list from issue
        # 127866) against numpy prod, with and without an explicit bool dtype.
        vals = [
            [True, True],
            [True, False],
            [False, False],
            [],
            [False] * 256,  # https://github.com/pytorch/pytorch/issues/127866
        ]
        for val in vals:
            result = torch.prod(torch.tensor(val, device=device), dtype=torch.bool).item()
            expect = np.prod(np.array(val), dtype=bool)
            self.assertEqual(result, expect)

            result = torch.prod(torch.tensor(val, device=device)).item()
            expect = np.prod(np.array(val))
            self.assertEqual(result, expect)

    @onlyAccelerator
    @skipIfMPS
    def test_max_mixed_devices(self, device):
        # [FUNCTION DOC] Accelerator-only: checks that max and amax raise RuntimeError when
        # the input is on CPU but the out tensors are on the accelerator. Skipped on MPS.
        a = torch.randn(10)
        if torch.accelerator.is_available():
            values = torch.randn(10).to(device)
            indices = torch.LongTensor().to(device)
            self.assertRaises(RuntimeError,
                              lambda: torch.max(a, 0, out=(values, indices)))
            self.assertRaises(RuntimeError,
                              lambda: torch.amax(a, 0, out=values))

    @onlyAccelerator
    @skipIfMPS
    def test_min_mixed_devices(self, device):
        # [FUNCTION DOC] Accelerator-only: checks that min and amin raise RuntimeError when
        # the input is on CPU but the out tensors are on the accelerator. Skipped on MPS.
        a = torch.randn(10)
        if torch.accelerator.is_available():
            values = torch.randn(10).to(device)
            indices = torch.LongTensor().to(device)
            self.assertRaises(RuntimeError,
                              lambda: torch.min(a, 0, out=(values, indices)))
            self.assertRaises(RuntimeError,
                              lambda: torch.amin(a, 0, out=values))

    # TODO: consider refactoring with bincount test
    @skipIfMPS  # AssertionError: UserWarning not triggered by <lambda>
    def test_bucketization(self, device):
        # [FUNCTION DOC] Exercises bucketize and searchsorted: 3-D values with out and right
        # variants, float boundaries with int32 output, empty values, nan values, type
        # promotion with noncontiguous inputs (expects a noncontiguous UserWarning except on
        # xla), scalar inputs, error cases (dimension mismatches, wrong out dtype, bad side,
        # wrong sorter size/dtype/range) and bfloat16 values/boundaries on CPU. Skipped on MPS.
        values_1d = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], device=device)
        values_3d = torch.tensor([[[1, 3, 5], [2, 4, 6]], [[1, 2, 3], [4, 5, 6]]], device=device)

        # simple 1d boundary and 3d input value
        boundaries = torch.tensor([1, 2, 3, 4, 5, 6], device=device)
        expected_result = torch.tensor([[[0, 2, 4], [1, 3, 5]], [[0, 1, 2], [3, 4, 5]]], device=device)
        output = torch.empty(2, 2, 3, device=device, dtype=torch.int64)
        self.assertEqual(torch.bucketize(values_3d, boundaries), expected_result)
        self.assertEqual(torch.bucketize(values_3d, boundaries, out=output), expected_result)
        expected_result = torch.tensor([[[1, 3, 5], [2, 4, 6]], [[1, 2, 3], [4, 5, 6]]], device=device)
        self.assertEqual(torch.bucketize(values_3d, boundaries, right=True), expected_result)
        self.assertEqual(torch.bucketize(values_3d, boundaries, out=output, right=True), expected_result)

        # simple float 1d boundary and 1d input with output int32 type
        for dtype in [torch.float32, torch.float16]:
            values_1d_float = values_1d.to(dtype)
            boundaries = torch.tensor([0.9, 1, 2, 2, 3, 3, 4, 4.1, 9, 9], device=device, dtype=dtype)
            expected_result = torch.tensor([1, 2, 4, 6, 8, 8, 8, 8, 8], device=device, dtype=torch.int32)
            self.assertEqual(torch.searchsorted(boundaries, values_1d_float, out_int32=True), expected_result)
            self.assertEqual(torch.bucketize(values_1d_float, boundaries, out_int32=True), expected_result)

        # multiple dimension input with 0 elements
        boundaries = torch.tensor([1, 2, 3, 4, 5, 6], device=device, dtype=torch.int64)
        values_0_el = torch.tensor([[[]]], device=device, dtype=torch.int64)
        expected_result = values_0_el.to(torch.int64)
        self.assertEqual(torch.searchsorted(boundaries, values_0_el), expected_result)
        self.assertEqual(torch.bucketize(values_0_el, boundaries), expected_result)

        # nan input
        values_nan = torch.tensor([1.0, float('nan'), 2.0, float('nan')], device=device, dtype=highest_precision_float(device))
        boundaries = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device, dtype=highest_precision_float(device))
        expected_result = torch.tensor([1, 4, 2, 4], device=device)
        self.assertEqual(torch.searchsorted(boundaries, values_nan), expected_result)
        expected_result = torch.tensor([2, 4, 3, 4], device=device)
        self.assertEqual(torch.searchsorted(boundaries, values_nan, right=True), expected_result)
        self.assertEqual(torch.searchsorted(boundaries, values_nan, side='right'), expected_result)

        # type promotion and non contiguous tensors
        values_3d_permute = values_3d.permute(2, 1, 0).to(torch.int32)
        boundaries_permute = values_3d.permute(2, 1, 0).to(highest_precision_float(device))
        expected_result = torch.tensor([[[0, 0], [0, 1]], [[2, 0], [0, 1]], [[2, 0], [0, 0]]], device=device)
        if self.device_type != 'xla':
            self.assertWarnsRegex(
                UserWarning, "tensor is non-contiguous",
                lambda: self.assertEqual(torch.searchsorted(boundaries_permute, values_3d_permute), expected_result))
        else:
            # All tensors in XLA is contiguous even doing permute, no warning msg will be generate in XLA
            self.assertEqual(torch.searchsorted(boundaries_permute, values_3d_permute), expected_result)

        # scalar type
        boundaries = torch.tensor([1.5, 2.5, 3.5], device=device)
        expected_result = torch.tensor(1, device=device)
        self.assertEqual(torch.searchsorted(boundaries, 2), expected_result)
        self.assertEqual(torch.bucketize(torch.tensor(2, device=device), boundaries), expected_result)
        expected_result = torch.tensor(3, device=device)
        scalar_tensor_nan = torch.tensor(float('nan'), device=device)
        self.assertEqual(torch.searchsorted(boundaries, scalar_tensor_nan), expected_result)
        self.assertEqual(torch.bucketize(float('nan'), boundaries, right=True), expected_result)

        # invalid input dimensions
        boundaries = torch.tensor([[1, 2, 3], [4, 5, 6]], device=device)
        with self.assertRaisesRegex(
                RuntimeError, "first N-1 dimensions of boundaries tensor and input value tensor must match"):
            torch.searchsorted(boundaries, values_3d)
        with self.assertRaisesRegex(
                RuntimeError, "boundaries tensor must be 1 dimension"):
            torch.bucketize(values_3d, boundaries)
        with self.assertRaisesRegex(
                RuntimeError, "only when boundaries tensor dimension is 1"):
            torch.searchsorted(boundaries, 1)

        # incompatible output tensor's dtype
        def test_output_dtype(dtype, is_int32):
            # [FUNCTION DOC] HELPER: Assert searchsorted raises RuntimeError about a wrong
            # output dtype when out has the given dtype with the given out_int32 flag.
            output = values_1d.to(dtype)
            with self.assertRaisesRegex(
                    RuntimeError, "output tensor's dtype is wrong"):
                torch.searchsorted(values_1d, values_1d, out=output, out_int32=is_int32)

        test_output_dtype(torch.float32, False)
        test_output_dtype(torch.int32, False)
        test_output_dtype(torch.int64, True)

        # invalid side argument
        with self.assertRaisesRegex(RuntimeError, "side can only be 'left' or 'right'"):
            torch.searchsorted(values_1d, values_1d, side='bad')

        # invalid sorter argument, wrong size
        with self.assertRaisesRegex(RuntimeError, "boundary and sorter must have the same size"):
            sequence = torch.rand_like(values_1d, dtype=torch.float)
            _, sorted_idx = torch.sort(sequence)
            torch.searchsorted(sequence, values_1d, sorter=sorted_idx[:-1])

        # invalid sorter argument, is not dtype long
        with self.assertRaisesRegex(RuntimeError, "sorter must be a tensor of long dtype"):
            sequence = torch.rand_like(values_1d, dtype=torch.float)
            _, sorted_idx = torch.sort(sequence)
            torch.searchsorted(sequence, values_1d, sorter=sorted_idx.to(torch.float32))

        # invalid sorter value, out of bound (>= innermost size)
        with self.assertRaisesRegex(RuntimeError, "sorter index out of range"):
            torch.searchsorted(torch.tensor([1, 2, 3]), 2.5, sorter=torch.tensor([0, 1, 3]))

        # invalid sorter value, out of bound (< 0)
        with self.assertRaisesRegex(RuntimeError, "sorter index out of range"):
            torch.searchsorted(torch.tensor([1, 2, 3]), 2.5, sorter=torch.tensor([-1, 1, 2]))

        # scalar type bfloat16
        if self.device_type == 'cpu':
            def test_dtype_bfloat16(values_bf16=False, boundaries_bf16=False):
                # [FUNCTION DOC] HELPER: Run bucketize on float32 values and
                # boundaries, optionally converting values and/or boundaries to
                # bfloat16, with int32 output and compare to the expected indices.
                values_1d_float = values_1d.to(torch.float32)
                boundaries = torch.tensor([0.9, 1, 2, 2, 3, 3, 4, 4.1, 9, 9], device=device, dtype=torch.float32)
                if values_bf16:
                    values_1d_float = values_1d_float.to(torch.bfloat16)
                if boundaries_bf16:
                    boundaries = boundaries.to(torch.bfloat16)
                expected_result = torch.tensor([1, 2, 4, 6, 8, 8, 8, 8, 8], device=device, dtype=torch.int32)
                self.assertEqual(torch.bucketize(values_1d_float, boundaries, out_int32=True), expected_result)

            test_dtype_bfloat16(True, False)
            test_dtype_bfloat16(False, True)
            test_dtype_bfloat16(True, True)

    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    @skipIfMPS
    def test_nansum(self, device, dtype):
        # [FUNCTION DOC] Compares nansum to sum on a copy with nans replaced by zero, for
        # contiguous and noncontiguous 17 by 17 inputs, over dim 0, dim 1 and all dims,
        # for every dtype (nans only injected for floating dtypes). Skipped on MPS.
        args = product(
            (True, False),  # noncontiguous
            (0, 1, None),   # dim
        )
        zero = torch.zeros((), device=device, dtype=dtype)

        for noncontiguous, dim in args:
            # Randomly scale the values
            scale = random.randint(10, 100)
            x = make_tensor((17, 17), device=device, dtype=dtype,
                            low=-scale, high=scale, noncontiguous=noncontiguous)

            if dtype.is_floating_point:
                nan_mask = x < 0.2 * scale
                x_nonan = torch.where(nan_mask, zero, x)
                x[nan_mask] = np.nan
            else:
                x_nonan = x

            dim_kwargs = {} if dim is None else {"dim": dim}
            expect = torch.sum(x_nonan, **dim_kwargs)
            actual = torch.nansum(x, **dim_kwargs)
            self.assertEqual(expect, actual)

    def _test_reduction_function_with_numpy(self, torch_func, np_func, device, dtype,
                                            with_extremal=False, atol=None, rtol=None,
                                            exact_dtype=True, with_keepdim=False):
        # [FUNCTION DOC] HELPER: Compare a torch reduction with its numpy
        # counterpart for random tensors of ranks 0 to 3, over the default (no dim)
        # case and all permutations of all combinations of dims, optionally with
        # keepdim and extremal values; atol, rtol and dtype exactness are forwarded.
        # Test 0-d to 3-d tensors.
        for ndims in range(4):
            shape = _rand_shape(ndims, min_size=5, max_size=10)
            for n in range(ndims + 1):
                for c in combinations(list(range(ndims)), n):
                    for count_dim in permutations(c):
                        # Generate Input.
                        x = _generate_input(shape, dtype, device, with_extremal)

                        if count_dim == ():
                            # Default `dims=None` case
                            self.compare_with_numpy(torch_func, np_func, x, device=None, dtype=None,
                                                    atol=atol, rtol=rtol, exact_dtype=exact_dtype)
                        else:
                            # With `dims: tuple of ints` case
                            if with_keepdim:
                                torch_func_partial = partial(torch_func, keepdim=True, dim=count_dim)
                                np_func_partial = partial(np_func, keepdims=True, axis=count_dim)
                            else:
                                torch_func_partial = partial(torch_func, dim=count_dim)
                                np_func_partial = partial(np_func, axis=count_dim)
                            self.compare_with_numpy(torch_func_partial, np_func_partial, x, device=None, dtype=None,
                                                    atol=atol, rtol=rtol, exact_dtype=exact_dtype)

    @dtypes(*all_types_and_complex_and(torch.half))
    @skipIfMPS
    def test_count_nonzero(self, device, dtype):
        # [FUNCTION DOC] Compares count_nonzero with numpy for many
        # dtypes, without and with extremal values. Skipped on MPS.
        self._test_reduction_function_with_numpy(torch.count_nonzero, np.count_nonzero, device, dtype)
        self._test_reduction_function_with_numpy(torch.count_nonzero, np.count_nonzero, device, dtype, True)

    # TODO: Investigate why the output is not close to numpy.
    def _get_relaxed_tolerances_for(self, dtype):
        # [FUNCTION DOC] HELPER: Return (atol, rtol) relaxed tolerances
        # for float16 and float32 and (None, None) otherwise.
        if dtype == torch.float16:
            atol = 0.4
            rtol = 1e-2
        elif dtype == torch.float32:
            atol = 7e-05
            rtol = 3e-06
        else:
            # Default values
            atol = None
            rtol = None
        return atol, rtol

    def _test_sum_reduction_vs_numpy(self, torch_fn, np_fn, device, dtype, with_keepdim=False, with_extremal=False):
        # [FUNCTION DOC] HELPER: Compare a sum-like reduction with numpy using
        # relaxed tolerances; dtype exactness is turned off for uint8 and, on
        # Windows, for integral dtypes because numpy and torch promote differently.
        def is_integral(dtype):
            # [FUNCTION DOC] HELPER: Return whether dtype is one of the integral dtypes.
            return dtype in integral_types()

        exact_dtype = True
        # On Windows CI, the current version of `numpy` promotes all lower integers
        # dtypes to int32 while `torch` promotes them to int64. Hence we skip on checking
        # the exact dtype.
        # PR : https://github.com/pytorch/pytorch/pull/38628#issuecomment-655905370
        if IS_WINDOWS and is_integral(dtype):
            exact_dtype = False
        # For uint8, numpy promotes to uint64 while torch promotes to int64.
        # So we must skip this as well.
        if dtype == torch.uint8:
            exact_dtype = False

        # TODO: Investigate why the output is not close to numpy.
        atol, rtol = self._get_relaxed_tolerances_for(dtype)
        self._test_reduction_function_with_numpy(torch_fn, np_fn, device, dtype,
                                                 atol=atol, rtol=rtol, exact_dtype=exact_dtype,
                                                 with_keepdim=with_keepdim, with_extremal=with_extremal)

    @onlyNativeDeviceTypes
    @dtypes(*set(all_types_and(torch.half)) - {torch.uint8})
    @skipIfMPS
    def test_sum_vs_numpy(self, device, dtype):
        # [FUNCTION DOC] Compares torch.sum against numpy sum for non-uint8 dtypes, with
        # default, extremal and keepdim settings. Native device types; skipped on MPS.
        self._test_sum_reduction_vs_numpy(torch.sum, np.sum, device, dtype)
        self._test_sum_reduction_vs_numpy(torch.sum, np.sum, device, dtype, with_extremal=True)
        self._test_sum_reduction_vs_numpy(torch.sum, np.sum, device, dtype, with_keepdim=True)

    @onlyNativeDeviceTypes
    @dtypes(*set(all_types_and(torch.half)) - {torch.uint8})
    @skipIfMPS
    def test_nansum_vs_numpy(self, device, dtype):
        # [FUNCTION DOC] Compares torch.nansum against numpy nansum for non-uint8 dtypes,
        # with default, extremal and keepdim settings. Native device types; skipped on MPS.
        self._test_sum_reduction_vs_numpy(torch.nansum, np.nansum, device, dtype)
        self._test_sum_reduction_vs_numpy(torch.nansum, np.nansum, device, dtype, with_extremal=True)
        self._test_sum_reduction_vs_numpy(torch.nansum, np.nansum, device, dtype, with_keepdim=True)

    @dtypes(*all_types_and(torch.half))
    @skipIfMPS
    def test_nansum_out_dtype(self, device, dtype):
        # [FUNCTION DOC] Compares nansum with an explicit output dtype against
        # numpy nansum with the corresponding dtype, across suitable input dtypes
        # (floating if the output is floating, else integral). Skipped on MPS.
        out_dtype = dtype
        inp_dtypes = all_types_and(torch.half) if out_dtype.is_floating_point else integral_types()
        for inp_dtype in inp_dtypes:
            # TODO: Investigate why the output is not close to numpy.
            atol, rtol = self._get_relaxed_tolerances_for(dtype)
            shape = _rand_shape(random.randint(2, 5), min_size=5, max_size=10)
            x = _generate_input(shape, inp_dtype, device, with_extremal=False)
            torch_fn = partial(torch.nansum, dtype=out_dtype)
            np_out_dtype = torch_to_numpy_dtype_dict[out_dtype]
            np_fn = partial(np.nansum, dtype=np_out_dtype)
            self.compare_with_numpy(torch_fn, np_fn, x, device=None, dtype=None, atol=atol, rtol=rtol)

    @dtypes(torch.int32, torch.int64)
    def test_nansum_int_out_dtype_float_input(self, device, dtype):
        # [FUNCTION DOC] Regression test for issue 183318: nansum of float32 inputs containing
        # nan with an int32 or int64 output dtype ignores nans and matches the sum of the nan-
        # to-zero tensor, over dim 1 and over all dims; an all-nan input gives zero.
        # Regression for #183318.
        x = torch.tensor(
            [[float("nan"), 2.0], [3.0, float("nan")]],
            dtype=torch.float32,
            device=device,
        )
        expected = torch.tensor([2, 3], dtype=dtype, device=device)
        self.assertEqual(torch.nansum(x, dim=1, dtype=dtype), expected)

        y = torch.arange(1, 46, device=device, dtype=torch.float32).reshape(5, 9)
        for idx in [(0, 2), (1, 0), (2, 4), (4, 8)]:
            y[idx] = float("nan")
        ref_dim = torch.nan_to_num(y, nan=0.0).sum(dim=1, dtype=dtype)
        self.assertEqual(torch.nansum(y, dim=1, dtype=dtype), ref_dim)
        ref_all = torch.nan_to_num(y, nan=0.0).sum(dtype=dtype)
        self.assertEqual(torch.nansum(y, dtype=dtype), ref_all)

        all_nan = torch.full((6,), float("nan"), device=device, dtype=torch.float32)
        self.assertEqual(
            torch.nansum(all_nan, dtype=dtype),
            torch.zeros((), dtype=dtype, device=device),
        )

    @dtypes(torch.int32, torch.int64)
    @skipIfMPS
    def test_nansum_int_out_dtype_matches_inductor(self, device, dtype):
        # [FUNCTION DOC] Checks that eager nansum with an integer output
        # dtype on float input equals the result under the inductor
        # backend with full graph (issue 183318). Skipped on MPS.
        # Eager/inductor parity for #183318.
        out_dtype = dtype
        x = torch.tensor(
            [[float("nan"), 2.0], [3.0, float("nan")]],
            dtype=torch.float32,
            device=device,
        )
        eager = torch.nansum(x, dim=1, dtype=out_dtype)
        compiled = torch.compile(
            lambda t: torch.nansum(t, dim=1, dtype=out_dtype),
            backend="inductor",
            fullgraph=True,
        )(x)
        self.assertEqual(eager, compiled)

    @dtypes(*all_types_and(torch.half))
    @dtypesIfXPU(torch.half, torch.int8, torch.uint8, torch.float32)
    # Acc issue for other types on xpu, see https://github.com/intel/torch-xpu-ops/issues/2295
    @skipIfMPS
    def test_argminmax_multiple(self, device, dtype):
        # [FUNCTION DOC] Compares argmax and argmin with numpy for all-ones inputs,
        # inputs with a nan, random inputs with extremal values (contiguous and
        # transposed) and two sample cases from issue 41998, and (except half) the
        # index output of max and min over a random dim. Skipped on MPS.
        # Case: All Ones
        t = torch.ones(3, 3, device=device, dtype=dtype)
        self.compare_with_numpy(torch.argmax, np.argmax, t)
        self.compare_with_numpy(torch.argmin, np.argmin, t)

        # Case: With single `nan` present.
        if dtype in floating_types_and(torch.half, torch.bfloat16):
            t[2, 2] = float('nan')
            self.compare_with_numpy(torch.argmax, np.argmax, t)
            self.compare_with_numpy(torch.argmin, np.argmin, t)

        # Case: Randomly Generated Tensors
        for ndims in range(1, 5):
            shape = _rand_shape(ndims, min_size=5, max_size=10)
            for with_extremal in [False, True]:
                for contiguous in [False, True]:
                    # Generate Input.
                    x = _generate_input(shape, dtype, device, with_extremal)

                    if dtype == torch.half:
                        max_val = torch.max(x.to(torch.float))
                        min_val = torch.min(x.to(torch.float))
                    else:
                        max_val = torch.max(x)
                        min_val = torch.min(x)

                    mask = torch.randn(x.shape) > 0.5
                    x[mask] = torch.tensor(max_val + 1, dtype=dtype)

                    mask = torch.randn(x.shape) > 0.5
                    x[mask] = torch.tensor(min_val - 1, dtype=dtype)

                    if not contiguous:
                        x = x.T

                    self.compare_with_numpy(torch.argmax, np.argmax, x, device=None, dtype=None)
                    self.compare_with_numpy(torch.argmin, np.argmin, x, device=None, dtype=None)

                    # Verify indices returned by max and min.
                    if dtype != torch.half:
                        rand_dim = random.randint(0, ndims - 1)
                        self.compare_with_numpy(lambda x: torch.max(x, dim=rand_dim)[1],
                                                lambda x: np.argmax(x, axis=rand_dim), x, device=None, dtype=None)
                        self.compare_with_numpy(lambda x: torch.min(x, dim=rand_dim)[1],
                                                lambda x: np.argmin(x, axis=rand_dim), x, device=None, dtype=None)

        def verify_against_numpy(t):
            # [FUNCTION DOC] HELPER: Compare argmax and argmin over dim 1 with
            # numpy on t and its transpose; also the index output of max and
            # min on the enclosing x variable (not t) when dtype is not half.
            # KNOWN ISSUE: the inner max/min index checks use the
            # outer variable x from the enclosing scope instead of the
            # argument t, so they do not test the two sample tensors.
            # Argmax
            torch_fn = partial(torch.argmax, dim=1)
            np_fn = partial(np.argmax, axis=1)
            self.compare_with_numpy(torch_fn, np_fn, t)
            # Non-contiguous input
            self.compare_with_numpy(torch_fn, np_fn, t.T)

            # Verify indices returned by max.
            if dtype != torch.half:
                self.compare_with_numpy(lambda x: torch.max(x, dim=1)[1], np_fn, x, device=None, dtype=None)
                self.compare_with_numpy(lambda x: torch.max(x, dim=1)[1], np_fn, x.T, device=None, dtype=None)

            # Argmin
            torch_fn = partial(torch.argmin, dim=1)
            np_fn = partial(np.argmin, axis=1)
            self.compare_with_numpy(torch_fn, np_fn, t)
            # Non-contiguous input
            self.compare_with_numpy(torch_fn, np_fn, t.T)

            # Verify indices returned by min.
            if dtype != torch.half:
                self.compare_with_numpy(lambda x: torch.min(x, dim=1)[1], np_fn, x, device=None, dtype=None)
                self.compare_with_numpy(lambda x: torch.min(x, dim=1)[1], np_fn, x.T, device=None, dtype=None)

        # Case: Sample from issue: https://github.com/pytorch/pytorch/issues/41998
        t = torch.tensor([[1, 5],
                          [2, 10],
                          [3, 3]], device=device, dtype=dtype)
        verify_against_numpy(t)

        # Case: Sample from issue: https://github.com/pytorch/pytorch/issues/41998
        t = torch.tensor([[1, 5],
                          [2, 10],
                          [0, 0]], device=device, dtype=dtype)
        verify_against_numpy(t)

    @dtypes(*all_types_and_complex_and(torch.half, torch.bool))
    @skipIfMPS
    def test_all_any_vs_numpy(self, device, dtype):
        # [FUNCTION DOC] Compares all and any with numpy for many dtypes, shapes of rank 0 to 4,
        # transposed and strided inputs, values with and without extremals, all-zero and all-one
        # inputs, with dims and keepdim, out variants (only bool and uint8 are allowed as out,
        # else RuntimeError) and output dtype (uint8 for uint8 input, else bool). Skipped on MPS.
        # Note [all, any uint8 compatibility]: However for compatibility reason,
        # for `uint8`, they return Tensor of same dtype `uint8`.
        # Reference: https://github.com/pytorch/pytorch/pull/47878#issuecomment-747108561
        exact_dtype = dtype != torch.uint8

        def _test_all_any(x):
            # [FUNCTION DOC] HELPER: Compare torch all and any (no dim) against numpy for x.
            self.compare_with_numpy(torch.all, np.all, x)
            self.compare_with_numpy(torch.any, np.any, x)

        def _test_all_any_with_dim(x, dim):
            # [FUNCTION DOC] HELPER: Compare all and any with a given dim
            # against numpy using the exact_dtype setting of the enclosing test.
            torch_fn = partial(torch.all, dim=dim)
            np_fn = partial(np.all, axis=dim)
            self.compare_with_numpy(torch_fn, np_fn, x, exact_dtype=exact_dtype)

            torch_fn = partial(torch.any, dim=dim)
            np_fn = partial(np.any, axis=dim)
            self.compare_with_numpy(torch_fn, np_fn, x, exact_dtype=exact_dtype)

        def _test_out_variant(x, dim):
            # [FUNCTION DOC] HELPER: For bool and uint8 input check the out
            # variants of all and any equal the non-out results; for other
            # dtypes expect RuntimeError saying the result must be bool.
            out = torch.empty_like(x)
            if dtype == torch.bool or dtype == torch.uint8:
                expected = torch.all(x, dim)
                torch.all(x, dim, out=out)
                self.assertEqual(expected, out)

                expected = torch.any(x, dim)
                torch.any(x, dim, out=out)
                self.assertEqual(expected, out)
            else:
                with self.assertRaisesRegex(RuntimeError, "all only supports bool tensor for result, got"):
                    torch.all(x, dim, out=out)

                with self.assertRaisesRegex(RuntimeError, "any only supports bool tensor for result, got"):
                    torch.any(x, dim, out=out)

        def _test_all_any_with_dim_keepdim(x, dim, keepdim):
            # [FUNCTION DOC] HELPER: Compare all and any with dim and keepdim against numpy.
            torch_fn = partial(torch.all, dim=dim, keepdim=keepdim)
            np_fn = partial(np.all, axis=dim, keepdims=keepdim)
            self.compare_with_numpy(torch_fn, np_fn, x, exact_dtype=exact_dtype)

            torch_fn = partial(torch.any, dim=dim, keepdim=keepdim)
            np_fn = partial(np.any, axis=dim, keepdims=keepdim)
            self.compare_with_numpy(torch_fn, np_fn, x, exact_dtype=exact_dtype)

        def _test_output_dtype(x):
            # [FUNCTION DOC] HELPER: Assert the dtype of all/any results
            # (full and dim 0) is uint8 for uint8 input and bool otherwise.
            # This test will fail once the functions return bool output
            # for uint8 input.
            expected_dtype = torch.uint8 if dtype == torch.uint8 else torch.bool
            self.assertEqual(torch.all(x).dtype, expected_dtype)
            self.assertEqual(torch.any(x).dtype, expected_dtype)

            self.assertEqual(torch.all(x, dim=0).dtype, expected_dtype)
            self.assertEqual(torch.any(x, dim=0).dtype, expected_dtype)

        for ndim in range(5):
            shape = _rand_shape(ndim, 1, 5)
            x = _generate_input(shape, dtype, device, with_extremal=False)
            _test_all_any(x)
            _test_all_any(x.T)
            _test_all_any(x[..., ::2])

            x = _generate_input(shape, dtype, device, with_extremal=True)
            _test_all_any(x)
            _test_all_any(x.T)
            _test_all_any(x[..., ::2])

            x = torch.zeros_like(x)
            _test_all_any(x)
            _test_all_any(x.T)
            _test_all_any(x[..., ::2])

            x = torch.ones_like(x)
            _test_all_any(x)
            _test_all_any(x.T)
            _test_all_any(x[..., ::2])
            _test_output_dtype(x)
            for dim in range(ndim):
                x = _generate_input(shape, dtype, device, with_extremal=False)
                _test_all_any_with_dim(x, dim)
                _test_all_any_with_dim(x.T, dim)
                _test_all_any_with_dim(x[..., ::2], dim)
                _test_out_variant(x, dim)
                _test_all_any_with_dim_keepdim(x, dim, keepdim=True)
                _test_all_any_with_dim_keepdim(x, dim, keepdim=False)

                x = _generate_input(shape, dtype, device, with_extremal=True)
                _test_all_any_with_dim(x, dim)
                _test_all_any_with_dim(x.T, dim)
                _test_all_any_with_dim(x[..., ::2], dim)
                _test_out_variant(x, dim)
                _test_all_any_with_dim_keepdim(x, dim, keepdim=True)
                _test_all_any_with_dim_keepdim(x, dim, keepdim=False)

                x = torch.zeros_like(x)
                _test_all_any_with_dim(x, dim)
                _test_all_any_with_dim(x.T, dim)
                _test_all_any_with_dim(x[..., ::2], dim)
                _test_out_variant(x, dim)
                _test_all_any_with_dim_keepdim(x, dim, keepdim=True)
                _test_all_any_with_dim_keepdim(x, dim, keepdim=False)

                x = torch.ones_like(x)
                _test_all_any_with_dim(x, dim)
                _test_all_any_with_dim(x.T, dim)
                _test_all_any_with_dim(x[..., ::2], dim)
                _test_out_variant(x, dim)
                _test_all_any_with_dim_keepdim(x, dim, keepdim=True)
                _test_all_any_with_dim_keepdim(x, dim, keepdim=False)

    # TODO: part of this test covers torch.norm, with should be covered by test_linalg
    @onlyNativeDeviceTypes
    @skipIfMPS
    def test_repeated_dim(self, device):
        # [FUNCTION DOC] Checks that mean, sum, nansum, std, logsumexp, var and norm raise
        # RuntimeError saying a dim appears multiple times when given a repeated dim
        # (including a wrapped negative duplicate). Native device types; skipped on MPS.
        ops = [torch.mean, torch.sum, torch.nansum, torch.std, torch.logsumexp, torch.std, torch.var,
               torch.norm]
        x = torch.randn(3, 3, 3, 3, device=device)

        error_msg = r'appears multiple times in the list of dims'
        for op in ops:
            for dim in [(0, 0), (0, -4)]:
                with self.assertRaisesRegex(RuntimeError, error_msg):
                    op(x, dim=dim)

    # TODO: update this test to compare against NumPy
    @onlyAccelerator
    @skipIfMPS
    def test_var(self, device):
        # [FUNCTION DOC] Accelerator-only: compares var and std on the device with CPU
        # results for a three-dimensional and a one-dimensional tensor. Skipped on MPS.
        cpu_tensor = torch.randn(2, 3, 3)
        device_tensor = cpu_tensor.to(device)
        self.assertEqual(device_tensor.var(), cpu_tensor.var())
        self.assertEqual(device_tensor.var(1), cpu_tensor.var(1))
        self.assertEqual(device_tensor.var(2), cpu_tensor.var(2))
        self.assertEqual(device_tensor.std(), cpu_tensor.std())
        self.assertEqual(device_tensor.std(1), cpu_tensor.std(1))
        self.assertEqual(device_tensor.var(2), cpu_tensor.var(2))

        cpu_tensor = torch.randn(100)
        device_tensor = cpu_tensor.to(device)
        self.assertEqual(device_tensor.var(), cpu_tensor.var())

    # TODO: update this test to compare against NumPy
    @onlyAccelerator
    @skipIfMPS
    def test_var_large_input(self, device):
        # [FUNCTION DOC] Accelerator-only: compares var over the last dim for a
        # large, awkwardly sized tensor on the device against CPU. Skipped on MPS.
        # Large, not-nice input
        cpu_tensor = torch.randn(2 * 32 * 1024 + 1, 2, 67)
        device_tensor = cpu_tensor.to(device)

        self.assertEqual(cpu_tensor.var(2), device_tensor.var(2))

    # TODO: update this to compare against NumPy instead of CPU
    @onlyAccelerator
    @dtypes(torch.double)
    @skipIfMPS
    def test_sum_noncontig(self, device, dtype):
        # [FUNCTION DOC] Accelerator-only: compares full and two-dim sums of a permuted
        # (noncontiguous) double tensor on the device with CPU. Skipped on MPS.
        x = torch.randn(1, 75, 57, 20, dtype=dtype, device=device).permute(0, 3, 1, 2)
        y = x.cpu()
        self.assertEqual(x.sum().cpu(), y.sum())
        self.assertEqual(x.sum(dim=(-1, -2)).cpu(), y.sum(dim=(-1, -2)))
        self.assertEqual(x.sum(dim=(1, 3)).cpu(), y.sum(dim=(1, 3)))

    # TODO: update this to compare against NumPy instead of CPU
    @onlyAccelerator
    @skipIfMPS
    def test_min_max_nan(self, device):
        # [FUNCTION DOC] Accelerator-only: for min, max, amin, amax and their
        # dim variants, checks that nan positions and non-nan values from the
        # device match CPU on a matrix containing one nan. Skipped on MPS.
        tests = [(lambda x: x.min(), 'min'),
                 (lambda x: x.max(), 'max'),
                 (lambda x: x.amin(), 'amin'),
                 (lambda x: x.amax(), 'amax'),
                 (lambda x: x.min(0).values, 'min_dim'),
                 (lambda x: x.max(0).values, 'max_dim'),
                 (lambda x: x.amin(0), 'amin_dim'),
                 (lambda x: x.amax(0), 'amax_dim')]
        for f, name in tests:
            a = torch.arange(25.0).view(5, 5)
            a[2, 2] = nan
            actual = f(a.to(device)).cpu()
            expected = f(a).cpu()
            self.assertEqual(torch.isnan(actual), torch.isnan(expected), msg=lambda msg: f'{msg}\nnans for {name}')
            self.assertEqual(actual[~torch.isnan(actual)],
                             expected[~torch.isnan(expected)], msg=lambda msg: f'{msg}\nnans for {name}')

    # TODO: make this test generic using OpInfos
    @onlyAccelerator
    @skipIfMPS
    def test_sum_cpu_device_mismatch(self, device):
        # [FUNCTION DOC] Accelerator-only: checks that sum with an out tensor on CPU for
        # an input on the device raises RuntimeError naming the expected device,
        # including for half input with float32 dtype on non-CPU devices. Skipped on MPS.
        x = torch.randn(20, dtype=torch.float32, device=device)
        y = torch.randn(1, dtype=torch.float32)

        err_string = f"Expected out tensor to have device {device}, but got cpu instead"

        with self.assertRaisesRegex(RuntimeError, err_string):
            torch.sum(x, dim=[0], dtype=torch.float32, out=y)

        # tests half to float promotion
        if self.device_type != 'cpu':
            x = x.half()
            with self.assertRaisesRegex(RuntimeError, err_string):
                torch.sum(x, dim=[0], dtype=torch.float32, out=y)

    # Assert for illegal dtype would not be raised on XLA
    @onlyNativeDeviceTypes
    @skipIfMPS
    def test_minmax_illegal_dtype(self, device):
        # [FUNCTION DOC] Native device types: checks that max, min, amax and amin accept
        # correct out dtypes and raise RuntimeError about the scalar type or dtype when
        # values or indices out tensors have the wrong dtype. Skipped on MPS.
        x = torch.randn(5, 5, dtype=torch.float32, device=device)
        valid_values = torch.empty(5, dtype=torch.float32, device=device)
        valid_indices = torch.empty(5, dtype=torch.long, device=device)
        illegal_values = torch.empty(5, dtype=torch.int, device=device)
        illegal_indices = torch.empty(5, dtype=torch.double, device=device)
        torch.max(x, dim=0, out=(valid_values, valid_indices))
        torch.min(x, dim=0, out=(valid_values, valid_indices))
        torch.amax(x, dim=0, out=valid_values)
        torch.amin(x, dim=0, out=valid_values)
        rmsg = r'scalar type|dtype'
        with self.assertRaisesRegex(RuntimeError, rmsg):
            torch.max(x, dim=0, out=(illegal_values, valid_indices))
        with self.assertRaisesRegex(RuntimeError, rmsg):
            torch.min(x, dim=0, out=(illegal_values, valid_indices))
        with self.assertRaisesRegex(RuntimeError, rmsg):
            torch.max(x, dim=0, out=(valid_values, illegal_indices))
        with self.assertRaisesRegex(RuntimeError, rmsg):
            torch.min(x, dim=0, out=(valid_values, illegal_indices))
        with self.assertRaisesRegex(RuntimeError, rmsg):
            torch.max(x, dim=0, out=(illegal_values, illegal_indices))
        with self.assertRaisesRegex(RuntimeError, rmsg):
            torch.min(x, dim=0, out=(illegal_values, illegal_indices))

    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    @skipIfMPS
    def test_dim_arg_reduction_scalar(self, device, dtype):
        # [FUNCTION DOC] Checks that argmax and argmin of a zero-dimensional
        # tensor return index zero for default, dim None and dim 0, and a zero-
        # dimensional int64 result when keepdim is set with dim 0. Skipped on MPS.
        example = 4.0

        x = torch.tensor(example, device=device, dtype=dtype)
        self.assertEqual(x.argmax().item(), 0)
        self.assertEqual(x.argmax(dim=None).item(), 0)
        self.assertEqual(x.argmax(dim=0).item(), 0)
        self.assertEqual(x.argmax(dim=0, keepdim=True), torch.tensor(0, dtype=torch.int64))

        x = torch.tensor(example, device=device, dtype=dtype)
        self.assertEqual(x.argmin().item(), 0)
        self.assertEqual(x.argmin(dim=None).item(), 0)
        self.assertEqual(x.argmin(dim=0).item(), 0)
        self.assertEqual(x.argmin(dim=0, keepdim=True), torch.tensor(0, dtype=torch.int64))


    @precisionOverride({torch.float16: 1e-2, torch.bfloat16: 1e-2})
    @dtypes(*set(all_types_and(torch.half, torch.bfloat16)) - {torch.uint8})
    @skipIfMPS
    def test_dim_reduction(self, device, dtype):
        # [FUNCTION DOC] Legacy broad test of reductions on a small two by three int/float
        # matrix: sum, mean (float dtypes), prod, min/amin/argmin and max/amax/argmax with
        # dim, keepdim and noncontiguous slices, with the expected result dtypes (integers
        # accumulate in int64, others keep dtype). Skipped on MPS; uint8 excluded.
        example = [[-1, 2, 1], [5, 3, 6]]

        sum_dtype = {
            torch.bfloat16: torch.bfloat16,
            torch.double: torch.double,
            torch.float: torch.float,
            torch.half: torch.half,
            torch.int64: torch.int64,
            torch.int32: torch.int64,
            torch.int16: torch.int64,
            torch.int8: torch.int64
        }

        # This won't test for 256bit instructions, since we usually
        # only work on 1 cacheline (512bit) at a time and these
        # examples aren't big enough to trigger that.
        x = torch.tensor(example, device=device, dtype=dtype)
        self.assertEqual(x.sum().item(), 16)
        self.assertEqual(x.sum(0), torch.tensor([4, 5, 7], dtype=sum_dtype[dtype]))
        self.assertEqual(x.sum(1), torch.tensor([2, 14], dtype=sum_dtype[dtype]))
        y = torch.tensor(example, device=device, dtype=sum_dtype[dtype])
        torch.sum(x, 0, out=y)
        self.assertEqual(x.sum(0), y)

        # Mean not supported for Int types
        if dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            x = torch.tensor(example, device=device, dtype=dtype)
            self.assertEqual(x.mean().item(), 16.0 / 6)
            self.assertEqual(x.mean(0), torch.tensor([2.0, 2.5, 7.0 / 2], dtype=dtype))
            self.assertEqual(x.mean(1), torch.tensor([2.0 / 3, 14.0 / 3], dtype=dtype))
            self.assertEqual(x.mean(), x.mean((0, 1)))

        prod_dtype = {
            torch.bfloat16: torch.bfloat16,
            torch.double: torch.double,
            torch.float: torch.float,
            torch.float16: torch.float16,
            torch.int64: torch.int64,
            torch.int32: torch.int64,
            torch.int16: torch.int64,
            torch.int8: torch.int64,
        }

        x = torch.tensor(example, device=device, dtype=dtype)
        self.assertEqual(x.prod().item(), -180)
        self.assertEqual(x.prod(0), torch.tensor([-5, 6, 6], dtype=prod_dtype[dtype]))
        self.assertEqual(x.prod(1), torch.tensor([-2, 90], dtype=prod_dtype[dtype]))

        self.assertEqual(x.min().item(), -1)
        self.assertEqual(x.argmin().item(), 0)

        # TODO: torch.min does not support the same operation as argmin
        # for the same case, should we enable it?
        self.assertEqual(x.argmin(dim=None).item(), 0)

        self.assertEqual(x.min(0), (torch.tensor([-1, 2, 1], dtype=dtype),
                                    torch.tensor([0, 0, 0], dtype=torch.int64)))
        self.assertEqual(x.amin(0), torch.tensor([-1, 2, 1], dtype=dtype))
        self.assertEqual(x.argmin(0), torch.tensor([0, 0, 0], dtype=torch.int64))

        self.assertEqual(x.min(dim=0, keepdim=True), (torch.tensor([[-1, 2, 1]], dtype=dtype),
                         torch.tensor([[0, 0, 0]], dtype=torch.int64)))
        self.assertEqual(x.amin(dim=0, keepdim=True), torch.tensor([[-1, 2, 1]], dtype=dtype))
        self.assertEqual(x.argmin(dim=0, keepdim=True), torch.tensor([[0, 0, 0]], dtype=torch.int64))

        self.assertEqual(x.min(1), (torch.tensor([-1, 3], dtype=dtype),
                         torch.tensor([0, 1], dtype=torch.int64)))
        self.assertEqual(x.amin(1), torch.tensor([-1, 3], dtype=dtype))
        self.assertEqual(x.argmin(1), torch.tensor([0, 1], dtype=torch.int64))

        self.assertEqual(x.min(dim=1, keepdim=True), (torch.tensor([[-1], [3]], dtype=dtype),
                         torch.tensor([[0], [1]], dtype=torch.int64)))
        self.assertEqual(x.amin(dim=1, keepdim=True), torch.tensor([[-1], [3]], dtype=dtype))
        self.assertEqual(x.argmin(dim=1, keepdim=True), torch.tensor([[0], [1]], dtype=torch.int64))

        # test that non-contiguous tensors work
        self.assertEqual(x[:, :2].min().item(), -1)
        self.assertEqual(x[:, :2].amin().item(), -1)
        self.assertEqual(x[:, :2].argmin().item(), 0)

        x = torch.tensor(example, device=device, dtype=dtype)

        self.assertEqual(x.max().item(), 6)
        self.assertEqual(x.amax().item(), 6)
        self.assertEqual(x.argmax().item(), 5)

        self.assertEqual(x.max(0), (torch.tensor([5, 3, 6], dtype=dtype),
                                    torch.tensor([1, 1, 1], dtype=torch.int64)))
        self.assertEqual(x.amax(0), torch.tensor([5, 3, 6], dtype=dtype))
        self.assertEqual(x.argmax(dim=0), torch.tensor([1, 1, 1], dtype=torch.int64))

        self.assertEqual(x.max(dim=0, keepdim=True), (torch.tensor([[5, 3, 6]], dtype=dtype),
                                                      torch.tensor([[1, 1, 1]], dtype=torch.int64)))
        self.assertEqual(x.amax(dim=0, keepdim=True), torch.tensor([[5, 3, 6]], dtype=dtype))
        self.assertEqual(x.argmax(dim=0, keepdim=True), torch.tensor([[1, 1, 1]], dtype=torch.int64))

        self.assertEqual(x.max(1), (torch.tensor([2, 6], dtype=dtype),
                                    torch.tensor([1, 2], dtype=torch.int64)))
        self.assertEqual(x.amax(1), torch.tensor([2, 6], dtype=dtype))
        self.assertEqual(x.argmax(dim=1), torch.tensor([1, 2], dtype=torch.int64))

        self.assertEqual(x.max(1, keepdim=True), (torch.tensor([[2], [6]], dtype=dtype),
                                                  torch.tensor([[1], [2]], dtype=torch.int64)))
        self.assertEqual(x.amax(1, keepdim=True), torch.tensor([[2], [6]], dtype=dtype))
        self.assertEqual(x.argmax(dim=1, keepdim=True), torch.tensor([[1], [2]], dtype=torch.int64))

        # test that non-contiguous tensors work
        self.assertEqual(x[:, :2].max().item(), 5)
        self.assertEqual(x[:, :2].amax().item(), 5)
        self.assertEqual(x[:, :2].argmax().item(), 2)

    @dtypes(*integral_types_and(torch.bool))
    @skipIfMPS
    def test_nanmean_integral_types(self, device, dtype):
        # [FUNCTION DOC] Checks that nanmean on integral or bool tensors of many shapes raises
        # RuntimeError saying the input must have floating point or complex dtype. Skipped on MPS.

        # List of tensor shapes to test
        shapes = [
            (),
            (0,),
            (1,),
            (3, 4, 5),
            (2, 0, 3),
            (10, 10, 10),
            (2, 3, 0, 4),
            (100,),
            (1, 1, 1),
            (5, 5, 5, 5, 5),
        ]

        for shape in shapes:
            # Tensor of the specified shape and dtype
            t = make_tensor(shape, dtype=dtype, device=device)
            # Attempt to call torch.nanmean and expect a RuntimeError
            with self.assertRaisesRegex(
                RuntimeError,
                r"nanmean\(\): expected input to have floating point or complex dtype but got \w+"
            ):
                torch.nanmean(t)

    @skipIfMPS
    @precisionOverride({torch.float16: 1e-2, torch.bfloat16: 1e-2})
    @dtypes(*set(all_types_and(torch.half, torch.bfloat16)) - {torch.uint8})
    @parametrize("fn_name", [
        "mean", "median", "nanmedian", "mode", "norm", "prod",
        "std", "sum", "var", "max", "min", "amax", "amin"])
    def test_dim_reduction_fns(self, device, dtype, fn_name):
        # [FUNCTION DOC] Parametrized over a reduction name (mean, median, nanmedian, mode,
        # norm, prod, std, sum, var, max, min, amax, amin). For each, checks dim reduction
        # semantics: result rank drops by one without keepdim and is kept with keepdim and
        # equals the unsqueezed result; a one-element 1-D input gives a scalar; a singleton
        # dim is reduced correctly; and writing into an out argument (a column slice of a
        # larger tensor, with a values/indices pair for median, nanmedian, mode, max and
        # min) matches the plain result. Skipped on MPS; uint8 excluded.
        def normfn_attr(t, dim, keepdim=False, out=None):
            # [FUNCTION DOC] HELPER: Wrapper that calls torch.norm with the 2-norm on the
            # given dim, keepdim and out so norm can be called like the other reductions.
            attr = torch.norm
            return attr(t, 2, dim, keepdim, out=out)

        fn_attr = getattr(torch, fn_name) if fn_name != "norm" else normfn_attr

        def fn(x, dim, keepdim=False, out=None):
            # [FUNCTION DOC] HELPER: Call the selected reduction with dim, keepdim and
            # out; if it returns a tuple (values, indices) return only the first element.
            ans = fn_attr(x, dim, keepdim=keepdim, out=out)
            return ans if not isinstance(ans, tuple) else ans[0]

        def fn_tuple(x, dim, keepdim=False, out=None):
            # [FUNCTION DOC] HELPER: Call the selected reduction with dim, keepdim
            # and out and return the raw result (possibly a values/indices tuple).
            return fn_attr(x, dim, keepdim=keepdim, out=out)

        def test_multidim(x, dim):
            # [FUNCTION DOC] HELPER: Assert that reducing x over dim with
            # keepdim equals the unsqueezed plain reduction, and that the
            # ranks are one less (no keepdim) and unchanged (keepdim).
            self.assertEqual(fn(x, dim).unsqueeze(dim), fn(x, dim, keepdim=True))
            self.assertEqual(x.ndimension() - 1, fn(x, dim).ndimension())
            self.assertEqual(x.ndimension(), fn(x, dim, keepdim=True).ndimension())

        # general case
        x = torch.randn(3, 4, 5, device=device)
        dim = random.randint(0, 2)
        test_multidim(x, dim)

        # check 1-d behavior
        x = torch.randn(1, device=device)
        dim = 0
        self.assertEqual(fn(x, dim).shape, ())
        self.assertEqual(fn(x, dim, keepdim=True).shape, (1,))

        # check reducing of a singleton dimension
        dims = [3, 4, 5]
        singleton_dim = random.randint(0, 2)
        dims[singleton_dim] = 1
        x = torch.randn(dims, device=device)
        test_multidim(x, singleton_dim)

        # check reducing with output kwargs
        if fn_name in ['median', 'nanmedian', 'mode', 'max', 'min']:
            y = torch.randn(5, 3, device=device)
            values = torch.randn(5, 3, device=device)
            indices = torch.zeros(5, 3, device=device).long() - 1
            fn_tuple(y, 1, keepdim=False, out=(values[:, 1], indices[:, 1]))
            values_expected, indices_expected = fn_tuple(y, 1, keepdim=False)
            self.assertEqual(values[:, 1], values_expected,
                             msg=lambda msg: f'{msg}\n{fn_name} values with out= kwarg')
            self.assertEqual(indices[:, 1], indices_expected,
                             msg=lambda msg: f'{msg}\n{fn_name} indices with out= kwarg')
            return

        x = torch.randn(5, 3, device=device)
        y = torch.randn(5, 3, device=device)
        fn(y, 1, keepdim=False, out=x[:, 1])
        expected = fn(y, 1, keepdim=False)
        self.assertEqual(x[:, 1], expected, msg=lambda msg: f'{msg}\n{fn_name} with out= kwarg')

    @onlyAccelerator
    @skipIfMPS
    @largeTensorTest('10GB')
    def test_reduction_split(self, device):
        # [FUNCTION DOC] Accelerator-only regression test (issue 37583) for a reduction
        # needing a 32-bit indexing split: sums a 5 by 14400 by 14400 float tensor over dim
        # 0 and compares with the sum of its five slices. Needs about 10GB; skipped on MPS.
        # Test reduction when there is a 32bit-indexing split
        # https://github.com/pytorch/pytorch/issues/37583
        input_ = torch.randn(5, 14400, 14400, device=device)
        result = input_.sum(dim=0)
        expect = input_[0] + input_[1] + input_[2] + input_[3] + input_[4]
        self.assertEqual(result, expect)

    @onlyAccelerator
    @skipIfMPS
    @dtypes(torch.half, torch.float, torch.double, torch.bfloat16)
    def test_reduction_vectorize_along_input_corner(self, device, dtype):
        # [FUNCTION DOC] Accelerator-only: checks vectorised reduction corner cases where the
        # reduced input is misaligned or ends at awkward offsets. Sum and argmax over a 1-D
        # tensor of about 64M elements (and a shifted view of it) with a single one placed
        # near the start or end, then 2-D sum, max/argmax and min/argmin over the last dim of
        # a seven by about one million tensor, verifying values and indices. Skipped on MPS.
        # 1D case: sum
        size = 1024 * 1024 * 64 + 3
        shift = 1
        x = torch.zeros(size, dtype=dtype, device=device)
        y = x[shift:]
        for i in range(100):
            x.zero_()
            x[i] = 1
            self.assertEqual(x.sum(), 1.0)
            if i < shift:
                self.assertEqual(y.sum(), 0.0)
            else:
                self.assertEqual(y.sum(), 1.0)
        for i in range(1, 100):
            x.zero_()
            x[-i] = 1
            self.assertEqual(x.sum(), 1.0)
            self.assertEqual(y.sum(), 1.0)
        # 1D case: argmax
        size = 1024 * 1024 * 64 + 3
        shift = 1
        ysize = size - shift
        x = torch.zeros(size, dtype=dtype, device=device)
        y = x[shift:]
        for i in range(100):
            x.zero_()
            x[i] = 1
            self.assertEqual(x.argmax().item(), i)
            if i >= shift:
                self.assertEqual(y.argmax().item(), i - shift)
        for i in range(1, 100):
            x.zero_()
            x[-i] = 1
            self.assertEqual(x.argmax().item(), size - i)
            self.assertEqual(y.argmax().item(), ysize - i)
        # 2D case: sum
        size = (7, 1024 * 1024 + 3)
        x = torch.zeros(size, dtype=dtype, device=device)
        for i in range(100):
            x.zero_()
            for j in range(7):
                x[j][i] = j
            xs = x.sum(dim=-1)
            for j in range(7):
                self.assertEqual(xs[j].item(), float(j))
        for i in range(100):
            x.zero_()
            for j in range(7):
                x[j][-i] = j
            xs = x.sum(dim=-1)
            for j in range(7):
                self.assertEqual(xs[j].item(), float(j))
        # 2D case: max/argmax
        size = (7, 1024 * 1024 + 3)
        x = torch.zeros(size, dtype=dtype, device=device)
        for i in range(100):
            x.zero_()
            for j in range(7):
                x[j][i] = j + 1
            xs1 = x.argmax(dim=-1)
            xs2 = x.max(dim=-1).indices
            for j in range(7):
                self.assertEqual(xs1[j].item(), i)
                self.assertEqual(xs2[j].item(), i)
        for i in range(1, 100):
            x.zero_()
            for j in range(7):
                x[j][-i] = j + 1
            xs1 = x.argmax(dim=-1)
            xs2 = x.max(dim=-1).indices
            for j in range(7):
                self.assertEqual(xs1[j].item(), size[1] - i)
                self.assertEqual(xs2[j].item(), size[1] - i)
        # 2D case: min/argmin
        size = (7, 1024 * 1024 + 3)
        x = torch.zeros(size, dtype=dtype, device=device)
        for i in range(100):
            x.zero_()
            for j in range(7):
                x[j][i] = -(j + 1)
            xs1 = x.argmin(dim=-1)
            xs2 = x.min(dim=-1).indices
            for j in range(7):
                self.assertEqual(xs1[j].item(), i)
                self.assertEqual(xs2[j].item(), i)
        for i in range(1, 100):
            x.zero_()
            for j in range(7):
                x[j][-i] = -(j + 1)
            xs1 = x.argmin(dim=-1)
            xs2 = x.min(dim=-1).indices
            for j in range(7):
                self.assertEqual(xs1[j].item(), size[1] - i)
                self.assertEqual(xs2[j].item(), size[1] - i)

    @onlyAccelerator
    @skipIfMPS
    # Driver issue of XPU, see https://github.com/intel/torch-xpu-ops/issues/2295
    @dtypes(torch.half, torch.float, torch.double, torch.bfloat16)
    def test_reduction_vectorize_along_output(self, device, dtype):
        # [FUNCTION DOC] Accelerator-only: checks vectorised reduction along the output
        # dimension (argmax and sum over dim 0 of identity-like matrices) for aligned and
        # misaligned storage and for widths that give vector sizes 4, 2 and 1. Skipped on MPS.
        def run_test(input_):
            # [FUNCTION DOC] HELPER: Zero the input matrix, set its diagonal
            # to one, then assert argmax over dim 0 gives the diagonal
            # indices and sum over dim 0 gives ones for each diagonal column.
            M, N = input_.shape
            input_.zero_()
            for i in range(min(M, N)):
                input_[i][i] = 1
            output1 = input_.argmax(dim=0)
            output2 = input_.sum(dim=0)
            for i in range(min(M, N)):
                self.assertEqual(output1[i], i)
                self.assertEqual(output2[i], 1)
        # vec 4
        run_test(torch.zeros(64, 64, dtype=dtype, device=device))
        # vec 2
        run_test(torch.zeros(64 * 64 + 2, dtype=dtype, device=device)[2:].view(64, 64))
        run_test(torch.zeros(64, 62, dtype=dtype, device=device))
        run_test(torch.zeros(64, 2, dtype=dtype, device=device))
        # vec 1
        run_test(torch.zeros(64 * 64 + 1, dtype=dtype, device=device)[1:].view(64, 64))
        run_test(torch.zeros(64, 61, dtype=dtype, device=device))
        run_test(torch.zeros(64, 1, dtype=dtype, device=device))

    @onlyAccelerator
    def test_argminmax_large_axis(self, device):
        # [FUNCTION DOC] Accelerator-only regression test (gh-32863):
        # argmax/max and argmin/min along a 2^31-element int8 tensor
        # return the last index (with the last element set to 1 or -1).
        # Regression test for gh-32863
        x = torch.zeros(2**31, device=device, dtype=torch.int8)
        x[-1] = 1
        self.assertEqual(x.argmax(0), x.shape[0] - 1)
        self.assertEqual(x.max(0).indices, x.shape[0] - 1)
        x[-1] = -1
        self.assertEqual(x.argmin(0), x.shape[0] - 1)
        self.assertEqual(x.min(0).indices, x.shape[0] - 1)

    @onlyAccelerator
    @serialTest()
    @largeMPSBufferTest("5GB")
    @largeTensorTest("5GB")
    @parametrize("fn,expected", [
        subtest((lambda x: x.max(), 1), name="max"),
        subtest((lambda x: x.argmax(), (1 << 32) + 4095), name="argmax"),
        subtest((lambda x: x.min(), 0), name="min"),
        subtest((lambda x: x.argmin(), 0), name="argmin"),
        # sum promotes int8 to int64, materializing a 32GB cast of the input on CUDA
        subtest((lambda x: x.sum(), 1), name="sum", decorators=[largeTensorTest("36GB", device="cuda")]),
        subtest((lambda x: x.any(), True), name="any"),
        subtest((lambda x: x.all(), False), name="all"),
        subtest((lambda x: x.view(-1, 4096).max(0).indices[4095], 1 << 20), name="max_dim"),
    ])
    def test_reductions_above_uint32(self, device, fn, expected):
        # [FUNCTION DOC] Parametrized over a reduction function and expected value: on a tensor
        # of more than 2^32 int8 elements with only the last element set to one, checks the
        # reductions (max, argmax, min, argmin, sum, any, all and a per-column max index). On
        # MPS, which indexes with 32 bits, it expects NotImplementedError about 64-bit indexing
        # instead. Accelerator-only; needs about 5GB (36GB for sum on CUDA); serial test.
        x = torch.zeros((1 << 32) + 4096, device=device, dtype=torch.int8)
        x[-1] = 1
        # MPS reduction kernels index in 32 bits and should raise
        if self.device_type == "mps":
            self.assertRaisesRegex(NotImplementedError, "64-bit indexing", fn, x)
        else:
            self.assertEqual(fn(x), expected)

    def test_argminmax_axis_with_dim_one(self, device):
        # [FUNCTION DOC] Checks (issue 38922) that argmax and argmin along a dimension of size
        # one (dim 0 or -2) of a one by 32768 tensor return zeros, with and without keepdim.
        # See: https://github.com/pytorch/pytorch/issues/38922
        n = 32768
        x = torch.zeros(1, n)
        self.assertEqual(x.argmax(dim=0), torch.zeros(n, dtype=torch.int64))
        self.assertEqual(x.argmin(dim=0), torch.zeros(n, dtype=torch.int64))

        self.assertEqual(x.argmax(dim=-2), torch.zeros(n, dtype=torch.int64))
        self.assertEqual(x.argmin(dim=-2), torch.zeros(n, dtype=torch.int64))

        self.assertEqual(x.argmax(dim=0, keepdim=True), torch.zeros(1, n, dtype=torch.int64))
        self.assertEqual(x.argmin(dim=0, keepdim=True), torch.zeros(1, n, dtype=torch.int64))

        self.assertEqual(x.argmax(dim=-2, keepdim=True), torch.zeros(1, n, dtype=torch.int64))
        self.assertEqual(x.argmin(dim=-2, keepdim=True), torch.zeros(1, n, dtype=torch.int64))

    @dtypes(torch.int, torch.long, torch.float, torch.double)
    @dtypesIfCUDA(torch.int, torch.long, torch.half, torch.float, torch.double)
    @dtypesIfXPU(torch.int, torch.long, torch.half, torch.float, torch.double)
    @skipIfMPS
    def test_median_real_values(self, device, dtype):
        # [FUNCTION DOC] Checks median and nanmedian on random tensors of ranks 0 to 3: the
        # global result equals the sorted lower middle element and nanmedian, and numpy median
        # for odd counts; with dim and keepdim the values equal the sorted middle slice, equal
        # the gather by returned indices, and match numpy for odd sizes. Skipped on MPS.
        # Generate random 0-3D sizes
        sizes = [random.sample(range(1, 32), i) for i in range(4) for _ in range(2)]
        for size in sizes:
            # Create random input tensor
            t = torch.randn(size, device=device).type(dtype)
            t_numpy = t.cpu().numpy()
            res = t.median()
            self.assertEqual(res, t.nanmedian())
            k = int((t.numel() - 1) / 2)
            self.assertEqual(res, t.view(-1).sort()[0][k])
            if t.numel() % 2 == 1:
                # We can only test against numpy for odd reductions because numpy
                # returns the mean of the two medians and torch returns the lower
                self.assertEqual(res.cpu().numpy(), np.median(t_numpy))
            for dim in range(t.ndim):
                res = t.median(dim, True)
                self.assertEqual(res, t.nanmedian(dim, True))
                size = t.size(dim) if t.ndim > 0 else 1
                k = int((size - 1) / 2)
                self.assertEqual(res[0], (t.sort(dim)[0]).select(dim, k).unsqueeze_(dim))
                self.assertEqual(res[0], t.gather(dim, res[1]))
                if size % 2 == 1:
                    # We can only test against numpy for odd reductions because numpy
                    # returns the mean of the two medians and torch returns the lower
                    self.assertEqual(res[0].cpu().numpy(), np.median(t_numpy, dim, keepdims=True), exact_dtype=False)

    @dtypes(torch.float, torch.double)
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypesIfXPU(torch.half, torch.float, torch.double)
    @skipIfMPS
    def test_median_nan_values(self, device, dtype):
        # [FUNCTION DOC] Checks median and nanmedian on random tensors containing nans: median
        # returns nan when any nan is present (the last sorted element), nanmedian returns the
        # lower middle of the non-nan values; verified against sort, gather by indices, and
        # numpy for odd valid counts, globally and per dim with keepdim. Skipped on MPS.
        # Generate random 0-3D sizes
        sizes = [random.sample(range(1, 32), i) for i in range(4) for _ in range(2)]
        for size in sizes:
            # Create random input tensor with nan values
            t = torch.rand(size, device=device, dtype=dtype)
            t.masked_fill_(t < 0.1, float('nan'))
            t_numpy = t.cpu().numpy()
            for op in [torch.median, torch.nanmedian]:
                numpy_op = np.median if op == torch.median else np.nanmedian
                res = op(t)
                num_nan = t.isnan().sum()
                if op == torch.median and num_nan > 0:
                    k = t.numel() - 1
                else:
                    k = int((t.numel() - num_nan - 1) / 2)
                self.assertEqual(res, t.view(-1).sort()[0][k])
                if (t.numel() - num_nan) % 2 == 1:
                    # We can only test against numpy for odd reductions because numpy
                    # returns the mean of the two medians and torch returns the lower
                    self.assertEqual(res.item(), numpy_op(t.cpu().numpy()))
                for dim in range(t.ndim):
                    res = op(t, dim, True)
                    size = t.size(dim) if t.ndim > 0 else 1
                    num_nan = t.isnan().sum(dim, True)
                    if op == torch.median:
                        k = torch.where(num_nan > 0, size - 1, int((size - 1) / 2))
                    else:
                        k = ((size - num_nan - 1) / 2).type(torch.long)
                    self.assertEqual(res[0], (t.sort(dim)[0]).gather(dim, k))
                    self.assertEqual(res[0], t.gather(dim, res[1]))
                    # We can only test against numpy for odd reductions because numpy
                    # returns the mean of the two medians and torch returns the lower
                    mask = (size - num_nan) % 2 == 1
                    res = res[0].masked_select(mask).cpu()
                    ref = numpy_op(t_numpy, dim, keepdims=True)[mask.cpu().numpy()]
                    self.assertEqual(res, torch.from_numpy(ref))

    @skipIfMPS
    def test_median_corner_cases(self, device):
        # [FUNCTION DOC] Checks median and nanmedian corner cases: scalar
        # nan, empty input, one element nan, ragged nan rows with dim,
        # and strided, transposed or reshaped inputs. Skipped on MPS.
        def check(op, a, args, key):
            # [FUNCTION DOC] HELPER: Apply the op to a tensor made from a with the given extra
            # args; if there are no args compare directly with key, if key has one element
            # compare only the values, otherwise compare with a (values, indices) pair.
            t = torch.tensor(a, device=device)
            res = op(t, *args)
            if not args:
                key = torch.tensor(key, device=device)
            else:
                if len(key) == 1:
                    key = torch.tensor(key[0], device=device)
                    res = res[0]
                else:
                    key = (torch.tensor(key[0], device=device), torch.tensor(key[1], device=device))
            self.assertEqual(res, key)

        nan = float('nan')
        check(torch.median, nan, [], nan)
        check(torch.median, [], [], nan)
        check(torch.nanmedian, nan, [], nan)
        check(torch.median, nan, [0], [nan, 0])
        check(torch.nanmedian, nan, [0], [nan, 0])
        check(torch.median, [nan], [0, True], [[nan], [0]])
        check(torch.nanmedian, [nan], [0, True], [[nan], [0]])
        check(torch.median, [nan], [0, True], [[nan], [0]])
        check(torch.nanmedian, [nan], [0, True], [[nan], [0]])

        # Indices are not deterministic here so can only check values
        check(torch.median, [[nan, nan], [1, 2]], [0], [[nan, nan]])
        check(torch.nanmedian, [[nan, nan], [1, 2]], [0], [[1, 2.]])
        check(torch.median, [[nan, nan], [1, 2]], [1], [[nan, 1]])
        check(torch.nanmedian, [[nan, nan], [1, 2]], [1], [[nan, 1.]])

        # Discontiguous and strided tensors
        a = torch.arange(12, device=device)
        self.assertEqual(a[::2].median(), torch.tensor(4, device=device))
        self.assertEqual(a[::2].nanmedian(), torch.tensor(4, device=device))

        a.resize_(3, 4)
        self.assertEqual(a.T.median(), torch.tensor(5, device=device))
        self.assertEqual(a.T.nanmedian(), torch.tensor(5, device=device))
        self.assertEqual(a[::2, ::2].median(-1)[0], torch.tensor([0, 8], device=device))
        self.assertEqual(a[::2, ::2].nanmedian(-1)[0], torch.tensor([0, 8], device=device))

        a.resize_(2, 3, 2)
        self.assertEqual(a.T.median(), torch.tensor(5, device=device))
        self.assertEqual(a.T.nanmedian(), torch.tensor(5, device=device))
        self.assertEqual(a[:, ::2, :].median(-1)[0], torch.tensor([[0, 4], [6, 10]], device=device))
        self.assertEqual(a[:, ::2, :].nanmedian(-1)[0], torch.tensor([[0, 4], [6, 10]], device=device))

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/pull/138657 discovers a latent bug")
    @onlyNativeDeviceTypes
    @dtypes(torch.float, torch.double)
    @skipIfMPS
    def test_quantile(self, device, dtype):
        # [FUNCTION DOC] Compares quantile and nanquantile (all five interpolation modes,
        # every dim and the flattened tensor, keepdim on and off, scalar and vector q) with
        # numpy on random tensors and corner cases including nan; also checks the out variant
        # equals the plain result. Skipped under dynamo and on MPS; native device types only.
        # Generate some random test cases
        ops = ['quantile', 'nanquantile']
        inputs = [tuple(np.random.randint(2, 10, size=i)) for i in range(1, 4)]
        quantiles = [tuple(np.random.rand(i)) for i in range(5)]
        keepdims = [True, False]

        # Add corner cases
        inputs.extend([0.75, (1,), (1, 1), (1, 2, 1)])
        inputs.extend([[float('nan')], [[float('nan'), float('nan')], [1, 2]]])
        inputs.extend([[[float('nan'), float('nan')], [float('nan'), 2]]])
        quantiles.extend([0.5, [0., 1.], np.random.rand(10)])

        # Enumerate all input combinations
        for op, x, q, keepdim in product(ops, inputs, quantiles, keepdims):
            if type(x) is tuple:
                a = torch.randn(x, dtype=dtype, device=device)
                # Make some random elements NaN
                a.masked_fill_(torch.randint_like(a, 20) == 0, float('nan'))
            else:
                a = torch.tensor(x, dtype=dtype, device=device)

            q = torch.tensor(q, dtype=dtype, device=device)

            torch_op = getattr(torch, op)
            numpy_op = getattr(np, op)

            # Compute quantile along every dimension and flattened tensor
            interpolations = ('linear', 'lower', 'higher', 'midpoint', 'nearest')
            for interpolation, dim in product(interpolations,
                                              [None] + list(range(a.ndim))):
                result = torch_op(a, q, dim=dim, keepdim=keepdim, interpolation=interpolation)
                expected = numpy_op(a.cpu().numpy(), q.cpu().numpy(), dim,
                                    method=interpolation, keepdims=keepdim)
                self.assertEqual(result.cpu(), torch.from_numpy(np.array(expected)).type(result.type()))

                # Test out variation
                out = torch.empty_like(result)
                torch_op(a, q, dim=dim, keepdim=keepdim, interpolation=interpolation, out=out)
                self.assertEqual(out.cpu(), result.cpu())

    @skipIfMPS  # Fails only on macos-m1-stable
    def test_quantile_backward(self, device):
        # [FUNCTION DOC] Checks the gradient of quantile and nanquantile for small hand-
        # written inputs (odd and even lengths, vector q, edge q of zero or one, and nan
        # handling where quantile gives all gradient to the nan position). Skipped on MPS.
        def check(a, q, dim, expected_grad, ops=(torch.quantile, torch.nanquantile)):
            # [FUNCTION DOC] HELPER: For the given ops (quantile and nanquantile by
            # default) compute the quantile of a tensor made from a along dim, sum
            # and backpropagate, and compare the gradient with the expected list.
            for op in ops:
                t = torch.tensor(a, device=device, requires_grad=True)
                op(t, torch.tensor(q, device=device), dim).sum().backward()
                self.assertEqual(t.grad, expected_grad)

        check([1., 2, 3], 0.5, 0, [0, 1, 0])
        check([1., 2, 3, 4], 0.5, 0, [0, 0.5, 0.5, 0])
        check([3., 1, 4, 2], 0.5, 0, [0.5, 0, 0, 0.5])
        check([1., 2, 3, 4], [0.25, 0.5, 0.75], 0, [0.25, 1.25, 1.25, 0.25])
        check([[1., 2], [2, 1]], 0., 0, [[1, 0], [0, 1]])
        check([[1., 2], [4, 3]], 1., 1, [[0, 1], [1, 0]])
        check([1, float('nan'), 2], 0.5, 0, [0, 1, 0], [torch.quantile])
        check([1, float('nan'), 2], 0.5, 0, [0.5, 0, 0.5], [torch.nanquantile])

    def test_quantile_error(self, device):
        # [FUNCTION DOC] Checks the error messages of quantile: empty input, bad q shape, integer
        # input, q dtype mismatch, q outside zero to one, bad out dtype, bad interpolation name,
        # and on non-CPU devices q or out on a different device than the input.
        def check(a, q, args, kwargs, message):
            # [FUNCTION DOC] HELPER: Assert quantile raises RuntimeError whose message
            # matches the given text; q may be a list (converted to a tensor) or a scalar.
            with self.assertRaisesRegex(RuntimeError, r'quantile\(\) ' + message):
                at = torch.tensor(a, device=device)
                qt = torch.tensor(q, device=device) if isinstance(q, list) else q
                torch.quantile(at, qt, *args, **kwargs)

        check([], 0.5, [], {}, r'input tensor must be non-empty')
        check([1.], [[1.]], [], {}, r'q must be a scalar or 1D tensor')
        check([1], 0.5, [], {}, r'input tensor must be either float or double dtype')
        check([1.], [1], [], {}, r'q tensor must be same dtype as the input tensor')
        check([1.], -1., [], {}, r'q must be in the range \[0, 1\] but got -1')
        check([1.], 1.1, [], {}, r'q must be in the range \[0, 1\] but got 1.1')
        check([1.], 0.5, [], {'out': torch.empty([], dtype=torch.int32, device=device)},
              r'out tensor must be same dtype as the input tensor')
        check([1.], [1.], [None, False], {'interpolation': 'random_mode'},
              r"interpolation must be one of linear, lower, higher, midpoint or nearest, but got random_mode")

        if self.device_type == "cpu":
            check([1.], [0.5, 1.1, -1], [], {}, r'q values must be in the range \[0, 1\]')
        else:
            with self.assertRaisesRegex(
                    RuntimeError, r'quantile\(\) q tensor must be on the same device as the input tensor'):
                torch.randn(1, device=device).quantile(torch.tensor(0.5))
            with self.assertRaisesRegex(
                    RuntimeError, r'quantile\(\) out tensor must be on the same device as the input tensor'):
                torch.quantile(torch.randn(1, device=device), 0.5, out=torch.scalar_tensor(1))

    @skipIfMPS  # MPS caps float32 at 2^24 and has no float64, so >2^24 can't run there
    @onlyNativeDeviceTypes
    @dtypes(torch.float32, torch.float64)
    def test_quantile_large_input(self, device, dtype):
        # [FUNCTION DOC] Regression for gh-64947: quantile and nanquantile of 17 million element
        # float32/float64 tensors (with some nans for the nan variant) match numpy computed in
        # float64 for all five interpolation modes. Native device types; skipped on MPS.
        # gh-64947: quantile must stay correct past the old 2^24 cap. float32
        # works because ranks are computed in float64; float64 reaches 2^53.
        # Oracle is numpy in float64 (the true value) for both, all modes + nan.
        n = 17_000_000
        torch.manual_seed(0)
        a = torch.randn(n, dtype=dtype, device=device)
        a_nan = a.clone()
        a_nan[torch.randint(n, (32,), device=device)] = float('nan')
        q = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=dtype, device=device)
        a_np = a.double().cpu().numpy()
        a_nan_np = a_nan.double().cpu().numpy()
        q_np = q.double().cpu().numpy()
        for interpolation in ('linear', 'lower', 'higher', 'midpoint', 'nearest'):
            res = torch.quantile(a, q, interpolation=interpolation)
            expected = np.quantile(a_np, q_np, method=interpolation)
            self.assertEqual(res.cpu(), torch.from_numpy(np.asarray(expected)).to(res))

            res_nan = torch.nanquantile(a_nan, q, interpolation=interpolation)
            expected_nan = np.nanquantile(a_nan_np, q_np, method=interpolation)
            self.assertEqual(res_nan.cpu(), torch.from_numpy(np.asarray(expected_nan)).to(res_nan))

    @onlyNativeDeviceTypes
    def test_quantile_size_limit(self, device):
        # [FUNCTION DOC] Checks the quantile input size limit: on MPS an input of 2^24
        # elements works and one more element raises RuntimeError saying it is too
        # large; on other devices the larger input works. Native device types only.
        # float32 ranks are exact only to 2^24; computing them in float64 lifts
        # the limit to 2^53. MPS has no float64, so it keeps the 2^24 cap and
        # raises past it, while CPU/CUDA support larger float32 inputs.
        over_cap = (1 << 24) + 1
        if self.device_type == "mps":
            torch.quantile(torch.empty(1 << 24, dtype=torch.float32, device=device), 0.5)
            with self.assertRaisesRegex(RuntimeError, r'quantile\(\) input tensor is too large'):
                torch.quantile(torch.empty(over_cap, dtype=torch.float32, device=device), 0.5)
        else:
            torch.quantile(torch.empty(over_cap, dtype=torch.float32, device=device), 0.5)

    @dtypes(torch.float, torch.double)
    @dtypesIfMPS(torch.float)  # no float64 on MPS; exercises the sort fallback
    def test_quantile_partial_selection(self, device, dtype):
        # [FUNCTION DOC] Stress test of the quantile selection fast path (and the sort fallback):
        # many quantiles, short rows, adversarial value layouts (few distinct values, sorted
        # ascending, descending, all zeros), batched rows with ragged nan counts and a single
        # valid element per row, all compared with numpy. Seeds torch and numpy random generators.
        # Stresses the CPU shared-pass selection (and the sort fallback on CUDA):
        # many quantiles (deep recursion), adversarial value layouts, and batched
        # ragged nan counts. Oracle is numpy.
        torch.manual_seed(0)
        np.random.seed(0)
        all_interps = ('linear', 'lower', 'higher', 'midpoint', 'nearest')

        def check(a, q, dim, keepdim, op='quantile', interps=all_interps):
            # [FUNCTION DOC] HELPER: For each interpolation mode compute
            # torch quantile or nanquantile (as chosen by op) of a at q
            # along dim and compare with the numpy equivalent.
            tq, nq = getattr(torch, op), getattr(np, op)
            qt = torch.tensor(q, dtype=dtype, device=device)
            for interp in interps:
                r = tq(a, qt, dim=dim, keepdim=keepdim, interpolation=interp)
                e = nq(a.cpu().numpy(), np.asarray(q), dim, method=interp, keepdims=keepdim)
                self.assertEqual(r.cpu(), torch.from_numpy(np.asarray(e)).to(r.dtype))

        # Many quantiles exercise the recursion depth (K under the rank cap on long
        # rows so the selection runs). The non-interpolating modes round q * (L - 1) in
        # the input dtype, which can straddle an integer differently from numpy's
        # float64 for arbitrary q, so the recursion is checked with 'linear' (the value
        # is continuous in the rank there); the rounding modes are covered below on
        # integer ranks.
        qmany = sorted(np.random.rand(50).tolist())
        for shape, dim in [((16384,), None), ((16384,), 0), ((8, 16384), 1), ((4, 4, 16384), 2)]:
            a = torch.randn(*shape, dtype=dtype, device=device)
            check(a, qmany, dim, False, interps=('linear',))
            check(a, [0.0, 1.0], dim, True)
        # Past the cap (and on short rows) the sort fallback must still match numpy.
        check(torch.randn(500, dtype=dtype, device=device), qmany + qmany, 0, False, interps=('linear',))
        check(torch.randn(40, 30, dtype=dtype, device=device), qmany, 1, False, interps=('linear',))

        # Adversarial value layouts (n - 1 divisible by 4 keeps the ranks integral).
        base = torch.randn(257, dtype=dtype, device=device)
        for a in (torch.randint(0, 5, (257,), device=device).to(dtype),
                  base.sort().values, base.sort(descending=True).values,
                  torch.zeros(257, dtype=dtype, device=device)):
            check(a, [0.0, 0.25, 0.5, 0.75, 1.0], 0, False)

        # Batched ragged nan: every row a different valid-count, plus an all-nan row.
        a = torch.randn(12, 401, dtype=dtype, device=device)
        for i in range(12):
            a[i, :i * 33] = float('nan')
        a[0] = float('nan')
        check(a, [0.0, 0.25, 0.5, 0.75, 1.0], 1, False, op='nanquantile')
        check(a, [0.0, 0.25, 0.5, 0.75, 1.0], 1, True, op='nanquantile')

        # A single valid element per row.
        a = torch.full((4, 5), float('nan'), dtype=dtype, device=device)
        a[:, 2] = torch.randn(4, dtype=dtype, device=device)
        check(a, [0.0, 0.5, 1.0], 1, False, op='nanquantile')

    # gradcheck's undefined-grad probe feeds None grads that compiled autograd
    # cannot add; this is an eager-only autograd correctness check.
    @skipIfTorchDynamo("gradcheck undefined-grad is unsupported under compiled autograd")
    @skipIfMPS  # MPS has no float64; gradcheck requires double
    @dtypes(torch.double)
    def test_quantile_partial_selection_autograd(self, device, dtype):
        # [FUNCTION DOC] Checks autograd of the quantile fast path: for 1-D inputs forward values
        # and gradients must exactly equal a full-sort reference for every interpolation mode (with
        # and without nans), and for batched shapes runs gradcheck plus a gradgradcheck. Inputs are
        # made well separated to avoid ties. Double dtype only; skipped under dynamo and on MPS.
        # The CPU fast path computes order-statistic indices and gathers them, so the
        # gradient flows through the same gather as the sort path. Inputs are sized to
        # trip the fast path (num_ranks^2 <= reduction_len, num_ranks <= 100, where
        # num_ranks = (2 if interpolating else 1) * num_q); the same cases exercise the
        # sort fallback on CUDA. Values are distinct and >=~0.75 apart so the selected
        # rank is stable under gradcheck's perturbation (the quantile gradient is
        # ill-defined at ties regardless of which path runs).
        gen = torch.Generator(device=device).manual_seed(0)

        def separated(*shape):
            # [FUNCTION DOC] HELPER: Return a tensor of the given shape holding a random
            # permutation of integers plus small random noise (from the shared seeded
            # generator) so all values are distinct and at least about 0.75 apart.
            n = int(np.prod(shape))
            perm = torch.randperm(n, generator=gen, device=device).to(dtype)
            noise = 0.25 * torch.rand(n, generator=gen, device=device, dtype=dtype)
            return (perm + noise).reshape(shape)

        def sort_ref(a, q, interpolation, ignore_nan):
            # [FUNCTION DOC] HELPER: Reference quantile for 1-D input: sort (nan last), compute
            # ranks from q and the valid count (last valid index when ignoring nan, else last
            # index), gather lower, higher or nearest elements or linearly interpolate with a
            # weight of one half for midpoint or the fractional rank otherwise.
            # The pre-change algorithm for 1D input: full sort (NaN last), gather at the
            # integer ranks, lerp. Same ranks and weights as quantile_compute, so for
            # distinct values it must match the fast path bit for bit.
            s = a.sort().values
            last = (~a.isnan()).sum() - 1 if ignore_nan else a.numel() - 1
            ranks = q * last
            if interpolation == 'lower':
                return s.gather(0, ranks.floor().long())
            if interpolation == 'higher':
                return s.gather(0, ranks.ceil().long())
            if interpolation == 'nearest':
                return s.gather(0, ranks.round().long())
            below = ranks.floor()
            vb = s.gather(0, below.long())
            va = s.gather(0, ranks.ceil().long())
            w = torch.full_like(ranks, 0.5) if interpolation == 'midpoint' else ranks - below
            return torch.lerp(vb, va, w)

        interps = ('linear', 'lower', 'higher', 'midpoint', 'nearest')

        # 1D: forward and gradient must match the sort reference exactly (distinct
        # values mean the fast path and the sort select the same elements).
        for L, qv in [(64, [0.2, 0.5, 0.8]), (100, [0.1, 0.3, 0.5, 0.7, 0.9])]:
            q = torch.tensor(qv, device=device, dtype=dtype)
            for op, ignore_nan in (('quantile', False), ('nanquantile', True)):
                base = separated(L)
                if ignore_nan:
                    base[:7] = float('nan')  # some (not all) NaN: ragged valid count
                for interp in interps:
                    x1 = base.clone().requires_grad_(True)
                    x2 = base.clone().requires_grad_(True)
                    out = getattr(torch, op)(x1, q, interpolation=interp)
                    ref = sort_ref(x2, q, interp, ignore_nan)
                    self.assertEqual(out, ref, atol=0, rtol=0)
                    go = torch.randn(out.shape, generator=gen, device=device, dtype=dtype)
                    out.backward(go)
                    ref.backward(go)
                    self.assertEqual(x1.grad, x2.grad, atol=0, rtol=0)

        # Batched / multi-dim: numerical gradcheck, plus one double-backward check.
        for shape, dim, qv, interp in [
            ((4, 96), 1, [0.25, 0.75], 'linear'),
            ((3, 64), 1, [0.2, 0.5, 0.8], 'midpoint'),
            ((2, 3, 100), 2, [0.1, 0.9], 'lower'),
        ]:
            q = torch.tensor(qv, device=device, dtype=dtype)
            x = separated(*shape).requires_grad_(True)
            self.assertTrue(gradcheck(lambda t: torch.quantile(t, q, dim=dim, interpolation=interp), (x,)))
        x = separated(4, 96).requires_grad_(True)
        q = torch.tensor([0.25, 0.75], device=device, dtype=dtype)
        self.assertTrue(gradgradcheck(lambda t: torch.quantile(t, q, dim=1, interpolation='linear'), (x,)))

    def test_std_mean(self, device):
        # [FUNCTION DOC] Checks std_mean over each single dim with
        # biased/unbiased and keepdim settings equals separate std and mean.
        x = torch.rand(100, 50, 20, device=device)
        for dim in range(x.dim()):
            for unbiased in [False, True]:
                for keepdim in [False, True]:
                    std1, mean1 = torch.std_mean(x, dim=dim, unbiased=unbiased, keepdim=keepdim)
                    std2 = x.std(dim=dim, unbiased=unbiased, keepdim=keepdim)
                    mean2 = x.mean(dim=dim, keepdim=keepdim)
                    self.assertEqual(std1, std2)
                    self.assertEqual(mean1, mean2)

    def test_std_mean_all_dims(self, device):
        # [FUNCTION DOC] Checks std_mean over all elements,
        # biased and unbiased, equals separate std and mean.
        x = torch.rand(100, 50, 20, device=device)
        for unbiased in [False, True]:
            std1, mean1 = torch.std_mean(x, unbiased=unbiased)
            std2 = x.std(unbiased=unbiased)
            mean2 = x.mean()
            self.assertEqual(std1, std2)
            self.assertEqual(mean1, mean2)

    def test_var_mean(self, device):
        # [FUNCTION DOC] Checks var_mean over each single dim with
        # biased/unbiased and keepdim settings equals separate var and mean.
        x = torch.rand(100, 300, 50, device=device)
        for dim in range(x.dim()):
            for unbiased in [False, True]:
                for keepdim in [False, True]:
                    var1, mean1 = torch.var_mean(x, dim=dim, unbiased=unbiased, keepdim=keepdim)
                    var2 = x.var(dim=dim, unbiased=unbiased, keepdim=keepdim)
                    mean2 = x.mean(dim=dim, keepdim=keepdim)
                    self.assertEqual(var1, var2)
                    self.assertEqual(mean1, mean2)

    def test_var_mean_all_dims(self, device):
        # [FUNCTION DOC] Checks var_mean over all elements,
        # biased and unbiased, equals separate var and mean.
        x = torch.rand(100, 50, 20, device=device)
        for unbiased in [False, True]:
            var1, mean1 = torch.var_mean(x, unbiased=unbiased)
            var2 = x.var(unbiased=unbiased)
            mean2 = x.mean()
            self.assertEqual(var1, var2)
            self.assertEqual(mean1, mean2)

    def test_std_mean_some_dims(self, device):
        # [FUNCTION DOC] Checks std_mean over every combination of two to four dims of a rank
        # 5 tensor, with biased/unbiased and keepdim settings, equals separate std and mean.
        sizes = (4, 6, 7, 5, 3)
        dims = len(sizes)
        x = torch.rand(sizes, device=device)
        for num_of_dims in range(2, dims):
            dim_list = list(combinations(list(range(dims)), r=num_of_dims))
            for dim in dim_list:
                for unbiased in [False, True]:
                    for keepdim in [False, True]:
                        std1, mean1 = torch.std_mean(x, dim=dim, unbiased=unbiased, keepdim=keepdim)
                        std2 = x.std(dim=dim, unbiased=unbiased, keepdim=keepdim)
                        mean2 = x.mean(dim=dim, keepdim=keepdim)
                        self.assertEqual(std1, std2)
                        self.assertEqual(mean1, mean2)

    def _compare_std_var_with_numpy(self, op, device, dtype, input, dim,
                                    keepdim, unbiased, use_out):
        # [FUNCTION DOC] HELPER: Compare torch var or std (selected by the op name; any
        # other name fails the test) with numpy on the given input. The ddof is 1 when
        # unbiased. With no dim the full reduction is used; otherwise dim and keepdim are
        # passed, optionally through an out tensor. bfloat16 inputs are converted to float32
        # for numpy. Dtype exactness is not required for bfloat16 and complex inputs.
        a = input.cpu().numpy() if input.dtype is not torch.bfloat16 else input.float().cpu().numpy()
        numpy_kwargs = {
            'axis' : dim,
            'keepdims' : keepdim,
            'ddof' : 1 if unbiased else 0,
        }

        if dim is None:
            del numpy_kwargs['axis']
            del numpy_kwargs['keepdims']

        if op == 'var':
            torch_op = torch.var
            numpy_op = np.var
        elif op == 'std':
            torch_op = torch.std
            numpy_op = np.std
        else:
            self.fail("Unknown op!")

        numpy_result = numpy_op(a, **numpy_kwargs)

        if dim is None and use_out is False:
            torch_result = torch_op(input, unbiased)
        elif dim is not None and use_out is False:
            torch_result = torch_op(input, dim, unbiased, keepdim)
        elif dim is not None and use_out is True:
            out = torch.empty(0, device=device, dtype=dtype)
            torch_result = torch_op(input, dim, unbiased, keepdim, out=out)
        else:
            out = torch.empty(0, device=device, dtype=dtype)
            torch_result = torch_op(input, dim, unbiased, keepdim, out=out)

        exact_dtype = input.dtype not in (torch.bfloat16, torch.complex32, torch.complex64, torch.complex128)
        self.assertEqual(torch_result, numpy_result, exact_dtype=exact_dtype)

    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    # Driver issue for float64 on XPU, see https://github.com/intel/torch-xpu-ops/issues/2295
    @dtypesIfXPU(torch.float, torch.cfloat, torch.cdouble)
    @skipIfMPS
    def test_var_vs_numpy(self, device, dtype):
        # [FUNCTION DOC] Compares var with numpy for a random 20 by 20
        # tensor over dim of None, 0 or 1 combined with unbiased, keepdim
        # and out settings (float and complex dtypes). Skipped on MPS.
        _size = (20, 20)

        for test_case in product((torch.randn(_size, device=device, dtype=dtype),),
                                 (None, 0, 1),
                                 (False, True),
                                 (False, True),
                                 (False, True),):
            self._compare_std_var_with_numpy('var', device, dtype, *test_case)

    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    # Driver issue for float64 on XPU, see https://github.com/intel/torch-xpu-ops/issues/2295
    @dtypesIfXPU(torch.float, torch.cfloat, torch.cdouble)
    @skipIfMPS
    def test_std_vs_numpy(self, device, dtype):
        # [FUNCTION DOC] Compares std with numpy for a random 20 by 20
        # tensor over dim of None, 0 or 1 combined with unbiased, keepdim
        # and out settings (float and complex dtypes). Skipped on MPS.
        _size = (20, 20)

        for test_case in product((torch.randn(_size, device=device, dtype=dtype),),
                                 (None, 0, 1),
                                 (False, True),
                                 (False, True),
                                 (False, True),):
            self._compare_std_var_with_numpy('std', device, dtype, *test_case)

    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    # Driver issue for float64 on XPU, see https://github.com/intel/torch-xpu-ops/issues/2295
    @dtypesIfXPU(torch.float, torch.cfloat, torch.cdouble)
    @skipIfMPS
    def test_var_correction_vs_numpy(self, device, dtype):
        # [FUNCTION DOC] Compares var with the correction argument (default,
        # 0, 10, 30 and a negative value) against numpy with the matching
        # ddof for several dims and keepdim settings; infinite results are
        # mapped to nan on both sides before comparing. Skipped on MPS.
        _size = (20, 20)
        test_args = [
            *product(
                # dim
                (None, 0, 1),
                # correction
                (None, 0, 10, 30),
                # keepdim
                (False, True),
            ),
            [None, -100, True],  # Negative correction
        ]

        tensor = make_tensor(_size, device=device, dtype=dtype)
        array = tensor.cpu().numpy()

        for dim, correction, keepdim in test_args:
            numpy_kwargs = dict(axis=dim, ddof=correction, keepdims=keepdim)
            if correction is None:
                # NumPy default is not compatible with torch.std (gh-50010)
                numpy_kwargs['ddof'] = 1

            numpy_res = np.asarray(np.var(array, **numpy_kwargs))
            torch_res = torch.var(tensor, dim=dim, correction=correction, keepdim=keepdim)

            # inf vs. nan results are sensitive to machine precision,
            # just treat them as equivalent
            numpy_res[np.isinf(numpy_res)] = np.nan
            torch_res[torch_res.isinf()] = np.nan

            self.assertEqual(torch_res, numpy_res)

    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    # Driver issue for float64 on XPU, see https://github.com/intel/torch-xpu-ops/issues/2295
    @dtypesIfXPU(torch.float, torch.cfloat, torch.cdouble)
    @skipIfMPS
    def test_std_correction_vs_numpy(self, device, dtype):
        # [FUNCTION DOC] Compares std with the correction argument (default,
        # 0, 10, 30 and a negative value) against numpy with the matching
        # ddof for several dims and keepdim settings; infinite results are
        # mapped to nan on both sides before comparing. Skipped on MPS.
        _size = (20, 20)
        test_args = [
            *product(
                # dim
                (None, 0, 1),
                # correction
                (None, 0, 10, 30),
                # keepdim
                (False, True),
            ),
            [None, -100, True],  # Negative correction
        ]

        tensor = make_tensor(_size, device=device, dtype=dtype)
        array = tensor.cpu().numpy()

        for dim, correction, keepdim in test_args:
            numpy_kwargs = dict(axis=dim, ddof=correction, keepdims=keepdim)
            if correction is None:
                # NumPy default is incompatible with torch.std (gh-50010)
                numpy_kwargs['ddof'] = 1

            numpy_res = np.asarray(np.std(array, **numpy_kwargs))
            torch_res = torch.std(tensor, dim=dim, correction=correction, keepdim=keepdim)

            # inf vs. nan results are sensitive to machine precision,
            # just treat them as equivalent
            numpy_res[np.isinf(numpy_res)] = np.nan
            torch_res[torch_res.isinf()] = np.nan

            self.assertEqual(torch_res, numpy_res)

    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    # Driver issue on XPU, see https://github.com/intel/torch-xpu-ops/issues/2295
    @dtypesIfXPU(torch.float, torch.cfloat)
    @skipIfMPS
    def test_std_mean_correction(self, device, dtype):
        # [FUNCTION DOC] Checks std_mean with correction, dim and keepdim equals separate std
        # and mean (for no dim the mean is reshaped to all ones when keepdim). Skipped on MPS.
        _size = (20, 20)
        test_args = [
            *product(
                # dim
                (None, 0, 1),
                # correction
                (None, 0, 10, 30),
                # keepdim
                (False, True),
            ),
            [None, -100, True],  # Negative correction
        ]

        tensor = make_tensor(_size, device=device, dtype=dtype)

        for dim, correction, keepdim in test_args:
            kwargs = dict(dim=dim, correction=correction, keepdim=keepdim)
            std1 = torch.std(tensor, **kwargs)
            if dim is not None:
                mean1 = torch.mean(tensor, dim=dim, keepdim=keepdim)
            else:
                mean1 = torch.mean(tensor)
                if keepdim:
                    mean1 = mean1.reshape((1,) * tensor.ndim)
            std2, mean2 = torch.std_mean(tensor, **kwargs)

            self.assertEqual(std1, std2)
            self.assertEqual(mean1, mean2)

    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    # Driver issue on XPU, see https://github.com/intel/torch-xpu-ops/issues/2295
    @dtypesIfXPU(torch.float, torch.cfloat)
    @skipIfMPS
    def test_var_mean_correction(self, device, dtype):
        # [FUNCTION DOC] Checks var_mean with correction, dim and keepdim equals separate var
        # and mean (for no dim the mean is reshaped to all ones when keepdim). Skipped on MPS.
        _size = (20, 20)
        test_args = [
            *product(
                # dim
                (None, 0, 1),
                # correction
                (None, 0, 10, 30),
                # keepdim
                (False, True),
            ),
            [None, -100, True],  # Negative correction
        ]

        tensor = make_tensor(_size, device=device, dtype=dtype)

        for dim, correction, keepdim in test_args:
            kwargs = dict(dim=dim, correction=correction, keepdim=keepdim)
            var1 = torch.var(tensor, **kwargs)
            if dim is not None:
                mean1 = torch.mean(tensor, dim=dim, keepdim=keepdim)
            else:
                mean1 = torch.mean(tensor)
                if keepdim:
                    mean1 = mean1.reshape((1,) * tensor.ndim)
            var2, mean2 = torch.var_mean(tensor, **kwargs)

            self.assertEqual(var1, var2)
            self.assertEqual(mean1, mean2)

    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    @skipIfMPS
    def test_warn_invalid_degrees_of_freedom(self, device, dtype):
        # [FUNCTION DOC] Checks that std, var, var_mean and std_mean emit
        # a warning mentioning degrees of freedom being non-positive when
        # the correction equals the reduced size. Skipped on MPS.
        def _assert_warning(_func, _tensor, _correction):
            # [FUNCTION DOC] HELPER: Call the function on the tensor along the last
            # dim with the correction while recording warnings, and assert the
            # first recorded warning message contains the degrees of freedom text.
            with warnings.catch_warnings(record=True) as w:
                _func(_tensor, dim=-1, correction=_correction)
            self.assertIn('degrees of freedom is <= 0', str(w[0].message))

        correction = 20
        size = (10, correction)
        tensor = make_tensor(size, dtype=dtype, device=device)
        for f in [torch.std, torch.var, torch.var_mean, torch.std_mean]:
            _assert_warning(f, tensor, correction)

    def test_amin_amax_some_dims(self, device):
        # [FUNCTION DOC] Checks amin and amax over multiple dims equal repeated
        # single-dim reductions (adjusting the dim index when not keepdim), for
        # all combinations of two to four dims of a rank 5 tensor.
        sizes = (4, 6, 7, 5, 3)
        dims = len(sizes)
        x = torch.rand(sizes, device=device)
        for num_of_dims in range(2, dims):
            dim_list = list(combinations(list(range(dims)), r=num_of_dims))
            for dim in dim_list:
                for keepdim in [False, True]:
                    amin1 = torch.amin(x, dim=dim, keepdim=keepdim)
                    amax1 = torch.amax(x, dim=dim, keepdim=keepdim)
                    amin2 = x
                    amax2 = x
                    for i, d in enumerate(dim):
                        if not keepdim:
                            d -= i
                        amin2 = torch.amin(amin2, dim=d, keepdim=keepdim)
                        amax2 = torch.amax(amax2, dim=d, keepdim=keepdim)
                    self.assertEqual(amin1, amin2)
                    self.assertEqual(amax1, amax2)

    @skipIfMPS
    def test_histc(self, device):
        # [FUNCTION DOC] Exercises torch.histc: negative bins error, empty input, default
        # bins, constant input, ranges with no elements, integral and non-integral bin sizes,
        # double and mixed dtype, scalar input, inf and nan handling with and without explicit
        # min/max (RuntimeError when the range is not finite), min greater than max error, and
        # comparison with numpy histogram for several shapes and strides. Skipped on MPS.
        # negative nbins throws
        with self.assertRaisesRegex(RuntimeError, 'bins must be > 0'):
            torch.histc(torch.tensor([1], dtype=torch.float, device=device), bins=-1)
        # empty tensor
        actual = torch.histc(torch.tensor([], device=device), min=0, max=3)
        expected = torch.zeros(100, dtype=torch.float, device=device)
        self.assertEqual(expected, actual)

        # without nbins
        actual = torch.histc(
            torch.tensor([2, 5], dtype=torch.float, device=device))
        expected = torch.zeros(100, dtype=torch.float, device=device)
        expected[0] = 1
        expected[99] = 1
        self.assertEqual(expected, actual)
        # tensor with the same element
        actual = torch.histc(torch.ones(5, dtype=torch.float, device=device), bins=5)
        self.assertEqual(
            torch.tensor([0, 0, 5, 0, 0], dtype=torch.float, device=device),
            actual)
        # no element falls between [min, max]
        actual = torch.histc(
            torch.ones(5, dtype=torch.float, device=device), bins=5, min=2, max=3)
        self.assertEqual(
            torch.tensor([0, 0, 0, 0, 0], dtype=torch.float, device=device),
            actual)
        # element falls below min + integral bin size and
        actual = torch.histc(
            torch.tensor([2, 4, 2, 2, 5, 4], dtype=torch.float, device=device),
            bins=5, min=1, max=5)
        self.assertEqual(
            torch.tensor([0, 3, 0, 2, 1], dtype=torch.float, device=device),
            actual)
        # non-integral bin size
        actual = torch.histc(
            torch.tensor([1, 2, 1], dtype=torch.float, device=device),
            bins=4, min=0, max=3)
        self.assertEqual(
            torch.tensor([0, 2, 1, 0], dtype=torch.float, device=device),
            actual)
        # double input
        actual = torch.histc(
            torch.tensor([1, 2, 1], dtype=torch.double, device=device), bins=4, min=0, max=3)
        self.assertEqual(
            torch.tensor([0, 2, 1, 0], dtype=torch.double, device=device),
            actual)
        self.assertEqual(actual.dtype, torch.double)
        # mixed input
        actual = torch.histc(
            torch.tensor([1., 2, 1], dtype=torch.float, device=device),
            bins=4, min=0, max=3)
        self.assertEqual(
            torch.tensor([0, 2, 1, 0], dtype=torch.float, device=device),
            actual)
        self.assertEqual(actual.dtype, torch.float)
        # scalar input and 1 bin -- should return a 1-dimensional tensor, not a scalar.
        actual = torch.histc(
            torch.tensor(0, dtype=torch.float, device=device),
            bins=1, min=0, max=3)
        self.assertEqual(
            torch.tensor([1], dtype=torch.float, device=device),
            actual)
        # tensors with inf; min, max not provided -- should throw a RuntimeError
        with self.assertRaisesRegex(RuntimeError, r'range of \[[\w,+\-\.\ ]+\] is not finite'):
            torch.histc(torch.tensor([float("inf")], dtype=torch.float, device=device))
        with self.assertRaisesRegex(RuntimeError, r'range of \[[\w,+\-\.\ ]+\] is not finite'):
            torch.histc(torch.tensor([float("-inf")], dtype=torch.float, device=device))
        with self.assertRaisesRegex(RuntimeError, r'range of \[[\w,+\-\.\ ]+\] is not finite'):
            torch.histc(torch.tensor([float("-inf"), float("inf")], dtype=torch.float, device=device))
        with self.assertRaisesRegex(RuntimeError, r'range of \[[\w,+\-\.\ ]+\] is not finite'):
            torch.histc(torch.tensor([1., 2., float("inf")], dtype=torch.float, device=device))
        # tensors with inf; min, max provided
        self.assertEqual(
            torch.histc(torch.tensor([float("inf")], dtype=torch.float, device=device),
                        bins=1, min=0, max=3),
            torch.tensor([0], dtype=torch.float, device=device))
        self.assertEqual(
            torch.histc(torch.tensor([1., 2., float("inf")], dtype=torch.float, device=device),
                        bins=4, max=3),
            torch.tensor([0, 1, 1, 0], dtype=torch.float, device=device))
        # tensor with nan; min, max not provided -- should throw a RuntimeError
        with self.assertRaisesRegex(RuntimeError, r'range of \[nan, nan\] is not finite'):
            torch.histc(torch.tensor([float("nan")], dtype=torch.float, device=device))
        # tensor with nan; min, max provided -- nan is ignored
        self.assertEqual(
            torch.histc(torch.tensor([1., 2., float("nan")], dtype=torch.float, device=device),
                        bins=4, max=3),
            torch.tensor([0, 1, 1, 0], dtype=torch.float, device=device))
        # tensors with min > max -- should throw a RuntimeError
        with self.assertRaisesRegex(RuntimeError, "max must be larger than min"):
            torch.histc(torch.tensor([1., 2., 3.], dtype=torch.float, device=device),
                        bins=4, min=5, max=1)

        # test against numpy.histogram()
        def test_against_np(tensor, bins=100, min=0, max=0):
            # [FUNCTION DOC] HELPER: Compute histc with the given bins and
            # range (taking min and max from the tensor when both are zero) and
            # compare with numpy histogram counts converted to the output type.
            if min == 0 and max == 0:
                min = tensor.min().item()
                max = tensor.max().item()
            nparr = tensor.cpu().numpy()
            actual = torch.histc(tensor, bins=bins, min=min, max=max)
            expected = torch.from_numpy(np.histogram(nparr, bins=bins, range=(min, max))[0])
            actual_cpu = actual.cpu()
            # NB: Numpy returns a int64 tensor, like normal people...
            self.assertEqual(actual, expected.to(actual_cpu))

        test_against_np(torch.tensor([1., 2, 1], device=device))
        test_against_np(torch.randn(5000, device=device))

        # Test bins arg
        test_against_np(torch.randn(301, device=device), bins=10)

        # Test truncated range
        test_against_np(torch.randn(201, device=device), min=0.1, max=1)

        noncontig = torch.randn(100, 3, device=device)[:, 2]
        test_against_np(noncontig)

        multidim = torch.randn(3, 5, 7, 2, device=device)
        test_against_np(multidim)

        expanded = torch.randn(1, 5, 1, 2, device=device).expand(3, 5, 7, 2)
        test_against_np(expanded)

        linear = torch.linspace(0, 0.99 - 5.0e-7, 101).to(device)
        test_against_np(linear, bins=20, min=0, max=0.99)

    @dtypes(torch.uint8, torch.int8, torch.int, torch.long, torch.float, torch.double)
    @skipIfMPS
    def test_histc_min_max_errors(self, device, dtype):
        # [FUNCTION DOC] Checks that histc with min greater than max raises
        # RuntimeError for integer and floating dtypes. Skipped on MPS.
        with self.assertRaisesRegex(RuntimeError, "max must be larger than min"):
            torch.histc(torch.tensor([1., 2., 3.], dtype=dtype, device=device), bins=4, min=5, max=1)

    @dtypes(torch.float, torch.double)
    @skipIfMPS
    def test_histc_min_max_corner_cases(self, device, dtype):
        # [FUNCTION DOC] Checks histc for equal min and max (both five) on float input,
        # expecting two counts in the first bin and one in the last. Skipped on MPS.
        actual = torch.histc(
            torch.tensor([1., 2, 1], dtype=dtype, device=device),
            bins=4, min=5, max=5)
        self.assertEqual(
            torch.tensor([2, 0, 0, 1], dtype=dtype, device=device),
            actual)

    @onlyCPU
    @dtypes(torch.float, torch.double)
    def test_histc_value_corner_cases(self, device, dtype):
        # [FUNCTION DOC] CPU only: checks that histc of three copies of the minimum or of
        # the maximum finite value of the dtype, with four bins, counts all three elements.
        min_val = torch.finfo(dtype).min
        actual = torch.histc(
            torch.tensor([min_val, min_val, min_val], dtype=dtype, device=device),
            bins=4)
        self.assertEqual(3.0, actual.sum())
        max_val = torch.finfo(dtype).max
        actual = torch.histc(
            torch.tensor([max_val, max_val, max_val], dtype=dtype, device=device),
            bins=4)
        self.assertEqual(3.0, actual.sum())

    @onlyAccelerator
    @skipIfMPS
    @dtypes(torch.uint8, torch.int8, torch.int, torch.long)
    def test_histc_min_max_corner_cases_device(self, device, dtype):
        # [FUNCTION DOC] Accelerator-only: checks histc with equal min and max (five) on integer
        # dtypes gives the same counts (two, zero, zero, one) as the float case. Skipped on MPS.
        actual = torch.histc(
            torch.tensor([1., 2, 1], dtype=dtype, device=device),
            bins=4, min=5, max=5)
        self.assertEqual(
            torch.tensor([2, 0, 0, 1], dtype=dtype, device=device),
            actual)

    # Tests to ensure that reduction functions employing comparison operators are usable when there
    # exists a zero dimension (i.e. when the tensors are empty) in the tensor. These tests specifically
    # cater to functions where specifying the `dim` parameter is necessary.
    def test_tensor_compare_ops_empty(self, device):
        # [FUNCTION DOC] For amax, amin, max and min (values only) and median on a
        # tensor of shape two by zero by four, checks reduction along the last dim,
        # with negative dim and keepdim gives empty tensors of the right shape (also
        # compared to numpy), and that reducing the zero-sized dim raises IndexError.
        shape = (2, 0, 4)
        master_input = torch.randn(shape, device=device)
        np_input = np.empty(shape)
        test_functions = [
            ('amax', torch.amax, np.amax),
            ('amin', torch.amin, np.amin),
            ('max', lambda *args, **kwargs: torch.max(*args, **kwargs).values, np.max),
            ('min', lambda *args, **kwargs: torch.min(*args, **kwargs).values, np.min),
            ('median', lambda *args, **kwargs: torch.median(*args, **kwargs).values, np.median),
        ]

        for name, fn, np_function in test_functions:
            # Check if reduction happens along the specified dim with and without keepdim. Check with
            # numpy to maintain compatibility with numpy functions.
            error_msg = f"test function: {name}"
            self.assertEqual(torch.empty((2, 0), device=device), fn(master_input, dim=2), msg=error_msg)
            self.assertEqual(np_function(np_input, axis=2),
                             fn(master_input, dim=2).cpu().numpy(), msg=error_msg, exact_dtype=False)

            self.assertEqual(torch.empty((2, 0), device=device), fn(master_input, dim=-1), msg=error_msg)
            self.assertEqual(np_function(np_input, axis=-1),
                             fn(master_input, dim=-1).cpu().numpy(), msg=error_msg, exact_dtype=False)

            self.assertEqual(torch.empty((2, 0, 1), device=device), fn(master_input, dim=2, keepdim=True),
                             msg=error_msg)
            self.assertEqual(np_function(np_input, axis=2, keepdims=True),
                             fn(master_input, dim=2, keepdim=True).cpu().numpy(), msg=error_msg, exact_dtype=False)

            self.assertEqual(torch.empty((2, 0, 1), device=device), fn(master_input, dim=-1, keepdim=True),
                             msg=error_msg)
            self.assertEqual(np_function(np_input, axis=-1, keepdims=True),
                             fn(master_input, dim=-1, keepdim=True).cpu().numpy(), msg=error_msg, exact_dtype=False)

            # Check if function raises error on specified zero'd dimension as reduction dim.
            self.assertRaisesRegex(IndexError, "Expected reduction dim", lambda: fn(master_input, dim=1))

    # Tests to ensure that reduction of zero-dim tensors (i.e. empty tensors) using comparison operators
    # raises an error if no `dim` parameter is specified. This exists separately from tests in
    # test_tensot_compare_ops_empty because not specifying a `dim` parameter in the former tests does
    # not throw errors. Also, checking the return type of argmax requires supplying a different dtype
    # argument than that for the input tensor. There is also variation in numpy testing.
    @skipIfMPS
    def test_tensor_compare_ops_argmax_argmix_kthvalue_dim_empty(self, device):
        # [FUNCTION DOC] For argmax, argmin and kthvalue on a tensor of shape two by zero by four,
        # checks reduction along non-empty dims gives empty tensors of the right shape (int64 for
        # the arg functions, compared to numpy), with keepdim variants, and that reducing the zero-
        # sized dim raises IndexError (also with no dim for argmax and argmin). Skipped on MPS.
        shape = (2, 0, 4)
        master_input = torch.randn(shape, device=device)
        np_input = np.empty(shape)
        test_functions = [
            ('argmax', torch.argmax, {'dtype': torch.int64}, np.argmax),
            ('argmin', torch.argmin, {'dtype': torch.int64}, np.argmin),
            ('kthvalue', lambda *args, k=1, **kwargs: torch.kthvalue(*args, k=1, **kwargs).values,
             {}, lambda *args, k=1, axis=None, **kwargs: np.partition(*args, k, **kwargs).take(k - 1, axis=axis))
        ]

        for name, fn, dtype, np_function in test_functions:
            error_msg = f"test function: {name}"
            self.assertEqual(torch.empty((2, 0), device=device, **dtype), fn(master_input, dim=2), msg=error_msg)
            self.assertEqual(
                np_function(np_input, axis=2), fn(master_input, dim=2).cpu().numpy(), msg=error_msg, exact_dtype=False
            )

            self.assertEqual(torch.empty((2, 0), device=device, **dtype), fn(master_input, dim=-1), msg=error_msg)
            self.assertEqual(
                np_function(np_input, axis=-1), fn(master_input, dim=-1).cpu().numpy(), msg=error_msg, exact_dtype=False
            )

            # keepdim variant does not exist for numpy
            self.assertEqual(torch.empty((2, 0, 1), device=device, **dtype), fn(master_input, dim=2, keepdim=True),
                             msg=error_msg)
            self.assertEqual(torch.empty((2, 0, 1), device=device, **dtype), fn(master_input, dim=-1, keepdim=True),
                             msg=error_msg)

            # Check if function raises error on specified zero'd dimension as reduction dim.
            self.assertRaisesRegex(IndexError, "Expected reduction dim", lambda: fn(master_input, dim=1))
            if name != 'kthvalue':
                self.assertRaisesRegex(IndexError, "Expected reduction dim", lambda: fn(master_input))

    # Tests to ensure that reduction of zero-dim tensors (i.e. empty tensors) using math operators works when a
    # non-zero dim is specified for the reduction and throws an error when the dim specified is 0. Although
    # there is some repetition with test_tensor_compare_ops_optional_dim_empty and test_tensor_compare_ops_empty,
    # these tests are kept separate since tests for math operators also require checking for correctness of the
    # returned data using allclose() or isinf() which does not exists in the former tests.
    @skipIfNoSciPy
    def test_tensor_reduce_ops_empty(self, device):
        # [FUNCTION DOC] For prod, sum, norm, mean, var, std and logsumexp on a tensor of shape
        # two by zero by four, checks reduction along non-empty dims returns empty tensors
        # matching numpy, and reducing the empty dim gives the identity-like value (one, zero,
        # zero, nan, nan, nan, -inf) with correct shapes, also with keepdim; with no dim all
        # except logsumexp (which raises TypeError) give the same scalar. Requires scipy.
        from scipy.special import logsumexp
        shape = (2, 0, 4)
        master_input = torch.randn(shape, device=device)
        np_input = np.empty(shape)
        test_functions = [
            ('prod', torch.prod, 1., np.prod),
            ('sum', torch.sum, 0., np.sum),
            ('norm', torch.norm, 0., np.linalg.norm),
            ('mean', torch.mean, nan, np.mean),
            ('var', torch.var, nan, np.var),
            ('std', torch.std, nan, np.std),
            ('logsumexp', torch.logsumexp, -inf, logsumexp),
        ]

        for name, fn, return_value, np_function in test_functions:
            # Check if reduction happens along the specified dimension.
            error_msg = f"test function: {name}"
            self.assertEqual(torch.empty((2, 0), device=device), fn(master_input, dim=2), msg=error_msg)
            self.assertEqual(np_function(np_input, axis=2), fn(master_input, dim=2).cpu().numpy(), msg=error_msg,
                             exact_dtype=False)

            self.assertEqual(torch.empty((2, 0), device=device), fn(master_input, dim=-1), msg=error_msg)
            self.assertEqual(np_function(np_input, axis=-1), fn(master_input, dim=-1).cpu().numpy(), msg=error_msg,
                             exact_dtype=False)

            self.assertEqual(torch.empty((2, 0, 1), device=device), fn(master_input, dim=2, keepdim=True),
                             msg=error_msg)
            self.assertEqual(np_function(np_input, axis=2, keepdims=True), fn(master_input, dim=2, keepdim=True),
                             msg=error_msg, exact_dtype=False)

            self.assertEqual(torch.empty((2, 0, 1), device=device), fn(master_input, dim=-1, keepdim=True),
                             msg=error_msg)
            self.assertEqual(np_function(np_input, axis=-1, keepdims=True), fn(master_input, dim=-1, keepdim=True),
                             msg=error_msg, exact_dtype=False)

            self.assertEqual(torch.full((2, 4), return_value, device=device), fn(master_input, dim=1), msg=error_msg)
            self.assertEqual(torch.full((2, 4), return_value, device=device), fn(master_input, dim=-2), msg=error_msg)
            self.assertEqual(torch.full((2, 1, 4), return_value, device=device), fn(master_input, dim=1, keepdim=True),
                             msg=error_msg)
            self.assertEqual(torch.full((2, 1, 4), return_value, device=device), fn(master_input, dim=-2, keepdim=True),
                             msg=error_msg)

            if name != 'logsumexp':
                # The scipy function does not work for reduction the zero dimension
                self.assertEqual(np.float32(np_function(np_input, axis=1)), fn(master_input, dim=1).cpu().numpy(),
                                 msg=error_msg)
                self.assertEqual(np.float32(np_function(np_input, axis=-2)), fn(master_input, dim=-2).cpu().numpy(),
                                 msg=error_msg)
                self.assertEqual(np.float32(np_function(np_input, axis=1, keepdims=True)),
                                 fn(master_input, dim=1, keepdim=True).cpu().numpy(),
                                 msg=error_msg)
                self.assertEqual(np.float32(np_function(np_input, axis=-2, keepdims=True)),
                                 fn(master_input, dim=-2, keepdim=True).cpu().numpy(),
                                 msg=error_msg)

                # logsumexp throws a type error when not specifying dim so test separately.
                self.assertEqual(torch.full((), return_value, device=device), fn(master_input), msg=error_msg)
            else:
                self.assertRaises(TypeError, lambda: fn(master_input))

    # Tests to ensure that any() and all() functions work with zero-dim tensors. Kept separate from
    # other tests for checking reduction with zero-dim tensors because these tests have significantly
    # different testing behaviour than that used for the former tests.
    @skipIfMPS
    def test_reduction_empty_any_all(self, device):
        # [FUNCTION DOC] Checks any and all on a tensor of shape two by zero by four
        # converted to many dtypes: reductions over the non-empty dims keep empty shape,
        # over the empty dim any gives false and all gives true (output dtype uint8 for
        # uint8 input, otherwise bool), including the full reduction. Skipped on MPS.
        shape = (2, 0, 4)
        x = torch.randn(shape, device=device)

        for dtype in all_types_and_complex_and(torch.half, torch.bool):
            # Refer: [all, any uint8 compatibility]
            if dtype == torch.uint8:
                out_dtype = torch.uint8
            else:
                out_dtype = torch.bool  # output of all/any is bool irrespective of input dtype

            xb = x.to(dtype)

            # any
            self.assertEqual((2, 0), xb.any(2).shape)
            self.assertEqual((2, 0, 1), xb.any(2, keepdim=True).shape)
            self.assertEqual(torch.zeros((2, 4), device=device, dtype=out_dtype), xb.any(1))
            self.assertEqual(torch.zeros((2, 1, 4), device=device, dtype=out_dtype), xb.any(1, keepdim=True))
            self.assertEqual(torch.zeros((), device=device, dtype=out_dtype), xb.any())

            # all
            self.assertEqual((2, 0), xb.all(2).shape)
            self.assertEqual((2, 0, 1), xb.all(2, keepdim=True).shape)
            self.assertEqual(torch.ones((2, 4), device=device, dtype=out_dtype), xb.all(1))
            self.assertEqual(torch.ones((2, 1, 4), device=device, dtype=out_dtype), xb.all(1, keepdim=True))
            self.assertEqual(torch.ones((), device=device, dtype=out_dtype), xb.all())

    # TODO: can these be merged with their respective OpInfos?
    @skipIfMPS
    def test_reduce_dtype(self, device):
        # [FUNCTION DOC] Checks that for sum, prod, cumsum, cumprod and logcumsumexp,
        # computing in an explicit double dtype (or converting the input to double
        # for logcumsumexp) yields the same gradient as the float computation, with
        # float gradient dtype, with and without dim. Skipped on MPS.
        def test_reduction(op, has_no_dim, takes_dtype=True):
            # [FUNCTION DOC] HELPER: For one op, build a float tensor requiring grad and
            # compare gradients via autograd between the plain op and the double variant
            # (for the no-dim form too when has_no_dim is set); takes_dtype selects
            # between passing a dtype argument and converting the input to double.
            x = torch.randn(3, 3, dtype=torch.float, requires_grad=True, device=device)

            if has_no_dim:
                grad1, = torch.autograd.grad([op(x)], [x])
                grad2, = torch.autograd.grad([op(x, dtype=torch.double)], [x])
                self.assertEqual(grad1, grad2)
                self.assertEqual(grad2.dtype, torch.float)

            gi = torch.randn(op(x, dim=0).shape, dtype=torch.float, device=device)
            grad1, = torch.autograd.grad([op(x, dim=0)], [x], gi)
            if takes_dtype:
                grad2, = torch.autograd.grad([op(x, dim=0, dtype=torch.double)], [x], gi.double())
            else:
                grad2, = torch.autograd.grad([op(x.double(), dim=0)], [x], gi.double())
            self.assertEqual(grad1, grad2)
            self.assertEqual(grad2.dtype, torch.float)

        test_reduction(torch.sum, True)
        test_reduction(torch.prod, True)
        test_reduction(torch.cumsum, False)
        test_reduction(torch.cumprod, False)
        test_reduction(torch.logcumsumexp, False, takes_dtype=False)

    @ops(reference_masked_ops)
    def test_reference_masked(self, device, dtype, op):
        # [FUNCTION DOC] Runs masked reduction ops from reference_masked_ops on strided tensors
        # using each op's sample inputs and compares to the numpy reference, mapping the masked
        # identity to numpy initial value. bfloat16 is converted to float32; dtype exactness is
        # relaxed for bfloat16, int to float promotion and 32/64-bit unsigned numpy results.
        """Test masked reduction operations on strided-only tensors using
        numpy reductions as reference.
        """

        def to_numpy(input):
            # [FUNCTION DOC] HELPER: Convert a tensor to a numpy
            # array on the CPU, going through float32 for bfloat16.
            if input.dtype is torch.bfloat16:
                return input.cpu().to(torch.float32).numpy()
            else:
                return input.cpu().numpy()

        samples = op.sample_inputs_func(op, device, dtype, requires_grad=False)
        for sample_input in samples:
            t = sample_input.input
            actual = op(t, *sample_input.args, **sample_input.kwargs)
            exact_dtype = not (t.dtype is torch.bfloat16
                               or (op.promotes_int_to_float and not torch.is_floating_point(t)))
            expected = op.ref(to_numpy(t), *sample_input.args,
                              **dict(
                                  # `identity` is mapped to numpy reduction `initial` argument
                                  identity=torch.masked._reduction_identity(op.name, t),
                                  **sample_input.kwargs))

            # Workaround https://github.com/pytorch/pytorch/issues/66556
            expected = np.asarray(expected)  # transform numpy scalars to numpy.ndarray instances

            # Numpy differs, producing uint32 on Windows
            if expected.dtype in [np.uint64, np.uint32]:
                exact_dtype = False

            msg = ("Failed to produce expected results! Input tensor was"
                   f" {t}, torch result is {actual}, and reference result is"
                   f" {expected}.") if t.numel() < 10 else None

            self.assertEqual(actual, expected, msg, exact_dtype=exact_dtype)

    @onlyAccelerator
    @serialTest()
    @largeMPSBufferTest(lambda self, device, dtype: 2**31 * dtype.itemsize)
    @largeTensorTest(lambda self, device, dtype: 2**31 * dtype.itemsize)
    @dtypes(torch.half, torch.chalf, torch.bfloat16)
    # skip chalf and half when XPU, see issues https://github.com/intel/torch-xpu-ops/issues/1973
    @dtypesIfXPU(torch.bfloat16)
    def test_reductions_large_half_tensors(self, device, dtype):
        # [FUNCTION DOC] Accelerator-only: sums a half/chalf/bfloat16 tensor of 2^31 elements (half
        # ones, half minus ones) and expects zero; mean is expected the same except that complex
        # half raises RuntimeError about not being implemented. Serial and large-memory test.
        t = torch.ones(2**31, device=device, dtype=dtype)
        t[2**30:] = -1
        expected = torch.tensor(0, device=device, dtype=dtype)
        self.assertEqual(torch.sum(t), expected)

        # mean_cuda is not implemented for ComplexHalf
        err_msg = "not implemented for 'ComplexHalf'"
        ctx = self.assertRaisesRegex(
            RuntimeError, err_msg) if dtype is torch.chalf else contextlib.nullcontext()
        with ctx:
            self.assertEqual(torch.mean(t), expected)

    def test_scalar_tensor_as_dim_argument(self):
        # [FUNCTION DOC] Checks that a scalar integer tensor given as the dim argument of sum
        # works like an int, for several integer dtypes (regression for a PythonArgParser bug).
        """Tests that scalar tensors work correctly as dimension arguments.

        This tests the fix for the PythonArgParser bug where scalar Tensors
        passed to IntList/SymIntList parameters would be incorrectly handled.
        """
        x = torch.ones(1, 2, 3, 4, 5)

        # Scalar tensors should work correctly (same as passing an int)
        result_tensor = x.sum(dim=torch.tensor(3))
        result_int = x.sum(dim=3)
        self.assertEqual(result_tensor.shape, result_int.shape)
        self.assertEqual(result_tensor.shape, torch.Size([1, 2, 3, 5]))

        # Test with different integer dtypes
        for dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
            dim_tensor = torch.tensor(1, dtype=dtype)
            result = x.sum(dim=dim_tensor)
            expected = x.sum(dim=1)
            self.assertEqual(result.shape, expected.shape)

    @skipIfTorchDynamo("Test uses random.randint which creates FakeTensors")
    def test_scalar_tensor_dim_compiled_mode(self):
        # [FUNCTION DOC] Checks that a sum whose dim comes from a python random integer gives the
        # same shape (two by two) in eager mode and under torch.compile. Skipped under dynamo.
        """Tests that scalar FakeTensors from random.randint work correctly in compiled mode."""
        def foo():
            # [FUNCTION DOC] HELPER: Eager function that sums
            # a ones tensor over a random dim (always 0).
            x = torch.ones(2, 2, 2)
            return x.sum(dim=random.randint(0, 0))

        @torch.compile
        def foo_compile():
            # [FUNCTION DOC] HELPER: Compiled version of the
            # same function, used to compare with eager mode.
            x = torch.ones(2, 2, 2)
            return x.sum(dim=random.randint(0, 0))

        result_eager = foo()
        result_compiled = foo_compile()
        self.assertEqual(result_eager.shape, result_compiled.shape)
        self.assertEqual(result_eager.shape, torch.Size([2, 2]))

class TestReductionsOnCPU(TestCase):
    # [CLASS DOC] CPU-only reduction tests (not device-instantiated): elementwise max/min
    # via a map-based reference, integer-upcast behavior of sum/prod/cumsum/cumprod,
    # accumulation scalar types, nansum on complex, low-precision histc, and extensive
    # numpy-based checks of histogram and histogramdd with error-handling cases.
    # TODO: kill map2_ (and similar) uses and update to compare with NumPy
    # only works on CPU since this uses map2_, which is only supported on CPU
    def _testCSelection(self, torchfn, mathfn):
        # [FUNCTION DOC] HELPER: Compare an elementwise binary torch selection
        # function with a python function applied elementwise through map2_
        # (CPU only) on two random 100 by 100 tensors, with zero tolerance.
        # Two tensors
        size = (100, 100)
        a = torch.rand(*size)
        b = torch.rand(*size)
        c = torchfn(a, b)
        expected_c = torch.zeros(*size)
        expected_c.map2_(a, b, lambda _, a, b: mathfn(a, b))
        self.assertEqual(expected_c, c, atol=0, rtol=0)

    def test_max_elementwise(self):
        # [FUNCTION DOC] Checks torch.max of two tensors against python max elementwise.
        self._testCSelection(torch.max, max)

    def test_min_elementwise(self):
        # [FUNCTION DOC] Checks torch.min of two tensors against python min elementwise.
        self._testCSelection(torch.min, min)

    # TODO: update this and tests that use it to handle device properly
    def _test_reduce_integer_upcast(self, fn, has_out=True, test_complex=True):
        # [FUNCTION DOC] HELPER: Check dtype promotion for integer-upcasting reductions.
        # For each CPU math dtype except half, the result dtype of fn is the dtype
        # itself for floating and complex inputs and int64 for integral inputs, an
        # explicit dtype argument is honored (also for mixed float/int/complex), and, if
        # has_out, results written to out tensors keep the out dtype and mismatching
        # dtype with out raises RuntimeError. test_complex is accepted but unused.
        shape = (3, 4, 5)
        reduced_shape = fn(torch.ones(shape)).shape

        def _test_out(dtype, other_dtype):
            # [FUNCTION DOC] HELPER: With an out tensor of the given dtype (taking
            # the input x from the enclosing loop), check fn returns a result of the
            # out dtype with the right values, also with a matching dtype argument,
            # and raises RuntimeError when the dtype argument disagrees with out.
            out = torch.ones(reduced_shape, dtype=dtype)
            result = fn(x, out=out)
            self.assertIs(out.dtype, result.dtype)
            self.assertEqual(fn(x.to(dtype)), result, exact_dtype=False)
            result = fn(x, out=out, dtype=dtype)
            self.assertIs(out.dtype, result.dtype)
            self.assertEqual(fn(x.to(dtype)), result, exact_dtype=False)
            # 'out' is favored over dtype, check error
            self.assertRaises(RuntimeError, lambda: fn(x, out=out, dtype=other_dtype))

        for dtype in [dtype for dtype in get_all_math_dtypes('cpu') if dtype != torch.float16]:
            x = torch.ones(shape, dtype=dtype)
            expected_dtype = dtype if dtype.is_floating_point or dtype.is_complex else torch.int64
            self.assertIs(expected_dtype, fn(x).dtype)
            self.assertEqual(fn(x.to(expected_dtype)), fn(x))

            if dtype.is_floating_point:
                other_dtype = torch.float32 if dtype == torch.float64 else torch.float64
            elif dtype.is_complex:
                other_dtype = torch.complex64 if dtype == torch.complex128 else torch.complex128
            else:
                other_dtype = torch.int32 if dtype != torch.int32 else torch.int16
            self.assertIs(other_dtype, fn(x, dtype=other_dtype).dtype)
            self.assertEqual(fn(x.to(other_dtype)), fn(x, dtype=other_dtype), exact_dtype=False)

            # test mixed int/float/complex
            if dtype.is_floating_point:
                mixed_dtypes = [torch.int32, torch.complex64]
            elif dtype.is_complex:
                mixed_dtypes = [torch.int32, torch.float32]
            else:
                mixed_dtypes = [torch.float32, torch.complex64]

            for mixed_dtype in mixed_dtypes:
                self.assertIs(mixed_dtype, fn(x, dtype=mixed_dtype).dtype)
                self.assertEqual(fn(x.to(mixed_dtype)), fn(x, dtype=mixed_dtype), exact_dtype=False)

                if has_out:
                    _test_out(dtype, other_dtype)
                    _test_out(dtype, mixed_dtype)

    def test_sum_integer_upcast(self):
        # [FUNCTION DOC] Runs the integer-upcast checks for sum
        # with no dim (without out variants) and with dim 0.
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.sum(x, **kwargs), False)
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.sum(x, 0, **kwargs))

    def test_prod_integer_upcast(self):
        # [FUNCTION DOC] Runs the integer-upcast checks for prod
        # with no dim (without out variants) and with dim 0.
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.prod(x, **kwargs), False)
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.prod(x, 0, **kwargs))

    def test_cumsum_integer_upcast(self):
        # [FUNCTION DOC] Runs the integer-upcast checks for cumsum along dim 0.
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.cumsum(x, 0, **kwargs))

    def test_cumprod_integer_upcast(self):
        # [FUNCTION DOC] Runs the integer-upcast checks for cumprod along dim 0.
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.cumprod(x, 0, **kwargs))

    # TODO: make work on CUDA, too
    def test_accreal_type(self) -> None:
        # [FUNCTION DOC] Checks that item of a full sum is a python float for double and
        # float tensors and a python int for long, int, short, char and byte tensors.
        x = torch.ones(2, 3, 4)
        self.assertIsInstance(x.double().sum().item(), float)
        self.assertIsInstance(x.float().sum().item(), float)
        self.assertIsInstance(x.long().sum().item(), int)
        self.assertIsInstance(x.int().sum().item(), int)
        self.assertIsInstance(x.short().sum().item(), int)
        self.assertIsInstance(x.char().sum().item(), int)
        self.assertIsInstance(x.byte().sum().item(), int)

    def test_nansum_complex(self):
        # [FUNCTION DOC] Checks that nansum on CPU for complex tensors
        # raises RuntimeError saying complex inputs are not supported.
        for dtype in complex_types():
            x = torch.randn((3, 3, 3), dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, "nansum on CPU does not support complex inputs"):
                torch.nansum(x)

    # CUDA histc uses AT_DISPATCH_ALL_TYPES (no half/bfloat16), so this
    # low-precision coverage only exists on CPU
    def test_histc_lowp(self):
        # [FUNCTION DOC] Checks histc on CPU for bfloat16 and half inputs gives
        # the expected counts (zero, two, one, zero) with the same output dtype.
        for dtype in (torch.bfloat16, torch.half):
            actual = torch.histc(
                torch.tensor([1, 2, 1], dtype=dtype), bins=4, min=0, max=3)
            self.assertEqual(torch.tensor([0, 2, 1, 0], dtype=dtype), actual)
            self.assertEqual(actual.dtype, dtype)

    """
    Runs torch.histogram and numpy.histogram on the specified input parameters
    and asserts that their output is equal.
    """
    def _test_histogram_numpy(self, t, bins, bin_range, weights, density, eq_func=None):
        # [FUNCTION DOC] HELPER: Run torch.histogram and numpy histogram on the same input,
        # bins (count or tensor), range, weights and density, and compare histograms and bin
        # edges using eq_func (default assertEqual). When bins is not a tensor, torch edges
        # are first checked against numpy within a tolerance and then fed to numpy to avoid
        # linspace discrepancies (issue 58758). Also runs torch.histogram with noncontiguous
        # out tensors and compares them. The range argument is only passed when given.
        def to_np(t):
            # [FUNCTION DOC] HELPER: Return the argument unchanged unless
            # it is a tensor, in which case return its CPU numpy array.
            if not torch.is_tensor(t):
                return t
            return t.cpu().numpy()

        # Wrapper around numpy.histogram performing conversions between torch tensors and numpy arrays.
        def reference_histogram(t, bins, bin_range, weights, density, dtype):
            # [FUNCTION DOC] HELPER: Run numpy histogram after converting tensors to numpy
            # and return the histogram and bin edges as torch tensors of the requested dtype.
            np_t, np_bins, np_weights = map(to_np, [t, bins, weights])
            np_hist, np_bin_edges = np.histogram(
                np_t, np_bins, range=bin_range, weights=np_weights, density=density
            )
            return (
                torch.from_numpy(np_hist).to(dtype),
                torch.from_numpy(np_bin_edges).to(dtype),
            )

        if eq_func is None:
            eq_func = self.assertEqual

        # Doesn't pass a 'range' kwarg unless necessary because the override of
        # histogram with Tensor bins doesn't accept one.
        if bin_range:
            actual_hist, actual_bin_edges = torch.histogram(
                t, bins, range=bin_range, weight=weights, density=density
            )
        else:
            actual_hist, actual_bin_edges = torch.histogram(
                t, bins, weight=weights, density=density
            )

        expected_hist, expected_bin_edges = reference_histogram(
            t, bins, bin_range, weights, density, actual_hist.dtype
        )

        """
        Works around linspace discrepancies by passing torch's constructed bin_edges to numpy.
        When bin edges are not explicitly defined, histogram uses the linspace operator internally
        to construct the sequence of bin edges. In some cases, torch.linspace output differs slightly
        from numpy.linspace output.
        Issue: https://github.com/pytorch/pytorch/issues/58758
        """
        if not torch.is_tensor(bins):
            eq_func(actual_bin_edges, expected_bin_edges, atol=1e-5, rtol=1e-5)
            # Calls numpy.histogram again, passing torch's actual_bin_edges as the bins
            # argument.
            expected_hist, expected_bin_edges = reference_histogram(
                t, actual_bin_edges, bin_range, weights, density, actual_hist.dtype,
            )

        eq_func(actual_hist, expected_hist)
        eq_func(actual_bin_edges, expected_bin_edges)

        # Test passing non-contiguous output tensors
        hist_out = make_tensor(
            expected_hist.shape,
            device=expected_hist.device,
            dtype=expected_hist.dtype,
            noncontiguous=True,
        )
        bin_edges_out = make_tensor(
            expected_bin_edges.shape,
            device=expected_bin_edges.device,
            dtype=expected_bin_edges.dtype,
            noncontiguous=True,
        )

        # Doesn't pass a 'range' kwarg unless necessary because the override of
        # histogram with Tensor bins doesn't accept one.
        if bin_range:
            torch.histogram(
                t,
                bins,
                range=bin_range,
                weight=weights,
                density=density,
                out=(hist_out, bin_edges_out),
            )
        else:
            torch.histogram(
                t, bins, weight=weights, density=density, out=(hist_out, bin_edges_out)
            )

        eq_func(hist_out, expected_hist)
        eq_func(bin_edges_out, expected_bin_edges)

    def test_histogram(self):
        # [FUNCTION DOC] Compares torch.histogram with numpy over many combinations of shapes
        # (scalar to rank 3), contiguity, bin counts, weights and density, with input cases
        # for a bin count, a random range, equal min and max (looser tolerance), explicit bin
        # edges (contiguous or not), constant input, and input equal to the bin edges;
        # finally checks that default arguments equal explicit defaults. CPU, float32.
        shapes = (
            (),
            (0,),
            (1,),
            (1, 5),
            (3, 5),
            (1, 5, 1),
            (2, 3, 5))

        for contig, bins_contig, bin_ct, weighted, density, shape in \
                product([True, False], [True, False], range(1, 10), [True, False], [True, False], shapes):
            values = make_tensor(shape, dtype=torch.float32, device="cpu", low=-9, high=9, noncontiguous=not contig)
            weights = make_tensor(shape, dtype=torch.float32, device="cpu", low=0, high=9, noncontiguous=not contig) if weighted else None

            # Tests passing just the bin_ct
            self._test_histogram_numpy(values, bin_ct, None, weights, density)

            # Tests with caller-specified histogram range
            bin_range = sorted((random.uniform(-9, 9), random.uniform(-9, 9)))
            self._test_histogram_numpy(values, bin_ct, bin_range, weights, density)

            # Tests with range min=max
            bin_range[1] = bin_range[0]
            self._test_histogram_numpy(
                values,
                bin_ct,
                bin_range,
                weights,
                density,
                # TODO: investigate why torch.histogram differs from numpy.histogram
                # so strongly on this particular test.  There seems to be more
                # differences here than the linspace issue, which is itself fairly
                # easily patched around.  Likely, the other tests also differ
                # significantly, but below the default threshold for assertEqual.
                eq_func=partial(self.assertEqual, rtol=3e-5, atol=0.0),
            )

            # Tests with caller-specified bin edges
            bin_edges = make_tensor(bin_ct + 1, dtype=torch.float32, device="cpu", low=-9, high=9).msort()
            if not bins_contig:
                # Necessary because msort always produces contiguous output
                bin_edges_noncontig = make_tensor(bin_ct + 1, dtype=torch.float32, device="cpu", noncontiguous=not bins_contig)
                bin_edges_noncontig.copy_(bin_edges)
                bin_edges = bin_edges_noncontig
            self.assertEqual(bin_edges.is_contiguous(), bins_contig)
            self._test_histogram_numpy(values, bin_edges, None, weights, density)

            # Tests with input tensor in which all elements are equal
            elt = random.uniform(-9, 9)
            values = make_tensor(shape, dtype=torch.float32, device="cpu", low=elt, high=elt, noncontiguous=not contig)
            self._test_histogram_numpy(values, bin_ct, bin_range, weights, density)
            self._test_histogram_numpy(values, bin_edges, None, weights, density)

            # Tests with input equal to bin_edges
            weights = (
                make_tensor(bin_ct + 1, dtype=torch.float32, device="cpu", low=0, high=9, noncontiguous=not contig)
                if weighted
                else None
            )
            self._test_histogram_numpy(bin_edges, bin_edges, None, weights, density)

        # Tests values of default args
        for bin_ct, shape in product(range(1, 10), shapes):
            values = make_tensor(shape, dtype=torch.float32, device="cpu", low=-9, high=9)
            (actual_hist, actual_bin_edges) = torch.histogram(values, bin_ct)
            (expected_hist, expected_bin_edges) = torch.histogram(
                values, bin_ct, range=None, weight=None, density=False)
            self.assertEqual(actual_hist, expected_hist)
            self.assertEqual(actual_bin_edges, expected_bin_edges)


    """
    Runs torch.histogramdd and numpy.histogramdd on the specified input parameters
    and asserts that their output is equal.
    """
    def _test_histogramdd_numpy(self, t, bins, bin_range, weights, density):
        # [FUNCTION DOC] HELPER: Run torch.histogramdd and numpy histogramdd on the same
        # input, bins, range, weights and density and compare histograms and per-dimension bin
        # edges. Handles numpy's requirement for two-dimensional input by reshaping, special-
        # cases a zero-size last dim, and when bins are not tensors compares torch edges with
        # numpy within a tolerance and reruns numpy with torch's edges (issue 58758).
        def to_np(t):
            # [FUNCTION DOC] HELPER: Recursively convert a list of tensors, or
            # a tensor, to numpy arrays; other values pass through unchanged.
            if type(t) is list:
                return list(map(to_np, t))
            if not torch.is_tensor(t):
                return t
            return t.cpu().numpy()

        # Wrapper around numpy.histogram performing conversions between torch tensors and numpy arrays.
        def reference_histogramdd(t, bins, bin_range, weights, density, dtype):
            # [FUNCTION DOC] HELPER: Reshape the input to two dimensions, run numpy
            # histogramdd (with the range converted to per-dimension pairs) and
            # return torch tensors for the histogram and bin edges; for a zero-size
            # last dim it returns nan (density) or zero and no edges instead.
            (np_t, np_bins, np_weights) = map(to_np, [t, bins, weights])

            # numpy.histogramdd accepts only (N, D) shapes
            D = np_t.shape[-1]
            N = np.prod(np_t.shape[:-1])
            reshaped_t = np.reshape(np_t, (N, D))
            reshaped_wt = np.reshape(np_weights, (N,)) if np_weights is not None else None

            # numpy.histogramdd throws an error for D=0
            if D == 0:
                return (torch.tensor(float('nan') if density else 0.), [])

            # numpy.histogramdd expects range to be specified as a sequence of D (lower, upper) tuples
            reshaped_range = None if not bin_range else [(bin_range[2 * i], bin_range[2 * i + 1]) for i in range(D)]

            (np_hist, np_bin_edges) = np.histogramdd(reshaped_t, np_bins,
                                                     range=reshaped_range, weights=reshaped_wt, density=density)

            return (torch.from_numpy(np_hist).to(dtype), [torch.from_numpy(t).to(dtype) for t in np_bin_edges])

        (actual_hist, actual_bin_edges) = torch.histogramdd(t, bins, range=bin_range, weight=weights, density=density)
        (expected_hist, expected_bin_edges) = reference_histogramdd(t, bins, bin_range, weights, density, actual_hist.dtype)

        D = len(actual_bin_edges)
        self.assertEqual(D, len(expected_bin_edges))

        """
        Works around linspace discrepancies by passing torch's constructed bin_edges to numpy.
        When bin edges are not explicitly defined, histogram uses the linspace operator internally
        to construct the sequence of bin edges. In some cases, torch.linspace output differs slightly
        from numpy.linspace output.
        Issue: https://github.com/pytorch/pytorch/issues/58758
        """
        if not torch.is_tensor(bins):
            for dim in range(D):
                self.assertEqual(actual_bin_edges[dim], expected_bin_edges[dim], atol=1e-5, rtol=1e-5)
            # Calls numpy.histogram again, passing torch's actual_bin_edges as the bins argument
            (expected_hist, expected_bin_edges) = reference_histogramdd(
                t, actual_bin_edges, bin_range, weights, density, actual_hist.dtype)
            self.assertEqual(D, len(expected_bin_edges))

        self.assertEqual(actual_hist, expected_hist)
        for dim in range(D):
            self.assertEqual(actual_bin_edges[dim], expected_bin_edges[dim])

    def test_histogramdd(self):
        # [FUNCTION DOC] Compares torch.histogramdd with numpy over shapes of rank 2
        # to 4 (including zero-size ones), contiguity, weights and density, using a
        # single bin count, per-dimension counts, random range, equal min and max
        # range, and explicit bin edges (contiguous or noncontiguous). CPU, float32.
        shapes = (
            (1, 5),
            (3, 5),
            (1, 5, 1),
            (2, 3, 5),
            (7, 7, 7, 7),
            (16, 8, 4, 2),
            (10, 10, 10),
            (7, 0, 3),
            (5, 0),)

        for contig, bins_contig, weighted, density, shape in \
                product([True, False], [True, False], [True, False], [True, False], shapes):
            D = shape[-1]

            values = make_tensor(shape, dtype=torch.float32, device="cpu", low=-9, high=9, noncontiguous=not contig)
            weights = (
                make_tensor(shape[:-1], dtype=torch.float32, device="cpu", low=0, high=9, noncontiguous=not contig)
                if weighted
                else None
            )

            # Tests passing a single bin count
            bin_ct = random.randint(1, 5)
            self._test_histogramdd_numpy(values, bin_ct, None, weights, density)

            # Tests passing a bin count for each dimension
            bin_ct = [random.randint(1, 5) for dim in range(D)]
            self._test_histogramdd_numpy(values, bin_ct, None, weights, density)

            # Tests with caller-specified histogram range
            bin_range_tuples = [sorted((random.uniform(-9, 9), random.uniform(-9, 9))) for dim in range(D)]
            bin_range = [elt for t in bin_range_tuples for elt in t]
            self._test_histogramdd_numpy(values, bin_ct, bin_range, weights, density)

            # Tests with range min=max
            for dim in range(D):
                bin_range[2 * dim + 1] = bin_range[2 * dim]
            self._test_histogramdd_numpy(values, bin_ct, bin_range, weights, density)

            # Tests with caller-specified bin edges
            bin_edges = [make_tensor(ct + 1, dtype=torch.float32, device="cpu", low=-9, high=9).msort() for ct in bin_ct]
            if not bins_contig:
                # Necessary because msort always produces contiguous output
                bin_edges_noncontig = [
                    make_tensor(ct + 1, dtype=torch.float32, device="cpu", noncontiguous=not bins_contig)
                    for ct in bin_ct
                ]
                for dim in range(D):
                    bin_edges_noncontig[dim].copy_(bin_edges[dim])
                bin_edges = bin_edges_noncontig
            for dim in range(D):
                self.assertEqual(bin_edges[dim].is_contiguous(), bins_contig)
            self._test_histogramdd_numpy(values, bin_edges, None, weights, density)

    def test_histogram_error_handling(self):
        # [FUNCTION DOC] Checks that torch.histogram raises the expected errors:
        # unsupported integer dtype, dtype mismatches among input, bins, weight and
        # out tensors, bins that are not one-dimensional or empty, non-positive bin
        # count, weight with the wrong shape, bin tensor combined with range
        # (TypeError), range with min above max, and a non-finite (nan) range.
        with self.assertRaisesRegex(RuntimeError, 'not implemented for'):
            values = make_tensor((), dtype=torch.int32, device="cpu")
            torch.histogram(values, 1)

        inconsistent_dtype = torch.float64

        with self.assertRaisesRegex(RuntimeError, 'input tensor and bins tensors should have the same dtype'):
            values = make_tensor((), dtype=torch.float32, device="cpu")
            bins = make_tensor((), dtype=inconsistent_dtype, device="cpu")
            torch.histogram(values, bins)

        with self.assertRaisesRegex(RuntimeError, 'input tensor and weight tensor should have the same dtype'):
            values = make_tensor((), dtype=torch.float32, device="cpu")
            weight = make_tensor((), dtype=inconsistent_dtype, device="cpu")
            torch.histogram(values, 1, weight=weight)

        with self.assertRaisesRegex(RuntimeError, 'input tensor and hist tensor should have the same dtype'):
            values = make_tensor((), dtype=torch.float32, device="cpu")
            hist = make_tensor((), dtype=inconsistent_dtype, device="cpu")
            bin_edges = make_tensor((), dtype=torch.float32, device="cpu")
            torch.histogram(values, 1, out=(hist, bin_edges))

        with self.assertRaisesRegex(RuntimeError, 'input tensor and bin_edges tensor should have the same dtype'):
            values = make_tensor((), dtype=torch.float32, device="cpu")
            hist = make_tensor((), dtype=torch.float32, device="cpu")
            bin_edges = make_tensor((), dtype=inconsistent_dtype, device="cpu")
            torch.histogram(values, 1, out=(hist, bin_edges))

        with self.assertRaisesRegex(RuntimeError, 'bins tensor should have one dimension'):
            t = make_tensor((2, 2), dtype=torch.float32, device="cpu")
            torch.histogram(t, t)

        with self.assertRaisesRegex(RuntimeError, 'bins tensor should have at least 1 element'):
            t = make_tensor((0), dtype=torch.float32, device="cpu")
            torch.histogram(t, t)

        with self.assertRaisesRegex(RuntimeError, 'bins must be > 0'):
            values = make_tensor((), dtype=torch.float32, device="cpu")
            torch.histogram(values, -1)

        with self.assertRaisesRegex(RuntimeError, 'if weight tensor is provided it should have the same shape \
as the input tensor excluding its innermost dimension'):
            values = make_tensor((2, 2), dtype=torch.float32, device="cpu")
            weight = make_tensor((1), dtype=torch.float32, device="cpu")
            torch.histogram(values, 1, weight=weight)

        with self.assertRaisesRegex(TypeError, 'received an invalid combination of arguments'):
            values = make_tensor((), dtype=torch.float32, device="cpu")
            bin_edges = make_tensor((), dtype=torch.float32, device="cpu")
            torch.histogram(values, bin_edges, range=(0, 1))

        with self.assertRaisesRegex(RuntimeError, 'min should not exceed max'):
            values = make_tensor((), dtype=torch.float32, device="cpu")
            torch.histogram(values, 2, range=(1, 0))

        with self.assertRaisesRegex(RuntimeError, r'range \[nan, nan\] is not finite'):
            values = torch.tensor([float("nan")], device="cpu", dtype=torch.float32)
            torch.histogram(values, 2)

instantiate_device_type_tests(TestReductions, globals(), allow_xpu=True, allow_mps=True)

if __name__ == '__main__':
    run_tests()
