# Owner(s): ["module: sparse"]
# ruff: noqa: F841
# [MODULE DOC] Tests for sparse tensors in PyTorch: the COO layout (TestSparse) and the
# compressed layouts CSR, CSC, BSR and BSC (TestSparseAny), plus sparse unary ufuncs, masked
# reductions, meta and fake tensors, legacy constructor deprecations and one-off device mixing
# checks. It defines helpers to parametrize over layouts and gradcheck semantics, a gradcheck
# wrapper for low precision dtypes, and a fake-tensor cross-reference mode. At import it wraps
# the module gradcheck, builds OpInfo lists with sparse support (reductions, binary ufuncs,
# like-functions), optionally imports scipy, and at the bottom instantiates the device-type and
# parametrized test classes (excluding meta, allowing MPS for some) before run_tests when run as
# a script.

import torch
import itertools
import functools
import operator
import random
import unittest
import warnings
from torch.testing import make_tensor
from torch.testing._internal.common_utils import TestCase, run_tests, do_test_dtypes, \
    load_tests, TEST_NUMPY, TEST_SCIPY, IS_WINDOWS, gradcheck, coalescedonoff, \
    DeterministicGuard, first_sample, TEST_WITH_CROSSREF, TEST_WITH_ROCM, skipIfTorchDynamo, \
    parametrize, subtest, is_coalesced_indices, suppress_warnings, instantiate_parametrized_tests, \
    skipIfCrossRef, set_warn_always_context
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_mps import mps_ops_modifier
from numbers import Number
from typing import Any
from packaging import version
from torch.testing._internal.common_cuda import \
    (SM80OrLater, TEST_MULTIGPU)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, ops, dtypes, dtypesIfCUDA, dtypesIfMPS, onlyCPU, onlyCUDA, precisionOverride,
     deviceCountAtLeast, OpDTypes, onlyNativeDeviceTypes, skipCUDAIf, expectedFailureMPS,
     largeTensorTest)
from torch.testing._internal.common_methods_invocations import \
    (op_db, reduction_ops, sparse_unary_ufuncs, sparse_masked_reduction_ops, binary_ufuncs)
from torch.testing._internal.common_dtype import (
    all_types, all_types_and_complex, all_mps_types, all_types_and_complex_and, floating_and_complex_types,
    floating_and_complex_types_and, highest_precision_complex, highest_precision_float,
    integral_types, floating_types_and,
)
from torch.testing._internal.opinfo.definitions.sparse import validate_sample_input_sparse
from torch.testing._internal.opinfo.refs import (
    ElementwiseBinaryPythonRefInfo,
    ReductionPythonRefInfo
)
from torch.testing._internal.common_utils import (
    IS_LINUX,
    IS_MACOS,
)

def _op_supports_any_sparse(op):
    # [FUNCTION DOC] HELPER: Returns a truthy value when the OpInfo entry declares support for
    # any sparse layout (COO, CSR, CSC, BSR or BSC) by or-ing the five supports_sparse flags on
    # it. Used to filter op_db into the module-level sparse op lists.
    return (op.supports_sparse
            or op.supports_sparse_csr
            or op.supports_sparse_csc
            or op.supports_sparse_bsr
            or op.supports_sparse_bsc)


reduction_ops_with_sparse_support = [
    op for op in reduction_ops if 'masked.' not in op.name and
    _op_supports_any_sparse(op) and not isinstance(op, ReductionPythonRefInfo)]

binary_ufuncs_with_sparse_support = [
    op for op in binary_ufuncs if _op_supports_any_sparse(op) and
    not isinstance(op, ElementwiseBinaryPythonRefInfo)]

like_fns_with_sparse_support = [op for op in op_db if _op_supports_any_sparse(op) and '_like' in op.name]

if TEST_SCIPY:
    import scipy.sparse

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

def _make_lowp_aware_gradcheck(gradcheck_fn):
    # [FUNCTION DOC] HELPER: Decorator factory that wraps a gradcheck callable. The returned
    # wrapper accepts a function and inputs (a single tensor or a sequence). If every input
    # already has float64 or complex128 dtype it delegates to the original gradcheck unchanged.
    # Otherwise it computes gradients on the original inputs and on CPU float64/complex128
    # clones (sparse clones are coalesced) and compares them densely with allclose, raising
    # AssertionError that reports the input index, dtypes, devices, shapes and max abs
    # difference on mismatch, and returning True on success. The wrapper copies the masked
    # attribute from the wrapped callable when present.
    """
    Wraps a gradcheck function to handle low precision dtypes

    For float64/complex128 inputs: runs gradcheck directly
    For lower precision inputs: compares backward() on device against
    backward() on CPU in float64/complex128
    """
    HIGHP_DTYPES = (torch.float64, torch.complex128)

    def needs_backward_comparison(inputs):
        # [FUNCTION DOC] HELPER: Returns True if any tensor in the inputs has a dtype outside
        # float64/complex128, meaning gradients must be compared against a high precision CPU
        # reference instead of numerically gradchecked.
        return any(inp.dtype not in HIGHP_DTYPES for inp in inputs)

    def clone_inputs_cpu(inputs):
        # [FUNCTION DOC] HELPER: Returns a tuple of detached CPU copies of the inputs: tensors
        # are cast to complex128 (if complex) or float64, sparse ones are coalesced, and
        # requires_grad is restored to match the original; non-tensor inputs are passed
        # through unchanged.
        cloned = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                cloned.append(inp)
                continue
            gradcheck_dtype = torch.complex128 if inp.dtype.is_complex else torch.float64
            c = inp.detach().clone().to("cpu").to(gradcheck_dtype)
            if c.is_sparse:
                c = c.coalesce()
            c = c.requires_grad_(inp.requires_grad)
            cloned.append(c)
        return tuple(cloned)

    def compute_grads(fn, inputs):
        # [FUNCTION DOC] HELPER: Runs the function on the inputs and returns a pair (gradients,
        # differentiable inputs). Gradients are taken of the output with an all-ones grad
        # output with respect to the tensor inputs that require grad, allowing unused inputs
        # (gradient None).
        grad_inputs = [x for x in inputs if isinstance(x, torch.Tensor) and x.requires_grad]
        out = fn(*inputs)
        grads = torch.autograd.grad(out, grad_inputs, torch.ones_like(out), allow_unused=True)
        return grads, grad_inputs

    @functools.wraps(gradcheck_fn)
    def wrapped(fn, inputs, *args, **kwargs):
        # [FUNCTION DOC] HELPER: The wrapper returned by the low-precision-aware gradcheck
        # factory. Normalizes inputs to a tuple, forwards to the original gradcheck when all
        # inputs are high precision, else compares reference (CPU, double precision) gradients
        # with those from the real inputs and raises AssertionError on any mismatch; returns
        # True otherwise.
        inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
        if not needs_backward_comparison(inputs):
            return gradcheck_fn(fn, inputs, *args, **kwargs)

        ref_grads, ref_inputs = compute_grads(fn, clone_inputs_cpu(inputs))
        orig_grads, orig_inputs = compute_grads(fn, inputs)

        for i, (og, rg, o_inp, r_inp) in enumerate(zip(orig_grads, ref_grads, orig_inputs, ref_inputs)):
            og_dense = og.to_dense() if og.is_sparse else og
            rg_dense = rg.to_dense() if rg.is_sparse else rg
            og_dense = og_dense.to('cpu')
            rg_dense = rg_dense.to(device='cpu', dtype=og_dense.dtype)
            if not torch.allclose(og_dense, rg_dense):
                max_diff = (og_dense - rg_dense).abs().max()
                raise AssertionError(
                    f"Gradient mismatch for input {i}:\n"
                    f"  input dtype/device: orig={o_inp.dtype}/{o_inp.device}, ref={r_inp.dtype}/{r_inp.device}\n"
                    f"  shapes: {tuple(og_dense.shape)} vs {tuple(rg_dense.shape)}\n"
                    f"  max abs diff: {max_diff}"
                )
        return True
    if hasattr(gradcheck_fn, 'masked'):
        wrapped.masked = gradcheck_fn.masked
    return wrapped

# batched grad doesn't support sparse
gradcheck = functools.partial(gradcheck, check_batched_grad=False)
gradcheck = _make_lowp_aware_gradcheck(gradcheck)

CUSPARSE_SPMM_COMPLEX128_SUPPORTED = (
    IS_WINDOWS and torch.version.cuda
) or (not IS_WINDOWS and not TEST_WITH_ROCM)

HIPSPARSE_SPMM_COMPLEX128_SUPPORTED = torch.version.hip and version.parse(torch.version.hip.split("-")[0]) >= version.parse("6.0")

def all_sparse_layouts(test_name='layout', include_strided=False):
    # [FUNCTION DOC] HELPER: Builds a parametrize decorator over sparse layouts under the given
    # argument name. The cases are named Strided, SparseCOO, SparseCSR, SparseCSC, SparseBSR and
    # SparseBSC; the strided case is dropped unless include_strided is true.
    return parametrize(test_name, [
        subtest(torch.strided, name='Strided'),
        subtest(torch.sparse_coo, name='SparseCOO'),
        subtest(torch.sparse_csr, name='SparseCSR'),
        subtest(torch.sparse_csc, name='SparseCSC'),
        subtest(torch.sparse_bsr, name='SparseBSR'),
        subtest(torch.sparse_bsc, name='SparseBSC'),
    ][(0 if include_strided else 1):])

def gradcheck_semantics(test_name='gradcheck'):
    # [FUNCTION DOC] HELPER: Builds a parametrize decorator with two cases, sparse and masked,
    # that supply a gradcheck callable under the given argument name. Both callables are the
    # module gradcheck bound to masked false or true, wrapped by the low precision aware
    # wrapper, and tagged with a masked attribute that tests read.
    gradcheck_sparse = functools.partial(gradcheck, masked=False)
    gradcheck_masked = functools.partial(gradcheck, masked=True)
    gradcheck_sparse = _make_lowp_aware_gradcheck(gradcheck_sparse)
    gradcheck_masked = _make_lowp_aware_gradcheck(gradcheck_masked)
    gradcheck_sparse.masked = False
    gradcheck_masked.masked = True
    return parametrize(test_name, [
        subtest(gradcheck_sparse, name='sparse'),
        subtest(gradcheck_masked, name='masked')])


class CrossRefSparseFakeMode(torch._subclasses.CrossRefFakeMode):
    # [CLASS DOC] Fake-tensor cross-reference mode used when TEST_WITH_CROSSREF is set. It
    # runs each op both normally and under fake tensors and compares results, with stride
    # and aliasing checks disabled and a small set of sparse ops skipped via ignore_op. Used
    # by TestSparseBase.run.
    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes the parent cross-ref fake mode with ignore_op as the op
        # filter and with stride checking and alias checking both turned off.
        super().__init__(
            self.ignore_op, check_strides=False,
            check_aliasing=False,
        )  # TODO: enable stride/alias checking

    # empty_like excluded for now due to sparse complex
    # aten._to_dense.default this one is getting called with csc
    @staticmethod
    def ignore_op(func):
        # [FUNCTION DOC] HELPER: Static predicate telling the cross-ref mode whether to skip an
        # op. Returns True when the op overload is one of a fixed tuple of aten overloads
        # (empty_like, set_ with storage offset, sspaddmm out, _spdiags, _to_dense, indices,
        # _indices, values, _values).
        return func in (
            torch.ops.aten.empty_like.default,
            torch.ops.aten.set_.source_Storage_storage_offset,
            torch.ops.aten.sspaddmm.out,
            torch.ops.aten._spdiags.default,
            torch.ops.aten._to_dense.default,
            torch.ops.aten.indices.default,
            torch.ops.aten._indices.default,
            torch.ops.aten.values.default,
            torch.ops.aten._values.default,
        )

class TestSparseLegacyAndDeprecation(TestCase):
    # [CLASS DOC] Tests that the deprecated torch.sparse legacy typed constructors still produce
    # correct tensors and emit a UserWarning exactly once.

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_legacy_warnings(self):
        # [FUNCTION DOC] Defines five nested checks (one per legacy constructor form: no args,
        # cdata, indices and values, indices values and shape, and shape only). The docstring of
        # each is used as the expected warning message. Each check builds a reference with the
        # modern sparse_coo_tensor and compares it to the legacy result. The main loop runs each
        # check twice inside assertWarns and asserts that exactly one warning was recorded (warn
        # once). Skipped under TorchDynamo.

        def f1():
            # [FUNCTION DOC] HELPER: Checks the zero-argument legacy double sparse constructor
            # yields an empty sparse tensor equal to the modern empty one. Its docstring doubles
            # as the expected warning text.
            "torch.sparse.SparseTensor() is deprecated."\
                "  Please use torch.sparse_coo_tensor((0,), dtype=)"
            x_ref = torch.sparse_coo_tensor((0,), dtype=torch.float64)
            x = torch.sparse.DoubleTensor()
            self.assertEqual(x, x_ref)

        def f2():
            # [FUNCTION DOC] HELPER: Checks the legacy constructor taking cdata (a private
            # handle of an existing sparse tensor) matches the source tensor, and that
            # rebuilding from its indices, values and shape also matches. Docstring is the
            # expected warning text.
            "torch.sparse.SparseTensor(cdata=x._cdata) is deprecated."\
                "  Please use torch.sparse_coo_tensor(x._indices(), x._values(), x.shape)"
            x_ref = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64).to_sparse()
            x = torch.sparse.DoubleTensor(cdata=x_ref._cdata)
            y = torch.sparse_coo_tensor(x._indices(), x._values(), x.shape)
            self.assertEqual(x, x_ref)
            self.assertEqual(y, x_ref)

        def f3():
            # [FUNCTION DOC] HELPER: Checks the legacy constructor with indices and values only
            # equals the modern constructor result for the same data in double precision.
            # Docstring is the expected warning text.
            "torch.sparse.SparseTensor(indices, values, *, device=) is deprecated."\
                "  Please use torch.sparse_coo_tensor(indices, values, dtype=, device=)"
            x_ref = torch.sparse_coo_tensor([[0, 0, 1, 1], [0, 1, 0, 1]], [1, 2, 3, 4], dtype=torch.float64)
            x = torch.sparse.DoubleTensor(torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]]),
                                          torch.tensor([1, 2, 3, 4], dtype=torch.float64))
            self.assertEqual(x, x_ref)

        def f4():
            # [FUNCTION DOC] HELPER: Checks the legacy constructor with indices, values and an
            # explicit shape equals the modern constructor result. Docstring is the expected
            # warning text.
            "torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated."\
                "  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=)"
            x_ref = torch.sparse_coo_tensor([[0, 0, 1, 1], [0, 1, 0, 1]], [1, 2, 3, 4], (2, 3), dtype=torch.float64)
            x = torch.sparse.DoubleTensor(torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]]),
                                          torch.tensor([1, 2, 3, 4], dtype=torch.float64), (2, 3))
            self.assertEqual(x, x_ref)

        def f5():
            # [FUNCTION DOC] HELPER: Checks the legacy constructor given only size arguments
            # creates an empty sparse tensor of that shape equal to the modern one. Docstring is
            # the expected warning text.
            "torch.sparse.SparseTensor(shape, *, device=) is deprecated."\
                "  Please use torch.sparse_coo_tensor(shape, dtype=, device=)"
            x_ref = torch.sparse_coo_tensor((2, 3), dtype=torch.float64)
            x = torch.sparse.DoubleTensor(2, 3)
            self.assertEqual(x, x_ref)

        for test_f in [f1, f2, f3, f4, f5]:

            with self.assertWarns(UserWarning, msg=test_f.__doc__) as cm:
                test_f()
                test_f()

            # Check warn-once:
            self.assertEqual(len(cm.warnings), 1)


class TestSparseBase(TestCase):
    # [CLASS DOC] Base class for the sparse tests. Its only job is to run each test inside
    # CrossRefSparseFakeMode when TEST_WITH_CROSSREF is enabled.
    def run(self, result=None):
        # [FUNCTION DOC] Overrides the unittest run entry point: under cross-ref testing it runs
        # the test inside a CrossRefSparseFakeMode context and returns the parent result,
        # otherwise it just defers to the parent run.
        if TEST_WITH_CROSSREF:
            with CrossRefSparseFakeMode():
                return super().run(result)
        else:
            return super().run(result)

class TestSparse(TestSparseBase):
    # [CLASS DOC] Main device-generic suite for sparse COO tensors: construction, coalescing,
    # conversions, arithmetic, matmul variants, reductions, softmax, factories, resizing and
    # broadcasting. setUp installs the factory helpers index_tensor, sparse_empty and
    # sparse_tensor on the instance. Most tests are parametrized by dtype and by coalesced or
    # uncoalesced inputs.

    def setUp(self):
        # [FUNCTION DOC] Calls the parent setUp and then stores three helpers on the test
        # instance: index_tensor (builds an int64 tensor), sparse_empty (torch.empty defaulting
        # to the sparse COO layout) and sparse_tensor (thin wrapper around sparse_coo_tensor).
        super().setUp()

        self.index_tensor = lambda *args, **kwargs: torch.tensor(*args, **kwargs, dtype=torch.int64)

        def sparse_empty_factory(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Creates an empty tensor with the given arguments,
            # defaulting the layout keyword to sparse COO when not supplied. Exposed
            # as self.sparse_empty.
            kwargs['layout'] = kwargs.get('layout', torch.sparse_coo)
            return torch.empty(*args, **kwargs)
        self.sparse_empty = sparse_empty_factory

        def sparse_tensor_factory(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Forwards all arguments to sparse_coo_tensor. Exposed
            # as self.sparse_tensor.
            return torch.sparse_coo_tensor(*args, **kwargs)
        self.sparse_tensor = sparse_tensor_factory

    def _gen_sparse(self, sparse_dim, nnz, with_size, dtype, device, coalesced):
        # [FUNCTION DOC] HELPER: Generates a random sparse COO tensor with the given sparse
        # dimension count, nnz, size, dtype and device via genSparseTensor. A numeric with_size
        # is expanded to that size in every sparse dimension. When coalesced is false it
        # verifies the result is uncoalesced. Returns the tensor, its indices and its values.
        if isinstance(with_size, Number):
            with_size = [with_size] * sparse_dim

        x, i, v = self.genSparseTensor(with_size, sparse_dim, nnz, not coalesced, dtype=dtype, device=device)

        if not coalesced:
            self.assert_uncoalesced(x)

        return x, i, v

    def assert_uncoalesced(self, x):
        # [FUNCTION DOC] HELPER: Sanity check for the uncoalesced generator. Raises
        # AssertionError if the tensor reports itself coalesced; then scans indices and returns
        # True upon finding a duplicate index column, else falls through returning None.
        """
        Test if a CPU tensor is uncoalesced.  This is used to ensure
        correctness of the uncoalesced tensor generation algorithm.
        """
        if x.is_coalesced():
            raise AssertionError("expected tensor to be uncoalesced")
        existing_indices = set()
        indices = x._indices()
        for i in range(x._nnz()):
            index = str(indices[:, i])
            if index in existing_indices:
                return True
            else:
                existing_indices.add(index)

    def randn(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Like torch.randn but works for any device: allocates an empty
        # tensor from the arguments and fills it in place with normal samples.
        """
        Variant of torch.randn that also works in the TEST_CUDA case.
        """
        # TODO: Put this in torch.cuda.randn
        return torch.empty(*args, **kwargs).normal_()

    @dtypes(torch.double)
    @dtypesIfMPS(torch.float32)
    def test_print_coalesced(self, device, dtype):
        # [FUNCTION DOC] Runs the shared print test for coalesced inputs (double, or float32
        # on MPS).
        self._test_print(device, dtype, True)

    @dtypes(torch.double)
    @dtypesIfMPS(torch.float32)
    def test_print_uncoalesced(self, device, dtype):
        # [FUNCTION DOC] Runs the shared print test for uncoalesced inputs (double, or float32
        # on MPS).
        self._test_print(device, dtype, False)

    def _test_print(self, device, dtype, coalesced):
        # [FUNCTION DOC] HELPER: Builds sparse tensors for several shape, sparse_dim and nnz
        # combos (including empty and zero-size cases), makes indices valid and optionally
        # uncoalesced, then for int32 and a float dtype records the printed repr of the
        # tensor, of requires_grad_ and addition results (floating only), and of its _indices
        # and _values. The accumulated text is compared with the stored expected output
        # through assertExpected.
        shape_sparse_dim_nnz = [
            ((), 0, 2),
            ((0,), 0, 10),
            ((2,), 0, 3),
            ((100, 3), 1, 3),
            ((100, 20, 3), 2, 0),
            ((10, 0, 3), 0, 3),
            ((10, 0, 3), 0, 0),
        ]
        printed = []
        for shape, sparse_dim, nnz in shape_sparse_dim_nnz:
            indices_shape = torch.Size((sparse_dim, nnz))
            values_shape = torch.Size((nnz,) + shape[sparse_dim:])
            printed.append(f"# shape: {torch.Size(shape)}")
            printed.append(f"# nnz: {nnz}")
            printed.append(f"# sparse_dim: {sparse_dim}")
            printed.append(f"# indices shape: {indices_shape}")
            printed.append(f"# values shape: {values_shape}")

            indices = torch.arange(indices_shape.numel(), dtype=self.index_tensor(0).dtype,
                                   device=device).view(indices_shape)
            for d in range(sparse_dim):
                indices[d].clamp_(max=(shape[d] - 1))  # make it valid index
            if not coalesced and indices.numel() > 0:
                indices[:, -1] = indices[:, 0]  # make it uncoalesced
            values_numel = values_shape.numel()
            values = torch.arange(values_numel, dtype=dtype,
                                  device=device).view(values_shape).div_(values_numel / 2.)
            sp_tensor = self.sparse_tensor(indices, values, shape, dtype=dtype, device=device)

            dtypes = [torch.int32]
            if values.dtype == torch.double:
                dtypes.append(torch.float)
            else:
                dtypes.append(highest_precision_float(values.device))
            for dtype in dtypes:
                printed.append(f"########## {dtype} ##########")
                x = sp_tensor.detach().to(dtype)
                printed.append("# sparse tensor")
                printed.append(str(x))
                if x.dtype.is_floating_point:
                    printed.append("# after requires_grad_")
                    printed.append(str(x.requires_grad_()))
                    printed.append("# after addition")
                    printed.append(str(x + x))
                printed.append("# _indices")
                printed.append(str(x._indices()))
                printed.append("# _values")
                printed.append(str(x._values()))
            printed.append('')
        self.assertExpected('\n'.join(printed))

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_basic(self, device, dtype, coalesced):
        # [FUNCTION DOC] Checks basic COO invariants for coalesced and uncoalesced inputs:
        # indices and values round trip, ndimension and size, nnz after coalesce, and that the
        # public indices and values accessors raise RuntimeError on uncoalesced tensors but
        # equal the private accessors when coalesced. Also checks that duplicate indices are
        # merged on coalesce (9 unique entries from 16).
        def test_shape(sparse_dims, nnz, with_size):
            # [FUNCTION DOC] HELPER: Generates a sparse tensor with the given sparse dimensions,
            # nnz and size (a number is expanded per dimension) and runs the invariant
            # assertions described above on it.
            if isinstance(with_size, Number):
                with_size = [with_size] * sparse_dims
            x, i, v = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)
            self.assertEqual(i, x._indices())
            self.assertEqual(v, x._values())
            self.assertEqual(x.ndimension(), len(with_size))
            self.assertEqual(x.coalesce()._nnz(), nnz if x.is_coalesced() else nnz // 2)
            self.assertEqual(list(x.size()), with_size)

            # Test .indices() and .values()
            if not coalesced:
                with self.assertRaisesRegex(RuntimeError, "Cannot get indices on an uncoalesced tensor"):
                    x.indices()
                with self.assertRaisesRegex(RuntimeError, "Cannot get values on an uncoalesced tensor"):
                    x.values()
            else:
                self.assertEqual(x.indices(), x._indices())
                self.assertEqual(x.values(), x._values())

        test_shape(3, 10, 100)
        test_shape(3, 10, [100, 100, 100])
        test_shape(3, 10, [100, 100, 100, 5, 5, 5, 0])
        test_shape(3, 0, [0, 0, 100, 5, 5, 5, 0])

        # Make sure that coalesce handles duplicate indices correctly
        i = self.index_tensor([[9, 0, 0, 0, 8, 1, 1, 1, 2, 7, 2, 2, 3, 4, 6, 9]], device=device)
        v = torch.tensor([[idx**2, idx] for idx in range(i.size(1))], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([10, 2]), dtype=dtype, device=device)
        self.assertEqual(x.coalesce()._nnz(), 9)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble, torch.bfloat16)
    @dtypesIfMPS(torch.float32, torch.complex64)
    @precisionOverride({torch.bfloat16: 1e-2})
    def test_coalesce(self, device, dtype, coalesced):
        # [FUNCTION DOC] Tests coalesce against a Python reference for all combinations of empty
        # indices, empty values and empty nnz, for coalesced and uncoalesced inputs (double,
        # cdouble, bfloat16 with looser tolerance).

        def _test_coalesce(t):
            # [FUNCTION DOC] HELPER: Coalesces the tensor, checks dense equality and the
            # coalesced flag, then recomputes the result by summing duplicate index entries in a
            # dict, sorting indices and rebuilding values, and asserts the coalesced indices and
            # values match. A zero nnz tensor is only compared directly and returned. Coalesced
            # inputs must be unchanged.
            tc = t.coalesce()
            self.assertEqual(tc.to_dense(), t.to_dense())
            self.assertTrue(tc.is_coalesced())
            # Our code below doesn't work when nnz is 0, because
            # then it's a 0D tensor, not a 2D tensor.
            if t._nnz() == 0:
                self.assertEqual(t._indices(), tc._indices())
                self.assertEqual(t._values(), tc._values())
                return tc

            value_map: dict[Any, Any] = {}
            for idx, val in zip(t._indices().t(), t._values()):
                idx_tup = tuple(idx.tolist())
                if idx_tup in value_map:
                    value_map[idx_tup] += val
                else:
                    value_map[idx_tup] = val.clone() if isinstance(val, torch.Tensor) else val

            new_indices = sorted(value_map.keys())
            _new_values = [value_map[idx] for idx in new_indices]
            if t._values().ndimension() < 2:
                new_values = t._values().new(_new_values)
            else:
                new_values = torch.stack(_new_values)

            new_indices = t._indices().new(new_indices).t()
            tg = t.new(new_indices, new_values, t.size())

            self.assertEqual(tc._indices(), tg._indices())
            self.assertEqual(tc._values(), tg._values())

            if t.is_coalesced():
                self.assertEqual(tc._indices(), t._indices())
                self.assertEqual(tc._values(), t._values())

        for empty_i, empty_v, empty_nnz in itertools.product([True, False], repeat=3):
            sparse_size = [] if empty_i else [2, 1]
            dense_size = [1, 0, 2] if empty_v else [1, 2]
            nnz = 0 if empty_nnz else 5

            t, _, _ = self._gen_sparse(len(sparse_size), nnz, sparse_size + dense_size, dtype, device, coalesced)
            _test_coalesce(t)  # this tests correctness

    @onlyCUDA
    @largeTensorTest("30GB", "cuda")
    @skipCUDAIf(not SM80OrLater and not TEST_WITH_ROCM, "CUDA capability < SM80 and not ROCM")
    @dtypes(torch.float)
    def test_coalesce_accepts_large_tensor(self, device, dtype):
        # [FUNCTION DOC] CUDA-only large tensor test (needs 30GB, SM80 or ROCm): builds a COO
        # matrix with hundreds of millions of random nnz and checks that coalesce completes
        # without raising. Makes no assertions on the result.
        N = 22500000
        NNZ = 272500000
        rows = torch.randint(0, N, (NNZ,), dtype=torch.int64, device=device)
        cols = torch.randint(0, N, (NNZ,), dtype=torch.int64, device=device)
        indices = torch.stack([rows, cols], dim=0)
        values = torch.randn(NNZ, dtype=dtype, device=device)
        sparse_matrix = torch.sparse_coo_tensor(indices, values, size=(N, N), dtype=torch.float32, device=device)
        sparse_matrix = sparse_matrix.coalesce()

    @dtypes(torch.float32)
    @onlyCPU
    # test_warn_on_sparse_tensor_invariant_checks_disabled must be called exactly once
    def test_warn_on_sparse_tensor_invariant_checks_disabled(self, device, dtype):
        # [FUNCTION DOC] CPU-only: with sparse tensor invariant checks set to the implicit
        # default (None), constructing a sparse COO tensor must emit the UserWarning saying
        # invariant checks are implicitly disabled. The test must run only once because the
        # warning is emitted once per process.
        indices = torch.tensor([[0, 1, 2], [2, 0, 1]])
        values = torch.tensor([1, 2, 3])
        shape = torch.Size([3, 3])
        with torch.sparse.check_sparse_tensor_invariants(None):
            msg = "Sparse invariant checks are implicitly disabled."
            with self.assertWarnsRegex(UserWarning, msg):
                x = torch.sparse_coo_tensor(indices, values, shape)

    @dtypes(torch.float32)
    @expectedFailureMPS
    @skipIfCrossRef
    def test_no_warn_when_check_invariants_is_explicit(self, device, dtype):
        # [FUNCTION DOC] Regression test: the implicit-disabled warning should be emitted only
        # when check_invariants is left as None and not when it is explicitly True or False,
        # across the COO, CSR and CSC constructors with and without size. Not run under cross-
        # ref and expected to fail on MPS.
        # Regression test for https://github.com/pytorch/pytorch/issues/178274
        def assert_invariant_warning(fn, check):
            # [FUNCTION DOC] HELPER: Calls the given constructor thunk with invariant checks
            # unset and warn-always on, recording warnings; asserts that warnings mentioning
            # implicit disabling appear (count nonzero) when the check argument is None and are
            # absent otherwise.
            with torch.sparse.check_sparse_tensor_invariants(None), \
                    set_warn_always_context(True):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    fn()
                    invariant_warnings = [
                        x for x in w if "implicitly disabled" in str(x.message)
                    ]
                    if check is None:
                        self.assertNotEqual(len(invariant_warnings), 0)
                    else:
                        self.assertEqual(len(invariant_warnings), 0)

        for check in [True, False, None]:
            assert_invariant_warning(
                lambda: torch.sparse_coo_tensor(
                    torch.tensor([[0, 1], [2, 0]], device=device),
                    torch.tensor([1, 2], dtype=dtype, device=device),
                    check_invariants=check,
                ), check
            )
            assert_invariant_warning(
                lambda: torch.sparse_coo_tensor(
                    torch.tensor([[0, 1], [2, 0]], device=device),
                    torch.tensor([1, 2], dtype=dtype, device=device),
                    (3, 3),
                    check_invariants=check,
                ), check
            )
            assert_invariant_warning(
                lambda: torch.sparse_coo_tensor(
                    (3, 3),
                    device=device,
                    check_invariants=check,
                ), check
            )
            assert_invariant_warning(
                lambda: torch.sparse_csr_tensor(
                    torch.tensor([0, 2, 4], dtype=torch.int64, device=device),
                    torch.tensor([0, 1, 0, 1], dtype=torch.int64, device=device),
                    torch.tensor([1, 2, 3, 4], dtype=dtype, device=device),
                    (2, 2),
                    check_invariants=check,
                ), check
            )
            assert_invariant_warning(
                lambda: torch.sparse_csc_tensor(
                    torch.tensor([0, 2, 4], dtype=torch.int64, device=device),
                    torch.tensor([0, 1, 0, 1], dtype=torch.int64, device=device),
                    torch.tensor([1, 2, 3, 4], dtype=dtype, device=device),
                    check_invariants=check,
                ), check
            )

    @dtypes(torch.double)
    @dtypesIfMPS(torch.float32)
    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/89395")
    def test_coalesce_reference_cycle(self, device, dtype):
        # [FUNCTION DOC] Verifies that coalesce does not create an autograd reference cycle
        # (gh-52253): first sanity checks that the weak tensor reference helper expires after
        # the tensor is deleted, then asserts a coalesced tensor built inside a nested function
        # is freed after the function returns. Skipped under TorchDynamo.
        # Test coalesce doesn't create autograd graph cycles (gh-52253)

        # Sanity check that the helper class works as expected
        t = torch.rand(2)
        t_ref = torch._C._WeakTensorRef(t)
        self.assertFalse(t_ref.expired())

        del t
        self.assertTrue(t_ref.expired())

        def test_sparse_sum():
            # [FUNCTION DOC] HELPER: Builds a small sparse tensor, coalesces it, sets
            # requires_grad, coalesces again, checks the result is coalesced and returns a weak
            # reference to the second coalesced tensor so the caller can check it expired.
            i = torch.tensor([[0], [4]], dtype=torch.long, device=device)
            v = torch.tensor([[[-0.4567, -1.8797, 0.0380, 1.4316]]],
                             dtype=dtype, device=device)
            S = torch.sparse_coo_tensor(i, v)
            S = S.coalesce()
            S.requires_grad_(True)
            S2 = S.coalesce()
            self.assertTrue(S2.is_coalesced())
            return torch._C._WeakTensorRef(S2)

        ref = test_sparse_sum()
        self.assertTrue(ref.expired())

    @dtypes(torch.double)
    @dtypesIfMPS(torch.float32)
    def test_ctor_large_sizes(self, device, dtype):
        # [FUNCTION DOC] Checks that constructing a sparse COO tensor whose dimensions are so
        # large that numel overflows raises RuntimeError (gh-57416).
        # Test that integer overflow is detected when computing numel
        # of a sparse tensor with large dimensions (gh-57416). Notice
        # that numel is computed internally when constructing a
        # tensor, hence the overflow may appear during the tensor
        # construction step.
        N = 100000
        indices = torch.tensor([[N, N - 1]] * 4, dtype=torch.int64, device=device)
        values = torch.tensor([1, 2], dtype=dtype, device=device)
        self.assertRaises(RuntimeError,
                          lambda: torch.sparse_coo_tensor(
                              indices, values, (N + 1,) * 4, device=device))

    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_ctor_size_checks(self, device, dtype):
        # [FUNCTION DOC] Checks that sparse_coo_tensor raises RuntimeError when the indices are
        # inconsistent with the requested size, and when the values shape is inconsistent with
        # the size.
        indices = self.index_tensor([
            [0, 0, 0],
            [0, 3, 0],
            [0, 0, 0],
            [0, 0, 0],
        ], device=device)
        values = torch.tensor([2, 1, 3, 4], dtype=dtype, device=device)

        # indices inconsistent with size
        self.assertRaises(
            RuntimeError,
            lambda: self.sparse_tensor(indices, values, torch.Size([2, 1, 1])))

        # values inconsistent with size
        values = torch.tensor([
            [2, 1, 2, 1],
            [1, 0, 5, 2],
        ], dtype=dtype, device=device)
        self.assertRaises(
            RuntimeError,
            lambda: self.sparse_tensor(indices, values, torch.Size([2, 4, 2, 1])))

    @expectedFailureMPS
    @coalescedonoff
    @dtypes(torch.double)
    @dtypesIfMPS(torch.float32)
    def test_ctor_is_coalesced_with_gradcheck(self, device, dtype, coalesced):
        # [FUNCTION DOC] For coalesced and uncoalesced tensors, runs torch.autograd.gradcheck on
        # constructing a COO tensor with check_invariants and the is_coalesced flag, with and
        # without an explicit shape. Passing is_coalesced false always works; passing true on
        # uncoalesced indices must raise RuntimeError saying is_coalesced cannot be true for
        # uncoalesced indices. Expected to fail on MPS.
        for sparse_size, nnz in (((3, 3), 5), ((2, 3, 1, 5), 11)):
            t, _, _ = self._gen_sparse(len(sparse_size), nnz, sparse_size, dtype, device, coalesced)
            self.assertEqual(t.is_coalesced(), coalesced)

            def func(indices, values, shape, is_coalesced):
                # [FUNCTION DOC] HELPER: Constructs a sparse COO tensor from indices and values
                # (with the shape only if given) using invariant checks and the is_coalesced
                # flag, asserts the coalesced state matches the flag, and returns the dense form
                # without masked gradients.
                if shape is None:
                    s = torch.sparse_coo_tensor(indices, values, check_invariants=True, is_coalesced=is_coalesced)
                else:
                    s = torch.sparse_coo_tensor(indices, values, shape, check_invariants=True, is_coalesced=is_coalesced)
                self.assertEqual(s.is_coalesced(), is_coalesced)
                return s.to_dense(masked_grad=False)

            for shape in {t.shape, None}:
                if coalesced:
                    torch.autograd.gradcheck(func, (t._indices(), t._values().requires_grad_(True), shape, False))
                    torch.autograd.gradcheck(func, (t._indices(), t._values().requires_grad_(True), shape, True))
                else:
                    torch.autograd.gradcheck(func, (t._indices(), t._values().requires_grad_(True), shape, False))
                    with self.assertRaisesRegex(RuntimeError,
                                                "cannot set is_coalesced to true if indices correspond to uncoalesced COO tensor"):
                        torch.autograd.gradcheck(func, (t._indices(), t._values().requires_grad_(True), shape, True))

    @dtypes(*floating_and_complex_types_and(torch.float16, torch.bfloat16))
    @dtypesIfMPS(*all_mps_types())
    @unittest.skipIf(TEST_WITH_CROSSREF, "generator unsupported triggers assertion error")
    @gradcheck_semantics()
    def test_to_dense_with_gradcheck(self, device, dtype, gradcheck):
        # [FUNCTION DOC] Tests to_dense on hand-built COO tensors (scalar and empty-values
        # hybrid cases) for the highest-precision float and complex value types, comparing
        # against expected dense results, and for float64 additionally running the gradcheck
        # variant (sparse or masked semantics) supplied by the parametrization. Skipped
        # under cross-ref.

        def test_tensor(x, res):
            # [FUNCTION DOC] HELPER: Calls to_dense several times (to catch memory corruption),
            # compares both to_dense and the safeToDense helper against the expected dense
            # tensor after casting. For float64 only, enables requires_grad and gradchecks
            # to_dense with the masked setting of the current gradcheck variant.
            x.to_dense()  # Tests triple to_dense for memory corruption
            x.to_dense()
            x.to_dense()
            dense_x = x.to_dense()
            safe_dense_x = self.safeToDense(x)
            dense_x = dense_x.to(res.dtype)
            safe_dense_x = safe_dense_x.to(res.dtype)
            self.assertEqual(res, dense_x)
            self.assertEqual(res, safe_dense_x)

            # Only run autograd test for float64
            if x.dtype != torch.float64:
                return

            def fn(x):
                # [FUNCTION DOC] HELPER: Function under gradcheck that returns the dense version
                # of its input using the masked_grad setting of the active gradcheck variant.
                return x.to_dense(masked_grad=gradcheck.masked)
            x.requires_grad_(True)
            gradcheck(fn, (x,))

        values_types = [highest_precision_float(device), highest_precision_complex(device)]
        for value_type in values_types:
            i = self.index_tensor([
                [0, 1, 2, 2],
                [0, 0, 0, 3],
                [0, 0, 1, 4],
            ], device=device)
            # we don't have to_dense for half types on CPU because it is implemented
            # with a slower add_ operation
            v = torch.tensor([2, 1, 3, 4], dtype=dtype, device=device)
            x = self.sparse_tensor(i, v, torch.Size([3, 4, 5]), dtype=value_type, device=device)
            res = torch.tensor([
                [[2, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]],
                [[1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]],
                [[0, 3, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 4]],
            ], dtype=dtype, device=device)

            test_tensor(x, res)
            test_tensor(res, res)

            i = self.index_tensor([
                [0, 1, 2, 2],
                [0, 0, 0, 3],
                [0, 0, 1, 4],
            ], device=device)
            v = torch.empty(4, 0, dtype=dtype, device=device)
            x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 0]), dtype=value_type, device=device)
            res = torch.empty((3, 4, 5, 0), dtype=dtype, device=device)
            test_tensor(x, res)

    @coalescedonoff
    @dtypes(torch.float16, torch.bfloat16, torch.float64, torch.int, torch.cfloat, torch.cdouble)
    @dtypesIfMPS(torch.float16, torch.bfloat16, torch.float32, torch.int, torch.cfloat)
    def test_to_sparse(self, device, dtype, coalesced):
        # [FUNCTION DOC] Tests dense to sparse conversion: for increasing numbers of sparse
        # dimensions of a shape, generates a random sparse tensor of several nnz, densifies it
        # and converts back with to_sparse given the sparse dim count; asserts dense equality,
        # equal size and the requested sparse dim.
        shape = [5, 2, 10, 4]
        max_nnz = 1
        dtypes = [highest_precision_float(device), highest_precision_complex(device)]
        for value_type in dtypes:
            for dim, dim_sz in enumerate(shape, 1):
                max_nnz *= dim_sz
                rnnz = torch.randint(2, max_nnz, (1,)).item()
                for nnz in [0, 1, rnnz]:
                    expected, _, _ = self._gen_sparse(dim, nnz, shape, dtype=value_type, device=device,
                                                      coalesced=coalesced)
                    expected = expected.to(dtype)

                    d = expected.to_dense()
                    result = d.to_sparse(dim)
                    self.assertEqual(d, result.to_dense())
                    self.assertEqual(expected.size(), result.size())
                    self.assertEqual(dim, result.sparse_dim())

    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_sparse_bool(self, device, dtype):
        # [FUNCTION DOC] Round trips a boolean tensor through to_sparse and to_dense and
        # checks equality.
        a = torch.tensor([True, False], dtype=dtype, device=device).to(torch.bool)
        b = a.to_sparse().to_dense()
        self.assertEqual(a, b)

    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/183891")
    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/108667")
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_scalar(self, device, dtype):
        # [FUNCTION DOC] Tests zero-dimensional (scalar) COO tensors, with one and with several
        # values: values numel, clone equality, coalesce, conversion to dense equals the scalar,
        # and dense to sparse round trip. Skipped under TorchDynamo.
        # tensor with value
        a = self.sparse_tensor(self.index_tensor([], device=device).unsqueeze(1), 12.3, [], dtype=dtype, device=device)
        self.assertEqual(1, a._values().numel())
        self.assertEqual(a, a.clone())
        a_coalesced = a.coalesce()
        self.assertTrue(a_coalesced.is_coalesced())
        self.assertEqual(torch.tensor(12.3, dtype=dtype, device=device), a.to_dense())
        self.assertEqual(a, a.to_dense().to_sparse())

        # tensor with multiple values
        a = self.sparse_tensor(self.index_tensor([], device=device).unsqueeze(1).expand(0, 2),
                               [12.3, 12.3], [], dtype=dtype, device=device)
        self.assertEqual(2, a._values().numel())
        self.assertEqual(a, a.clone())
        a_coalesced = a.coalesce()
        self.assertTrue(a_coalesced.is_coalesced())
        self.assertEqual(torch.tensor(12.3 * 2, dtype=dtype, device=device), a.to_dense())
        self.assertEqual(a.coalesce(), a.coalesce().to_dense().to_sparse())

        # tensor without value
        a = self.sparse_empty((), dtype=dtype, device=device)
        self.assertEqual(0, a._values().numel())
        self.assertEqual(a, a.clone())
        a_coalesced = a.coalesce()
        self.assertTrue(a_coalesced.is_coalesced())
        self.assertEqual(torch.tensor(0, dtype=dtype, device=device), a.to_dense())
        self.assertEqual(a, a.to_dense().to_sparse())

    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_shared(self, device, dtype):
        # [FUNCTION DOC] Checks that a COO tensor shares storage with the indices and values it
        # was built from: mutating the values tensor or the indices tensor after construction
        # changes the dense result. Repeats with empty-values hybrid tensor for indices.
        i = self.index_tensor([[2]], device=device)
        v = torch.tensor([5], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3]))
        v[0] = 6
        self.assertEqual(torch.tensor([0, 0, 6], dtype=dtype, device=device), self.safeToDense(x))
        i[0][0] = 0
        self.assertEqual(torch.tensor([6, 0, 0], dtype=dtype, device=device), self.safeToDense(x))

        i = self.index_tensor([[2]], device=device)
        v = torch.empty((1, 0), dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 0]))
        i[0][0] = 0
        self.assertEqual(torch.empty((3, 0), dtype=dtype, device=device), self.safeToDense(x))

    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    @unittest.skipIf(TEST_WITH_CROSSREF, "generator unsupported triggers assertion error")
    @gradcheck_semantics()
    def test_to_dense_hybrid(self, device, dtype, gradcheck):
        # [FUNCTION DOC] Tests to_dense on hybrid COO tensors (sparse plus dense dims,
        # including a zero-size dense dim) against hand-written expected dense tensors, then
        # gradchecks to_dense with the masked setting of the current gradcheck variant. Skipped
        # under cross-ref.

        def test_tensor(x, res):
            # [FUNCTION DOC] HELPER: Calls to_dense repeatedly, compares to_dense and
            # safeToDense against the expected tensor, then enables requires_grad and runs
            # gradcheck over to_dense with the masked setting.
            x.to_dense()  # Tests double to_dense for memory corruption
            x.to_dense()
            x.to_dense()
            self.assertEqual(res, x.to_dense())
            self.assertEqual(res, self.safeToDense(x))

            def fn(x):
                # [FUNCTION DOC] HELPER: Function under gradcheck returning the dense version of
                # its input with the masked_grad setting of the current gradcheck variant.
                return x.to_dense(masked_grad=gradcheck.masked)
            x.requires_grad_(True)
            gradcheck(fn, (x,))

        i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
        ], device=device)
        v = torch.tensor([[2, 3], [1, 2], [3, 4], [4, 5]], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 2]))
        res = torch.tensor([
            [[2, 3],
             [0, 0],
             [0, 0],
             [0, 0]],
            [[1, 2],
             [0, 0],
             [0, 0],
             [0, 0]],
            [[3, 4],
             [0, 0],
             [0, 0],
             [4, 5]],
        ], dtype=dtype, device=device)
        test_tensor(x, res)

        i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
        ], device=device)
        v = torch.empty((4, 2, 0), dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 2, 0]))
        res = torch.empty((3, 4, 2, 0), dtype=dtype, device=device)
        test_tensor(x, res)

    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_contig(self, device, dtype):
        # [FUNCTION DOC] Tests that coalesce of non-hybrid COO tensors sorts indices
        # lexicographically and sums duplicates, with hand-written expected indices and values,
        # including unsorted indices, empty trailing dense dimension, and duplicated entries.
        def test_tensor(x, exp_i, exp_v):
            # [FUNCTION DOC] HELPER: Coalesces the tensor and asserts its private indices and
            # values equal the expected ones.
            x = x.coalesce()
            self.assertEqual(exp_i, x._indices())
            self.assertEqual(exp_v, x._values())

        i = self.index_tensor([
            [1, 0, 35, 14, 39, 6, 71, 66, 40, 27],
            [92, 31, 62, 50, 22, 65, 89, 74, 56, 34],
        ], device=device)
        v = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([100, 100]))
        exp_i = self.index_tensor([
            [0, 1, 6, 14, 27, 35, 39, 40, 66, 71],
            [31, 92, 65, 50, 34, 62, 22, 56, 74, 89],
        ], device=device)
        exp_v = torch.tensor([2, 1, 6, 4, 10, 3, 5, 9, 8, 7], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

        i = self.index_tensor([
            [2, 0, 2, 1],
            [0, 0, 3, 0],
            [1, 0, 4, 0],
        ], device=device)
        v = torch.tensor([3, 2, 4, 1], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5]))
        exp_i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ], device=device)
        exp_v = torch.tensor([2, 1, 3, 4], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

        i = self.index_tensor([
            [2, 0, 2, 1],
            [0, 0, 3, 0],
            [1, 0, 4, 0],
        ], device=device)
        v = torch.empty([4, 0], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 0]))
        exp_i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ], device=device)
        exp_v = torch.empty([4, 0], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

        # Duplicate indices
        i = self.index_tensor([
            [0, 0, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 4, 0],
        ], device=device)
        v = torch.tensor([3, 2, 4, 1], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5]))
        exp_i = self.index_tensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ], device=device)
        exp_v = torch.tensor([6, 4], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

        i = self.index_tensor([
            [0, 0, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 4, 0],
        ], device=device)
        v = torch.empty([4, 0], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 0]))
        exp_i = self.index_tensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ], device=device)
        exp_v = torch.empty([2, 0], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_contig_hybrid(self, device, dtype):
        # [FUNCTION DOC] Same as test_contig but for hybrid tensors whose values have dense
        # dimensions (including a zero-size one): checks coalesce ordering and the summing of
        # duplicate index entries row by row.
        def test_tensor(x, exp_i, exp_v):
            # [FUNCTION DOC] HELPER: Coalesces the tensor and asserts its private indices and
            # values equal the expected ones.
            x = x.coalesce()
            self.assertEqual(exp_i, x._indices())
            self.assertEqual(exp_v, x._values())

        i = self.index_tensor([
            [1, 0, 35, 14, 39, 6, 71, 66, 40, 27],
            [92, 31, 62, 50, 22, 65, 89, 74, 56, 34],
        ], device=device)
        v = torch.tensor([
            [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
            [6, 7], [7, 8], [8, 9], [9, 10], [10, 11],
        ], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([100, 100, 2]))
        exp_i = self.index_tensor([
            [0, 1, 6, 14, 27, 35, 39, 40, 66, 71],
            [31, 92, 65, 50, 34, 62, 22, 56, 74, 89],
        ], device=device)
        exp_v = torch.tensor([
            [2, 3], [1, 2], [6, 7], [4, 5], [10, 11],
            [3, 4], [5, 6], [9, 10], [8, 9], [7, 8],
        ], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

        i = self.index_tensor([
            [2, 0, 2, 1],
            [0, 0, 3, 0],
            [1, 0, 4, 0],
        ], device=device)
        v = torch.tensor([[3, 3, 3], [2, 2, 2], [4, 4, 4], [1, 1, 1]], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 3]))
        exp_i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ], device=device)
        exp_v = torch.tensor([[2, 2, 2], [1, 1, 1], [3, 3, 3], [4, 4, 4]], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

        i = self.index_tensor([
            [2, 0, 2, 1],
            [0, 0, 3, 0],
            [1, 0, 4, 0],
        ], device=device)
        v = torch.empty([4, 3, 0], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 3, 0]))
        exp_i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ], device=device)
        exp_v = torch.empty([4, 3, 0], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

        # Duplicate indices
        i = self.index_tensor([
            [0, 0, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 4, 0],
        ], device=device)
        v = torch.tensor([[3, 2, 3], [2, 1, 1], [4, 3, 4], [1, 1, 1]], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 3]))
        exp_i = self.index_tensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ], device=device)
        exp_v = torch.tensor([[6, 4, 5], [4, 3, 4]], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

        i = self.index_tensor([
            [0, 0, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 4, 0],
        ], device=device)
        v = torch.empty([4, 3, 0], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 3, 0]))
        exp_i = self.index_tensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ], device=device)
        exp_v = torch.empty([2, 3, 0], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

    @coalescedonoff
    @dtypesIfMPS(torch.float32, torch.complex64)
    @dtypes(torch.double, torch.cdouble)
    def test_clone(self, device, dtype, coalesced):
        # [FUNCTION DOC] Checks that clone preserves the coalesced flag: an uncoalesced input
        # clones to an uncoalesced tensor, and after coalescing the clone is coalesced, over
        # several sizes including empty ones.
        def test_shape(sparse_dims, nnz, with_size):
            # [FUNCTION DOC] HELPER: Generates a sparse tensor, checks clone keeps the
            # uncoalesced state (when generated uncoalesced), then coalesces and checks the
            # clone is coalesced.
            x = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)[0]
            if not coalesced:
                self.assertFalse(x.is_coalesced())
                y = x.clone()
                self.assertFalse(y.is_coalesced())
            x = x.coalesce()
            self.assertTrue(x.is_coalesced())
            y = x.clone()
            self.assertTrue(y.is_coalesced())

        test_shape(4, 20, 5)
        test_shape(3, 10, [100, 100, 100, 5, 5, 5, 0])
        test_shape(3, 0, [0, 0, 100, 5, 5, 5, 0])

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble, torch.bfloat16)
    @dtypesIfMPS(torch.float32, torch.complex64, torch.bfloat16)
    @precisionOverride({torch.bfloat16: 2e-2})
    def test_Sparse_to_Sparse_copy_(self, device, dtype, coalesced):
        # [FUNCTION DOC] Tests copy_ between sparse tensors: dense equality after copy, the
        # destination dtype being kept when the source has another dtype, RuntimeError when
        # shapes differ (no broadcasting) or when copying from a dense tensor, and autograd:
        # gradient flows to the source (twice the incoming grad when the destination is
        # multiplied by two) while the destination gets no grad.
        # This is for testing torch.copy_(SparseTensor, SparseTensor)
        sparse_dims = 3
        nnz = 10
        sizes = [2, 3, 4, 5]  # hybrid sparse
        x1, _, _ = self._gen_sparse(sparse_dims, nnz, sizes, dtype, device, coalesced)
        x2, _, _ = self._gen_sparse(sparse_dims, nnz + 10, sizes, dtype, device, coalesced)

        # test copy
        x2_dense = x2.to_dense()
        x1.copy_(x2)
        self.assertEqual(x2_dense, x1.to_dense())

        # test type conversion (when x1.copy_(x2), x1.dtype should stay the same)
        x1 = x1.to(torch.float32)

        x2 = x2.to(torch.float16)
        x1_dtype = x1.dtype
        x1.copy_(x2)
        self.assertEqual(x1_dtype, x1.dtype)

        x2 = x2.to(highest_precision_float(device))
        x1_dtype = x1.dtype
        x1.copy_(x2)
        self.assertEqual(x1_dtype, x1.dtype)

        # test no broadcast
        self.assertRaises(RuntimeError, lambda: x1.copy_(x2.narrow_copy(0, 0, 1)))

        # test raise error on copy_() between dense and sparse Tensors
        self.assertRaises(RuntimeError, lambda: x1.copy_(torch.randn(5, 5)))

        # test autograd
        x1, _, _ = self._gen_sparse(sparse_dims, nnz, sizes, dtype, device, coalesced)
        x2, _, _ = self._gen_sparse(sparse_dims, nnz + 10, sizes, dtype, device, coalesced)
        x2.requires_grad_(True)
        x1.copy_(x2)
        y = x1 * 2
        x2_clone = x2.clone()
        y.backward(x2_clone)
        expected_grad = x2_clone * 2
        self.assertEqual(expected_grad.to_dense(), x2.grad.to_dense())
        self.assertEqual(None, x1.grad)

    @coalescedonoff
    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    @dtypes(torch.double, torch.cdouble)
    def test_Sparse_to_Sparse_copy_multi_gpu(self, device, dtype, coalesced):
        # [FUNCTION DOC] Multi-GPU variant of the sparse copy_ test (skipped without multiple
        # GPUs): copies from a tensor on the second GPU and from CPU into a tensor on the first
        # GPU, checks values and that the destination device is unchanged, and checks autograd
        # gradient across devices.
        # This is for testing torch.copy_(SparseTensor, SparseTensor) across GPU devices
        sparse_dims = 3
        nnz = 10
        sizes = [2, 3, 4, 5]  # hybrid sparse
        x1, _, _ = self._gen_sparse(sparse_dims, nnz, sizes, dtype, device, coalesced)
        x2, _, _ = self._gen_sparse(sparse_dims, nnz + 10, sizes, dtype, device, coalesced)
        x1 = x1.to('cuda:0')

        def test_cross_device(x1, x2):
            # [FUNCTION DOC] HELPER: Records the destination device, copies the source into the
            # destination, asserts dense equality with the source moved to the first GPU and
            # that the destination device did not change.
            x1_device = x1.device
            x1.copy_(x2)
            self.assertEqual(x2.to('cuda:0').to_dense(), x1.to_dense())
            self.assertEqual(x1_device, x1.device)

        test_cross_device(x1, x2.to('cuda:1'))  # test across gpu devices
        test_cross_device(x1, x2.to('cpu'))  # test between cpu and gpu

        # test autograd
        x2 = x2.to('cuda:1')
        x2.requires_grad_(True)
        x1.copy_(x2)
        y = x1 * 2
        x2_clone = x2.clone().to('cuda:0')
        y.backward(x2_clone)
        expected_grad = x2_clone * 2
        self.assertEqual(expected_grad.to_dense(), x2.grad.to('cuda:0').to_dense())
        self.assertEqual(None, x1.grad)

    @onlyCUDA
    def test_cuda_empty(self, device):
        # [FUNCTION DOC] CUDA-only: checks that moving empty sparse COO tensors (float32,
        # float16, and one with a trailing zero-size dense dim) to the device and back to CPU
        # preserves sparse_dim and dense_dim.
        def test_tensor(x):
            # [FUNCTION DOC] HELPER: Moves the tensor to the test device, compares sparse and
            # dense dimension counts, brings it back to CPU and compares again.
            y = x.to(device)
            self.assertEqual(x.sparse_dim(), y.sparse_dim())
            self.assertEqual(x.dense_dim(), y.dense_dim())
            x = y.cpu()
            self.assertEqual(y.sparse_dim(), x.sparse_dim())
            self.assertEqual(y.dense_dim(), x.dense_dim())

        x = torch.sparse_coo_tensor((2, 3, 4), dtype=torch.float32)
        test_tensor(x)

        x = torch.sparse_coo_tensor((2, 3, 4), dtype=torch.float16)
        test_tensor(x)

        x = torch.sparse_coo_tensor((2, 3, 4), dtype=torch.float16)
        test_tensor(x)

        x = torch.sparse_coo_tensor((2, 3, 4, 0), dtype=torch.float32)
        test_tensor(x)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_transpose(self, device, dtype, coalesced):
        # [FUNCTION DOC] Checks in-place and out-of-place transpose of COO tensors for every
        # pair among the first four dimensions against the dense transpose, for sizes including
        # zero-size dims and zero nnz.
        def test_shape(sparse_dims, nnz, with_size):
            # [FUNCTION DOC] HELPER: Generates a sparse tensor and its dense form, then for
            # every dimension pair applies transpose_ (in place) and transpose (copy) to both
            # and asserts dense equality each time.
            x = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)[0]
            y = self.safeToDense(x)

            for i, j in itertools.combinations(range(4), 2):
                x = x.transpose_(i, j)
                y = y.transpose(i, j)
                self.assertEqual(self.safeToDense(x), y)

                x = x.transpose(i, j)
                y = y.transpose(i, j)
                self.assertEqual(self.safeToDense(x), y)

        test_shape(4, 6, 3)
        test_shape(4, 3, [7, 7, 7, 3, 3, 3, 0])
        test_shape(4, 0, [0, 0, 7, 3, 3, 3, 0])

    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/184598")
    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    @unittest.skipIf(TEST_WITH_CROSSREF, "generator unsupported triggers assertion error")
    @gradcheck_semantics()
    def test_permute(self, device, dtype, coalesced, gradcheck):
        # [FUNCTION DOC] Tests permute on COO tensors. Trivial checks: wrong number of dims
        # and duplicate dims raise RuntimeError, and permute with an empty tuple on a scalar
        # works (earlier segfault). Then for all permutations of dims: valid ones (not mixing
        # sparse and dense dims) must match dense permute, keep the coalesced flag only if dim
        # 0 stays first (else uncoalesced), and pass gradcheck; invalid ones must raise
        # RuntimeError about transpositions between sparse and dense dimensions. Skipped under
        # dynamo and cross-ref.
        # trivial checks
        s = torch.rand(3, 3, 3, device=device, dtype=dtype).to_sparse()
        with self.assertRaisesRegex(RuntimeError, "does not match the length"):
            s.permute(dims=(1, 0))
        with self.assertRaisesRegex(RuntimeError, "duplicate dims"):
            s.permute(dims=(1, 1, 1))
        # Calling permute on a sparse tensor with an empty tuple used to segfault,
        # see https://github.com/pytorch/pytorch/issues/116325
        x = torch.rand((), device=device, dtype=dtype).to_sparse()
        x.permute(())
        self.assertEqual(len(x.values()), 1)

        def test_shape(sparse_dims, nnz, with_size):
            # [FUNCTION DOC] HELPER: Enumerates every permutation of dimensions for a generated
            # sparse tensor, deciding validity by sorting the sparse part and dense part of the
            # permutation, and runs the correct-result and gradcheck checks or the expected-
            # error check.
            ndim = len(with_size)
            valid_sparse_dims = torch.arange(-ndim, -ndim + sparse_dims)
            valid_dense_dims = torch.arange(-ndim + sparse_dims, 0)

            for dims in itertools.permutations(range(-ndim, 0)):
                s = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)[0]
                d = self.safeToDense(s)

                dims_sparse, _ = torch.tensor(dims[:sparse_dims]).sort()
                dims_dense, _ = torch.tensor(dims[sparse_dims:]).sort()

                if (valid_sparse_dims == dims_sparse).all() and (valid_dense_dims == dims_dense).all():
                    # if valid permutation, test for correctness
                    s_permuted = s.permute(dims)
                    self.assertEqual(s_permuted, d.permute(dims))

                    # if s is coalesced, and perm does not touch 0-dim,
                    # the result has to be coalesced as well
                    if dims[0] == 0:
                        self.assertEqual(s_permuted.is_coalesced(), s.is_coalesced())
                    else:
                        self.assertFalse(s_permuted.is_coalesced())

                    gradcheck(lambda t: t.permute(dims).to_dense(masked_grad=gradcheck.masked), s.requires_grad_())
                else:
                    # otherwise check if exception is thrown
                    fail_message = "transpositions between sparse and dense dimensions are not allowed"
                    with self.assertRaisesRegex(RuntimeError, fail_message):
                        s.permute(dims)

        test_shape(2, 3, [2, 3, 4, 5])
        test_shape(2, 3, [2, 2, 0])
        # if nnz=0, it is not true that t == t.to_dense().to_sparse()
        # unless t.sparse_dim == t.dim (i.e. t is not hybrid)
        test_shape(3, 0, [0, 0, 2])

    @coalescedonoff
    @onlyCPU
    @dtypes(torch.double)
    def test_coalesce_transpose_mm(self, device, dtype, coalesced):
        # [FUNCTION DOC] CPU-only: checks that a coalesced sparse matrix transposed with t is
        # coalesced only when it has no nnz or zero rows, and that mm with the transposed sparse
        # matrix equals the dense matmul, for several sizes including empty ones.
        def test_shape(di, dj, dk, nnz):
            # [FUNCTION DOC] HELPER: For the given sizes, builds a sparse matrix and a random
            # dense matrix, coalesces, transposes, asserts the coalesced flag expectation and
            # compares sparse mm to dense mm.
            x, _, _ = self._gen_sparse(2, nnz, [dj, di], dtype, device, coalesced)
            y = torch.randn(dj, dk, dtype=dtype, device=device)

            x_coalesced = x.coalesce()
            self.assertTrue(x_coalesced.is_coalesced())

            x_coalesced_t = x_coalesced.t()
            # Transpose is `colasced`-preserving if the indices tensor is empty.
            self.assertEqual(x_coalesced_t.is_coalesced(), di * nnz == 0)

            res = torch.mm(x_coalesced_t, y)
            expected = torch.mm(self.safeToDense(x_coalesced_t), y)
            self.assertEqual(res, expected)

        test_shape(10, 20, 30, 20)
        test_shape(0, 20, 30, 0)
        test_shape(10, 0, 30, 0)
        test_shape(10, 20, 0, 0)
        test_shape(10, 20, 0, 20)

    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1166")
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_t_empty(self, device, dtype):
        # [FUNCTION DOC] Checks transpose (in place and not in place) of empty sparse matrices:
        # the shape is swapped, indices and values stay empty, and sparse_dim and dense_dim
        # remain 2 and 0. Skipped under TorchDynamo.
        def test_in_place(x):
            # [FUNCTION DOC] HELPER: Applies in-place transpose to the tensor and asserts
            # swapped shape, empty indices and values, sparse_dim 2 and dense_dim 0.
            shape_original = x.shape
            x.t_()
            self.assertEqual(torch.Size([shape_original[1], shape_original[0]]), x.size())
            self.assertEqual(0, x._indices().numel())
            self.assertEqual(0, x._values().numel())
            self.assertEqual(x.sparse_dim(), 2)
            self.assertEqual(x.dense_dim(), 0)

        def test_not_in_place(x):
            # [FUNCTION DOC] HELPER: Same assertions as the in-place variant, but using the out-
            # of-place transpose and checking the returned tensor; also checks the input still
            # reports sparse_dim 2 and dense_dim 0.
            shape_original = x.shape
            y = x.t()
            self.assertEqual(torch.Size([shape_original[1], shape_original[0]]), y.size())
            self.assertEqual(0, y._indices().numel())
            self.assertEqual(0, y._values().numel())
            self.assertEqual(x.sparse_dim(), 2)
            self.assertEqual(x.dense_dim(), 0)

        x = self.sparse_empty(2, 3, dtype=dtype, device=device)
        test_in_place(x)
        test_not_in_place(x)

        x = self.sparse_empty(2, 0, dtype=dtype, device=device)
        test_in_place(x)
        test_not_in_place(x)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_add_zeros(self, device, dtype, coalesced):
        # [FUNCTION DOC] Checks that adding an all-zero (empty) sparse tensor on either side of
        # a random sparse tensor returns a tensor equal to the random one.
        def test_shape(sparse_dims, nnz, sizes):
            # [FUNCTION DOC] HELPER: Generates a sparse tensor of the given size, adds an
            # empty sparse tensor of the same size on both sides and asserts both sums equal
            # the original.
            x, _, _ = self._gen_sparse(sparse_dims, nnz, sizes, dtype, device, coalesced)
            zeros = torch.sparse_coo_tensor(sizes, device=x.device)
            r1 = zeros + x
            r2 = x + zeros
            self.assertEqual(r1, x)
            self.assertEqual(r2, x)

        test_shape(1, 20, [1])
        test_shape(4, 20, [3, 17, 19, 5])
        test_shape(2, 20, [3, 17, 19, 5])
        test_shape(2, 20, [3, 17, 19, 0])

    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_add_sub_nnz(self, device, dtype):
        # [FUNCTION DOC] Regression test (gh-34964): repeated in-place add and sub of a sparse
        # tensor with itself must not grow nnz beyond the original 10 (coalescing duplicates).
        # nnz should not grow unbounded (gh-34964)
        x = torch.randn(10, dtype=dtype, device=device).to_sparse()
        x.add_(x)
        x.add_(x)
        self.assertLessEqual(x._nnz(), 10)

        x.sub_(2 * x)
        x.sub_(2 * x)
        self.assertLessEqual(x._nnz(), 10)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_cat(self, device, dtype, coalesced):
        # [FUNCTION DOC] Tests torch.cat on sparse tensors along various dims, including hybrid
        # tensors, dense dims, a negative (wrapped) dim, and error cases: mismatched shapes,
        # mismatched sparse and dense dimension counts, and mixing in a dense tensor each raise
        # RuntimeError with the expected message.
        # shapes: list of tuples (sparse_dims, nnz, sizes)
        def test_shapes(shapes, dim, fail_message=None):
            # [FUNCTION DOC] HELPER: Builds the inputs from (sparse dim, nnz, size) triples.
            # When a failure message is given it asserts that cat raises RuntimeError
            # matching it; otherwise it compares the dense of the sparse cat to the cat of
            # the dense inputs.
            inputs = [self._gen_sparse(shape[0], shape[1], shape[2], dtype, device, coalesced)[0]
                      for shape in shapes]
            if fail_message:
                with self.assertRaisesRegex(RuntimeError, fail_message):
                    torch.cat(inputs, dim)
            else:
                result = torch.cat(inputs, dim)
                dense_result = torch.cat([t.to_dense() for t in inputs], dim)
                self.assertEqual(dense_result, result.to_dense())

        test_shapes(
            [(3, 10, [2, 3, 4]), (3, 10, [2, 1, 4]), (3, 10, [2, 4, 4])], 1)

        # mismatched sizes
        test_shapes([(3, 10, [2, 3, 4]), (3, 10, [2, 1, 4])], 0,
                    "All tensors must have the same shape: \\[2, 3, 4].*\\[2, 1, 4]")
        # hybrid sparse/dense
        test_shapes(
            [(2, 10, [2, 3, 4]), (2, 10, [2, 1, 4]), (2, 10, [2, 4, 4])], 1)
        # cat along dense dim
        test_shapes([(2, 10, [2, 3, 4]), (2, 10, [2, 3, 7])], 2)
        test_shapes([(1, 10, [2, 3, 4]), (1, 10, [2, 3, 4])], 1)
        test_shapes([(1, 10, [2, 3, 4]), (1, 10, [2, 3, 4])], 2)
        # mismatched dimensions
        test_shapes([(2, 10, [2, 3, 4]), (3, 10, [2, 3, 4])], 0,
                    "All tensors must have the same.*2, 1, but tensor at position 1 has 3, 0.")
        # wrapped dimension
        test_shapes(
            [(3, 10, [2, 3, 4]), (3, 10, [2, 1, 4]), (3, 10, [2, 4, 4])], -2)

        # sparse with dense
        sp = self._gen_sparse(3, 10, [2, 3, 4], dtype, device, coalesced)[0]
        dn = sp.to_dense()
        with self.assertRaisesRegex(RuntimeError,
                                    "Concatenating sparse tensors, but a dense tensor was found at position 1."):
            torch.cat((sp, dn))

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_unsqueeze(self, device, dtype, coalesced):
        # [FUNCTION DOC] Tests unsqueeze on sparse tensors for sparse and dense positions,
        # positive and negative (wrapped) dims, and checks IndexError with message "Dimension
        # out of range" for out of bound dims.
        def test_shape(sparse_dims, nnz, sizes, unsqueeze_dim, fail_message=None):
            # [FUNCTION DOC] HELPER: Generates a sparse tensor then either asserts unsqueeze
            # raises IndexError matching the message, or compares the unsqueezed sparse tensor
            # in dense form against the dense unsqueeze.
            x, _, _ = self._gen_sparse(sparse_dims, nnz, sizes, dtype, device, coalesced)
            if fail_message:
                with self.assertRaisesRegex(IndexError, fail_message):
                    torch.unsqueeze(x, unsqueeze_dim)
            else:
                result = torch.unsqueeze(x, unsqueeze_dim)
                dense_result = torch.unsqueeze(x.to_dense(), unsqueeze_dim)
                self.assertEqual(dense_result, result.to_dense())

        # basic case
        test_shape(3, 10, [5, 7, 11], 0)

        # hybrid sparse/dense, unsqueeze along sparse dim
        test_shape(3, 10, [5, 7, 11, 13, 17], 0)
        test_shape(3, 10, [5, 7, 11, 13, 17], 3)

        # unsqueeze along dense dimensions
        test_shape(3, 10, [5, 7, 11, 13, 17], 4)
        test_shape(3, 10, [5, 7, 11, 13, 17], 5)

        # wrapped dimensions
        test_shape(3, 10, [5, 7, 11, 13, 17], -1)
        test_shape(3, 10, [5, 7, 11, 13, 17], -6)

        # bounds
        test_shape(3, 10, [5, 7, 11, 13, 17], -7, "Dimension out of range")
        test_shape(3, 10, [5, 7, 11, 13, 17], 6, "Dimension out of range")

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_select(self, device, dtype, coalesced):
        # [FUNCTION DOC] Tests select on sparse tensors: selecting a sparse dim (dense or sparse
        # result depending on hybrid layout), selecting a dense dim, and an out of range index
        # raising IndexError.
        def test_shape(sparse_dims, nnz, sizes, select_dim, select_index, fail_message=None):
            # [FUNCTION DOC] HELPER: Generates a sparse tensor then either asserts select raises
            # IndexError matching the message, or compares the result (densified if sparse) with
            # dense select.
            x, _, _ = self._gen_sparse(sparse_dims, nnz, sizes, dtype, device, coalesced)
            if fail_message:
                with self.assertRaisesRegex(IndexError, fail_message):
                    torch.select(x, select_dim, select_index)
            else:
                result = torch.select(x, select_dim, select_index)
                if result.is_sparse:
                    result = result.to_dense()
                dense_result = torch.select(x.to_dense(), select_dim, select_index)
                self.assertEqual(dense_result, result)


        sizes = [5, 7, 11, 13, 17]
        # hybrid sparse/dense, select sparse dim, result is dense
        for i in range(sizes[0]):
            test_shape(1, 10, sizes, 0, i)
        test_shape(1, 10, sizes, 0, sizes[0] + 1, r'select[(][)][:] index \d out of range.*')

        # hybrid sparse/dense, select sparse dim, result is sparse
        for d in range(3):
            for i in range(sizes[d]):
                test_shape(3, 10, sizes, d, i)

        # hybrid sparse/dense, select dense dim, result is sparse
        for d in range(1, 3):
            for i in range(sizes[d]):
                test_shape(1, 10, sizes, d, i)

    @dtypes(*integral_types())
    def test_select_no_type_promotion(self, device, dtype):
        # [FUNCTION DOC] Regression test (issue 82150) for integral dtypes: selecting from a
        # sparse tensor, including one scaled to all zeros, must not promote the dtype, both for
        # empty results and for summed duplicates.
        # see https://github.com/pytorch/pytorch/issues/82150
        idx = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]])
        val = torch.ones(6, dtype=dtype)
        s = torch.sparse_coo_tensor(idx, val, size=(3, 3))

        for t in (s, s * torch.tensor(0, dtype=dtype)):
            # empty checks
            self.assertEqual(t.dtype, t[2].dtype)
            self.assertEqual(t.dtype, t[0, 1].dtype)
            # sum should not promote
            self.assertEqual(t.dtype, t[0, 0].dtype)
            self.assertEqual(t.dtype, t[1, 1].dtype)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_index_select(self, device, dtype, coalesced):
        # [FUNCTION DOC] Tests index_select on sparse tensors across every dimension with
        # single indices and index lists and several sparse-dim counts, comparing against the
        # dense result.
        def test_shape(sparse_dims, nnz, sizes, select_dim, select_index, fail_message=None):
            # [FUNCTION DOC] HELPER: Converts the index (int or list) into a long tensor,
            # generates a sparse tensor, and either asserts an IndexError with the message or
            # compares sparse index_select (densified) against the dense one.
            if isinstance(select_index, int):
                select_index = [select_index]
            if isinstance(select_index, list):
                select_index = torch.tensor(select_index, device=device, dtype=torch.long)
            x, _, _ = self._gen_sparse(sparse_dims, nnz, sizes, dtype, device, coalesced)
            if fail_message:
                with self.assertRaisesRegex(IndexError, fail_message):
                    torch.index_select(x, select_dim, select_index)
            else:
                result = torch.index_select(x, select_dim, select_index)
                if result.is_sparse:
                    result = result.to_dense()
                dense_result = torch.index_select(x.to_dense(), select_dim, select_index)
                self.assertEqual(dense_result, result)

        sizes = [5, 7, 11, 13, 17]
        for d in range(len(sizes)):
            for index in [0, sizes[d] - 1, [0, sizes[d] // 2, sizes[d] - 1]]:
                test_shape(1, 10, sizes, d, index)
                test_shape(len(sizes) // 2, 10, sizes, d, index)
                test_shape(len(sizes), 10, sizes, d, index)

    def _test_index_select_exhaustive_index(self, sizes, dims, device, dtype, coalesced):
        # [FUNCTION DOC] HELPER: Exhaustive index_select check. For each listed dim it
        # enumerates every length-size index tuple built from negative indices, and compares the
        # sparse result with the dense result (indices shifted by the dim size because dense
        # does not accept negatives) for a dense-converted sparse tensor (nnz larger than the
        # dim) and a small sparse tensor with only two nnz (nnz not larger than the dim).
        t = make_tensor(sizes, dtype=dtype, device=device)
        t_sparse = t.to_sparse().coalesce() if coalesced else t.to_sparse()
        t_small_sparse, _, _ = self._gen_sparse(len(sizes), 2, sizes, dtype, device, coalesced)
        t_small = t_small_sparse.to_dense()
        for d in dims:
            # NOTE: indices are negative
            idx_dim_d_range = list(range(-sizes[d], 0))
            for idx_len in range(sizes[d], sizes[d] + 1):
                # creates all possible valid indices into dim d of length idx_len
                for idx in itertools.product(*itertools.repeat(idx_dim_d_range, idx_len)):
                    t_idx = torch.tensor(idx, dtype=torch.long, device=device)

                    # NOTE: index_select for dense does not support negative indices,
                    # hence + sizes[d]. See https://github.com/pytorch/pytorch/issues/76347

                    # tests the nnz > sizes[d] branch
                    dense_result = t.index_select(d, t_idx + sizes[d])
                    sparse_result = t_sparse.index_select(d, t_idx)
                    self.assertEqual(dense_result, sparse_result)

                    # tests the nnz <= sizes[d] branch
                    small_dense_result = t_small.index_select(d, t_idx + sizes[d])
                    small_sparse_result = t_small_sparse.index_select(d, t_idx)
                    self.assertEqual(small_dense_result, small_sparse_result)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_index_select_exhaustive_index_small(self, device, dtype, coalesced):
        # [FUNCTION DOC] Runs the exhaustive index_select check on a small shape so that the
        # brute-force implementation path is taken.
        # will trigger brute-force algo
        self._test_index_select_exhaustive_index((3, 3, 4), range(3), device, dtype, coalesced)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_index_select_exhaustive_index_large(self, device, dtype, coalesced):
        # [FUNCTION DOC] Runs the exhaustive index_select check on a larger shape and on the
        # last two dims, to exercise the non brute-force algorithms.
        # will trigger more sophisticated algos
        self._test_index_select_exhaustive_index((100, 50, 3, 3), (2, 3), device, dtype, coalesced)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_index_select_empty_and_non_contiguous_index(self, device, dtype, coalesced):
        # [FUNCTION DOC] Checks sparse index_select with an empty index and with a non-
        # contiguous index, compared with dense, both when nnz exceeds the dim size and when it
        # does not.
        # empty index
        idx_empty = torch.tensor([], dtype=torch.long, device=device)
        t = make_tensor((5, 5), dtype=dtype, device=device)
        res_dense = t.index_select(0, idx_empty)
        res_sparse = t.to_sparse().index_select(0, idx_empty)
        self.assertEqual(res_dense, res_sparse)

        # non-contiguous index
        idx = torch.randint(low=0, high=5, size=(10, 2), device=device)[:, 0]

        def run_test(sizes):
            # [FUNCTION DOC] HELPER: For the given size, compares index_select with the shared
            # non-contiguous index on a dense-converted tensor and on a small generated sparse
            # tensor against the dense result.
            # case nnz > size[d]
            t = make_tensor(sizes, dtype=dtype, device=device)
            res_dense = t.index_select(0, idx)
            res_sparse = t.to_sparse().index_select(0, idx)
            self.assertEqual(res_dense, res_sparse)

            # case nnz <= size[d]
            t_small_sparse, _, _ = self._gen_sparse(len(sizes), 2, sizes, dtype, device, coalesced)
            res_sparse = t_small_sparse.index_select(0, idx)
            res_dense = t_small_sparse.to_dense().index_select(0, idx)
            self.assertEqual(res_dense, res_sparse)

        # brute-force
        run_test((10, 10))
        # more sophisticated algos
        run_test((10, 100, 100))

    @onlyCPU
    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_index_select_parallelization(self, device, dtype, coalesced):
        # [FUNCTION DOC] CPU-only: compares sparse index_select with the dense result on 1-D
        # tensors large enough (70000 elements, over twice the parallel grain size) to take the
        # parallel code path, for both the nnz not above dim size and nnz above dim size cases,
        # with indices smaller and larger than nnz.
        """
        Test with sizes that will trigger parallelization (i.e. with sizes
        that are >= at::internal::GRAIN_SIZE)
        """
        def run_test(nnz, size):
            # [FUNCTION DOC] HELPER: Builds a 1-D sparse tensor of the given nnz and size and
            # compares index_select with a short random index (half of nnz) and a long one
            # (twice nnz) against the dense result.
            t_sparse, _, _ = self._gen_sparse(1, nnz, (size,), dtype, device, coalesced)
            t_dense = t_sparse.to_dense()

            # idx_small to (sort) and (binary) search into t_sparse
            idx_small = torch.randint(size, (nnz // 2,), device=device)
            # idx_large to (sort) and (binary) search into idx_large
            # NOTE: when coalesced=True, the (binary) search will be
            # done over t_sparse anyway, as it is already sorted.
            idx_large = torch.randint(size, (nnz * 2,), device=device)
            for idx in (idx_small, idx_large):
                res_dense = t_dense.index_select(0, idx)
                res_sparse = t_sparse.index_select(0, idx)
                self.assertEqual(res_dense, res_sparse)

        # NOTE: GRAIN_SIZE = 32768
        # case nnz <= size[d]
        tlen = 70000  # > 2 * GRAIN_SIZE
        run_test(tlen, tlen)

        # case nnz > size[d]
        run_test(tlen, tlen // 2)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_mm(self, device, dtype, coalesced):
        # [FUNCTION DOC] Compares sparse-dense addmm (with and without alpha and beta) and mm
        # against the dense computation, for several sizes including empty ones.
        def test_shape(di, dj, dk, nnz):
            # [FUNCTION DOC] HELPER: Builds a random sparse matrix and dense operands, then
            # asserts addmm with random alpha and beta, default addmm, and mm all match the
            # dense results.
            x, _, _ = self._gen_sparse(2, nnz, [di, dj], dtype, device, coalesced)
            t = torch.randn(di, dk, dtype=dtype, device=device)
            y = torch.randn(dj, dk, dtype=dtype, device=device)
            alpha = random.random()
            beta = random.random()

            res = torch.addmm(t, x, y, beta=beta, alpha=alpha)
            expected = torch.addmm(t, self.safeToDense(x), y, beta=beta, alpha=alpha)
            self.assertEqual(res, expected)

            res = torch.addmm(t, x, y)
            expected = torch.addmm(t, self.safeToDense(x), y)
            self.assertEqual(res, expected)

            res = torch.mm(x, y)
            expected = torch.mm(self.safeToDense(x), y)
            self.assertEqual(res, expected)

        test_shape(10, 100, 100, 20)
        test_shape(100, 1000, 200, 20)
        test_shape(64, 10000, 300, 20)
        test_shape(0, 100, 100, 0)
        test_shape(10, 0, 100, 0)
        test_shape(10, 100, 0, 0)
        test_shape(10, 100, 0, 20)

    @coalescedonoff
    @dtypes(torch.double)
    @dtypesIfMPS(torch.float32)
    def test_bmm(self, device, dtype, coalesced):
        # [FUNCTION DOC] Tests sparse batched matmul: for several sizes (including empty ones)
        # each batch result of bmm on a stacked sparse tensor must equal the per-matrix mm; then
        # edge batches (all-zero, lower-triangular, identity matrices) are checked against per-
        # matrix products, and the transposed-operands product must equal the original result.
        def test_shape(num_mats, dim_i, dim_j, dim_k, nnz):
            # [FUNCTION DOC] HELPER: Builds lists of sparse and dense matrices, stacks them,
            # runs bmm and compares each batch entry to the mm of the corresponding pair.
            a_list = []
            b_list = []
            for _ in range(num_mats):
                a_mat = self._gen_sparse(2, nnz, [dim_i, dim_j], dtype, device, coalesced)[0]
                b_mat = torch.randn([dim_j, dim_k], dtype=dtype, device=device)
                a_list.append(a_mat)
                b_list.append(b_mat)

            a = torch.stack(a_list)
            b = torch.stack(b_list)
            ab = a.bmm(b)

            # Compare each matrix against result from mm()
            for mat_idx in range(num_mats):
                a_mat = a_list[mat_idx]
                b_mat = b_list[mat_idx]
                ab_mat_bmm = ab[mat_idx]
                ab_mat_mm = a_mat.mm(b_mat)
                self.assertEqual(ab_mat_bmm, ab_mat_mm)

        test_shape(10, 10, 100, 99, 20)
        test_shape(10, 100, 1000, 200, 20)
        test_shape(10, 64, 10000, 300, 20)
        test_shape(10, 0, 100, 99, 0)
        test_shape(10, 10, 0, 100, 0)
        test_shape(10, 10, 100, 0, 0)
        test_shape(10, 10, 100, 0, 20)
        test_shape(10, 10, 100, 0, 20)

        a = torch.rand([10, 23, 32], dtype=dtype, device=device)
        a[3].zero_()
        a[6].zero_()
        a[7].tril_()
        a[8].zero_().diagonal().fill_(1)
        a = a.to_sparse()
        b = torch.rand([10, 32, 10], dtype=dtype, device=device)
        b[4].zero_()
        b[6].zero_()
        b[7].tril_()
        b[8].zero_().diagonal().fill_(1)
        ab = a.bmm(b)
        for mat_idx in range(ab.size(0)):
            ab_mat = ab[mat_idx]
            ab_mat_check = a[mat_idx].mm(b[mat_idx])
            self.assertEqual(ab_mat, ab_mat_check)

        ab_traspose_check = b.transpose(1, 2).to_sparse().bmm(
            a.transpose(1, 2).to_dense()
        ).transpose(1, 2)
        self.assertEqual(ab, ab_traspose_check)

    @onlyCUDA
    @coalescedonoff
    @dtypes(torch.double)
    def test_bmm_deterministic(self, device, dtype, coalesced):
        # [FUNCTION DOC] CUDA-only: runs sparse bmm with deterministic algorithms off and on,
        # and asserts the results are either exactly equal or within a relative difference of
        # 0.001 (nan treated as zero), for several sizes including empty ones.
        def test_shape(num_mats, dim_i, dim_j, dim_k, nnz):
            # [FUNCTION DOC] HELPER: Builds the stacked sparse and dense operands on CUDA,
            # computes bmm under a DeterministicGuard first with nondeterministic and then
            # deterministic algorithms and asserts the results agree within tolerance.
            a_list = []
            b_list = []
            for _ in range(num_mats):
                a_list.append(self._gen_sparse(2, nnz, [dim_i, dim_j], dtype, device, coalesced)[0])
                b_list.append(torch.randn([dim_j, dim_k], dtype=dtype, device=device))

            a = torch.stack(a_list).cuda()
            b = torch.stack(b_list).cuda()
            with DeterministicGuard(torch.are_deterministic_algorithms_enabled()):
                torch.use_deterministic_algorithms(False)
                ab_nondeterministic = torch.bmm(a, b)
                torch.use_deterministic_algorithms(True)
                ab_deterministic = torch.bmm(a, b)
            diff_abs = (ab_deterministic - ab_nondeterministic).abs()
            diff_rel = diff_abs / ab_deterministic.abs()
            diff_rel[torch.isnan(diff_rel)] = 0

            # deterministic and non-deterministic results should either be
            # equal or within a small relative difference
            equal_abs_or_rel = diff_abs.eq(0).logical_or(diff_rel.lt(0.001))
            self.assertTrue(equal_abs_or_rel.all())

        test_shape(10, 10, 100, 99, 20)
        test_shape(10, 100, 1000, 200, 20)
        test_shape(10, 64, 10000, 300, 20)
        test_shape(10, 0, 100, 99, 0)
        test_shape(10, 10, 0, 100, 0)
        test_shape(10, 10, 100, 0, 0)
        test_shape(10, 10, 100, 0, 20)
        test_shape(10, 10, 100, 0, 20)

    @onlyCUDA
    def test_bmm_oob(self, device):
        # [FUNCTION DOC] CUDA-only regression test (issue 131977): sparse bmm where the first
        # batch has no non-zero entries must not read out of bounds and must return zeros.
        # Empties the cache first so a bad access would cause an illegal memory access.
        # Targets an out of bounds error when the sparse tensor has no non-zero
        # values in the first batch dimension (#131977).
        # NOTE: This test is separated from the other bmm tests to avoid
        # interference from prior memory allocations on the device. Since CUDA
        # doesn't perform bounds checking, we need the error to cause an
        # illegal memory access (by indexing into unallocated memory) for the
        # test to fail.
        torch.cuda.empty_cache()
        indices = torch.tensor([[1], [0], [0]], device=device)
        values = torch.tensor([1.], device=device)
        a = torch.sparse_coo_tensor(indices, values, size=(2, 1, 1))
        b = torch.zeros((2, 1, 1), device=device)
        ab = torch.bmm(a, b)
        self.assertEqual(ab, torch.zeros((2, 1, 1), device=device))

    @onlyCUDA
    @dtypes(torch.double)
    @dtypesIfMPS(torch.float32)
    def test_bmm_coo_row_index_alignment(self, device, dtype):
        # [FUNCTION DOC] CUDA regression test (issue 167901) for COO bmm with a very tall batch.
        # Checks the row index pointers alternate between 16 byte aligned (odd batches) and 8
        # byte offset (even batches) positions, then asserts the sparse bmm result equals the
        # dense bmm.
        # See https://github.com/pytorch/pytorch/issues/167901
        # cuSPARSE SpMV has a bug before cuda 13.1 which assumes
        # 16-byte alignment for COO row index array.
        batch = 5
        nrows = 291105

        m1 = torch.randn(batch, nrows, 1, device=device, dtype=dtype)
        m2 = torch.randn(batch, 1, 1, device=device, dtype=dtype)
        ref = torch.bmm(m1, m2)

        m1_sparse = m1.to_sparse()
        row_indices = m1_sparse.indices()[1]
        del m1

        # Row indices ptr for odd matrices should be 16-byte aligned
        for i in range(1, batch, 2):
            self.assertTrue(
                (row_indices[i * nrows].data_ptr() % 16) == 0
            )
        # Row indices ptr for even matrices should be 8-byte aligned
        for i in range(0, batch, 2):
            self.assertTrue(
                (row_indices[i * nrows].data_ptr() % 16) == 8
            )

        res = torch.bmm(m1_sparse, m2)
        self.assertEqual(ref, res)

    @onlyCPU
    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_saddmm(self, device, dtype, coalesced):
        # [FUNCTION DOC] CPU-only: compares saddmm (with and without alpha and beta) and smm
        # outputs, converted to dense, against the dense addmm and mm, for several sizes.
        def test_shape(di, dj, dk, nnz):
            # [FUNCTION DOC] HELPER: Builds sparse and dense operands and asserts saddmm
            # with random scale factors, saddmm with defaults and smm all match their
            # dense counterparts.
            x = self._gen_sparse(2, nnz, [di, dj], dtype, device, coalesced)[0]
            t = self._gen_sparse(2, nnz, [di, dk], dtype, device, coalesced)[0]
            y = torch.randn(dj, dk, dtype=dtype, device=device)
            alpha = random.random()
            beta = random.random()

            res = torch.saddmm(t, x, y, beta=beta, alpha=alpha)
            expected = torch.addmm(self.safeToDense(t), self.safeToDense(x), y, beta=beta, alpha=alpha)
            self.assertEqual(self.safeToDense(res), expected)

            res = torch.saddmm(t, x, y)
            expected = torch.addmm(self.safeToDense(t), self.safeToDense(x), y)
            self.assertEqual(self.safeToDense(res), expected)

            res = torch.smm(x, y)
            expected = torch.mm(self.safeToDense(x), y)
            self.assertEqual(self.safeToDense(res), expected)

        test_shape(7, 5, 3, 20)
        test_shape(1000, 100, 100, 20)
        test_shape(3000, 64, 300, 20)
        test_shape(0, 100, 100, 0)
        test_shape(1000, 0, 100, 0)
        test_shape(1000, 100, 0, 0)

    @onlyCPU
    @coalescedonoff
    # adding a graph break before self.assertFalse(weight._indices().is_contiguous())
    # makes the test pass so some existent sparse related bug
    @skipIfTorchDynamo("skip")
    @dtypes(torch.double, torch.cdouble)
    def test_sspaddmm(self, device, dtype, coalesced):
        # [FUNCTION DOC] CPU-only: compares sspaddmm against dense addmm for several sizes, then
        # runs a regression for issue 45113 using a coalesced weight whose indices tensor is
        # non-contiguous (checked) with sspaddmm on a concatenated bias, comparing against the
        # dense result. Skipped under TorchDynamo.

        def test_shape(di, dj, dk, nnz):
            # [FUNCTION DOC] HELPER: Builds sparse and dense operands and asserts sspaddmm with
            # random alpha and beta, and with defaults, match the dense addmm.
            x = self._gen_sparse(2, nnz, [di, dj], dtype, device, coalesced)[0]
            t = self._gen_sparse(2, nnz, [di, dk], dtype, device, coalesced)[0]
            y = torch.randn(dj, dk, dtype=dtype, device=device)
            alpha = random.random()
            beta = random.random()

            res = t.sspaddmm(x, y, beta=beta, alpha=alpha)
            expected = torch.addmm(self.safeToDense(t), self.safeToDense(x), y, beta=beta, alpha=alpha)
            self.assertEqual(self.safeToDense(res), expected)

            res = t.sspaddmm(x, y)
            expected = torch.addmm(self.safeToDense(t), self.safeToDense(x), y)
            self.assertEqual(self.safeToDense(res), expected)

        test_shape(7, 5, 3, 20)
        test_shape(1000, 100, 100, 20)
        test_shape(3000, 64, 300, 20)
        test_shape(0, 100, 100, 0)
        test_shape(1000, 0, 100, 0)
        test_shape(1000, 100, 0, 0)

        # Test code from issue https://github.com/pytorch/pytorch/issues/45113
        batch_size, input_size, hidden_size = 5, 3, 7

        # Create coalesced sparse tensor with non-contiguous indices
        weight = torch.randn(hidden_size, input_size, dtype=dtype, device=device).to_sparse()
        self.assertTrue(weight.is_coalesced())
        non_contig_indices = weight.indices().mT.contiguous().mT
        weight = torch.sparse_coo_tensor(
            indices=non_contig_indices, values=weight.values(), size=weight.shape)
        weight._coalesced_(True)
        self.assertFalse(weight._indices().is_contiguous())
        # Create un/coalesced sparse tensor
        bias = torch.randn((hidden_size, 1), dtype=dtype, device=device).to_sparse()
        bias = torch.cat([bias] * batch_size, dim=1)

        if coalesced:
            bias = bias.coalesce()

        x = torch.randn(input_size, batch_size, dtype=dtype, device=device)
        res = bias.sspaddmm(weight, x)

        true_result = (bias.to_dense() + torch.matmul(weight.to_dense(), x)).to_sparse()
        self.assertEqual(self.safeToDense(res), self.safeToDense(true_result))

    @onlyCPU
    @dtypes(torch.double, torch.cdouble)
    def test_sspaddmm_wrong_mat_types_error_messages(self, device, dtype):
        # [FUNCTION DOC] CPU-only: checks sspaddmm raises RuntimeError with the expected message
        # when mat1 is dense rather than sparse, and when mat2 is sparse rather than strided.
        m, k, n = 2, 2, 2
        self_sp = self._gen_sparse(2, 4, [m, n], dtype, device, True)[0]
        mat1_dense = torch.randn(m, k, dtype=dtype, device=device)
        mat2_dense = torch.randn(k, n, dtype=dtype, device=device)
        with self.assertRaisesRegex(
                RuntimeError,
                "sspaddmm: expected 'mat1' to have sparse layout, got 'mat1' with layout Strided"):
            self_sp.sspaddmm(mat1_dense, mat2_dense)

        mat1_sparse = self._gen_sparse(2, 3, [m, k], dtype, device, True)[0]
        mat2_sparse = self._gen_sparse(2, 3, [k, n], dtype, device, True)[0]
        with self.assertRaisesRegex(
                RuntimeError,
                "sspaddmm: expected 'mat2' to have strided layout, got 'mat2' with layout Sparse"):
            self_sp.sspaddmm(mat1_sparse, mat2_sparse)

    @coalescedonoff
    @precisionOverride({torch.bfloat16: 5e-2, torch.float16: 5e-2})
    @dtypes(torch.double, torch.cdouble, torch.bfloat16, torch.float16)
    @dtypesIfMPS(torch.float32, torch.complex64, torch.bfloat16, torch.float16)
    def test_sparse_addmm(self, device, dtype, coalesced):
        # [FUNCTION DOC] Tests torch.sparse.addmm against dense addmm for double, complex,
        # bfloat16 and half (skipped for bf16 and half on CUDA), with broadcast and full-shaped
        # bias and several alpha and beta values; for double precision types it also gradchecks
        # with masked semantics.
        if (dtype is torch.bfloat16 or dtype is torch.float16) and device.startswith("cuda"):
            self.skipTest('addmm_sparse_cuda is not implemented for BFloat16 and Half')


        def test_shape(m, n, p, nnz, broadcast, alpha_beta=None):
            # [FUNCTION DOC] HELPER: Builds a bias (scalar when broadcast, else full), dense
            # matrix and sparse matrix requiring grad, compares sparse addmm with dense addmm,
            # and gradchecks it when dtype is double or complex double.
            if alpha_beta is None:
                alpha = random.random()
                beta = random.random()
            else:
                alpha, beta = alpha_beta
            if broadcast:
                D1 = make_tensor((), dtype=dtype, device=device, requires_grad=True)
            else:
                D1 = make_tensor([n, p], dtype=dtype, device=device, requires_grad=True)
            D2 = make_tensor([m, p], dtype=dtype, device=device, requires_grad=True)
            S = self._gen_sparse(2, nnz, [n, m], dtype, device, coalesced)[0]
            S_dense = S.to_dense().requires_grad_(True)
            S.requires_grad_(True)
            Y = torch.sparse.addmm(D1, S, D2, beta=beta, alpha=alpha)
            Y_dense = torch.addmm(D1, S_dense, D2, beta=beta, alpha=alpha)
            self.assertEqual(Y, Y_dense)

            if dtype not in {torch.double, torch.cdouble}:
                # gradcheck will likely fail with low-precision input dtypes.
                return

            def fn(S, D1, D2, beta=beta, alpha=alpha):
                # [FUNCTION DOC] HELPER: Function under gradcheck: calls sparse addmm with
                # the given sparse matrix, bias and dense matrix, with alpha and beta bound
                # as defaults.
                return torch.sparse.addmm(D1, S, D2, beta=beta, alpha=alpha)
            gradcheck(fn, (S, D1, D2), masked=True)

        test_shape(7, 8, 9, 20, False, None)
        test_shape(7, 8, 9, 20, True, None)
        test_shape(7, 8, 9, 20, False, (1, 0))
        test_shape(7, 8, 9, 20, True, (1, 0))
        test_shape(7, 8, 9, 20, False, (1, 1))
        test_shape(7, 8, 9, 20, True, (1, 1))

    @coalescedonoff
    @dtypes(torch.double)
    @dtypesIfMPS(torch.float32)
    @unittest.skipIf(TEST_WITH_CROSSREF, "generator unsupported triggers assertion error")
    def test_sparse_mm(self, device, dtype, coalesced):
        # [FUNCTION DOC] Tests torch.sparse.mm against dense mm, with a contiguous and a
        # transposed dense operand, and gradchecks it with masked semantics. Skipped
        # under cross-ref.
        def test_shape(d1, d2, d3, nnz, transposed):
            # [FUNCTION DOC] HELPER: Builds the dense (optionally transposed) and sparse
            # operands requiring grad, compares sparse mm with dense mm, then gradchecks.
            if transposed:
                D = torch.randn(d3, d2, dtype=dtype,
                                device=device).t_().requires_grad_(True)
            else:
                D = torch.randn(d2, d3, dtype=dtype, device=device).requires_grad_(True)
            S = self._gen_sparse(2, nnz, [d1, d2], dtype, device, coalesced)[0]
            S_dense = S.to_dense().requires_grad_(True)
            S.requires_grad_(True)
            self.assertEqual(torch.sparse.mm(S, D), torch.mm(S_dense, D))

            def fn(S, D):
                # [FUNCTION DOC] HELPER: Function under gradcheck returning sparse mm of the
                # sparse and dense inputs.
                return torch.sparse.mm(S, D)

            gradcheck(fn, (S, D), masked=True)

        test_shape(7, 8, 9, 20, False)
        test_shape(7, 8, 9, 20, True)

    @unittest.skipIf(IS_LINUX or IS_MACOS or TEST_WITH_ROCM or IS_WINDOWS, "https://github.com/pytorch/pytorch/issues/174389")
    @coalescedonoff
    @dtypes(torch.double)
    @dtypesIfMPS(torch.float32)
    @unittest.skipIf(TEST_WITH_CROSSREF, "generator unsupported triggers assertion error")
    @gradcheck_semantics()
    def test_sparse_mul(self, device, dtype, coalesced, gradcheck):
        # [FUNCTION DOC] Gradchecks elementwise multiplication of sparse tensors (and the sum of
        # the product) in sparse and masked semantics, and checks the dense equality of the
        # product for several shapes. Also skipped under cross-ref.
        # KNOWN ISSUE: the first skip condition is true on Linux, macOS, ROCm and Windows, so
        # the test effectively never runs (issue 174389).
        # check_batched_grad=False: slow gradcheck's batched/vmap Jacobian
        # path calls aten::view which is unsupported for sparse tensors.
        # https://github.com/pytorch/pytorch/issues/79914
        a = torch.tensor([[0., 1]], dtype=dtype, device=device).to_sparse().requires_grad_(True)
        b = torch.tensor([[0., 1]], dtype=dtype, device=device).to_sparse().requires_grad_(True)
        gradcheck(
            lambda x, y: torch.sparse.sum(x * y).to_dense(masked_grad=gradcheck.masked),
            [a, b],
            check_batched_grad=False,
        )

        def test_shape(sparse_dims, nnz, with_shape):
            # [FUNCTION DOC] HELPER: Generates two sparse tensors requiring grad, asserts the
            # product matches the dense product and gradchecks the product (and its sparse sum
            # with masked semantics).
            a = self._gen_sparse(sparse_dims, nnz, with_shape, dtype, device, coalesced)[0].requires_grad_(True)
            b = self._gen_sparse(sparse_dims, nnz, with_shape, dtype, device, coalesced)[0].requires_grad_(True)

            self.assertEqual((a * b).to_dense(), a.to_dense() * b.to_dense())
            gradcheck(lambda x, y: (x * y).to_dense(), [a, b], check_batched_grad=False)
            # Issues with 0-dim indices/values
            gradcheck(lambda x, y: torch.sparse.sum(x * y).to_dense(), [a, b], masked=True, check_batched_grad=False)

        test_shape(2, 3, [2, 3, 4, 5])
        test_shape(2, 3, [2, 2, 0])
        test_shape(2, 3, [4, 5])

    @coalescedonoff
    @dtypes(torch.double)
    @dtypesIfMPS(torch.float32)
    def test_dsmm(self, device, dtype, coalesced):
        # [FUNCTION DOC] Compares dsmm (dense result of sparse times dense) with dense mm for
        # several sizes including empty ones.
        def test_shape(di, dj, dk, nnz):
            # [FUNCTION DOC] HELPER: Builds a sparse and a dense matrix and asserts dsmm equals
            # dense mm.
            x = self._gen_sparse(2, nnz, [di, dj], dtype, device, coalesced)[0]
            y = self.randn(dj, dk, dtype=dtype, device=device)

            res = torch.dsmm(x, y)
            expected = torch.mm(self.safeToDense(x), y)
            self.assertEqual(res, expected)

        test_shape(7, 5, 3, 20)
        test_shape(1000, 100, 100, 20)
        test_shape(3000, 64, 300, 20)
        test_shape(0, 100, 100, 0)
        test_shape(1000, 0, 100, 0)
        test_shape(1000, 100, 0, 0)
        test_shape(1000, 100, 0, 20)

    @coalescedonoff
    @expectedFailureMPS
    @dtypes(torch.double)
    @dtypesIfMPS(torch.float32)
    def test_hsmm(self, device, dtype, coalesced):
        # [FUNCTION DOC] Compares hsmm (hybrid sparse result of sparse times dense, densified)
        # with dense mm for several sizes. Expected to fail on MPS.
        def test_shape(di, dj, dk, nnz):
            # [FUNCTION DOC] HELPER: Builds a sparse and a dense matrix and asserts the
            # densified hsmm result equals dense mm.
            x = self._gen_sparse(2, nnz, [di, dj], dtype, device, coalesced)[0]
            y = self.randn(dj, dk, dtype=dtype, device=device)

            res = torch.hsmm(x, y)
            expected = torch.mm(self.safeToDense(x), y)
            self.assertEqual(res.to_dense(), expected)

        test_shape(7, 5, 3, 20)
        test_shape(1000, 100, 100, 20)
        test_shape(3000, 64, 300, 20)
        test_shape(0, 100, 100, 0)
        test_shape(1000, 0, 100, 0)
        test_shape(1000, 100, 0, 0)
        test_shape(1000, 100, 0, 20)

    @coalescedonoff
    @dtypes(torch.double)
    @dtypesIfMPS(torch.float32)
    def test_spadd(self, device, dtype, coalesced):
        # [FUNCTION DOC] Tests adding a sparse tensor into a dense tensor with a scale factor
        # for plain and hybrid shapes (including zero-size ones), covering a transposed (non-
        # contiguous) dense operand and sparse operands built from strided slices of the
        # indices, of the values, and of both.

        def _test_spadd_shape(nnz, shape_i, shape_v=None):
            # [FUNCTION DOC] HELPER: For one shape and nnz, checks torch.add of a dense tensor
            # and a sparse tensor with random alpha against the dense formula for a contiguous
            # dense operand, a non-contiguous dense operand, and three sparse variants with non-
            # contiguous indices and/or values.
            shape = shape_i + (shape_v or [])
            x, _, _ = self._gen_sparse(len(shape_i), nnz, shape, dtype, device, coalesced)
            y = self.randn(*shape, dtype=dtype, device=device)
            r = random.random()

            res = torch.add(y, x, alpha=r)
            expected = y + r * self.safeToDense(x)

            self.assertEqual(res, expected)

            # Non contiguous dense tensor
            s = list(shape)
            s[0] = shape[-1]
            s[-1] = shape[0]
            y = self.randn(*s, dtype=dtype, device=device)
            y.transpose_(0, len(s) - 1)
            r = random.random()

            res = torch.add(y, x, alpha=r)
            expected = y + r * self.safeToDense(x)

            self.assertEqual(res, expected)

            x, i, v = self._gen_sparse(len(shape_i), nnz, shape, dtype, device, coalesced)
            nnz = i.size(1)

            # Non contiguous sparse indices tensor
            x_ = self.sparse_tensor(i[:, ::2], v[:(nnz + 1) // 2], x.shape, dtype=dtype, device=device)
            res = torch.add(y, x_, alpha=r)
            expected = y + r * self.safeToDense(x_)
            self.assertEqual(res, expected)

            # Non contiguous sparse values tensor

            x_ = self.sparse_tensor(i[:, :(nnz + 1) // 2], v[::2], x.shape, dtype=dtype, device=device)
            res = torch.add(y, x_, alpha=r)
            expected = y + r * self.safeToDense(x_)
            self.assertEqual(res, expected)

            # Non contiguous sparse indices and values tensors
            x_ = self.sparse_tensor(i[:, 1::2], v[1::2], x.shape, dtype=dtype, device=device)
            res = torch.add(y, x_, alpha=r)
            expected = y + r * self.safeToDense(x_)
            self.assertEqual(res, expected)

        def _test_spadd():
            # [FUNCTION DOC] HELPER: Runs the sparse-dense add check for non-hybrid shapes
            # including zero nnz and zero-size dims.
            _test_spadd_shape(10, [5, 6])
            _test_spadd_shape(10, [10, 10, 10])
            _test_spadd_shape(10, [50, 30, 20])
            _test_spadd_shape(10, [5, 5, 5, 5, 5, 5])
            _test_spadd_shape(0, [0, 30, 20])
            _test_spadd_shape(0, [50, 0, 20])
            _test_spadd_shape(0, [50, 30, 0])

        def _test_spadd_hybrid():
            # [FUNCTION DOC] HELPER: Runs the sparse-dense add check for hybrid shapes including
            # zero-size dense dims.
            _test_spadd_shape(10, [5, 6], [2, 3])
            _test_spadd_shape(10, [10, 10, 10], [3])
            _test_spadd_shape(10, [50, 30, 20], [2])
            _test_spadd_shape(10, [5, 5, 5, 5, 5, 5], [2])
            _test_spadd_shape(0, [0, 30, 20], [2, 0])
            _test_spadd_shape(0, [50, 0, 20], [2, 0])
            _test_spadd_shape(0, [50, 30, 0], [2, 0])
            _test_spadd_shape(10, [50, 30, 20], [2, 0])

        _test_spadd()
        _test_spadd_hybrid()

    @coalescedonoff
    @dtypes(torch.float)
    def test_sparse_add_out_bfloat16(self, device, dtype, coalesced):
        # [FUNCTION DOC] Checks that adding two sparse float tensors converted to bfloat16 gives
        # a result that matches the float32 sum within an absolute tolerance of 0.01.
        # fp32
        x, _, _ = self._gen_sparse(3, 5, 10, dtype, device, coalesced)
        y, _, _ = self._gen_sparse(3, 5, 10, dtype, device, coalesced)
        res_fp32 = torch.add(x, y)

        # bfloat16
        x = x.bfloat16()
        y = y.bfloat16()
        res_bf16 = torch.add(x, y)
        res_bf16 = res_bf16.float()  # to compare with reference
        self.assertEqual(res_fp32, res_bf16, atol=1e-2, rtol=0)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_norm(self, device, dtype, coalesced):
        # [FUNCTION DOC] Checks vector_norm of a sparse tensor equals the norm of its coalesced
        # values for several shapes, and that unsupported arguments (keepdim, dim, and dtype
        # with ord zero) raise RuntimeError with the expected messages.
        def test_shape(sparse_dims, nnz, with_size):
            # [FUNCTION DOC] HELPER: Generates a sparse tensor and asserts vector_norm of it
            # equals vector_norm of the values of its coalesced copy.
            x, _, _ = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)
            y = x.coalesce()
            self.assertEqual(
                torch.linalg.vector_norm(x),
                torch.linalg.vector_norm(y._values()),
            )

        test_shape(3, 10, 100)
        test_shape(4, 10, [100, 100, 100, 5, 5, 5, 0])
        test_shape(4, 0, [0, 0, 100, 5, 5, 5, 0])

        # Unsupported arguments should error
        kwarg_error_pairs = [
            ({'keepdim': True},
             RuntimeError, r'norm_sparse currently does not support keepdim=True'),
            ({'dim': 0},
             RuntimeError, r'norm_sparse currently only supports full reductions'),
            ({'dtype': torch.double, 'ord': 0},
             RuntimeError, r"norm_sparse currently does not support 'dtype' argument"),
        ]
        x = self._gen_sparse(3, 10, 100, dtype, device, coalesced)[0]
        for kwargs, err, msg in kwarg_error_pairs:
            with self.assertRaisesRegex(err, msg):
                torch.linalg.vector_norm(x, **kwargs)

    @coalescedonoff
    @dtypes(torch.double)
    @dtypesIfMPS(torch.float32)
    @unittest.skipIf(TEST_WITH_CROSSREF, "fallback triggers cuda device error")
    def test_sparse_sum(self, device, dtype, coalesced):
        # [FUNCTION DOC] Tests torch.sparse.sum over all, and over every combination of dims, of
        # a hybrid sparse tensor with gradcheck. Also checks dims equivalence for negative dim,
        # IndexError for out of range dim, RuntimeError for repeated dims, summing an empty
        # tensor (forward and backward). Skipped under cross-ref.

        def run_tests(S, td=None):
            # [FUNCTION DOC] HELPER: For a sparse tensor requiring grad and optional dims,
            # compares the sparse sum with the dense sum and gradchecks the sum with
            # masked semantics.
            D = S.coalesce().to_dense().detach().requires_grad_(True)
            if td is None:
                S_sum = torch.sparse.sum(S)
                D_sum = D.sum()
                self.assertEqual(S_sum.item(), D_sum.item())

                def fn(S):
                    # [FUNCTION DOC] HELPER: Function under gradcheck returning the full
                    # sparse sum.
                    return torch.sparse.sum(S)
                gradcheck(fn, (S,), masked=True)
            else:
                S_sum = torch.sparse.sum(S, td)
                D_sum = D.sum(td)
                self.assertEqual(S_sum.to_dense() if S_sum.is_sparse else S_sum, D_sum)

                def fn(S):
                    # [FUNCTION DOC] HELPER: Function under gradcheck returning the sparse sum
                    # over the selected dims as a dense tensor with masked gradient.
                    res = torch.sparse.sum(S, td)
                    return res.to_dense(masked_grad=True)
                gradcheck(fn, (S,), masked=True)

        nnz = 10
        sparse_dims = 2
        with_size = [5, 5, 1, 4]  # use a dense dim = 1 to test for squeeze
        test_dims = []
        for i in range(1, 5):
            test_dims += itertools.combinations(range(len(with_size)), i)

        # https://github.com/pytorch/pytorch/issues/16501
        x = torch.tensor([[1., 0., 0., 1.],
                          [0., 1., 0., 0.],
                          [0., 1., 1., 0.],
                          [0., 1., 0., 2.]], dtype=dtype, device=device).to_sparse()
        self.assertEqual(torch.sparse.sum(x, dim=0), torch.sparse.sum(x, dim=-2))
        self.assertEqual(torch.sum(x.to_dense(), dim=0), torch.sparse.sum(x, dim=0).to_dense())

        S = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)[0]

        # dim out of range
        self.assertRaises(IndexError, lambda: torch.sparse.sum(S, 5))

        # dim 0 appears multiple times in the list of dims
        self.assertRaises(RuntimeError, lambda: torch.sparse.sum(S, [0, 0]))

        # sum an empty tensor
        empty_S = torch.sparse_coo_tensor(size=with_size, dtype=dtype, device=device)
        self.assertEqual(torch.sparse.sum(empty_S, [0]).to_dense(), torch.sum(empty_S.to_dense(), [0]))
        self.assertEqual(torch.sparse.sum(empty_S), torch.tensor(0, dtype=dtype, device=device))
        empty_S.requires_grad_(True)
        empty_S_sum = torch.sparse.sum(empty_S)
        empty_S_sum.backward()
        self.assertEqual(empty_S.grad.to_dense(), empty_S.detach().clone().to_dense())

        # test values().sum()
        S = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)[0]
        run_tests(S.requires_grad_(True))

        for test_dim in test_dims:
            S = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)[0]
            run_tests(S.requires_grad_(True), test_dim)

    def _test_basic_ops_shape(self, nnz_x1, nnz_x2, shape_i, shape_v, dtype, device, coalesced):
        # [FUNCTION DOC] HELPER: Generates two sparse tensors of the given nnz and shape and
        # checks add, sub, mul, scalar mul, scalar div, floor divide and pow with 2 (both
        # operator and in-place forms where supported) and zero_ against the dense results, then
        # checks that coalesce does not change the coalesced flag of the original, and is in
        # place when already coalesced and out of place otherwise.
        shape = shape_i + (shape_v)
        x1, _, _ = self._gen_sparse(len(shape_i), nnz_x1, shape, dtype, device, coalesced)
        x2, _, _ = self._gen_sparse(len(shape_i), nnz_x2, shape, dtype, device, coalesced)

        y1 = x1 + x2
        y2 = x1.clone()
        y2.add_(x2)
        expected = self.safeToDense(x1) + self.safeToDense(x2)
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        y1 = x1 - x2
        y2 = x1.clone()
        y2.sub_(x2)
        expected = self.safeToDense(x1) - self.safeToDense(x2)
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        y1 = x1 * x2
        y2 = x1.clone()
        y2.mul_(x2)
        expected = self.safeToDense(x1) * self.safeToDense(x2)
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        y1 = x1 * 37.5
        y2 = x1.clone()
        y2.mul_(37.5)
        expected = self.safeToDense(x1) * 37.5
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        y1 = x1 / 37.5
        y2 = x1.clone()
        y2.div_(37.5)
        expected = self.safeToDense(x1) / 37.5
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        y1 = x1 // 37.5
        y2 = x1.clone()
        y2.floor_divide_(37.5)
        expected = self.safeToDense(x1) // 37.5
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        # TODO: add back inplace support
        y1 = x1 ** 2
        y2 = x1.clone()
        y2 = y2.pow(2)
        expected = self.safeToDense(x1) ** 2
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        y = x1.clone()
        y.zero_()
        expected = torch.zeros(x1.size(), dtype=dtype, device=device)
        self.assertEqual(self.safeToDense(y), expected)

        self.assertEqual(x1.is_coalesced(), coalesced)
        y = x1.coalesce()
        z = x1.coalesce()
        self.assertEqual(x1.is_coalesced(), coalesced)
        self.assertTrue(y.is_coalesced())
        y._values().add_(1)
        if not x1.is_coalesced():
            # check that coalesce is out of place if the original tensor is not
            # coalesced.
            self.assertEqual(z._values() + 1, y._values())
        else:
            # check that coalesce is in-place if the original tensor is
            # coalesced.
            self.assertEqual(z._values(), y._values())

    @coalescedonoff
    @dtypes(torch.double)
    @dtypesIfMPS(torch.float32)
    def test_basic_ops(self, device, dtype, coalesced):
        # [FUNCTION DOC] Runs the basic elementwise operation checks for a range of non-hybrid
        # and hybrid shapes, including empty and zero-size cases.

        def _test_basic_ops():
            # [FUNCTION DOC] HELPER: Runs the shared elementwise operation check for non-
            # hybrid shapes.
            self._test_basic_ops_shape(9, 12, [5, 6], [], dtype, device, coalesced)
            self._test_basic_ops_shape(9, 12, [10, 10, 10], [], dtype, device, coalesced)
            self._test_basic_ops_shape(9, 12, [50, 30, 20], [], dtype, device, coalesced)
            self._test_basic_ops_shape(9, 12, [5, 5, 5, 5, 5, 5], [], dtype, device, coalesced)
            self._test_basic_ops_shape(0, 12, [10, 10, 10], [], dtype, device, coalesced)
            self._test_basic_ops_shape(9, 0, [10, 10, 10], [], dtype, device, coalesced)
            self._test_basic_ops_shape(0, 0, [10, 10, 10], [], dtype, device, coalesced)
            self._test_basic_ops_shape(0, 0, [10, 10, 0], [], dtype, device, coalesced)
            self._test_basic_ops_shape(0, 0, [], [], dtype, device, coalesced)

        def _test_basic_ops_hybrid():
            # [FUNCTION DOC] HELPER: Runs the shared elementwise operation check for
            # hybrid shapes.
            self._test_basic_ops_shape(9, 12, [5, 6], [2, 3], dtype, device, coalesced)
            self._test_basic_ops_shape(9, 12, [10, 10, 10], [3], dtype, device, coalesced)
            self._test_basic_ops_shape(9, 12, [50, 30, 20], [2], dtype, device, coalesced)
            self._test_basic_ops_shape(9, 12, [5, 5, 5, 5, 5, 5], [2], dtype, device, coalesced)
            self._test_basic_ops_shape(0, 12, [10, 10, 10], [2], dtype, device, coalesced)
            self._test_basic_ops_shape(9, 0, [10, 10, 10], [2], dtype, device, coalesced)
            self._test_basic_ops_shape(0, 0, [10, 10, 10], [2], dtype, device, coalesced)
            self._test_basic_ops_shape(9, 12, [10, 10, 10], [2, 0], dtype, device, coalesced)
            self._test_basic_ops_shape(0, 12, [10, 10, 10], [2, 0], dtype, device, coalesced)
            self._test_basic_ops_shape(9, 0, [10, 10, 10], [2, 0], dtype, device, coalesced)
            self._test_basic_ops_shape(0, 0, [10, 10, 10], [2, 0], dtype, device, coalesced)
            self._test_basic_ops_shape(0, 0, [10, 10, 0], [2, 0], dtype, device, coalesced)

        _test_basic_ops()
        _test_basic_ops_hybrid()

    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_add_dense_sparse_mismatch(self, device, dtype):
        # [FUNCTION DOC] Checks that adding a dense tensor and a sparse tensor of mismatched
        # size raises RuntimeError with the message that self and other must have the same size.
        def test_shape(dense_size, sparse_dims_shape, dense_dims_shape, sparse_size):
            # [FUNCTION DOC] HELPER: Builds a dense tensor and a sparse tensor from the
            # given shapes and asserts that adding them raises RuntimeError with the size-
            # mismatch message.
            x = torch.zeros(dense_size, dtype=dtype, device=device)
            sparse_y = self.sparse_tensor(torch.zeros(sparse_dims_shape, dtype=torch.int64, device=device),
                                          torch.randn(dense_dims_shape, dtype=dtype, device=device),
                                          torch.Size(sparse_size))
            with self.assertRaisesRegex(
                    RuntimeError,
                    "add: expected 'self' and 'other' to have same size"):
                x + sparse_y

        test_shape([3, 4], [1, 4], [4, 4, 4], [3, 4, 4])
        test_shape([3, 4, 0], [1, 4], [4, 4, 4, 0], [3, 4, 4, 0])

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_add_noncontiguous(self, device, dtype):
        # [FUNCTION DOC] Checks addition of a sparse tensor whose values are a non-contiguous
        # expanded view (asserted non-contiguous) to itself equals the dense sum. Skipped
        # under TorchDynamo.
        indices = self.index_tensor([[1, 2], [0, 2]], device=device)
        values = torch.tensor([1.], dtype=dtype, device=device).expand(2, 3, 4, 5)
        x = self.sparse_tensor(indices, values, dtype=dtype, device=device)
        if x._values().is_contiguous():
            raise AssertionError("expected values to be non-contiguous")
        y = x + x
        expected = self.safeToDense(x) + self.safeToDense(x)
        self.assertEqual(self.safeToDense(y), expected)

    def _test_sparse_mask_shape(self, nnz_x1, nnz_x2, shape_i, shape_v, dtype, device, coalesced):
        # [FUNCTION DOC] HELPER: Generates two sparse tensors and asserts the out-of-place and
        # in-place add match the dense sum.
        # KNOWN ISSUE: despite the name it never exercises sparse_mask; only addition
        # is verified.
        shape = shape_i + (shape_v or [])
        x1, _, _ = self._gen_sparse(len(shape_i), nnz_x1, shape, dtype, device, coalesced)
        x2, _, _ = self._gen_sparse(len(shape_i), nnz_x2, shape, dtype, device, coalesced)

        y1 = x1 + x2
        y2 = x1.clone()
        y2.add_(x2)
        expected = self.safeToDense(x1) + self.safeToDense(x2)
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_sparse_mask(self, device, dtype, coalesced):
        # [FUNCTION DOC] Tests sparse_mask with fixed dense and sparse sources (including a
        # zero-size dense dim) and the coalesced flag side effects, then sweeps several shapes
        # using the shared mask shape helper, and further repetition checks later in the body.
        def _test_sparse_mask_fixed():
            # [FUNCTION DOC] HELPER: Builds a coalesced sparse mask, applies sparse_mask from a
            # dense tensor and from its sparse form, compares to the expected sparse values (7,
            # 14, 3, 20 pattern) after coalescing, asserts the source remains coalesced, and
            # repeats with an empty trailing dimension.
            i = self.index_tensor([
                [1, 3, 0, 4],
                [2, 1, 2, 3],
            ], device=device)
            v = torch.tensor([1, 2, 3, 4], dtype=dtype, device=device)
            x = self.sparse_tensor(i, v, torch.Size([5, 4]), dtype=dtype, device=device).coalesce()
            dense = torch.tensor([
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
                [17, 18, 19, 20],
            ], dtype=dtype, device=device)
            exp_v = torch.tensor([7, 14, 3, 20], dtype=dtype, device=device)
            res_dense_lhs = dense.sparse_mask(x)
            sparse = dense.to_sparse()
            res_sparse_lhs = sparse.sparse_mask(x)
            expected = self.sparse_tensor(i, exp_v, torch.Size([5, 4]), dtype=dtype, device=device)
            self.assertEqual(res_dense_lhs.coalesce(), expected.coalesce())
            # check no side effects for the coalesce flag.
            self.assertTrue(sparse.is_coalesced())
            self.assertEqual(res_sparse_lhs.coalesce(), expected.coalesce())

            i = self.index_tensor([
                [1, 3, 0, 4],
                [2, 1, 2, 3],
            ], device=device)
            v = torch.empty([4, 0], dtype=dtype, device=device)
            x = self.sparse_tensor(i, v, torch.Size([5, 4, 0])).coalesce()
            dense = torch.empty([5, 4, 0], dtype=dtype, device=device)
            exp_v = torch.empty([4, 0], dtype=dtype, device=device)
            res_dense_lhs = dense.sparse_mask(x)
            sparse = dense.to_sparse(2)
            res_sparse_lhs = sparse.sparse_mask(x)
            expected = self.sparse_tensor(i, exp_v, torch.Size([5, 4, 0]), dtype=dtype, device=device)
            self.assertEqual(res_dense_lhs.coalesce(), expected.coalesce())
            # check no side effects for the coalesce flag.
            self.assertTrue(sparse.is_coalesced())
            self.assertEqual(res_sparse_lhs.coalesce(), expected.coalesce())

        _test_sparse_mask_fixed()

        self._test_sparse_mask_shape(9, 12, [5, 6], [], dtype, device, coalesced)
        self._test_sparse_mask_shape(9, 12, [10, 10, 10], [], dtype, device, coalesced)
        self._test_sparse_mask_shape(9, 12, [50, 30, 20], [], dtype, device, coalesced)
        self._test_sparse_mask_shape(9, 12, [5, 5, 5, 5, 5, 5], [], dtype, device, coalesced)
        self._test_sparse_mask_shape(0, 12, [10, 10, 10], [], dtype, device, coalesced)
        self._test_sparse_mask_shape(9, 0, [10, 10, 10], [], dtype, device, coalesced)
        self._test_sparse_mask_shape(0, 0, [10, 10, 10], [], dtype, device, coalesced)
        self._test_sparse_mask_shape(0, 0, [10, 10, 0], [], dtype, device, coalesced)

        # check repetitions and matchings in the intersection
        lhs = torch.randint(0, 5, (100,), device=device)
        rhs = torch.randint(0, 5, (100,), device=device).to_sparse()
        self.assertEqual(lhs.to_sparse().sparse_mask(rhs), lhs.sparse_mask(rhs))

        # check coalesce
        sparse_c = torch.rand(3, 3, device=device).to_sparse()
        sparse_unc = torch.rand(3, 3, device=device).to_sparse()._coalesced_(False)
        for lhs, rhs in [(sparse_c, sparse_unc), (sparse_unc, sparse_c)]:
            res_all_sparse = lhs.sparse_mask(rhs)
            res_dense_sparse = lhs.to_dense().sparse_mask(rhs)
            self.assertEqual(res_all_sparse.coalesce(), res_dense_sparse.coalesce())
            self.assertEqual(rhs.is_coalesced(), res_all_sparse.is_coalesced())

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_sparse_mask_hybrid(self, device, dtype, coalesced):
        # [FUNCTION DOC] Hybrid variant of the sparse_mask test: fixed dense and sparse sources
        # with a dense trailing dimension (and a zero-size one) checked against expected values,
        # then the shared mask shape helper over hybrid shapes with empty cases.
        def _test_sparse_mask_hybrid_fixed():
            # [FUNCTION DOC] HELPER: Builds a coalesced hybrid mask, applies sparse_mask from a
            # dense and from a sparse source, compares to the expected hybrid values after
            # coalescing and asserts the sparse source remains coalesced; repeats with a zero-
            # size trailing dim.
            i = self.index_tensor([
                [1, 3, 0, 4],
                [2, 1, 2, 3],
            ])
            v = torch.tensor([[1, 2], [2, 3], [3, 4], [4, 5]])
            # TODO: This is also testing that, if coalesce is a no-op,
            # the indices don't get permuted. I don't know if we actually
            # want to give this invariant.
            x = self.sparse_tensor(i, v, torch.Size([5, 4, 2])).coalesce()
            dense = torch.tensor([
                [[1, 3], [2, 2], [3, 3], [4, 2]],
                [[5, 7], [6, 7], [7, 9], [8, 9]],
                [[9, 2], [10, 4], [11, 1], [12, 3]],
                [[13, 5], [14, 1], [15, 1], [16, 6]],
                [[17, 7], [18, 2], [19, 7], [20, 1]],
            ])
            res_dense_lhs = dense.sparse_mask(x)
            sparse = dense.to_sparse(2)
            res_sparse_lhs = sparse.sparse_mask(x)
            exp_v = torch.tensor([[7, 9], [14, 1], [3, 3], [20, 1]])
            expected = self.sparse_tensor(i, exp_v, torch.Size([5, 4, 2]))
            self.assertEqual(res_dense_lhs.coalesce(), expected.coalesce())
            # check no side effects for the coalesce flag
            self.assertTrue(sparse.is_coalesced())
            self.assertEqual(res_sparse_lhs.coalesce(), expected.coalesce())

            i = self.index_tensor([
                [1, 3, 0, 4],
                [2, 1, 2, 3],
            ])
            v = torch.empty(4, 2, 0)
            x = self.sparse_tensor(i, v, torch.Size([5, 4, 2, 0])).coalesce()
            dense = torch.empty(5, 4, 2, 0)
            res_dense_lhs = dense.sparse_mask(x)
            sparse = dense.to_sparse(2)
            res_sparse_lhs = sparse.sparse_mask(x)
            exp_v = torch.empty(4, 2, 0)
            expected = self.sparse_tensor(i, exp_v, torch.Size([5, 4, 2, 0]))
            self.assertEqual(res_dense_lhs.coalesce(), expected.coalesce())
            # check no side effects for the coalesce flag
            self.assertTrue(sparse.is_coalesced())
            self.assertEqual(res_sparse_lhs.coalesce(), expected.coalesce())

        _test_sparse_mask_hybrid_fixed()

        self._test_sparse_mask_shape(9, 12, [5, 6], [2, 3], dtype, device, coalesced)
        self._test_sparse_mask_shape(9, 12, [10, 10, 10], [3], dtype, device, coalesced)
        self._test_sparse_mask_shape(9, 12, [50, 30, 20], [2], dtype, device, coalesced)
        self._test_sparse_mask_shape(9, 12, [5, 5, 5, 5, 5, 5], [2], dtype, device, coalesced)
        self._test_sparse_mask_shape(0, 12, [10, 10, 10], [2], dtype, device, coalesced)
        self._test_sparse_mask_shape(9, 0, [10, 10, 10], [2], dtype, device, coalesced)
        self._test_sparse_mask_shape(0, 0, [10, 10, 10], [2], dtype, device, coalesced)
        self._test_sparse_mask_shape(9, 12, [10, 10, 10], [2, 0], dtype, device, coalesced)
        self._test_sparse_mask_shape(0, 12, [10, 10, 10], [2, 0], dtype, device, coalesced)
        self._test_sparse_mask_shape(9, 0, [10, 10, 10], [2, 0], dtype, device, coalesced)
        self._test_sparse_mask_shape(0, 0, [10, 10, 10], [2, 0], dtype, device, coalesced)
        self._test_sparse_mask_shape(0, 0, [10, 10, 0], [2, 0], dtype, device, coalesced)

    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    @skipIfCrossRef
    def test_sparse_mask_backward(self, device, dtype):
        # [FUNCTION DOC] Gradchecks sparse_mask backward for nnz values 0, 5, 15, 25 and all
        # four combinations of coalesced and uncoalesced lhs and rhs, with masked semantics
        # (lhs masked by its own pattern first, then rhs) and unmasked semantics. Skipped
        # under cross-ref.
        from itertools import product, repeat

        shape = (5, 5)
        sparse_dims = len(shape)
        nnzs = (0, 5, 15, 25)

        lhs_data = torch.arange(1, 26, device=device).reshape(shape).to(dtype).to_sparse(sparse_dims)
        for nnz in nnzs:
            for lhs_is_coalesced, rhs_is_coalesced in product(*repeat((True, False), 2)):
                lhs = torch.sparse_coo_tensor(
                    lhs_data._indices()[:, :nnz],
                    lhs_data._values()[:nnz],
                    lhs_data.shape
                ).clone()._coalesced_(lhs_is_coalesced).requires_grad_(True)

                rhs = torch.sparse_coo_tensor(
                    lhs_data._indices()[:, -nnz:],
                    lhs_data._values()[-nnz:],
                    lhs_data.shape
                ).clone()._coalesced_(rhs_is_coalesced)

                # To test masked semantics we need to make sure that
                # sparsity_pattern(lhs) == sparsity_pattern(lhs.grad).
                # lhs.sparse_mask(lhs_mask) accomplishes that.
                lhs_mask = lhs.detach().clone()

                def op_masked(x):
                    # [FUNCTION DOC] HELPER: Function under masked gradcheck. Moves the mask and
                    # rhs to the input device if needed, applies sparse_mask with both, and
                    # returns the dense result with masked gradient.
                    m, r = lhs_mask, rhs
                    if x.device != m.device:
                        m = m.to(device=x.device)
                        r = r.to(device=x.device)
                    return x.sparse_mask(m).sparse_mask(r).to_dense(masked_grad=True)

                gradcheck(
                    op_masked,
                    (lhs,),
                    masked=True
                )

                def op_unmasked(x):
                    # [FUNCTION DOC] HELPER: Function under unmasked gradcheck. Moves rhs to the
                    # input device if needed, applies sparse_mask and returns the dense result
                    # with unmasked gradient.
                    r = rhs
                    if x.device != r.device:
                        r = r.to(device=x.device)
                    return x.sparse_mask(r).to_dense(masked_grad=False)

                gradcheck(
                    op_unmasked,
                    (lhs, ),
                    masked=False
                )

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_zeros(self, device, dtype, coalesced):
        # [FUNCTION DOC] Tests torch.zeros with an out argument that is a sparse tensor: the out
        # tensor must be resized to the requested shape and left with empty indices and values,
        # for non-hybrid and hybrid inputs.
        def _test_zeros(nnzs, shape, out_shape_i, out_shape_v=None):
            # [FUNCTION DOC] HELPER: For each nnz, generates a sparse out tensor of the given
            # sparse and dense shapes, calls torch.zeros with the requested shape writing into
            # it, and asserts the size equals the shape, indices and values are empty, nnz is
            # zero, sparse_dim equals the number of requested dims and dense_dim is zero.
            out_shape = out_shape_i + (out_shape_v or [])
            for nnz in nnzs:
                out, _, _ = self._gen_sparse(len(out_shape_i), nnz, out_shape, dtype, device, coalesced)
                torch.zeros(*shape, out=out, dtype=dtype, device=device)
                self.assertEqual(tuple(out.size()), tuple(shape))
                self.assertTrue(out._indices().numel() == out._values().numel() == 0)
                self.assertEqual(out._nnz(), 0)
                self.assertEqual(out.sparse_dim(), len(shape))
                self.assertEqual(out.dense_dim(), 0)

        def test_shape(i_shapes, v_shapes, shape, nnzs):
            # [FUNCTION DOC] HELPER: Loops over every prefix of the sparse shape list (at least
            # one entry) and every prefix of the dense shape list (possibly empty) and runs the
            # zeros-out check for each combination.
            for i_dim in range(1, len(i_shapes) + 1):
                for v_dim in range(len(v_shapes) + 1):
                    _test_zeros(nnzs, shape, i_shapes[:i_dim], v_shapes[:v_dim])
        test_shape([2, 3, 4], [3, 4, 5, 6], [2, 3, 4], [9, 12])
        test_shape([0, 3, 4], [3, 4, 5, 6], [2, 3, 4], [0])
        test_shape([2, 3, 4], [0, 4, 5, 6], [2, 3, 4], [9, 12])
        test_shape([2, 3, 4], [3, 4, 5, 6], [2, 3, 0], [9, 12])
        test_shape([0, 3, 4], [3, 4, 5, 6], [2, 3, 0], [0])
        test_shape([2, 3, 4], [0, 4, 5, 6], [2, 3, 0], [9, 12])

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_zeros_like(self, device, dtype, coalesced):
        # [FUNCTION DOC] Tests zeros_like on sparse tensors: size, empty indices and values,
        # zero nnz, and preserved sparse and dense dimension counts over many shape
        # combinations; asserts RuntimeError when a memory format is given for a sparse input
        # but success when layout strided is also given; and that zeros_like of a dense tensor
        # with sparse COO layout gives the same shape and empty index and value shapes as a
        # fresh empty sparse tensor.
        def _test_zeros_like(nnzs, template_shape_i, template_shape_v=None):
            # [FUNCTION DOC] HELPER: Generates a sparse template per nnz, calls zeros_like and
            # asserts size, empty indices and values, zero nnz and that sparse_dim and dense_dim
            # equal those of the template.
            template_shape_v = template_shape_v or []
            template_shape = template_shape_i + template_shape_v
            for nnz in nnzs:
                t, _, _ = self._gen_sparse(len(template_shape_i), nnz, template_shape, dtype, device, coalesced)
                res = torch.zeros_like(t)
                self.assertEqual(tuple(res.size()), tuple(template_shape))
                self.assertTrue(res._indices().numel() == res._values().numel() == 0)
                self.assertEqual(res._nnz(), 0)
                self.assertEqual(res.sparse_dim(), len(template_shape_i))
                self.assertEqual(res.dense_dim(), len(template_shape_v))

        def test_shape(i_shapes, v_shapes, nnzs):
            # [FUNCTION DOC] HELPER: Loops over every prefix of the sparse and dense shape lists
            # and runs the zeros_like check for each combination.
            for i_dim in range(1, len(i_shapes) + 1):
                for v_dim in range(len(v_shapes) + 1):
                    _test_zeros_like(nnzs, i_shapes[:i_dim], v_shapes[:v_dim])
        test_shape([2, 3, 4], [3, 4, 5, 6], [9, 12])
        test_shape([0, 3, 4], [3, 4, 5, 6], [0])
        test_shape([2, 3, 4], [0, 4, 5, 6], [9, 12])
        test_shape([2, 3, 4], [3, 4, 5, 6], [9, 12])
        test_shape([0, 3, 4], [3, 4, 5, 6], [0])
        test_shape([2, 3, 4], [0, 4, 5, 6], [9, 12])

        sparse_tensor, _, _ = self._gen_sparse(len([2, 3]), 9, [2, 3] + [5, 6], dtype, device, coalesced)
        data = (sparse_tensor, sparse_tensor, sparse_tensor, sparse_tensor.unsqueeze(0))
        mem_formats = [torch.channels_last, torch.contiguous_format, torch.preserve_format, torch.channels_last_3d]
        for x, mem_format in zip(data, mem_formats):

            with self.assertRaisesRegex(RuntimeError, "memory format option is only supported by strided tensors"):
                result = torch.zeros_like(x, memory_format=mem_format)

            result = torch.zeros_like(x, layout=torch.strided, memory_format=mem_format)
            self.assertTrue(result.layout == torch.strided)

        dense_tensor = sparse_tensor.to_dense()
        result = torch.zeros_like(dense_tensor, layout=torch.sparse_coo)
        self.assertEqual(dense_tensor.shape, result.shape)
        self.assertEqual(result.layout, torch.sparse_coo)

        sparse_zeros = torch.sparse_coo_tensor(dense_tensor.shape)
        self.assertEqual(result._indices().shape, sparse_zeros._indices().shape)
        self.assertEqual(result._values().shape, sparse_zeros._values().shape)

    def _assert_sparse_invars(self, t):
        # [FUNCTION DOC] HELPER: Asserts the invariants of a COO tensor: sparse_dim plus
        # dense_dim equals the number of dimensions, indices have shape sparse_dim by nnz, and
        # values have shape nnz followed by the dense part of the tensor shape.
        # SparseTensor has the following invariants:
        # - sparse_dim + dense_dim = len(SparseTensor.shape)
        # - SparseTensor._indices().shape = (sparse_dim, nnz)
        # - SparseTensor._values().shape = (nnz, SparseTensor.shape[sparse_dim:])
        self.assertEqual(t.sparse_dim() + t.dense_dim(), len(t.shape))
        self.assertEqual(tuple(t._indices().shape), (t.sparse_dim(), t._nnz()))
        self.assertEqual(tuple(t._values().shape), (t._nnz(), ) + t.shape[t.sparse_dim():])

    def _test_empty_like(self, sparse_tensor, dtype, device, coalesced):
        # [FUNCTION DOC] HELPER: Checks empty_like on a sparse tensor: result is sparse,
        # satisfies the COO invariants and has the same shape, dtype, device, sparse_dim and
        # dense_dim. Then for each of four memory formats (channels_last, contiguous, preserve,
        # channels_last_3d, the last paired with an unsqueezed input) asserts RuntimeError when
        # only a memory format is given and a strided result when layout strided is also passed.
        # Finally asserts that empty_like of a dense tensor with sparse COO layout raises
        # RuntimeError about the missing empty_strided kernel for a Sparse backend.

        result = torch.empty_like(sparse_tensor)
        self.assertTrue(result.is_sparse)
        self._assert_sparse_invars(result)
        self.assertEqual(result.shape, sparse_tensor.shape)
        self.assertEqual(result.dtype, sparse_tensor.dtype)
        self.assertEqual(result.device, sparse_tensor.device)
        self.assertEqual(result.sparse_dim(), sparse_tensor.sparse_dim())
        self.assertEqual(result.dense_dim(), sparse_tensor.dense_dim())

        sparse_tensor, _, _ = self._gen_sparse(len([2, 3]), 9, [2, 3] + [5, 6], dtype, device, coalesced)
        data = (sparse_tensor, sparse_tensor, sparse_tensor, sparse_tensor.unsqueeze(0))
        mem_formats = [torch.channels_last, torch.contiguous_format, torch.preserve_format, torch.channels_last_3d]
        for x, mem_format in zip(data, mem_formats):

            with self.assertRaisesRegex(RuntimeError, "memory format option is only supported by strided tensors"):
                result = torch.empty_like(x, memory_format=mem_format)

            result = torch.empty_like(x, layout=torch.strided, memory_format=mem_format)
            self.assertTrue(result.layout == torch.strided)

        with self.assertRaisesRegex(
            RuntimeError, r"Could not run 'aten::empty_strided' with arguments from the 'Sparse(CPU|CUDA|MPS)' backend"
        ):
            dense_tensor = sparse_tensor.to_dense()
            result = torch.empty_like(dense_tensor, layout=torch.sparse_coo)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_empty_like(self, device, dtype, coalesced):
        # [FUNCTION DOC] Regression test for issue 43699: runs the empty_like checks on a
        # coalesced 1-D tensor and a coalesced hybrid tensor when coalesced, and on an
        # uncoalesced tensor with duplicate indices and on an empty tensor with zero-size dims
        # when not coalesced.
        # tests https://github.com/pytorch/pytorch/issues/43699

        if coalesced:
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0, 1, 2]]),
                values=torch.tensor([3.0, -4.0, 5.0]),
                size=[3, ],
                dtype=dtype,
                device=device
            ).coalesce()
            self._test_empty_like(input_coalesced, dtype, device, coalesced)

            # hybrid sparse input
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[1, 3], [2, 4]]),
                values=torch.tensor([[-1.0, 3.0], [-5.0, 7.0]]),
                size=[4, 5, 2],
                dtype=dtype,
                device=device
            ).coalesce()
            self._test_empty_like(input_coalesced, dtype, device, coalesced)

        if not coalesced:
            # test uncoalesced input
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0], [1], [2], [0], [1], [2]]).transpose(1, 0),
                values=torch.tensor([2.0, -3.0, -4.0, 1.0, -1.0, 1.5]),
                size=[3, ],
                dtype=dtype,
                device=device
            )
            self._test_empty_like(input_uncoalesced, dtype, device, coalesced)

            # test on empty sparse tensor
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.zeros([2, 0]),
                values=torch.zeros([0, 5, 5, 5, 5, 5, 5, 0]),
                size=[0, 0, 5, 5, 5, 5, 5, 5, 0],
                dtype=dtype,
                device=device
            )
            self._test_empty_like(input_uncoalesced, dtype, device, coalesced)

    def _test_narrow(self, input, narrow_args):
        # [FUNCTION DOC] HELPER: Asserts narrow_copy on a sparse tensor equals dense narrow of
        # its dense version for the given dim, start and length.
        expected = input.to_dense().narrow(*narrow_args)
        self.assertEqual(expected, input.narrow_copy(*narrow_args).to_dense())

    def _all_narrow_combs(self, shape):
        # [FUNCTION DOC] HELPER: Generator yielding every valid (dim, start, length) triple for
        # the shape, where start ranges below the dim size and length ranges so that start plus
        # length stays below the dim size (strictly, so full-extent narrows are excluded).
        for dim, dim_sz in enumerate(shape):
            for start in range(dim_sz):
                for length in range(dim_sz - start):
                    yield [dim, start, length]

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_narrow(self, device, dtype, coalesced):
        # [FUNCTION DOC] Tests narrow_copy over all valid combinations on a 4-D sparse tensor
        # with all-sparse dims and on one with two dense dims, and checks RuntimeError for a
        # negative dim, dim beyond the number of dims, start beyond the size and start plus
        # length beyond the size.
        shape = [3, 3, 4, 2]
        input, _, _ = self._gen_sparse(4, 19, shape, dtype, device, coalesced)
        for narrow_args in self._all_narrow_combs(shape):
            self._test_narrow(input, narrow_args)

        self.assertRaises(RuntimeError, lambda: input.narrow_copy(-1, 0, 3))  # dim < 0
        self.assertRaises(RuntimeError, lambda: input.narrow_copy(10, 0, 3))  # dim > input.dim()
        self.assertRaises(RuntimeError, lambda: input.narrow_copy(0, shape[0] + 1, 3))  # start > size of dim
        self.assertRaises(RuntimeError, lambda: input.narrow_copy(0, 2, shape[0]))  # start+length > size of dim

        with_dense, _, _ = self._gen_sparse(2, 7, shape, dtype, device, coalesced)
        for narrow_args in self._all_narrow_combs(shape):
            self._test_narrow(with_dense, narrow_args)

        self.assertRaises(RuntimeError, lambda: with_dense.narrow_copy(10, 0, 3))  # dim > sparseDim + denseDim

    def _test_log1p_tensor(self, sparse_tensor, coalesced):
        # [FUNCTION DOC] HELPER: Checks log1p on a sparse tensor: out of place result equals
        # dense log1p; in-place on a coalesced copy equals it for floating types and raises
        # RuntimeError about result type cast for integral types; in-place on an uncoalesced
        # input raises RuntimeError requiring coalesced input; for integral dtype requires_grad_
        # raises RuntimeError.
        def is_integral(dtype):
            # [FUNCTION DOC] HELPER: Returns True if the dtype is one of the integral types.
            return dtype in integral_types()

        dense_tensor = sparse_tensor.to_dense()
        expected_output = dense_tensor.log1p()
        is_integral_dtype = is_integral(sparse_tensor.dtype)
        self.assertEqual(expected_output, sparse_tensor.log1p().to_dense())
        if is_integral_dtype:
            with self.assertRaisesRegex(RuntimeError, "result type .* can't be cast to"):
                sparse_tensor.coalesce().log1p_()
        else:
            self.assertEqual(expected_output, sparse_tensor.coalesce().log1p_().to_dense())

        if not coalesced:
            # test in-place op on uncoalesced input
            with self.assertRaisesRegex(RuntimeError, "log1p_ requires coalesced input"):
                sparse_tensor.log1p_()

        if is_integral_dtype:
            with self.assertRaisesRegex(RuntimeError, "only Tensors of floating point dtype can require gradients"):
                sparse_tensor.requires_grad_()

    @coalescedonoff
    @dtypesIfMPS(*all_mps_types())
    @dtypes(*all_types())
    def test_log1p(self, device, dtype, coalesced):
        # [FUNCTION DOC] Tests log1p over all dtypes with a coalesced 1-D, a hybrid, and (when
        # uncoalesced) an uncoalesced duplicate-index tensor and an empty tensor forced to the
        # uncoalesced state, delegating checks to the log1p helper.
        if coalesced:
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0], [1], [2]]).transpose(1, 0),
                values=torch.tensor([3.0, 4.0, 5.0]),
                size=[3, ],
                device=device,
                dtype=dtype
            ).coalesce()
            self._test_log1p_tensor(input_coalesced, coalesced)

            # hybrid sparse input
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[1, 3], [2, 4]]),
                values=torch.tensor([[1.0, 3.0], [5.0, 7.0]]),
                size=[4, 5, 2],
                device=device,
                dtype=dtype
            ).coalesce()
            self._test_log1p_tensor(input_coalesced, coalesced)

        if not coalesced:
            # test uncoalesced input
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0], [1], [2], [0], [1], [2]]).transpose(1, 0),
                values=torch.tensor([2.0, 3.0, 4.0, 1.0, 1.0, 1.0]),
                size=[3, ],
                device=device,
                dtype=dtype
            )
            self._test_log1p_tensor(input_uncoalesced, coalesced)

            # test on empty sparse tensor
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.zeros([2, 0]),
                values=torch.zeros([0, 5, 5, 5, 5, 5, 5, 0]),
                size=[0, 0, 5, 5, 5, 5, 5, 5, 0],
                device=device,
                dtype=dtype
            )
            # empty tensors are coalesced at creation (nnz < 2) we must force the uncoalesced state
            input_uncoalesced._coalesced_(False)
            self._test_log1p_tensor(input_uncoalesced, coalesced)

    def _test_neg_negative(self, sparse_tensor):
        # [FUNCTION DOC] HELPER: Checks every spelling of negation (neg and negative as
        # function, method, in-place method, and the unary minus operator) on a clone of the
        # sparse tensor against the dense neg, and the out variant of neg and negative into a
        # zeros_like tensor.
        dense_tensor = sparse_tensor.to_dense()
        expected_output = dense_tensor.neg()

        ops = (
            torch.neg, torch.Tensor.neg, torch.Tensor.neg_,
            torch.negative, torch.Tensor.negative, torch.Tensor.negative_,
            operator.neg
        )
        for op in ops:
            sparse_tensor_copy = sparse_tensor.clone()
            self.assertEqual(expected_output, op(sparse_tensor_copy).to_dense())

            if op in (torch.neg, torch.negative):
                sparse_tensor_out = torch.zeros_like(sparse_tensor)
                op(sparse_tensor, out=sparse_tensor_out)
                self.assertEqual(expected_output, sparse_tensor_out.to_dense())

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_neg_negative(self, device, dtype, coalesced):
        # [FUNCTION DOC] Tests neg and negative with coalesced 1-D and hybrid inputs, an
        # uncoalesced duplicate-index tensor and an empty tensor, via the negation helper.

        if coalesced:
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0, 1, 2]]),
                values=torch.tensor([3.0, -4.0, 5.0]),
                size=[3, ],
                dtype=dtype,
                device=device
            ).coalesce()
            self._test_neg_negative(input_coalesced)

            # hybrid sparse input
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[1, 3], [2, 4]]),
                values=torch.tensor([[-1.0, 3.0], [-5.0, 7.0]]),
                size=[4, 5, 2],
                dtype=dtype,
                device=device
            ).coalesce()
            self._test_neg_negative(input_coalesced)

        if not coalesced:
            # test uncoalesced input
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0], [1], [2], [0], [1], [2]]).transpose(1, 0),
                values=torch.tensor([2.0, -3.0, -4.0, 1.0, -1.0, 1.5]),
                size=[3, ],
                dtype=dtype,
                device=device
            )
            self._test_neg_negative(input_uncoalesced)

            # test on empty sparse tensor
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.zeros([2, 0]),
                values=torch.zeros([0, 5, 5, 5, 5, 5, 5, 0]),
                size=[0, 0, 5, 5, 5, 5, 5, 5, 0],
                dtype=dtype,
                device=device
            )
            self._test_neg_negative(input_uncoalesced)

    def _test_asin_arcsin(self, sparse_tensor, coalesced):
        # [FUNCTION DOC] HELPER: Checks asin and arcsin (function and method forms, plus out
        # variants and in-place variants) against dense asin. For integral dtypes the out and
        # in-place coalesced forms must raise RuntimeError about result type casting; in-place
        # on an uncoalesced input must raise RuntimeError requiring coalesced input.
        def is_integral(dtype):
            # [FUNCTION DOC] HELPER: Returns True if the dtype is one of the integral types.
            return dtype in integral_types()
        is_integral_dtype = is_integral(sparse_tensor.dtype)

        dense_tensor = sparse_tensor.to_dense()
        expected_output = dense_tensor.asin()

        ops = (
            torch.asin, torch.Tensor.asin,
            torch.arcsin, torch.Tensor.arcsin,
        )
        for op in ops:
            self.assertEqual(expected_output, op(sparse_tensor).to_dense())
            if op in (torch.asin, torch.arcsin):
                sparse_tensor_out = torch.zeros_like(sparse_tensor)
                if not is_integral_dtype:
                    op(sparse_tensor, out=sparse_tensor_out)
                    self.assertEqual(expected_output, sparse_tensor_out.to_dense())
                else:
                    with self.assertRaisesRegex(RuntimeError, "result type .* can't be cast to"):
                        op(sparse_tensor, out=sparse_tensor_out)

        for op in (torch.Tensor.asin_, torch.Tensor.arcsin_):
            if is_integral_dtype:
                # test coalesce on integral dtype tensor
                with self.assertRaisesRegex(RuntimeError, "result type .* can't be cast to"):
                    op(sparse_tensor.clone().coalesce()).to_dense()
            else:
                self.assertEqual(expected_output, op(sparse_tensor.clone().coalesce()).to_dense())

            if not coalesced:
                # test in-place op on uncoalesced input
                with self.assertRaisesRegex(RuntimeError, "asin_ requires coalesced input"):
                    op(sparse_tensor)

    @coalescedonoff
    @dtypes(*all_types())
    @dtypesIfMPS(*all_mps_types())
    def test_asin_arcsin(self, device, dtype, coalesced):
        # [FUNCTION DOC] Tests asin and arcsin with values in the valid domain over all dtypes
        # using coalesced 1-D and hybrid inputs, uncoalesced duplicate-index input and an empty
        # tensor forced uncoalesced, via the asin helper.
        if coalesced:
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0, 1, 2, 3]]),
                values=torch.tensor([0.5, -0.5, 0.7, -0.7]),
                size=[4, ],
                dtype=dtype,
                device=device
            ).coalesce()
            self._test_asin_arcsin(input_coalesced, coalesced)

            # hybrid sparse input
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[1, 3], [2, 4]]),
                values=torch.tensor([[-0.1, 0.24], [-0.44, 0.1]]),
                size=[4, 5, 2],
                dtype=dtype,
                device=device
            ).coalesce()
            self._test_asin_arcsin(input_coalesced, coalesced)

        if not coalesced:
            # test uncoalesced input
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0], [1], [2], [0], [1], [2]]).transpose(1, 0),
                values=torch.tensor([0.3, -0.3, -0.4, 0.3, -0.5, 0.15]),
                size=[3, ],
                dtype=dtype,
                device=device
            )
            self._test_asin_arcsin(input_uncoalesced, coalesced)

            # test on empty sparse tensor
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.zeros([2, 0]),
                values=torch.zeros([0, 5, 5, 5, 5, 5, 5, 0]),
                size=[0, 0, 5, 5, 5, 5, 5, 5, 0],
                dtype=dtype,
                device=device
            )
            # empty tensors are coalesced at creation (nnz < 2) we must force the uncoalesced state
            input_uncoalesced._coalesced_(False)
            self._test_asin_arcsin(input_uncoalesced, coalesced)

    @coalescedonoff
    @dtypes(torch.double)
    @dtypesIfMPS(torch.float32)
    def test_mv(self, device, dtype, coalesced):
        # [FUNCTION DOC] Tests sparse matrix-vector multiplication (matmul with a vector)
        # against dense for several sizes, and checks RuntimeError for mismatched sizes and for
        # passing a sparse tensor as the vector.
        def test_shape(di, dj, dk, nnz):
            # [FUNCTION DOC] HELPER: Builds a sparse matrix and a dense vector and asserts
            # matmul equals the dense result.
            x, _, _ = self._gen_sparse(2, nnz, [di, dj], dtype, device, coalesced)
            t = torch.randn(dk, dtype=dtype, device=device)

            res = x.matmul(t)
            expected = self.safeToDense(x).matmul(t)
            self.assertEqual(res, expected)

        test_shape(10, 100, 100, 20)
        test_shape(100, 1000, 1000, 20)
        test_shape(64, 10000, 10000, 20)
        test_shape(0, 100, 100, 0)
        test_shape(10, 0, 0, 0)
        test_shape(10, 100, 100, 0)
        test_shape(10, 100, 100, 20)

        with self.assertRaisesRegex(RuntimeError, r"mv: expected self\.size\(-1\) == vec\.size\(-1\)"):
            test_shape(10, 100, 10, 20)

        with self.assertRaisesRegex(RuntimeError, "mv: two tensor dim should be 2 and 1"):
            x, _, _ = self._gen_sparse(2, 20, [10, 100], dtype, device, coalesced)
            y, _, _ = self._gen_sparse(2, 20, [10, 100], dtype, device, coalesced)
            res = x.mv(y)

    @dtypes(*floating_and_complex_types())
    @dtypesIfMPS(torch.float32, torch.bfloat16, torch.complex64)
    def test_sparse_add_coalesce(self, device, dtype):
        # [FUNCTION DOC] Adds two sparse vectors with a duplicated index (plain and with a zero-
        # size dense dim) and asserts the sum is not wrongly flagged coalesced: the check fails
        # only when the sum has an index count other than 2 yet claims to be coalesced.
        i = self.index_tensor([[1, 2, 1]], device=device)
        v = torch.tensor([3, 4, 5], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3]))
        y = self.sparse_tensor(i, v, torch.Size([3]))
        z = x + y

        self.assertFalse(z._indices().numel() != 2 and z.is_coalesced())

        i = self.index_tensor([[1, 2, 1]], device=device)
        v = torch.empty([3, 0], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 0]))
        y = self.sparse_tensor(i, v, torch.Size([3, 0]))
        z = x + y

        self.assertFalse(z._indices().numel() != 2 and z.is_coalesced())

    @onlyCUDA
    def test_storage_not_null(self, device):
        # [FUNCTION DOC] CUDA-only: empty sparse COO tensors (plain and with a zero-size dim)
        # must still report a valid device index (not minus one).
        x = torch.sparse_coo_tensor((2,), dtype=torch.float32, device=device)
        self.assertNotEqual(x.get_device(), -1)

        x = torch.sparse_coo_tensor((2, 0), dtype=torch.float32, device=device)
        self.assertNotEqual(x.get_device(), -1)

    @onlyCUDA
    @deviceCountAtLeast(2)
    def test_same_gpu(self, devices):
        # [FUNCTION DOC] CUDA test needing two GPUs: builds sparse tensors from index and value
        # tensors on the second GPU, and empty ones, with device index 1, and checks the tensor,
        # its values and its indices all report device 1.
        def check_device(x, device_id):
            # [FUNCTION DOC] HELPER: Asserts the sparse tensor, its values and its indices all
            # report the given device index.
            self.assertEqual(x.get_device(), device_id)
            self.assertEqual(x._values().get_device(), device_id)
            self.assertEqual(x._indices().get_device(), device_id)

        dev1, dev2 = devices[0], devices[1]

        i = self.index_tensor([[2]], device=dev2)
        v = torch.tensor([5], device=dev2)
        x = self.sparse_tensor(i, v, torch.Size([3]), device=1)
        check_device(x, 1)

        i = self.index_tensor([[2]], device=dev2)
        v = torch.empty(1, 0, device=dev2)
        x = self.sparse_tensor(i, v, torch.Size([3, 0]), device=1)
        check_device(x, 1)

        x = self.sparse_empty(3, device=1)
        check_device(x, 1)

        x = self.sparse_empty(3, 0, device=1)
        check_device(x, 1)

    def _test_new_device(self, size, device=torch.cuda):
        # [FUNCTION DOC] HELPER: Creates a double COO tensor of the given size under a CUDA
        # device context for the given index, asserts it is on that device, and asserts that new
        # called with no arguments and with a size both produce tensors on the same device.
        # KNOWN ISSUE: the default device argument is the torch.cuda module object, which is not
        # a valid device; all callers pass an explicit index.
        with torch.cuda.device(device):
            x = torch.sparse_coo_tensor(size, device='cuda', dtype=torch.float64)
        self.assertEqual(x.get_device(), device)
        x1 = x.new()
        x2 = x.new(2, 3)
        self.assertEqual(x1.get_device(), device)
        self.assertEqual(x2.get_device(), device)

    @onlyCUDA
    def test_new_device_single_gpu(self):
        # [FUNCTION DOC] CUDA-only: runs the new-device check for several sizes (scalar through
        # zero-size) on device 0.
        self._test_new_device((), 0)
        self._test_new_device((30, 20), 0)
        self._test_new_device((30, 20, 10), 0)
        self._test_new_device((30, 20, 10, 0), 0)

    @onlyCUDA
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_new_device_multi_gpu(self):
        # [FUNCTION DOC] CUDA-only, needs multiple GPUs: runs the new-device check for several
        # sizes on device 1.
        self._test_new_device((), 1)
        self._test_new_device((30, 20), 1)
        self._test_new_device((30, 20, 10), 1)
        self._test_new_device((30, 20, 10, 0), 1)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_new(self, device, dtype, coalesced):
        # [FUNCTION DOC] Tests the legacy new method: on non-CUDA tensors building from indices
        # and values equals the coalesced source (compared by indices and values), and building
        # with an explicit size equals the original tensor, including a zero-size case.
        def test_shape(sparse_dims, nnz, with_size):
            # [FUNCTION DOC] HELPER: Generates a sparse tensor and checks new with indices and
            # values (CPU only) and new with indices, values and size.
            x, indices, values = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)
            if not x.is_cuda:
                # CUDA sparse tensors currently requires the size to be
                # specified if nDimV > 0
                out = x.new(indices, values).coalesce()
                x_c = x.coalesce()
                self.assertEqual((out.indices(), out.values()), (x_c.indices(), x_c.values()))
            self.assertEqual(x.new(indices, values, x.size()), x)

        test_shape(3, 10, 100)
        test_shape(3, 0, [100, 100, 0])

    @onlyCPU  # not really, but we only really want to run this once
    @dtypes(torch.float64, torch.float32, torch.float16, torch.cfloat, torch.cdouble)
    def test_factory(self, device, dtype):
        # [FUNCTION DOC] CPU-only (run once): sweeps sparse_coo_tensor construction over empty
        # and non-empty values, with and without explicit size, indices and values given as
        # tensors or Python lists and scalars, and CPU or last CUDA device. Checks the stored
        # indices and values, the size (default inferred size when none given), the dtype, the
        # device and that requires_grad is true.
        for test_empty_tensor in [True, False]:
            if test_empty_tensor:
                default_size = torch.Size([1, 3, 0])
                size = torch.Size([3, 3, 0])
            else:
                default_size = torch.Size([1, 3])
                size = torch.Size([3, 3])
            for include_size in [True, False]:
                for use_tensor_idx in [True, False]:
                    for use_tensor_val in [True, False]:
                        for use_cuda in ([False] if not torch.cuda.is_available() else [True, False]):
                            # have to include size with cuda sparse tensors
                            include_size = include_size or use_cuda
                            long_dtype = torch.int64
                            device = torch.device('cpu') if not use_cuda else \
                                torch.device(torch.cuda.device_count() - 1)
                            indices = torch.tensor(([0], [2]), dtype=long_dtype) if use_tensor_idx else ([0], [2])
                            if test_empty_tensor:
                                values = torch.empty(1, 0).to(dtype)
                            else:
                                if use_tensor_val:
                                    values = torch.tensor([1.], dtype=dtype)
                                else:
                                    values = 1.
                            if include_size:
                                sparse_tensor = torch.sparse_coo_tensor(indices, values, size, dtype=dtype,
                                                                        device=device, requires_grad=True)
                            else:
                                sparse_tensor = torch.sparse_coo_tensor(indices, values, dtype=dtype,
                                                                        device=device, requires_grad=True)
                            self.assertEqual(indices, sparse_tensor._indices())
                            self.assertEqual(values, sparse_tensor._values())
                            self.assertEqual(size if include_size else default_size, sparse_tensor.size())
                            self.assertEqual(dtype, sparse_tensor.dtype)
                            if use_cuda:
                                self.assertEqual(device, sparse_tensor._values().device)
                            self.assertEqual(True, sparse_tensor.requires_grad)

    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_factory_size_check(self, device, dtype):
        # [FUNCTION DOC] Checks that sparse_coo_tensor raises RuntimeError for inconsistent
        # inputs: size inconsistent with indices (several variants including empty dims), found
        # negative index, and values with an incorrect size.
        indices = self.index_tensor([[1, 2],
                                    [0, 2]], device=device)
        values = torch.tensor([.5, .5], dtype=dtype, device=device)
        sizes = torch.Size([2, 3])
        with self.assertRaisesRegex(RuntimeError, "size is inconsistent with indices"):
            torch.sparse_coo_tensor(indices, values, sizes, dtype=dtype, device=device)

        indices.fill_(-1)
        with self.assertRaisesRegex(RuntimeError, "found negative index"):
            torch.sparse_coo_tensor(indices, values, sizes, dtype=dtype, device=device)

        indices = self.index_tensor([[1, 2],
                                    [0, 2]], device=device)
        values = torch.empty([2, 1, 0], dtype=dtype, device=device)
        sizes = torch.Size([2, 3, 1, 0])
        with self.assertRaisesRegex(RuntimeError, "size is inconsistent with indices"):
            torch.sparse_coo_tensor(indices, values, sizes, dtype=dtype, device=device)

        indices = self.index_tensor([[1, 2],
                                    [0, 2]], device=device)
        values = torch.empty([2, 2, 2], dtype=dtype, device=device)
        sizes = torch.Size([0, 0, 2, 2])
        with self.assertRaisesRegex(RuntimeError, "size is inconsistent with indices"):
            torch.sparse_coo_tensor(indices, values, sizes, dtype=dtype, device=device)

        indices = self.index_tensor([[1, 2],
                                    [0, 2]], device=device)
        values = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=dtype, device=device)
        sizes = torch.Size([3, 3, 2])
        with self.assertRaisesRegex(RuntimeError, "values has incorrect size"):
            torch.sparse_coo_tensor(indices, values, sizes, dtype=dtype, device=device)

        indices = self.index_tensor([[1, 2],
                                    [0, 2]], device=device)
        values = torch.empty([2, 1, 0], dtype=dtype, device=device)
        sizes = torch.Size([3, 3, 2, 0])
        with self.assertRaisesRegex(RuntimeError, "values has incorrect size"):
            torch.sparse_coo_tensor(indices, values, sizes, dtype=dtype, device=device)

    def test_factory_empty_indices(self, device):
        # [FUNCTION DOC] Checks that an empty sparse tensor created from a size has empty int64
        # indices whose first dimension equals the number of dimensions of the size.
        tensor = torch.sparse_coo_tensor(torch.Size([2, 0]), device=device)
        expected_indices = torch.empty((2, 0), dtype=torch.long, device=device)
        self.assertEqual(tensor._indices(), expected_indices)

        tensor = torch.sparse_coo_tensor(torch.Size([2, 2, 0]), device=device)
        expected_indices = torch.empty((3, 0), dtype=torch.long, device=device)
        self.assertEqual(tensor._indices(), expected_indices)

        tensor = torch.sparse_coo_tensor(torch.Size([2, 2, 0, 0]), device=device)
        expected_indices = torch.empty((4, 0), dtype=torch.long, device=device)
        self.assertEqual(tensor._indices(), expected_indices)

    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_factory_nnz(self, device, dtype):
        # [FUNCTION DOC] Checks that sparse_coo_tensor raises RuntimeError saying indices and
        # values must have the same nnz when their nnz differ, with normal and empty values.
        indices = self.index_tensor([[0]], device=device)  # (sparse_dim, nnz): (1, 1)
        values = torch.tensor([[1, 1], [1, 1]], dtype=dtype, device=device)  # (nnz, ...): (2, 2)
        sizes = torch.Size([2, 2])
        with self.assertRaisesRegex(RuntimeError, "indices and values must have same nnz"):
            torch.sparse_coo_tensor(indices, values, sizes, dtype=dtype, device=device)

        indices = self.index_tensor([[0]], device=device)  # (sparse_dim, nnz): (1, 1)
        values = torch.empty([2, 0], dtype=dtype, device=device)  # (nnz, ...): (2, 0)
        sizes = torch.Size([2, 0])
        with self.assertRaisesRegex(RuntimeError, "indices and values must have same nnz"):
            torch.sparse_coo_tensor(indices, values, sizes, dtype=dtype, device=device)

    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_factory_nnz_zero(self, device, dtype):
        # [FUNCTION DOC] Checks construction from empty indices and values: the stored
        # indices, values and size equal the expected ones, where the size is inferred from
        # the values shape plus the sparse dim count when no size is passed, or taken from the
        # explicit size.
        def test_shape(i_shape, v_shape, size, expected_size):
            # [FUNCTION DOC] HELPER: Builds a sparse tensor from empty index and value tensors
            # of the given shapes (with the size only if given) and asserts indices, values and
            # size equal the expected ones.
            if size:
                t = torch.sparse_coo_tensor(torch.empty(i_shape), torch.empty(v_shape), torch.Size(size),
                                            dtype=dtype, device=device)
            else:
                t = torch.sparse_coo_tensor(torch.empty(i_shape), torch.empty(v_shape), dtype=dtype, device=device)
            expected_indices = torch.empty(i_shape, device=device, dtype=torch.int64)
            expected_values = torch.empty(v_shape, device=device, dtype=dtype)
            expected_size = torch.Size(expected_size)
            self.assertEqual(t._indices(), expected_indices)
            self.assertEqual(t._values(), expected_values)
            self.assertEqual(t.size(), expected_size)

        test_shape([1, 0], [0, 2, 4, 0], None, [0, 2, 4, 0])
        test_shape([3, 0], [0, 2, 4, 0], None, [0, 0, 0, 2, 4, 0])
        test_shape([1, 0], [0, 2, 4, 0], [0, 2, 4, 0], [0, 2, 4, 0])
        test_shape([3, 0], [0, 2, 4, 0], [0, 0, 0, 2, 4, 0], [0, 0, 0, 2, 4, 0])
        test_shape([3, 0], [0, 2, 4, 0], [1, 2, 3, 2, 4, 0], [1, 2, 3, 2, 4, 0])

    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_factory_dense_dim(self, device, dtype):
        # [FUNCTION DOC] Checks that construction raises RuntimeError saying values has
        # incorrect size when the dense dimensions of the values disagree with the given size
        # (plain and with a zero-size dim).
        indices = self.index_tensor([[0]], device=device)
        values = torch.tensor([[[1, 1, 1], [1, 1, 1]]], dtype=dtype, device=device)
        sizes = torch.Size([1, 3, 4])
        with self.assertRaisesRegex(RuntimeError, "values has incorrect size"):
            torch.sparse_coo_tensor(indices, values, sizes)

        indices = self.index_tensor([[0]], device=device)
        values = torch.empty([1, 2, 3, 0], dtype=dtype, device=device)
        sizes = torch.Size([1, 3, 4, 0])
        with self.assertRaisesRegex(RuntimeError, "values has incorrect size"):
            torch.sparse_coo_tensor(indices, values, sizes)

    @onlyCPU
    @dtypes(torch.float16, torch.float32, torch.float64, torch.cfloat, torch.cdouble, torch.int64)
    def test_factory_type_inference(self, device, dtype):
        # [FUNCTION DOC] CPU-only: checks dtype inference when no dtype is given: it follows the
        # values tensor (including empty half, float, double and long values, and an integer
        # values tensor giving int64).
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.tensor([1.], dtype=dtype))
        self.assertEqual(dtype, t.dtype)
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.tensor([1]))
        self.assertEqual(torch.int64, t.dtype)

        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.HalfTensor(1, 0))
        self.assertEqual(torch.float16, t.dtype)
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.FloatTensor(1, 0))
        self.assertEqual(torch.float32, t.dtype)
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.DoubleTensor(1, 0))
        self.assertEqual(torch.float64, t.dtype)
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.LongTensor(1, 0))
        self.assertEqual(torch.int64, t.dtype)

    @onlyCUDA
    def test_factory_device_type_inference(self, device):
        # [FUNCTION DOC] CUDA-only: checks device inference for indices and values placed on
        # CPU or CUDA with explicit device CPU, CUDA or None: with no device and mismatched
        # indices and values it must raise RuntimeError; otherwise the result is on CUDA only
        # when the device is CUDA or, for no device, when the values are on CUDA, identically
        # for empty values.
        # both indices/values are CUDA

        cpu_cuda = ('cpu', 'cuda')
        cpu_cuda_none = cpu_cuda + (None,)
        for indices_device, values_device, device in itertools.product(cpu_cuda,
                                                                       cpu_cuda,
                                                                       cpu_cuda_none):
            indices = torch.tensor(([0], [2]), device=indices_device)
            values = torch.tensor([1.], device=values_device)
            empty_values = torch.empty(1, 0).to(values_device)
            shape = (1, 3)
            empty_shape = (1, 3, 0)
            if device is None and indices_device != values_device:
                with self.assertRaises(RuntimeError):
                    torch.sparse_coo_tensor(indices, values, shape, device=device)
                with self.assertRaises(RuntimeError):
                    torch.sparse_coo_tensor(indices, empty_values, empty_shape, device=device)
            else:
                t = torch.sparse_coo_tensor(indices, values, shape, device=device)
                t_empty = torch.sparse_coo_tensor(indices, empty_values, empty_shape, device=device)
                should_be_cuda = (device == 'cuda' or (device is None and values_device == 'cuda'))
                self.assertEqual(should_be_cuda, t.is_cuda)
                self.assertEqual(t.is_cuda, t_empty.is_cuda)

    @onlyCPU
    def test_factory_copy(self, device):
        # [FUNCTION DOC] CPU-only: checks whether construction copies its inputs by comparing
        # data pointers: indices are shared only when already int64 and values only when
        # already the requested float64 dtype (empty tensors always count as shared), including
        # complex values.
        def test_tensor(indices, values, indices_equal, values_equal):
            # [FUNCTION DOC] HELPER: Builds a double sparse tensor from the indices and values
            # and asserts the data pointers of its stored indices and values are equal or
            # different from the inputs according to the expected flags.
            sparse_tensor = torch.sparse_coo_tensor(indices, values, dtype=torch.float64, device=device)
            if indices_equal:
                self.assertEqual(indices.data_ptr(), sparse_tensor._indices().data_ptr())
            else:
                self.assertNotEqual(indices.data_ptr(), sparse_tensor._indices().data_ptr())
            if values_equal:
                self.assertEqual(values.data_ptr(), sparse_tensor._values().data_ptr())
            else:
                self.assertNotEqual(values.data_ptr(), sparse_tensor._values().data_ptr())

        # both correct
        indices = torch.tensor(([0], [2]), dtype=torch.int64)
        values = torch.tensor([1.], dtype=torch.float64)
        test_tensor(indices, values, True, True)

        indices = torch.tensor(([0], [2]), dtype=torch.int64)
        values = torch.DoubleTensor(1, 0)
        test_tensor(indices, values, True, True)

        # only indices correct
        indices = torch.tensor(([0], [2]), dtype=torch.int64)
        values = torch.tensor([1.], dtype=torch.float32)
        test_tensor(indices, values, True, False)

        indices = torch.tensor(([0], [2]), dtype=torch.int64)
        values = torch.tensor([1.], dtype=torch.float16)
        test_tensor(indices, values, True, False)

        indices = torch.tensor(([0], [2]), dtype=torch.int64)
        values = torch.FloatTensor(1, 0)
        test_tensor(indices, values, True, True)  # An empty tensor's data_ptr is always equal to 0

        # only values correct
        indices = torch.tensor(([0], [2]), dtype=torch.int32)
        values = torch.tensor([1.], dtype=torch.float64)
        test_tensor(indices, values, False, True)

        indices = torch.tensor(([0], [2]), dtype=torch.int32)
        values = torch.DoubleTensor(1, 0)
        test_tensor(indices, values, False, True)

        # neither correct
        indices = torch.tensor(([0], [2]), dtype=torch.int32)
        values = torch.tensor([1.], dtype=torch.float32)
        test_tensor(indices, values, False, False)

        indices = torch.tensor(([0], [2]), dtype=torch.int32)
        values = torch.FloatTensor(1, 0)
        test_tensor(indices, values, False, True)  # An empty tensor's data_ptr is always equal to 0

        # complex support
        indices = torch.tensor(([0], [2]), dtype=torch.int64)
        values = make_tensor([1, ], dtype=torch.cdouble, device=device)
        test_tensor(indices, values, True, False)

        indices = torch.tensor(([0], [2]), dtype=torch.int32)
        values = make_tensor([1, 1], dtype=torch.cdouble, device=device)
        test_tensor(indices, values, False, False)

    @onlyCPU  # just run once, we test both cpu and cuda
    def test_legacy_new_device(self, device):
        # [FUNCTION DOC] CPU-only: checks that the legacy new method raises RuntimeError
        # whenever the requested device differs from the device of the tensor, in every call
        # form (no args, indices and values, with size, size only), for CPU tensors and, when
        # CUDA is available, for CUDA tensors.
        i = torch.tensor([[0, 1, 1], [2, 0, 2]])
        v = torch.tensor([3., 4., 5.])
        size = torch.Size([2, 3])

        x = torch.sparse_coo_tensor(i, v, size, device='cpu')
        self.assertRaises(RuntimeError, lambda: x.new(device='cuda'))
        self.assertRaises(RuntimeError, lambda: x.new(i, v, device='cuda'))
        self.assertRaises(RuntimeError, lambda: x.new(i, v, size, device='cuda'))
        self.assertRaises(RuntimeError, lambda: x.new(torch.Size([2, 3, 4]), device='cuda'))

        if torch.cuda.is_available():
            x = torch.sparse_coo_tensor(i, v, size, device='cuda')
            self.assertRaises(RuntimeError, lambda: x.new(device='cpu'))
            self.assertRaises(RuntimeError, lambda: x.new(i, v, device='cpu'))
            self.assertRaises(RuntimeError, lambda: x.new(i, v, size, device='cpu'))
            self.assertRaises(RuntimeError, lambda: x.new(torch.Size([2, 3, 4]), device='cpu'))

    def test_legacy_new(self, device):
        # [FUNCTION DOC] Checks the legacy new method on a sparse tensor: no arguments or size
        # arguments give sparse COO layout, while passing a storage, a dense tensor or a Python
        # list raises TypeError.
        i = torch.tensor([[0, 1, 1], [2, 0, 2]])
        v = torch.tensor([3., 4., 5.])
        size = torch.Size([2, 3])
        s = torch.sparse_coo_tensor(i, v, size)

        self.assertEqual(torch.sparse_coo, s.new(device='cpu').layout)
        self.assertRaises(TypeError, lambda: s.new(v.untyped_storage()))
        self.assertRaises(TypeError, lambda: s.new(v))
        self.assertEqual(torch.sparse_coo, s.new(torch.Size([2, 3])).layout)
        self.assertRaises(TypeError, lambda: s.new([6]))

    @onlyCPU  # not really, but we only really want to run this once
    def test_dtypes(self, device):
        # [FUNCTION DOC] CPU-only (run once): runs the common dtype test helper for sparse COO
        # over all types, complex, half, bool and bfloat16, on CPU and on the first CUDA device
        # when available.
        all_sparse_dtypes = all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16)
        do_test_dtypes(self, all_sparse_dtypes, torch.sparse_coo, torch.device('cpu'))
        if torch.cuda.is_available():
            do_test_dtypes(self, all_sparse_dtypes, torch.sparse_coo, torch.device('cuda:0'))

    def _test_empty_full(self, device, dtype, requires_grad):
        # [FUNCTION DOC] HELPER: Checks properties of empty-like sparse tensors for a dtype
        # and requires_grad setting: sparse_coo_tensor of a shape, zeros with out, new_empty
        # (with and without dtype and device) and empty_like (with and without overrides) all
        # return a sparse COO tensor of the right shape, dtype, layout and device, with the
        # expected requires_grad.
        shape = (2, 3)
        layout = torch.sparse_coo

        def check_value(tensor, value=None, dtype=dtype, requires_grad=requires_grad):
            # [FUNCTION DOC] HELPER: Asserts shape, dtype, layout (sparse COO) and requires_grad
            # of the tensor, its device when it is CUDA and a device was given, and, when a
            # value is given, equality with a tensor of that shape filled with the value.
            self.assertEqual(shape, tensor.shape)
            self.assertIs(dtype, tensor.dtype)
            self.assertIs(layout, tensor.layout)
            self.assertEqual(tensor.requires_grad, requires_grad)
            if tensor.is_cuda and device is not None:
                self.assertEqual(device, tensor.device)
            if value is not None:
                fill = tensor.empty(shape, dtype=dtype).fill_(value)
                self.assertEqual(tensor, fill)

        v = torch.sparse_coo_tensor(shape, dtype=dtype, device=device, requires_grad=requires_grad)
        check_value(v)

        out = v.new()
        check_value(torch.zeros(shape, out=out, device=device, requires_grad=requires_grad))

        int64_dtype = torch.int64
        check_value(v.new_empty(shape), requires_grad=False)
        check_value(v.new_empty(shape, dtype=int64_dtype, device=device, requires_grad=False),
                    dtype=int64_dtype, requires_grad=False)
        check_value(torch.empty_like(v), requires_grad=False)
        check_value(torch.empty_like(v, dtype=int64_dtype, layout=layout, device=device, requires_grad=False),
                    dtype=int64_dtype, requires_grad=False)

    @onlyCPU  # not really, but we only really want to run this once
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @parametrize('requires_grad', (True, False))
    def test_empty_full(self, device, dtype, requires_grad):
        # [FUNCTION DOC] CPU-only: runs the empty and full helper for each dtype and
        # requires_grad setting, skipping grad-requiring non float or complex dtypes, on
        # the default device and, if CUDA is available, with no device and with the first
        # CUDA device.
        if requires_grad and not (dtype.is_floating_point or dtype.is_complex):
            self.skipTest(f'requires_grad==True requires float or complex dtype, got {dtype}')

        self._test_empty_full(device, dtype, requires_grad)
        if torch.cuda.is_available():
            self._test_empty_full(None, dtype, requires_grad)
            self._test_empty_full(torch.device('cuda:0'), dtype, requires_grad)

    def test_is_sparse(self, device):
        # [FUNCTION DOC] Checks is_sparse is false for dense tensors (including zero-size) and
        # true for a sparse empty tensor.
        x = torch.randn(3, 3)
        self.assertFalse(x.is_sparse)

        x = torch.randn(3, 3, 0)
        self.assertFalse(x.is_sparse)

        x = self.sparse_empty(1, 0, device=device)
        self.assertTrue(x.is_sparse)

    def test_resize_as(self, device):
        # [FUNCTION DOC] Checks resize_as_ on an empty sparse tensor used as a template for
        # shapes with a zero-size and non-zero size: the resized copy has the template shape
        # and, after zeroing, can be added to the template (needs matching sparse and dense
        # dim counts).
        def do_test(t):
            # [FUNCTION DOC] HELPER: Makes a new empty tensor, resizes it like the given sparse
            # tensor and zeroes it, asserts same shape, and asserts the sum with the template
            # equals the template.
            y = t.new().resize_as_(t).zero_()
            self.assertEqual(y.shape, t.shape)
            # Check that y can be added to t. Currently, this requires that
            # sparse_dim and dense_dim match.
            self.assertEqual(t, t + y)

        do_test(self.sparse_empty([3, 0], device=device))
        do_test(self.sparse_empty([3, 3], device=device))

    def _test_resize_shape(self, x_i, x_v, x_size, y_i, y_v, y_size, dtype, device):
        # [FUNCTION DOC] HELPER: Builds a sparse tensor x and a sparse template y from given
        # index, value and size shapes (x with consecutive values, y with ones), resizes x like
        # y and the dense forms alike, and asserts shape, sparse_dim, dense_dim equal y's and
        # that the original data of x is preserved in the first elements of the flat dense view.
        x_v_numel = torch.zeros(x_v).numel()
        x = torch.sparse_coo_tensor(torch.zeros(x_i),
                                    torch.arange(x_v_numel).resize_(x_v).to(torch.float),
                                    torch.Size(x_size), dtype=dtype, device=device)
        x_dense = x.to_dense()
        y = torch.sparse_coo_tensor(torch.zeros(y_i),
                                    torch.ones(y_v).to(torch.float),
                                    torch.Size(y_size), dtype=dtype, device=device)
        y_dense = y.to_dense()
        x.resize_as_(y)
        x_dense.resize_as_(y_dense)
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(x.sparse_dim(), y.sparse_dim())
        self.assertEqual(x.dense_dim(), y.dense_dim())
        self.assertEqual(x.shape, x_dense.shape)
        self.assertEqual(y.shape, y_dense.shape)
        # Here we make sure that the original data are preserved after resizing
        self.assertEqual(x.to_dense().view(-1)[0:x_v_numel].view(x_v),
                         x_dense.view(-1)[0:x_v_numel].view(x_v))

    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_resize(self, device, dtype):
        # [FUNCTION DOC] Documents the supported and unsupported resize_as_ cases on sparse
        # tensors: growing dense dims or sparse dims works; changing both sparse and dense dims
        # works only if nnz is zero; and RuntimeError is expected when adding or removing dense
        # dims, changing the sparse dim count of a non-empty tensor, shrinking sparse dims or
        # shrinking dense dims of a non-empty tensor, each with a distinct message.
        # 1. Expand the size of some dense dimensions [Supported]
        self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                [1, 1], [1, 2, 4], [2, 2, 4],
                                dtype=dtype, device=device)

        self._test_resize_shape([1, 1], [1, 2, 0], [2, 2, 0],
                                [1, 1], [1, 2, 4], [2, 2, 4],
                                dtype=dtype, device=device)

        # 2. Expand the size of some sparse dimensions [Supported]
        self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                [1, 1], [1, 2, 3], [4, 2, 3],
                                dtype=dtype, device=device)

        # 3. Change the shapes of both sparse and dense dimensions when nnz is zero [Supported]
        self._test_resize_shape([1, 0], [0, 2, 3], [2, 2, 3],
                                [2, 0], [0, 2, 4, 5], [1, 1, 2, 4, 5],
                                dtype=dtype, device=device)

        self._test_resize_shape([1, 0], [0, 2, 3], [2, 2, 3],
                                [2, 0], [0, 2, 4, 0], [1, 1, 2, 4, 0],
                                dtype=dtype, device=device)

        # 4. Add dims to dense dimensions [Not Supported]
        with self.assertRaisesRegex(RuntimeError, "changing the number of dense dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [1, 1], [1, 2, 3, 4], [2, 2, 3, 4],
                                    dtype=dtype, device=device)

        with self.assertRaisesRegex(RuntimeError, "changing the number of dense dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [1, 1], [1, 2, 3, 0], [2, 2, 3, 0],
                                    dtype=dtype, device=device)

        # 5. Remove dims from dense dimensions [Not Supported]
        with self.assertRaisesRegex(RuntimeError, "changing the number of dense dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [1, 1], [1, 2], [2, 2],
                                    dtype=dtype, device=device)

        # 6. Change the number of sparse dimensions on a non-empty sparse tensor [Not Supported]
        with self.assertRaisesRegex(RuntimeError, "changing the number of sparse dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [2, 1], [1, 2, 3], [1, 2, 2, 3],
                                    dtype=dtype, device=device)

        # 7. Shrink the size of some sparse dimensions on a non-empty sparse tensor [Not Supported]
        with self.assertRaisesRegex(RuntimeError, "shrinking the size of sparse dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [1, 1], [1, 2, 3], [1, 2, 3],
                                    dtype=dtype, device=device)

        # 8. Shrink the size of some dense dimensions on a non-empty sparse tensor [Not Supported]
        with self.assertRaisesRegex(RuntimeError, "shrinking the size of dense dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [1, 1], [1, 2, 2], [2, 2, 2],
                                    dtype=dtype, device=device)

        with self.assertRaisesRegex(RuntimeError, "shrinking the size of dense dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [1, 1], [1, 2, 0], [2, 2, 0],
                                    dtype=dtype, device=device)

    def test_is_nonzero(self, device):
        # [FUNCTION DOC] Tests is_nonzero: true for a single nonzero entry (real and complex,
        # with the cdouble cases skipped on MPS), false for single or multiple zero entries
        # including entries that sum to zero, true for a scalar sparse tensor, and RuntimeError
        # with the ambiguous Boolean value message for a tensor with empty values.
        self.assertTrue(torch.sparse_coo_tensor(([0],), 1., (1,), device=device).is_nonzero())
        self.assertFalse(torch.sparse_coo_tensor(([0],), 0., (1,), device=device).is_nonzero())
        self.assertFalse(torch.sparse_coo_tensor(([0], [0]), 0., (1, 1), device=device).is_nonzero())
        self.assertFalse(torch.sparse_coo_tensor(([0, 0],), (0., 0.), (1,), device=device).is_nonzero())
        self.assertFalse(torch.sparse_coo_tensor(([0, 0],), (-1., 1.), (1,), device=device).is_nonzero())

        # scalar sparse tensor
        self.assertTrue(torch.sparse_coo_tensor(torch.zeros(0, 1), 12.3, [], device=device).is_nonzero())
        with self.assertRaisesRegex(RuntimeError, "Boolean value of Tensor with no values is ambiguous"):
            torch.sparse_coo_tensor(([0, 1],), torch.empty(2, 0), (4, 0), device=device).is_nonzero()
        self.assertTrue(torch.sparse_coo_tensor(([0],), 2.3 - 4.5j, (1,), dtype=torch.cfloat, device=device)
                        .is_nonzero())
        if device != 'mps:0':
            self.assertTrue(torch.sparse_coo_tensor(([0],), 2.3 - 4.5j, (1,), dtype=torch.cdouble, device=device)
                            .is_nonzero())
        self.assertFalse(torch.sparse_coo_tensor(([0],), 0. + 0j, (1,), dtype=torch.cfloat, device=device)
                         .is_nonzero())
        if device != 'mps:0':
            self.assertFalse(torch.sparse_coo_tensor(([0],), 0. + 0j, (1,), dtype=torch.cdouble, device=device)
                             .is_nonzero())

    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_change_tensor_metadata(self, device, dtype):
        # [FUNCTION DOC] Checks that resizing, resizing as, restriding, setting storage or
        # transposing the original index and value tensors in place after construction does not
        # affect the metadata of the sparse tensor: the coalesced indices stay shape 2 by 1 and
        # values 1 by 3.
        i = self.index_tensor([[0], [1]], device=device)
        v = torch.tensor([[3, 4, 5]], dtype=dtype, device=device)
        t = torch.sparse_coo_tensor(i, v, torch.Size([1, 2, 3]), dtype=dtype, device=device)
        i.resize_(2, 3)
        v.resize_(4, 5)
        self.assertEqual(list(t.coalesce().indices().size()), [2, 1])
        self.assertEqual(list(t.coalesce().values().size()), [1, 3])

        i = self.index_tensor([[0], [1]], device=device)
        v = torch.tensor([[3, 4, 5]], dtype=dtype, device=device)
        t = torch.sparse_coo_tensor(i, v, torch.Size([1, 2, 3]))
        i.resize_as_(self.index_tensor([0, 1], device=device))
        v.resize_as_(torch.tensor([3, 4, 5], dtype=dtype, device=device))
        self.assertEqual(list(t.coalesce().indices().size()), [2, 1])
        self.assertEqual(list(t.coalesce().values().size()), [1, 3])

        i = self.index_tensor([[0], [1]], device=device)
        v = torch.tensor([[3, 4, 5]], dtype=dtype, device=device)
        t = torch.sparse_coo_tensor(i, v, torch.Size([1, 2, 3]))
        i.as_strided_((2, 1), (1, 1))
        v.as_strided_((1, 3), (1, 1))
        self.assertEqual(list(t.coalesce().indices().size()), [2, 1])
        self.assertEqual(list(t.coalesce().values().size()), [1, 3])

        i = self.index_tensor([[0], [1]], device=device)
        v = torch.tensor([[3, 4, 5]], dtype=dtype, device=device)
        t = torch.sparse_coo_tensor(i, v, torch.Size([1, 2, 3]))
        i.set_(self.index_tensor([0, 1], device=device))
        v.set_(torch.tensor([3, 4, 5], dtype=dtype, device=device))
        self.assertEqual(list(t.coalesce().indices().size()), [2, 1])
        self.assertEqual(list(t.coalesce().values().size()), [1, 3])

        i = self.index_tensor([[0], [1]], device=device)
        v = torch.tensor([[3, 4, 5]], dtype=dtype, device=device)
        t = torch.sparse_coo_tensor(i, v, torch.Size([1, 2, 3]))
        i.transpose_(0, 1)
        v.transpose_(0, 1)
        self.assertEqual(list(t.coalesce().indices().size()), [2, 1])
        self.assertEqual(list(t.coalesce().values().size()), [1, 3])

    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/182816")
    @coalescedonoff
    @dtypes(torch.double)
    @dtypesIfMPS(torch.float32)
    def test_pickle(self, device, dtype, coalesced):
        # [FUNCTION DOC] Round trips sparse tensors through pickle (dumps and loads) for several
        # shapes, sparse dims and nnz (including empty and, when not coalesced, duplicate
        # indices) and asserts the loaded tensor equals the original.
        import pickle

        shape_sparse_dim_nnz = [
            ((), 0, 2),
            ((0,), 0, 10),
            ((2,), 0, 3),
            ((100, 3), 1, 3),
            ((100, 20, 3), 2, 0),
            ((10, 0, 3), 0, 3),
            ((10, 0, 3), 0, 0),
        ]

        for shape, sparse_dim, nnz in shape_sparse_dim_nnz:
            indices_shape = torch.Size((sparse_dim, nnz))
            values_shape = torch.Size((nnz,) + shape[sparse_dim:])
            indices = torch.arange(indices_shape.numel(), dtype=self.index_tensor(0).dtype,
                                   device=device).view(indices_shape)
            for d in range(sparse_dim):
                indices[d].clamp_(max=(shape[d] - 1))  # make it valid index
            if not coalesced and indices.numel() > 0:
                indices[:, -1] = indices[:, 0]  # make it uncoalesced
            values_numel = values_shape.numel()
            values = torch.arange(values_numel, dtype=dtype,
                                  device=device).view(values_shape).div_(values_numel / 2.)
            sp_tensor = self.sparse_tensor(indices, values, shape)
            serialized = pickle.dumps(sp_tensor)
            sp_tensor_loaded = pickle.loads(serialized)
            self.assertEqual(sp_tensor, sp_tensor_loaded)

    def test_any(self, device):
        # [FUNCTION DOC] Checks torch.any on a sparse boolean tensor returns false for all false
        # values and true when any value is true.
        t = torch.sparse_coo_tensor(torch.tensor(([0, 0], [2, 0])), torch.tensor([False, False]), device=device)
        t_any = torch.tensor(False)
        self.assertEqual(torch.any(t), t_any)
        t = torch.sparse_coo_tensor(torch.tensor(([0, 0], [2, 0])), torch.tensor([True, False]), device=device)
        t_any = torch.tensor(True)
        self.assertEqual(torch.any(t), t_any)

    def test_isnan(self, device):
        # [FUNCTION DOC] Checks torch.isnan on sparse tensors returns a sparse tensor of false
        # for ordinary values and true at the NaN position (compared as ints).
        t = torch.sparse_coo_tensor(torch.tensor(([0, 0], [0, 2])), torch.tensor([1, 4]), device=device)
        t_nan = torch.sparse_coo_tensor(torch.tensor(([0, 0], [0, 2])), torch.tensor([False, False]), device=device)
        self.assertEqual(torch.isnan(t).int(), t_nan.int())
        t = torch.sparse_coo_tensor(torch.tensor(([0, 0], [0, 2])), torch.tensor([1, float("nan")]), device=device)
        t_nan = torch.sparse_coo_tensor(torch.tensor(([0, 0], [0, 2])), torch.tensor([False, True]), device=device)
        self.assertEqual(torch.isnan(t).int(), t_nan.int())

    @coalescedonoff
    @dtypes(torch.float32, torch.float64)
    @dtypesIfMPS(torch.float16, torch.float32)
    def test_div_rounding_mode(self, device, dtype, coalesced):
        # [FUNCTION DOC] Checks division by a scalar with rounding modes None, floor and trunc
        # for sparse tensors matches the dense result for the out of place, in place and out
        # argument forms.
        sparse, _, _ = self._gen_sparse(2, 10, (10, 10), dtype,
                                        device, coalesced)
        dense = self.safeToDense(sparse)

        for mode in (None, 'floor', 'trunc'):
            actual = sparse.div(-2, rounding_mode=mode)
            expect = dense.div(-2, rounding_mode=mode)
            self.assertEqual(self.safeToDense(actual), expect)

            # Test inplace
            actual = sparse.clone().div_(-2, rounding_mode=mode)
            self.assertEqual(self.safeToDense(actual), expect)

            # Test out argument
            actual.zero_()
            torch.div(sparse, -2, rounding_mode=mode, out=actual)
            self.assertEqual(self.safeToDense(actual), expect)

    def test_div_by_sparse_error(self, device):
        # [FUNCTION DOC] Checks that dividing a sparse tensor by a sparse tensor raises
        # RuntimeError with the message that sparse division requires a dense or scalar divisor.
        self.assertRaisesRegex(RuntimeError, 'Sparse division requires',
                               lambda: torch.tensor(1., device=device).to_sparse()
                               / torch.tensor(1., device=device).to_sparse())

    def test_floor_divide_by_sparse_error(self, device):
        # [FUNCTION DOC] Checks that floor-dividing a sparse tensor by a sparse tensor
        # raises RuntimeError with the message that sparse floor division requires a dense
        # or scalar divisor.
        self.assertRaisesRegex(RuntimeError, 'Sparse floor division requires',
                               lambda: torch.tensor(1., device=device).to_sparse()
                               // torch.tensor(1., device=device).to_sparse())

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    @onlyCPU
    def test_sparse_to_numpy(self, device):
        # [FUNCTION DOC] CPU-only, needs NumPy: converting a sparse tensor to numpy must
        # raise TypeError.
        t = torch.sparse_coo_tensor(torch.tensor(([0, 0], [2, 0])), torch.tensor([1, 4]))
        self.assertRaises(TypeError, lambda: t.numpy())

    @coalescedonoff
    @dtypes(torch.double)
    @dtypesIfMPS(torch.float32)
    def test_softmax(self, device, dtype, coalesced):
        # [FUNCTION DOC] Comprehensive sparse softmax and log_softmax test. Defines pure Python
        # reference implementations (a sparse softmax treating unspecified entries as negative
        # infinity, a dense reference, analytic and autograd based Jacobians) and for many
        # shapes compares torch.sparse.softmax and log_softmax outputs, dimension bounds
        # errors, negative dims, the dtype argument and Jacobians (including the log relation)
        # against them.
        import torch.nn.functional as F

        def to_dense(sparse, fill_value=None):
            # [FUNCTION DOC] HELPER: Returns a dense tensor from a sparse one, filling
            # unspecified entries with the given fill value (default zero uses the normal
            # to_dense; otherwise coalesces and writes the values into a tensor full of the
            # fill value).
            """
            Return dense tensor from a sparse tensor using given fill value.
            """
            if fill_value is None or fill_value == 0:
                return sparse.to_dense()
            sparse = sparse.coalesce()
            dense = torch.full(sparse.shape, fill_value, dtype=sparse.dtype, device=sparse.device)
            for idx, value in zip(sparse._indices().t(), sparse._values()):
                dense[tuple(idx)] = value
            return dense

        def softmax_to_dense(sparse, dim):
            # [FUNCTION DOC] HELPER: Dense reference of sparse softmax. Fills unspecified
            # entries with negative infinity, applies dense softmax along the dim and replaces
            # NaNs (rows with no specified entries) by zero to preserve sparsity.
            """Dense softmax of a sparse tensor. Useful only for testing softmax
            correctness.

            When computing softmax of a sparse tensor, the value of
            unspecified items is negative infinity rather than zero so
            that

              softmax(sparse.to_dense(fill_value=-inf), dim) == softmax(sparse, dim).to_dense()

            holds for non-empty lines. One empty lines, the softmax
            values are defined as 0 in order to preserve the sparsity
            of result.

            Note that in PyTorch, ``to_dense`` method does not
            implement the ``fill_value`` keyword argument.
            """
            dtype = sparse.dtype
            device = sparse.device
            dense = to_dense(sparse, fill_value=-float('inf'))
            r = F.softmax(dense, dim)
            # softmax on empty lines results nan, replace with zeros to match the definition
            r[r != r] = 0
            return r

        def sparse_softmax(sparse, dim):
            # [FUNCTION DOC] HELPER: Pure Python prototype of sparse softmax. Coalesces; for a
            # sparse dim it groups entries by their index in the other sparse dims, computes
            # per-group max, exponentiates shifted values, sums per group and normalizes; for a
            # dense dim it applies dense softmax to the values along the matching axis; raises
            # ValueError if the dim is not smaller than the number of sparse plus dense dims.
            # Returns a sparse tensor with the same indices.
            """Pure Python softmax of a sparse tensor. Assuming -inf for
            unspecified sparse tensor data. This is a prototype of
            sparse softmax algorithm in Python.
            """
            dtype = sparse.dtype
            device = sparse.device

            # softmax is non-linear operation, so sparse tensors must
            # be coalesced.
            sparse = sparse.coalesce()
            inf = float('inf')
            indices = sparse._indices()
            values = sparse._values()

            if dim < sparse.sparse_dim():
                nnz = sparse._nnz()

                # compute pool indices
                size = sparse.size()
                strides = torch.ones((sparse.sparse_dim(), 1), dtype=indices.dtype, device=indices.device)
                for i in reversed(range(sparse.sparse_dim() - 1)):
                    strides[i, 0] = strides[i + 1, 0] * size[i + 1]
                strides[dim, 0] = 0

                pool = (indices * strides).sum(dim=0)
                i2p = {}
                for i in range(nnz):
                    c = int(pool[i])
                    if c not in i2p:
                        i2p[c] = len(i2p)
                    pool[i] = i2p[c]

                # compute max
                dense_size = tuple(size[sparse.sparse_dim():])
                mx = torch.empty((pool.max() + 1,) + dense_size, dtype=dtype, device=device)
                mx[:] = -inf
                for n in range(nnz):
                    p = pool[n]
                    mx[p] = torch.max(mx[p], values[n])

                # apply exp to (v - mx) and sum the results
                exp_values = torch.empty_like(values)
                exp_sums = torch.zeros_like(mx)
                for n in range(nnz):
                    p = pool[n]
                    v = exp_values[n] = (values[n] - mx[p]).exp()
                    exp_sums[p] = exp_sums[p] + v

                # normalize with the sum of exponents
                for n in range(nnz):
                    p = pool[n]
                    exp_values[n] = exp_values[n] / exp_sums[p]

                return torch.sparse_coo_tensor(indices,
                                               exp_values,
                                               sparse.size(),
                                               dtype=dtype, device=device)

            elif dim < sparse.sparse_dim() + sparse.dense_dim():
                return torch.sparse_coo_tensor(indices,
                                               F.softmax(values, dim - sparse.sparse_dim() + 1),
                                               sparse.size(),
                                               dtype=dtype, device=device)
            else:
                raise ValueError(
                    f'`dim(={dim})` must be smaller than `sparse_dim(={sparse.sparse_dim()}) + dense_dim(={sparse.dense_dim()})`')

        def softmax_jacobian_analytic(x, dim):
            # [FUNCTION DOC] HELPER: Builds the softmax Jacobian of a dense input with the
            # analytic formula (output times identity minus output), replacing NaNs by zeros,
            # returning a tensor with a leading axis for the differentiated position.
            """Return Jacobian of softmax using analytic formula

               D_jS_i = S_i * (1[i==j] - S_j).

            where S = softmax(x, dim), x is dense tensor, i,j in
            range(x.shape[dim]).
            """
            y = F.softmax(x, dim)
            y[y != y] = 0  # replace nan-s with zeros
            J = torch.zeros((x.shape[dim],) + tuple(x.shape), dtype=x.dtype, device=x.device)
            si = [slice(None)] * len(y.shape)
            sj = [slice(None)] * len(y.shape)
            s = [slice(None)] * len(J.shape)
            for i in range(y.shape[dim]):
                si[dim] = i
                s[dim + 1] = i
                yi = y[tuple(si)]
                for j in range(y.shape[dim]):
                    sj[dim] = j
                    s[0] = j
                    if i == j:
                        J[tuple(s)] = yi * (1 - yi)
                    else:
                        yj = y[tuple(sj)]
                        J[tuple(s)] = - yi * yj
                    sj[dim] = slice(None)
                si[dim] = slice(None)
                s[dim + 1] = slice(None)
            return J

        def softmax_jacobian_autograd(x, dim, log=False):
            # [FUNCTION DOC] HELPER: Builds the softmax (or log_softmax if requested) Jacobian
            # by autograd, backpropagating one-hot gradients for each position along the dim;
            # supports dense and sparse (coalesced) input, builds sparse one-hot gradients,
            # replaces NaNs by zero for dense and returns a tensor with a leading axis for
            # each position.
            """Return Jacobian of softmax using PyTorch autograd feature.

            x can be dense or sparse tensor.
            """
            import itertools

            if x.is_sparse:
                x = x.coalesce()

            dtype = x.dtype
            device = x.device
            shape = tuple(x.shape)
            J = torch.zeros((shape[dim],) + shape, dtype=dtype, device=device)
            for i in range(shape[dim]):
                if x.is_sparse:
                    sparse_dim = x.sparse_dim()
                    dense_dim = x.dense_dim()
                    if dim < sparse_dim:
                        ranges = []
                        for j, sz in enumerate(shape[:sparse_dim]):
                            if dim == j:
                                ranges.append([i])
                            else:
                                ranges.append(list(range(sz)))
                        indices = torch.tensor(list(itertools.product(*ranges)), dtype=torch.long, device=device).t()
                        values = torch.ones((indices.shape[1],) + shape[sparse_dim:], dtype=dtype, device=device)
                    else:
                        ranges = []
                        for sz in shape[:sparse_dim]:
                            ranges.append(list(range(sz)))
                        indices = torch.tensor(list(itertools.product(*ranges)), dtype=torch.long, device=device).t()
                        values = torch.zeros((indices.shape[1],) + shape[sparse_dim:], dtype=dtype, device=device)
                        sv = [slice(None)] * (dense_dim + 1)
                        sv[dim - sparse_dim + 1] = i
                        values[tuple(sv)] = 1
                    v = torch.sparse_coo_tensor(indices, values, shape, dtype=dtype, device=device)
                else:
                    v = torch.zeros_like(x)
                    sv = [slice(None)] * len(v.shape)
                    sv[dim] = i
                    v[tuple(sv)] = 1
                x_ = x.clone()
                x_.requires_grad_(True)

                if log:
                    if x_.is_sparse:
                        y = torch.sparse.log_softmax(x_, dim)
                    else:
                        y = F.log_softmax(x_, dim)
                else:
                    if x_.is_sparse:
                        y = torch.sparse.softmax(x_, dim)
                    else:
                        y = F.softmax(x_, dim)
                        # replace nan-s with zeros
                        y.data[y != y] = 0
                y.backward(v)
                g = x_.grad
                if not g.is_sparse:
                    # replace nan-s with zeros
                    g.data[g != g] = 0
                J[i] = g.to_dense() if g.is_sparse else g
            return J

        @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1166")
        def test_op(sparse_dims, nnz, with_size, coalesced):
            # [FUNCTION DOC] HELPER: Runs all the sparse softmax checks for one random sparse
            # tensor (skipped under TorchDynamo): dim out of range errors, comparison of the
            # Python and C++ implementations and log_softmax, Jacobian agreement between
            # analytic, dense autograd and sparse autograd, the log_softmax relation, and the
            # dtype argument for dim zero.
            if isinstance(with_size, Number):
                with_size = [with_size] * sparse_dims

            x, i, v = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)

            def sparse_log(x):
                # [FUNCTION DOC] HELPER: Returns a sparse tensor with the same indices as the
                # input and the elementwise log of its values.
                return torch.sparse_coo_tensor(x._indices(), x._values().log(),
                                               x.size(), dtype=x.dtype, device=x.device)

            # Check dim out of bounds
            with self.assertRaisesRegex(IndexError, r"Dimension out of range"):
                torch.sparse.softmax(x, x.dim())
            with self.assertRaisesRegex(IndexError, r"Dimension out of range"):
                torch.sparse.softmax(x, -x.dim() - 1)

            for dim in range(x.dim()):
                # Check sparse softmax definition

                # check Python sparse softmax
                y = sparse_softmax(x, dim)
                r1 = softmax_to_dense(x, dim)
                r2 = y.to_dense()
                self.assertEqual(r1, r2)

                # check C++ sparse softmax
                for d in (dim, dim - x.dim()):
                    y1 = torch.sparse.softmax(x, d)
                    self.assertEqual(y, y1)

                    # check C++ sparse log_softmax
                    ly1 = torch.sparse.log_softmax(x, d)
                    self.assertEqual(ly1, sparse_log(y1))

                # Check autograd support on sparse softmax

                # check softmax Jacobian definition for dense input
                x1 = to_dense(x, fill_value=float('-inf'))
                J = softmax_jacobian_analytic(x1, dim)
                if J.shape[0] != x.shape[dim]:
                    raise AssertionError(f"J.shape[0] mismatch: {J.shape[0]} != {x.shape[dim]}")
                if J.shape[dim + 1] != x.shape[dim]:
                    raise AssertionError(f"J.shape[{dim + 1}] mismatch: {J.shape[dim + 1]} != {x.shape[dim]}")

                # check softmax Jacobian from autograd, dense input
                J2 = softmax_jacobian_autograd(x1, dim)
                self.assertEqual(J, J2)

                # check softmax Jacobian from autograd, sparse input
                J3 = softmax_jacobian_autograd(x, dim)
                self.assertEqual(J, J3)

                '''
                y = softmax(x, dim)
                z = log(y) = log_softmax(x, dim)
                Dy/Dx = J
                Dz/Dx = Dz/Dy Dy/Dx = 1/y * J
                => J = J_log * y
                '''
                # log_softmax Jacobian from autograd, dense input
                J2_log = softmax_jacobian_autograd(x1, dim, log=True)

                # log_softmax Jacobian from autograd, sparse input
                J3_log = softmax_jacobian_autograd(x, dim, log=True)

                J = J.transpose(0, dim + 1)
                J2_log = J2_log.transpose(0, dim + 1)
                J3_log = J3_log.transpose(0, dim + 1)
                self.assertEqual(J, J2_log * r1)
                self.assertEqual(J, J3_log * r1)

                if dim == 0:
                    # check dtype argument
                    other_dtype = torch.float32
                    y2 = torch.sparse.softmax(x, dim, dtype=other_dtype)
                    self.assertEqual(y2.dtype, other_dtype)
                    self.assertEqual(y2, y1.type(other_dtype))

                    ly2 = torch.sparse.log_softmax(x, dim, dtype=other_dtype)
                    self.assertEqual(ly2.dtype, other_dtype)
                    self.assertEqual(ly2, ly1.type(other_dtype))

        test_op(1, 10, [3], coalesced)
        test_op(1, 10, [2, 3], coalesced)
        test_op(1, 10, [3, 2], coalesced)
        test_op(2, 10, [2, 3, 4], coalesced)
        test_op(2, 10, [3, 4], coalesced)
        test_op(2, 5, [5, 4], coalesced)
        test_op(2, 10, [3, 4, 2], coalesced)
        test_op(3, 10, [3, 4, 2], coalesced)
        test_op(3, 100, [3, 4, 2], coalesced)
        test_op(3, 100, [3, 4, 2, 3], coalesced)
        test_op(3, 100, [3, 4, 2, 3, 5, 2], coalesced)
        test_op(4, 100, [3, 4, 2, 3, 5, 2], coalesced)


    def _check_zero_nnz_softmax_op(self, func, ndim, device, dtype):
        # [FUNCTION DOC] HELPER: Builds a sparse tensor of the given number of dims with no
        # specified values and a trailing dimension of size 3, applies the sparse softmax-like
        # function along dim 0, asserts the result equals zeros, then gradchecks the function
        # with masked semantics.
        # create a sparse tensor with shape (0,..., 3) it has no materialize values
        t = torch.sparse_coo_tensor([[] for _ in range(ndim)], [], (0,) * (ndim - 1) + (3,), device=device, dtype=dtype)
        out = func(t, 0)
        self.assertEqual(out, torch.zeros_like(t))

        # gradient
        t = t.requires_grad_()
        gradcheck(lambda x: func(x, 0).to_dense(), (t,), masked=True)


    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/183435")
    @dtypes(torch.double, torch.float)
    @dtypesIfMPS(torch.float32)
    @unittest.skipIf(TEST_WITH_CROSSREF, "generator unsupported triggers assertion error")
    def test_softmax_zero_nnz(self, device, dtype):
        # [FUNCTION DOC] Runs the zero-nnz check for sparse softmax with 1 and 10 dimensions.
        # Skipped under TorchDynamo and cross-ref.
        self._check_zero_nnz_softmax_op(torch.sparse.softmax, 1, device, dtype)
        self._check_zero_nnz_softmax_op(torch.sparse.softmax, 10, device, dtype)

    @dtypes(torch.double, torch.float)
    @dtypesIfMPS(torch.float32)
    @unittest.skipIf(TEST_WITH_CROSSREF, "generator unsupported triggers assertion error")
    def test_log_softmax_zero_nnz(self, device, dtype):
        # [FUNCTION DOC] Runs the zero-nnz check for sparse log_softmax with 1 and 10
        # dimensions. Skipped under cross-ref.
        self._check_zero_nnz_softmax_op(torch.sparse.log_softmax, 1, device, dtype)
        self._check_zero_nnz_softmax_op(torch.sparse.log_softmax, 10, device, dtype)

    @dtypes(torch.float)
    def test_log_softmax_float(self, device, dtype):
        # [FUNCTION DOC] Checks sparse log_softmax in low precision: a tensor with large
        # negative values must give the same result as the double (or float on MPS) computation
        # cast back, where float16 replaces float on MPS.
        # float16 -> float32 comparison for mps since mps doesn't have float64
        dtype = torch.float16 if device == "mps:0" else dtype
        x = (torch.rand(4, 3, dtype=dtype, device=device) - 10000000.0).to_sparse()
        out = torch.sparse.log_softmax(x, dim=1).to_dense()
        x_double = x.double() if device != "mps:0" else x.float()
        out_double = torch.sparse.log_softmax(x_double, dim=1).to_dense()
        self.assertEqual(out, out_double.to(dtype=dtype))

    # TODO: Check after why ROCm's cusparseXcsrgemm2Nnz function doesn't return the same nnz value as CUDA
    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/182600")
    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/184329")
    @coalescedonoff
    @dtypes(*floating_and_complex_types())
    @dtypesIfMPS(*all_mps_types())
    @dtypesIfCUDA(*floating_types_and(*[torch.half] if not TEST_WITH_ROCM else [],
                                      *[torch.bfloat16] if SM80OrLater and not TEST_WITH_ROCM else [],
                                      torch.complex64,
                                      *[torch.complex128]
                                      if CUSPARSE_SPMM_COMPLEX128_SUPPORTED or HIPSPARSE_SPMM_COMPLEX128_SUPPORTED
                                      else []))
    @unittest.skipIf(TEST_WITH_CROSSREF, "not working with fake tensor")
    @precisionOverride({torch.bfloat16: 1e-2, torch.float16: 1e-2, torch.complex64: 1e-2, torch.float32: 1e-2})
    def test_sparse_matmul(self, device, dtype, coalesced):
        # [FUNCTION DOC] Tests sparse-sparse matrix multiplication with torch.sparse.mm:
        # compares with the dense product, verifies the result is truly coalesced, gradchecks
        # for double types, compares gradients restricted to custom sparsity patterns, covers
        # empty matrices, error cases and a non-contiguous grad regression. Runs for floating
        # and complex types with looser precision, and skipped under cross-ref and certain
        # dynamo issues.
        """
        This function test `torch.sparse.mm` when both the mat1 and mat2 are sparse tensors.
        """

        def ref_sparse_mm(a, b):
            # [FUNCTION DOC] HELPER: Reference product computed by densifying both
            # sparse inputs.
            return a.to_dense() @ b.to_dense()

        def grad_with_custom_sparsity_pattern_test_helper(sparse_dims, nnz, shape_a, shape_b):
            # [FUNCTION DOC] HELPER: Generates two sparse matrices requiring grad, multiplies
            # them sparsely, backpropagates a random gradient restricted to the product pattern,
            # and compares both input gradients (densified) against the gradients from the same
            # dense computation masked to the input patterns.
            def test_grad_dense(a_s, b_s, g_s):
                # [FUNCTION DOC] HELPER: Computes the dense matmul of densified detached inputs
                # and backpropagates the densified grad, returning the input gradients masked
                # with the original sparsity patterns of a and b.
                a = a_s.to_dense().detach()
                b = b_s.to_dense().detach()
                g = g_s.to_dense().detach()

                a.requires_grad_(True)
                b.requires_grad_(True)
                c = a @ b
                c.backward(g)
                return a.grad.sparse_mask(a_s.coalesce()), b.grad.sparse_mask(b_s.coalesce())

            a, _, _ = self._gen_sparse(sparse_dims, nnz, shape_a, dtype, device, coalesced)
            b, _, _ = self._gen_sparse(sparse_dims, nnz, shape_b, dtype, device, coalesced)
            a.requires_grad_(True)
            b.requires_grad_(True)

            c = torch.sparse.mm(a, b)
            c2 = c.to_dense().detach()
            c2 = torch.rand_like(c2)
            g = c2.sparse_mask(c.coalesce())

            c.backward(g)

            a_grad, b_grad = test_grad_dense(a, b, g)

            # We convert grad to dense since dense and sparse mm
            # implementations handle materialized zeroes differently.
            self.assertEqual(a.grad.to_dense(), a_grad.to_dense())
            self.assertEqual(b.grad.to_dense(), b_grad.to_dense())

        def test_sparse_matmul(sparse_dims, nnz, shape_a, shape_b):
            # [FUNCTION DOC] HELPER: Generates two sparse matrices, asserts torch.sparse.mm
            # equals the dense product and is truly coalesced, and for double types gradchecks
            # it with masked semantics (with a looser nondeterminism tolerance on CUDA) and runs
            # the custom sparsity pattern gradient comparison.
            a, i_a, v_a = self._gen_sparse(sparse_dims, nnz, shape_a, dtype, device, coalesced)
            b, i_b, v_b = self._gen_sparse(sparse_dims, nnz, shape_b, dtype, device, coalesced)

            # dense implementation
            r1 = ref_sparse_mm(a, b)

            # cpp implementation
            r2 = torch.sparse.mm(a, b)
            self.assertEqual(r1, r2.to_dense())

            # Check result is truly coalesced
            self.assertTrue(r2.is_coalesced() and is_coalesced_indices(r2))

            if dtype in [torch.double, torch.cdouble]:
                a.requires_grad_(True)
                b.requires_grad_(True)

                # check autograd support on sparse matmul
                def fn(D1, D2):
                    # [FUNCTION DOC] HELPER: Function under gradcheck returning the dense result
                    # of sparse mm of its two inputs.
                    return torch.sparse.mm(D1, D2).to_dense()

                if a.is_cuda:
                    # For cuda, `nondet_tol` is set with `1e-5`
                    # This is because cuSparse sometimes returns approximate zero values like `~e-323`
                    # TODO: Check this cuSparse issue.
                    # This happens when you do chain multiplication `torch.sparse.mm` operations
                    gradcheck(fn, (a, b), nondet_tol=1e-5, masked=True)
                else:
                    gradcheck(fn, (a, b), masked=True)
                grad_with_custom_sparsity_pattern_test_helper(sparse_dims, nnz, shape_a, shape_b)

        def test_error_cases():
            # [FUNCTION DOC] HELPER: Checks error cases: a three dimensional input is rejected
            # with RuntimeError, incompatible shapes raise RuntimeError with the cannot be
            # multiplied message, and mixed float32 and float16 inputs raise RuntimeError with
            # the dtype mismatch message.
            def fn(sparse_dims, nnz, shape_a, shape_b):
                # [FUNCTION DOC] HELPER: Generates two sparse tensors of the given shapes
                # and multiplies them with sparse mm, discarding the result (used to
                # trigger errors).
                a, i_a, v_a = self._gen_sparse(sparse_dims, nnz, shape_a, dtype, device, coalesced)
                b, i_b, v_b = self._gen_sparse(sparse_dims, nnz, shape_b, dtype, device, coalesced)
                r2 = torch.sparse.mm(a, b)

            # This is not a matrix
            self.assertRaises(RuntimeError, lambda: fn(3, 4, [2, 2, 2], [2, 2, 2]))

            # Shapes does not
            self.assertRaisesRegex(RuntimeError,
                                   r"mat1 and mat2 shapes cannot be multiplied \(2x3 and 4x2\)",
                                   lambda: fn(2, 10, [2, 3], [4, 2]))

            def different_dtypes():
                # [FUNCTION DOC] HELPER: Multiplies a float32 sparse tensor with a float16 one
                # with sparse mm to trigger a dtype mismatch error. Two tensors are generated
                # but only the first is used, in both casts.
                a, i_a, v_a = self._gen_sparse(2, 10, [2, 2], dtype, device, coalesced)
                b, i_b, v_b = self._gen_sparse(2, 10, [2, 2], dtype, device, coalesced)
                r2 = torch.sparse.mm(a.to(torch.float32), a.to(torch.float16))

            self.assertRaisesRegex(RuntimeError, 'mat1 dtype Float does not match mat2 dtype Half', different_dtypes)

        def test_backward_noncontiguous():
            # [FUNCTION DOC] HELPER: Regression check (issue 102493) that the backward of sparse
            # mm with identity matrices gives gradient equal to the input repeatedly, even with
            # non-contiguous grads.
            # Sparse.mm backward used to wrong with non-contiguous grads,
            # see https://github.com/pytorch/pytorch/issues/102493.
            n_reps = 7
            for _ in range(n_reps):
                A = torch.eye(5).to_sparse().requires_grad_(True)
                B = torch.eye(5).to_sparse()
                out = torch.sparse.mm(A, B)
                out.coalesce().values().sum().backward()
                self.assertEqual(A.grad, A)

        for n in range(2, 5):
            for m in range(2, 8):
                for p in range(2, 8):
                    test_sparse_matmul(2, 10, [n, m], [m, p])

        test_sparse_matmul(2, 0, [0, 0], [0, 0])
        test_sparse_matmul(2, 0, [0, 10], [10, 0])
        test_error_cases()
        test_backward_noncontiguous()

    @coalescedonoff
    @dtypes(torch.double)
    def test_assign(self, device, dtype, coalesced):
        # [FUNCTION DOC] Checks that item assignment on a sparse tensor raises TypeError.
        def assign_to():
            # [FUNCTION DOC] HELPER: Generates a sparse tensor and attempts to assign a number
            # to its first element, expected to raise TypeError.
            a, i_a, v_a = self._gen_sparse(2, 5, [2, 3], dtype, device, coalesced)
            a[0] = 100

        self.assertRaises(TypeError, assign_to)

    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_full_broadcast_to(self, device, dtype):
        # [FUNCTION DOC] Tests torch._sparse_broadcast_to against the dense broadcast_to over
        # combinations of shapes and sparse dim counts: validates result coo arguments, checks
        # is_coalesced estimation and dense equality; for non-broadcastable shapes expects
        # RuntimeError saying it does not broadcast.
        def can_broadcast(s0, s1):
            # [FUNCTION DOC] HELPER: Returns True when the first shape can be broadcast to the
            # second, comparing from the trailing dimension with each dim being either 1 or
            # equal; the check indexes the second shape by the position in the first and so
            # assumes the second shape has at least as many dims.
            s0 = tuple(reversed(s0))
            s1 = tuple(reversed(s1))
            for i in range(len(s0)):
                if s0[i] != 1 and s0[i] != s1[i]:
                    return False
            return True
        sizes = (
            (), (1,), (2,), (1, 1), (3, 1), (3, 2), (4, 1, 1), (4, 3, 2)
        )
        for s0, s1 in itertools.combinations(sizes, r=2):
            t = make_tensor(s0, dtype=dtype, device=device, low=-9, high=9)
            for sparse_dims in range(1, len(s0) + 1):
                s = t.to_sparse(sparse_dims)
                if can_broadcast(s0, s1):
                    t_res = torch.broadcast_to(t, s1)
                    s_res = torch._sparse_broadcast_to(s, s1)
                    torch._validate_sparse_coo_tensor_args(s_res._indices(), s_res._values(), s_res.shape)
                    if s_res.is_coalesced():
                        # ensure that is_coalesced is estimated correctly
                        self.assertEqual(s_res, torch.sparse_coo_tensor(s_res._indices(), s_res._values(), s_res.shape).coalesce())
                    self.assertEqual(s_res.to_dense(), t_res)
                else:
                    with self.assertRaisesRegex(RuntimeError,
                                                r"does not broadcast"):
                        torch._sparse_broadcast_to(s, s1)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_sparse_broadcast_to(self, device, dtype, coalesced):
        # [FUNCTION DOC] Checks sparse broadcast_to for hybrid tensors with zero-size dims and
        # leading new dims against the dense broadcast.
        def test(sparse_dims, nnz, with_size, new_size):
            # [FUNCTION DOC] HELPER: Generates a sparse tensor, broadcasts it with the sparse op
            # to the new size and compares densified results with the dense broadcast.
            x = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)[0]
            y = self.safeToDense(x)
            x1 = torch._sparse_broadcast_to(x, new_size)
            y1 = y.broadcast_to(new_size)
            self.assertEqual(self.safeToDense(x1), y1)

        test(4, 6, [7, 3, 1, 3, 0], [7, 3, 4, 3, 0])
        test(4, 6, [7, 3, 1, 3, 0], [2, 7, 3, 1, 3, 0])
        test(4, 6, [7, 3, 1, 3, 1, 3], [7, 3, 1, 3, 2, 3])
        test(4, 6, [7, 3, 1, 3, 2, 1], [7, 3, 1, 3, 2, 3])

    def _test_mul_skips(self, device, dtype, coalesced):
        # [FUNCTION DOC] HELPER: Skips the calling multiplication test (through skipTest) when
        # inputs are uncoalesced and the dtype is float16 or bfloat16 (coalescing would lose
        # precision) or when dtype is bool on a CUDA device (to_dense of uncoalesced boolean
        # CUDA tensors is problematic, issue 81648); otherwise does nothing.
        skipTestIfUncoalesced = False
        # This case always coalesce inputs and that could lead to loss of precision,
        # hence it is inhibited for float16/bfloat16 by providing already coalesced tensors.
        if not coalesced and dtype in {torch.float16, torch.bfloat16}:
            skipTestIfUncoalesced = True
        # to_dense is problematic for boolean non-coalesced CUDA tensors
        # see https://github.com/pytorch/pytorch/issues/81648
        if not coalesced and dtype == torch.bool and torch.device(device).type == "cuda":
            skipTestIfUncoalesced = True

        if skipTestIfUncoalesced:
            self.skipTest(f"Test with dtype={dtype}, device={device} runs only with coalesced inputs")

    @coalescedonoff
    # NOTE: addcmul_out is not implemented for bool.
    @dtypes(*all_types_and_complex_and(torch.bfloat16, torch.float16))
    @dtypesIfMPS(*all_mps_types())
    @precisionOverride({torch.bfloat16: 1e-2, torch.float16: 1e-2})
    def test_sparse_sparse_mul(self, device, dtype, coalesced):
        # [FUNCTION DOC] Tests sparse times sparse elementwise multiplication against the dense
        # product for many sub-shapes, empty cases, and broadcasting of size-one dense dims,
        # after the shared skip helper. Covers all real and complex types plus half and bfloat16
        # with looser tolerance.
        self._test_mul_skips(device, dtype, coalesced)

        shape = (2, 3, 4, 10)
        nnz = 10

        def check(self, x, y):
            # [FUNCTION DOC] HELPER: Multiplies two sparse tensors and asserts the densified
            # product equals the product of the densified inputs. Takes the test case as an
            # explicit first argument.
            res_sparse = x * y
            res_dense = x.to_dense() * y.to_dense()
            self.assertEqual(res_sparse.to_dense(), res_dense)

        def check_empty(sparse_shape, nnz, dense_shape, coalesce):
            # [FUNCTION DOC] HELPER: For each of nnz and zero nnz, and for an empty and a
            # nonempty extra trailing dim, generates a sparse tensor and checks that squaring it
            # matches dense.
            # KNOWN ISSUE: it reads the sparse dimension count from the enclosing test scope
            # (assigned later in the loop) and the dense shape it computes is never used.
            from itertools import product
            for nnz_val, shape_suffix in product((nnz, 0), ((), (0,))):
                empty_sparse_shape = sparse_shape + shape_suffix
                empty_dense_shape = dense_shape + shape_suffix
                x = self._gen_sparse(sparse_dim, nnz_val, empty_sparse_shape, dtype, device, coalesce)[0]
                check(self, x, x)

        # TODO: uncomment once backward is implemented for sparse tensors that broadcast in dense dims.
        # def check_autograd(x, y):
        #     if dtype in {torch.double, torch.cdouble}:
        #         xa = x.detach().clone().requires_grad_(True)
        #         ya = y.detach().clone().requires_grad_(True)
        #         gradcheck(lambda a, b: (a * b).to_dense(), (xa, ya), masked=True)
        #         gradcheck(lambda a, b: (a * b).to_dense(), (ya, xa), masked=True)

        for dim in range(len(shape) + 1):
            sub_shape = shape[dim:]
            sparse_dim = len(sub_shape) // 2

            check_empty(sub_shape, nnz, shape, coalesced)

            x = self._gen_sparse(sparse_dim, nnz, sub_shape, dtype, device, coalesced)[0]
            y = self._gen_sparse(sparse_dim, nnz, sub_shape, dtype, device, coalesced)[0]
            check(self, x, y)
            # TODO: uncomment once supported
            # check_autograd(x, y)

            # check broadcasting in dense dims
            for d in range(sparse_dim, len(sub_shape)):
                new_shape = sub_shape[:d] + (1,) + sub_shape[d + 1:]
                y = self._gen_sparse(sparse_dim, nnz, new_shape, dtype, device, coalesced)[0]
                check(self, x, y)
                # TODO: uncomment once supported
                # check_autograd(x, y)

    @coalescedonoff
    @expectedFailureMPS
    @dtypes(*all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16))
    @dtypesIfMPS(*all_mps_types())
    @precisionOverride({torch.bfloat16: 1e-2, torch.float16: 1e-2})
    def test_sparse_dense_mul(self, device, dtype, coalesced):
        # [FUNCTION DOC] Tests sparse times dense multiplication over many shapes:
        # commutativity, correctness against the dense product, in-place results for dense and
        # sparse operands when dimension counts allow, scalar multiplication (bool, int, float,
        # and as 0-dim sparse tensors), an uncoalesced 0-dim sparse scalar, and broadcasting of
        # the sparse over the dense and the dense over the sparse, plus empty cases. Expected to
        # fail on MPS.
        self._test_mul_skips(device, dtype, coalesced)

        shape = (2, 3, 4, 10)
        nnz = 10

        def check(self, s, d):
            # [FUNCTION DOC] HELPER: Checks that the product of dense and sparse commutes,
            # equals the dense product after densifying, and that the in-place multiplications
            # (dense by sparse when the dense dim count is at least the sparse one, sparse by
            # dense when the reverse) match the dense in-place results. Takes the test case as
            # an explicit first argument.
            res = d * s

            # check commutativity
            self.assertEqual(res, s * d)

            # check correctness
            self.assertEqual(res.to_dense(), s.to_dense() * d)

            # check in-placeness for dense
            if d.dim() >= s.dim():
                dc = d.clone()
                self.assertEqual(d.mul_(s), dc.mul_(s.to_dense()))

            # check in-placeness for sparse
            if s.dim() >= d.dim():
                # for sparse
                sc = s.clone()
                self.assertEqual(s.mul_(d).to_dense(), sc.to_dense().mul_(d))

        for dim in range(len(shape) + 1):
            sub_shape = shape[dim:]
            sparse_dim = len(sub_shape) // 2

            def check_empty(sparse_shape, nnz, dense_shape, coalesce):
                # [FUNCTION DOC] HELPER: For nnz and zero nnz, and an empty and a nonempty extra
                # trailing dim, generates a sparse tensor and a random dense tensor of the
                # matching shapes and runs the sparse-dense product check. Uses the sparse dim
                # count from the enclosing loop iteration.
                from itertools import product
                for nnz_val, shape_suffix in product((nnz, 0), ((), (0,))):
                    empty_sparse_shape = sparse_shape + shape_suffix
                    empty_dense_shape = dense_shape + shape_suffix
                    s = self._gen_sparse(sparse_dim, nnz_val, empty_sparse_shape, dtype, device, coalesce)[0]
                    d = make_tensor(empty_dense_shape, dtype=dtype, device=device)
                    check(self, s, d)

            # check scalar multiplication
            s = self._gen_sparse(sparse_dim, nnz, sub_shape, dtype, device, coalesced)[0]
            for scalar in (True, 1, 1.0):
                res_sparse_right = s * scalar
                res_sparse_left = scalar * s
                res_dense = s.to_dense() * scalar
                # check correctness and dtype
                self.assertEqual(s.to(res_sparse_right.dtype), res_sparse_right)
                self.assertEqual(res_sparse_right, res_sparse_left)
                self.assertEqual(res_sparse_right.dtype, res_dense.dtype)
                self.assertEqual(res_sparse_left.dtype, res_dense.dtype)
                # check scalar as 0-dim sparse tensor
                tscalar = torch.tensor(scalar, device=device)
                sscalar = tscalar.to_sparse()
                res_sparse_right = s * sscalar
                res_sparse_left = sscalar * s
                self.assertEqual(res_sparse_right, res_sparse_left)
                self.assertEqual(s.to(res_sparse_right.dtype), res_sparse_right)

            # check non-coalesced 0-dim scalar
            # we skip torch.bool because for such tensors
            # coalesce.to_dense != to_dense
            if dtype == torch.bool:
                return

            for scalar_dtype in (int, float):
                scalar = scalar_dtype(1)
                idx = torch.tensor([], device=device).reshape(0, 2)
                val = torch.tensor([scalar, scalar], device=device)
                sscalar = torch.sparse_coo_tensor(idx, val, ())
                res_dense = s.to_dense() * sscalar.to_dense()
                self.assertEqual((s * sscalar).to_dense(), res_dense)
                self.assertEqual((sscalar * s).to_dense(), res_dense)

            # Case 1: sparse broadcasts over dense
            s = self._gen_sparse(sparse_dim, nnz, sub_shape, dtype, device, coalesced)[0]
            d = make_tensor(shape, dtype=dtype, device=device)
            check(self, s, d)
            check_empty(sub_shape, nnz, shape, coalesced)

            # Case 2: dense broadcasts over sparse
            s = self._gen_sparse(3, nnz, shape, dtype, device, coalesced)[0]
            d = make_tensor(sub_shape, dtype=dtype, device=device)
            check(self, s, d)
            check_empty(shape, nnz, sub_shape, coalesced)

    @unittest.skipIf(not TEST_NUMPY, "NumPy is not available")
    @onlyCPU
    @dtypes(*all_types_and_complex_and(torch.bool))
    def test_sparse_spdiags(self, device, dtype):
        # [FUNCTION DOC] CPU test (needs NumPy) for torch.sparse.spdiags against a reference
        # (scipy when available, otherwise a manual diagonal fill): many valid cases including
        # zero-size shapes, non-contiguous diags and offsets, empty diag sets, rotation of upper
        # diagonals, vector diags, scalar and out of range offsets and CSR or CSC output
        # layouts, plus invalid inputs that must raise RuntimeError with specific messages
        # (wrong shape rank, offsets rank, diags rank, count mismatch, unsupported layout,
        # duplicate offsets, non-long offsets).

        make_diags = functools.partial(make_tensor, dtype=dtype, device=device)
        make_offsets = functools.partial(torch.tensor, dtype=torch.long, device=device)

        if TEST_SCIPY:
            def reference(diags, offsets, shape):
                # [FUNCTION DOC] HELPER: Reference for spdiags using scipy: builds the dense
                # array for the diagonals, offsets and shape.
                return scipy.sparse.spdiags(diags, offsets, *shape).toarray()

        else:
            def reference(diags, offsets, shape):
                # [FUNCTION DOC] HELPER: Fallback reference for spdiags when scipy is not
                # available: for each offset copies the corresponding row of diags (dropping the
                # first offset elements when positive) into the matching diagonal of a zero
                # matrix, limited by the diagonal length.
                result = torch.zeros(shape, dtype=dtype, device=device)
                for i, off in enumerate(offsets):
                    res_view = result.diagonal(off)
                    data = diags[i]
                    if off > 0:
                        data = data[off:]

                    m = min(res_view.shape[0], data.shape[0])
                    res_view[:m] = data[:m]
                return result

        def check_valid(diags, offsets, shape, layout=None):
            # [FUNCTION DOC] HELPER: Calls spdiags with the layout, asserts the layout is the
            # expected (COO by default) and that the densified output equals the reference.
            ref_out = reference(diags, offsets, shape)
            out = torch.sparse.spdiags(diags, offsets, shape, layout=layout)
            if layout is None:
                ex_layout = torch.sparse_coo
            else:
                ex_layout = layout
            out_dense = out.to_dense()
            self.assertTrue(out.layout == ex_layout, lambda msg: f"{msg}\nOutput layout {out.layout} expected {ex_layout}")
            self.assertEqual(out_dense, ref_out, lambda msg: f"{msg}\nResult:\n{out_dense} does not match reference:\n{ref_out}")

        def check_invalid(args, error):
            # [FUNCTION DOC] HELPER: Asserts calling spdiags with the argument tuple raises
            # RuntimeError matching the error pattern.
            with self.assertRaisesRegex(RuntimeError, error):
                torch.sparse.spdiags(*args)

        def valid_cases():
            # [FUNCTION DOC] HELPER: Generator of tuples of diags, offsets, shape and optional
            # layout that spdiags must handle correctly.
            # some normal cases
            yield (make_diags((1, 5)), make_offsets([0]), (5, 5))
            yield (make_diags((3, 3)), make_offsets([-1, 0, 1]), (4, 4))
            # zero-dimension shapes (regression test for issue #171505)
            yield (make_diags((1, 3)), make_offsets([0]), (0, 4))
            yield (make_diags((1, 3)), make_offsets([0]), (4, 0))
            yield (make_diags((1, 3)), make_offsets([0]), (0, 0))
            yield (make_diags((1, 3)), make_offsets([0]), (0, 4), torch.sparse_csr)
            yield (make_diags((1, 3)), make_offsets([0]), (0, 4), torch.sparse_csc)
            # non-contiguous diags
            yield (make_diags((5, 4), noncontiguous=True), make_offsets([-1, 1, 0, 2, -2]), (5, 5))
            # non-contiguous offsets
            yield (make_diags((3, 4)), make_offsets([1, -1, 0, -2, 2])[::2], (5, 5))
            # non-contiguous diags + offsets
            yield (make_diags((3, 4), noncontiguous=True), make_offsets([1, -1, 0, -2, 2])[::2], (5, 5))
            # correct dimensionality, 2d, 2d , and shapes match, but the number of diagonals is zero
            yield (make_diags((0, 3)), make_offsets([]), (3, 3))
            # forward rotation of upper diagonals
            yield (make_diags((3, 8)), make_offsets([1, 2, 3]), (4, 4))
            # rotation exausts input space to read from
            yield (make_diags((2, 3)), make_offsets([2, 1]), (3, 3))
            # Simple cases repeated with special output format
            yield (make_diags((1, 5)), make_offsets([0]), (5, 5), torch.sparse_csc)
            yield (make_diags((3, 3)), make_offsets([-1, 0, 1]), (4, 4), torch.sparse_csr)
            # vector diags
            yield (make_diags((3, )), make_offsets([1]), (4, 4))
            # Scalar offset
            yield (make_diags((1, 3)), make_offsets(2), (4, 4))
            # offsets out of range
            yield (make_diags((1, 3)), make_offsets([3]), (3, 3))
            yield (make_diags((1, 3)), make_offsets([-3]), (3, 3))

        for case in valid_cases():
            check_valid(*case)

        def invalid_cases():
            # [FUNCTION DOC] HELPER: Generator of pairs of invalid argument tuples and the error
            # regular expression that spdiags must raise.
            yield (make_diags((1, 3)), make_offsets([0]), (3, 2, 3)), "Output shape must be 2d"
            yield (make_diags((2, 3)), make_offsets([[1, 2], [0, 3]]), (3, 3)), "Offsets must be scalar or vector"
            yield (make_diags((3, 2, 3)), make_offsets([0, 1, 2]), (4, 4)), "Diagonals must be vector or matrix"
            yield (make_diags((3, 3)), make_offsets([-1, 0]), (3, 3)), \
                r"Number of diagonals \(\d\) does not match the number of offsets \(\d\)"
            yield (make_diags((5,)), make_offsets([0, 1, 2, 3, 4]), (3, 3)), \
                r"Number of diagonals \(\d\) does not match the number of offsets \(\d\)"
            yield (make_diags((2, 2)), make_offsets([-1, 0]), (2, 3), torch.strided), \
                r"Only output layouts \(\w+, \w+, \w+\) are supported, got \w+"
            yield (make_diags((2, 5)), make_offsets([0, 0]), (5, 5)), "Offset tensor contains duplicate values"
            yield (make_diags((1, 5)), make_offsets([0]).to(torch.int32), (5, 5)), r"Offset Tensor must have dtype Long but got \w+"


        for case, error_regex in invalid_cases():
            check_invalid(case, error_regex)

    def test_small_nnz_coalesced(self):
        # [FUNCTION DOC] Checks the coalesced flag after creation: a COO tensor with zero or one
        # nnz is coalesced, but with two or more nnz it is not flagged coalesced even when
        # indices are unique.
        # creating a coo tensor with nnz == 0 is always coalesced
        self.assertTrue(torch.sparse_coo_tensor([[], []], [], (2, 2)).is_coalesced())
        # same for a coo tensor with only 1 nnz
        self.assertTrue(torch.sparse_coo_tensor([[0], [0]], [1], (2, 2)).is_coalesced())
        # two or more nnz coalesced is false as it can't be verified without an expensive check
        self.assertFalse(torch.sparse_coo_tensor([[0, 0], [0, 0]], [1, 2], (2, 2)).is_coalesced())
        # even if there are no duplicates
        self.assertFalse(torch.sparse_coo_tensor([[0, 1], [0, 1]], [1, 2], (2, 2)).is_coalesced())

    @coalescedonoff
    @dtypes(*all_types_and_complex_and(torch.bool))
    @dtypesIfMPS(*all_mps_types())
    def test_sum(self, device, dtype, coalesced):
        # [FUNCTION DOC] Checks that sum of a sparse tensor equals the sum of its values, and
        # for float and complex types that backward of the sum's absolute value gives a gradient
        # equal to ones times the sign of the sum, for zero, some and all entries nnz.
        def run_test(shape, nnz):
            # [FUNCTION DOC] HELPER: Generates a sparse matrix with the given nnz and shape and
            # runs the sum and gradient checks described above.
            a = self._gen_sparse(2, nnz, shape, dtype, device, coalesced)[0]
            self.assertEqual(a.sum(), a._values().sum())
            if dtype.is_floating_point or dtype.is_complex:
                a.requires_grad_(True)
                a_inter = a.sum()
                a_inter.abs().backward()
                with torch.no_grad():
                    self.assertEqual(a.grad, torch.ones(shape, dtype=dtype, device=device) * torch.sgn(a_inter))
        for shape in [(10, 5), (10, 10)]:
            run_test(shape, 0)
            run_test(shape, max(shape))
            run_test(shape, shape[0] * shape[1])


class TestSparseOneOff(TestCase):
    # [CLASS DOC] Small one-off tests for CUDA and CPU mixing with sparse tensors that need no
    # device-type instantiation.
    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_cuda_from_cpu(self):
        # [FUNCTION DOC] Skipped without CUDA: checks that building a sparse tensor from CUDA
        # indices and CPU values raises RuntimeError saying all tensors must be on the same
        # device, for plain, hybrid zero-size dense and fully empty cases.
        with self.assertRaisesRegex(
                RuntimeError,
                "Expected all tensors to be on the same device"):
            torch.sparse_coo_tensor(torch.zeros(1, 4).long().cuda(),
                                    torch.randn(4, 4, 4),
                                    [3, 4, 4])

        with self.assertRaisesRegex(
                RuntimeError,
                "Expected all tensors to be on the same device"):
            torch.sparse_coo_tensor(torch.zeros(1, 4).long().cuda(),
                                    torch.randn(4, 4, 4, 0),
                                    [3, 4, 4, 0])

        with self.assertRaisesRegex(
                RuntimeError,
                "Expected all tensors to be on the same device"):
            torch.sparse_coo_tensor(torch.empty(1, 0).long().cuda(),
                                    torch.randn(0, 4, 4, 0),
                                    [0, 4, 4, 0])

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_cuda_sparse_cpu_dense_add(self):
        # [FUNCTION DOC] Skipped without CUDA: checks that adding a sparse CUDA tensor to a
        # dense CPU tensor raises RuntimeError saying the first argument should be a CUDA
        # tensor, for plain, trailing zero-size and fully empty shapes.
        x = torch.zeros(3, 4, 4)
        sparse_y = torch.sparse_coo_tensor(torch.zeros(1, 4).long().cuda(),
                                           torch.randn(4, 4, 4).cuda(),
                                           [3, 4, 4])
        with self.assertRaisesRegex(RuntimeError, "add: expected 'self' to be a CUDA tensor, but got a CPU tensor"):
            x + sparse_y

        x = torch.zeros(3, 4, 4, 0)
        sparse_y = torch.sparse_coo_tensor(torch.zeros(1, 4).long().cuda(),
                                           torch.randn(4, 4, 4, 0).cuda(),
                                           [3, 4, 4, 0])
        with self.assertRaisesRegex(RuntimeError, "add: expected 'self' to be a CUDA tensor, but got a CPU tensor"):
            x + sparse_y

        x = torch.zeros(0, 4, 4, 0)
        sparse_y = torch.sparse_coo_tensor(torch.empty(1, 0).long().cuda(),
                                           torch.randn(0, 4, 4, 0).cuda(),
                                           [0, 4, 4, 0])
        with self.assertRaisesRegex(RuntimeError, "add: expected 'self' to be a CUDA tensor, but got a CPU tensor"):
            x + sparse_y


def _sparse_to_dense(tensor):
    # [FUNCTION DOC] HELPER: Returns the dense version of a sparse tensor. For non-bool it uses
    # to_dense with masked gradient; for bool it converts via int8, since coalesce is not
    # implemented for bool.
    if tensor.dtype != torch.bool:
        return tensor.to_dense(masked_grad=True)

    # to_dense uses coalesce which isn't implemented for bool
    return tensor.to(torch.int8).to_dense().to(torch.bool)


_sparse_unary_ops = ops(mps_ops_modifier(sparse_unary_ufuncs, sparse=True), dtypes=OpDTypes.supported,
                        allowed_dtypes=all_types_and_complex())
class TestSparseUnaryUfuncs(TestCase):
    # [CLASS DOC] OpInfo based tests of sparse COO support for the unary ufuncs in
    # sparse_unary_ufuncs, instantiated over supported dtypes through the shared ops decorator
    # built at module level. exact_dtype is set so dtype mismatches fail. Covers consistency
    # with dense, out and in-place variants, empty tensors and gradcheck.
    exact_dtype = True


    @_sparse_unary_ops
    def test_sparse_consistency(self, device, dtype, op):
        # [FUNCTION DOC] Takes the first sample of the op, applies the op to the dense input and
        # to its sparse form, and asserts the densified sparse result equals the dense result;
        # raises AssertionError if the sample input or either result is not a tensor.
        sample = first_sample(self, op.sample_inputs(device, dtype))
        if not isinstance(sample.input, torch.Tensor):
            raise AssertionError(f"expected sample.input to be Tensor, got {type(sample.input)}")

        expected = op(sample.input, *sample.args, **sample.kwargs)
        if not torch.is_tensor(expected):
            raise AssertionError(f"expected tensor, got {type(expected)}")
        output = op(sample.input.to_sparse(), *sample.args, **sample.kwargs)
        if not torch.is_tensor(output):
            raise AssertionError(f"expected tensor, got {type(output)}")
        self.assertEqual(_sparse_to_dense(output), expected)

    @_sparse_unary_ops
    def test_out(self, device, dtype, op):
        # [FUNCTION DOC] Skips if the op has no out variant. Otherwise applies the op to the
        # first sample converted to sparse, then calls it again with a sparse COO out tensor of
        # the result dtype and asserts out equals the expected result.
        if not op.supports_out:
            self.skipTest("Skipped! Out not supported")

        sample = first_sample(self, op.sample_inputs(device, dtype))
        sample.input = sample.input.to_sparse()
        expect = op(sample.input, *sample.args, **sample.kwargs)

        out = torch.sparse_coo_tensor(sample.input.shape, device=device,
                                      dtype=expect.dtype)
        op(sample.input, *sample.args, **sample.kwargs, out=out)
        self.assertEqual(out, expect)

    @_sparse_unary_ops
    def test_inplace(self, device, dtype, op):
        # [FUNCTION DOC] Skips if the op has no in-place variant. Uses the first sample as a
        # coalesced sparse tensor; if the result dtype cannot be cast to the input dtype it
        # asserts RuntimeError about result type casting; otherwise asserts the in-place call
        # returns the same object and that it equals the out of place result.
        if op.inplace_variant is None:
            self.skipTest("Skipped! Out not supported")

        sample = first_sample(self, op.sample_inputs(device, dtype))
        sample.input = sample.input.to_sparse().coalesce()
        expect = op(sample.input, *sample.args, **sample.kwargs)

        if not torch.can_cast(expect.dtype, dtype):
            with self.assertRaisesRegex(RuntimeError, "result type .* can't be cast to"):
                op.inplace_variant(sample.input, *sample.args, **sample.kwargs)
            return

        actual = op.inplace_variant(sample.input, *sample.args, **sample.kwargs)
        self.assertIs(actual, sample.input)
        self.assertEqual(actual, expect)

    @_sparse_unary_ops
    def test_sparse_zero_dims(self, device, dtype, op):
        # [FUNCTION DOC] Checks the op on a 0 by 0 sparse COO tensor: the result must equal a
        # sparse tensor with the same indices and the op applied to the (empty) values.
        # test 0x0 sparse_coo_tensor
        indices = torch.empty(2, 0, dtype=torch.int64, device=device)
        values = torch.empty(0, dtype=dtype, device=device)
        sparse_0x0 = torch.sparse_coo_tensor(indices, values, (0, 0))
        expected = torch.sparse_coo_tensor(indices, op(values), (0, 0))
        actual = op(sparse_0x0)
        self.assertEqual(expected, actual)

    @_sparse_unary_ops
    def test_sparse_zeros(self, device, dtype, op):
        # [FUNCTION DOC] Checks the op on a scalar-shaped sparse tensor with no entries: the
        # densified result equals the op applied to a dense scalar zero. The sample inputs are
        # fetched but not used.
        samples = op.sample_inputs(device, dtype)

        zero_input = torch.zeros((), device=device, dtype=dtype)
        sparse_input = torch.sparse_coo_tensor((), dtype=dtype, device=device)

        expect = op(zero_input)
        actual = op(sparse_input)
        self.assertEqual(expect, _sparse_to_dense(actual))

    @ops(sparse_unary_ufuncs, dtypes=OpDTypes.supported,
         allowed_dtypes=[torch.double, torch.cdouble])
    def test_sparse_fn_grad(self, device, dtype, op):
        # [FUNCTION DOC] Autograd test for double and complex double: for each sample, converts
        # the input to a detached sparse tensor requiring grad and gradchecks the op with masked
        # semantics (honoring the op's nondeterminism tolerance and fast mode), asserting
        # gradcheck passes. Skipped if the op lacks autograd support.
        if not op.supports_autograd:
            self.skipTest("Skipped! Op doesn't support autograd")

        for sample in op.sample_inputs(device, dtype):
            sparse_input = sample.input.to_sparse().detach().requires_grad_(True)

            def fn(x):
                # [FUNCTION DOC] HELPER: Function under gradcheck: applies the op with the
                # sample arguments and returns the densified result.
                return _sparse_to_dense(
                    op(x, *sample.args, **sample.kwargs))

            self.assertTrue(gradcheck(
                fn,
                (sparse_input,),
                check_batched_grad=False,
                check_grad_dtypes=True,
                nondet_tol=op.gradcheck_nondet_tol,
                fast_mode=op.gradcheck_fast_mode,
                masked=True))


class TestSparseMaskedReductions(TestCase):
    # [CLASS DOC] Tests masked reductions (torch.masked ops in sparse_masked_reduction_ops)
    # on sparse COO inputs. fp16_low_precision_list names ops that need looser tolerance for
    # half precision.
    exact_dtype = True

    fp16_low_precision_list = {
        'masked.prod',
    }

    @ops(sparse_masked_reduction_ops)
    def test_future_empty_dim(self, device, dtype, op):
        # [FUNCTION DOC] Checks that, for samples whose dim is 0, calling the masked reduction
        # on the sparse input with an empty tuple of dims (meaning reduce over all dims
        # currently) returns a sparse COO result equal to the dense reduction converted to
        # sparse. Skips samples without dim 0 and, for prod, amax and amin without a mask
        # (unsupported for sparse COO). Uses looser tolerances for half precision on ops in the
        # low precision list.
        """Currently, `dim=()` in reductions operations means "reduce over
        all dimensions" while in future, it will read "no reduce". See
        https://github.com/pytorch/pytorch/issues/29137

        For sparse masked reductions, we'll implement the current behavior.

        For testing, we'll use samples with `dim=0` and map it to
        `dim=()` until
        torch.testing._internal.common_methods_invocations._generate_reduction_kwargs
        is made to generate samples with `dim=()` for non-scalar
        inputs. With this and after gh-29137 is resolved, this test
        can be deleted. See also `torch.masked._canonical_dim`
        implementation about changing the `dim=()` behavior.
        """

        samples = op.sample_inputs_func(op, device, dtype, requires_grad=False)
        op_name = op.name.replace('masked.', '')
        for sample_input in samples:
            if sample_input.kwargs.get('dim') != 0:
                continue
            sample_input_kwargs = dict(sample_input.kwargs)
            sample_input_kwargs['dim'] = ()    # reduce over all dimensions

            t = sample_input.input
            mask = sample_input_kwargs.get('mask')
            if mask is None and op_name in {'prod', 'amax', 'amin'}:
                # FIXME: for now reductions with non-zero reduction identity and
                # unspecified mask are not supported for sparse COO
                # tensors, see torch.masked.prod implementation
                # for details.
                continue
            sparse_op_kwargs = dict(sample_input_kwargs)
            actual = op(t.to_sparse(), *sample_input.args, **sample_input_kwargs)
            self.assertEqual(actual.layout, torch.sparse_coo)

            expected = op(t, *sample_input.args, **sample_input_kwargs).to_sparse()
            atol = None
            rtol = None
            if op.name in self.fp16_low_precision_list and dtype == torch.half:
                atol = 1e-5
                rtol = 2e-3
            self.assertEqual(actual, expected, atol=atol, rtol=rtol)


class TestSparseMeta(TestCase):
    # [CLASS DOC] Tests sparse tensors on the meta device and fake tensors: construction,
    # printing, conversion to meta, zeros_like, sum and add, for all sparse layouts. exact_dtype
    # is set. Uses the generate_simple_inputs helper provided by the common TestCase base class.
    exact_dtype = True

    def _test_meta_sparse_coo(self, dtype):
        # [FUNCTION DOC] HELPER: Creates sparse COO tensors on the meta device through several
        # factories and checks they are meta, the resize and clear methods give expected sparse
        # and dense dims and nnz zero, the coalesced flag is true for empty and can be forced
        # false and true again, and indices and values (private and public) are empty meta
        # tensors of the right shapes and dtypes.
        r = torch.empty(4, 4, layout=torch.sparse_coo, device='meta', dtype=dtype)
        self.assertTrue(r.is_meta)
        self.assertEqual(r.device.type, "meta")
        r2 = torch.empty_like(r)
        self.assertTrue(r2.is_meta)
        self.assertEqual(r, r2)
        r3 = torch.sparse_coo_tensor(size=(4, 4), device='meta', dtype=dtype)
        self.assertTrue(r3.is_meta)
        self.assertEqual(r, r3)
        r.sparse_resize_((4, 4), 1, 1)
        r.sparse_resize_and_clear_((4, 4, 4), 2, 1)
        self.assertEqual(r.sparse_dim(), 2)
        self.assertEqual(r.dense_dim(), 1)
        self.assertEqual(r._dimV(), 1)
        self.assertEqual(r._nnz(), 0)
        # nnz zero sparse tensors should always be coalesced at creation
        self.assertEqual(r.is_coalesced(), True)
        # but we can force them into the uncoalesed state
        r._coalesced_(False)
        self.assertEqual(r.is_coalesced(), False)
        # return the coalesced state for indices/values access
        r._coalesced_(True)
        # TODO: this sort of aliasing will need to be handled by
        # functionalization
        self.assertEqual(r._indices(), torch.empty(2, 0, device='meta', dtype=torch.int64))
        self.assertEqual(r._values(), torch.empty(0, 4, device='meta', dtype=dtype))
        self.assertEqual(r.indices(), torch.empty(2, 0, device='meta', dtype=torch.int64))
        self.assertEqual(r.values(), torch.empty(0, 4, device='meta', dtype=dtype))

    def _test_meta_sparse_compressed(self, dtype, layout, batchsize, densesize):
        # [FUNCTION DOC] HELPER: Builds an empty compressed-layout (CSR, CSC, BSR or BSC) meta
        # tensor with the given batch and dense sizes, asserting it is meta with expected sparse
        # and dense dims, nnz zero, block size, compressed and plain indices and values shapes,
        # that empty_like keeps it meta and equal, and, for CSR and CSC without dense dims, that
        # torch.empty with that layout on the meta device gives an equal tensor.
        index_dtype = torch.int64
        blocksize = (2, 3) if layout in {torch.sparse_bsr, torch.sparse_bsc} else ()
        sparsesize = (4, 6)
        nnz = 0

        shape = (*batchsize, *sparsesize, *densesize)
        compressed_dim = 0 if layout in {torch.sparse_csr, torch.sparse_bsr} else 1
        nof_compressed_indices = (sparsesize[compressed_dim] // blocksize[compressed_dim] + 1 if blocksize
                                  else sparsesize[compressed_dim] + 1)
        compressed_indices = torch.empty((*batchsize, nof_compressed_indices), device='meta', dtype=index_dtype)
        plain_indices = torch.empty((*batchsize, nnz), device='meta', dtype=index_dtype)

        values = torch.empty((*batchsize, nnz, *blocksize, *densesize), device='meta', dtype=dtype)
        r = torch.sparse_compressed_tensor(
            compressed_indices,
            plain_indices,
            values,
            shape,
            layout=layout
        )
        self.assertTrue(r.is_meta)
        self.assertEqual(r.device.type, "meta")

        self.assertEqual(r.sparse_dim(), 2)
        self.assertEqual(r.dense_dim(), len(densesize))
        self.assertEqual(r._nnz(), nnz)
        batch_dims = r.ndim - r.sparse_dim() - r.dense_dim()
        r_blocksize = r.values().shape[batch_dims + 1: batch_dims + 1 + len(blocksize)]
        self.assertEqual(r_blocksize, blocksize)

        r_compressed_indices = r.crow_indices() if layout in {torch.sparse_csr, torch.sparse_bsr} else r.ccol_indices()
        r_plain_indices = r.col_indices() if layout in {torch.sparse_csr, torch.sparse_bsr} else r.row_indices()

        self.assertEqual(r_compressed_indices,
                         torch.empty((*batchsize, nof_compressed_indices), device='meta', dtype=index_dtype))
        self.assertEqual(r_plain_indices, torch.empty((*batchsize, nnz), device='meta', dtype=index_dtype))
        self.assertEqual(r.values(), torch.empty((*batchsize, nnz, *blocksize, *densesize), device='meta', dtype=dtype))

        r2 = torch.empty_like(r)
        self.assertTrue(r2.is_meta)
        self.assertEqual(r2, r)

        if layout in {torch.sparse_csr, torch.sparse_csc}:
            r3 = torch.empty((*batchsize, *sparsesize), dtype=dtype, layout=layout, device="meta")
            self.assertTrue(r3.is_meta)
            if not densesize:
                # dense dimensions cannot be specified for torch.empty
                self.assertEqual(r3, r)

    @all_sparse_layouts('layout', include_strided=False)
    @parametrize("dtype", [torch.float64])
    def test_meta(self, dtype, layout):
        # [FUNCTION DOC] For one layout: runs the COO meta checks for COO, or the compressed
        # meta checks over all combinations of no batch or one batch dim and no dense or one
        # dense dim.
        if layout is torch.sparse_coo:
            self._test_meta_sparse_coo(dtype)
        else:
            for batchsize, densesize in itertools.product([(), (2,)], [(), (3,)]):
                self._test_meta_sparse_compressed(dtype, layout, batchsize, densesize)

    def _test_print_meta_data(self, dtype, layout, batchsize, sparsesize, densesize):
        # [FUNCTION DOC] HELPER: Builds an empty meta sparse tensor (COO or compressed layout)
        # for the given sizes and returns a list of text lines: a header line describing dtype,
        # index dtype and sizes, a label line and the printed repr of the tensor.
        index_dtype = torch.int64
        nnz = 0
        blocksize = (2, 3) if layout in {torch.sparse_bsr, torch.sparse_bsc} else ()
        shape = (*batchsize, *sparsesize, *densesize)
        values = torch.empty((*batchsize, nnz, *blocksize, *densesize), device='meta', dtype=dtype)
        if layout is torch.sparse_coo:
            indices = torch.empty((len(sparsesize), nnz), device='meta', dtype=index_dtype)
            x = torch.sparse_coo_tensor(indices, values, shape)
        else:
            compressed_dim = 0 if layout in {torch.sparse_csr, torch.sparse_bsr} else 1
            nof_compressed_indices = (sparsesize[compressed_dim] // blocksize[compressed_dim] + 1 if blocksize
                                      else sparsesize[compressed_dim] + 1)
            compressed_indices = torch.empty((*batchsize, nof_compressed_indices), device='meta', dtype=index_dtype)
            plain_indices = torch.empty((*batchsize, nnz), device='meta', dtype=index_dtype)
            x = torch.sparse_compressed_tensor(
                compressed_indices,
                plain_indices,
                values,
                shape,
                layout=layout
            )

        printed = []
        printed.append(f"########## {dtype}/{index_dtype}/size={batchsize}+{sparsesize}+{blocksize}+{densesize} ##########")
        printed.append("# sparse meta tensor")
        printed.append(str(x))

        return printed

    @all_sparse_layouts('layout', include_strided=False)
    @parametrize("dtype", [torch.float64])
    def test_print_meta(self, dtype, layout):
        # [FUNCTION DOC] Concatenates the printed meta tensor descriptions over combinations of
        # batch, sparse and dense sizes valid for the layout (COO has no batch dims; compressed
        # layouts need exactly two sparse dims) and compares them with the stored expected
        # output, with the unlimited diff size temporarily set and restored.
        printed = []
        for batchsize, sparsesize, densesize in itertools.product(
                [(), (2,)], [(4, 6), (3, 5, 7)], [(), (3,)]
        ):
            if layout is torch.sparse_coo and batchsize:
                # COO tensors don't have batch dimensions
                continue
            if layout is not torch.sparse_coo and len(sparsesize) != 2:
                # CSR/CSC/BSR/BSC tensors must have 2 sparse dimensions
                continue
            printed += self._test_print_meta_data(dtype, layout, batchsize, sparsesize, densesize)

        orig_maxDiff = self.maxDiff
        self.maxDiff = None
        try:
            self.assertExpected('\n'.join(printed))
            self.maxDiff = orig_maxDiff
        except Exception:
            self.maxDiff = orig_maxDiff
            raise

    def assertEqualMeta(self, x, y, expected_nnz):
        # [FUNCTION DOC] Asserts two sparse tensors have the same layout, shape, dtype, sparse
        # and dense dims, and that their indices and values components have the same dtype,
        # layout and shape with the number of stored elements replaced by the expected nnz;
        # device is compared only when the first is not meta.
        self.assertEqual(x.layout, y.layout)
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(x.dtype, y.dtype)
        self.assertEqual(x.sparse_dim(), y.sparse_dim())
        self.assertEqual(x.dense_dim(), y.dense_dim())

        def assertEqualAttrs(x, y, expected_shape):
            # [FUNCTION DOC] HELPER: Asserts x has the expected shape and the same dtype and
            # layout as y, and the same device when x is not on the meta device.
            self.assertEqual(x.shape, expected_shape)
            self.assertEqual(x.dtype, y.dtype)
            self.assertEqual(x.layout, y.layout)
            if not x.is_meta:
                self.assertEqual(x.device, y.device)

        if x.layout is torch.sparse_coo:
            assertEqualAttrs(x._indices(), y._indices(), (*y._indices().shape[:-1], expected_nnz))
            assertEqualAttrs(x._values(), y._values(), (expected_nnz, *y._values().shape[1:]))
        elif x.layout in {torch.sparse_csr, torch.sparse_bsr}:
            assertEqualAttrs(x.crow_indices(), y.crow_indices(), y.crow_indices().shape)
            assertEqualAttrs(x.col_indices(), y.col_indices(), (*y.col_indices().shape[:-1], expected_nnz))
            batch_dim = x.col_indices().ndim - 1
            values_shape = (*y.values().shape[:batch_dim], expected_nnz, *y.values().shape[batch_dim + 1:])
            self.assertEqual(x.values().layout, y.values().layout)
            self.assertEqual(x.values().dtype, y.values().dtype)
            self.assertEqual(x.values().shape, values_shape)
        elif x.layout in {torch.sparse_csc, torch.sparse_bsc}:
            assertEqualAttrs(x.ccol_indices(), y.ccol_indices(), y.ccol_indices().shape)
            assertEqualAttrs(x.row_indices(), y.row_indices(), (*y.row_indices().shape[:-1], expected_nnz))
            batch_dim = x.row_indices().ndim - 1
            values_shape = (*y.values().shape[:batch_dim], expected_nnz, *y.values().shape[batch_dim + 1:])
            self.assertEqual(x.values().layout, y.values().layout)
            self.assertEqual(x.values().dtype, y.values().dtype)
            self.assertEqual(x.values().shape, values_shape)

    @all_sparse_layouts('layout', include_strided=False)
    @parametrize("dtype", [torch.float64])
    def test_to_meta(self, dtype, layout):
        # [FUNCTION DOC] For each simple sparse input of the layout on CPU, moves it to the
        # meta device and checks the device is meta and the metadata matches the original with
        # nnz zero.
        index_dtype = torch.int64
        device = 'cpu'
        for t in self.generate_simple_inputs(layout, device=device, dtype=dtype, index_dtype=index_dtype):
            m = t.to(device="meta")
            self.assertEqual(m.device.type, "meta")
            self.assertEqualMeta(m, t, 0)

    @all_sparse_layouts('layout', include_strided=False)
    @parametrize("dtype", [torch.float64])
    def test_zeros_like_meta(self, dtype, layout):
        # [FUNCTION DOC] For each simple sparse input, zeros_like on the meta device must be
        # meta and match the input metadata with nnz zero.
        index_dtype = torch.int64
        device = 'cpu'
        for t in self.generate_simple_inputs(layout, device=device, dtype=dtype, index_dtype=index_dtype):
            m = torch.zeros_like(t, device="meta")
            self.assertEqual(m.device.type, "meta")
            self.assertEqualMeta(m, t, 0)

    @all_sparse_layouts('layout', include_strided=False)
    @parametrize("dtype", [torch.float64])
    def test_fake(self, dtype, layout):
        # [FUNCTION DOC] Converts each simple sparse input into a FakeTensor and checks it is a
        # FakeTensor with matching metadata and nnz zero, also after detach.
        from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
        fake_mode = FakeTensorMode()
        index_dtype = torch.int64
        device = 'cpu'
        for t in self.generate_simple_inputs(layout, device=device, dtype=dtype, index_dtype=index_dtype):
            f = FakeTensor.from_tensor(t, fake_mode)
            self.assertIsInstance(f, FakeTensor)
            self.assertEqualMeta(f, t, 0)

            d = f.detach()
            self.assertIsInstance(d, FakeTensor)
            self.assertEqualMeta(d, t, 0)

    @all_sparse_layouts('layout', include_strided=False)
    @parametrize("dtype", [torch.float64])
    def test_zeros_like_fake(self, dtype, layout):
        # [FUNCTION DOC] Checks zeros_like of a fake tensor (run without dispatch, using the
        # fake device) equals zeros_like of the real tensor and has matching metadata with
        # nnz zero.
        from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
        from torch.utils._mode_utils import no_dispatch
        fake_mode = FakeTensorMode()
        index_dtype = torch.int64
        device = 'cpu'
        for t in self.generate_simple_inputs(layout, device=device, dtype=dtype, index_dtype=index_dtype):
            f = FakeTensor.from_tensor(t, fake_mode)
            expected = torch.zeros_like(t)
            with no_dispatch():
                result = torch.zeros_like(f, device=f.fake_device)
            self.assertEqual(result, expected)
            self.assertEqualMeta(result, expected, 0)

    @all_sparse_layouts('layout', include_strided=False)
    @parametrize("dtype", [torch.float64])
    def test_sum_meta(self, dtype, layout):
        # [FUNCTION DOC] Checks sum of a meta sparse tensor is meta and its metadata equals that
        # of the real sum moved to meta.
        device = 'cpu'
        index_dtype = torch.int64
        for t in self.generate_simple_inputs(layout, device=device, dtype=dtype, index_dtype=index_dtype):
            m = t.to(device='meta')
            r = torch.sum(m)
            expected = torch.sum(t).to(device="meta")
            self.assertTrue(r.is_meta)
            self.assertEqualMeta(r, expected, 0)

    @all_sparse_layouts('layout', include_strided=False)
    @parametrize("dtype", [torch.float64])
    def test_add_meta(self, dtype, layout):
        # [FUNCTION DOC] Checks adding two meta sparse tensors gives metadata equal to the real
        # sum moved to meta.
        device = 'cpu'
        index_dtype = torch.int64
        for t in self.generate_simple_inputs(layout, device=device, dtype=dtype, index_dtype=index_dtype):
            expected = torch.add(t, t).to(device='meta')
            m = t.to(device='meta')
            r = torch.add(m, m)
            self.assertEqualMeta(r, expected, 0)


class _SparseDataset(torch.utils.data.Dataset):
    # [CLASS DOC] HELPER: Minimal map-style dataset wrapping a list of sparse tensors, used
    # by TestSparseAny.test_dataloader.
    # An utility class used in TestSparseAny.test_dataloader method.

    def __init__(self, sparse_tensors):
        # [FUNCTION DOC] HELPER: Stores the list of sparse tensors.
        self.sparse_tensors = sparse_tensors

    def __len__(self):
        # [FUNCTION DOC] HELPER: Returns the number of stored tensors.
        return len(self.sparse_tensors)

    def __getitem__(self, index):
        # [FUNCTION DOC] HELPER: Returns the stored tensor at the given index.
        return self.sparse_tensors[index]


class TestSparseAny(TestCase):
    # [CLASS DOC] Layout-generic sparse tests across COO, CSR, CSC, BSR and BSC: invariant
    # checking, input generation, constructors with autograd, conversions, reductions, binary
    # operations, like-functions, sparse_mask, dataloader, pinned memory and complex views. Uses
    # the generate_simple_inputs helper provided by the common TestCase base class.

    @onlyCPU
    @all_sparse_layouts('layout', include_strided=False)
    @torch.sparse.check_sparse_tensor_invariants(enable=False)
    def test_check_sparse_tensor_invariants(self, layout):
        # [FUNCTION DOC] CPU-only test, run with sparse tensor invariant checks globally
        # disabled, for every sparse layout. Builds an invalid tensor (out of range column or
        # row index) and verifies the invariant check context manager: disabled by default so
        # construction succeeds, check_invariants true in the constructor raises RuntimeError
        # with the layout specific message, a local enabled context makes even default
        # construction raise, explicit false overrides it, state is restored after leaving
        # contexts and after exceptions, nesting of predefined context managers works, and
        # reusing an already active manager instance raises RuntimeError.

        if layout is torch.sparse_coo:

            def create_invalid_tensor(check_invariants=None):
                # [FUNCTION DOC] HELPER: Builds a COO tensor with a column index out of range,
                # passing check_invariants only if not None, and returns it (raises if checking
                # is enabled).
                shape = (2, 2)
                invalid_indices = torch.tensor([[0], [3]])  # column index is out of range
                values = torch.tensor([1])
                if check_invariants is None:
                    return torch.sparse_coo_tensor(invalid_indices, values, shape)
                else:
                    return torch.sparse_coo_tensor(invalid_indices, values, shape, check_invariants=check_invariants)

            expected_exception_message = 'size is inconsistent with indices: for dim 1, size is 2 but found index 3'

        elif layout in {torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}:

            def create_invalid_tensor(check_invariants=None):
                # [FUNCTION DOC] HELPER: Builds a compressed-layout tensor of the layout under
                # test with a plain index out of range (block values for BSR and BSC), passing
                # check_invariants only if not None, and returns it.
                shape = (2, 2)
                compressed_indices = torch.tensor([0, 0, 1])
                invalid_plain_indices = torch.tensor([3])  # index is out of range
                if layout in {torch.sparse_bsr, torch.sparse_bsc}:
                    values = torch.tensor([[[1]]])
                else:
                    values = torch.tensor([1])
                if check_invariants is None:
                    return torch.sparse_compressed_tensor(compressed_indices, invalid_plain_indices, values, shape, layout=layout)
                else:
                    return torch.sparse_compressed_tensor(compressed_indices, invalid_plain_indices, values, shape, layout=layout,
                                                          check_invariants=check_invariants)

            if layout in {torch.sparse_csr, torch.sparse_bsr}:
                expected_exception_message = r'`0 <= col_indices < ncols` is not satisfied.'
            else:
                expected_exception_message = r'`0 <= row_indices < nrows` is not satisfied.'

        else:
            raise NotImplementedError(layout)

        # First, consider the case where invariant checks are disabled
        # "globally" (read: within the context of this test method
        # caller) as defined by check_sparse_tensor_invariants(False)
        # decorator:
        self.assertFalse(torch.sparse.check_sparse_tensor_invariants.is_enabled())

        # Enable the invariant checks in a local context:
        with torch.sparse.check_sparse_tensor_invariants():
            self.assertTrue(torch.sparse.check_sparse_tensor_invariants.is_enabled())

        # Leaving the local context must restore the "global" state of
        # the invariant check feature:
        self.assertFalse(torch.sparse.check_sparse_tensor_invariants.is_enabled())

        # Since invariant checks are disabled by default, we can
        # create an invalid sparse tensor without raising an
        # exception:
        r = create_invalid_tensor()
        self.assertEqual(r.layout, layout)

        # Or, when disabling the invariants check explicitly:
        r = create_invalid_tensor(check_invariants=False)
        self.assertEqual(r.layout, layout)

        # Enabling invariant check via constructor's optional argument
        # will raise an exception when sparse tensor invariants are
        # violated:
        with self.assertRaisesRegex(RuntimeError, expected_exception_message):
            create_invalid_tensor(check_invariants=True)

        # Check that the global invariant check flag has been restored
        # after raising the exception above:
        self.assertFalse(torch.sparse.check_sparse_tensor_invariants.is_enabled())

        # Next, consider the case where invariant checks are enabled
        # within a local context:
        with torch.sparse.check_sparse_tensor_invariants():
            self.assertTrue(torch.sparse.check_sparse_tensor_invariants.is_enabled())

            # Since invariant checks are now enabled by default, an
            # attempt to create an invalid sparse tensor will lead to
            # an exception:
            with self.assertRaisesRegex(RuntimeError, expected_exception_message):
                create_invalid_tensor()

            # Similarly, when enabling the invariant checks
            # explicitly, invalid sparse tensor construction will lead
            # to an exception:
            with self.assertRaisesRegex(RuntimeError, expected_exception_message):
                create_invalid_tensor(check_invariants=True)

            # However, invariants check can be disabled via
            # constructor's optional argument so that the invalid
            # tensor is successfully constructed:
            r = create_invalid_tensor(check_invariants=False)
            self.assertEqual(r.layout, layout)

            # Check that the invariant check flag has been restored
            # when leaving the constructor:
            self.assertTrue(torch.sparse.check_sparse_tensor_invariants.is_enabled())

        # Double-check restoring the global state when leaving the
        # local context:
        self.assertFalse(torch.sparse.check_sparse_tensor_invariants.is_enabled())

        # Test nesting of pre-defined context managers
        check_ctx = torch.sparse.check_sparse_tensor_invariants(True)
        no_check_ctx = torch.sparse.check_sparse_tensor_invariants(False)
        with check_ctx:
            self.assertTrue(torch.sparse.check_sparse_tensor_invariants.is_enabled())
            with no_check_ctx:
                self.assertFalse(torch.sparse.check_sparse_tensor_invariants.is_enabled())
            self.assertTrue(torch.sparse.check_sparse_tensor_invariants.is_enabled())
        self.assertFalse(torch.sparse.check_sparse_tensor_invariants.is_enabled())

        # Test an attempt to reuse an activate context manager instance
        check_ctx2 = torch.sparse.check_sparse_tensor_invariants(True)
        with check_ctx:
            self.assertTrue(torch.sparse.check_sparse_tensor_invariants.is_enabled())
            with no_check_ctx:
                self.assertFalse(torch.sparse.check_sparse_tensor_invariants.is_enabled())
                with self.assertRaisesRegex(RuntimeError, "This context manager instance is already activated."
                                            " Use a different context manager instance for context nesting"):
                    with check_ctx:
                        self.assertTrue(torch.sparse.check_sparse_tensor_invariants.is_enabled())
                self.assertFalse(torch.sparse.check_sparse_tensor_invariants.is_enabled())
                with check_ctx2:
                    self.assertTrue(torch.sparse.check_sparse_tensor_invariants.is_enabled())
                self.assertFalse(torch.sparse.check_sparse_tensor_invariants.is_enabled())
            self.assertTrue(torch.sparse.check_sparse_tensor_invariants.is_enabled())
        self.assertFalse(torch.sparse.check_sparse_tensor_invariants.is_enabled())

    def test_generate_simple_inputs(self):
        # [FUNCTION DOC] Checks the generate_simple_inputs generator: zipping the generators for
        # all layouts gives semantically equal tensors with the requested layouts, and records
        # the combination of layout, hybrid, batch, nontrivial block size and index and value
        # contiguity of each; raises AssertionError listing any combination of those properties
        # that was never produced.
        layouts = [torch.strided, torch.sparse_coo, torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc]

        tested_combinations = set()
        for tensors in zip(*map(self.generate_simple_inputs, layouts)):
            for i, t in enumerate(tensors):
                self.assertEqual(t.layout, layouts[i])

                # all layouts must produce semantically the same tensors
                self.assertEqual(t, tensors[0])

                if t.layout is torch.strided:
                    is_hybrid = None
                else:
                    is_hybrid = t.dense_dim() > 0
                if t.layout in {torch.sparse_csr, torch.sparse_bsr}:
                    is_batch = t.crow_indices().ndim > 1
                elif t.layout in {torch.sparse_csc, torch.sparse_bsc}:
                    is_batch = t.ccol_indices().ndim > 1
                else:
                    is_batch = None
                if t.layout in {torch.sparse_bsr, torch.sparse_bsc}:
                    blocksize = t.values().shape[1:3]
                    nontrivial_blocksize = 1 not in blocksize
                else:
                    nontrivial_blocksize = None
                if t.layout in {torch.sparse_csr, torch.sparse_bsr}:
                    contiguous_indices = t.crow_indices().is_contiguous() and t.col_indices().is_contiguous()
                    contiguous_values = t.values().is_contiguous()
                elif t.layout in {torch.sparse_csc, torch.sparse_bsc}:
                    contiguous_indices = t.ccol_indices().is_contiguous() and t.row_indices().is_contiguous()
                    contiguous_values = t.values().is_contiguous()
                elif t.layout is torch.sparse_coo:
                    contiguous_indices = t._indices().is_contiguous()
                    contiguous_values = t._values().is_contiguous()
                else:
                    contiguous_indices = None
                    contiguous_values = t.is_contiguous()

                tested_combinations.add((t.layout, is_hybrid, is_batch, nontrivial_blocksize,
                                         contiguous_indices, contiguous_values))

        # Ensure that the inputs generation covers all layout,
        # non-hybrid/hybrid, non-batch/batch, and contiguity
        # combinations:
        untested_combinations = set()
        for layout in layouts:
            for is_hybrid in [False, True]:
                if layout is torch.strided:
                    is_hybrid = None
                for is_batch in [False, True]:
                    if layout in {torch.sparse_coo, torch.strided}:
                        is_batch = None
                    for nontrivial_blocksize in [False, True]:
                        if layout not in {torch.sparse_bsr, torch.sparse_bsc}:
                            nontrivial_blocksize = None
                        for contiguous_indices in [False, True]:
                            if layout is torch.strided:
                                contiguous_indices = None
                            elif not is_batch:
                                # indices are contiguous per-patch
                                contiguous_indices = True
                            for contiguous_values in [False, True]:
                                key = (layout, is_hybrid, is_batch, nontrivial_blocksize,
                                       contiguous_indices, contiguous_values)
                                if key not in tested_combinations:
                                    untested_combinations.add(
                                        f'layout={layout}, is_hybrid={is_hybrid}, is_batch={is_batch},'
                                        f' nontrivial_blocksize={nontrivial_blocksize},'
                                        f' contiguous_indices{contiguous_indices}, contiguous_values={contiguous_values}')
        if untested_combinations:
            raise AssertionError(f"untested combinations: {untested_combinations}")

    @all_sparse_layouts('layout', include_strided=False)
    def test_constructor_autograd(self, device, layout):
        # [FUNCTION DOC] Gradchecks sparse constructors (layout specific and generic, only the
        # specific one for COO) for generated simple inputs without batch dims: the result
        # requires grad exactly when the values do, and when it does, the constructors with and
        # without explicit size pass gradcheck with both masked settings.

        def specific_constructor(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Calls the constructor matching the layout under test (COO,
            # CSR, CSC, BSC or BSR) with the given arguments; raises NotImplementedError for
            # other layouts.
            if layout is torch.sparse_csr:
                return torch.sparse_csr_tensor(*args, **kwargs)
            elif layout is torch.sparse_csc:
                return torch.sparse_csc_tensor(*args, **kwargs)
            elif layout is torch.sparse_bsc:
                return torch.sparse_bsc_tensor(*args, **kwargs)
            elif layout is torch.sparse_bsr:
                return torch.sparse_bsr_tensor(*args, **kwargs)
            elif layout is torch.sparse_coo:
                return torch.sparse_coo_tensor(*args, **kwargs)
            else:
                raise NotImplementedError(layout)

        def generic_constructor(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Calls the generic sparse_compressed_tensor constructor with
            # the layout keyword for compressed layouts, or the COO constructor for COO; raises
            # NotImplementedError otherwise.
            if layout in {torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}:
                kwargs.update(layout=layout)
                return torch.sparse_compressed_tensor(*args, **kwargs)
            elif layout is torch.sparse_coo:
                return torch.sparse_coo_tensor(*args, **kwargs)
            else:
                raise NotImplementedError(layout)

        if layout is torch.sparse_coo:
            constructors = (specific_constructor,)
        else:
            constructors = (specific_constructor, generic_constructor)

        for args, kwargs in self.generate_simple_inputs(
                layout, device=device, dtype=torch.float64,
                enable_batch=False,  # TODO: remove after gh-104868 is resolved
                output_tensor=False):
            values_offset = 1 if layout is torch.sparse_coo else 2

            for cnstr in constructors:
                for requires_grad in (False, True):
                    values = args[values_offset].detach().requires_grad_(requires_grad)
                    args = (*args[:values_offset], values, *args[values_offset + 1:])
                    kwargs_ = dict(kwargs)
                    args_ = args + (kwargs_.pop('size'),)

                    sparse = cnstr(*args, **kwargs)

                    self.assertEqual(sparse.requires_grad, requires_grad)

                    if requires_grad:
                        for masked in (False, True):
                            if layout is torch.sparse_coo:
                                torch.autograd.gradcheck(
                                    lambda i, v: cnstr(i, v, **kwargs).to_dense(masked_grad=masked),
                                    args, masked=masked)
                                torch.autograd.gradcheck(
                                    lambda i, v, sz: cnstr(i, v, sz, **kwargs_).to_dense(masked_grad=masked),
                                    args_, masked=masked)
                            else:
                                torch.autograd.gradcheck(
                                    lambda ci, pi, v: cnstr(ci, pi, v, **kwargs).to_dense(masked_grad=masked),
                                    args, masked=masked)
                                torch.autograd.gradcheck(
                                    lambda ci, pi, v, sz: cnstr(ci, pi, v, sz, **kwargs_).to_dense(masked_grad=masked),
                                    args_, masked=masked)

    @all_sparse_layouts('from_layout', include_strided=False)
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @parametrize("index_dtype", [torch.int32, torch.int64])
    def test_to_dense(self, from_layout, device, dtype, index_dtype):
        # [FUNCTION DOC] Checks conversion from each sparse layout to strided via to_dense for
        # simple generated inputs: the result is strided and equals the input.
        """
        This test tests conversion from any layout to strided layout.
        """
        for t in self.generate_simple_inputs(
                from_layout, device=device, dtype=dtype, index_dtype=index_dtype):
            r = t.to_dense()
            self.assertEqual(r.layout, torch.strided)
            self.assertEqual(r, t)

    @all_sparse_layouts('from_layout', include_strided=False)
    @dtypes(torch.float64, torch.complex128)
    @parametrize("index_dtype", [torch.int64])
    @gradcheck_semantics()
    def test_gradcheck_to_dense(self, from_layout, device, dtype, index_dtype, gradcheck):
        # [FUNCTION DOC] Gradchecks to_dense for sparse layouts in double and complex double
        # with the masked setting of the gradcheck variant; inputs with batch dimensions are
        # skipped as not supported yet.
        for t in self.generate_simple_inputs(
                from_layout, device=device, dtype=dtype, index_dtype=index_dtype):
            batch_dim = t.dim() - t.dense_dim() - t.sparse_dim()
            if batch_dim > 0:
                # TODO: implement batch support in _convert_indices_from_csr_to_coo
                continue
            t = t.detach().clone().requires_grad_(True)
            r = gradcheck(lambda x: torch.Tensor.to_dense(x, masked_grad=gradcheck.masked), t)
            self.assertTrue(r)

    @all_sparse_layouts('from_layout', include_strided=True)
    @all_sparse_layouts('to_layout', include_strided=False)
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @parametrize("index_dtype", [torch.int32, torch.int64])
    def test_to_sparse(self, from_layout, to_layout, device, dtype, index_dtype):
        # [FUNCTION DOC] Checks conversion from every layout (strided included) to every sparse
        # layout, for generated inputs. Computes the block size for block layouts, verifies
        # expected RuntimeError for unsupported batched compressed to block conversions, COO
        # with sparse_dim other than two to compressed layouts, and block to non-block
        # compressed conversions (then skips the test as not implemented). For supported
        # conversions validates the result arguments and index and value dtypes, preserves
        # coalesced state for COO, compares equality with the input, checks consistency with the
        # explicit conversion methods, and round trips block formats. Also includes two
        # regression cases for gh-90910 and gh-91007.
        """
        This test tests conversion from any layout to any sparse layout.
        """
        for t in self.generate_simple_inputs(
                from_layout, device=device, dtype=dtype, index_dtype=index_dtype,
                enable_hybrid=(
                    # TODO: to support conversion strided->hybrid
                    # CSR/CSC/BSR/BSC, to_sparse() requires extra keyword
                    # argument, either nof_batch_dims or
                    # nof_dense_dims
                    not (from_layout is torch.strided and to_layout in
                         {torch.sparse_bsr, torch.sparse_bsc, torch.sparse_csr, torch.sparse_csc}))):

            if to_layout in {torch.sparse_bsr, torch.sparse_bsc}:
                if from_layout == torch.sparse_bsr:
                    batch_ndim = t.crow_indices().dim() - 1
                    blocksize = t.values().shape[batch_ndim + 1:batch_ndim + 3]
                elif from_layout == torch.sparse_bsc:
                    batch_ndim = t.ccol_indices().dim() - 1
                    blocksize = t.values().shape[batch_ndim + 1:batch_ndim + 3]
                else:
                    blocksize = (1, 1)
            else:
                blocksize = None

            if from_layout is torch.strided:
                is_batch = None
                is_hybrid = None
            else:
                is_batch = t.dim() > (t.sparse_dim() + t.dense_dim())
                is_hybrid = t.dense_dim() > 0

            def explicit_to_sparse(x):
                # [FUNCTION DOC] HELPER: Converts using the explicit method matching the target
                # layout (COO, CSR, CSC, BSR or BSC with the block size); raises AssertionError
                # for any other layout.
                # Used to check that the explicit conversion methods
                # are consistent with the `to_sparse(*, layout,
                # blocksize)` method.
                if to_layout is torch.sparse_coo:
                    return x.to_sparse_coo()
                elif to_layout is torch.sparse_csr:
                    return x.to_sparse_csr()
                elif to_layout is torch.sparse_csc:
                    return x.to_sparse_csc()
                elif to_layout is torch.sparse_bsr:
                    return x.to_sparse_bsr(blocksize)
                elif to_layout is torch.sparse_bsc:
                    return x.to_sparse_bsc(blocksize)
                else:
                    raise AssertionError(f"unreachable: to_layout={to_layout}")

            # TODO: The following exception cases all correspond to
            # not implemented conversions
            if from_layout in {
                    torch.sparse_csr, torch.sparse_csc} and to_layout in {torch.sparse_bsr, torch.sparse_bsc} and is_batch:
                with self.assertRaisesRegex(
                        RuntimeError,
                        r"conversion from Sparse(Csr|Csc) to Sparse(Bsr|Bsc) for batched inputs is not supported"):
                    t.to_sparse(layout=to_layout, blocksize=blocksize)
                with self.assertRaisesRegex(
                        RuntimeError,
                        r"conversion from Sparse(Csr|Csc) to Sparse(Bsr|Bsc) for batched inputs is not supported"):
                    explicit_to_sparse(t)
                continue
            elif from_layout is torch.sparse_coo and to_layout in {
                    torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc} and t.sparse_dim() != 2:
                with self.assertRaisesRegex(
                        RuntimeError,
                        r"conversion from Sparse to .* for input tensors with sparse_dim\(\)!=2 is not supported"):
                    t.to_sparse(layout=to_layout, blocksize=blocksize)
                with self.assertRaisesRegex(
                        RuntimeError,
                        r"conversion from Sparse to .* for input tensors with sparse_dim\(\)!=2 is not supported"):
                    explicit_to_sparse(t)
                continue
            elif (from_layout, to_layout) in {(torch.sparse_bsc, torch.sparse_csr), (torch.sparse_bsc, torch.sparse_csc),
                                              (torch.sparse_bsr, torch.sparse_csr), (torch.sparse_bsr, torch.sparse_csc)}:
                with self.assertRaisesRegex(
                        RuntimeError,
                        r"sparse_compressed_to_sparse_(csr|csc|bsr|bsc): expected\s*(Sparse(Csc|Csr)[,]|)\s*Sparse(Csr|Bsr)"
                        " or Sparse(Csc|Bsc) layout but got Sparse(Csr|Csc|Bsr|Bsc)"):
                    t.to_sparse(layout=to_layout, blocksize=blocksize)
                with self.assertRaisesRegex(
                        RuntimeError,
                        r"sparse_compressed_to_sparse_(csr|csc|bsr|bsc): expected\s*(Sparse(Csc|Csr)[,]|)\s*Sparse(Csr|Bsr)"
                        " or Sparse(Csc|Bsc) layout but got Sparse(Csr|Csc|Bsr|Bsc)"):
                    explicit_to_sparse(t)
                self.skipTest('NOT IMPL')
            else:
                r = t.to_sparse(layout=to_layout, blocksize=blocksize)

                self.assertEqual(r.layout, to_layout)

                # to_sparse method uses unsafe construction of sparse
                # tensors. Here we explicitly validate the results to
                # make sure that the sparse tensors are consistent
                # with the corresponding sparse tensor invariants.
                if r.layout in {torch.sparse_csr, torch.sparse_bsr, torch.sparse_csc, torch.sparse_bsc}:
                    if r.layout in {torch.sparse_csr, torch.sparse_bsr}:
                        compressed_indices, plain_indices = r.crow_indices(), r.col_indices()
                    else:
                        compressed_indices, plain_indices = r.ccol_indices(), r.row_indices()
                    torch._validate_sparse_compressed_tensor_args(compressed_indices, plain_indices, r.values(),
                                                                  r.shape, r.layout)
                    if from_layout in {torch.strided, torch.sparse_coo}:
                        self.assertEqual(compressed_indices.dtype, torch.int64)
                        self.assertEqual(plain_indices.dtype, torch.int64)
                    else:
                        self.assertEqual(compressed_indices.dtype, index_dtype)
                        self.assertEqual(plain_indices.dtype, index_dtype)
                    self.assertEqual(r.values().dtype, dtype)
                elif r.layout is torch.sparse_coo:
                    if t.layout is torch.sparse_coo:
                        self.assertEqual(t.is_coalesced(), r.is_coalesced())

                    # Check r is truly coalesced when r.is_coalesced == True
                    if r.is_coalesced():
                        self.assertTrue(is_coalesced_indices(r))

                    torch._validate_sparse_coo_tensor_args(r._indices(), r._values(), r.shape)
                    self.assertEqual(r._indices().dtype, torch.int64)
                    self.assertEqual(r._values().dtype, dtype)
                else:
                    raise AssertionError("unreachable")

                # Finally, we'll test tensor equality:
                self.assertEqual(r, t)

                # Also, check consistency with explicit conversion methods:
                r2 = explicit_to_sparse(t)
                self.assertEqual(r2, r)

                # Check inverse conversion from sparse compressed block tensors
                if from_layout == torch.sparse_bsr:
                    batch_ndim = t.crow_indices().dim() - 1
                    from_blocksize = t.values().shape[batch_ndim + 1:batch_ndim + 3]
                elif from_layout == torch.sparse_bsc:
                    batch_ndim = t.ccol_indices().dim() - 1
                    from_blocksize = t.values().shape[batch_ndim + 1:batch_ndim + 3]
                else:
                    continue
                if r.ndim != 2:
                    continue

                t2 = r.to_sparse(layout=from_layout, blocksize=from_blocksize)
                self.assertEqual(t2, t)

        # extra tests
        if (from_layout, to_layout) == (torch.sparse_csr, torch.sparse_bsr):
            # See gh-90910
            t = torch.tensor([[0, 0, 1, 0], [0, 1, 0, 0]], dtype=dtype, device=device).to_sparse_csr()
            r = t.to_sparse_bsr((2, 2))
            torch._validate_sparse_compressed_tensor_args(r.crow_indices(), r.col_indices(), r.values(), r.shape, r.layout)
            self.assertEqual(r, t)

        if (from_layout, to_layout) in {(torch.sparse_csr, torch.sparse_csc),
                                        (torch.sparse_csc, torch.sparse_csr)}:
            # See gh-91007
            compressed_indices = torch.tensor([0, 4, 8, 8, 12, 16, 20], dtype=index_dtype, device=device)
            plain_indices = torch.tensor([0, 1, 2, 3] * 5, dtype=index_dtype, device=device)
            t = torch.sparse_compressed_tensor(compressed_indices, plain_indices, range(20),
                                               dtype=dtype, device=device, layout=from_layout)
            r = t.to_sparse(layout=to_layout)
            if r.layout in {torch.sparse_csr, torch.sparse_bsr}:
                compressed_indices, plain_indices = r.crow_indices(), r.col_indices()
            else:
                compressed_indices, plain_indices = r.ccol_indices(), r.row_indices()
            torch._validate_sparse_compressed_tensor_args(compressed_indices, plain_indices, r.values(), r.shape, r.layout)
            self.assertEqual(r, t)

    @onlyNativeDeviceTypes
    @suppress_warnings
    @ops(reduction_ops_with_sparse_support)
    @precisionOverride({torch.bfloat16: 5e-4, torch.float16: 5e-3})
    @all_sparse_layouts('layout', include_strided=False)
    def test_reductions(self, layout, device, dtype, op):
        # [FUNCTION DOC] For each sample the op's sparse reduction result for the given layout
        # must equal the same reduction applied to the densified input. Skips the test if the op
        # yields no sparse samples. Native device types only.
        count = 0
        for sample in op.sample_inputs_sparse(layout, device, dtype):
            count += 1

            t_inp, t_args, t_kwargs = sample.input, sample.args, sample.kwargs
            result = op.op(t_inp, *t_args, **t_kwargs)

            #  Checking invariant rop(inp, ...).to_dense() == rop(inp.to_dense(), ...)
            dense = op.op(t_inp.to_dense(), *t_args, **t_kwargs)
            self.assertEqual(result, dense)

        if count == 0:
            # we count samples to avoid false-positive test reports
            self.skipTest('no sample inputs')

    @onlyNativeDeviceTypes
    @suppress_warnings
    @ops(reduction_ops_with_sparse_support, allowed_dtypes=(torch.float32, torch.float64, torch.complex64, torch.complex128))
    @all_sparse_layouts('layout', include_strided=False)
    def test_reductions_backward(self, layout, device, dtype, op):
        # [FUNCTION DOC] For floating and complex dtypes, evaluates the reduction on samples
        # requiring grad and, only for the sum op, backpropagates the absolute value of the
        # (summed) result and checks the input gradient equals ones times the sign of the
        # result; other ops skip the test as not implemented, as does having no samples.
        count = 0
        for sample in op.sample_inputs_sparse(layout, device, dtype, requires_grad=True):
            t_inp, t_args, t_kwargs = sample.input, sample.args, sample.kwargs
            r = op.op(t_inp, *t_args, **t_kwargs)
            if r.numel() != 0:
                r = r.sum()

            if op.name == 'sum':
                count += 1
                r.abs().backward()
                self.assertEqual(t_inp.grad, torch.ones(t_inp.shape, dtype=dtype, device=device) * torch.sgn(r))
            else:
                self.skipTest('NOT IMPL')

        if count == 0:
            # we count samples to avoid false-positive test reports
            self.skipTest('no sample inputs')

    @onlyNativeDeviceTypes
    @suppress_warnings
    @parametrize("mth", [subtest(mth, name=mth.__name__)
                         for mth in [torch.Tensor.is_coalesced,
                                     torch.Tensor.coalesce,
                                     torch.Tensor.indices,
                                     torch.Tensor.values,
                                     torch.Tensor.crow_indices,
                                     torch.Tensor.col_indices,
                                     torch.Tensor.ccol_indices,
                                     torch.Tensor.row_indices,
                                     ]])
    @all_sparse_layouts('layout', include_strided=True)
    def test_unsupported_backend_error_message(self, mth, layout, device):
        # [FUNCTION DOC] Checks that layout specific accessors (is_coalesced, coalesce, indices,
        # values, crow_indices, col_indices, ccol_indices, row_indices) work on supported
        # layouts and raise RuntimeError with the matching message on every other layout,
        # strided included, using an input of a 2 by 2 matrix converted to the layout (with
        # block size one for block layouts).
        inp = torch.tensor([[1, 2], [3, 4]], device=device).to_sparse(
            layout=layout,
            blocksize=(1, 1) if layout in {torch.sparse_bsr, torch.sparse_bsc} else None)
        if inp.layout is not layout:
            raise AssertionError(f"expected layout {layout}, got {inp.layout}")

        expected_behaviour = dict(
            # <mth name> = (<supported layouts>, <exception message on other layouts>)
            is_coalesced=({torch.sparse_coo},
                          "is_coalesced expected sparse coordinate tensor layout but got (Sparse(Csr|Csc|Bsr|Bsc)|Strided)"),
            coalesce=({torch.sparse_coo},
                      "coalesce expected sparse coordinate tensor layout but got (Sparse(Csr|Csc|Bsr|Bsc)|Strided)"),
            indices=({torch.sparse_coo},
                     "indices expected sparse coordinate tensor layout but got (Sparse(Csr|Csc|Bsr|Bsc)|Strided)"),
            values=({torch.sparse_coo, torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc},
                    "values expected sparse tensor layout but got Strided"),
            crow_indices=({torch.sparse_csr, torch.sparse_bsr},
                          "crow_indices expected sparse row compressed tensor layout but got (Sparse(Csc|Bsc|)|Strided)"),
            col_indices=({torch.sparse_csr, torch.sparse_bsr},
                         "col_indices expected sparse row compressed tensor layout but got (Sparse(Csc|Bsc|)|Strided)"),
            ccol_indices=({torch.sparse_csc, torch.sparse_bsc},
                          "ccol_indices expected sparse column compressed tensor layout but got (Sparse(Csr|Bsr|)|Strided)"),
            row_indices=({torch.sparse_csc, torch.sparse_bsc},
                         "row_indices expected sparse column compressed tensor layout but got (Sparse(Csr|Bsr|)|Strided)"),
        )[mth.__name__]

        if layout in expected_behaviour[0]:
            mth(inp)
        else:
            with self.assertRaisesRegex(RuntimeError, expected_behaviour[1]):
                mth(inp)

    @onlyNativeDeviceTypes
    @all_sparse_layouts('layout', include_strided=not True)
    @dtypes(torch.float64, torch.cdouble)
    @parametrize("masked", [subtest(False, name='sparse'), subtest(True, name='masked')])
    @parametrize("fast_mode", [subtest(False, name='slow'), subtest(True, name='fast')])
    def test_gradcheck_mm(self, layout, dtype, device, masked, fast_mode):
        # [FUNCTION DOC] Gradchecks mm (sparse.mm when masked, else torch.mm) between a sparse
        # matrix of the layout and a dense matrix for double and complex, in slow and fast mode.
        # For BSR unmasked and BSC, expects RuntimeError saying addmm is not implemented for
        # that combination; for CSC, BSR and BSC masked expects RuntimeError about unsupported
        # layouts or missing MKL; these two branches then skip as not implemented; otherwise
        # runs the gradcheck. Batch and hybrid inputs are not covered.
        # This function does not check the following cases:
        # - batch or hybrid tensors because addmm does not support
        #   such inputs yet
        # - check_forward_ad=True because of the lack of sparse tensor
        #   support in aten::view_as_real, torch._VF._make_dual, etc.

        ref_x = torch.tensor([[1, 2, 0, 0],
                              [0, 6, 0, 0],
                              [0, 0, 0, 0],
                              [13, 14, 0, 15]], dtype=dtype, device=device)
        ref_y = torch.tensor([[11, 12, 13, 14],
                              [21, 22, 23, 24],
                              [31, 32, 33, 34],
                              [41, 42, 43, 44]],
                             dtype=dtype, device=device)

        mm = torch.sparse.mm if masked else torch.mm

        blocksize = (2, 2) if layout in {torch.sparse_bsr, torch.sparse_bsc} else None
        x = ref_x.to_sparse(layout=layout, blocksize=blocksize).requires_grad_(True)
        y = ref_y.requires_grad_(True)

        if layout is torch.sparse_bsr and not masked or layout is torch.sparse_bsc:
            with self.assertRaisesRegex(
                    RuntimeError,
                    r"addmm: computation on (CPU|CUDA) is not implemented for Strided \+ Sparse(Bsr|Bsc) @ Strided"):
                torch.autograd.gradcheck(mm, (x, y), fast_mode=fast_mode, masked=masked)
            self.skipTest('NOT IMPL')
        elif layout in {torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc} and masked:
            with self.assertRaisesRegex(
                    RuntimeError,
                    r"(sparse_addmm_sparse_backward: unsupported combination of layouts,"
                    r" grad: Strided, mat1: Sparse(Csc|Bsr|Bsc), mat2: Strided"
                    r"|addmm: computation on (CPU|CUDA) is not implemented for "
                    r"Strided \+ Sparse(Csc|Bsr|Bsc) @ Strided without MKL)"):
                torch.autograd.gradcheck(mm, (x, y), fast_mode=fast_mode, masked=masked)
            self.skipTest('NOT IMPL')
        else:
            torch.autograd.gradcheck(mm, (x, y), fast_mode=fast_mode, masked=masked)

    @onlyNativeDeviceTypes
    @suppress_warnings
    @ops(binary_ufuncs_with_sparse_support)
    @all_sparse_layouts('layout', include_strided=False)
    def test_binary_operation(self, layout, device, dtype, op):
        # [FUNCTION DOC] Tests binary ufuncs with sparse support on each sparse layout: skips
        # layouts the OpInfo does not support, skips samples rejected by the sample validator,
        # and checks the result's shape, sparse dim and dense dim equal the input's and that the
        # result equals the op applied to the densified operands (silently skipping a known
        # strided ComplexHalf limitation).
        if not op.supports_sparse_layout(layout):
            self.skipTest(f'{layout} is not supported in `{op.name}` OpInfo definition. Skipping!')

        for sample in op.sample_inputs_sparse(layout, device, dtype):
            if validate_sample_input_sparse(op, sample, check_validate=False) is not sample:
                # that is, the validation returns the sparse sample
                # wrapped within ErrorInput instance
                continue
            t_inp, t_args, t_kwargs = sample.input, sample.args, sample.kwargs
            batch_dim = t_inp.dim() - t_inp.dense_dim() - t_inp.sparse_dim()
            result = op.op(t_inp, *t_args, **t_kwargs)

            # Check rop(inp, ...).shape == inp.shape
            self.assertEqual(result.shape, t_inp.shape)

            # Check rop(inp, ...).sparse_dim() == inp.sparse_dim()
            self.assertEqual(result.sparse_dim(), t_inp.sparse_dim())

            # Check rop(inp, ...).dense_dim() == inp.dense_dim()
            self.assertEqual(result.dense_dim(), t_inp.dense_dim())

            # Check invariant rop(inp, ...).to_dense() == rop(inp.to_dense(), ...)
            try:
                dense = op.op(t_inp.to_dense(), *(t_args[0].to_dense(), *t_args[1:]), **t_kwargs)
            except Exception as msg:
                # this is strided op issue, so skipping the sample silently here
                if "\"cpublas_axpy_impl\" not implemented for 'ComplexHalf'" in str(msg):
                    continue
                raise
            self.assertEqual(result, dense)

    @onlyCPU
    @all_sparse_layouts('layout', include_strided=True)
    @dtypes(torch.double)
    def test_to_sparse_identity(self, device, layout, dtype):
        # [FUNCTION DOC] CPU-only: for identity matrices and sparse dim arguments, converting a
        # sparse tensor to sparse with the same sparse dim keeps that sparse dim, and with a
        # different sparse dim raises RuntimeError saying the conversion is not supported.
        # KNOWN ISSUE: the layout argument is never used, so the same check runs for
        # every layout.
        for dense_dim in range(4):
            x_dense = torch.eye(dense_dim, dtype=dtype, device=device)
            for sparse_dim_in in range(1, dense_dim):
                x_sparse = x_dense.to_sparse(sparse_dim_in)
                for sparse_dim_out in range(dense_dim):
                    if sparse_dim_out == sparse_dim_in:
                        self.assertTrue(x_sparse.to_sparse(sparse_dim_out).sparse_dim() == sparse_dim_out)
                    else:
                        with self.assertRaisesRegex(
                                RuntimeError,
                                r"to_sparse: conversion from Sparse to Sparse with sparse_dim argument !=self.sparse_dim\(\)"
                                " is not supported"):
                            x_sparse.to_sparse(sparse_dim_out)


    @onlyNativeDeviceTypes
    @suppress_warnings
    @ops(like_fns_with_sparse_support)
    @all_sparse_layouts('layout', include_strided=False)
    def test_like_fns(self, layout, device, dtype, op):
        # [FUNCTION DOC] Tests like-functions (OpInfos with sparse support) for each sparse
        # layout: the result must have the expected dtype, device type and layout (from kwargs
        # or input), the same block size for block layouts, the same shape, and sparse and dense
        # dims consistent with the output layout (strided gives dense dims equal to dim, COO
        # folds batch dims into sparse dims, compressed layouts keep both) with COO and
        # compressed arguments validated.

        for sample in op.sample_inputs_sparse(layout, device, dtype):
            t_inp, t_args, t_kwargs = sample.input, sample.args, sample.kwargs
            batch_dim = t_inp.dim() - t_inp.dense_dim() - t_inp.sparse_dim()
            if t_inp.layout in {torch.sparse_bsr, torch.sparse_bsc}:
                expected_blocksize = t_inp.values().shape[batch_dim + 1:batch_dim + 3]
            else:
                expected_blocksize = None
            expected_dtype = t_kwargs.get('dtype', dtype)
            expected_device = torch.device(t_kwargs.get('device', device))
            expected_layout = t_kwargs.get('layout', layout)

            result = op.op(t_inp, *t_args, **t_kwargs)

            self.assertEqual(result.dtype, expected_dtype)
            self.assertEqual(result.device.type, expected_device.type)
            self.assertEqual(result.layout, expected_layout)

            if result.layout in {torch.sparse_bsr, torch.sparse_bsc}:
                result_batch_dim = result.dim() - result.dense_dim() - result.sparse_dim()
                blocksize = result.values().shape[result_batch_dim + 1:result_batch_dim + 3]
                self.assertEqual(blocksize, expected_blocksize)

            # Check op(inp).shape == inp.shape
            self.assertEqual(result.shape, t_inp.shape)

            if expected_layout is torch.strided:
                self.assertEqual(result.sparse_dim(), 0)
                # Check op(inp, layout=torch.strided).dense_dim() == inp.dim()
                self.assertEqual(result.dense_dim(), t_inp.dim())
            elif expected_layout is torch.sparse_coo:
                # Check op(inp, layout=torch.sparse_coo).sparse_dim() == batch_dim + inp.sparse_dim()
                self.assertEqual(result.sparse_dim(), batch_dim + t_inp.sparse_dim())
                # Check op(inp, layout=torch.sparse_coo).dense_dim() == inp.dense_dim()
                self.assertEqual(result.dense_dim(), t_inp.dense_dim())

                torch._validate_sparse_coo_tensor_args(result._indices(), result._values(), result.shape)
            else:
                # Check op(inp).sparse_dim() == inp.sparse_dim()
                self.assertEqual(result.sparse_dim(), t_inp.sparse_dim())
                # Check op(inp).dense_dim() == inp.dense_dim()
                self.assertEqual(result.dense_dim(), t_inp.dense_dim())

                if result.layout in {torch.sparse_csr, torch.sparse_bsr}:
                    compressed_indices, plain_indices = result.crow_indices(), result.col_indices()
                else:
                    compressed_indices, plain_indices = result.ccol_indices(), result.row_indices()

                torch._validate_sparse_compressed_tensor_args(compressed_indices, plain_indices, result.values(),
                                                              result.shape, result.layout)

    @all_sparse_layouts('mask_layout', include_strided=False)
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_sparse_mask(self, mask_layout, device, dtype):
        # [FUNCTION DOC] Tests sparse_mask of a strided-origin sparse tensor with masks of every
        # sparse layout: the result must have the same indices as the mask, its dense form must
        # equal the input times a ones tensor with the mask pattern, and the dense form of the
        # mask masked by itself must equal the mask.
        input_layout = torch.strided
        mask_dtype = torch.bool
        for mask in self.generate_simple_inputs(mask_layout, dtype=mask_dtype, device=device,
                                                enable_hybrid=False, enable_batch=False):

            x = make_tensor(mask.shape, dtype=dtype, device=device).to_sparse(layout=input_layout)

            result = x.sparse_mask(mask)

            # Check invariant `x.sparse_mask(mask).<indices> == mask.<indices>`
            if mask_layout is torch.sparse_coo:
                self.assertEqual(result._indices(), mask._indices())
                ones = torch.sparse_coo_tensor(mask._indices(),
                                               torch.ones_like(mask._values(), dtype=x.dtype),
                                               mask.shape,
                                               is_coalesced=mask.is_coalesced())
            elif mask_layout in {torch.sparse_csr, torch.sparse_bsr}:
                self.assertEqual(result.crow_indices(), mask.crow_indices())
                self.assertEqual(result.col_indices(), mask.col_indices())
                ones = torch.sparse_compressed_tensor(mask.crow_indices(), mask.col_indices(),
                                                      torch.ones_like(mask.values(), dtype=x.dtype),
                                                      mask.shape, layout=mask.layout)
            else:
                self.assertEqual(result.ccol_indices(), mask.ccol_indices())
                self.assertEqual(result.row_indices(), mask.row_indices())
                ones = torch.sparse_compressed_tensor(mask.ccol_indices(), mask.row_indices(),
                                                      torch.ones_like(mask.values(), dtype=x.dtype),
                                                      mask.shape, layout=mask.layout)

            # Check invariant:
            #  x.sparse_mask(mask).to_dense() == x.mul(sparse_xyz_tensor(<mask indices>,
            #                                          ones_like(<mask values>)).to_dense())
            expected = x.mul(ones.to_dense())

            self.assertEqual(result.to_dense(), expected)

            # Check invariant `mask.to_dense().sparse_mask(mask) == mask`
            result = mask.to_dense().sparse_mask(mask)
            self.assertEqual(result, mask)

    @all_sparse_layouts('layout', include_strided=False)
    @parametrize("masked", [subtest(False, name='nonmasked'), subtest(True, name='masked')])
    @parametrize("fast_mode", [subtest(False, name='slow'), subtest(True, name='fast')])
    def test_as_sparse_gradcheck(self, layout, device, masked, fast_mode):
        # [FUNCTION DOC] Tests torch.sparse.as_sparse_gradcheck wrapped around gradcheck for
        # to_dense, sum, identity, to_sparse and values over generated inputs of the layout,
        # with masked and slow or fast variants. Batch inputs are only enabled for COO and
        # hybrid inputs are disabled for compressed layouts with sum or to_sparse (known
        # failures noted in the body). COO inputs are coalesced before calling values.
        gradcheck = torch.sparse.as_sparse_gradcheck(torch.autograd.gradcheck)
        sparse_compressed_layouts = {torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}

        def identity(x):
            # [FUNCTION DOC] HELPER: Returns its argument unchanged.
            return x

        for func in (torch.Tensor.to_dense,
                     torch.Tensor.sum,
                     identity,
                     torch.Tensor.to_sparse,
                     torch.Tensor.values,
                     ):
            for x in self.generate_simple_inputs(
                    layout,
                    device=device,
                    dtype=torch.float64,
                    # TODO: fix gh-104868  to enable batched samples:
                    enable_batch=layout not in sparse_compressed_layouts,
                    enable_hybrid=not (
                        layout in sparse_compressed_layouts and (
                            # FIXME: RuntimeError: sparse_mask(): the
                            # number of sparse dimensions in `self`
                            # should match that of the `mask`. Got
                            # `self.sparse_dim() == 3` !=
                            # `mask.sparse_dim() == 2
                            func.__name__ == 'sum'
                            # FIXME: RuntimeError: expected
                            # col_indices to be a contiguous tensor
                            # per batch
                            or func.__name__ == 'to_sparse'
                        ))):
                if layout is torch.sparse_coo and func.__name__ == 'values':
                    x = x.coalesce()

                gradcheck(func, x.requires_grad_(True), masked=masked, fast_mode=fast_mode)

    @onlyCPU
    @all_sparse_layouts('layout', include_strided=False)
    @dtypes(torch.double)
    def test_dataloader(self, device, layout, dtype):
        # [FUNCTION DOC] CPU-only: wraps generated sparse inputs in a dataset, loads them
        # through a DataLoader with no batching and two worker processes, and asserts the loaded
        # list equals the original.

        data = list(self.generate_simple_inputs(layout, device=device, dtype=dtype))

        dataset = _SparseDataset(data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=2)

        loaded_data = list(loader)
        self.assertEqual(data, loaded_data)

    @onlyCPU
    def test_invalid_blocksize(self):
        # [FUNCTION DOC] CPU-only: to_sparse with a block size argument that is not exactly two
        # entries (tuple, list or Size of length 1 or 3) must raise RuntimeError reporting the
        # number of entries received.
        # Blocksize should be a tuple/list/torch.Size containing two values
        with self.assertRaisesRegex(RuntimeError, ".*blocksize.*, but got 1"):
            torch.randn(1).to_sparse(blocksize=(1,))
        with self.assertRaisesRegex(RuntimeError, ".*blocksize.*, but got 1"):
            torch.randn(1).to_sparse(blocksize=[1])
        with self.assertRaisesRegex(RuntimeError, ".*blocksize.*, but got 1"):
            torch.randn(1).to_sparse(blocksize=torch.Size((1,)))
        with self.assertRaisesRegex(RuntimeError, ".*blocksize.*, but got 3"):
            torch.randn(1).to_sparse(blocksize=(1, 1, 1))
        with self.assertRaisesRegex(RuntimeError, ".*blocksize.*, but got 3"):
            torch.randn(1).to_sparse(blocksize=[1, 1, 1])
        with self.assertRaisesRegex(RuntimeError, ".*blocksize.*, but got 3"):
            torch.randn(1).to_sparse(blocksize=torch.Size((1, 1, 1)))

    @unittest.skipIf(not torch.cuda.is_available(), 'requires cuda')
    @onlyCPU
    @all_sparse_layouts('layout', include_strided=True)
    def test_constructor_pin_memory(self, device, layout):
        # [FUNCTION DOC] Skipped without CUDA, CPU-only. For generated inputs of each layout
        # (strided included, zero-size excluded) built with the pin_memory option, asserts that
        # every component (indices and values, whichever the layout has) and the tensor itself
        # report as pinned.
        """Tests sparse_xyz_tensor(indices, values, pin_memory=True)
        """
        self.assertEqual(device, "cpu")
        for t in self.generate_simple_inputs(
                layout, device=device, dtype=torch.float64,
                enable_zero_sized=False,  # pinning zero-sized tensors is a no-op
                pin_memory=True,
        ):
            if layout is torch.sparse_coo:
                self.assertTrue(t._indices().is_pinned())
                self.assertTrue(t._values().is_pinned())
            elif layout in {torch.sparse_csr, torch.sparse_bsr}:
                self.assertTrue(t.crow_indices().is_pinned())
                self.assertTrue(t.col_indices().is_pinned())
                self.assertTrue(t.values().is_pinned())
            elif layout in {torch.sparse_csc, torch.sparse_bsc}:
                self.assertTrue(t.ccol_indices().is_pinned())
                self.assertTrue(t.row_indices().is_pinned())
                self.assertTrue(t.values().is_pinned())
            elif layout is torch.strided:
                pass
            else:
                raise AssertionError(f"unreachable: layout={layout}")
            self.assertTrue(t.is_pinned())

    @unittest.skipIf(not torch.cuda.is_available(), 'requires cuda')
    @onlyCPU
    @all_sparse_layouts('layout', include_strided=True)
    def test_method_pin_memory(self, device, layout):
        # [FUNCTION DOC] Skipped without CUDA, CPU-only. For unpinned generated inputs, calling
        # pin_memory yields a pinned tensor with pinned components while the original and its
        # components stay unpinned (pinning clones), and pinning an already pinned tensor
        # returns the same object.
        """Tests sparse_xyz_tensor(indices, values, pin_memory=False).pin_memory()
        """

        for t_ in self.generate_simple_inputs(
                layout, device=device, dtype=torch.float64,
                enable_zero_sized=False,  # pinning zero-sized tensors is a no-op
                pin_memory=False,         # no pinning
        ):
            t = t_.pin_memory()
            self.assertTrue(t.is_pinned())

            # registering a non-pinned tensor with CUDA memory is a
            # clone operation
            self.assertFalse(t_.is_pinned())

            # registering already pinned tensor with CUDA memory is an
            # identity operation:
            t2 = t.pin_memory()
            self.assertTrue(t2 is t)

            if layout is torch.sparse_coo:
                self.assertTrue(t._indices().is_pinned())
                self.assertTrue(t._values().is_pinned())
                self.assertFalse(t_._indices().is_pinned())
                self.assertFalse(t_._values().is_pinned())
            elif layout in {torch.sparse_csr, torch.sparse_bsr}:
                self.assertTrue(t.crow_indices().is_pinned())
                self.assertTrue(t.col_indices().is_pinned())
                self.assertTrue(t.values().is_pinned())
                self.assertFalse(t_.crow_indices().is_pinned())
                self.assertFalse(t_.col_indices().is_pinned())
                self.assertFalse(t_.values().is_pinned())
            elif layout in {torch.sparse_csc, torch.sparse_bsc}:
                self.assertTrue(t.ccol_indices().is_pinned())
                self.assertTrue(t.row_indices().is_pinned())
                self.assertTrue(t.values().is_pinned())
                self.assertFalse(t_.ccol_indices().is_pinned())
                self.assertFalse(t_.row_indices().is_pinned())
                self.assertFalse(t_.values().is_pinned())
            elif layout is torch.strided:
                pass
            else:
                raise AssertionError(f"unreachable: layout={layout}")


    @unittest.skipIf(not torch.cuda.is_available(), 'requires cuda')
    @onlyCPU
    @all_sparse_layouts('layout', include_strided=True)
    def test_constructor_pinned_memory(self, device, layout):
        # [FUNCTION DOC] Skipped without CUDA, CPU-only. Builds tensors from indices and values
        # that were pinned individually, with the constructor pin_memory option unspecified, and
        # asserts every component and the result are pinned.
        """Tests sparse_xyz_tensor(indices.pin_memory(device), values.pin_memory(device))
        """
        pin_memory_device = "cuda"
        for t in self.generate_simple_inputs(
                layout, device=device, dtype=torch.float64,
                enable_zero_sized=False,     # pinning zero-sized tensors is a no-op
                pin_memory=None,             # constructor does not specify pin_memory=...
                members_pin_memory=True,     # indices and values are pinned
        ):
            if layout is torch.sparse_coo:
                self.assertTrue(t._indices().is_pinned())
                self.assertTrue(t._values().is_pinned())
            elif layout in {torch.sparse_csr, torch.sparse_bsr}:
                self.assertTrue(t.crow_indices().is_pinned())
                self.assertTrue(t.col_indices().is_pinned())
                self.assertTrue(t.values().is_pinned())
            elif layout in {torch.sparse_csc, torch.sparse_bsc}:
                self.assertTrue(t.ccol_indices().is_pinned())
                self.assertTrue(t.row_indices().is_pinned())
                self.assertTrue(t.values().is_pinned())
            elif layout is torch.strided:
                pass
            else:
                raise AssertionError(f"unreachable: layout={layout}")
            self.assertTrue(t.is_pinned())

    @unittest.skipIf(not torch.cuda.is_available(), 'requires cuda')
    @onlyCPU
    @all_sparse_layouts('layout', include_strided=False)
    def test_constructor_mismatched_pinned_memory(self, device, layout):
        # [FUNCTION DOC] Skipped without CUDA, CPU-only. Builds tensors with pinned indices but
        # unpinned values and vice versa, and asserts the constructor raises RuntimeError
        # saying the memory pinning of indices must match that of values, with the actual flags
        # in the message.
        """Test the failure to construct sparse tensor from indices and values
        that have different pinning states.
        """
        def generic_constructor(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Calls the generic compressed constructor with the layout
            # keyword for compressed layouts, or the COO constructor for COO; raises
            # NotImplementedError otherwise.
            if layout in {torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}:
                kwargs.update(layout=layout)
                return torch.sparse_compressed_tensor(*args, **kwargs)
            elif layout is torch.sparse_coo:
                return torch.sparse_coo_tensor(*args, **kwargs)
            else:
                raise NotImplementedError(layout)

        for args, kwargs in self.generate_simple_inputs(
                layout, device=device, dtype=torch.float64,
                enable_zero_sized=False,     # pinning zero-sized tensors is a no-op
                output_tensor=False):

            # indices are pinned, values is a non-pinned tensor
            args1 = (args[0].pin_memory(), *args[1:])

            # indices are non-pinned, values is a pinned tensor
            args2 = (*args[:-1], args[-1].pin_memory())

            with self.assertRaisesRegex(
                    RuntimeError, r"memory pinning of \w*indices \(=1\) must match memory pinning of values \(=0\)"):
                generic_constructor(*args1, **kwargs)

            with self.assertRaisesRegex(
                    RuntimeError, r"memory pinning of \w*indices \(=0\) must match memory pinning of values \(=1\)"):
                generic_constructor(*args2, **kwargs)

    @onlyNativeDeviceTypes
    @dtypes(torch.complex32, torch.complex64, torch.complex128)
    def test_view_as_real(self, device, dtype):
        # [FUNCTION DOC] For complex dtypes, view_as_real of a sparse COO tensor keeps the COO
        # layout and indices, appends a trailing dimension of two, and has values whose last
        # axis holds the real and imaginary parts; the dense form is compared except for
        # ComplexHalf on CPU; and view_as_complex of the result round trips to the original.
        for xs in self.generate_simple_inputs(torch.sparse_coo, device=device, dtype=dtype):
            res = torch.view_as_real(xs)
            self.assertEqual(res.layout, torch.sparse_coo)
            self.assertEqual(res._indices(), xs._indices())
            self.assertEqual(res.shape, xs.shape + (2,))
            self.assertEqual(res._values()[..., 0], xs._values().real)
            self.assertEqual(res._values()[..., 1], xs._values().imag)
            if not (
                dtype in (torch.complex32, torch.bcomplex32)
                and torch.device(device).type == "cpu"
            ):
                # ComplexHalf to_dense() is not supported on CPU.
                self.assertEqual(res.to_dense(), torch.view_as_real(xs.to_dense()))
            self.assertEqual(torch.view_as_complex(torch.view_as_real(xs)), xs)

    @onlyNativeDeviceTypes
    @dtypes(torch.float16, torch.float32, torch.float64)
    def test_view_as_complex(self, device, dtype):
        # [FUNCTION DOC] For float dtypes, view_as_complex of a sparse COO tensor: when it
        # raises RuntimeError the test accepts only the documented messages (last dim not two or
        # no dense dims, or a values stride issue); otherwise checks layout, indices, shape
        # without the last dim, real and imaginary parts equal the two slices of the values,
        # dense equality except float16 on CPU, and that view_as_real round trips.
        for xs in self.generate_simple_inputs(torch.sparse_coo, device=device, dtype=dtype):
            try:
                res = torch.view_as_complex(xs)
            except RuntimeError as e:
                if xs.shape[-1] != 2 or xs.dense_dim() == 0:
                    self.assertIn(
                        "view_as_complex_sparse is only supported for sparse tensors with the last dim == 2 and dense_dim > 0.",
                        str(e))
                elif xs._values().stride()[-1] != 1:
                    self.assertIn("Tensor must have a last dimension with stride 1", str(e))
                else:
                    raise
                continue
            self.assertEqual(res.layout, torch.sparse_coo)
            self.assertEqual(res._indices(), xs._indices())
            self.assertEqual(res.shape, xs.shape[:-1])
            self.assertEqual(res._values().real, xs._values()[..., 0])
            self.assertEqual(res._values().imag, xs._values()[..., 1])
            if not (dtype is torch.float16 and torch.device(device).type == "cpu"):
                # ComplexHalf to_dense() is not supported on CPU.
                self.assertEqual(res.to_dense(), torch.view_as_complex(xs.to_dense()))
            self.assertEqual(torch.view_as_real(torch.view_as_complex(xs)), xs)

# e.g., TestSparseUnaryUfuncsCPU and TestSparseUnaryUfuncsCUDA
instantiate_device_type_tests(TestSparseUnaryUfuncs, globals(), allow_mps=True, except_for='meta')

instantiate_device_type_tests(TestSparseMaskedReductions, globals(), except_for='meta')

# e.g., TestSparseCPU and TestSparseCUDA
instantiate_device_type_tests(TestSparse, globals(), allow_mps=True, except_for='meta')

instantiate_device_type_tests(TestSparseAny, globals(), except_for='meta')

instantiate_parametrized_tests(TestSparseMeta)

instantiate_parametrized_tests(TestSparseLegacyAndDeprecation)

if __name__ == '__main__':
    run_tests()
