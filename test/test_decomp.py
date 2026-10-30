# Owner(s): ["module: decompositions"]
# [MODULE DOC] Tests that operator decompositions (torch._decomp and the inductor
# decomposition table) agree numerically with the native ATen kernels. Provides a
# TorchDispatchMode that runs each decomposable op both ways and compares results (strict
# tolerance for most dtypes, or an error-versus-float64 comparison for float16/bfloat16),
# driven over OpInfo samples (forward and backward), plus one-off correctness tests and tests
# about decomposition-table coverage. Import-time side effects: builds the lists of
# decomposable OpInfos from the global tables; defines module-level tolerance tables,
# exclusion sets and expected-failure sets (the core backward set is extended when slow tests
# are not enabled); creates global counters of called and decomposed ops; and instantiates the
# device-generic test classes TestDecomp and DecompOneOffTests into this module's namespace.

import functools
import itertools
import re
import unittest
from collections import defaultdict
from functools import partial

import torch._inductor.decomposition
import torch.autograd
from torch import Tensor
from torch._decomp import (
    core_aten_decompositions,
    decomposition_table,
    get_decompositions,
)
from torch._dispatch.python import enable_python_dispatcher
from torch._export.utils import _is_cia_op
from torch._ops import DispatchKey
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import SM70OrLater, tf32_off
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
    onlyNativeDeviceTypes,
    ops,
    skip,
    skipOps,
    xfail,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_modules import module_db, modules
from torch.testing._internal.common_utils import (
    is_iterable_of_tensors,
    run_tests,
    skipIfCrossRef,
    skipIfTorchDynamo,
    suppress_warnings,
    TEST_WITH_ASAN,
    TEST_WITH_SLOW,
    TestCase,
    unMarkDynamoStrictTest,
)
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten


aten = torch.ops.aten


# TODO: this isn't going to work with non-aten namespaces
def overload_to_aten_name(op):
    # [FUNCTION DOC] HELPER: Returns the bare ATen operator name of an overload, taken from
    # the part of the schema name after the namespace separator. Only valid for names with a
    # namespace prefix (a todo notes it does not work for non-aten namespaces).
    return op._schema.name.split("::")[1]


# All operators that can have decomp tests
decomposition_names = {
    overload_to_aten_name(k)
    for k in decomposition_table
    if isinstance(k, torch._ops.OpOverload)
}
core_decomposition_names = {
    overload_to_aten_name(k)
    for k in core_aten_decompositions()
    if isinstance(k, torch._ops.OpOverload) and not _is_cia_op(k)
}
_decomp_test_ops = [
    op
    for op in op_db
    if op.aten_name in decomposition_names
    or op.aten_backward_name in decomposition_names
]
_decomp_test_ops_core_autograd = [
    op
    for op in op_db
    if op.aten_name in core_decomposition_names and op.supports_autograd
]
_sdpa_op_info = [op for op in op_db if "scaled_dot_product_attention" in op.aten_name]


def diff_arg(arg, requires_grad=True):
    # [FUNCTION DOC] HELPER: Decides whether a test argument takes part in differentiation. A
    # tensor is differentiable if it requires grad (when requires_grad is true) or if it has a
    # floating point or complex dtype (when false). For an iterable of tensors, returns true
    # if all elements qualify and false if none do, and raises RuntimeError for a mix. Any
    # non-tensor, non-iterable argument yields false.
    def is_differentiable_arg(arg):
        # [FUNCTION DOC] HELPER: Closure over requires_grad. Returns whether one tensor
        # requires grad, or, when the enclosing flag is false, whether it is floating point or
        # complex.
        if requires_grad:
            return arg.requires_grad
        else:
            return arg.is_floating_point() or arg.is_complex()

    if is_iterable_of_tensors(arg):
        if all(is_differentiable_arg(a) for a in arg):
            return True
        if all(not is_differentiable_arg(a) for a in arg):
            return False
        raise RuntimeError("NYI: The test runner can't handle this")
    return isinstance(arg, Tensor) and is_differentiable_arg(arg)


# Version of autograd.grad with some differences:
#   - pytree inputs is allowed (but leaves of the pytree have to all
#     be tensors)
#   - if an input is not used as part of derivatives, we will return a
#     zero-filled tensor for the result
def _autograd_grad(
    outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=True
):
    # [FUNCTION DOC] HELPER: Variant of torch.autograd.grad that accepts pytree inputs whose
    # leaves are all tensors. Only inputs requiring grad (and outputs requiring grad, paired
    # with their grad_outputs when given) take part; the underlying call allows unused inputs.
    # Returns the gradients restructured like inputs, with zero tensors for inputs that do not
    # require grad or were unused. create_graph defaults to true and retain_graph to false.
    inputs, inputs_spec = tree_flatten(inputs)
    diff_inputs = tuple(inp for inp in inputs if inp.requires_grad)
    if grad_outputs is None:
        diff_outputs = tuple(out for out in outputs if out.requires_grad)
    else:
        diff_grad_outputs = [
            (out, go) for out, go in zip(outputs, grad_outputs) if out.requires_grad
        ]
        if len(diff_grad_outputs) == 0:
            diff_outputs, grad_outputs = (), ()
        else:
            diff_outputs, grad_outputs = zip(*diff_grad_outputs)
    grad_inputs = torch.autograd.grad(
        diff_outputs,
        diff_inputs,
        grad_outputs,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )
    result = []
    grad_inputs_iter = iter(grad_inputs)
    for inp in inputs:
        if inp.requires_grad:
            grad_input = next(grad_inputs_iter)
            if grad_input is None:
                result.append(torch.zeros_like(inp))
            else:
                result.append(grad_input)
        else:
            result.append(torch.zeros_like(inp))
    return tree_unflatten(result, inputs_spec)


def _as_tuple(val):
    # [FUNCTION DOC] HELPER: Returns the value unchanged if it is a tuple, else wraps it in a
    # one-element tuple.
    if isinstance(val, tuple):
        return val
    return (val,)


def ref_vjp_no_create(f, *primals):
    # [FUNCTION DOC] HELPER: Reference vector-Jacobian product without graph creation. Runs f
    # on the primals and returns the result together with a function mapping cotangents to
    # input gradients, computed via _autograd_grad with create_graph false and retain_graph
    # true so it can be applied repeatedly.
    result = f(*primals)

    def wrapped(cotangents):
        # [FUNCTION DOC] HELPER: Closure returned by ref_vjp_no_create. Takes cotangents
        # (single value or tuple) and returns the gradients of the captured result with
        # respect to the captured primals, using _autograd_grad without graph creation.
        return _autograd_grad(
            _as_tuple(result),
            primals,
            _as_tuple(cotangents),
            create_graph=False,
            retain_graph=True,
        )

    return result, wrapped


dtype_precisions = {
    torch.float16: (0.001, 1e-5),
    torch.bfloat16: (0.016, 1e-4),
    torch.float32: (1.3e-6, 1e-5),
    torch.float64: (1e-7, 1e-7),
    torch.complex32: (0.001, 1e-5),
    torch.complex64: (1.3e-6, 1e-5),
    torch.complex128: (1e-7, 1e-7),
}
# Returns the "default" rtol and atol for comparing scalars or
# tensors of the given dtypes.


def _getDefaultRtolAndAtol(dtype0, dtype1):
    # [FUNCTION DOC] HELPER: Returns the default relative and absolute tolerances for
    # comparing two dtypes: the larger rtol and the larger atol among the dtype_precisions
    # entries of each dtype (dtypes missing from the table count as zero).
    rtol = max(
        dtype_precisions.get(dtype0, (0, 0))[0], dtype_precisions.get(dtype1, (0, 0))[0]
    )
    atol = max(
        dtype_precisions.get(dtype0, (0, 0))[1], dtype_precisions.get(dtype1, (0, 0))[1]
    )
    return rtol, atol


op_assert_ref_tol_table = {
    (torch.bfloat16, torch.ops.aten.native_layer_norm.default): 1e-5,
    (torch.float16, torch.ops.aten.native_layer_norm.default): 1e-5,
    (torch.float16, torch.ops.aten.native_layer_norm_backward.default): 1e-3,
    (torch.bfloat16, torch.ops.aten.native_layer_norm_backward.default): 2e-2,
    (torch.bfloat16, torch.ops.aten.native_batch_norm.default): 1e-5,
    (torch.float16, torch.ops.aten.native_batch_norm.default): 1e-5,
    (torch.bfloat16, torch.ops.aten._native_batch_norm_legit.default): 1e-5,
    (torch.bfloat16, torch.ops.aten._native_batch_norm_legit.no_stats): 1e-5,
    (torch.float16, torch.ops.aten._native_batch_norm_legit.default): 1e-5,
    (torch.float16, torch.ops.aten._native_batch_norm_legit.no_stats): 1e-5,
    (torch.bfloat16, torch.ops.aten.linalg_vector_norm.default): 1e-4,
    (torch.float16, torch.ops.aten.linalg_vector_norm.default): 1e-4,
    (torch.bfloat16, torch.ops.aten.var_mean.correction): 5e-7,
    (torch.float16, torch.ops.aten.var_mean.correction): 5e-7,
    (torch.bfloat16, torch.ops.aten.var_mean.dim): 5e-7,
    (torch.float16, torch.ops.aten.var_mean.dim): 5e-7,
    (torch.float16, torch.ops.aten.nll_loss_forward.default): 1e-2,
    (torch.bfloat16, torch.ops.aten.nll_loss_forward.default): 1e-1,
    (torch.float16, torch.ops.aten.nll_loss2d_forward.default): 1e-2,
    (torch.bfloat16, torch.ops.aten.nll_loss2d_forward.default): 2e-1,
    (torch.float16, torch.ops.aten.hardswish.default): 2e-7,
    (torch.bfloat16, torch.ops.aten.hardswish.default): 2e-7,
    (torch.float16, torch.ops.aten.multi_margin_loss.default): 3e-2,
    (torch.bfloat16, torch.ops.aten.multi_margin_loss.default): 5e-2,
    (torch.float16, torch.ops.aten.multilabel_margin_loss_forward.default): 3e-2,
    (torch.bfloat16, torch.ops.aten.multilabel_margin_loss_forward.default): 3e-2,
    (torch.float16, torch.ops.aten.reflection_pad1d_backward.default): 5e-3,
    (torch.bfloat16, torch.ops.aten.reflection_pad1d_backward.default): 5e-3,
    (torch.float16, torch.ops.aten.reflection_pad2d_backward.default): 5e-3,
    (torch.bfloat16, torch.ops.aten.reflection_pad2d_backward.default): 5e-3,
    (torch.float16, torch.ops.aten.reflection_pad3d_backward.default): 5e-3,
    (torch.bfloat16, torch.ops.aten.reflection_pad3d_backward.default): 5e-2,
    (torch.float16, torch.ops.aten._batch_norm_with_update.default): 2e-7,
    (torch.bfloat16, torch.ops.aten._batch_norm_with_update.default): 5e-7,
    # see https://github.com/pytorch/pytorch/pull/96264
    (torch.float16, torch.ops.aten.mv.default): 2e-5,
    (torch.bfloat16, torch.ops.aten.mv.default): 1e-5,
    (torch.float16, torch.ops.aten.dot.default): 2e-6,
    (torch.float16, torch.ops.aten._softmax_backward_data.default): 3e-7,
    (torch.bfloat16, torch.ops.aten._softmax_backward_data.default): 2e-7,
    # decomp for addcmul is x + y * z, but it typically compiles into an FMA for the
    # eager operator, causing a significant difference on float16
    (torch.bfloat16, torch.ops.aten.addcmul.default): 1e-5,
    (torch.float16, torch.ops.aten.addcmul.default): 1e-5,
}


def op_assert_ref(test_case, op, test_dtype, i, orig, decomp, ref, args, kwargs):
    # [FUNCTION DOC] HELPER: Checks one decomposition output against the native output using a
    # higher-precision reference. Raises AssertionError on a dtype mismatch, on a numel
    # mismatch when either output is empty (and returns when both are empty), or on a shape
    # mismatch. If the reference is floating point, computes the max absolute error of each
    # output against it and raises RuntimeError if the decomposition error exceeds the native
    # error plus a tolerance looked up by dtype and op in op_assert_ref_tol_table (default
    # 1e-7), with a message listing args and kwargs. For non-floating references it asserts
    # the two outputs are equal via the test case.
    if orig.dtype != decomp.dtype:
        raise AssertionError(
            f"{i} Operation: {op} dtype mismatch: {orig.dtype} != {decomp.dtype}"
        )
    if orig.numel() == 0 or decomp.numel() == 0:
        if orig.numel() != decomp.numel():
            raise AssertionError(f"numel mismatch: {orig.numel()} != {decomp.numel()}")
        return
    if orig.shape != decomp.shape:
        raise AssertionError(
            f"{i} Operation: {op} shape mismatch: {orig.shape} != {decomp.shape}"
        )
    if ref.is_floating_point():
        orig_diff = (orig - ref).abs().max()
        decomp_diff = (decomp - ref).abs().max()
        atol = op_assert_ref_tol_table.get((test_dtype, op), 1e-7)
        if decomp_diff > orig_diff + atol:
            raise RuntimeError(
                f"Difference from float64 is larger with decomposition {op.__name__}"
                f" than original on output {i}. Original max diff: {orig_diff}, Decomp max diff: {decomp_diff}\n"
                f"atol = {atol}\n"
                f"args = {args}\n"
                f"kwargs = {kwargs}"
            )
    else:
        test_case.assertEqual(
            orig,
            decomp,
            msg=lambda msg: f"{msg}\n{op.__name__}\nargs = {args}\nkwargs = {kwargs}",
        )


# Before adding an entry to this table, make sure your decomposition is right :)
op_assert_equal_tol_table = {
    # Due to strange epsilon behaviors, see https://github.com/pytorch/pytorch/issues/73161
    (torch.float32, torch.ops.aten.native_layer_norm.default): (1e-3, 1e-3),
    (torch.float32, torch.ops.aten.native_layer_norm_backward.default): (1e-3, 1e-3),
    (torch.float64, torch.ops.aten.native_layer_norm.default): (1e-6, 1e-6),
    # Due to strange epsilon behaviors, see https://github.com/pytorch/pytorch/issues/73161
    (torch.float32, torch.ops.aten.native_group_norm.default): (1e-4, 5e-6),
    # This exceeds default tolerances only on CPU, on CUDA it's fine
    (torch.float32, torch.ops.aten.grid_sampler_2d.default): (7e-6, 3e-5),
    # Exceeds tolerances on CUDA, likely due to fma
    (torch.float32, torch.ops.aten.mv.default): (1e-5, 3e-5),
    (torch.complex64, torch.ops.aten.mv.default): (5e-5, 5e-5),
    (torch.float64, torch.ops.aten.upsample_bicubic2d.vec): (1e-5, 5e-4),
    (torch.float64, torch.ops.aten.upsample_bicubic2d.default): (1e-5, 5e-4),
    # The decomposition is TOO correct. It computes everything in int64, so sometimes
    # there's an off-by-one error. See
    # https://github.com/pytorch/pytorch/issues/81996
    # https://github.com/pytorch/pytorch/issues/82230
    (torch.int8, torch.ops.aten.linspace.default): (0, 1),
    (torch.uint8, torch.ops.aten.linspace.default): (0, 1),
    (torch.int16, torch.ops.aten.linspace.default): (0, 1),
    (torch.int32, torch.ops.aten.linspace.default): (0, 1),
    (torch.int64, torch.ops.aten.linspace.default): (0, 1),
    (torch.int8, torch.ops.aten.linspace.Tensor_Tensor): (0, 1),
    (torch.uint8, torch.ops.aten.linspace.Tensor_Tensor): (0, 1),
    (torch.int16, torch.ops.aten.linspace.Tensor_Tensor): (0, 1),
    (torch.int32, torch.ops.aten.linspace.Tensor_Tensor): (0, 1),
    (torch.int64, torch.ops.aten.linspace.Tensor_Tensor): (0, 1),
    (torch.int8, torch.ops.aten.linspace.Tensor_Scalar): (0, 1),
    (torch.uint8, torch.ops.aten.linspace.Tensor_Scalar): (0, 1),
    (torch.int16, torch.ops.aten.linspace.Tensor_Scalar): (0, 1),
    (torch.int32, torch.ops.aten.linspace.Tensor_Scalar): (0, 1),
    (torch.int64, torch.ops.aten.linspace.Tensor_Scalar): (0, 1),
    (torch.int8, torch.ops.aten.linspace.Scalar_Tensor): (0, 1),
    (torch.uint8, torch.ops.aten.linspace.Scalar_Tensor): (0, 1),
    (torch.int16, torch.ops.aten.linspace.Scalar_Tensor): (0, 1),
    (torch.int32, torch.ops.aten.linspace.Scalar_Tensor): (0, 1),
    (torch.int64, torch.ops.aten.linspace.Scalar_Tensor): (0, 1),
}


def op_assert_equal(test_case, op, test_dtype, orig, decomp, args, kwargs):
    # [FUNCTION DOC] HELPER: Strict comparison of native and decomposed outputs. Asserts equal
    # dtypes, then asserts equal values with rtol and atol taken from
    # op_assert_equal_tol_table keyed by decomposition dtype and op, or else from the default
    # tolerances of both dtypes. Failure messages include the op name, args and kwargs.
    test_case.assertEqual(
        orig.dtype,
        decomp.dtype,
        lambda msg: f"{msg}\nOperation: {op}, orig.dtype: {orig.dtype}, "
        f"decomp.dtype: {decomp.dtype}, {args}, {kwargs}",
    )
    if (decomp.dtype, op) in op_assert_equal_tol_table:
        rtol, atol = op_assert_equal_tol_table[(decomp.dtype, op)]
    else:
        rtol, atol = _getDefaultRtolAndAtol(orig.dtype, decomp.dtype)

    test_case.assertEqual(
        orig,
        decomp,
        rtol=rtol,
        atol=atol,
        msg=lambda msg: f"{msg}\n{op.__name__}\nargs = {args}\nkwargs = {kwargs}",
    )


# Given f, returns an f' such that:
# - f' takes only positional arguments
# - All arguments to f' are floating-point Tensors
# - All outputs of f' are floating-point Tensors
def normalize_op_input_output2(
    f, args, kwargs, output_process_fn_grad=None, requires_grad=True
):
    # [FUNCTION DOC] HELPER: Wraps f into a function over only the differentiable positional
    # inputs. Flattens args as a pytree, selects leaves that diff_arg accepts (raising
    # AssertionError if there are none), and returns the wrapper and the tuple of those leaves
    # as primals. The wrapper substitutes new values for those leaves, rebuilds args, calls f
    # with kwargs, applies output_process_fn_grad if given, and for tuple results keeps only
    # floating or complex tensor outputs (AssertionError if none remain).
    flat_args, args_spec = tree_flatten(args)
    diff_argnums = tuple(
        i
        for i, arg in enumerate(flat_args)
        if diff_arg(arg, requires_grad=requires_grad)
    )
    if len(diff_argnums) <= 0:
        raise AssertionError("expected diff_argnums to be non-empty")
    primals = tuple(flat_args[i] for i in diff_argnums)

    @functools.wraps(f)
    def wrapped(*primals):
        # [FUNCTION DOC] HELPER: Closure returned by normalize_op_input_output2 (it keeps the
        # name and metadata of f). Takes replacement values for the differentiable leaves,
        # runs f on the rebuilt arguments and returns the processed, filtered result as
        # described there.
        _args = list(flat_args)
        for num, arg in zip(diff_argnums, primals):
            _args[num] = arg
        _args = tree_unflatten(_args, args_spec)
        result = f(*_args, **kwargs)
        if output_process_fn_grad is not None:
            result = output_process_fn_grad(result)
        if isinstance(result, tuple):
            # TODO We should check that the integer outputs also agree
            result = tuple(
                r
                for r in result
                if isinstance(r, Tensor) and (r.is_floating_point() or r.is_complex())
            )
            if len(result) <= 0:
                raise AssertionError("expected result to be non-empty")
        return result

    return wrapped, primals


# NB: This also upcasts dtype arguments
# TODO: handle complex correctly
def upcast_tensor(x, dtype=torch.float32):
    # [FUNCTION DOC] HELPER: Converts a floating point tensor to the given dtype (default
    # float32); a dtype argument equal to float16, bfloat16 or float32 is replaced by that
    # dtype; any other value is returned unchanged. A todo says complex is not handled.
    if isinstance(x, Tensor) and x.dtype.is_floating_point:
        return x.to(dtype=dtype)
    elif isinstance(x, torch.dtype) and x in [
        torch.float16,
        torch.bfloat16,
        torch.float,
    ]:
        return dtype
    else:
        return x


def normalize_op_input_output(f, sample, requires_grad=True):
    # [FUNCTION DOC] HELPER: Convenience wrapper that builds the positional argument tuple
    # (input followed by args) from a sample input and delegates to normalize_op_input_output2
    # with the sample kwargs and its gradient output processing function. Returns the wrapper
    # and primals.
    args = tuple([sample.input] + list(sample.args))
    return normalize_op_input_output2(
        f,
        args,
        sample.kwargs,
        sample.output_process_fn_grad,
        requires_grad=requires_grad,
    )


CROSS_REF_EXCLUDE_SET = {
    # CUBLAS_STATUS_NOT_SUPPORTED when calling
    # `cublasGemmStridedBatchedExFix(handle, opa, opb, (int)m, (int)n, (int)k,
    # (void*)&falpha, a, CUDA_R_16BF, (int)lda, stridea, b, CUDA_R_16BF,
    # (int)ldb, strideb, (void*)&fbeta, c, CUDA_R_16BF, (int)ldc, stridec,
    # (int)num_batches, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`
    ("cuda", torch.bfloat16, "nn.functional.bilinear"),
    # randomness
    (None, None, "special.ndtr"),  # aten.special_ndtr was not decomposed
    (None, None, "new_empty"),
    (None, None, "empty_like"),
    (None, None, "empty"),
    # AssertionError: False is not true : aten.item was not decomposed, saw calls for: aten._local_scalar_dense.default.
    (None, None, "item"),
    # It's the only in-place op without an out-of-place equivalent in the Python API
    # Its OpInfo wrongly registers it as `torch.zero_(x.clone())`.
    (None, None, "zero_"),
    # No idea what's going on here
    # In the recursive test logsumexp.default fails with args = (torch.tensor(-math.inf), [])
    # in the test, but it seems to pass when tested locally and in the logsumexp test
    (None, torch.float32, "masked.logsumexp"),
    (None, torch.float64, "masked.logsumexp"),
    # exp_vml_cpu not implemented for Half
    (torch.cpu, torch.float16, "signal.windows.exponential"),
    (torch.cpu, torch.float16, "signal.windows.gaussian"),
    # sin_vml_cpu not implemented for Half
    (torch.cpu, torch.float16, "signal.windows.cosine"),
    # CompositeAutogradImplicit
    # See https://github.com/pytorch/pytorch/issues/81669
    (None, None, "nn.functional.relu6"),
    # This decomp runs before autograd.
    (None, None, "nn.functional.rrelu"),
    (None, None, "meshgrid"),
    # Decomposition registered as Autograd
    (None, None, "nn.functional.hardshrink"),
    (None, None, "nn.functional.softshrink"),
    # diag was not decomposed (it just registers a decomp for diag_out, torch.diag is CompImplicit)
    (None, None, "diag"),
    # _softmax_backward_data's CPU kernel for bfloat16 always return the grad_input as float32
    ("cpu", torch.bfloat16, "_softmax_backward_data"),
    (None, None, "norm"),
    # native_batch_norm is only implicit when python dispatcher is on (and noncomposite otherwise)
    (None, None, "native_batch_norm"),
    (None, None, "_upsample_bilinear2d_aa"),
    (None, None, "empty_strided"),  # aten.empty_strided was not decomposed
    (
        None,
        None,
        "bernoulli",
    ),  # bernoulli is a function of randomness, so couldn't do cross-reference.
    # Decomposition is intentionally partial: returns NotImplemented for
    # non-evenly-divisible output sizes.
    (None, None, "nn.functional.adaptive_max_pool1d"),
    (None, None, "nn.functional.adaptive_max_pool2d"),
    (None, None, "nn.functional.adaptive_max_pool3d"),
}

CROSS_REF_BACKWARD_EXCLUDE_SET = {
    # Decomposed backward formula is not as precise
    ("cpu", torch.bfloat16, "nn.functional.hardswish"),
    ("cuda", torch.float16, "nn.functional.cross_entropy"),
    (
        None,
        None,
        "bernoulli",
    ),  # bernoulli is a function of randomness, so couldn't do cross-reference.
}

all_decomposed = set()
all_called = defaultdict(int)

# Helpful snippet for testing coverage
"""
import atexit
def check_coverage():
    print("missing coverage:")
    print("\n".join(map(str, decomposition_table.keys() - all_decomposed)))
atexit.register(check_coverage)
"""

# Helpful snippet for Horace to create his google sheet :)
"""
import atexit
def dump_ops():
    with open('run_ops.txt', 'w') as f, open('count_ops.txt', 'w') as g:
        for op, count in sorted(all_called.items(), key=lambda x: x[0].__name__):
            f.write(f'{op.__name__}\n')
            g.write(f'{count}\n')
    with open('run_decompositions.txt', 'w') as f:
        for op in sorted([i.__name__ for i in all_decomposed]):
            f.write(f'{op}\n')

atexit.register(dump_ops)
"""


def any_unsupported(args, kwargs):
    # [FUNCTION DOC] HELPER: Returns true if any leaf of args and kwargs is something
    # decompositions are not expected to handle: a plain tensor or parameter that is sparse,
    # sparse CSR, mkldnn, quantized, nested or a functional tensor, or any other tensor-like
    # object (such as a subclass). Otherwise false.
    def test_unsupported(t):
        # [FUNCTION DOC] HELPER: Predicate for one leaf. For exact tensor or parameter types
        # returns true if it has any unsupported layout or wrapper property; for other tensor-
        # like objects returns true; otherwise false.
        if type(t) is torch.Tensor or type(t) is torch.nn.Parameter:
            # These are all things that we haven't coded decompositions
            # to handle correctly.  Maybe they should.
            return any(
                [
                    t.is_sparse_csr,
                    t.is_sparse,
                    t.is_mkldnn,
                    t.is_quantized,
                    t.is_nested,
                    torch._is_functional_tensor(t),
                ]
            )
        elif torch.overrides.is_tensor_like(t):
            # Decompositions will generally change the behavior of Tensor-like
            # subclasses, so bypass tests in this case too
            return True
        else:
            return False

    flat_args = pytree.arg_tree_leaves(*args, **kwargs)
    return any(test_unsupported(x) for x in flat_args)


core_backward_failures = {
    skip("_softmax_backward_data"),  # slow: fails with --timeout=360 secs
    skip("deg2rad"),  # slow: fails with --timeout=360 secs
    skip("diag_embed"),  # slow: fails with --timeout=360 secs
    skip("frac"),  # slow: fails with --timeout=360 secs
    skip("grid_sampler_2d"),  # slow: fails with --timeout=360 secs
    skip("logaddexp"),  # slow: fails with --timeout=360 secs
    skip("native_dropout_backward"),  # slow: fails with --timeout=360 secs
    skip("nn.functional.glu"),  # slow: fails with --timeout=360 secs
    xfail("nn.functional.hardshrink"),
    xfail("nn.functional.softshrink"),
    skip("nn.functional.unfold"),  # slow: fails with --timeout=360 secs
    xfail("norm"),
    xfail("norm", "fro"),
    xfail("norm", "inf"),
    xfail("norm", "nuc"),
    skip("rad2deg"),  # slow: fails with --timeout=360 secs
    skip("renorm"),  # slow: fails with --timeout=360 secs
    skip("rot90"),  # slow: fails with --timeout=360 secs
    skip("rsub"),  # slow: fails with --timeout=360 secs
    skip("sgn"),  # slow: fails with --timeout=360 secs
    skip("special.xlog1py"),  # slow: fails with --timeout=360 secs
    xfail("stack"),
    skip("tril"),  # slow: fails with --timeout=360 secs
    skip("triu"),  # slow: fails with --timeout=360 secs
    skip("unfold_copy"),  # slow: fails with --timeout=360 secs
    skip("xlogy"),  # slow: fails with --timeout=360 secs
    xfail("zero_"),
}
if not TEST_WITH_SLOW:
    core_backward_failures.update(
        {
            skip("addr"),  # slow: takes 46 sec on A100
            skip("baddbmm"),  # slow: takes 800+ sec on A100
            skip("clamp_min"),  # slow: takes 800 sec on A100
            skip("clamp_max"),  # slow: takes 800 sec on A100
            skip("logit"),  # slow: takes 44 sec on A100
            skip("nn.functional.hardswish"),  # slow: takes 60 sec on A100
            skip("std_mean"),  # slow: takes 170 sec on A100
            skip("split", variant_name="list_args"),  # slow: takes 118 sec on A100
            skip("transpose"),  # slow: takes 50 sec on A100
            skip("unbind"),  # slow: takes 70 sec on A100
            skip("unsafe_split"),  # slow: takes 49 sec on A100
        }
    )

comprehensive_failures = {
    xfail(
        "nn.functional.interpolate", "bilinear", dtypes=(torch.uint8,)
    ),  # off by one error
    xfail(
        "nn.functional.interpolate", "bicubic", dtypes=(torch.uint8,)
    ),  # off by one error
    xfail(
        "nn.functional.upsample_bilinear", "", dtypes=(torch.uint8,)
    ),  # off by one error
}


@unMarkDynamoStrictTest
class TestDecomp(TestCase):
    # [CLASS DOC] Device-generic tests running decomposition cross-reference checks and a few
    # targeted comparisons (instantiated per device type at import). Sets long messages on.
    # Holds the nested DecompCrossRefMode, and the shared drivers check_decomposed and
    # do_cross_ref used by the OpInfo-based tests.
    longMessage = True

    # NB: This actually overlaps with test_comprehensive, but it only
    # runs on things that are definitely decomposed so it's a lot faster
    # to run
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    @suppress_warnings
    @ops(_decomp_test_ops)
    def test_quick(self, device, dtype, op):
        # [FUNCTION DOC] Quick cross-reference run over OpInfos known to have decompositions:
        # calls do_cross_ref with run_all false, so each sample is checked and the op must
        # actually have been decomposed. Runs only on native device types and is skipped under
        # cross-ref mode.
        self.do_cross_ref(device, dtype, op, run_all=False)

    @skipOps(core_backward_failures)
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    @suppress_warnings
    @ops(_decomp_test_ops_core_autograd, allowed_dtypes=(torch.float64,))
    def test_quick_core_backward(self, device, dtype, op):
        # [FUNCTION DOC] Checks decompositions of core ATen ops with autograd in float64 using
        # gradcheck. Skips when the op appears in the backward exclusion set. For each sample
        # input with requires_grad, runs gradcheck of the op under DecompCrossRefMode and the
        # Python dispatcher, then asserts via check_decomposed that the op was decomposed.
        test_keys = [
            (torch.device(device).type, dtype, op.name),
            (None, dtype, op.name),
            (None, None, op.name),
        ]
        if any(key in CROSS_REF_BACKWARD_EXCLUDE_SET for key in test_keys):
            self.skipTest(f"{op.name} in {dtype} not supported")
        for sample_input in op.sample_inputs(device, dtype, requires_grad=True):
            aten_name = op.decomp_aten_name or op.aten_name
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs
            func = partial(op.get_op(), **kwargs)
            with (
                self.DecompCrossRefMode(
                    self, self.precision, self.rel_tol, dtype, run_all=False
                ) as mode,
                enable_python_dispatcher(),
            ):
                torch.autograd.gradcheck(func, args)
            self.check_decomposed(aten_name, mode)

    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    @skipOps(comprehensive_failures)
    @suppress_warnings
    @ops(op_db)
    def test_comprehensive(self, device, dtype, op):
        # [FUNCTION DOC] Comprehensive cross-reference run over every OpInfo (skipped under
        # ASAN and for expected failures) calling do_cross_ref with run_all true, which also
        # checks recursive decompositions but does not require the op itself to be decomposed.
        self.do_cross_ref(device, dtype, op, run_all=True)

    def test_hann_window_decomp(self, device):
        # [FUNCTION DOC] Compares the hann_window decomposition to the native kernel for
        # window lengths 0, 1, 8 and 9 (default periodic form), for the periodic overload with
        # both flag values, and for float64 dtype forwarding (checking the result dtype too).
        # KNOWN ISSUE: the leading comment promises coverage of the out and periodic_out
        # overloads, but the test only exercises the default and periodic overloads.
        # Verify the hann_window decomp matches the native kernel for all four
        # overloads: .default, .periodic, .out, .periodic_out.
        from torch._decomp.decompositions import hann_window, hann_window_periodic

        for n in (0, 1, 8, 9):
            # .default (periodic=True)
            ref = torch.hann_window(n, device=device)
            res = hann_window(n, device=device)
            self.assertEqual(ref, res)

            # .periodic overload, explicit periodic flag
            for periodic in (True, False):
                ref = torch.hann_window(n, periodic, device=device)
                res = hann_window_periodic(n, periodic, device=device)
                self.assertEqual(ref, res)

        # dtype forwarding
        ref = torch.hann_window(8, dtype=torch.float64, device=device)
        res = hann_window_periodic(8, dtype=torch.float64, device=device)
        self.assertEqual(ref, res)
        self.assertEqual(res.dtype, torch.float64)

    def test_uniform(self, device):
        # [FUNCTION DOC] Seeds the RNG with 123, calls the native uniform op on a random
        # float32 tensor, reseeds, calls the decomposition with the same bounds (0.3 to 0.9),
        # and asserts equal outputs.
        size = (2, 3, 4, 5)
        dtype = torch.float32
        x = make_tensor(size, dtype=dtype, device=device)
        low = 0.3
        high = 0.9

        torch.manual_seed(123)
        ref = torch.ops.aten.uniform(x, low, high)
        torch.manual_seed(123)
        res = torch._decomp.decompositions.uniform(x, low=low, high=high)
        self.assertEqual(ref, res)

    def test_bernoulli_default(self, device):
        # [FUNCTION DOC] Draws bernoulli samples from a 5x5 tensor of probability 0.3 with the
        # native op and the decomposition under the same seed, and asserts the sample means
        # are within a loose tolerance (6 percent) rather than elementwise equal.
        # KNOWN ISSUE: the device argument is unused; the probability tensor is always on CPU.
        p = 0.3
        p_t = p * torch.ones(5, 5)
        torch.manual_seed(123)
        ref = torch.ops.aten.bernoulli.default(p_t)
        torch.manual_seed(123)
        res = torch._decomp.decompositions.bernoulli(p_t)
        ref_p = ref.sum() / torch.prod(torch.tensor(ref.size()))
        res_p = res.sum() / torch.prod(torch.tensor(res.size()))
        self.assertEqual(ref_p, res_p, atol=0.06 * p, rtol=0.06)

    def test_broadcasting_index_copy(self, device):
        # [FUNCTION DOC] Checks the index_copy_ decomposition broadcasts its source: copies a
        # 1x10 tensor into row zero of a 2x10 ones tensor in place, and compares to manually
        # assigning that row.
        x = torch.zeros([1, 10], device=device)
        xs = torch.ones([2, 10], device=device)

        def index_copy(xs, x):
            # [FUNCTION DOC] HELPER: Calls the decomposition of in-place index_copy_ on xs
            # along dim 0 with a scalar zero index on the device and source x. Mutates xs;
            # returns nothing.
            torch._decomp.decompositions.index_copy_(
                xs, 0, torch.tensor(0).to(device), x
            )

        index_copy(xs, x)

        xs_two = torch.ones([2, 10], device=device)
        xs_two[0] = x

        self.assertEqual(xs, xs_two)

    def test_cat_single_input(self, device):
        # [FUNCTION DOC] Compares the inductor decomposition of cat against torch.cat for a
        # list of ten references to one large (2048x2048) tensor, along dims minus one, zero
        # and one.
        decomp_table = torch._inductor.decomposition.select_decomp_table()
        cat_inductor = decomp_table[torch.ops.aten.cat.default]

        inp = torch.rand([2048, 2048], device=device)
        inps = [inp for _ in range(10)]

        for dim in (-1, 0, 1):
            self.assertEqual(torch.cat(inps, dim), cat_inductor(inps, dim))

    @suppress_warnings
    @tf32_off()
    # only tests RNNs since we have py dispsatcher decomps for them
    @modules(
        filter(
            lambda m: m.module_cls in (torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU),
            module_db,
        )
    )
    def test_rnn_decomp_module(self, device, dtype, module_info, training):
        # [FUNCTION DOC] For RNN, LSTM and GRU modules from module_db (TF32 off, warnings
        # suppressed): for each module input that has a forward input, builds the module,
        # moves it to device and dtype, runs it under DecompCrossRefMode (run_all true) and
        # the Python dispatcher, then runs it normally and asserts the outputs are equal,
        # which catches wrong Python dispatcher decompositions that the dispatch-level checks
        # cannot.
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(
            module_info,
            device=device,
            dtype=dtype,
            requires_grad=True,
            training=training,
        )
        for module_input in module_inputs:
            if module_input.forward_input is None:
                continue
            args, kwargs = (
                module_input.constructor_input.args,
                module_input.constructor_input.kwargs,
            )
            m = module_cls(*args, **kwargs)
            m.to(device).to(dtype)

            args, kwargs = (
                module_input.forward_input.args,
                module_input.forward_input.kwargs,
            )
            with (
                self.DecompCrossRefMode(
                    self, self.precision, self.rel_tol, dtype, run_all=True
                ),
                enable_python_dispatcher(),
            ):
                decomp_out = m(*args, **kwargs)

            non_decomp_out = m(*args, **kwargs)
            # without this check, incorrect decomps at the python dispatcher level can still pass because
            # they're checking aten decomps at the torch_dispatch level
            self.assertEqual(decomp_out, non_decomp_out)

    def test_batch_norm_unflatten_weight_bias(self, device):
        # [FUNCTION DOC] Regression test for a batch norm decomposition given weight of shape
        # 3x1x1x1: runs the decomposition in eval mode and asserts the output shape equals the
        # input shape.
        # https://github.com/pytorch/pytorch/issues/100970
        shape = (1, 3, 2, 2)
        input = torch.randn(shape, device=device)
        weight = torch.randn((3, 1, 1, 1), device=device)
        bias = torch.randn(3, device=device)
        mean = torch.randn(3, device=device)
        var = torch.randn(3, device=device)
        res = torch._decomp.decompositions.native_batch_norm(
            input, weight, bias, mean, var, False, 1, 1e-05
        )
        self.assertEqual(shape, res[0].shape)

    def test_batch_norm_eval_emits_rsqrt(self, device):
        # [FUNCTION DOC] Traces the no-training legit batch norm op with make_fx using the
        # decomposition table and asserts the generated graph code contains rsqrt and no
        # reciprocal.
        # The eval/inference branch of native_batch_norm_helper computes the
        # inverse std as rsqrt(running_var + eps), matching the training branch,
        # rather than the un-fused 1 / sqrt(running_var + eps). Assert the
        # decomposed graph contains a single rsqrt and no reciprocal + sqrt pair.
        from torch.fx.experimental.proxy_tensor import make_fx

        def func(input, weight, bias, mean, var):
            # [FUNCTION DOC] HELPER: Traced function calling the no-training legit batch norm
            # ATen op with momentum 0.1 and eps 1e-5, returning its outputs.
            return torch.ops.aten._native_batch_norm_legit_no_training.default(
                input, weight, bias, mean, var, 0.1, 1e-05
            )

        shape = (2, 3, 4, 4)
        input = torch.randn(shape, device=device)
        weight = torch.randn(3, device=device)
        bias = torch.randn(3, device=device)
        mean = torch.randn(3, device=device)
        var = torch.rand(3, device=device) + 1.0

        fx_g = make_fx(func, decomposition_table=decomposition_table)(
            input, weight, bias, mean, var
        )
        graph_str = fx_g.code
        self.assertIn("rsqrt", graph_str)
        self.assertNotIn("reciprocal", graph_str)

    def test_arange_graph(self, device):
        # [FUNCTION DOC] Traces a function that builds a float32 arange of the input length,
        # with or without a start, using make_fx and the decomposition table. After stripping
        # device and requires_grad text, asserts the graph equals the expected text built from
        # prims iota, mul, add and convert_element_type (length 10 offset 0, then length 9
        # offset 1).
        from torch.fx.experimental.proxy_tensor import make_fx

        def func(x, start):
            # [FUNCTION DOC] HELPER: Traced function: returns a float32 arange over the last
            # dimension size of x, beginning at start unless start is None.
            le = x.shape[-1]
            if start is None:
                a = torch.arange(le, dtype=torch.float32, device=x.device)
            else:
                a = torch.arange(start, le, dtype=torch.float32, device=x.device)
            return a

        pattern = r", device = device\(.+\), requires_grad = False"

        cfunc = make_fx(func, decomposition_table=decomposition_table)
        fx_g = cfunc(torch.rand(10, device=device), None)
        fx_g_code = fx_g.code.strip()
        # Remove device and requires_grad
        fx_g_code = re.sub(pattern, "", fx_g_code)
        self.assertExpectedInline(
            fx_g_code,
            """\
def forward(self, x_1, start_1):
    iota = torch.ops.prims.iota.default(10, start = 0, step = 1, dtype = torch.int64)
    mul = torch.ops.prims.mul.default(iota, 1);  iota = None
    add = torch.ops.prims.add.default(mul, 0);  mul = None
    convert_element_type = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
    return convert_element_type""",
        )

        fx_g = cfunc(torch.rand(10, device=device), 1)
        fx_g_code = fx_g.code.strip()
        # Remove device and requires_grad
        fx_g_code = re.sub(pattern, "", fx_g_code)
        self.assertExpectedInline(
            fx_g_code,
            """\
def forward(self, x_1, start_1):
    iota = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64)
    mul = torch.ops.prims.mul.default(iota, 1);  iota = None
    add = torch.ops.prims.add.default(mul, 1);  mul = None
    convert_element_type = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
    return convert_element_type""",
        )

    def test_masked_fill(self, device):
        # [FUNCTION DOC] Skips unless the device is xpu, cuda or the privateuse1 backend.
        # Traces masked_fill with make_fx and the decomposition table and asserts the graph is
        # a single prims where call.
        from torch.fx.experimental.proxy_tensor import make_fx

        if torch.device(device).type not in [
            "xpu",
            "cuda",
            torch._C._get_privateuse1_backend_name(),
        ]:
            self.skipTest("only runs on XPU and CUDA and PrivateUse1.")

        def func(scores, mask, value):
            # [FUNCTION DOC] HELPER: Traced function returning scores with masked positions
            # replaced by value.
            return scores.masked_fill(mask, value)

        scores_t = torch.tensor([1, 2, 3, 4], device=device)
        mask_t = torch.tensor([True, True, True, True], device=device)
        value_t = torch.tensor(0, dtype=scores_t.dtype)
        cfunc = make_fx(func, decomposition_table=decomposition_table)
        fx_g = cfunc(scores_t, mask_t, value_t)
        self.assertExpectedInline(
            fx_g.code.strip(),
            """\
def forward(self, scores_1, mask_1, value_1):
    where = torch.ops.prims.where.default(mask_1, value_1, scores_1);  mask_1 = value_1 = scores_1 = None
    return where""",
        )

    class DecompCrossRefMode(TorchDispatchMode):
        # [CLASS DOC] TorchDispatchMode that cross-checks decompositions. For every dispatched
        # op that has a decomposition it runs the decomposition and the real op and compares
        # them. State: the owning test case, saved precision and relative tolerance (restored
        # on every call), the test dtype, a run_all flag (decompose recursively), and sets
        # called and decomposed of ops seen. Also updates the module-level all_called and
        # all_decomposed records.
        def __init__(self, test_case, saved_precision, saved_rel_tol, dtype, run_all):
            # [FUNCTION DOC] Stores the test case, saved tolerances, dtype and run_all flag,
            # and creates empty sets of called and decomposed ops. No other side effects.
            self.test_case = test_case
            self.saved_precision = saved_precision
            self.saved_rel_tol = saved_rel_tol
            self.test_dtype = dtype
            self.run_all = run_all

            # We check the correctness of each decomposition right after running it.
            # So, when we encounter a decomposition, we run the function normally, and
            # then run the decomposition, and ensure they're identical.
            self.called = set()
            self.decomposed = set()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            # [FUNCTION DOC] Intercepts one op call. Resets the test case precision and
            # rel_tol to the saved values, records the op as called (and in the global
            # counter). Ops lacking a decomposition, in a fixed ignore list (detach, empty and
            # random constructors, dropout), tagged seeded nondeterministic, with unsupported
            # args, or in place are simply executed. Otherwise records the op as decomposed,
            # runs the decomposition (recursively under this mode when run_all) and then the
            # real op, raises AssertionError if the number of outputs differs, and compares
            # each output: for float16/bfloat16 against a float64 (float32 on mps)
            # recomputation using op_assert_ref, else with op_assert_equal; non-tensor outputs
            # must match in type and value (AssertionError). Returns the real op result.
            self.test_case.precision = self.saved_precision
            self.test_case.rel_tol = self.saved_rel_tol

            self.called.add(func)
            all_called[func] += 1

            # Stuff we shouldn't bother testing
            # (TODO: remove detach from the decomp table?)
            # N.b. Testing in-place ops would need dedicated logic
            in_place = func._schema.name.endswith("_")
            ignored_ops = [
                torch.ops.aten.detach.default,
                # non-deterministic ops
                torch.ops.aten.empty.memory_format,
                torch.ops.aten.empty_like.default,
                torch.ops.aten.new_empty.default,
                torch.ops.aten.empty_strided.default,
                torch.ops.aten.new_empty_strided.default,
                torch.ops.aten.randn.default,
                torch.ops.aten.native_dropout.default,
            ]
            if (
                func not in decomposition_table
                or func in ignored_ops
                or torch.Tag.nondeterministic_seeded in func.tags
                or any_unsupported(args, kwargs)
                or in_place
            ):
                return func(*args, **kwargs)

            self.decomposed.add(func)
            all_decomposed.add(func)

            # We take 2 main strategies for verifying correctness/numerical stability of decompositions
            # The first one is simply tolerance checking between decomp_out and pytorch_out
            # However, for fp16/bf16 and reductions, this becomes very
            # finicky, as there are not many guarantees we can make.
            # So, for fp16/bf16, we instead compare the difference of
            # {decomp_out, pytorch_out_64} and {pytorch_out,
            # pytorch_out_64}. In other words, we compare how far the
            # decomposition and pytorch are from the "ground truth" (i.e.
            # fp64). If the decomposition results in more error, we error

            # We also decompose the decomposition recursively for
            # further coverage, as some paths not be exercised directly by
            # OpInfos (sadly) but just by other ops

            decomposition = decomposition_table[func]

            do_relative_check = self.test_dtype in [torch.float16, torch.bfloat16]
            if self.run_all:
                # Execute recursively via DFS, to find the root of a possible error first
                with self:
                    decomp_out = pytree.tree_leaves(decomposition(*args, **kwargs))
            else:
                decomp_out = pytree.tree_leaves(decomposition(*args, **kwargs))

            # At this stage we should not be decomposing an in-place op
            # We'd like to have decompositions that decompose out-of-place ops into out-of-place ops
            #  because decompositions are run after functionalisation and we would not like them to
            #  de-functionalise the graph, as that would break AoTAutograd
            # We run the real function *after* the decomposition to make sure that the
            # decomposition does not modify any of the inputs in-place. If it does
            # real_out should be different than decom_out so we should catch this
            real_out_unflat = func(*args, **kwargs)
            real_out = pytree.tree_leaves(real_out_unflat)

            if len(real_out) != len(decomp_out):
                raise AssertionError(
                    f"output length mismatch: {len(real_out)} != {len(decomp_out)}"
                )

            if do_relative_check:
                device_arg = kwargs.get("device", None)

                def upcast(x):
                    # [FUNCTION DOC] HELPER: Upcasts a value for the reference computation:
                    # tensors on mps (or when the device kwarg is mps) go to float32,
                    # everything else to float64, via upcast_tensor.
                    if (isinstance(x, Tensor) and x.device.type == "mps") or (
                        device_arg and torch.device(device_arg).type == "mps"
                    ):
                        return upcast_tensor(x, dtype=torch.float32)
                    else:
                        return upcast_tensor(x, dtype=torch.float64)

                real_out_double, _ = tree_flatten(
                    func(*tree_map(upcast, args), **tree_map(upcast, kwargs))
                )
                for i, (orig, decomp, ref) in enumerate(
                    zip(real_out, decomp_out, real_out_double)
                ):
                    if not isinstance(orig, torch.Tensor):
                        if type(orig) is not type(decomp):
                            raise AssertionError(
                                f"type mismatch: {type(orig)} != {type(decomp)}"
                            )
                        if orig != decomp:
                            raise AssertionError(f"value mismatch: {orig} != {decomp}")
                        continue
                    op_assert_ref(
                        self.test_case,
                        func,
                        self.test_dtype,
                        i,
                        orig,
                        decomp,
                        ref,
                        args,
                        kwargs,
                    )
            else:
                for orig, decomp in zip(real_out, decomp_out):
                    if not isinstance(orig, torch.Tensor):
                        if type(orig) is not type(decomp):
                            raise AssertionError(
                                f"type mismatch: {type(orig)} != {type(decomp)}"
                            )
                        if orig != decomp:
                            raise AssertionError(f"value mismatch: {orig} != {decomp}")
                        continue
                    op_assert_equal(
                        self.test_case,
                        func,
                        self.test_dtype,
                        orig,
                        decomp,
                        args,
                        kwargs,
                    )

            return real_out_unflat

    def check_decomposed(self, aten_name, mode):
        # [FUNCTION DOC] Asserts that at least one op in mode.decomposed has the given base
        # aten name. The failure message lists the ops seen and suggests adding
        # CompositeImplicitAutograd ops to CROSS_REF_EXCLUDE_SET.
        self.assertTrue(
            any(overload_to_aten_name(c) == aten_name for c in mode.decomposed),
            msg=(
                f"aten.{aten_name} was not decomposed, saw calls for: "
                f"{', '.join(map(str, list(mode.called)))}. If your op is  "
                f"CompositeImplicitAutograd you should skip this test "
                f"by updating CROSS_REF_EXCLUDE_SET."
            ),
        )

    @onlyCPU
    @skipIfCrossRef
    def test_decomp_crossref_mode_skips_overloaded_inplace(self, device):
        # [FUNCTION DOC] CPU only. Runs an in-place pow overload on a tensor under
        # DecompCrossRefMode and the Python dispatcher and asserts it was called but not
        # recorded as decomposed.
        x = torch.tensor(2.0, device=device)
        mode = self.DecompCrossRefMode(
            self, self.precision, self.rel_tol, x.dtype, run_all=False
        )
        with mode, enable_python_dispatcher():
            aten.pow_.Scalar(x, 3.0)

        self.assertIn(aten.pow_.Scalar, mode.called)
        self.assertNotIn(aten.pow_.Scalar, mode.decomposed)

    @onlyCPU
    @skipIfCrossRef
    def test_decomp_crossref_mode_decomposes_mutable_functional_ops(self, device):
        # [FUNCTION DOC] CPU only. Runs the batch_norm_with_update op (which mutates running
        # stats) on random inputs under DecompCrossRefMode and the Python dispatcher and
        # asserts it appears in both the called and decomposed sets.
        x = torch.randn(2, 3, 4, 4, device=device)
        weight = torch.randn(3, device=device)
        bias = torch.randn(3, device=device)
        running_mean = torch.randn(3, device=device)
        running_var = torch.rand(3, device=device) + 1
        mode = self.DecompCrossRefMode(
            self, self.precision, self.rel_tol, x.dtype, run_all=False
        )

        with mode, enable_python_dispatcher():
            aten._batch_norm_with_update.default(
                x, weight, bias, running_mean, running_var, 0.1, 1e-5
            )

        self.assertIn(aten._batch_norm_with_update.default, mode.called)
        self.assertIn(aten._batch_norm_with_update.default, mode.decomposed)

    @skipIfTorchDynamo("Test does not work with TorchDynamo")
    def do_cross_ref(self, device, dtype, op, *, run_all):
        # [FUNCTION DOC] Shared driver for OpInfo cross-reference tests (skipped under
        # Dynamo). Skips the test if the op, dtype and device are in CROSS_REF_EXCLUDE_SET.
        # Decides whether to test backward (op supports autograd, dtype has backward support,
        # not complex32) and whether the backward exclusion set applies. For each sample it
        # runs the op, or its vector-Jacobian product with random cotangents, under
        # DecompCrossRefMode and the Python dispatcher, and again without the dispatcher when
        # a composite implicit kernel is involved. Unless run_all, it asserts the forward and
        # backward ops were decomposed. If only backward is decomposed but the dtype lacks
        # autograd it skips the test. Argument run_all is keyword only.
        test_keys = [
            (torch.device(device).type, dtype, op.name),
            (None, dtype, op.name),
            (None, None, op.name),
        ]
        if any(key in CROSS_REF_EXCLUDE_SET for key in test_keys):
            self.skipTest(f"{op.name} in {dtype} not supported")

        skip_decomp_vjp = any(
            key in CROSS_REF_BACKWARD_EXCLUDE_SET for key in test_keys
        )

        requires_grad = (
            op.supports_autograd
            and dtype in op.supported_backward_dtypes(torch.device(device).type)
            # TODO: OpInfo really ought to error out for this case, but it's
            # not exercised in test_ops_gradients atm.  The problem is not
            # complex32 per-se (which is supported by data movement only ops)
            # but that when we do backwards we expect other ops like add to work
            and dtype != torch.complex32
        )
        samples = op.sample_inputs(device, dtype, requires_grad=requires_grad)

        aten_name = op.decomp_aten_name or op.aten_name

        func = op.get_op()

        def run_without_python_dispatcher(mode):
            # [FUNCTION DOC] HELPER: Returns true if the func under test or any op decomposed
            # in the given mode is an op overload that has a CompositeImplicitAutograd kernel,
            # meaning a second run without the Python dispatcher is needed.
            return any(
                isinstance(op, torch._ops.OpOverload)
                and op.has_kernel_for_dispatch_key(
                    DispatchKey.CompositeImplicitAutograd
                )
                for op in mode.decomposed.union([func])
            )

        for sample_input in samples:
            if requires_grad:
                fn, primals = normalize_op_input_output(func, sample_input)

                # Once https://github.com/pytorch/pytorch/pull/75965/ I can
                # store the called list on the mode object instance and no
                # explicit clearing is necessary as I will create a fresh mode
                # for each region
                with (
                    self.DecompCrossRefMode(
                        self, self.precision, self.rel_tol, dtype, run_all
                    ) as mode,
                    enable_python_dispatcher(),
                ):
                    decomp_out, decomp_vjp_fn = ref_vjp_no_create(fn, *primals)
                if run_without_python_dispatcher(mode):
                    # without this check, incorrect decomps at the python dispatcher level can still pass because
                    # they're checking aten decomps at the torch_dispatch level.
                    with self.DecompCrossRefMode(
                        self, self.precision, self.rel_tol, dtype, run_all
                    ) as mode:
                        decomp_out, decomp_vjp_fn = ref_vjp_no_create(fn, *primals)
                if aten_name in decomposition_names:
                    self.check_decomposed(aten_name, mode)

                if not skip_decomp_vjp and (
                    op.aten_backward_name in decomposition_names or run_all
                ):
                    cotangents = tree_map(lambda x: torch.randn_like(x), decomp_out)

                    with (
                        self.DecompCrossRefMode(
                            self, self.precision, self.rel_tol, dtype, run_all
                        ) as mode,
                        enable_python_dispatcher(),
                    ):
                        decomp_vjp_fn(cotangents)
                    if run_without_python_dispatcher(mode):
                        # without this check, incorrect decomps at the python dispatcher level can still pass because
                        # they're checking aten decomps at the torch_dispatch level.
                        with self.DecompCrossRefMode(
                            self, self.precision, self.rel_tol, dtype, run_all
                        ) as mode:
                            decomp_vjp_fn(cotangents)
                    if not run_all:
                        self.check_decomposed(op.aten_backward_name, mode)

            elif aten_name in decomposition_names or run_all:
                args = [sample_input.input] + list(sample_input.args)
                kwargs = sample_input.kwargs
                # A failure here might be because the decomposition for the op is wrong or because a
                # decomposition used by the particular op is wrong.
                with (
                    self.DecompCrossRefMode(
                        self, self.precision, self.rel_tol, dtype, run_all
                    ) as mode,
                    enable_python_dispatcher(),
                ):
                    func(*args, **kwargs)

                if run_without_python_dispatcher(mode):
                    # without this check, incorrect decomps at the python dispatcher level can still pass because
                    # they're checking aten decomps at the torch_dispatch level.
                    with self.DecompCrossRefMode(
                        self, self.precision, self.rel_tol, dtype, run_all
                    ) as mode:
                        func(*args, **kwargs)

                if not run_all:
                    self.check_decomposed(aten_name, mode)
            else:
                if not op.supports_autograd:
                    raise AssertionError("expected op.supports_autograd")
                self.skipTest(
                    "only backwards is decomposed, but dtype doesn't support AD"
                )

    def test_binary_cross_entropy_with_logits_decomp(self, device):
        # [FUNCTION DOC] Compares the native binary_cross_entropy_with_logits with the
        # inductor decomposition for bfloat16 self and target of shape 4x5x6, a float32
        # weight, and reduction mode 2, using assert_close with dtype checking.
        op_config = {
            "self": torch.randn([4, 5, 6], dtype=torch.bfloat16, device=device),
            "target": torch.randn([4, 5, 6], dtype=torch.bfloat16, device=device),
            "weight": torch.randn([6], dtype=torch.float32, device=device),
            "reduction": 2,
        }

        ref = torch.ops.aten.binary_cross_entropy_with_logits.default(**op_config)

        decomp_table = torch._inductor.decomposition.select_decomp_table()
        bce_decomp = decomp_table[
            torch.ops.aten.binary_cross_entropy_with_logits.default
        ]
        res = bce_decomp(**op_config)

        torch.testing.assert_close(ref, res, check_dtype=True)


instantiate_device_type_tests(TestDecomp, globals())


class DecompOneOffTests(TestCase):
    # [CLASS DOC] Device-generic one-off correctness tests of individual decompositions
    # (stride, dtype and value checks against native ops, fused dropout edge cases, SDPA,
    # layer norm, addmm, addmv and vector norm). Instantiated per device type at import.
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    def test_contiguous_softmax(self, device):
        # [FUNCTION DOC] Builds a float32 tensor with a non-contiguous stride and asserts the
        # softmax decomposition produces the same strides as the native op along the last dim.
        size = (2, 4, 3, 3)
        stride = (9, 18, 3, 1)
        dtype = torch.float32

        x = torch.randn(size, dtype=dtype, device=device)
        x = torch.as_strided(x, size, stride)

        ref = torch.ops.aten._softmax(x, -1, False)
        res = torch._decomp.decompositions._softmax(x, -1, False)
        self.assertEqual(ref.stride(), res.stride())

    @onlyNativeDeviceTypes
    @skipIfCrossRef
    def test_contiguous_log_softmax(self, device):
        # [FUNCTION DOC] Same as the softmax stride test but for log softmax: asserts the
        # decomposition output strides equal the native op strides for a strided input.
        size = (2, 4, 3, 3)
        stride = (9, 18, 3, 1)

        dtype = torch.float32
        x = torch.randn(size, dtype=dtype, device=device)
        x = torch.as_strided(x, size, stride)

        ref = torch.ops.aten._log_softmax(x, -1, False)
        res = torch._decomp.decompositions._log_softmax(x, -1, False)
        self.assertEqual(ref.stride(), res.stride())

    @onlyCUDA
    def test_exponential_non_inf(self, device):
        # [FUNCTION DOC] CUDA only. With RNG state preserved around the in-place exponential_,
        # compares it with the refs exponential on a 4x400x256 tensor, asserts equality, and
        # that the refs result has no infinite values.
        inp = torch.empty((4, 400, 256), device=device)

        with torch._dynamo.utils.preserve_rng_state():
            exp_ref = inp.exponential_()
        exp = torch._refs.exponential(inp)

        self.assertEqual(exp, exp_ref)
        self.assertFalse(exp.isinf().any())

    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @skipIfCrossRef
    @onlyCUDA
    def test_amp_batch_norm_backward(self):
        # [FUNCTION DOC] CUDA only (hard-coded device, no device argument; skipped under
        # ASAN). Runs native batch norm backward and its decomposition on float16 activations
        # with float32 weights, running stats and empty saved stats, in eval mode with all
        # gradients requested, and asserts every output has the same stride and dtype in both.
        device = "cuda"
        grad_out = torch.randn((1, 2, 16, 16), dtype=torch.float16, device=device)
        x = torch.randn((1, 2, 16, 16), dtype=torch.float16, device=device)
        weight = torch.randn((2,), dtype=torch.float32, device=device)
        rmean = torch.randn((2,), dtype=torch.float32, device=device)
        rvar = torch.randn((2,), dtype=torch.float32, device=device)
        mean = torch.randn((0,), dtype=torch.float32, device=device)

        ref = torch.ops.aten.native_batch_norm_backward(
            grad_out,
            x,
            weight,
            rmean,
            rvar,
            mean,
            mean,
            False,
            1e-05,
            [True, True, True],
        )
        res = torch._decomp.decompositions.native_batch_norm_backward(
            grad_out,
            x,
            weight,
            rmean,
            rvar,
            mean,
            mean,
            False,
            1e-05,
            [True, True, True],
        )
        for a, b in zip(ref, res):
            self.assertEqual(a.stride(), b.stride())
            self.assertEqual(a.dtype, b.dtype)

    @onlyNativeDeviceTypes
    @skipIfCrossRef
    def test_elu_backward(self, device):
        # [FUNCTION DOC] Compares elu backward (native versus decomposition) on random float32
        # inputs with alpha 1.0, scale 1, input scale 1 and is_result true; asserts equal
        # values.
        size = (2, 4, 3, 3)
        dtype = torch.float32
        grad_out = torch.randn(size, dtype=dtype, device=device)
        out = torch.randn(size, dtype=dtype, device=device)

        ref = torch.ops.aten.elu_backward(grad_out, 1.0, 1, 1, True, out)
        res = torch._decomp.decompositions.elu_backward(grad_out, 1.0, 1, 1, True, out)
        self.assertEqual(ref, res)

    @onlyNativeDeviceTypes
    @skipIfCrossRef
    def test_threshold_backward_dtype(self, device):
        # [FUNCTION DOC] Calls threshold backward with integer tensors natively and via the
        # decomposition and asserts the result dtypes are the same.
        grad = torch.randint(10, (4,), device=device)
        input_tensor = torch.randint(10, (4,), device=device)

        ref = torch.ops.aten.threshold_backward(grad, input_tensor, 1)
        res = torch._decomp.decompositions.threshold_backward(grad, input_tensor, 1)
        self.assertEqual(ref.dtype, res.dtype)

    @onlyNativeDeviceTypes
    @skipIfCrossRef
    def test_weight_norm_interface(self, device):
        # [FUNCTION DOC] Compares _weight_norm_interface native versus decomposition: with dim
        # 2 on a 3x10x10 and 1x1x10 pair (allclose on both outputs), and with default dim on a
        # 30x10 and 30x1 pair (equality of the output tuples).
        g = torch.randn((3, 10, 10), device=device)
        v = torch.randn((1, 1, 10), device=device)

        ref = torch.ops.aten._weight_norm_interface(g, v, 2)
        res = torch._decomp.decompositions._weight_norm_interface(g, v, 2)
        self.assertTrue(torch.allclose(ref[0], res[0]))
        self.assertTrue(torch.allclose(ref[1], res[1]))

        inp = torch.rand([30, 10], device=device)
        inp2 = torch.rand([30, 1], device=device)

        self.assertEqual(
            torch.ops.aten._weight_norm_interface(inp, inp2),
            torch._decomp.decompositions._weight_norm_interface(inp, inp2),
        )

    @onlyCUDA
    @skipIfCrossRef
    def test_fused_dropout_decomposition_extreme_p(self, device):
        # [FUNCTION DOC] CUDA only. Compares the fused dropout decomposition with the native
        # op for probability parameter 0.0 (all-NaN output, all-false mask, equal with NaN
        # equality) and 1.0 (output equals input, all-true mask), asserting the decomposition
        # mask dtype is uint8, and that dtypes and values match the native results.
        input_tensor = torch.randn(10, 10, dtype=torch.float32, device=device)

        ref_out, ref_mask = torch.ops.aten._fused_dropout(input_tensor, 0.0, None)
        decomp_out, decomp_mask = (
            torch._decomp.decompositions._fused_dropout_decomposition(
                input_tensor, 0.0, None
            )
        )
        self.assertTrue(torch.isnan(ref_out).all().item())
        self.assertFalse(ref_mask.bool().any().item())
        self.assertEqual(decomp_out.dtype, ref_out.dtype)
        self.assertEqual(decomp_mask.dtype, torch.uint8)
        self.assertEqual(decomp_out, ref_out, equal_nan=True)
        self.assertEqual(decomp_mask, ref_mask)

        ref_out, ref_mask = torch.ops.aten._fused_dropout(input_tensor, 1.0, None)
        decomp_out, decomp_mask = (
            torch._decomp.decompositions._fused_dropout_decomposition(
                input_tensor, 1.0, None
            )
        )
        self.assertEqual(ref_out, input_tensor)
        self.assertTrue(ref_mask.bool().all().item())
        self.assertEqual(decomp_out.dtype, ref_out.dtype)
        self.assertEqual(decomp_mask.dtype, torch.uint8)
        self.assertEqual(decomp_out, ref_out)
        self.assertEqual(decomp_mask, ref_mask)

    @onlyCUDA
    @skipIfCrossRef
    def test_fused_dropout_compile_extreme_p(self, device):
        # [FUNCTION DOC] CUDA only. Compiles the fused dropout op and a mask-dtype probe with
        # the eager backend, and for probability parameter 0.0 and 1.0 asserts the compiled
        # output and mask have the same dtype and values (NaNs equal) as eager, and the probed
        # mask dtype matches.
        def fn(input_tensor, p, generator):
            # [FUNCTION DOC] HELPER: Calls the native fused dropout op with the given input,
            # probability and generator and returns its output and mask.
            return torch.ops.aten._fused_dropout(input_tensor, p, generator)

        def mask_dtype(input_tensor, p):
            # [FUNCTION DOC] HELPER: Returns the dtype of the mask output of the native fused
            # dropout op, called with no generator.
            return torch.ops.aten._fused_dropout(input_tensor, p, None)[1].dtype

        input_tensor = torch.randn(10, 10, dtype=torch.float32, device=device)
        compiled_fn = torch.compile(fn, backend="eager")
        compiled_mask_dtype = torch.compile(mask_dtype, backend="eager")

        for p in (0.0, 1.0):
            ref_out, ref_mask = fn(input_tensor, p, None)
            res_out, res_mask = compiled_fn(input_tensor, p, None)
            self.assertEqual(res_out.dtype, ref_out.dtype)
            self.assertEqual(res_mask.dtype, ref_mask.dtype)
            self.assertEqual(compiled_mask_dtype(input_tensor, p), ref_mask.dtype)
            self.assertEqual(res_out, ref_out, equal_nan=True)
            self.assertEqual(res_mask, ref_mask)

    @onlyCPU
    @skipIfCrossRef
    @skipOps(
        [
            xfail(
                "nn.functional.scaled_dot_product_attention",
                dtypes=[torch.half],
            ),
        ]
    )
    @ops(_sdpa_op_info)
    def test_sdpa(self, device, dtype, op):
        # [FUNCTION DOC] CPU only (half is an expected failure). With random 1x128x100x64
        # query, key and value, tries no mask (with causal flag) and an all-true boolean mask.
        # Runs the CPU flash attention decomposition with zero dropout, asserts the first
        # output is contiguous after permuting dims, and compares it to the OpInfo operator
        # result with allclose.
        # KNOWN ISSUE: the tolerance table stores relative tolerance first, but it is unpacked
        # as atol then rtol, so the two tolerances are swapped.
        # SDPA doesn't support float16, this is aligned with aten/src/ATen/native/transformers/attention.cpp. If we
        # add support for float16 over there we should update this test as well.
        query_layer = torch.randn(1, 128, 100, 64, device=device, dtype=dtype)
        key_layer = torch.randn(1, 128, 100, 64, device=device, dtype=dtype)
        value_layer = torch.randn(1, 128, 100, 64, device=device, dtype=dtype)
        masks = [None, torch.ones((1, 1, 100, 100), device=device, dtype=torch.bool)]

        atol, rtol = dtype_precisions[dtype]

        for mask in masks:
            is_causal = mask is None
            decomposed_res = (
                torch._decomp.decompositions.scaled_dot_product_flash_attention_for_cpu(
                    query_layer, key_layer, value_layer, 0.0, is_causal, attn_mask=mask
                )
            )
            actual_res = decomposed_res[0]
            # Output has form (N, H, L, E), but should be continuous on (L, N, H, E)
            # in order for subsequent view(L * N, H * E) to be valid.
            # So permute(2, 0, 1, 3) before checking that tensor is contiguous
            self.assertTrue(actual_res.permute(2, 0, 1, 3).is_contiguous())

            eager_res = op(
                query_layer,
                key_layer,
                value_layer,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=is_causal,
            )

            self.assertTrue(torch.allclose(actual_res, eager_res, atol=atol, rtol=rtol))

    @onlyCPU
    def test_native_layer_norm_cpu_decomp(self, device):
        # [FUNCTION DOC] CPU only. Computes native_layer_norm on bfloat16 tensors (eps 0.5),
        # then repeats it with fresh tensors under the Python dispatcher and FakeTensorMode,
        # and asserts each output dtype equals the real run.
        def f(x, w, b):
            # [FUNCTION DOC] HELPER: Calls native layer norm over normalized shape 1x2x3 with
            # weight, bias and eps 0.5; returns its outputs.
            return torch.ops.aten.native_layer_norm.default(x, [1, 2, 3], w, b, eps=0.5)

        x = torch.randn(1, 2, 3, dtype=torch.bfloat16, device="cpu")
        w = torch.randn(1, 2, 3, dtype=torch.bfloat16, requires_grad=True, device="cpu")
        b = torch.randn(1, 2, 3, dtype=torch.bfloat16, requires_grad=True, device="cpu")
        out_ref = f(x, w, b)

        from torch._subclasses.fake_tensor import FakeTensorMode

        with enable_python_dispatcher(), FakeTensorMode():
            x = torch.randn(1, 2, 3, dtype=torch.bfloat16, device="cpu")
            w = torch.randn(
                1, 2, 3, dtype=torch.bfloat16, requires_grad=True, device="cpu"
            )
            b = torch.randn(
                1, 2, 3, dtype=torch.bfloat16, requires_grad=True, device="cpu"
            )
            out = f(x, w, b)

        for o_ref, o in zip(out_ref, out):
            self.assertEqual(o_ref.dtype, o.dtype)

    @onlyCUDA
    @unittest.skipIf(not SM70OrLater, "triton")
    def test_rms_norm_decomp_cuda(self, device):
        # [FUNCTION DOC] CUDA only, needs SM70 or later. Compiles rms_norm followed by sinh,
        # runs forward and backward capturing generated code, and asserts the code strings
        # contain the expected fused triton kernel names for forward and for backward.
        @torch.compile
        def rms_norm_sinh(a, b, c):
            # [FUNCTION DOC] HELPER: Compiled function applying rms_norm with given normalized
            # shape and weight to a, then sinh to the result.
            output = torch.nn.functional.rms_norm(a, b, c)
            return torch.sinh(output)

        normalized_shape_arg = (3, 3, 3)
        input_tensor = torch.randn(3, 3, 3, device=device, requires_grad=True)
        weight_tensor = torch.randn(3, 3, 3, device=device, requires_grad=True)

        def forward_pass_fn():
            # [FUNCTION DOC] HELPER: Zero-argument closure running the compiled rms_norm_sinh
            # function on the enclosing input, normalized shape and weight tensors.
            return rms_norm_sinh(input_tensor, normalized_shape_arg, weight_tensor)

        model_output, generated_codes = torch._inductor.utils.run_fw_bw_and_get_code(
            forward_pass_fn
        )

        # check RMSNorm was fused with sinh
        self.assertTrue("triton_per_fused__fused_rms_norm_sinh" in generated_codes[0])
        self.assertTrue(
            "triton_per_fused__fused_rms_norm__fused_rms_norm_backward_cosh_mul"
            in generated_codes[1]
        )

    @onlyCUDA
    @skipIfCrossRef
    def test_addmm_out_dtype_decomp(self, device):
        # [FUNCTION DOC] CUDA only. For three beta/alpha settings, compares addmm with output
        # dtype float32 on bfloat16 matrices and float32 bias against the decomposition,
        # asserting result dtype float32 and equal values.
        cases = [
            {"beta": 1, "alpha": 1},
            {"beta": 0, "alpha": 1},
            {"beta": 2, "alpha": 3},
        ]
        for kwargs in cases:
            a = torch.randn(4, 8, dtype=torch.bfloat16, device=device)
            b = torch.randn(8, 4, dtype=torch.bfloat16, device=device)
            c = torch.randn(4, 4, dtype=torch.float32, device=device)

            ref = torch.ops.aten.addmm.dtype(c, a, b, torch.float32, **kwargs)
            res = torch._decomp.decompositions.addmm_dtype(
                c, a, b, out_dtype=torch.float32, **kwargs
            )

            self.assertEqual(res.dtype, torch.float32)
            self.assertEqual(res, ref)

    @onlyCPU
    @skipIfCrossRef
    def test_addmv_decomp_rejects_mixed_dtype(self, device):
        # [FUNCTION DOC] CPU only. Fetches the addmv decomposition and asserts it raises
        # RuntimeError mentioning same dtype when the input is float32 and matrix and vector
        # are int32.
        addmv_decomp = get_decompositions([aten.addmv.default])[aten.addmv.default]
        input = torch.tensor([2.7], dtype=torch.float32, device=device)
        mat = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device=device)
        vec = torch.tensor([1, 2], dtype=torch.int32, device=device)

        with self.assertRaisesRegex(RuntimeError, "same dtype"):
            addmv_decomp(input, mat, vec)

    @onlyCPU
    @skipIfCrossRef
    def test_linalg_vector_norm_decomp_correctness(self, device):
        # [FUNCTION DOC] CPU only. Compares the vector norm decomposition to
        # torch.linalg.vector_norm over cases of shapes, real and complex dtypes, orders
        # (including zero, infinities and fractions), dims, keepdim and output dtype,
        # asserting equality. Then runs gradcheck on the decomposition for three float64
        # inputs.
        decomp = decomposition_table[aten.linalg_vector_norm.default]

        def make_input(shape, dtype):
            # [FUNCTION DOC] HELPER: Makes a random tensor of the given shape and dtype on the
            # device; complex dtypes are built from random real and imaginary parts.
            if dtype.is_complex:
                real = torch.randn(shape, device=device)
                imag = torch.randn(shape, device=device)
                return (real + 1j * imag).to(dtype)
            return torch.randn(shape, device=device, dtype=dtype)

        cases = [
            ((2, 3), torch.float32, 2, None, False, None),
            ((2, 3), torch.float32, 0, 1, True, torch.float64),
            ((2, 1, 3), torch.float64, float("inf"), (-2, -1), False, None),
            ((2, 3), torch.float32, 3.5, (), True, None),
            ((1,), torch.float64, 6, (), False, None),
            ((2, 3), torch.complex64, 1, 1, True, torch.complex128),
            ((2, 3), torch.complex64, float("-inf"), None, False, None),
        ]
        for shape, input_dtype, ord, dim, keepdim, dtype in cases:
            x = make_input(shape, input_dtype)
            actual = decomp(x, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype)
            expected = torch.linalg.vector_norm(
                x, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype
            )
            self.assertEqual(actual, expected)

        gradcheck_cases = [
            (
                torch.tensor(-4.826984902407649, device=device, dtype=torch.float64),
                6,
                None,
                False,
            ),
            (
                torch.randn(2, 3, device=device, dtype=torch.float64) + 0.5,
                3.5,
                (),
                True,
            ),
            (torch.randn(2, 3, device=device, dtype=torch.float64) + 0.5, 2, 1, True),
        ]
        for x, ord, dim, keepdim in gradcheck_cases:
            x.requires_grad_()
            self.assertTrue(
                torch.autograd.gradcheck(
                    lambda x: decomp(x, ord=ord, dim=dim, keepdim=keepdim), (x,)
                )
            )


instantiate_device_type_tests(DecompOneOffTests, globals())


class HasDecompTest(TestCase):
    # [CLASS DOC] Checks coverage of the decomposition tables against the registered ATen
    # operators and tests a few inductor decompositions. Uses expect files for the lists.
    def setUp(self):
        # [FUNCTION DOC] Calls the base setUp and removes the diff size limit so long
        # expected-output differences print in full.
        super().setUp()
        self.maxDiff = None

    @staticmethod
    def _can_appear_in_trace(op: torch._ops.OpOverload) -> bool:
        # [FUNCTION DOC] HELPER: Static predicate: true if the op has a tensor type among its
        # schema arguments or returns and is not a CompositeImplicitAutograd op. A
        # RuntimeError whose message says the key does not exist (some jit-registered ops)
        # gives false; other RuntimeErrors propagate.
        has_tensor_arg = any(
            "Tensor" in str(a.type)
            for a in itertools.chain(op._schema.arguments, op._schema.returns)
        )
        if not has_tensor_arg:
            return False

        try:
            # CompositeImplicitAutograd ops are transparent to the tracer, so don't need decompositions
            return not _is_cia_op(op)
        except RuntimeError as e:
            # has_key fails for some jit-registered ops, which shouldn't be
            # relevant here anyway
            if "does not exist" in str(e):
                return False
            raise

    def test_has_decomposition(self):
        # [FUNCTION DOC] Collects all ATen overloads that can appear in a trace and lack an
        # entry in the global decomposition table (excluding a small allow list of ops only
        # registered in some CI configurations) and compares the sorted newline-joined op
        # names with the expect file.
        def all_aten_overloads():
            # [FUNCTION DOC] HELPER: Generator over every registered operator name in the aten
            # namespace, yielding the overload object (overload default when the name has no
            # suffix). Raises AssertionError if the packet is not an OpOverloadPacket.
            for name in torch._C._dispatch_get_all_op_names():
                if not name.startswith("aten::"):
                    continue

                name = name[6:]
                if "." in name:
                    packet_name, overload_name = name.split(".")
                else:
                    packet_name, overload_name = name, "default"

                packet = getattr(aten, packet_name)
                if not isinstance(packet, torch._ops.OpOverloadPacket):
                    raise AssertionError(
                        f"expected OpOverloadPacket, got {type(packet)}"
                    )
                op = getattr(packet, overload_name)
                yield op

        # This is for operators that are only registered in some CI
        # configurations, so would cause the test to fail
        allow_list = {aten.get_gradients.default}

        overloads_wanting_decomp = {
            op for op in all_aten_overloads() if self._can_appear_in_trace(op)
        }
        ops_missing_decomp = overloads_wanting_decomp - decomposition_table.keys()
        ops_missing_decomp -= allow_list
        self.assertExpected(
            "".join(sorted(op.name() + "\n" for op in ops_missing_decomp))
        )

    def test_aten_core_operators(self):
        # [FUNCTION DOC] Computes decomposition-table ops that can appear in a trace but are
        # not in the core ATen decompositions (so they decompose core operators) and compares
        # their sorted names with the expect file. Accept the new list with the
        # EXPECTTEST_ACCEPT environment variable, or add the decomposition to the core set.
        # If a decomposition isn't included in the core decompositions,
        # then it must decompose a core ATen operator.
        #
        # See NOTE [Core ATen Ops]
        #
        # If this test fails then either:
        # - Add the decomposition to torch._decomp.core_aten_decompositions,
        #   if decomposition should be used by inductor (not a core operator).
        # - Run this test again with EXPECTTEST_ACCEPT=1 to update the list of
        #   core ATen operators (and inductor will not use the decomposition).

        # Some decompositions are registered for CompositeImplicitAutograd
        # operators, which never appear in AOTAutograd's graph so are never used.
        useful_decomps = {
            op
            for op in decomposition_table
            if isinstance(op, torch._ops.OpOverload) and self._can_appear_in_trace(op)
        }
        core_decomps = torch._decomp.core_aten_decompositions().keys()
        core_aten_ops = useful_decomps - core_decomps
        self.assertExpected("".join(sorted(op.name() + "\n" for op in core_aten_ops)))

    def test_conv1d_decomposition(self):
        # [FUNCTION DOC] Compares native conv1d with the inductor conv1d to conv2d
        # decomposition across several settings of stride, padding, dilation, kernel size and
        # groups, using allclose.
        from torch._inductor.decomposition import conv1d_to_conv2d

        def check_case(
            # [FUNCTION DOC] HELPER: Seeds the RNG with zero, builds random input, weight and
            # bias with the given shapes, dtype and device, runs native conv1d and the
            # decomposition with the same list-valued stride, padding, dilation and groups,
            # and asserts allclose at 1e-5 relative and absolute tolerance.
            N=2,
            C_in=3,
            C_out=5,
            L=37,
            K=5,
            stride=2,
            padding=3,
            dilation=1,
            groups=1,
            dtype=torch.float32,
            device="cpu",
        ):
            torch.manual_seed(0)
            x = torch.randn(N, C_in, L, dtype=dtype, device=device)
            w = torch.randn(C_out, C_in // groups, K, dtype=dtype, device=device)
            b = torch.randn(C_out, dtype=dtype, device=device)

            ref = torch.ops.aten.conv1d.default(
                x,
                w,
                b,
                stride=[stride],
                padding=[padding],
                dilation=[dilation],
                groups=groups,
            )
            got = conv1d_to_conv2d(
                x,
                w,
                b,
                stride=[stride],
                padding=[padding],
                dilation=[dilation],
                groups=groups,
            )
            self.assertTrue(torch.allclose(ref, got, atol=1e-5, rtol=1e-5))

        # A few cases
        check_case()  # default
        check_case(stride=1, padding=0, K=3)
        check_case(stride=3, padding=4, K=7)
        check_case(dilation=2, padding=6, K=5)  # dilation
        check_case(groups=1, C_in=8, C_out=12)  # groups=1 bigger
        check_case(groups=2, C_in=8, C_out=12)  # grouped conv

    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    @torch._dynamo.config.patch("capture_dynamic_output_shape_ops", True)
    def test_mm_decompose_mm_dde(self):
        # [FUNCTION DOC] Compiles a long generated chain of batched matmuls (with embedding
        # lookup over indices from a data-dependent nonzero, rms_norm and gelu) with full
        # graph and dynamic shapes, with scalar-output and dynamic-shape-op capture enabled,
        # and asserts the result matches eager (allclose 1e-5). The inline comment says the
        # point is to avoid a NameError.
        def fuzzed_program(
            arg_0,
            arg_1,
            arg_2,
            arg_3,
            arg_4,
            arg_5,
            arg_6,
            arg_7,
            arg_8,
            arg_9,
            arg_10,
            arg_11,
            arg_12,
            arg_13,
            arg_14,
            arg_15,
            arg_16,
            arg_17,
            arg_18,
            sentinel,
        ):
            # [FUNCTION DOC] HELPER: Generated program taking nineteen tensor arguments plus a
            # sentinel. Multiplies float64 matrices and constants in a nested chain, uses an
            # embedding lookup with nonzero-derived indices, and returns the final result
            # times the sentinel (real part if complex).
            var_node_6 = (
                arg_0  # size=(9, 9, 9), stride=(81, 9, 1), dtype=float64, device=cuda
            )
            var_node_7 = (
                arg_1  # size=(9, 9, 11), stride=(99, 11, 1), dtype=float64, device=cuda
            )
            var_node_5 = torch.matmul(
                var_node_6.to(torch.float64), var_node_7.to(torch.float64)
            )  # size=(9, 9, 11), stride=(99, 11, 1), dtype=float64, device=cuda
            var_node_9 = torch.full(
                (9, 11, 12), 1.5758497316910556, dtype=torch.float64
            )  # size=(9, 11, 12), stride=(132, 12, 1), dtype=float64, device=cuda
            var_node_10 = (
                arg_2  # size=(9, 12, 8), stride=(96, 8, 1), dtype=float64, device=cuda
            )
            var_node_8 = torch.matmul(
                var_node_9.to(torch.float64), var_node_10.to(torch.float64)
            )  # size=(9, 11, 8), stride=(88, 8, 1), dtype=float64, device=cuda
            var_node_4 = torch.matmul(
                var_node_5.to(torch.float64), var_node_8.to(torch.float64)
            )  # size=(9, 9, 8), stride=(72, 8, 1), dtype=float64, device=cuda
            var_node_13 = arg_3  # size=(9, 8, 13), stride=(104, 13, 1), dtype=float64, device=cuda
            var_node_14 = (
                arg_4  # size=(9, 13, 7), stride=(91, 7, 1), dtype=float64, device=cuda
            )
            var_node_12 = torch.matmul(
                var_node_13.to(torch.float64), var_node_14.to(torch.float64)
            )  # size=(9, 8, 7), stride=(56, 7, 1), dtype=float64, device=cuda
            var_node_15 = arg_5  # size=(9, 7, 16), stride=(112, 16, 1), dtype=float64, device=cuda
            var_node_11 = torch.matmul(
                var_node_12.to(torch.float64), var_node_15.to(torch.float64)
            )  # size=(9, 8, 16), stride=(128, 16, 1), dtype=float64, device=cuda
            var_node_3 = torch.matmul(
                var_node_4.to(torch.float64), var_node_11.to(torch.float64)
            )  # size=(9, 9, 16), stride=(144, 16, 1), dtype=float64, device=cuda
            var_node_17 = arg_6  # size=(9, 16, 12), stride=(192, 12, 1), dtype=float64, device=cuda
            var_node_18 = arg_7  # size=(9, 12, 11), stride=(132, 11, 1), dtype=float64, device=cuda
            var_node_16 = torch.matmul(
                var_node_17.to(torch.float64), var_node_18.to(torch.float64)
            )  # size=(9, 16, 11), stride=(176, 11, 1), dtype=float64, device=cuda
            var_node_2 = torch.matmul(
                var_node_3.to(torch.float64), var_node_16.to(torch.float64)
            )  # size=(9, 9, 11), stride=(99, 11, 1), dtype=float64, device=cuda
            var_node_23 = torch.full(
                (156, 8), -0.5249394453404403, dtype=torch.float64
            )  # size=(156, 8), stride=(8, 1), dtype=float64, device=cuda
            var_node_24 = torch.full(
                (8, 9), 0.9331226188585692, dtype=torch.float64
            )  # size=(8, 9), stride=(9, 1), dtype=float64, device=cuda
            var_node_22 = torch.matmul(
                var_node_23.to(torch.float64), var_node_24.to(torch.float64)
            )  # size=(156, 9), stride=(9, 1), dtype=float64, device=cuda
            var_node_26 = torch.full(
                (9, 13), -0.9276381954691514, dtype=torch.float64
            )  # size=(9, 13), stride=(13, 1), dtype=float64, device=cuda
            var_node_27 = torch.full(
                (13, 16), 0.024752238943232543, dtype=torch.float64
            )  # size=(13, 16), stride=(16, 1), dtype=float64, device=cuda
            var_node_25 = torch.matmul(
                var_node_26.to(torch.float64), var_node_27.to(torch.float64)
            )  # size=(9, 16), stride=(16, 1), dtype=float64, device=cuda
            var_node_21 = torch.matmul(
                var_node_22.to(torch.float64), var_node_25.to(torch.float64)
            )  # size=(156, 16), stride=(16, 1), dtype=float64, device=cuda
            var_node_29 = arg_8
            _x_nz = torch.zeros(
                (9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                dtype=torch.bool,
                device=var_node_29.device,
            )
            _x_nz_flat = _x_nz.reshape(-1)
            _x_nz_flat[:9] = True
            var_node_28 = torch.nonzero(
                _x_nz
            )  # size=(9, 11), stride=(11, 1), dtype=int64, device=cuda
            var_node_20 = torch.nn.functional.embedding(
                torch.clamp(var_node_28.to(torch.int64), 0, var_node_21.size(0) - 1),
                var_node_21,
            )  # size=(9, 11, 16), stride=(176, 16, 1), dtype=float64, device=cuda
            var_node_33 = torch.full(
                (9, 16, 5), 1.0707914920634904, dtype=torch.float64
            )  # size=(9, 16, 5), stride=(80, 5, 1), dtype=float64, device=cuda
            var_node_34 = torch.full(
                (9, 5, 10), -0.44934093079047227, dtype=torch.float64
            )  # size=(9, 5, 10), stride=(50, 10, 1), dtype=float64, device=cuda
            var_node_32 = torch.matmul(
                var_node_33.to(torch.float64), var_node_34.to(torch.float64)
            )  # size=(9, 16, 10), stride=(160, 10, 1), dtype=float64, device=cuda
            var_node_36 = (
                arg_9  # size=(9, 10, 1), stride=(10, 1, 1), dtype=float64, device=cuda
            )
            var_node_37 = torch.full(
                (9, 1, 11), -1.874293687140311, dtype=torch.float64
            )  # size=(9, 1, 11), stride=(11, 11, 1), dtype=float64, device=cuda
            var_node_35 = torch.matmul(
                var_node_36.to(torch.float64), var_node_37.to(torch.float64)
            )  # size=(9, 10, 11), stride=(110, 11, 1), dtype=float64, device=cuda
            var_node_31 = torch.matmul(
                var_node_32.to(torch.float64), var_node_35.to(torch.float64)
            )  # size=(9, 16, 11), stride=(176, 11, 1), dtype=float64, device=cuda
            var_node_40 = torch.full(
                (990, 2), 0.4084376380351558, dtype=torch.float64
            )  # size=(990, 2), stride=(2, 1), dtype=float64, device=cuda
            var_node_41 = torch.full(
                (2,), 0.982671965550022, dtype=torch.float64
            )  # size=(2,), stride=(1,), dtype=float64, device=cuda
            var_node_39 = torch.matmul(
                var_node_40.to(torch.float64), var_node_41.to(torch.float64)
            )  # size=(990,), stride=(1,), dtype=float64, device=cuda
            var_node_38 = torch.reshape(
                var_node_39, [9, 11, 10]
            )  # size=(9, 11, 10), stride=(110, 10, 1), dtype=float64, device=cuda
            var_node_30 = torch.matmul(
                var_node_31.to(torch.float64), var_node_38.to(torch.float64)
            )  # size=(9, 16, 10), stride=(160, 10, 1), dtype=float64, device=cuda
            var_node_19 = torch.matmul(
                var_node_20.to(torch.float64), var_node_30.to(torch.float64)
            )  # size=(9, 11, 10), stride=(110, 10, 1), dtype=float64, device=cuda
            var_node_1 = torch.matmul(
                var_node_2.to(torch.float64), var_node_19.to(torch.float64)
            )  # size=(9, 9, 10), stride=(90, 10, 1), dtype=float64, device=cuda
            var_node_47 = arg_10  # size=(9, 10, 15), stride=(150, 15, 1), dtype=float64, device=cuda
            var_node_48 = torch.full(
                (9, 15, 2), -0.3349339402390618, dtype=torch.float64
            )  # size=(9, 15, 2), stride=(30, 2, 1), dtype=float64, device=cuda
            var_node_46 = torch.matmul(
                var_node_47.to(torch.float64), var_node_48.to(torch.float64)
            )  # size=(9, 10, 2), stride=(20, 2, 1), dtype=float64, device=cuda
            var_node_50 = (
                arg_11  # size=(9, 2, 7), stride=(14, 7, 1), dtype=float64, device=cuda
            )
            var_node_51 = (
                arg_12  # size=(9, 7, 2), stride=(14, 2, 1), dtype=float64, device=cuda
            )
            var_node_49 = torch.matmul(
                var_node_50.to(torch.float64), var_node_51.to(torch.float64)
            )  # size=(9, 2, 2), stride=(4, 2, 1), dtype=float64, device=cuda
            var_node_45 = torch.matmul(
                var_node_46.to(torch.float64), var_node_49.to(torch.float64)
            )  # size=(9, 10, 2), stride=(20, 2, 1), dtype=float64, device=cuda
            var_node_52 = torch.full(
                (9, 2, 1), -0.4046675639434615, dtype=torch.float64
            )  # size=(9, 2, 1), stride=(2, 1, 1), dtype=float64, device=cuda
            var_node_44 = torch.matmul(
                var_node_45.to(torch.float64), var_node_52.to(torch.float64)
            )  # size=(9, 10, 1), stride=(10, 1, 1), dtype=float64, device=cuda
            var_node_56 = (
                arg_13  # size=(9, 1, 1), stride=(1, 1, 1), dtype=float64, device=cuda
            )
            var_node_55 = torch.nn.functional.rms_norm(
                var_node_56.to(torch.float64), (1,)
            )  # size=(9, 1, 1), stride=(1, 1, 1), dtype=float64, device=cuda
            var_node_57 = torch.full(
                (9, 1, 8), 0.17877664640931384, dtype=torch.float64
            )  # size=(9, 1, 8), stride=(8, 8, 1), dtype=float64, device=cuda
            var_node_54 = torch.matmul(
                var_node_55.to(torch.float64), var_node_57.to(torch.float64)
            )  # size=(9, 1, 8), stride=(8, 8, 1), dtype=float64, device=cuda
            var_node_60 = arg_14  # size=(9, 8, 10), stride=(80, 10, 1), dtype=float64, device=cuda
            var_node_61 = torch.full(
                (9, 10, 6), 0.43614806380221494, dtype=torch.float64
            )  # size=(9, 10, 6), stride=(60, 6, 1), dtype=float64, device=cuda
            var_node_59 = torch.matmul(
                var_node_60.to(torch.float64), var_node_61.to(torch.float64)
            )  # size=(9, 8, 6), stride=(48, 6, 1), dtype=float64, device=cuda
            var_node_63 = (
                arg_15  # size=(9, 6, 3), stride=(18, 3, 1), dtype=float64, device=cuda
            )
            var_node_64 = torch.full(
                (9, 3, 8), -0.042774422041922854, dtype=torch.float64
            )  # size=(9, 3, 8), stride=(24, 8, 1), dtype=float64, device=cuda
            var_node_62 = torch.matmul(
                var_node_63.to(torch.float64), var_node_64.to(torch.float64)
            )  # size=(9, 6, 8), stride=(48, 8, 1), dtype=float64, device=cuda
            var_node_58 = torch.matmul(
                var_node_59.to(torch.float64), var_node_62.to(torch.float64)
            )  # size=(9, 8, 8), stride=(64, 8, 1), dtype=float64, device=cuda
            var_node_53 = torch.matmul(
                var_node_54.to(torch.float64), var_node_58.to(torch.float64)
            )  # size=(9, 1, 8), stride=(8, 8, 1), dtype=float64, device=cuda
            var_node_43 = torch.matmul(
                var_node_44.to(torch.float64), var_node_53.to(torch.float64)
            )  # size=(9, 10, 8), stride=(80, 8, 1), dtype=float64, device=cuda
            var_node_68 = arg_16  # size=(9, 8, 16), stride=(128, 16, 1), dtype=float64, device=cuda
            var_node_70 = torch.full(
                (9, 16, 15), 0.24947808634496438, dtype=torch.float64
            )  # size=(9, 16, 15), stride=(240, 15, 1), dtype=float64, device=cuda
            var_node_71 = torch.full(
                (9, 15, 7), -0.09035245509773453, dtype=torch.float64
            )  # size=(9, 15, 7), stride=(105, 7, 1), dtype=float64, device=cuda
            var_node_69 = torch.matmul(
                var_node_70.to(torch.float64), var_node_71.to(torch.float64)
            )  # size=(9, 16, 7), stride=(112, 7, 1), dtype=float64, device=cuda
            var_node_67 = torch.matmul(
                var_node_68.to(torch.float64), var_node_69.to(torch.float64)
            )  # size=(9, 8, 7), stride=(56, 7, 1), dtype=float64, device=cuda
            var_node_74 = torch.full(
                (9, 7, 1), 0.05671950481832341, dtype=torch.float64
            )  # size=(9, 7, 1), stride=(7, 1, 1), dtype=float64, device=cuda
            var_node_73 = torch.nn.functional.gelu(
                var_node_74
            )  # size=(9, 7, 1), stride=(7, 1, 1), dtype=float64, device=cuda
            var_node_76 = torch.full(
                (9, 1, 2), -0.019912810353597852, dtype=torch.float64
            )  # size=(9, 1, 2), stride=(2, 2, 1), dtype=float64, device=cuda
            var_node_77 = (
                arg_17  # size=(9, 2, 7), stride=(14, 7, 1), dtype=float64, device=cuda
            )
            var_node_75 = torch.matmul(
                var_node_76.to(torch.float64), var_node_77.to(torch.float64)
            )  # size=(9, 1, 7), stride=(7, 7, 1), dtype=float64, device=cuda
            var_node_72 = torch.matmul(
                var_node_73.to(torch.float64), var_node_75.to(torch.float64)
            )  # size=(9, 7, 7), stride=(49, 7, 1), dtype=float64, device=cuda
            var_node_66 = torch.matmul(
                var_node_67.to(torch.float64), var_node_72.to(torch.float64)
            )  # size=(9, 8, 7), stride=(56, 7, 1), dtype=float64, device=cuda
            var_node_78 = arg_18  # size=(9, 7, 13), stride=(91, 13, 1), dtype=float64, device=cuda
            var_node_65 = torch.matmul(
                var_node_66.to(torch.float64), var_node_78.to(torch.float64)
            )  # size=(9, 8, 13), stride=(104, 13, 1), dtype=float64, device=cuda
            var_node_42 = torch.matmul(
                var_node_43.to(torch.float64), var_node_65.to(torch.float64)
            )  # size=(9, 10, 13), stride=(130, 13, 1), dtype=float64, device=cuda
            var_node_0 = torch.matmul(
                var_node_1.to(torch.float64), var_node_42.to(torch.float64)
            )  # size=(9, 9, 13), stride=(117, 13, 1), dtype=float64, device=cuda
            # Ensure gradient computation by multiplying with sentinel and taking real part
            result = var_node_0 * sentinel
            if result.is_complex():
                result = result.real
            return result

        # Sentinel tensor to ensure gradient computation
        sentinel = torch.tensor(1.0, requires_grad=True)

        arg_0 = torch.as_strided(
            torch.randn(729).to(torch.float64), (9, 9, 9), (81, 9, 1)
        )
        arg_1 = torch.as_strided(
            torch.randn(891).to(torch.float64), (9, 9, 11), (99, 11, 1)
        )
        arg_2 = torch.as_strided(
            torch.randn(864).to(torch.float64), (9, 12, 8), (96, 8, 1)
        )
        arg_3 = torch.as_strided(
            torch.randn(936).to(torch.float64), (9, 8, 13), (104, 13, 1)
        )
        arg_4 = torch.as_strided(
            torch.randn(819).to(torch.float64), (9, 13, 7), (91, 7, 1)
        )
        arg_5 = torch.as_strided(
            torch.randn(1008).to(torch.float64), (9, 7, 16), (112, 16, 1)
        )
        arg_6 = torch.as_strided(
            torch.randn(1728).to(torch.float64), (9, 16, 12), (192, 12, 1)
        )
        arg_7 = torch.as_strided(
            torch.randn(1188).to(torch.float64), (9, 12, 11), (132, 11, 1)
        )
        arg_8 = torch.as_strided(
            torch.randint(0, 2, (1,), dtype=torch.int8).bool(),
            (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        )
        arg_9 = torch.as_strided(
            torch.randn(90).to(torch.float64), (9, 10, 1), (10, 1, 1)
        )
        arg_10 = torch.as_strided(
            torch.randn(1350).to(torch.float64), (9, 10, 15), (150, 15, 1)
        )
        arg_11 = torch.as_strided(
            torch.randn(126).to(torch.float64), (9, 2, 7), (14, 7, 1)
        )
        arg_12 = torch.as_strided(
            torch.randn(126).to(torch.float64), (9, 7, 2), (14, 2, 1)
        )
        arg_13 = torch.as_strided(
            torch.randn(9).to(torch.float64), (9, 1, 1), (1, 1, 1)
        )
        arg_14 = torch.as_strided(
            torch.randn(720).to(torch.float64), (9, 8, 10), (80, 10, 1)
        )
        arg_15 = torch.as_strided(
            torch.randn(162).to(torch.float64), (9, 6, 3), (18, 3, 1)
        )
        arg_16 = torch.as_strided(
            torch.randn(1152).to(torch.float64), (9, 8, 16), (128, 16, 1)
        )
        arg_17 = torch.as_strided(
            torch.randn(126).to(torch.float64), (9, 2, 7), (14, 7, 1)
        )
        arg_18 = torch.as_strided(
            torch.randn(819).to(torch.float64), (9, 7, 13), (91, 13, 1)
        )

        args = (
            arg_0,
            arg_1,
            arg_2,
            arg_3,
            arg_4,
            arg_5,
            arg_6,
            arg_7,
            arg_8,
            arg_9,
            arg_10,
            arg_11,
            arg_12,
            arg_13,
            arg_14,
            arg_15,
            arg_16,
            arg_17,
            arg_18,
        ) + (sentinel,)
        result_original = fuzzed_program(*args)
        compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
        result_compiled = compiled_program(*args)

        # Both should succeed without NameError
        self.assertTrue(
            torch.allclose(result_original, result_compiled, rtol=1e-5, atol=1e-5)
        )


if __name__ == "__main__":
    run_tests()
