# mypy: ignore-errors

# [MODULE DOC]
# OpInfo database for the torch.masked operators (masked.sum, prod, amax, amin, mean, norm,
# var, std, softmax, log_softmax, softmin, cumsum, cumprod, logaddexp, logsumexp, normalize
# and similar). Provides sample-input generators (dense, sparse COO and sparse CSR variants),
# a mask generator, a numpy reference wrapper for masked std/var, and the module-level list
# op_db of OpInfo/ReductionOpInfo entries consumed by the generic operator tests.
# Import-time side effects: builds op_db only; no global state is mutated.

import unittest
from collections.abc import Sequence
from functools import partial

import numpy as np

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import tol, toleranceOverride
from torch.testing._internal.common_dtype import (
    all_types_and,
    all_types_and_complex_and,
    complex_types,
    floating_and_complex_types_and,
    floating_types_and,
    integral_types,
)
from torch.testing._internal.opinfo.core import (
    DecorateInfo,
    gradcheck_wrapper_masked_operation,
    gradcheck_wrapper_masked_pointwise_operation,
    M,
    OpInfo,
    ReductionOpInfo,
    S,
    sample_inputs_reduction,
    SampleInput,
)
from torch.testing._internal.opinfo.utils import prod_numpy, reference_reduction_numpy


# Used for log_softmax, softmax, softmin
def sample_inputs_softmax_variant(
    op_info,
    device,
    dtype,
    requires_grad,
    with_dtype=False,
    use_zero_dimensions=True,
    **kwargs,
):
    # [FUNCTION DOC] Generator-style helper (returns a generator expression) of SampleInput
    # objects for softmax-like ops. Inputs are tensors of several shapes (including an empty
    # one when use_zero_dimensions is true, and a 0-d one except on xla), each paired with a
    # dim argument passed as positional args. With with_dtype, a dtype keyword (bfloat16 on
    # mps, else float64) is passed. Takes op_info, device, dtype, requires_grad.
    # KNOWN ISSUE: the incoming **kwargs argument is discarded because the name is rebound
    # to the dtype keyword mapping (or None).
    make_arg = partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )
    cases = [
        ((S,), (0,)),
        ((S, S), (0,)),
        ((S, S), (1,)),
        ((S, S), (-1,)),
        ((S, M, S), (2,)),
        *([((S, 0, 0), (-1,))] if use_zero_dimensions else []),
    ]
    kwargs = (
        dict(
            dtype=(
                torch.bfloat16 if torch.device(device).type == "mps" else torch.float64
            )
        )
        if with_dtype
        else None
    )

    # PyTorch on XLA throws an error when passed with dim argument for 0d tensor.
    # See https://github.com/pytorch/xla/issues/3061 for more details.
    if torch.device(device).type != "xla":
        cases.append(((), (0,)))

    return (
        SampleInput(make_arg(shape), args=dim, kwargs=kwargs) for shape, dim in cases
    )


def _generate_masked_op_mask(input_shape, device, **kwargs):
    # [FUNCTION DOC] HELPER: Generator yielding candidate masks for an input of input_shape:
    # first None (no mask), then a random bool mask of the full shape. For inputs with more
    # than two dims it also yields masks broadcastable to the input: last, middle and first
    # dim collapsed to size one, a mask with one fewer leading dim, and a 1-d mask matching
    # the last dim. Masks never require gradients. Extra kwargs are ignored.
    make_arg = partial(
        make_tensor, dtype=torch.bool, device=device, requires_grad=False
    )
    yield None
    yield make_arg(input_shape)
    if len(input_shape) > 2:
        # broadcast last mask dimension:
        yield make_arg(input_shape[:-1] + (1,))
        # broadcast middle mask dimension:
        yield make_arg(input_shape[:1] + (1,) + input_shape[2:])
        # broadcast first mask dimension:
        yield make_arg((1,) + input_shape[1:])
        # mask.ndim < input.ndim
        yield make_arg(input_shape[1:])
        # mask.ndim == 1
        yield make_arg(input_shape[-1:])
        # masks that require broadcasting of inputs (mask.ndim >
        # input.ndim) will not be supported, however, we may
        # reconsider this if there will be demand on this kind of
        # degenerate cases.


def sample_inputs_masked_reduction(op_info, device, dtype, requires_grad, **kwargs):
    # [FUNCTION DOC] Generator of SampleInput objects for masked reductions. Takes every
    # sample from the generic reduction sample generator and crosses it with each mask from
    # the mask generator, passing the mask as a keyword. Reads supports_multiple_dims from
    # op_info and stores it in kwargs. For non-grad floating 2-d inputs with a full-shape
    # mask it also yields variants whose diagonal is filled with inf, -inf and nan.
    """Sample inputs for masked reduction operators.

    Masked reduction operator is a reduction operator with trailing
    mask optional argument. A mask is a bool tensor with the same
    shape as input or a shape that is broadcastable to input shape.
    """
    kwargs["supports_multiple_dims"] = op_info.supports_multiple_dims

    for sample_input in sample_inputs_reduction(
        op_info, device, dtype, requires_grad, **kwargs
    ):
        for mask in _generate_masked_op_mask(
            sample_input.input.shape, device, **kwargs
        ):
            sample_input_args, sample_input_kwargs = (
                sample_input.args,
                dict(mask=mask, **sample_input.kwargs),
            )
            yield SampleInput(
                sample_input.input.detach().requires_grad_(requires_grad),
                args=sample_input_args,
                kwargs=sample_input_kwargs,
            )
            if (
                not requires_grad
                and dtype.is_floating_point
                and sample_input.input.ndim == 2
                and mask is not None
                and mask.shape == sample_input.input.shape
            ):
                for v in [torch.inf, -torch.inf, torch.nan]:
                    t = sample_input.input.detach()
                    t.diagonal(0, -2, -1).fill_(v)
                    yield SampleInput(
                        t.requires_grad_(requires_grad),
                        args=sample_input_args,
                        kwargs=sample_input_kwargs,
                    )


def sample_inputs_sparse_coo_masked_reduction(
    op_info, device, dtype, requires_grad, **kwargs
):
    # [FUNCTION DOC] Generator of sparse COO sample inputs for masked reductions. Yields
    # nothing unless op_info.supports_sparse. Converts the input and any mask of each dense
    # masked-reduction sample to sparse COO. Samples without a mask are skipped for prod,
    # amax and amin because their reduction identity is nonzero.
    """Sample inputs for masked reduction operators that support inputs
    with sparse coo layouts.
    """
    if op_info.supports_sparse:
        op_name = op_info.name.replace("masked.", "")
        for sample_input in sample_inputs_masked_reduction(
            op_info, device, dtype, requires_grad, **kwargs
        ):
            mask = sample_input.kwargs.get("mask")
            if mask is not None:
                sample_input_kwargs = sample_input.kwargs.copy()
                sample_input_kwargs.update(mask=mask.to_sparse())
                yield SampleInput(
                    sample_input.input.to_sparse(),
                    args=sample_input.args,
                    kwargs=sample_input_kwargs,
                )
            else:
                if op_name in {"prod", "amax", "amin"}:
                    # FIXME: for now reductions with non-zero reduction identity and
                    # unspecified mask are not supported for sparse COO
                    # tensors, see torch.masked.prod implementation
                    # for details.
                    continue
                yield SampleInput(
                    sample_input.input.to_sparse(),
                    args=sample_input.args,
                    kwargs=sample_input.kwargs,
                )


def sample_inputs_sparse_csr_masked_reduction(
    op_info, device, dtype, requires_grad, **kwargs
):
    # [FUNCTION DOC] Generator of sparse CSR sample inputs for masked reductions. Yields
    # nothing unless op_info.supports_sparse_csr. Keeps only 2-d samples with keepdim set,
    # converts input and mask to CSR, and skips unmasked samples for prod, amax, amin and
    # mean. For each sample with dim equal to zero it also yields a copy with dim one so
    # both inner and outer dimension implementations are covered.
    """Sample inputs for masked reduction operators that support inputs
    with sparse csr layouts.
    """
    if op_info.supports_sparse_csr:
        op_name = op_info.name.replace("masked.", "")
        for sample_input in sample_inputs_masked_reduction(
            op_info, device, dtype, requires_grad, **kwargs
        ):
            if not (
                sample_input.input.ndim == 2 and sample_input.kwargs.get("keepdim")
            ):
                # - sparse CSR tensors are always 2-D tensors
                # - masked reduction on CSR tensors are defined only if keepdim is True.
                continue
            mask = sample_input.kwargs.get("mask")
            if mask is not None:
                sample_input_kwargs = sample_input.kwargs.copy()
                sample_input_kwargs.update(mask=mask.to_sparse_csr())
                new_sample = SampleInput(
                    sample_input.input.to_sparse_csr(),
                    args=sample_input.args,
                    kwargs=sample_input_kwargs,
                )
            else:
                if op_name in ["prod", "amax", "amin", "mean"]:
                    # reductions with non-zero reduction identity and
                    # unspecified mask is not supported for sparse CSR
                    # tensors, see torch.masked.prod implementation
                    # for details.
                    continue
                new_sample = SampleInput(
                    sample_input.input.to_sparse_csr(),
                    args=sample_input.args,
                    kwargs=sample_input.kwargs,
                )
            yield new_sample
            if sample_input.kwargs["dim"] == 0:
                # Reductions of CSR tensors use different implementations for
                # inner and/or outer dimensions. So, as a minimum of testing CSR
                # implementations the following kwargs must be generated:
                #   dict(dim=0, keepdim=True)
                #   dict(dim=1, keepdim=True)
                #   dict(dim=(0, 1), keepdim=True)
                # Here we generate the dim=1 case from the dim=0 case.
                sample_input_kwargs = new_sample.kwargs.copy()
                sample_input_kwargs.update(dim=1)
                yield SampleInput(
                    new_sample.input.clone(),
                    args=sample_input.args,
                    kwargs=sample_input_kwargs,
                )


def sample_inputs_masked_norm(op_info, device, dtype, requires_grad, **kwargs):
    # [FUNCTION DOC] Generator of SampleInput objects for masked norm: for each norm order
    # (2.0, 1, inf, -inf, 0) it takes every masked-reduction sample and prepends the order
    # to the positional args, using a cloned input.
    """Sample inputs for masked norm."""
    for ord in [2.0, 1, float("inf"), float("-inf"), 0]:
        for sample_input in sample_inputs_masked_reduction(
            op_info, device, dtype, requires_grad, **kwargs
        ):
            sample_input_args, sample_input_kwargs = (
                (ord,) + sample_input.args,
                sample_input.kwargs.copy(),
            )
            yield SampleInput(
                sample_input.input.clone().requires_grad_(requires_grad),
                args=sample_input_args,
                kwargs=sample_input_kwargs,
            )


def reference_masked_std_var(
    numpy_fn,
):
    # [FUNCTION DOC] HELPER: Builds a numpy reference function for masked std/var from the
    # given numpy function (such as std or var) by wrapping it with the generic numpy
    # reduction reference. Returns the wrapper function.
    ref = reference_reduction_numpy(numpy_fn)

    # Translate unbiased or correction arguments into ddof
    def func(
        input,
        dim=None,
        unbiased=None,
        *,
        correction=None,
        **kwargs,
    ):
        # [FUNCTION DOC] HELPER: Reference call. Computes the delta degrees of freedom:
        # default one, zero or one from unbiased if given, and correction overrides both.
        # A sequence dim becomes a tuple. Returns the wrapped numpy reduction result.
        ddof = 1
        if unbiased is not None:
            ddof = 1 if unbiased else 0
        if correction is not None:
            ddof = correction

        if isinstance(dim, Sequence):
            dim = tuple(dim)

        return ref(input, dim, ddof=ddof, **kwargs)

    return func


def sample_inputs_masked_std_var(op_info, device, dtype, requires_grad, **kwargs):
    # [FUNCTION DOC] Generator of SampleInput objects for masked std and var. Builds samples
    # from the generic std/var generator crossed with masks (and inf/nan diagonal variants),
    # skipping samples whose first positional arg is a bool. Then drops samples where the
    # smallest count of unmasked elements along dim is at most correction plus one, since
    # these would give nan. Correction defaults to the unbiased flag (true) as an integer.
    """Sample inputs for masked std/var."""
    kwargs["supports_multiple_dims"] = op_info.supports_multiple_dims
    from torch.testing._internal.common_methods_invocations import sample_inputs_std_var

    def masked_samples():
        # [FUNCTION DOC] HELPER: Generator of std/var samples crossed with masks, same
        # scheme as the masked reduction generator, minus samples with a bool first arg.
        for sample_input in sample_inputs_std_var(
            op_info, device, dtype, requires_grad, **kwargs
        ):
            if len(sample_input.args) and isinstance(sample_input.args[0], bool):
                continue  # masked.{std, var} doesn't support `.var(unbiased)`

            for mask in _generate_masked_op_mask(
                sample_input.input.shape, device, **kwargs
            ):
                sample_input_args, sample_input_kwargs = (
                    sample_input.args,
                    dict(mask=mask, **sample_input.kwargs),
                )
                yield SampleInput(
                    sample_input.input.detach().requires_grad_(requires_grad),
                    args=sample_input_args,
                    kwargs=sample_input_kwargs,
                )
                if (
                    not requires_grad
                    and dtype.is_floating_point
                    and sample_input.input.ndim == 2
                    and mask is not None
                    and mask.shape == sample_input.input.shape
                ):
                    for v in [torch.inf, -torch.inf, torch.nan]:
                        t = sample_input.input.detach()
                        t.diagonal(0, -2, -1).fill_(v)
                        yield SampleInput(
                            t.requires_grad_(requires_grad),
                            args=sample_input_args,
                            kwargs=sample_input_kwargs,
                        )

    for sample_input in masked_samples():
        correction = sample_input.kwargs.get("correction")
        if correction is None:
            correction = int(sample_input.kwargs.get("unbiased", True))

        dim = sample_input.kwargs.get("dim", None)

        if sample_input.kwargs.get("mask") is None:
            orig_count = torch.masked.sum(
                torch.ones(sample_input.input.shape, dtype=torch.int64),
                dim,
                keepdim=True,
            )
        else:
            inmask = torch.masked._input_mask(
                sample_input.input, *sample_input.args, **sample_input.kwargs
            )
            orig_count = torch.masked.sum(
                inmask.new_ones(sample_input.input.shape, dtype=torch.int64),
                dim,
                keepdim=True,
                mask=inmask,
            )
        if orig_count.min() <= correction + 1:
            # Skip samples that lead to nans in var computation
            continue

        yield sample_input


def sample_inputs_masked_softmax(
    op_info, device, dtype, requires_grad, with_dtype=False, **kwargs
):
    # [FUNCTION DOC] Generator of SampleInput objects for masked softmax, log_softmax and
    # softmin. Crosses every softmax-variant sample (optionally with a dtype keyword when
    # with_dtype is set) with each generated mask, passing the mask as a keyword and a
    # cloned input.
    """Sample inputs for masked softmax, log_softmax, and softmin.

    Masked normalization operator is a reduction operator with
    trailing mask optional argument. A mask is a bool tensor with the
    same shape as input or a shape that is broadcastable to input
    shape.
    """
    for sample_input in sample_inputs_softmax_variant(
        op_info, device, dtype, requires_grad, with_dtype=with_dtype, **kwargs
    ):
        for mask in _generate_masked_op_mask(
            sample_input.input.shape, device, **kwargs
        ):
            yield SampleInput(
                sample_input.input.clone().requires_grad_(requires_grad),
                *sample_input.args,
                mask=mask,
                **sample_input.kwargs,
            )


def sample_inputs_masked_cumops(op_info, device, dtype, requires_grad, **kwargs):
    # [FUNCTION DOC] Generator of SampleInput objects for masked cumsum and cumprod. Uses
    # softmax-variant samples crossed with masks, but only real tensor masks (None is
    # skipped). Removes any keepdim keyword and ensures dim is passed positionally, skipping
    # samples with no dim at all.
    """Sample inputs for masked cumsum and cumprod."""
    for sample_input in sample_inputs_softmax_variant(
        op_info, device, dtype, requires_grad, **kwargs
    ):
        for mask in _generate_masked_op_mask(
            sample_input.input.shape, device, **kwargs
        ):
            if type(mask) is not torch.Tensor:
                continue
            sample_input_args, sample_input_kwargs = (
                sample_input.args,
                dict(mask=mask, **sample_input.kwargs),
            )
            if "keepdim" in sample_input_kwargs:
                sample_input_kwargs.pop("keepdim")
            # dimension is required
            if sample_input_args:
                dim = sample_input.args[0]
            else:
                if "dim" not in sample_input_kwargs:
                    continue
                dim = sample_input_kwargs.pop("dim")
                sample_input_args = (dim,)
            yield SampleInput(
                sample_input.input.clone().requires_grad_(requires_grad),
                *sample_input_args,
                **sample_input_kwargs,
            )


def sample_inputs_masked_logaddexp(op_info, device, dtype, requires_grad, **kwargs):
    # [FUNCTION DOC] Generator of SampleInput objects for masked logaddexp. For three shapes
    # it pairs two random tensors with an input_mask and an other_mask given as keywords;
    # the masks are drawn from two separately generated lists of the same length, paired
    # position by position (strict zip, raises on length mismatch).
    """Sample inputs for masked logaddexp."""
    shapes = [(S,), (S, S), (S, M, S)]
    input_mask_lists = [
        list(_generate_masked_op_mask(shape, device, **kwargs)) for shape in shapes
    ]
    other_mask_lists = [
        list(_generate_masked_op_mask(shape, device, **kwargs)) for shape in shapes
    ]

    make_arg = partial(
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
    )
    for shape, input_masks, other_masks in zip(
        shapes, input_mask_lists, other_mask_lists, strict=True
    ):
        for input_mask, other_mask in zip(input_masks, other_masks, strict=True):
            yield SampleInput(
                make_arg(shape),
                make_arg(shape),
                input_mask=input_mask,
                other_mask=other_mask,
            )


def sample_inputs_masked_normalize(op_info, device, dtype, requires_grad, **kwargs):
    # [FUNCTION DOC] Generator of SampleInput objects for masked normalize: for each norm
    # order (2.0, 1, inf, -inf, 0) and each softmax-variant sample without empty-dimension
    # shapes, yields a cloned input with the order as first positional arg, followed by
    # the sample's args and kwargs. No mask is generated here.
    """Sample inputs for masked normalize."""
    for ord in [2.0, 1, float("inf"), float("-inf"), 0]:
        for sample_input in sample_inputs_softmax_variant(
            op_info, device, dtype, requires_grad, use_zero_dimensions=False, **kwargs
        ):
            yield SampleInput(
                sample_input.input.clone().requires_grad_(requires_grad),
                ord,
                *sample_input.args,
                **sample_input.kwargs,
            )


op_db: list[OpInfo] = [
    ReductionOpInfo(
        "masked.sum",
        ref=reference_reduction_numpy(np.sum),
        method_variant=None,
        identity=0,
        nan_policy="propagate",
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        supports_sparse=True,
        supports_sparse_csr=True,
        promotes_int_to_int64=True,
        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
        skips=(
            DecorateInfo(
                unittest.skip("Failing on some jobs"),
                "TestReductions",
                "test_reference_masked",
                dtypes=(torch.bool, torch.int8, torch.int16, torch.int32),
            ),
            # FIXME: improve precision
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestReductions",
                "test_reference_masked",
                dtypes=(torch.float16,),
            ),
            DecorateInfo(
                unittest.expectedFailure,
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
            # FIXME: sum reduces all dimensions when dim=[]
            DecorateInfo(unittest.expectedFailure, "TestReductions", "test_dim_empty"),
            DecorateInfo(
                unittest.expectedFailure, "TestReductions", "test_dim_empty_keepdim"
            ),
            # RuntimeError: undefined value tensor
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
            # Driver issue of XPU, see https://github.com/intel/torch-xpu-ops/issues/2295
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestReductions",
                "test_ref_small_input",
                device_type="xpu",
                dtypes=[torch.complex128],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestReductions",
                "test_reference_masked",
                device_type="xpu",
                dtypes=[torch.complex128],
            ),
        ),
        decorators=[
            DecorateInfo(
                toleranceOverride(
                    {
                        torch.bfloat16: tol(atol=1e-03, rtol=5e-2),
                        torch.float16: tol(atol=1e-03, rtol=5e-3),
                    }
                ),
                "TestReductions",
                "test_reference_masked",
            ),
            DecorateInfo(
                toleranceOverride({torch.float16: tol(atol=1e-02, rtol=1e-03)}),
                "TestReductions",
                "test_ref_small_input",
            ),
            DecorateInfo(
                toleranceOverride(
                    {
                        torch.bfloat16: tol(atol=0.1, rtol=0.1),
                        torch.float16: tol(atol=5e-3, rtol=5e-3),
                    }
                ),
                "TestMasked",
                "test_mask_layout",
            ),
        ],
        sample_inputs_func=sample_inputs_masked_reduction,
        sample_inputs_sparse_coo_func=sample_inputs_sparse_coo_masked_reduction,
        sample_inputs_sparse_csr_func=sample_inputs_sparse_csr_masked_reduction,
    ),
    ReductionOpInfo(
        "masked.prod",
        ref=prod_numpy,
        method_variant=None,
        identity=1,
        nan_policy="propagate",
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        supports_sparse=True,
        supports_sparse_csr=True,
        promotes_int_to_int64=True,
        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
        skips=(
            DecorateInfo(
                unittest.expectedFailure,
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
            DecorateInfo(
                unittest.skip("Failing on some jobs"),
                "TestReductions",
                "test_reference_masked",
                dtypes=(torch.bool, torch.int8, torch.int16, torch.int32),
            ),
            DecorateInfo(
                "TestReductions",
                "test_ref_small_input",
                dtypes=(torch.int8, torch.int16, torch.int32),
            ),
            # FIXME: "cuda_scatter_gather_base_kernel_func" not implemented for ... (used for sparse_coo inputs)
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestMasked",
                "test_mask_layout",
                device_type="cuda",
                dtypes=(torch.bool, *integral_types(), *complex_types()),
            ),
            # Driver issue of XPU, see https://github.com/intel/torch-xpu-ops/issues/2295
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestReductions",
                "test_reference_masked",
                device_type="xpu",
                dtypes=(torch.complex128, torch.int64, torch.uint8),
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestReductions",
                "test_ref_small_input",
                device_type="xpu",
                dtypes=(
                    torch.int64,
                    torch.int32,
                    torch.int16,
                    torch.int8,
                    torch.complex128,
                ),
            ),
        ),
        decorators=[
            DecorateInfo(
                toleranceOverride({torch.float16: tol(atol=1e-03, rtol=1e-02)}),
                "TestReductions",
                "test_reference_masked",
            ),
            DecorateInfo(
                toleranceOverride({torch.float16: tol(atol=1e-03, rtol=1e-03)}),
                "TestReductions",
                "test_ref_duplicate_values",
            ),
            DecorateInfo(
                toleranceOverride({torch.float16: tol(atol=1e-03, rtol=1e-03)}),
                "TestReductions",
                "test_ref_small_input",
            ),
            DecorateInfo(
                toleranceOverride({torch.float16: tol(atol=1e-02, rtol=1.5e-03)}),
                "TestMasked",
                "test_mask_layout",
                device_type="cpu",
            ),
            DecorateInfo(
                toleranceOverride({torch.float32: tol(atol=1e-05, rtol=1e-05)}),
                "TestOperators",
                "test_jvp",
                device_type="cuda",
            ),
        ],
        sample_inputs_func=sample_inputs_masked_reduction,
        sample_inputs_sparse_coo_func=sample_inputs_sparse_coo_masked_reduction,
        sample_inputs_sparse_csr_func=sample_inputs_sparse_csr_masked_reduction,
    ),
    OpInfo(
        "masked.cumsum",
        dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16),
        method_variant=None,
        # Runs very slowly on slow gradcheck - alternatively reduce input sizes
        gradcheck_fast_mode=True,
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        skips=(
            DecorateInfo(
                unittest.expectedFailure,
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
            # NotSupportedError: Compiled functions can't ... use keyword-only arguments with defaults
            DecorateInfo(
                unittest.skip("Skipped!"), "TestJit", "test_variant_consistency_jit"
            ),
            # The following dtypes worked in forward but are not listed by the OpInfo: {torch.bool}.
            DecorateInfo(
                unittest.expectedFailure, "TestCommon", "test_dtypes", device_type="mps"
            ),
        ),
        # Can reuse the same inputs; dim is required in both
        sample_inputs_func=sample_inputs_masked_cumops,
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,
    ),
    OpInfo(
        "masked.cumprod",
        dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16),
        method_variant=None,
        # Runs very slowly on slow gradcheck - alternatively reduce input sizes
        gradcheck_fast_mode=True,
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        skips=(
            # NotSupportedError: Compiled functions can't ... use keyword-only arguments with defaults
            DecorateInfo(
                unittest.expectedFailure,
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
            # NotSupportedError: Compiled functions can't ... use keyword-only arguments with defaults
            DecorateInfo(
                unittest.skip("Skipped!"), "TestJit", "test_variant_consistency_jit"
            ),
            DecorateInfo(
                toleranceOverride({torch.float32: tol(atol=1e-5, rtol=1e-5)}),
                "TestCompositeCompliance",
                "test_backward",
                device_type="cuda",
            ),
            DecorateInfo(
                toleranceOverride({torch.float16: tol(atol=1e-2, rtol=2.6e-3)}),
                "TestInductorOpInfo",
                "test_comprehensive",
                device_type="cuda",
            ),
            # The following dtypes worked in forward but are not listed by the OpInfo: {torch.bool}.
            DecorateInfo(
                unittest.expectedFailure, "TestCommon", "test_dtypes", device_type="mps"
            ),
        ),
        # Can reuse the same inputs; dim is required in both
        sample_inputs_func=sample_inputs_masked_cumops,
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,
    ),
    ReductionOpInfo(
        "masked.amax",
        nan_policy="propagate",
        supports_out=False,
        dtypes=all_types_and(torch.float16, torch.bfloat16),
        supports_sparse=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        supports_sparse_csr=True,
        ref=reference_reduction_numpy(np.amax),
        skips=(
            DecorateInfo(
                unittest.expectedFailure,
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
            # FIXME: amax reduces all dimensions when dim=[]
            DecorateInfo(unittest.expectedFailure, "TestReductions", "test_dim_empty"),
            DecorateInfo(
                unittest.expectedFailure, "TestReductions", "test_dim_empty_keepdim"
            ),
            # RuntimeError: Unknown builtin op: aten::iinfo
            DecorateInfo(
                unittest.skip("Skipped!"), "TestJit", "test_variant_consistency_jit"
            ),
            # FIXME: "cuda_scatter_gather_base_kernel_func" not implemented for ... (used for sparse_coo inputs)
            # FIXME: "_segment_reduce_lengths_cpu/cuda" not implemented for ... (used for sparse_csr inputs)
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestMasked",
                "test_mask_layout",
                dtypes=(torch.bool, *integral_types(), *complex_types()),
            ),
            # Driver issue of XPU, see https://github.com/intel/torch-xpu-ops/issues/2295
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestReductions",
                "test_reference_masked",
                device_type="xpu",
                dtypes=[torch.int64],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestReductions",
                "test_ref_small_input",
                device_type="xpu",
                dtypes=[torch.int64],
            ),
        ),
        sample_inputs_func=sample_inputs_masked_reduction,
        sample_inputs_sparse_coo_func=sample_inputs_sparse_coo_masked_reduction,
        sample_inputs_sparse_csr_func=sample_inputs_sparse_csr_masked_reduction,
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,
    ),
    ReductionOpInfo(
        "masked.amin",
        nan_policy="propagate",
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        dtypes=all_types_and(torch.float16, torch.bfloat16),
        supports_sparse=True,
        supports_sparse_csr=True,
        ref=reference_reduction_numpy(np.amin),
        skips=(
            DecorateInfo(
                unittest.expectedFailure,
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
            # FIXME: amax reduces all dimensions when dim=[]
            DecorateInfo(unittest.expectedFailure, "TestReductions", "test_dim_empty"),
            DecorateInfo(
                unittest.expectedFailure, "TestReductions", "test_dim_empty_keepdim"
            ),
            # RuntimeError: Unknown builtin op: aten::iinfo
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
            # FIXME: "cuda_scatter_gather_base_kernel_func" not implemented for ... (used for sparse_coo inputs)
            # FIXME: "_segment_reduce_lengths_cpu/cuda" not implemented for ... (used for sparse_csr inputs)
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestMasked",
                "test_mask_layout",
                dtypes=(torch.bool, *integral_types(), *complex_types()),
            ),
            # Driver issue of XPU, see https://github.com/intel/torch-xpu-ops/issues/2295
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestReductions",
                "test_reference_masked",
                device_type="xpu",
                dtypes=[torch.int64],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestReductions",
                "test_ref_small_input",
                device_type="xpu",
                dtypes=[torch.int64],
            ),
        ),
        sample_inputs_func=sample_inputs_masked_reduction,
        sample_inputs_sparse_coo_func=sample_inputs_sparse_coo_masked_reduction,
        sample_inputs_sparse_csr_func=sample_inputs_sparse_csr_masked_reduction,
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,
    ),
    ReductionOpInfo(
        "masked.argmax",
        supports_out=False,
        supports_multiple_dims=False,
        supports_autograd=False,
        dtypes=all_types_and(torch.float16, torch.bfloat16),
        ref=reference_reduction_numpy(np.argmax, supports_keepdims=False),
        skips=(
            DecorateInfo(
                unittest.expectedFailure,
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
            # initial is not a keyword for argmax
            DecorateInfo(
                unittest.expectedFailure, "TestReductions", "test_reference_masked"
            ),
            # NotSupportedError: Compiled functions can't ... use keyword-only arguments with defaults
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
            # Driver issue of XPU, see https://github.com/intel/torch-xpu-ops/issues/2295
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestReductions",
                "test_ref_small_input",
                device_type="xpu",
                dtypes=floating_types_and(
                    torch.int8, torch.int16, torch.int32, torch.int64, torch.float16
                ),
            ),
        ),
        sample_inputs_func=sample_inputs_masked_reduction,
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,
    ),
    ReductionOpInfo(
        "masked.argmin",
        supports_out=False,
        supports_multiple_dims=False,
        supports_autograd=False,
        dtypes=all_types_and(torch.float16, torch.bfloat16),
        ref=reference_reduction_numpy(np.argmin, supports_keepdims=False),
        skips=(
            DecorateInfo(
                unittest.expectedFailure,
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
            # initial is not a keyword for argmin
            DecorateInfo(
                unittest.expectedFailure, "TestReductions", "test_reference_masked"
            ),
            # NotSupportedError: Compiled functions can't ... use keyword-only arguments with defaults
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
            # Driver issue of XPU, see https://github.com/intel/torch-xpu-ops/issues/2295
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestReductions",
                "test_ref_small_input",
                device_type="xpu",
                dtypes=floating_types_and(
                    torch.int8, torch.int16, torch.int32, torch.int64, torch.float16
                ),
            ),
        ),
        sample_inputs_func=sample_inputs_masked_reduction,
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,
    ),
    ReductionOpInfo(
        "masked.mean",
        ref=reference_reduction_numpy(np.mean)
        if np.lib.NumpyVersion(np.__version__) >= "1.20.2"
        else None,
        method_variant=None,
        nan_policy="propagate",
        supports_out=False,
        supports_sparse_csr=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        promotes_int_to_float=True,
        dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
        skips=(
            DecorateInfo(
                unittest.expectedFailure,
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
            # FIXME: sum reduces all dimensions when dim=[]
            DecorateInfo(unittest.expectedFailure, "TestReductions", "test_dim_empty"),
            DecorateInfo(
                unittest.expectedFailure, "TestReductions", "test_dim_empty_keepdim"
            ),
            # RuntimeError: undefined value tensor
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
            # FIXME: "_segment_reduce_lengths_cpu/cuda" not implemented for ... (used for sparse_csr inputs)
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestMasked",
                "test_mask_layout",
                dtypes=(torch.bool, *integral_types(), *complex_types()),
            ),
            # Driver issue of XPU, see https://github.com/intel/torch-xpu-ops/issues/2295
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestReductions",
                "test_ref_small_input",
                device_type="xpu",
                dtypes=[torch.complex128],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestReductions",
                "test_reference_masked",
                device_type="xpu",
                dtypes=[torch.complex128],
            ),
        ),
        decorators=[
            DecorateInfo(
                toleranceOverride(
                    {
                        torch.bfloat16: tol(atol=1e-03, rtol=0.05),
                        torch.float16: tol(atol=1e-03, rtol=1e-03),
                    }
                ),
                "TestReductions",
                "test_reference_masked",
            ),
            DecorateInfo(
                toleranceOverride({torch.float16: tol(atol=1e-03, rtol=1e-03)}),
                "TestReductions",
                "test_ref_small_input",
            ),
            DecorateInfo(
                toleranceOverride({torch.float16: tol(atol=1e-03, rtol=2e-03)}),
                "TestSparseCompressed",
                "test_consistency",
                device_type="cuda",
            ),
        ],
        sample_inputs_func=sample_inputs_masked_reduction,
        sample_inputs_sparse_csr_func=sample_inputs_sparse_csr_masked_reduction,
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,
    ),
    OpInfo(
        "masked.median",
        dtypes=floating_types_and(torch.bfloat16, torch.float16),
        method_variant=None,
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        skips=(
            DecorateInfo(
                unittest.expectedFailure,
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
            # NotSupportedError: Compiled functions can't ... use keyword-only arguments with defaults
            DecorateInfo(
                unittest.skip("Skipped!"), "TestJit", "test_variant_consistency_jit"
            ),
            # masked.median raises ValueError for fully-masked rows on
            # non-floating dtypes, but the trivial 0-d sample exercised by
            # test_dtypes succeeds for ints / bool / complex64, fooling the
            # detector into believing these dtypes are supported.
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_dtypes",
                device_type="mps",
            ),
        ),
        sample_inputs_func=partial(
            sample_inputs_masked_softmax, use_zero_dimensions=False
        ),
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,
    ),
    ReductionOpInfo(
        "masked.norm",
        identity=0,
        method_variant=None,
        nan_policy="propagate",
        supports_out=False,
        promotes_int_to_float=True,
        dtypes=floating_types_and(torch.float16, torch.bfloat16),
        skips=(
            DecorateInfo(
                unittest.expectedFailure,
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
            # FIXME: sum reduces all dimensions when dim=[]
            DecorateInfo(unittest.expectedFailure, "TestReductions", "test_dim_empty"),
            DecorateInfo(
                unittest.expectedFailure, "TestReductions", "test_dim_empty_keepdim"
            ),
            # torch.jit.frontend.NotSupportedError: Compiled functions
            # can't take variable number of arguments or use
            # keyword-only arguments with defaults
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
        ),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_masked_norm,
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,
    ),
    ReductionOpInfo(
        "masked.var",
        ref=reference_masked_std_var(np.var)
        if np.lib.NumpyVersion(np.__version__) >= "1.20.2"
        else None,
        method_variant=None,
        nan_policy="propagate",
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        promotes_int_to_float=True,
        dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16),
        skips=(
            # Issue with conj and torch dispatch, see https://github.com/pytorch/pytorch/issues/82479
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestSchemaCheckModeOpInfo",
                "test_schema_correctness",
                dtypes=(torch.complex64, torch.complex128),
            ),
            DecorateInfo(
                unittest.expectedFailure,
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
            # FIXME: sum reduces all dimensions when dim=[]
            DecorateInfo(unittest.expectedFailure, "TestReductions", "test_dim_empty"),
            DecorateInfo(
                unittest.expectedFailure, "TestReductions", "test_dim_empty_keepdim"
            ),
            # RuntimeError: undefined value tensor
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
            # Driver issue of XPU, see https://github.com/intel/torch-xpu-ops/issues/2295
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestReductions",
                "test_ref_small_input",
                device_type="xpu",
                dtypes=[torch.complex128],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestReductions",
                "test_reference_masked",
                device_type="xpu",
                dtypes=[torch.complex128],
            ),
        ),
        decorators=[
            DecorateInfo(
                toleranceOverride(
                    {
                        torch.float16: tol(atol=1e-02, rtol=1e-02),
                        torch.bfloat16: tol(atol=1e-03, rtol=1e-03),
                    }
                ),
                "TestReductions",
                "test_reference_masked",
            ),
            DecorateInfo(
                toleranceOverride({torch.float16: tol(atol=1e-02, rtol=1e-02)}),
                "TestReductions",
                "test_ref_small_input",
            ),
            DecorateInfo(
                toleranceOverride({torch.float16: tol(atol=1e-02, rtol=1e-02)}),
                "TestMasked",
                "test_reference_masked",
            ),
            DecorateInfo(
                toleranceOverride(
                    {
                        torch.float16: tol(atol=1e-02, rtol=1e-02),
                        torch.bfloat16: tol(atol=1e-03, rtol=1e-03),
                    }
                ),
                "TestMasked",
                "test_reference_masked",
            ),
            DecorateInfo(
                toleranceOverride(
                    {
                        torch.float16: tol(atol=4e-5, rtol=2e-2),
                    }
                ),
                "TestInductorOpInfo",
                "test_comprehensive",
                device_type="cuda",
            ),
        ],
        sample_inputs_func=sample_inputs_masked_std_var,
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,
        check_batched_grad=True,
    ),
    ReductionOpInfo(
        "masked.std",
        ref=reference_masked_std_var(np.std)
        if np.lib.NumpyVersion(np.__version__) >= "1.20.2"
        else None,
        method_variant=None,
        nan_policy="propagate",
        # Runs very slowly on slow gradcheck - alternatively reduce input sizes
        gradcheck_fast_mode=True,
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        promotes_int_to_float=True,
        dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
        skips=(
            # Issue with conj and torch dispatch, see https://github.com/pytorch/pytorch/issues/82479
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestSchemaCheckModeOpInfo",
                "test_schema_correctness",
                dtypes=(torch.complex64, torch.complex128),
            ),
            DecorateInfo(
                unittest.expectedFailure,
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
            # FIXME: sum reduces all dimensions when dim=[]
            DecorateInfo(unittest.expectedFailure, "TestReductions", "test_dim_empty"),
            DecorateInfo(
                unittest.expectedFailure, "TestReductions", "test_dim_empty_keepdim"
            ),
            # RuntimeError: undefined value tensor
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
            # Driver issue of XPU, see https://github.com/intel/torch-xpu-ops/issues/2295
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestReductions",
                "test_ref_small_input",
                device_type="xpu",
                dtypes=[torch.complex128],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestReductions",
                "test_reference_masked",
                device_type="xpu",
                dtypes=[torch.complex128],
            ),
        ),
        decorators=[
            DecorateInfo(
                toleranceOverride(
                    {
                        torch.bfloat16: tol(atol=1e-02, rtol=1e-02),
                        torch.float16: tol(atol=1e-02, rtol=1e-02),
                    }
                ),
                "TestReductions",
                "test_reference_masked",
            ),
            DecorateInfo(
                toleranceOverride({torch.float16: tol(atol=1e-02, rtol=1e-02)}),
                "TestReductions",
                "test_ref_small_input",
            ),
            DecorateInfo(
                toleranceOverride(
                    {
                        torch.float16: tol(atol=1e-02, rtol=1e-02),
                        torch.bfloat16: tol(atol=5e-03, rtol=5e-04),
                    }
                ),
                "TestMasked",
                "test_reference_masked",
            ),
        ],
        sample_inputs_func=sample_inputs_masked_std_var,
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,
        check_batched_grad=True,
    ),
    OpInfo(
        "masked.softmax",
        method_variant=None,
        dtypes=floating_types_and(torch.half, torch.bfloat16),
        sample_inputs_func=sample_inputs_masked_softmax,
        skips=(
            DecorateInfo(
                unittest.expectedFailure,
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
        ),
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        supports_out=False,
    ),
    OpInfo(
        "masked.log_softmax",
        method_variant=None,
        dtypes=floating_types_and(torch.half, torch.bfloat16),
        dtypesIfMPS=floating_types_and(
            torch.half,
            torch.bfloat16,
            torch.uint8,
            torch.int32,
            torch.int16,
            torch.int8,
        ),
        sample_inputs_func=sample_inputs_masked_softmax,
        skips=(
            DecorateInfo(
                unittest.expectedFailure,
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
            # NotImplementedError: "log_softmax_lastdim_kernel_impl" not implemented for *
            DecorateInfo(
                unittest.expectedFailure,
                "TestConsistency",
                device_type="mps",
                dtypes=(
                    torch.uint8,
                    torch.int32,
                    torch.int16,
                    torch.int8,
                ),
            ),
        ),
        decorators=[
            DecorateInfo(
                toleranceOverride({torch.bfloat16: tol(atol=1e-02, rtol=1e-02)}),
                "TestMasked",
                "test_reference_masked",
            ),
        ],
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        supports_out=False,
    ),
    OpInfo(
        "masked.softmin",
        method_variant=None,
        dtypes=floating_types_and(torch.half, torch.bfloat16),
        sample_inputs_func=sample_inputs_masked_softmax,
        skips=(
            DecorateInfo(
                unittest.expectedFailure,
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
            # FIXME:
            # Mismatched elements: 2 / 2 (100.0%)
            # Greatest absolute difference: nan at index (0,) (up to 0.0001 allowed)
            # Greatest relative difference: nan at index (0,) (up to 0.0001 allowed
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestOperators",
                "test_vmapvjpvjp",
                device_type="cpu",
            ),
        ),
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        supports_out=False,
    ),
    OpInfo(
        "masked.normalize",
        method_variant=None,
        dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16),
        sample_inputs_func=sample_inputs_masked_normalize,
        decorators=[
            DecorateInfo(
                toleranceOverride({torch.float16: tol(atol=2e-5, rtol=6e-3)}),
                "TestInductorOpInfo",
                "test_comprehensive",
                device_type="cuda",
            ),
        ],
        skips=(
            DecorateInfo(
                unittest.expectedFailure,
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
        ),
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,
        # Runs very slowly on slow gradcheck - alternatively reduce input sizes
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        supports_out=False,
    ),
    OpInfo(
        "masked.logaddexp",
        dtypes=floating_types_and(torch.float16, torch.bfloat16),
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        check_batched_forward_grad=False,
        skips=(
            DecorateInfo(
                unittest.expectedFailure,
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
            # NotSupportedError: Compiled functions can't ... use keyword-only arguments with defaults
            DecorateInfo(
                unittest.skip("Skipped!"), "TestJit", "test_variant_consistency_jit"
            ),
        ),
        sample_inputs_func=sample_inputs_masked_logaddexp,
        gradcheck_wrapper=gradcheck_wrapper_masked_pointwise_operation,
    ),
    ReductionOpInfo(
        "masked.logsumexp",
        dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
        method_variant=None,
        nan_policy="propagate",
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
            # FIXME: reduces all dimensions when dim=[]
            DecorateInfo(unittest.skip("Skipped!"), "TestReductions", "test_dim_empty"),
            DecorateInfo(
                unittest.skip("Skipped!"), "TestReductions", "test_dim_empty_keepdim"
            ),
            # Identity can't be -torch.inf without overflow
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestReductions",
                "test_empty_tensor_empty_slice",
            ),
            # NotSupportedError: Compiled functions can't ... use keyword-only arguments with defaults
            DecorateInfo(
                unittest.skip("Skipped!"), "TestJit", "test_variant_consistency_jit"
            ),
            # all the values are the same except for -inf vs nan
            DecorateInfo(unittest.skip("Skipped!"), "TestDecomp", "test_comprehensive"),
            # FIXME:
            # Mismatched elements: 2 / 12 (16.7%)
            # Greatest absolute difference: 9223372034707292160 at index (0, 0, 0, 0)
            # Greatest relative difference: 0.0 at index (0, 0, 0, 1)
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestInductorOpInfo",
                "test_comprehensive",
                device_type="cpu",
            ),
        ),
        sample_inputs_func=sample_inputs_masked_reduction,
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,
    ),
]
