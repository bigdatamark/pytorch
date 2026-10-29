# mypy: ignore-errors

# [MODULE DOC]
# Helpers for generating and validating sample and error inputs of OpInfos on sparse tensors
# (COO, CSR, CSC, BSR, BSC layouts). Provides generators for sparse reductions (sum), sparse
# elementwise binary ops (mul), like-functions, and a dispatching validator that classifies each
# sample as valid (SampleInput) or known-failing (ErrorInput with the expected error message).
# The environment variable PYTORCH_TEST_CHECK_VALIDATE_SPARSE_SAMPLES set to 1 makes the
# validation actually run each op to confirm the classification. No import-time side effects.

import os

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import highest_precision_float
from torch.testing._internal.opinfo.core import (
    BinaryUfuncInfo,
    ErrorInput,
    generate_elementwise_binary_tensors,
    ReductionOpInfo,
    sample_inputs_reduction,
    SampleInput,
)


def _check_validate(op_info, sample):
    # [FUNCTION DOC]
    # HELPER: Confirms a classified sample by calling the op on it. For an ErrorInput the call
    # must raise the expected error type; for a SampleInput it must succeed. Raises
    # AssertionError describing the mismatch otherwise. Runs the operator (side effects of op).
    def _check_fail(sample):
        # [FUNCTION DOC]
        # HELPER: Calls the op on the wrapped sample of an ErrorInput; swallows the expected
        # error type, raises AssertionError for a different exception or for no exception.
        # KNOWN ISSUE: only the error type is checked, the error_regex is never matched.
        try:
            op_info(
                sample.sample_input.input,
                *sample.sample_input.args,
                **sample.sample_input.kwargs,
            )
        except sample.error_type:
            pass
        except Exception as msg:
            raise AssertionError(  # noqa: B904
                f"{op_info.name} on {sample.sample_input=} expected exception "
                f"{sample.error_type}: {sample.error_regex}, got {type(msg).__name__}: {msg}"
            )
        else:
            raise AssertionError(
                f"{op_info.name} on {sample.sample_input=} expected exception "
                f"{sample.error_type}: {sample.error_regex}, got none."
            )

    def _check_success(sample):
        # [FUNCTION DOC]
        # HELPER: Calls the op on a SampleInput; raises AssertionError wrapping any exception.
        try:
            op_info(sample.input, *sample.args, **sample.kwargs)
        except Exception as msg:
            raise AssertionError(  # noqa: B904
                f"{op_info.name} on {sample=} expected to succeed "
                f", got {type(msg).__name__}: {msg}"
            )

    if isinstance(sample, ErrorInput):
        _check_fail(sample)
    else:
        _check_success(sample)


def _sample_inputs_sparse(
    sample_inputs,
    maybe_failing_sample_inputs,
    validate_sample_input,
    op_info,
    *args,
    **kwargs,
):
    # [FUNCTION DOC]
    # HELPER: Generator combining regular and maybe-failing sample generators. Each produced
    # sample goes through validate_sample_input (with check_validate read from the environment
    # variable PYTORCH_TEST_CHECK_VALIDATE_SPARSE_SAMPLES equal to 1) and only those classified
    # as SampleInput are yielded; ErrorInput results are dropped here.
    check_validate = (
        os.environ.get("PYTORCH_TEST_CHECK_VALIDATE_SPARSE_SAMPLES", "0") == "1"
    )
    for sample in sample_inputs(op_info, *args, **kwargs):
        sample = validate_sample_input(op_info, sample, check_validate=check_validate)
        if isinstance(sample, SampleInput):
            yield sample
        # Error inputs are handled in error_inputs_sparse

    for sample in maybe_failing_sample_inputs(op_info, *args, **kwargs):
        sample = validate_sample_input(op_info, sample, check_validate=check_validate)
        if isinstance(sample, SampleInput):
            yield sample


def _error_inputs_sparse(
    maybe_failing_sample_inputs, validate_sample_input, op_info, *args, **kwargs
):
    # [FUNCTION DOC]
    # HELPER: Generator counterpart of the sample generator: runs the maybe-failing samples
    # through validate_sample_input (check_validate from the environment variable
    # PYTORCH_TEST_CHECK_VALIDATE_SPARSE_SAMPLES equal to 1) and yields only ErrorInput results.
    check_validate = (
        os.environ.get("PYTORCH_TEST_CHECK_VALIDATE_SPARSE_SAMPLES", "0") == "1"
    )
    for sample in maybe_failing_sample_inputs(op_info, *args, **kwargs):
        sample = validate_sample_input(op_info, sample, check_validate=check_validate)
        if isinstance(sample, ErrorInput):
            yield sample
        # Sample inputs are handled in sample_inputs_sparse


def _apply_requires_grad_to_samples(sample_inputs):
    # [FUNCTION DOC]
    # HELPER: Decorator for sample generator functions. Returns a wrapper generator that, when
    # requires_grad is set, clones/detaches every floating or complex tensor in each sample and
    # marks it as requiring grad (so samples sharing tensor instances stay independent);
    # otherwise passes the samples through unchanged.
    """Decorator to _maybe_failing_sample_inputs_... generator functions
    that clones and sets requires_grad argument to tensors in sample
    input arguments. This is needed when the generated samples share
    tensor instances.
    """

    def wrapper(op_info, device, dtype, requires_grad, layout, **kwargs):
        # [FUNCTION DOC]
        # HELPER: The decorated generator; see the enclosing decorator for behavior.
        def apply_requires_grad(x):
            # [FUNCTION DOC]
            # HELPER: Returns x unchanged unless it is a floating/complex tensor that does not
            # yet require grad (and requires_grad is wanted); then returns a detached clone that
            # requires grad.
            if (
                not isinstance(x, torch.Tensor)
                or x.requires_grad
                or not requires_grad
                or not (x.is_floating_point() or x.is_complex())
            ):
                return x
            return x.detach().clone().requires_grad_(requires_grad)

        if requires_grad:
            for sample_input in sample_inputs(
                op_info, device, dtype, requires_grad, layout, **kwargs
            ):
                yield sample_input.transform(apply_requires_grad)
        else:
            yield from sample_inputs(
                op_info, device, dtype, requires_grad, layout, **kwargs
            )

    return wrapper


def sample_inputs_sparse_reduction(
    op_info, device, dtype, requires_grad, layout, blocksize=None, **kwargs
):
    # [FUNCTION DOC]
    # Generator of sparse-tensor samples for reduction ops: starts from the dense reduction
    # samples and converts each input to the given layout (blocksize defaults to 1x1 for block
    # layouts). Yields nothing if the op does not support the layout. Skips scalar inputs,
    # compressed layouts with fewer than 2 dims, and batched compressed samples with explicit
    # zeros. Also yields an uncoalesced COO variant (float/complex dtypes) and a hybrid variant
    # (dense dims) for inputs with more than 2 dims. Raises AssertionError if the uncoalesced
    # tensor turns out coalesced.
    """Sample inputs for reduction operations on sparse tensors."""
    layout_name = str(layout).split(".", 1)[-1].rsplit("_coo", 1)[0]
    op_supports_layout = getattr(op_info, "supports_" + layout_name)
    if not op_supports_layout:
        return

    for sample_input in sample_inputs_reduction(
        op_info, device, dtype, requires_grad, **kwargs
    ):
        if sample_input.input.ndim == 0:
            # scalar sparse tensors are not supported
            continue

        if layout in {
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        }:
            if sample_input.input.ndim < 2:
                # conversion to sparse compressed tensors requires at
                # least 2 dimensional tensors
                continue
            if sample_input.input.ndim > 2 and (sample_input.input == 0).any():
                # Skip batched sparse compressed samples that contain
                # explicit zeros because to_sparse(layout=..) will
                # fail, see gh-98495.
                # TODO: remove this if-block after gh-98495 is fixed.
                continue

        if layout in {torch.sparse_bsr, torch.sparse_bsc} and blocksize is None:
            blocksize = (1, 1)

        yield SampleInput(
            sample_input.input.detach()
            .to_sparse(layout=layout, blocksize=blocksize)
            .requires_grad_(requires_grad),
            args=sample_input.args,
            kwargs=sample_input.kwargs,
        )

        if layout is torch.sparse_coo and (dtype.is_floating_point or dtype.is_complex):
            # uncoalesced samples
            inp = sample_input.input.detach().to_sparse(layout=layout)
            inp = torch.sparse_coo_tensor(
                inp.indices().repeat(1, 2),
                inp.values().repeat(2),
                inp.shape,
                dtype=inp.dtype,
                device=inp.device,
            )
            if inp.is_coalesced():
                raise AssertionError("Expected inp to not be coalesced")
            yield SampleInput(
                inp.requires_grad_(requires_grad),
                args=sample_input.args,
                kwargs=sample_input.kwargs,
            )

        if sample_input.input.ndim > 2:
            # hybrid samples
            yield SampleInput(
                sample_input.input.detach()
                .to_sparse(
                    layout=layout,
                    blocksize=blocksize,
                    dense_dim=sample_input.input.ndim - 2,
                )
                .requires_grad_(requires_grad),
                args=sample_input.args,
                kwargs=sample_input.kwargs,
            )


def _validate_sample_input_sparse_reduction(op_info, sample, check_validate=False):
    # [FUNCTION DOC]
    # HELPER: Classifies a reduction sample. Returns the sample unchanged if supported, or an
    # ErrorInput carrying the known error type and message for op specific unsupported cases:
    # sum (delegates to the sum validator), masked sum, and masked amax/amin/mean/prod (missing
    # mask, keepdim false, batched or 3D inputs, mask layout/nnz mismatches). If check_validate
    # is set, also runs the op to confirm the classification.
    """Return the specified sample when it is valid and supported by the
    operation. Otherwise, return the sample as ErrorInput instance.

    When check_validate is True, the result is validated against
    calling the op on the sample.
    """
    UNSPECIFIED = object()
    if op_info.name == "sum":
        sample = _validate_sample_input_sparse_reduction_sum(sample)

    if op_info.name == "masked.sum":
        mask = sample.kwargs.get("mask", UNSPECIFIED)
        if (
            mask not in {None, UNSPECIFIED}
            and mask.ndim > 2
            and mask.layout is torch.strided
            and (mask == 0).any()
        ):
            # TODO: remove this if-block after gh-98495 is fixed.
            sample = ErrorInput(
                sample,
                error_regex="Expect the same number of specified elements per batch.",
            )
        elif not sample.kwargs.get("keepdim"):
            sample = ErrorInput(
                sample,
                error_type=(AssertionError, RuntimeError),
                error_regex="reduction operations on (CSR|CSC) tensors with keepdim=False is unsupported",
            )
        elif mask is UNSPECIFIED:
            sample = ErrorInput(
                sample,
                error_type=ValueError,
                error_regex="masked (.*) expects explicit mask for sparse_csr tensor input",
            )
        elif sample.input.ndim > 2:
            sample = ErrorInput(
                sample,
                error_regex="crow_indices is supposed to be a vector, but got 3 dimensional tensor.",
            )

    if op_info.name in {"masked.amax", "masked.amin", "masked.mean", "masked.prod"}:
        t_inp = sample.input
        mask = sample.kwargs.get("mask")
        if (
            mask is not None
            and mask.ndim > 2
            and mask.layout is torch.strided
            and (mask == 0).any()
        ):
            # TODO: remove this if-block after gh-98495 is fixed.
            sample = ErrorInput(
                sample,
                error_regex="Expect the same number of specified elements per batch.",
            )
        elif mask is None:
            sample = ErrorInput(
                sample,
                error_type=ValueError,
                error_regex="masked (.*) expects explicit mask for sparse_csr tensor input",
            )
        elif (
            mask.layout is sample.input.layout
            and mask.ndim > 2
            and op_info.name == "masked.mean"
        ):
            sample = ErrorInput(
                sample,
                error_type=TypeError,
                error_regex=(
                    "where[(][)] received an invalid combination of arguments"
                    " - got [(]Tensor, Tensor, NoneType[)]"
                ),
            )
        elif not sample.kwargs.get("keepdim"):
            sample = ErrorInput(
                sample,
                error_type=(AssertionError, RuntimeError),
                error_regex="reduction operations on (CSR|CSC) tensors with keepdim=False is unsupported",
            )
        elif (
            sample.input.ndim > 2
            and (sample.kwargs.get("dim") not in {0, 1})
            and mask.ndim > 2
            and mask.layout is not torch.strided
        ):
            if sample.kwargs.get("dim") == (0, -1):
                sample = ErrorInput(
                    sample,
                    error_regex="tensor dimensionality must be sum of batch, base, and dense dimensionalities",
                )
            elif op_info.name == "masked.prod":
                sample = ErrorInput(
                    sample,
                    error_regex="input_dim == 2 INTERNAL ASSERT FAILED at",
                )
            else:
                sample = ErrorInput(
                    sample,
                    error_type=AssertionError,
                    error_regex="Sparse CSR tensors are 2D and only support reduction along dim 0 or 1.",
                )
        elif sample.input.ndim > 2:
            sample = ErrorInput(
                sample,
                error_regex="crow_indices is supposed to be a vector, but got 3 dimensional tensor.",
            )
        elif (
            mask.layout is t_inp.layout
            and mask._nnz() != t_inp._nnz()
            and t_inp.dense_dim() > 0
        ):
            sample = ErrorInput(
                sample,
                error_regex="Index tensor must have the same number of dimensions as src tensor",
            )

    if check_validate:
        _check_validate(op_info, sample)

    return sample


def _validate_sample_input_sparse_reduction_sum(sample, check_validate=False):
    # [FUNCTION DOC]
    # HELPER: For sum with an explicit dim on a sparse compressed input, returns an ErrorInput
    # with the known message for unsupported cases (non-CSR compressed layouts, CSR/CSC with
    # keepdim false, non-2D input, bool CSR, complex32 CSR); otherwise returns the sample.
    # KNOWN ISSUE: the check_validate argument is accepted but never used.
    # NOTE: When fixing a failing sample case, remove the
    #       corresponding if-block
    t_inp, t_kwargs = sample.input, sample.kwargs
    dim = t_kwargs.get("dim")
    keepdim = t_kwargs.get("keepdim")
    layout = t_inp.layout
    if isinstance(dim, (int, list, tuple)):
        if layout in {
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        }:
            if layout in {torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}:
                return ErrorInput(
                    sample,
                    error_regex=(
                        "Currently the only compressed sparse format supported for sum.dim_IntList is CSR, but got layout"
                    ),
                )
            if layout in {torch.sparse_csr, torch.sparse_csc} and not keepdim:
                return ErrorInput(
                    sample,
                    error_regex=(
                        "reduction operations on CSR tensors with keepdim=False is unsupported"
                    ),
                )
            if t_inp.dim() != 2:
                return ErrorInput(
                    sample,
                    error_regex=("input_dim == 2 INTERNAL ASSERT"),
                )
            if layout == torch.sparse_csr:
                if t_inp.dtype == torch.bool:
                    return ErrorInput(
                        sample,
                        error_regex=("_sparse_csr_sum_cpu not implemented for 'Bool'"),
                    )
                if t_inp.dtype == torch.complex32:
                    return ErrorInput(
                        sample,
                        error_regex=(
                            "_sparse_csr_sum_cuda not implemented for 'ComplexHalf'"
                        ),
                    )
    return sample


def _maybe_failing_sample_inputs_sparse_reduction_sum(
    op_info, device, dtype, requires_grad, layout, **kwargs
):
    # [FUNCTION DOC]
    # HELPER: Generator of sum samples (dim, keepdim variants on small 2D and hybrid sparse
    # tensors) for compressed layouts that are known to fail with NotImplementedError or
    # RuntimeError. Yields nothing for COO. Block layouts use explicit blocksizes.
    """Generator of samples that are known to fail or that were failing in past."""
    # NOTE: When fixing a failing case, remove the Exception comment
    #       but keep the `yield sample` statement.
    if layout in [
        torch.sparse_csr,
        torch.sparse_csc,
    ]:
        # NotImplementedError: Could not run 'aten::sum.IntList_out' with arguments from the 'SparseCsrCPU' backend.
        yield SampleInput(
            torch.tensor([[0, 1], [2, 3]], dtype=dtype)
            .to_sparse(layout=layout)
            .requires_grad_(requires_grad),
            kwargs=dict(dim=0, keepdim=True),
        )
        yield SampleInput(
            torch.tensor([[[0, 1]], [[2, 3]]], dtype=dtype)
            .to_sparse(layout=layout, dense_dim=1)
            .requires_grad_(requires_grad),
            kwargs=dict(dim=0),
        )
        yield SampleInput(
            torch.tensor([[0, 1], [2, 3]], dtype=dtype)
            .to_sparse(layout=layout)
            .requires_grad_(requires_grad),
            kwargs=dict(dim=(0,)),
        )
        yield SampleInput(
            torch.tensor([[0, 1], [2, 3]], dtype=dtype)
            .to_sparse(layout=layout)
            .requires_grad_(requires_grad),
            kwargs=dict(dim=(0,), keepdim=True),
        )
        yield SampleInput(
            torch.tensor([[[0, 1]], [[2, 3]]], dtype=dtype)
            .to_sparse(layout=layout, dense_dim=1)
            .requires_grad_(requires_grad),
            kwargs=dict(dim=(0,)),
        )

        # RuntimeError: torch.empty: Only batched sparse compressed (non-block) tensors are supported, but got size [2]
        yield SampleInput(
            torch.tensor([[0, 1], [2, 3]], dtype=dtype)
            .to_sparse(layout=layout)
            .requires_grad_(requires_grad),
            kwargs=dict(dim=0),
        )

    if layout in [
        torch.sparse_bsr,
        torch.sparse_bsc,
    ]:
        # RuntimeError: empty_sparse_compressed expected sparse compressed (non-block) tensor layout but got SparseBsr
        yield SampleInput(
            torch.tensor([[0, 1], [2, 3]], dtype=dtype)
            .to_sparse(layout=layout, blocksize=(2, 2))
            .requires_grad_(requires_grad),
            kwargs=dict(dim=0, keepdim=True),
        )
        yield SampleInput(
            torch.tensor([[[0, 1]], [[2, 3]]], dtype=dtype)
            .to_sparse(layout=layout, dense_dim=1, blocksize=(1, 1))
            .requires_grad_(requires_grad),
            kwargs=dict(dim=0),
        )
        yield SampleInput(
            torch.tensor([[0, 1], [2, 3]], dtype=dtype)
            .to_sparse(layout=layout, blocksize=(1, 1))
            .requires_grad_(requires_grad),
            kwargs=dict(dim=(0,)),
        )
        yield SampleInput(
            torch.tensor([[0, 1], [2, 3]], dtype=dtype)
            .to_sparse(layout=layout, blocksize=(1, 1))
            .requires_grad_(requires_grad),
            kwargs=dict(dim=(0,), keepdim=True),
        )
        yield SampleInput(
            torch.tensor([[[0, 1]], [[2, 3]]], dtype=dtype)
            .to_sparse(layout=layout, blocksize=(1, 1), dense_dim=1)
            .requires_grad_(requires_grad),
            kwargs=dict(dim=(0,)),
        )

        # RuntimeError: torch.empty: Only batched sparse compressed (non-block) tensors are supported, but got size [2]
        yield SampleInput(
            torch.tensor([[0, 1], [2, 3]], dtype=dtype)
            .to_sparse(layout=layout, blocksize=(1, 1))
            .requires_grad_(requires_grad),
            kwargs=dict(dim=0),
        )


def sample_inputs_sparse_reduction_sum(
    op_info, device, dtype, requires_grad, layout, **kwargs
):
    # [FUNCTION DOC]
    # Generator of valid sample inputs for sum on sparse tensors: regular reduction samples plus
    # the maybe-failing ones that validate as supported.
    """Sample inputs for sum on sparse tensors."""
    yield from _sample_inputs_sparse(
        sample_inputs_sparse_reduction,
        _maybe_failing_sample_inputs_sparse_reduction_sum,
        _validate_sample_input_sparse_reduction,
        op_info,
        device,
        dtype,
        requires_grad,
        layout,
        **kwargs,
    )


def error_inputs_sparse_reduction_sum(op_info, device, layout, **kwargs):
    # [FUNCTION DOC]
    # Generator of ErrorInput cases for sum on sparse tensors, built from the maybe-failing
    # samples that validate as errors. Uses float64 and no requires_grad.
    """Error inputs for sum on sparse tensors."""
    dtype = torch.float64
    requires_grad = False
    yield from _error_inputs_sparse(
        _maybe_failing_sample_inputs_sparse_reduction_sum,
        _validate_sample_input_sparse_reduction,
        op_info,
        device,
        dtype,
        requires_grad,
        layout,
        **kwargs,
    )


def sample_inputs_sparse_elementwise_binary_operation(
    op_info, device, dtype, requires_grad, layout, **kwargs
):
    # [FUNCTION DOC]
    # Generator of sparse samples for elementwise binary ops. For each dense binary sample
    # (zeros excluded) it converts both operands to the layout for every valid dense_dim (and
    # block sizes 1x1 and full block for block layouts), yielding op(sparse, sparse) and
    # op(sparse, scalar tensor). Compressed layouts skip inputs with fewer than 2 dims.
    """Sample inputs for elementwise binary operations on sparse tensors.

    The samples include regular, zero-sized, batched, and hybrid
    sparse tensors as well as rhs scalars. All tensors are full tensors.
    """

    def _to_sparse(tensor, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Detaches the tensor, converts it with the given to_sparse keywords and sets
        # requires_grad as requested by the enclosing generator.
        return tensor.detach().to_sparse(**kwargs).requires_grad_(requires_grad)

    for sample_input in generate_elementwise_binary_tensors(
        op_info,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        exclude_zero=True,
        **kwargs,
    ):
        lhs, rhs = sample_input.input, sample_input.args[0]
        min_dense_dim = 0
        max_dense_dim = lhs.ndim - 1
        if layout in {
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        }:
            if lhs.ndim < 2:
                # sparse compressed tensors sparse_dim must be 2
                continue
            max_dense_dim = lhs.ndim - 2

        for dense_dim in range(min_dense_dim, max_dense_dim + 1):
            if layout in {torch.sparse_bsr, torch.sparse_bsc}:
                blocksizes = [(1, 1)]
                if lhs.numel() > 0:
                    blocksizes.append(
                        (
                            lhs.shape[lhs.ndim - 2 - dense_dim],
                            lhs.shape[lhs.ndim - 1 - dense_dim],
                        )
                    )
            else:
                blocksizes = [None]
            for blocksize in blocksizes:
                to_sparse_kwargs = dict(
                    layout=layout, dense_dim=dense_dim, blocksize=blocksize
                )
                lhs_sparse = _to_sparse(lhs, **to_sparse_kwargs)
                rhs_sparse = _to_sparse(rhs, **to_sparse_kwargs)
                # op(sparse, sparse)
                yield SampleInput(
                    lhs_sparse,
                    args=(rhs_sparse, *sample_input.args[1:]),
                    kwargs=sample_input.kwargs,
                )
                # op(sparse, scalar)
                yield SampleInput(
                    lhs_sparse,
                    args=(
                        make_tensor(
                            (), dtype=dtype, device=device, requires_grad=requires_grad
                        ),
                        *sample_input.args[1:],
                    ),
                    kwargs=sample_input.kwargs,
                )


def _validate_sample_input_elementwise_binary_sparse_mul(sample):
    # [FUNCTION DOC]
    # HELPER: Classifies a mul sample on sparse input: returns an ErrorInput with the known
    # error message for unsupported combinations (batched CSR, CSC/BSR/BSC with non-scalar
    # other, bool or half dtypes on CPU, mixed strided/sparse CSR operands, shape mismatches);
    # otherwise returns the sample unchanged.
    # NOTE: When fixing a failing sample case, remove the
    #       corresponding if-block
    t_inp, t_args = sample.input, sample.args
    batch_dim = t_inp.dim() - t_inp.dense_dim() - t_inp.sparse_dim()
    layout = t_inp.layout
    dtype = t_inp.dtype
    if layout is torch.sparse_csr and batch_dim > 0 and t_args[0].ndim > 0:
        return ErrorInput(
            sample,
            error_regex=(
                "coo_to_sparse_csr: conversion from Sparse to SparseCsr for input"
                " tensors with sparse_dim[(][)]!=2 is not supported"
            ),
        )
    elif layout is torch.sparse_csc and t_args[0].ndim > 0:
        return ErrorInput(
            sample, error_regex="Expected result Tensor to be of format CSR"
        )
    elif layout is torch.sparse_bsr and t_args[0].ndim > 0:
        return ErrorInput(
            sample,
            error_regex="empty_sparse_compressed expected sparse compressed [(]non-block[)] tensor layout but got SparseBsr",
        )
    elif layout is torch.sparse_bsc and t_args[0].ndim > 0:
        return ErrorInput(
            sample,
            error_regex="empty_sparse_compressed expected sparse compressed [(]non-block[)] tensor layout but got SparseBsc",
        )
    elif (
        layout is torch.sparse_coo
        and dtype is torch.bool
        and t_args[0].ndim > 0
        and t_inp.is_cpu
        and t_inp.numel() > 0
        and t_inp.dense_dim() > 0
    ):
        return ErrorInput(
            sample, error_regex="\"addcmul_cpu_out\" not implemented for 'Bool'"
        )
    elif (
        layout in {torch.sparse_coo, torch.sparse_csr}
        and dtype is torch.bool
        and t_inp._nnz() > 0
        and t_args[0].ndim > 0
        and t_inp.is_cpu
        and t_inp.numel() > 0
    ):
        return ErrorInput(
            sample, error_regex="\"mul_out_sparse\" not implemented for 'Bool'"
        )
    elif (
        layout is torch.sparse_csr
        and t_args[0].layout is torch.strided
        and 0 < t_args[0].ndim
        and t_args[0].ndim < t_inp.ndim
    ):
        return ErrorInput(
            sample, error_regex="sparse_mask_sparse_csr expects self to be 2D"
        )
    elif layout is torch.sparse_csr and (
        (t_args[0].layout is torch.strided and 0 < t_args[0].ndim)
        or (t_args[0].layout is layout and t_inp.shape != t_args[0].shape)
    ):
        return ErrorInput(
            sample,
            error_regex=(
                "expects sparse inputs with equal dimensionality, number of sparse dimensions,"
                " and shape of sparse dimensions"
            ),
        )
    elif (
        layout is torch.sparse_csr
        and t_inp.dense_dim() > 0
        and t_inp._nnz() > 0
        and t_inp.is_cpu
        and dtype is torch.float16
        and t_args[0].ndim > 0
    ):
        return ErrorInput(
            sample, error_regex="\"addcmul_cpu_out\" not implemented for 'Half'"
        )
    return sample


@_apply_requires_grad_to_samples
def _maybe_failing_sample_inputs_sparse_elementwise_binary_mul(
    op_info, device, dtype, requires_grad, layout, **kwargs
):
    # [FUNCTION DOC]
    # HELPER: Generator (wrapped to give each sample its own grad-requiring tensors) of mul
    # samples known to fail per layout and dtype: batched or hybrid CSR, CSC/BSR/BSC regular
    # inputs, and complex32/bool/half cases for COO and CSR. Builds regular, batch and hybrid
    # sparse tensors on the given device and dtype.
    """Generator of samples that are known to fail or that were failing in past."""
    # NOTE: When fixing a failing case, remove the Exception comment
    #       but keep the `yield sample` statement.

    blocksize = (1, 1) if layout in {torch.sparse_bsr, torch.sparse_bsc} else None
    regular = torch.tensor([[1, 2], [3, 4]], device=device, dtype=dtype).to_sparse(
        layout=layout, dense_dim=0, blocksize=blocksize
    )
    batch = torch.tensor(
        [[[1, 2], [3, 4]], [[4, 5], [6, 7]]], device=device, dtype=dtype
    ).to_sparse(layout=layout, dense_dim=0, blocksize=blocksize)
    hybrid = torch.tensor(
        [[[1], [2]], [[3], [4]]], device=device, dtype=dtype
    ).to_sparse(layout=layout, dense_dim=1, blocksize=blocksize)

    if layout is torch.sparse_csr:
        # RuntimeError: crow_indices is supposed to be a vector, but got 2 dimensional tensor
        yield SampleInput(batch, args=(batch,))
        # RuntimeError: Only tensors with two sparse dimensions can be
        # converted to the SparseCsr layout, got self with 3 sparse
        # dimensions.
        yield SampleInput(
            torch.zeros_like(hybrid).requires_grad_(requires_grad),
            args=(torch.zeros_like(hybrid).requires_grad_(requires_grad),),
        )
        if dtype is torch.complex32:
            # RuntimeError: "mul_out_sparse" not implemented for 'ComplexHalf'
            yield SampleInput(regular, args=(regular,))
        if dtype is torch.bool and regular.is_cpu:
            # RuntimeError: "mul_out_sparse" not implemented for 'Bool'
            yield SampleInput(regular, args=(regular,))
    if layout is torch.sparse_csc:
        # RuntimeError: Expected result Tensor to be of format CSR
        yield SampleInput(regular, args=(regular,))
    if layout is torch.sparse_bsr:
        # RuntimeError: empty_sparse_compressed expected sparse compressed (non-block) tensor layout but got SparseBsr
        yield SampleInput(regular, args=(regular,))
    if layout is torch.sparse_bsc:
        # RuntimeError: empty_sparse_compressed expected sparse compressed (non-block) tensor layout but got SparseBsc
        yield SampleInput(regular, args=(regular,))
    if layout is torch.sparse_coo:
        if dtype is torch.complex32:
            # RuntimeError: "mul_out_sparse" not implemented for 'ComplexHalf'
            yield SampleInput(regular, args=(regular,))
        if dtype is torch.bool and regular.is_cpu:
            # RuntimeError: "mul_out_sparse" not implemented for 'Bool'
            yield SampleInput(regular, args=(regular,))
        if dtype in {torch.bool, torch.float16} and regular.is_cpu:
            # RuntimeError: "addcmul_cpu_out" not implemented for '(Bool|Half)'
            yield SampleInput(hybrid, args=(hybrid,))


def _validate_sample_input_sparse_elementwise_binary_operation(
    op_info, sample, check_validate=False
):
    # [FUNCTION DOC]
    # HELPER: Validator for elementwise binary ops on sparse input. Only mul is classified
    # (via the mul validator); other ops pass through. Optionally runs the op to confirm.
    if op_info.name == "mul":
        sample = _validate_sample_input_elementwise_binary_sparse_mul(sample)

    if check_validate:
        _check_validate(op_info, sample)
    return sample


def sample_inputs_sparse_mul(op_info, device, dtype, requires_grad, layout, **kwargs):
    # [FUNCTION DOC]
    # Generator of valid sample inputs for mul on sparse tensors, combining the generic sparse
    # binary samples with the maybe-failing mul samples that validate as supported.
    """Sample inputs for mul operation on sparse tensors."""
    yield from _sample_inputs_sparse(
        sample_inputs_sparse_elementwise_binary_operation,
        _maybe_failing_sample_inputs_sparse_elementwise_binary_mul,
        _validate_sample_input_sparse_elementwise_binary_operation,
        op_info,
        device,
        dtype,
        requires_grad,
        layout,
        **kwargs,
    )


def error_inputs_sparse_mul(op_info, device, layout, **kwargs):
    # [FUNCTION DOC]
    # Generator of ErrorInput cases for mul on sparse tensors (known failing samples that
    # validate as errors), using float64 and no requires_grad.
    """Error inputs for mul operation on sparse tensors."""
    dtype = torch.float64
    requires_grad = False
    yield from _error_inputs_sparse(
        _maybe_failing_sample_inputs_sparse_elementwise_binary_mul,
        _validate_sample_input_sparse_elementwise_binary_operation,
        op_info,
        device,
        dtype,
        requires_grad,
        layout,
        **kwargs,
    )


def _sample_inputs_sparse_like_fns(
    op_info, device, dtype, requires_grad, layout, **kwargs
):
    # [FUNCTION DOC]
    # HELPER: Generator of samples for like-functions (e.g. zeros_like) on sparse tensors. For
    # each simple input tensor of the layout (batch, hybrid, zero-sized) yields the tensor alone
    # and with explicit device/dtype/layout, a highest-precision float dtype, the other
    # device if CUDA is available, a related layout (csr<->csc, bsr<->bsc, else strided), and
    # COO for non-COO inputs. Imports TestCase lazily to build the inputs.
    from torch.testing._internal.common_utils import TestCase

    for tensor in TestCase().generate_simple_inputs(
        layout,
        device=device,
        dtype=dtype,
        enable_batch=True,
        enable_hybrid=True,
        enable_zero_sized=True,
        enable_non_contiguous_indices=False,
        enable_non_contiguous_values=False,
    ):
        yield SampleInput(tensor, args=(), kwargs={})
        yield SampleInput(
            tensor, args=(), kwargs=dict(device=device, dtype=dtype, layout=layout)
        )

        hpf = highest_precision_float(device)
        if dtype is not hpf:
            yield SampleInput(tensor, args=(), kwargs=dict(dtype=hpf))

        if torch.cuda.is_available():
            other_device = "cuda" if tensor.device.type == "cpu" else "cpu"
            yield SampleInput(tensor, args=(), kwargs=dict(device=other_device))

        if layout is torch.sparse_csr:
            other_layout = torch.sparse_csc
        elif layout is torch.sparse_csc:
            other_layout = torch.sparse_csr
        elif layout is torch.sparse_bsr:
            other_layout = torch.sparse_bsc
        elif layout is torch.sparse_bsc:
            other_layout = torch.sparse_bsr
        else:
            other_layout = torch.strided
        yield SampleInput(tensor, args=(), kwargs=dict(layout=other_layout))

        if layout is not torch.sparse_coo:
            yield SampleInput(tensor, args=(), kwargs=dict(layout=torch.sparse_coo))


def _validate_sample_input_sparse_like_fns(op_info, sample, check_validate=False):
    # [FUNCTION DOC]
    # HELPER: Classifies a like-function sample. Returns an ErrorInput for compressed inputs
    # (other than zeros_like) requesting a different layout, and for every COO input (the
    # random-normal fill is unsupported on sparse). Otherwise returns the sample, optionally
    # confirming by running the op when check_validate is set.
    if (
        sample.input.layout
        in {
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        }
        and op_info.name != "zeros_like"
    ):
        if sample.kwargs.get("layout", sample.input.layout) != sample.input.layout:
            return ErrorInput(
                sample,
                error_regex=(
                    "empty_like with different sparse layout is not supported"
                    " \\(self is Sparse(Csc|Csr|Bsc|Bsr) but you requested Sparse(Csr|Csc|Bsr|Bsc)\\)"
                ),
            )
    if sample.input.layout is torch.sparse_coo:
        return ErrorInput(
            sample,
            error_regex=(
                "Could not run 'aten::normal_' with arguments from the 'Sparse(CPU|CUDA)' backend."
            ),
        )
    if check_validate:
        _check_validate(op_info, sample)
    return sample


def _maybe_failing_sample_inputs_sparse_like_fns(
    op_info, device, dtype, requires_grad, layout, **kwargs
):
    # [FUNCTION DOC]
    # HELPER: Generator of like-function samples expected to fail. Only when CUDA is available
    # and the layout is not COO: yields a small 2D sparse tensor requesting the opposite device
    # type, and one requesting a different related layout.
    if torch.cuda.is_available() and layout is not torch.sparse_coo:
        other_device = "cuda" if torch.device(device).type == "cpu" else "cpu"
        if layout is torch.sparse_csr:
            other_layout = torch.sparse_csc
        elif layout is torch.sparse_csc:
            other_layout = torch.sparse_csr
        elif layout is torch.sparse_bsr:
            other_layout = torch.sparse_bsc
        elif layout is torch.sparse_bsc:
            other_layout = torch.sparse_bsr
        else:
            other_layout = torch.strided

        blocksize = (1, 1) if layout in {torch.sparse_bsr, torch.sparse_bsc} else None

        yield SampleInput(
            torch.tensor([[0, 1], [2, 3]], dtype=dtype, device=device).to_sparse(
                layout=layout, blocksize=blocksize
            ),
            kwargs=dict(device=other_device),
        )

        yield SampleInput(
            torch.tensor([[0, 1], [2, 3]], dtype=dtype, device=device).to_sparse(
                layout=layout, blocksize=blocksize
            ),
            kwargs=dict(layout=other_layout),
        )


def sample_inputs_sparse_like_fns(
    op_info, device, dtype, requires_grad, layout, **kwargs
):
    # [FUNCTION DOC]
    # Generator of valid sample inputs for like-functions on sparse tensors (regular plus the
    # maybe-failing ones that validate as supported).
    """Sample inputs for like-functions on sparse tensors."""
    yield from _sample_inputs_sparse(
        _sample_inputs_sparse_like_fns,
        _maybe_failing_sample_inputs_sparse_like_fns,
        _validate_sample_input_sparse_like_fns,
        op_info,
        device,
        dtype,
        requires_grad,
        layout,
        **kwargs,
    )


def error_inputs_sparse_like_fns(op_info, device, layout, **kwargs):
    # [FUNCTION DOC]
    # Generator of ErrorInput cases for like-functions on sparse tensors, using float64 and no
    # requires_grad.
    """Error inputs for like-functions on sparse tensors."""
    dtype = torch.float64
    requires_grad = False
    yield from _error_inputs_sparse(
        _maybe_failing_sample_inputs_sparse_like_fns,
        _validate_sample_input_sparse_like_fns,
        op_info,
        device,
        dtype,
        requires_grad,
        layout,
        **kwargs,
    )


def _validate_sample_input_sparse_default(op_info, sample, check_validate=False):
    # [FUNCTION DOC]
    # HELPER: Default validator. For to_sparse on a compressed input with an integer sparse_dim
    # argument other than 2, returns an ErrorInput; otherwise returns the sample. Optionally
    # runs the op to confirm when check_validate is set.
    if op_info.name == "to_sparse":
        if (
            sample.input.layout
            in {torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}
            and len(sample.args) == 1
            and isinstance(sample.args[0], int)
            and sample.args[0] != 2
        ):
            sample = ErrorInput(
                sample,
                error_regex="sparse dim argument must be 2 for sparse_compressed_to_sparse",
            )

    if check_validate:
        _check_validate(op_info, sample)
    return sample


def validate_sample_input_sparse(op_info, sample, check_validate=False):
    # [FUNCTION DOC]
    # Entry point validator: dispatches on the OpInfo type to the reduction validator
    # (ReductionOpInfo), the elementwise binary validator (BinaryUfuncInfo) or the default
    # validator. Returns the sample as is when valid, else an ErrorInput.
    """Return the specified sample when it is valid and supported by the
    operation. Otherwise, return the sample as ErrorInput instance.

    When check_validate is True, the result is validated against
    calling the op on the sample.
    """
    if isinstance(op_info, ReductionOpInfo):
        return _validate_sample_input_sparse_reduction(
            op_info, sample, check_validate=check_validate
        )
    elif isinstance(op_info, BinaryUfuncInfo):
        return _validate_sample_input_sparse_elementwise_binary_operation(
            op_info, sample, check_validate=check_validate
        )
    else:
        return _validate_sample_input_sparse_default(
            op_info, sample, check_validate=check_validate
        )
