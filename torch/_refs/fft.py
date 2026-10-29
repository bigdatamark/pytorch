# [MODULE DOC]
# Reference implementations of torch.fft transforms (1-D, 2-D, N-D, real/Hermitian variants and
# shift helpers) built on the fft_c2c, fft_r2c and fft_c2r prims, with dtype promotion,
# size padding/trimming and normalization helpers. Import time: registers each public function as
# a decomposition of the matching aten op.
import math
from collections.abc import Iterable, Sequence
from typing import Literal, NamedTuple

import torch
import torch._prims as prims
import torch._prims_common as utils
from torch._decomp import register_decomposition
from torch._prims_common import DimsType, ShapeType, TensorLikeType
from torch._prims_common.wrappers import _maybe_convert_to_dtype, out_wrapper


__all__ = [
    # Transforms
    "fft",
    "fft2",
    "fftn",
    "hfft",
    "hfft2",
    "hfftn",
    "rfft",
    "rfft2",
    "rfftn",
    "ifft",
    "ifft2",
    "ifftn",
    "ihfft",
    "ihfft2",
    "ihfftn",
    "irfft",
    "irfft2",
    "irfftn",
    # Helpers
    "fftshift",
    "ifftshift",
]

NormType = None | Literal["forward", "backward", "ortho"]
_NORM_VALUES = {None, "forward", "backward", "ortho"}
aten = torch._ops.ops.aten


def _apply_norm(
    x: TensorLikeType, norm: NormType, signal_numel: int, forward: bool
) -> TensorLikeType:
    # [FUNCTION DOC] HELPER: Scales an unnormalized FFT result x. Raises an error for an invalid norm mode.
    # ortho multiplies by 1/sqrt(signal_numel); otherwise multiplies by 1/signal_numel when
    # (inverse transform with norm None/backward) or (forward transform with norm forward).
    """Apply normalization to the un-normalized FFT result"""
    torch._check(norm in _NORM_VALUES, lambda: f"Invalid normalization mode: {norm}")

    if norm == "ortho":
        return x * (1 / math.sqrt(signal_numel))

    normalize = (not forward and (norm is None or norm == "backward")) or (
        forward and norm == "forward"
    )
    return x * (1 / signal_numel) if normalize else x


def _promote_type_fft(
    dtype: torch.dtype, require_complex: bool, device: torch.device
) -> torch.dtype:
    # [FUNCTION DOC] HELPER: Maps dtype to one supported by the FFT prims: complex kept; integral becomes the
    # default float dtype; float32/float64 allowed, plus float16 on cuda/meta/xpu (bfloat16 is
    # promoted to float32 there). Raises not-implemented for other dtypes. With require_complex,
    # returns the matching complex dtype.
    """Helper to promote a dtype to one supported by the FFT primitives"""
    if dtype.is_complex:
        return dtype

    # Promote integral to default float type
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()

    allowed_types = [torch.float32, torch.float64]
    maybe_support_half = device.type in ["cuda", "meta", "xpu"]

    if maybe_support_half:
        allowed_types.append(torch.float16)
        # bfloat16 is supported for FFT on CUDA/XPU devices, but there is no
        # corresponding "complex bfloat16" dtype (corresponding_complex_dtype
        # maps it to complex64). Promote bfloat16 -> float32 here so that:
        #   * the decomposition output dtype (complex64) matches aten, and
        #   * a real dtype conversion happens, so the result never aliases the
        #     input. Otherwise, for size-1 inputs the prim would return a view
        #     while aten returns a fresh tensor, breaking view-consistency
        #     (test_python_ref__refs_fft_*_cuda_bfloat16).
        if dtype == torch.bfloat16:
            dtype = torch.float32
    torch._check_not_implemented(
        dtype in allowed_types, lambda: f"Unsupported dtype {dtype}"
    )

    if require_complex:
        dtype = utils.corresponding_complex_dtype(dtype)

    return dtype


def _maybe_promote_tensor_fft(
    t: TensorLikeType, require_complex: bool = False
) -> TensorLikeType:
    # [FUNCTION DOC] HELPER: Converts tensor t to the dtype chosen by _promote_type_fft (no-op if unchanged).
    """Helper to promote a tensor to a dtype supported by the FFT primitives"""
    cur_type = t.dtype
    new_type = _promote_type_fft(cur_type, require_complex, t.device)
    return _maybe_convert_to_dtype(t, new_type)  # type: ignore[return-value]


def _resize_fft_input(
    x: TensorLikeType, dims: tuple[int, ...], sizes: tuple[int, ...]
) -> TensorLikeType:
    # [FUNCTION DOC] HELPER: Makes x have size sizes[i] along dims[i] by narrowing from index 0 (if larger) or
    # zero padding at the end (if smaller); a size of -1 leaves the dim alone. Raises if dims
    # and sizes differ in length. Returns x itself when no padding was needed and no narrowing occurred.
    """
    Fixes the shape of x such that x.size(dims[i]) == sizes[i],
    either by zero-padding, or by slicing x starting from 0.
    """
    if len(dims) != len(sizes):
        raise AssertionError(
            f"dims and sizes must have the same length, got {len(dims)} and {len(sizes)}"
        )
    must_copy = False
    x_sizes = x.shape
    pad_amount = [0] * len(x_sizes) * 2
    for i in range(len(dims)):
        if sizes[i] == -1:
            continue

        if x_sizes[dims[i]] < sizes[i]:
            must_copy = True
            pad_idx = len(pad_amount) - 2 * dims[i] - 1

            pad_amount[pad_idx] = sizes[i] - x_sizes[dims[i]]

        if x_sizes[dims[i]] > sizes[i]:
            x = x.narrow(dims[i], 0, sizes[i])

    return torch.constant_pad_nd(x, pad_amount) if must_copy else x


def _fft_c2r(
    func_name: str,
    input: TensorLikeType,
    n: int | None,
    dim: int,
    norm: NormType,
    forward: bool,
) -> TensorLikeType:
    # [FUNCTION DOC] HELPER: Shared 1-D complex-to-real FFT for irfft/hfft. Promotes input to complex, computes
    # output length n (default 2*(size-1)), errors if below 1, resizes input to n//2+1, conjugates
    # when forward (hfft), runs the c2r prim and applies normalization.
    """Common code for performing any complex to real FFT (irfft or hfft)"""
    input = _maybe_promote_tensor_fft(input, require_complex=True)
    dims = (utils.canonicalize_dim(input.ndim, dim, wrap_scalar=False),)
    last_dim_size = n if n is not None else 2 * (input.shape[dim] - 1)
    torch._check(
        last_dim_size >= 1,
        lambda: f"Invalid number of data points ({last_dim_size}) specified",
    )

    if n is not None:
        input = _resize_fft_input(input, dims=dims, sizes=(last_dim_size // 2 + 1,))

    if forward:
        input = torch.conj(input)

    output = prims.fft_c2r(input, dim=dims, last_dim_size=last_dim_size)
    return _apply_norm(output, norm=norm, signal_numel=last_dim_size, forward=forward)


def _fft_r2c(
    func_name: str,
    input: TensorLikeType,
    n: int | None,
    dim: int,
    norm: NormType,
    forward: bool,
    onesided: bool,
) -> TensorLikeType:
    # [FUNCTION DOC] HELPER: Shared 1-D real-to-complex FFT for rfft/ihfft (and fft/ifft on real input). Errors on
    # complex input or non-positive length, promotes dtype, resizes to n, runs the r2c prim
    # (onesided or not), normalizes, and conjugates the result unless forward.
    """Common code for performing any real to complex FFT (rfft or ihfft)"""
    torch._check_type(
        not input.dtype.is_complex,
        lambda: f"{func_name} expects a floating point input tensor, but got {input.dtype}",
    )
    input = _maybe_promote_tensor_fft(input)
    dims = (utils.canonicalize_dim(input.ndim, dim, wrap_scalar=False),)
    dim_size = n if n is not None else input.shape[dim]
    torch._check(
        dim_size >= 1, lambda: f"Invalid number of data points ({dim_size}) specified"
    )

    if n is not None:
        input = _resize_fft_input(input, dims, (n,))

    ret = prims.fft_r2c(input, dim=dims, onesided=onesided)
    ret = _apply_norm(ret, norm, dim_size, forward)
    return ret if forward else torch.conj(ret)


def _fft_c2c(
    func_name: str,
    input: TensorLikeType,
    n: int | None,
    dim: int,
    norm: NormType,
    forward: bool,
) -> TensorLikeType:
    # [FUNCTION DOC] HELPER: Shared 1-D complex-to-complex FFT for fft/ifft. Errors if input is not complex or
    # the length is below 1; resizes to n, runs the c2c prim in the given direction, normalizes.
    """Common code for performing any complex to complex FFT (fft or ifft)"""
    torch._check(
        input.dtype.is_complex,
        lambda: f"{func_name} expects a complex input tensor, but got {input.dtype}",
    )
    dims = (utils.canonicalize_dim(input.ndim, dim, wrap_scalar=False),)
    dim_size = n if n is not None else input.shape[dim]
    torch._check(
        dim_size >= 1, lambda: f"Invalid number of data points ({dim_size}) specified"
    )

    if n is not None:
        input = _resize_fft_input(input, dims, (n,))

    ret = prims.fft_c2c(input, dim=dims, forward=forward)
    return _apply_norm(ret, norm, dim_size, forward)


@register_decomposition(aten.fft_fft)
@out_wrapper()
def fft(
    input: TensorLikeType,
    n: int | None = None,
    dim: int = -1,
    norm: NormType = None,
) -> TensorLikeType:
    # [FUNCTION DOC] 1-D discrete Fourier transform along dim (decomposition of aten fft_fft). Complex input uses
    # the c2c path; real input uses the non-onesided r2c path. n pads/trims the signal; norm picks scaling.
    if input.dtype.is_complex:
        return _fft_c2c("fft", input, n, dim, norm, forward=True)
    else:
        return _fft_r2c("fft", input, n, dim, norm, forward=True, onesided=False)


@register_decomposition(aten.fft_ifft)
@out_wrapper()
def ifft(
    input: TensorLikeType,
    n: int | None = None,
    dim: int = -1,
    norm: NormType = None,
) -> TensorLikeType:
    # [FUNCTION DOC] 1-D inverse discrete Fourier transform along dim; complex input uses c2c, real input uses
    # the non-onesided r2c path with inverse direction. n pads/trims; norm picks scaling.
    if input.dtype.is_complex:
        return _fft_c2c("ifft", input, n, dim, norm, forward=False)
    else:
        return _fft_r2c("ifft", input, n, dim, norm, forward=False, onesided=False)


@register_decomposition(aten.fft_rfft)
@out_wrapper()
def rfft(
    input: TensorLikeType,
    n: int | None = None,
    dim: int = -1,
    norm: NormType = None,
) -> TensorLikeType:
    # [FUNCTION DOC] 1-D FFT of real input returning only the non-redundant half spectrum (n//2+1 entries).
    return _fft_r2c("rfft", input, n, dim, norm, forward=True, onesided=True)


@register_decomposition(aten.fft_irfft)
@out_wrapper()
def irfft(
    input: TensorLikeType,
    n: int | None = None,
    dim: int = -1,
    norm: NormType = None,
) -> TensorLikeType:
    # [FUNCTION DOC] 1-D inverse of rfft: complex half spectrum to real signal of length n (default 2*(size-1)).
    return _fft_c2r("irfft", input, n, dim, norm, forward=False)


@register_decomposition(aten.fft_hfft)
@out_wrapper()
def hfft(
    input: TensorLikeType,
    n: int | None = None,
    dim: int = -1,
    norm: NormType = None,
) -> TensorLikeType:
    # [FUNCTION DOC] 1-D FFT of a Hermitian-symmetric half signal, returning a real spectrum (complex-to-real, forward).
    return _fft_c2r("hfft", input, n, dim, norm, forward=True)


@register_decomposition(aten.fft_ihfft)
@out_wrapper()
def ihfft(
    input: TensorLikeType,
    n: int | None = None,
    dim: int = -1,
    norm: NormType = None,
) -> TensorLikeType:
    # [FUNCTION DOC] 1-D inverse of hfft: real signal to the one-sided Hermitian half spectrum (real-to-complex, inverse).
    return _fft_r2c("ihfft", input, n, dim, norm, forward=False, onesided=True)


class _ShapeAndDims(NamedTuple):
    # [CLASS DOC] HELPER: Named tuple pairing canonical transform sizes (shape) with the matching dims.
    shape: tuple[int, ...]
    dims: tuple[int, ...]


def _canonicalize_fft_shape_and_dim_args(
    input: TensorLikeType, shape: ShapeType | None, dim: DimsType | None
) -> _ShapeAndDims:
    # [FUNCTION DOC] HELPER: Resolves optional shape and dim into concrete tuples. dim is wrapped to non-negative
    # and must be unique; if shape is given with no dim, dims are the last len(shape) dims; -1 in
    # shape means the input size on that dim; with neither, all dims and sizes are used. Raises errors
    # if dim/shape lengths differ, shape is longer than ndim, or any size is not positive.
    # Returns a _ShapeAndDims.
    """Convert the shape and dim arguments into a canonical form where neither are optional"""
    input_dim = input.ndim
    input_sizes = input.shape

    if dim is not None:
        if not isinstance(dim, Sequence):
            dim = (dim,)
        ret_dims = utils.canonicalize_dims(input_dim, dim, wrap_scalar=False)

        # Check dims are unique
        torch._check(
            len(set(ret_dims)) == len(ret_dims), lambda: "FFT dims must be unique"
        )

    if shape is not None:
        if not isinstance(shape, Sequence):
            shape = (shape,)

        # Has shape, might have dim
        torch._check(
            dim is None or len(dim) == len(shape),
            lambda: "When given, dim and shape arguments must have the same length",
        )
        transform_ndim = len(shape)

        torch._check(
            transform_ndim <= input_dim,
            lambda: f"Got shape with {transform_ndim} values but input tensor "
            f"only has {input_dim} dimensions.",
        )

        # If shape is given, dims defaults to the last len(shape) dimensions
        if dim is None:
            ret_dims = tuple(range(input_dim - transform_ndim, input_dim))

        # Translate any -1 values in shape to the default length
        ret_shape = tuple(
            s if s != -1 else input_sizes[d]
            for (s, d) in zip(shape, ret_dims)  # type: ignore[possibly-undefined]
        )
    elif dim is None:
        # No shape, no dim
        ret_dims = tuple(range(input_dim))
        ret_shape = tuple(input_sizes)
    else:
        # No shape, has dim
        ret_shape = tuple(input_sizes[d] for d in ret_dims)  # type: ignore[possibly-undefined]

    for n in ret_shape:
        torch._check(n > 0, lambda: f"Invalid number of data points ({n}) specified")

    return _ShapeAndDims(shape=ret_shape, dims=ret_dims)  # type: ignore[possibly-undefined]


def _prod(xs: Iterable[int]) -> int:
    # [FUNCTION DOC] HELPER: Returns the product of an iterable of ints (1 if empty).
    """Compute product of a list"""
    prod = 1
    for x in xs:
        prod *= x
    return prod


def _fftn_c2c(
    function_name: str,
    input: TensorLikeType,
    shape: tuple[int, ...],
    dim: tuple[int, ...],
    norm: NormType,
    forward: bool,
) -> TensorLikeType:
    # [FUNCTION DOC] HELPER: Shared n-D complex-to-complex FFT for fftn/ifftn. Errors if input is not complex;
    # resizes to shape on dim, runs c2c prim, normalizes by the product of shape.
    """Common code for n-dimensional complex to complex FFTs (fftn or ifftn)"""
    torch._check(
        input.dtype.is_complex,
        lambda: f"{function_name} expects a complex input tensor, "
        f"but got {input.dtype}",
    )
    x = _resize_fft_input(input, dim, shape)
    output = prims.fft_c2c(x, dim=dim, forward=forward)
    return _apply_norm(output, norm=norm, signal_numel=_prod(shape), forward=forward)


@register_decomposition(aten.fft_fftn)
@out_wrapper()
def fftn(
    input: TensorLikeType,
    s: ShapeType | None = None,
    dim: DimsType | None = None,
    norm: NormType = None,
) -> TensorLikeType:
    # [FUNCTION DOC] N-D FFT over dim with optional sizes s (decomposition of aten fft_fftn). Promotes input to
    # complex, canonicalizes args, then runs the forward c2c transform.
    (shape, dim) = _canonicalize_fft_shape_and_dim_args(input, s, dim)
    x = _maybe_promote_tensor_fft(input, require_complex=True)
    return _fftn_c2c("fftn", x, shape, dim, norm, forward=True)


@register_decomposition(aten.fft_ifftn)
@out_wrapper()
def ifftn(
    input: TensorLikeType,
    s: ShapeType | None = None,
    dim: DimsType | None = None,
    norm: NormType = None,
) -> TensorLikeType:
    # [FUNCTION DOC] N-D inverse FFT over dim with optional sizes s; promotes to complex and runs inverse c2c.
    (shape, dim) = _canonicalize_fft_shape_and_dim_args(input, s, dim)
    x = _maybe_promote_tensor_fft(input, require_complex=True)
    return _fftn_c2c("ifftn", x, shape, dim, norm, forward=False)


@register_decomposition(aten.fft_rfftn)
@out_wrapper()
def rfftn(
    input: TensorLikeType,
    s: ShapeType | None = None,
    dim: DimsType | None = None,
    norm: NormType = None,
) -> TensorLikeType:
    # [FUNCTION DOC] N-D FFT of real input over dim returning the one-sided spectrum (last dim halved). Raises an
    # error for complex input.
    torch._check_type(
        not input.dtype.is_complex,
        lambda: f"rfftn expects a real-valued input tensor, but got {input.dtype}",
    )
    shape, dim = _canonicalize_fft_shape_and_dim_args(input, s, dim)
    input = _maybe_promote_tensor_fft(input, require_complex=False)
    input = _resize_fft_input(input, dim, shape)
    out = prims.fft_r2c(input, dim=dim, onesided=True)
    return _apply_norm(out, norm=norm, signal_numel=_prod(shape), forward=True)


@register_decomposition(aten.fft_ihfftn)
@out_wrapper()
def ihfftn(
    input: TensorLikeType,
    s: ShapeType | None = None,
    dim: DimsType | None = None,
    norm: NormType = None,
) -> TensorLikeType:
    # [FUNCTION DOC] N-D inverse of hfftn on real input: r2c on the last dim, then conjugate and inverse c2c over the
    # remaining dims, with inverse normalization. Errors for complex input or when no axis is transformed.
    torch._check_type(
        not input.dtype.is_complex,
        lambda: f"ihfftn expects a real-valued input tensor, but got {input.dtype}",
    )
    shape, dim = _canonicalize_fft_shape_and_dim_args(input, s, dim)
    torch._check(len(shape) > 0, lambda: "ihfftn must transform at least one axis")
    input = _maybe_promote_tensor_fft(input, require_complex=False)
    input = _resize_fft_input(input, dim, shape)

    tmp = prims.fft_r2c(input, dim=dim[-1:], onesided=True)

    if len(dim) == 1:
        tmp = _apply_norm(tmp, norm=norm, signal_numel=shape[0], forward=False)
        return prims.conj(tmp)

    tmp = prims.conj_physical(tmp)
    tmp = prims.fft_c2c(tmp, dim=dim[:-1], forward=False)
    return _apply_norm(tmp, norm=norm, signal_numel=_prod(shape), forward=False)


class _CanonicalizeC2rReturn(NamedTuple):
    # [CLASS DOC] HELPER: Named tuple of canonical shape, dims and the output last_dim_size for c2r transforms.
    shape: tuple[int, ...]
    dim: tuple[int, ...]
    last_dim_size: int


def _canonicalize_fft_c2r_shape_and_dim_args(
    fname: str,
    input: TensorLikeType,
    s: ShapeType | None,
    dim: DimsType | None,
) -> _CanonicalizeC2rReturn:
    # [FUNCTION DOC] HELPER: Canonicalizes shape/dim for N-D complex-to-real transforms and computes last_dim_size
    # (from s if given and not -1, else 2*(input size - 1) on the last transform dim). Errors if no axis
    # is transformed or last_dim_size is below 1. Returns shape with last entry replaced by
    # last_dim_size//2+1, the dims, and last_dim_size.
    """Canonicalize shape and dim arguments for n-dimensional c2r transforms,
    as well as calculating the last_dim_size which is shape[dim[-1]] for the output"""
    (shape, dim) = _canonicalize_fft_shape_and_dim_args(input, s, dim)
    torch._check(len(shape) > 0, lambda: f"{fname} must transform at least one axis")

    if s is None or s[-1] == -1:
        last_dim_size = 2 * (input.shape[dim[-1]] - 1)
    else:
        last_dim_size = shape[-1]

    torch._check(
        last_dim_size >= 1,
        lambda: f"Invalid number of data points ({last_dim_size}) specified",
    )

    shape_list = list(shape)
    shape_list[-1] = last_dim_size // 2 + 1
    return _CanonicalizeC2rReturn(
        shape=tuple(shape_list), dim=dim, last_dim_size=last_dim_size
    )


@register_decomposition(aten.fft_irfftn)
@out_wrapper()
def irfftn(
    input: TensorLikeType,
    s: ShapeType | None = None,
    dim: DimsType | None = None,
    norm: NormType = None,
) -> TensorLikeType:
    # [FUNCTION DOC] N-D inverse of rfftn: complex half spectrum to real output via the c2r prim over dim, with
    # inverse normalization over the output sizes.
    shape, dim, last_dim_size = _canonicalize_fft_c2r_shape_and_dim_args(
        "irfftn", input, s, dim
    )
    input = _maybe_promote_tensor_fft(input, require_complex=True)
    input = _resize_fft_input(input, dim, shape)
    out = prims.fft_c2r(input, dim=dim, last_dim_size=last_dim_size)
    return _apply_norm(out, norm, _prod(out.shape[d] for d in dim), forward=False)


@register_decomposition(aten.fft_hfftn)
@out_wrapper()
def hfftn(
    input: TensorLikeType,
    s: ShapeType | None = None,
    dim: DimsType | None = None,
    norm: NormType = None,
) -> TensorLikeType:
    # [FUNCTION DOC] N-D FFT of a Hermitian half signal to a real spectrum: forward c2c over all but the last dim,
    # forward normalization, physical conjugation, then c2r on the last dim.
    shape, dim, last_dim_size = _canonicalize_fft_c2r_shape_and_dim_args(
        "hfftn", input, s, dim
    )
    input = _maybe_promote_tensor_fft(input, require_complex=True)
    input = _resize_fft_input(input, dim, shape)

    tmp = prims.fft_c2c(input, dim=dim[:-1], forward=True) if len(dim) > 1 else input
    tmp = _apply_norm(tmp, norm, _prod(shape[:-1]), forward=True)
    tmp = prims.conj_physical(tmp)
    out = prims.fft_c2r(tmp, dim=dim[-1:], last_dim_size=last_dim_size)
    return _apply_norm(out, norm, last_dim_size, forward=True)


@register_decomposition(aten.fft_fft2)
@out_wrapper()
def fft2(
    input: TensorLikeType,
    s: ShapeType | None = None,
    dim: DimsType | None = (-2, -1),
    norm: NormType = None,
) -> TensorLikeType:
    # [FUNCTION DOC] 2-D FFT; calls fftn with default dim of the last two dims.
    return torch.fft.fftn(input, s=s, dim=dim, norm=norm)


@register_decomposition(aten.fft_ifft2)
@out_wrapper()
def ifft2(
    input: TensorLikeType,
    s: ShapeType | None = None,
    dim: DimsType | None = (-2, -1),
    norm: NormType = None,
) -> TensorLikeType:
    # [FUNCTION DOC] 2-D inverse FFT; calls ifftn with default dim of the last two dims.
    return torch.fft.ifftn(input, s=s, dim=dim, norm=norm)


@register_decomposition(aten.fft_rfft2)
@out_wrapper()
def rfft2(
    input: TensorLikeType,
    s: ShapeType | None = None,
    dim: DimsType | None = (-2, -1),
    norm: NormType = None,
) -> TensorLikeType:
    # [FUNCTION DOC] 2-D real FFT; calls rfftn with default dim of the last two dims.
    return torch.fft.rfftn(input, s=s, dim=dim, norm=norm)


@register_decomposition(aten.fft_irfft2)
@out_wrapper()
def irfft2(
    input: TensorLikeType,
    s: ShapeType | None = None,
    dim: DimsType | None = (-2, -1),
    norm: NormType = None,
) -> TensorLikeType:
    # [FUNCTION DOC] 2-D inverse real FFT; calls irfftn with default dim of the last two dims.
    return torch.fft.irfftn(input, s=s, dim=dim, norm=norm)


@register_decomposition(aten.fft_hfft2)
@out_wrapper()
def hfft2(
    input: TensorLikeType,
    s: ShapeType | None = None,
    dim: DimsType | None = (-2, -1),
    norm: NormType = None,
) -> TensorLikeType:
    # [FUNCTION DOC] 2-D Hermitian FFT; calls hfftn with default dim of the last two dims.
    return torch.fft.hfftn(input, s=s, dim=dim, norm=norm)


@register_decomposition(aten.fft_ihfft2)
@out_wrapper()
def ihfft2(
    input: TensorLikeType,
    s: ShapeType | None = None,
    dim: DimsType | None = (-2, -1),
    norm: NormType = None,
) -> TensorLikeType:
    # [FUNCTION DOC] 2-D inverse Hermitian FFT; calls ihfftn with default dim of the last two dims.
    return torch.fft.ihfftn(input, s=s, dim=dim, norm=norm)


def _default_alldims(dim: DimsType | None, x: TensorLikeType) -> list[int]:
    # [FUNCTION DOC] HELPER: Converts an optional dim (None, int or sequence) to a list, defaulting to all dims of x.
    """Convert Optional[DimsType] to a simple list, defaulting to all dimensions"""
    if dim is None:
        return list(range(x.ndim))
    elif not isinstance(dim, Sequence):
        return [dim]
    else:
        return list(dim)


@register_decomposition(aten.fft_fftshift)
def fftshift(input: TensorLikeType, dim: DimsType | None = None) -> TensorLikeType:
    # [FUNCTION DOC] Rolls the zero-frequency component to the center: shifts each chosen dim (default all) by size//2.
    dims = _default_alldims(dim, input)
    shift = [input.shape[d] // 2 for d in dims]
    return torch.roll(input, shift, dims)


@register_decomposition(aten.fft_ifftshift)
def ifftshift(input: TensorLikeType, dim: DimsType | None = None) -> TensorLikeType:
    # [FUNCTION DOC] Inverse of fftshift: shifts each chosen dim (default all) by (size+1)//2.
    dims = _default_alldims(dim, input)
    shift = [(input.shape[d] + 1) // 2 for d in dims]
    return torch.roll(input, shift, dims)
