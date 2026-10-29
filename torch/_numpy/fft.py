# [MODULE DOC] Numpy-compatible discrete Fourier transform API over torch.fft. Each
# transform is wrapped by a decorator that upcasts the input to the default 64-bit-
# style float or complex dtype (numpy fft returns 64-bit results) and by the argument
# normalizer that converts array-likes to tensors. Contents: 1D (fft, ifft, rfft,
# irfft, hfft, ihfft), N-D (fftn, ifftn, rfftn, irfftn), 2D (fft2, ifft2, rfft2,
# irfft2), frequency helpers (fftfreq, rfftfreq) and shifts (fftshift, ifftshift). No
# import-time side effects.
from __future__ import annotations

import functools
from typing import Concatenate, ParamSpec, TYPE_CHECKING, TypeVar

import torch

from . import _dtypes_impl, _util
from ._normalizations import ArrayLike, normalizer


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


_P = ParamSpec("_P")
_R = TypeVar("_R")


def upcast(
    func: Callable[Concatenate[torch.Tensor, _P], _R],
) -> Callable[Concatenate[torch.Tensor, _P], _R]:
    # [FUNCTION DOC] HELPER: Decorator for fft functions. The wrapped function casts its
    # first tensor argument to the default complex dtype if complex, else to the default
    # float dtype, then calls the original function with the remaining arguments. Returns
    # the wrapper.
    """NumPy fft casts inputs to 64 bit and *returns 64-bit results*."""

    @functools.wraps(func)
    def wrapped(tensor: torch.Tensor, *args: _P.args, **kwds: _P.kwargs) -> _R:
        # [FUNCTION DOC] HELPER: Wrapper that performs the dtype upcast of the first tensor
        # argument and forwards all arguments.
        target_dtype = (
            _dtypes_impl.default_dtypes().complex_dtype
            if tensor.is_complex()
            else _dtypes_impl.default_dtypes().float_dtype
        )
        tensor = _util.cast_if_needed(tensor, target_dtype)
        return func(tensor, *args, **kwds)

    return wrapped


@normalizer
@upcast
def fft(
    a: ArrayLike, n: int | None = None, axis: int = -1, norm: str | None = None
) -> torch.Tensor:
    # [FUNCTION DOC] 1D discrete Fourier transform along axis, with optional length n
    # (pad/trim) and normalization mode norm. Returns a complex tensor.
    return torch.fft.fft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def ifft(
    a: ArrayLike, n: int | None = None, axis: int = -1, norm: str | None = None
) -> torch.Tensor:
    # [FUNCTION DOC] 1D inverse discrete Fourier transform along axis, with optional
    # length n and normalization mode norm. Returns a complex tensor.
    return torch.fft.ifft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def rfft(
    a: ArrayLike, n: int | None = None, axis: int = -1, norm: str | None = None
) -> torch.Tensor:
    # [FUNCTION DOC] 1D FFT of real input along axis returning only the non-negative
    # frequency half; optional length n and norm.
    return torch.fft.rfft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def irfft(
    a: ArrayLike, n: int | None = None, axis: int = -1, norm: str | None = None
) -> torch.Tensor:
    # [FUNCTION DOC] Inverse of rfft along axis: maps a one-sided Hermitian spectrum to
    # real output of length n (default derived from input); optional norm.
    return torch.fft.irfft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def fftn(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] N-dimensional FFT over the given axes (all axes when None) with
    # optional output shape s and norm.
    return torch.fft.fftn(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def ifftn(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] N-dimensional inverse FFT over the given axes (all when None) with
    # optional output shape s and norm.
    return torch.fft.ifftn(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def rfftn(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] N-dimensional FFT of real input over the given axes, returning the
    # one-sided half along the last transformed axis; optional s and norm.
    return torch.fft.rfftn(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def irfftn(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Inverse of rfftn over the given axes, producing real output; optional
    # s and norm.
    return torch.fft.irfftn(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def fft2(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] 2D FFT over axes (default the last two) with optional shape s and
    # norm.
    return torch.fft.fft2(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def ifft2(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] 2D inverse FFT over axes (default the last two) with optional shape s
    # and norm.
    return torch.fft.ifft2(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def rfft2(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] 2D FFT of real input over axes (default the last two), one-sided
    # along the last axis; optional s and norm.
    return torch.fft.rfft2(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def irfft2(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Inverse of rfft2 over axes (default the last two), producing real
    # output; optional s and norm.
    return torch.fft.irfft2(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def hfft(
    a: ArrayLike, n: int | None = None, axis: int = -1, norm: str | None = None
) -> torch.Tensor:
    # [FUNCTION DOC] FFT of a Hermitian-symmetric signal (one-sided time domain) along
    # axis giving a real spectrum; optional length n and norm.
    return torch.fft.hfft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def ihfft(
    a: ArrayLike, n: int | None = None, axis: int = -1, norm: str | None = None
) -> torch.Tensor:
    # [FUNCTION DOC] Inverse of hfft: transforms a real signal along axis into the one-
    # sided Hermitian representation; optional length n and norm.
    return torch.fft.ihfft(a, n, dim=axis, norm=norm)


@normalizer
def fftfreq(n: int, d: float = 1.0) -> torch.Tensor:
    # [FUNCTION DOC] Sample frequencies for an n-point FFT with sample spacing d. Returns
    # a tensor. Not upcast.
    return torch.fft.fftfreq(n, d)


@normalizer
def rfftfreq(n: int, d: float = 1.0) -> torch.Tensor:
    # [FUNCTION DOC] Sample frequencies for an n-point real FFT (non-negative half) with
    # sample spacing d. Returns a tensor.
    return torch.fft.rfftfreq(n, d)


@normalizer
def fftshift(x: ArrayLike, axes: int | Sequence[int] | None = None) -> torch.Tensor:
    # [FUNCTION DOC] Shift the zero-frequency component to the centre along the given axes
    # (all axes when None). Returns a tensor.
    return torch.fft.fftshift(x, axes)


@normalizer
def ifftshift(x: ArrayLike, axes: int | Sequence[int] | None = None) -> torch.Tensor:
    # [FUNCTION DOC] Inverse of fftshift along the given axes (all when None). Returns a
    # tensor.
    return torch.fft.ifftshift(x, axes)
