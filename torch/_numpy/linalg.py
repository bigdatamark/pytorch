# [MODULE DOC] Numpy-compatible linear algebra functions implemented over
# torch.linalg. Contents: matrix and vector products (matrix_power, multi_dot),
# solving and inversion (solve, lstsq, inv, pinv, tensorsolve, tensorinv), norms and
# scalar quantities (det, slogdet, cond, matrix_rank, norm), decompositions (cholesky,
# qr, svd) and eigen routines (eig, eigh, eigvals, eigvalsh). Each public function is
# wrapped by an argument normalizer and an error-translation decorator that maps torch
# linear-algebra errors to LinAlgError; integer or bool inputs are promoted to
# floating point. Defines LinAlgError. No import-time side effects.
from __future__ import annotations

import functools
import math
from typing import ParamSpec, TYPE_CHECKING, TypeVar

import torch

from . import _dtypes_impl, _util
from ._normalizations import ArrayLike, KeepDims, normalizer, OutArray


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


_P = ParamSpec("_P")
_R = TypeVar("_R")


class LinAlgError(Exception):
    # [CLASS DOC] Exception raised for linear algebra failures (singular matrix, non-
    # convergence, undefined on empty input); the numpy-compatible counterpart of torch's
    # internal linear algebra error. No added state.
    pass


def _atleast_float_1(a: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Return the tensor unchanged if floating or complex; otherwise
    # convert it to the default float dtype.
    if not (a.dtype.is_floating_point or a.dtype.is_complex):
        a = a.to(_dtypes_impl.default_dtypes().float_dtype)
    return a


def _atleast_float_2(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] HELPER: Cast two tensors to their common result dtype, or to the
    # default float dtype if that is not floating or complex. Returns the pair of tensors.
    dtyp = _dtypes_impl.result_type_impl(a, b)
    if not (dtyp.is_floating_point or dtyp.is_complex):
        dtyp = _dtypes_impl.default_dtypes().float_dtype

    a = _util.cast_if_needed(a, dtyp)
    b = _util.cast_if_needed(b, dtyp)
    return a, b


def linalg_errors(func: Callable[_P, _R]) -> Callable[_P, _R]:
    # [FUNCTION DOC] HELPER: Decorator translating torch's internal linear algebra error
    # into LinAlgError (same arguments, original exception not chained explicitly). Other
    # exceptions propagate. Returns the wrapper.
    @functools.wraps(func)
    def wrapped(*args: _P.args, **kwds: _P.kwargs) -> _R:
        # [FUNCTION DOC] HELPER: Calls the wrapped function; on a torch linear algebra error
        # raises LinAlgError with the same arguments.
        try:
            return func(*args, **kwds)
        except torch._C._LinAlgError as e:  # pyrefly: ignore[missing-attribute]  # TODO
            raise LinAlgError(*e.args)  # noqa: B904

    return wrapped


# ### Matrix and vector products ###


@normalizer
@linalg_errors
def matrix_power(a: ArrayLike, n: int) -> torch.Tensor:
    # [FUNCTION DOC] Raise a square matrix (or batch) to the integer power n; negative n
    # uses the inverse. Input promoted to float.
    a = _atleast_float_1(a)
    return torch.linalg.matrix_power(a, n)


@normalizer
@linalg_errors
def multi_dot(
    inputs: Sequence[ArrayLike], *, out: OutArray | None = None
) -> torch.Tensor:
    # [FUNCTION DOC] Multiply a sequence of matrices in an order chosen to minimize cost.
    # out is accepted but unused here.
    return torch.linalg.multi_dot(inputs)


# ### Solving equations and inverting matrices ###


@normalizer
@linalg_errors
def solve(a: ArrayLike, b: ArrayLike) -> torch.Tensor:
    # [FUNCTION DOC] Solve the linear system a x = b. Both inputs are promoted to a common
    # floating dtype. Raises LinAlgError for singular a.
    a, b = _atleast_float_2(a, b)
    return torch.linalg.solve(a, b)


@normalizer
@linalg_errors
def lstsq(
    a: ArrayLike, b: ArrayLike, rcond: float | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Least-squares solution of a x = b with cutoff rcond. Returns a four-
    # tuple (solution, residuals, rank, singular values) from torch. Uses the gels driver
    # on CUDA inputs and gelsd otherwise.
    a, b = _atleast_float_2(a, b)
    # NumPy is using gelsd: https://github.com/numpy/numpy/blob/v1.24.0/numpy/linalg/umath_linalg.cpp#L3991
    # on CUDA, only `gels` is available though, so use it instead
    driver = "gels" if a.is_cuda or b.is_cuda else "gelsd"
    return torch.linalg.lstsq(a, b, rcond=rcond, driver=driver)


@normalizer
@linalg_errors
def inv(a: ArrayLike) -> torch.Tensor:
    # [FUNCTION DOC] Matrix inverse (batched). Input promoted to float. Raises LinAlgError
    # if singular.
    a = _atleast_float_1(a)
    result = torch.linalg.inv(a)
    return result


@normalizer
@linalg_errors
def pinv(a: ArrayLike, rcond: float = 1e-15, hermitian: bool = False) -> torch.Tensor:
    # [FUNCTION DOC] Moore-Penrose pseudo-inverse with relative cutoff rcond; hermitian
    # flag is forwarded to torch.
    a = _atleast_float_1(a)
    return torch.linalg.pinv(a, rtol=rcond, hermitian=hermitian)


@normalizer
@linalg_errors
def tensorsolve(
    a: ArrayLike, b: ArrayLike, axes: Sequence[int] | None = None
) -> torch.Tensor:
    # [FUNCTION DOC] Solve a tensor equation, optionally moving the given axes of a to the
    # end first. Inputs promoted to a common float dtype.
    a, b = _atleast_float_2(a, b)
    return torch.linalg.tensorsolve(a, b, dims=axes)


@normalizer
@linalg_errors
def tensorinv(a: ArrayLike, ind: int = 2) -> torch.Tensor:
    # [FUNCTION DOC] Inverse of a tensor with respect to the first ind dimensions (default
    # 2). Input promoted to float.
    a = _atleast_float_1(a)
    return torch.linalg.tensorinv(a, ind=ind)


# ### Norms and other numbers ###


@normalizer
@linalg_errors
def det(a: ArrayLike) -> torch.Tensor:
    # [FUNCTION DOC] Determinant of a square matrix or batch; input promoted to float.
    a = _atleast_float_1(a)
    return torch.linalg.det(a)


@normalizer
@linalg_errors
def slogdet(a: ArrayLike) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Sign and natural log of the absolute determinant; returns the pair
    # from torch.
    a = _atleast_float_1(a)
    return torch.linalg.slogdet(a)


@normalizer
@linalg_errors
def cond(x: ArrayLike, p: int | str | None = None) -> torch.Tensor:
    # [FUNCTION DOC] Condition number of a matrix for norm order p. Raises LinAlgError for
    # an empty matrix. NaN results are replaced by infinity.
    x = _atleast_float_1(x)

    # check if empty
    # cf: https://github.com/numpy/numpy/blob/v1.24.0/numpy/linalg/linalg.py#L1744
    if x.numel() == 0 and math.prod(x.shape[-2:]) == 0:
        raise LinAlgError("cond is not defined on empty arrays")

    result = torch.linalg.cond(x, p=p)

    # Convert nans to infs (numpy does it in a data-dependent way, depending on
    # whether the input array has nans or not)
    # XXX: NumPy does this: https://github.com/numpy/numpy/blob/v1.24.0/numpy/linalg/linalg.py#L1744
    return torch.where(torch.isnan(result), float("inf"), result)


@normalizer
@linalg_errors
def matrix_rank(
    a: ArrayLike, tol: float | None = None, hermitian: bool = False
) -> torch.Tensor | int:
    # [FUNCTION DOC] Matrix rank via singular values. For inputs with fewer than two
    # dimensions returns an int, 1 if any element is nonzero else 0. With tol None uses a
    # relative tolerance of the larger matrix dimension times machine epsilon; otherwise
    # tol is an absolute tolerance. hermitian forwarded.
    a = _atleast_float_1(a)

    if a.ndim < 2:
        return int((a != 0).any())

    if tol is None:
        # follow https://github.com/numpy/numpy/blob/v1.24.0/numpy/linalg/linalg.py#L1885
        atol = 0
        rtol = max(a.shape[-2:]) * torch.finfo(a.dtype).eps
    else:
        atol, rtol = tol, 0
    return torch.linalg.matrix_rank(a, atol=atol, rtol=rtol, hermitian=hermitian)


@normalizer
@linalg_errors
def norm(
    x: ArrayLike,
    ord: int | float | str | None = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: KeepDims = False,
) -> torch.Tensor:
    # [FUNCTION DOC] Vector or matrix norm of order ord over axis. keepdims is accepted
    # but not applied here.
    # KNOWN ISSUE: keepdims is ignored by this function.
    x = _atleast_float_1(x)
    return torch.linalg.norm(x, ord=ord, dim=axis)


# ### Decompositions ###


@normalizer
@linalg_errors
def cholesky(a: ArrayLike) -> torch.Tensor:
    # [FUNCTION DOC] Lower-triangular Cholesky factor of a positive-definite matrix; input
    # promoted to float. Raises LinAlgError if not positive definite.
    a = _atleast_float_1(a)
    return torch.linalg.cholesky(a)


@normalizer
@linalg_errors
def qr(
    a: ArrayLike, mode: str = "reduced"
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    # [FUNCTION DOC] QR decomposition. mode selects reduced, complete or r; for mode r
    # returns only the R factor (numpy behavior), otherwise the named tuple of Q and R.
    a = _atleast_float_1(a)
    result = torch.linalg.qr(a, mode=mode)
    if mode == "r":
        # match NumPy
        return result.R
    return result


@normalizer
@linalg_errors
def svd(
    a: ArrayLike,
    full_matrices: bool = True,
    compute_uv: bool = True,
    hermitian: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
    # [FUNCTION DOC] Singular value decomposition. When compute_uv is false returns only
    # singular values; otherwise returns the (U, S, Vh) tuple, honoring full_matrices.
    # hermitian is ignored.
    a = _atleast_float_1(a)
    if not compute_uv:
        return torch.linalg.svdvals(a)

    # NB: ignore the hermitian= argument (no pytorch equivalent)
    result = torch.linalg.svd(a, full_matrices=full_matrices)
    return result


# ### Eigenvalues and eigenvectors ###


@normalizer
@linalg_errors
def eig(a: ArrayLike) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Eigenvalues and right eigenvectors of a general matrix. If the input
    # is real and all eigenvalues have zero imaginary part, both results are converted to
    # real. Returns the pair (w, v).
    a = _atleast_float_1(a)
    w, vt = torch.linalg.eig(a)

    if not a.is_complex() and w.is_complex() and (w.imag == 0).all():
        w = w.real
        vt = vt.real
    return w, vt


@normalizer
@linalg_errors
def eigh(a: ArrayLike, UPLO: str = "L") -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Eigenvalues and eigenvectors of a Hermitian (or real symmetric)
    # matrix using the UPLO triangle. Returns the pair.
    a = _atleast_float_1(a)
    return torch.linalg.eigh(a, UPLO=UPLO)


@normalizer
@linalg_errors
def eigvals(a: ArrayLike) -> torch.Tensor:
    # [FUNCTION DOC] Eigenvalues of a general matrix; if the input is real and all
    # eigenvalues are real-valued, returns real dtype, else complex.
    a = _atleast_float_1(a)
    result = torch.linalg.eigvals(a)
    if not a.is_complex() and result.is_complex() and (result.imag == 0).all():
        result = result.real
    return result


@normalizer
@linalg_errors
def eigvalsh(a: ArrayLike, UPLO: str = "L") -> torch.Tensor:
    # [FUNCTION DOC] Eigenvalues of a Hermitian (or real symmetric) matrix using the UPLO
    # triangle.
    a = _atleast_float_1(a)
    return torch.linalg.eigvalsh(a, UPLO=UPLO)
