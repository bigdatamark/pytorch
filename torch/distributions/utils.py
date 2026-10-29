# [MODULE DOC]
# Shared helpers for torch.distributions: broadcasting of mixed number/tensor
# arguments, logits/probabilities conversion, probability clamping, a lazy
# caching property descriptor, sum over trailing dims, standard normal sampling
# and lower-triangular matrix/vector conversion. Defines the module constant
# euler_constant. No import-time side effects beyond imports and TypeVars.
from collections.abc import Callable, Sequence
from functools import update_wrapper
from typing import Any, Final, Generic, overload, TypeVar

import torch
import torch.nn.functional as F
from torch import SymInt, Tensor
from torch.overrides import is_tensor_like
from torch.types import _dtype, _Number, Device, Number


euler_constant: Final[float] = 0.57721566490153286060  # Euler Mascheroni Constant

__all__ = [
    "broadcast_all",
    "logits_to_probs",
    "clamp_probs",
    "probs_to_logits",
    "lazy_property",
    "tril_matrix_to_vec",
    "vec_to_tril_matrix",
]


# FIXME: Use (*values: *Ts) -> tuple[Tensor for T in Ts] if Mapping-Type is ever added.
#   See https://github.com/python/typing/issues/1216#issuecomment-2126153831
def broadcast_all(*values: Tensor | Number) -> tuple[Tensor, ...]:
    # [FUNCTION DOC]
    # Broadcasts a mix of numbers and tensor-like objects to a common shape.
    # Numbers become tensors using the dtype and device of the first real
    # tensor argument, or the default dtype if there is none. Returns a tuple
    # of broadcast tensors. Raises ValueError if any value is neither a number
    # nor tensor-like.
    r"""
    Given a list of values (possibly containing numbers), returns a list where each
    value is broadcasted based on the following rules:

    - `torch.*Tensor` instances are broadcasted as per :ref:`broadcasting-semantics`.
    - Number instances (scalars) are upcast to tensors having
      the same size and type as the first tensor passed to `values`.  If all the
      values are scalars, then they are upcasted to scalar Tensors.

    Args:
        values (list of `Number`, `torch.*Tensor` or objects implementing __torch_function__)

    Raises:
        ValueError: if any of the values is not a `Number` instance,
            a `torch.*Tensor` instance, or an instance implementing __torch_function__
    """
    if not all(is_tensor_like(v) or isinstance(v, _Number) for v in values):
        raise ValueError(
            "Input arguments must all be instances of Number, "
            "torch.Tensor or objects implementing __torch_function__."
        )
    if not all(is_tensor_like(v) for v in values):
        options: dict[str, Any] = dict(dtype=torch.get_default_dtype())
        for value in values:
            if isinstance(value, torch.Tensor):
                options = dict(dtype=value.dtype, device=value.device)
                break
        new_values = [
            v if is_tensor_like(v) else torch.tensor(v, **options) for v in values
        ]
        return torch.broadcast_tensors(*new_values)
    return torch.broadcast_tensors(*values)


def _standard_normal(
    shape: Sequence[int | SymInt],
    dtype: _dtype | None,
    device: Device | None,
) -> Tensor:
    # [FUNCTION DOC] HELPER: Returns a tensor of the given shape, dtype and
    # device filled with standard normal samples (consumes RNG state). Under
    # tracing it uses the two-argument normal op as a workaround instead of
    # the in-place fill.
    if torch._C._get_tracing_state():
        # [JIT WORKAROUND] lack of support for .normal_()
        return torch.normal(
            torch.zeros(shape, dtype=dtype, device=device),
            torch.ones(shape, dtype=dtype, device=device),
        )
    return torch.empty(shape, dtype=dtype, device=device).normal_()


def _sum_rightmost(value: Tensor, dim: int) -> Tensor:
    # [FUNCTION DOC] HELPER: Sums over the dim rightmost dimensions of value by
    # flattening them and reducing; returns value unchanged when dim is zero.
    # Behavior for negative dim is not defined.
    r"""
    Sum out ``dim`` many rightmost dimensions of a given tensor.

    Args:
        value (Tensor): A tensor of ``.dim()`` at least ``dim``.
        dim (int): The number of rightmost dims to sum out.
    """
    if dim == 0:
        return value
    required_shape = value.shape[:-dim] + (-1,)
    return value.reshape(required_shape).sum(-1)


def logits_to_probs(logits: Tensor, is_binary: bool = False) -> Tensor:
    # [FUNCTION DOC]
    # Converts logits to probabilities: elementwise sigmoid when binary,
    # otherwise softmax along the last dimension. Returns a tensor of the same
    # shape.
    r"""
    Converts a tensor of logits into probabilities. Note that for the
    binary case, each value denotes log odds, whereas for the
    multi-dimensional case, the values along the last dimension denote
    the log probabilities (possibly unnormalized) of the events.
    """
    if is_binary:
        return torch.sigmoid(logits)
    return F.softmax(logits, dim=-1)


def clamp_probs(probs: Tensor) -> Tensor:
    # [FUNCTION DOC]
    # Clamps probabilities into the closed range from eps to one minus eps,
    # where eps is the machine epsilon of the input dtype (so zero and one are
    # avoided). Returns a new tensor.
    """Clamps the probabilities to be in the open interval `(0, 1)`.

    The probabilities would be clamped between `eps` and `1 - eps`,
    and `eps` would be the smallest representable positive number for the input data type.

    Args:
        probs (Tensor): A tensor of probabilities.

    Returns:
        Tensor: The clamped probabilities.

    Examples:
        >>> probs = torch.tensor([0.0, 0.5, 1.0])
        >>> clamp_probs(probs)
        tensor([1.1921e-07, 5.0000e-01, 1.0000e+00])

        >>> probs = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        >>> clamp_probs(probs)
        tensor([2.2204e-16, 5.0000e-01, 1.0000e+00], dtype=torch.float64)

    """
    eps = torch.finfo(probs.dtype).eps
    return probs.clamp(min=eps, max=1 - eps)


def probs_to_logits(probs: Tensor, is_binary: bool = False) -> Tensor:
    # [FUNCTION DOC]
    # Clamps probs away from zero and one, then returns the log odds when
    # binary or the plain log otherwise (unnormalized logits).
    r"""
    Converts a tensor of probabilities into logits. For the binary case,
    this denotes the probability of occurrence of the event indexed by `1`.
    For the multi-dimensional case, the values along the last dimension
    denote the probabilities of occurrence of each of the events.
    """
    ps_clamped = clamp_probs(probs)
    if is_binary:
        return torch.log(ps_clamped) - torch.log1p(-ps_clamped)
    return torch.log(ps_clamped)


T = TypeVar("T", contravariant=True)
R = TypeVar("R", covariant=True)


class lazy_property(Generic[T, R]):
    # [CLASS DOC]
    # Non-data descriptor decorator that computes a method once per instance
    # and caches the result by storing it as an instance attribute of the same
    # name, which shadows the descriptor afterwards. State: wrapped (the
    # method). Accessed on the class it returns a property-like wrapper.
    r"""
    Used as a decorator for lazy loading of class attributes. This uses a
    non-data descriptor that calls the wrapped method to compute the property on
    first call; thereafter replacing the wrapped method into an instance
    attribute.
    """

    def __init__(self, wrapped: Callable[[T], R]) -> None:
        # [FUNCTION DOC] HELPER: Stores the wrapped method and copies its
        # metadata (name, docstring) onto the descriptor.
        self.wrapped: Callable[[T], R] = wrapped
        update_wrapper(self, wrapped)  # type:ignore[arg-type]

    @overload
    def __get__(
        self, instance: None, obj_type: Any = None
    ) -> "_lazy_property_and_property[T, R]": ...

    @overload
    def __get__(self, instance: T, obj_type: Any = None) -> R: ...

    def __get__(
        self, instance: T | None, obj_type: Any = None
    ) -> "R | _lazy_property_and_property[T, R]":
        # [FUNCTION DOC]
        # Descriptor access. On the class (no instance) returns a combined
        # lazy/property wrapper of the method. On an instance, computes the
        # value with gradients enabled, stores it as an instance attribute under
        # the method name (side effect: caches it) and returns it.
        if instance is None:
            return _lazy_property_and_property(self.wrapped)
        with torch.enable_grad():
            value = self.wrapped(instance)
        setattr(instance, self.wrapped.__name__, value)
        return value


class _lazy_property_and_property(lazy_property[T, R], property):
    # [CLASS DOC] HELPER: Object returned when a lazy_property is read from the
    # class; it is both a lazy_property and a builtin property so documentation
    # tools and distribution argument validation both recognize it.
    """We want lazy properties to look like multiple things.

    * property when Sphinx autodoc looks
    * lazy_property when Distribution validate_args looks
    """

    def __init__(self, wrapped: Callable[[T], R]) -> None:
        # [FUNCTION DOC] HELPER: Initializes only the builtin property part with
        # the wrapped method; lazy_property.__init__ is deliberately skipped.
        property.__init__(self, wrapped)


def tril_matrix_to_vec(mat: Tensor, diag: int = 0) -> Tensor:
    # [FUNCTION DOC]
    # Extracts the lower-triangular entries (up to the diag-th diagonal) of a
    # square matrix or batch of matrices in row order into a vector along the
    # last dimension. Raises ValueError when diag is outside the range from
    # minus n to n-1 (check skipped while tracing).
    r"""
    Convert a `D x D` matrix or a batch of matrices into a (batched) vector
    which comprises of lower triangular elements from the matrix in row order.
    """
    n = mat.shape[-1]
    if not torch._C._get_tracing_state() and (diag < -n or diag >= n):
        raise ValueError(f"diag ({diag}) provided is outside [{-n}, {n - 1}].")
    arange = torch.arange(n, device=mat.device)
    tril_mask = arange < arange.view(-1, 1) + (diag + 1)
    vec = mat[..., tril_mask]
    return vec


def vec_to_tril_matrix(vec: Tensor, diag: int = 0) -> Tensor:
    # [FUNCTION DOC]
    # Inverse of tril_matrix_to_vec: solves for the matrix size n from the last
    # dimension length, then builds a zero matrix of shape batch plus n by n and
    # fills the lower-triangular positions with vec in row order. Raises
    # ValueError if the length cannot form such a matrix (skipped when tracing).
    # KNOWN ISSUE: the check compares round(n) minus n against eps, which is
    # only positive when n rounds up, so non-integer n that rounds down (and
    # small deviations) are not rejected.
    r"""
    Convert a vector or a batch of vectors into a batched `D x D`
    lower triangular matrix containing elements from the vector in row order.
    """
    # +ve root of D**2 + (1+2*diag)*D - |diag| * (diag+1) - 2*vec.shape[-1] = 0
    n = (
        -(1 + 2 * diag)
        + ((1 + 2 * diag) ** 2 + 8 * vec.shape[-1] + 4 * abs(diag) * (diag + 1)) ** 0.5
    ) / 2
    eps = torch.finfo(vec.dtype).eps
    if not torch._C._get_tracing_state() and (round(n) - n > eps):
        raise ValueError(
            f"The size of last dimension is {vec.shape[-1]} which cannot be expressed as "
            + "the lower triangular part of a square D x D matrix."
        )
    n = round(n.item()) if isinstance(n, torch.Tensor) else round(n)
    mat = vec.new_zeros(vec.shape[:-1] + torch.Size((n, n)))
    arange = torch.arange(n, device=vec.device)
    tril_mask = arange < arange.view(-1, 1) + (diag + 1)
    mat[..., tril_mask] = vec
    return mat
