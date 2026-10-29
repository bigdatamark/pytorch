# [MODULE DOC] CuTeDSL override registrations for aten sum (inner-tree reduction) on CUDA. Registers the
# sum.dim_IntList and sum.IntList_out overrides, enabled only when the PYTORCH_SUM_INNER_TREE env var is
# truthy. Eligibility predicates accept only single-dim reductions of fp16/bf16/fp32/fp64 CUDA tensors
# with a contiguous fastest reduced dim and at most one outer dim; others fall through to ATen.
# The kernel itself is imported lazily from inner_tree_kernel. No import-time side effects.
"""CuTeDSL override registrations for ``aten::sum`` (inner-tree reduction).

CuTeDSL port of the Triton-style inner-tree reduction sum kernel that
otherwise lives in ``aten/src/ATen/native/cuda/ReduceSumProdKernel.cu``
(``try_inner_tree_reduction`` + the inner-tree kernels). This override
runs only when ``PYTORCH_SUM_INNER_TREE`` is set to a truthy value, which
is the feature-rollout gate for this operator -- it is *not* gated by the
global ``TORCH_DISABLE_NATIVE_JIT`` kill switch alone (that is handled by
``cutedsl_utils`` at registration time).

We register two ATen dispatcher entries on ``CUDA``:

* ``sum.dim_IntList`` -- functional/method ``x.sum(dim=...)``.
* ``sum.IntList_out`` -- the structured ``.out`` target.

``sum.dim_IntList`` is ``structured_delegate: sum.IntList_out``, so its
delegation to the ``.out`` kernel happens *below* the dispatcher; overriding
``.out`` alone would not intercept ``x.sum(dim=1)``. Both are separate
dispatcher entries and are registered explicitly (the same reason
``scatter_add_`` is registered separately from ``scatter_add.out``).

Eligibility mirrors ``try_inner_tree_reduction`` (ReduceSumProdKernel.cu):
build the reduction ``TensorIterator`` over ``(out, self)`` and accept only
the coalesced geometry the kernels know how to run -- a single contiguous
reduced (fastest) dimension whose non-reduced dims collapse to at most one
outer-strided dimension. Anything else (multi-dim reduction, non-contiguous
reduced dim, dtype-casting sum, integer/complex dtypes, non-collapsing outer
layout) falls through to aten.

Dispatch note: this CuTeDSL override is the only inner-tree path -- the ATen
CUDA backend has no inner-tree kernel of its own. When this predicate accepts,
the CuTeDSL kernel runs; when it rejects (or ``PYTORCH_SUM_INNER_TREE`` is
unset), the call falls through to the unchanged ATen reduction.
"""

from __future__ import annotations

import os

import torch
from torch._subclasses.fake_tensor import is_fake_tensor
from torch._tensor_iterator import reduce_op, TensorIterator

from ... import cutedsl_utils as cu


# Reduction input dtypes the kernel handles. fp32/fp64 are the bitwise-
# validated set (the inner-tree bitwise tests upcast fp16/bf16/fp8 data to
# fp32 before summing, and fp64 sums in fp64). fp16/bf16 native reductions
# (acc in fp32) are functionally supported but not part of the bitwise
# contract.
_SUPPORTED_DTYPES = frozenset(
    {torch.float16, torch.bfloat16, torch.float32, torch.float64}
)

# The kernel indexes rows/elements and sizes the launch grid with Int32. Decline
# reductions whose sizes would overflow that so we fall through to ATen rather
# than silently wrap (mirrors the magnitude guard from the CUDA review).
_INT32_MAX = 2**31 - 1


def _int32_indexing_safe(self: torch.Tensor, d: int) -> bool:
    # [FUNCTION DOC] HELPER: True if sizes of reducing dim d are safe for Int32 indexing and launch grid
    # (n, rows m, and m times the upper bound on batches all within int32 max); empty dims are safe.
    """Decline giant reductions so the kernel's Int32 row/element indexing and
    launch grid never wrap. The two-kernel grid is ``m * num_batches``;
    ``num_batches`` is bounded above by ``ceil(n / (32 * vec_size))`` (the
    smallest possible per-batch width), so this bound covers every path."""
    n = self.shape[d]
    if n == 0:
        return True
    m = self.numel() // n
    vec_size = max(1, 16 // self.element_size())
    num_batches_ub = -(-n // (32 * vec_size))
    return n <= _INT32_MAX and m <= _INT32_MAX and m * num_batches_ub <= _INT32_MAX


def _inner_tree_enabled() -> bool:
    # [FUNCTION DOC] HELPER: True if the PYTORCH_SUM_INNER_TREE env var is set to something other than empty or 0.
    # The CUDA gate is `use_inner_tree_reduction()`; the CuTeDSL predicate
    # requires the explicit feature flag (an empty/"0" value is off).
    return os.getenv("PYTORCH_SUM_INNER_TREE", "") not in ("", "0")


def _normalize_dim(dim: int, ndim: int) -> int:
    # [FUNCTION DOC] HELPER: Converts a negative dim to non-negative by adding ndim; no range check.
    return dim + ndim if dim < 0 else dim


def _single_dim(dim, ndim: int) -> int | None:
    # [FUNCTION DOC] HELPER: Returns the single normalized reduced dim from an int or sequence of dims, or
    # None if dim is None or the number of dims is not exactly one.
    """Return the one normalized reduced dim, or ``None`` if this is not a
    single-dim reduction (matches ``num_reduce_dims == 1``)."""
    if dim is None:
        return None
    if isinstance(dim, int):
        dims = [dim]
    else:
        dims = list(dim)
    if len(dims) != 1:
        return None
    return _normalize_dim(dims[0], ndim)


def _keepdim_out_shape(self: torch.Tensor, d: int) -> list[int]:
    # [FUNCTION DOC] HELPER: Shape list of self with dimension d set to 1 (keepdim output shape).
    shape = list(self.shape)
    shape[d] = 1
    return shape


def _out_keepdim_view(
    self: torch.Tensor, d: int, keepdim: bool, out: torch.Tensor
) -> torch.Tensor | None:
    # [FUNCTION DOC] HELPER: Returns a keepdim-shaped view of out (unsqueezing dim d when keepdim is false so
    # it is always a view), or None if unsqueeze fails or the shape differs from the expected reduced shape.
    """Return a keepdim-shaped ``(.., 1, ..)`` view of ``out`` so the kernel
    writes ``out``'s storage directly. ``unsqueeze`` is always a view (unlike
    ``reshape``, which may copy a non-contiguous ``out``). Returns ``None`` if
    ``out`` is not the expected reduced shape."""
    expected = _keepdim_out_shape(self, d)
    try:
        out_kd = out if keepdim else out.unsqueeze(d)
    except IndexError:
        return None
    return out_kd if list(out_kd.shape) == expected else None


def _eligibility(
    self: torch.Tensor, d: int, out: torch.Tensor
) -> TensorIterator | None:
    # [FUNCTION DOC] HELPER: Builds the reduction TensorIterator over (out, self) and returns it only if
    # nonempty, 1D or 2D, with unit element stride on the input's fastest (reduced) axis and, for 2D, a
    # larger outer stride; returns None otherwise or if iterator construction raises a RuntimeError.
    """Return the reduction ``TensorIterator`` if ``(self, d, out)`` fits the
    kernel's expected geometry, else ``None``. ``d`` is already normalized
    and ``out`` is a keepdim-shaped output (broadcast-aligned with ``self``).

    Mirrors ``try_inner_tree_reduction``: build the reduction iterator, then
    require a single coalesced reduced (fastest) dimension with element
    stride 1 on the input, and at most one non-reduced (outer) dimension so
    the operands canonicalize to a single ``(M, N)`` view.
    """
    try:
        it = reduce_op(out, self)
    except RuntimeError:
        return None

    if it.numel == 0:
        return None
    if it.ndim == 0 or it.ndim > 2:
        return None

    input_index = it.ntensors - 1  # operands are (out, self); input is last
    es_in = it.element_strides(input_index)
    # Input must be contiguous on the reduced (fastest, dim-0) axis.
    if es_in[0] != 1:
        return None
    # Reduction must be on the fastest dim: either the whole tensor is the
    # reduction (ndim == 1) or the reduced-dim stride is the smallest.
    if it.ndim == 2 and not (es_in[0] < es_in[1]):
        return None
    return it


def _geometry(self: torch.Tensor, out: torch.Tensor, it: TensorIterator):
    # [FUNCTION DOC] HELPER: Derives (M, N, in_row_stride, out_row_stride) in element units from the iterator:
    # M output rows, N reduced elements per row; for 1D the strides default to N and 1.
    """Recover ``(M, N, in_row_stride, out_row_stride)`` (element units) from
    the coalesced reduction iterator. ``M`` output rows, each a contiguous
    ``N``-element reduction; ``in_row_stride`` / ``out_row_stride`` step
    between rows of the canonical ``(M, N)`` / ``(M,)`` views."""
    m = out.numel()
    n = self.numel() // m if m else 0
    if it.ndim == 2:
        in_row_stride = it.element_strides(it.ntensors - 1)[1]
        out_row_stride = it.element_strides(0)[1]
    else:
        in_row_stride = n
        out_row_stride = 1
    return m, n, in_row_stride, out_row_stride


def _base_cond_ok(self: torch.Tensor, dim, dtype) -> int | None:
    # [FUNCTION DOC] HELPER: Shared front gate; returns the normalized reduced dim when the feature flag is on,
    # tensor is a real CUDA tensor of a supported dtype, no dtype override is given, exactly one in-range
    # dim is reduced and int32 indexing is safe; otherwise None.
    """Shared front gate. Returns the normalized reduced dim if the call is a
    candidate, else ``None``."""
    if not _inner_tree_enabled():
        return None
    if not self.is_cuda or is_fake_tensor(self):
        return None
    # dtype-casting sum (an explicit out/result dtype) is out of scope.
    if dtype is not None:
        return None
    if self.dtype not in _SUPPORTED_DTYPES:
        return None
    d = _single_dim(dim, self.ndim)
    if d is None or not (0 <= d < self.ndim):
        return None
    if not _int32_indexing_safe(self, d):
        return None
    return d


def _cond(self, dim, keepdim=False, *, dtype=None) -> bool:
    # [FUNCTION DOC] HELPER: Dispatcher condition for sum.dim_IntList: passes the front gate and the geometry
    # check using a scratch empty keepdim output. Returns bool.
    d = _base_cond_ok(self, dim, dtype)
    if d is None:
        return False
    out = torch.empty(_keepdim_out_shape(self, d), dtype=self.dtype, device=self.device)
    return _eligibility(self, d, out) is not None


def _out_cond(self, dim, keepdim=False, *, dtype=None, out) -> bool:
    # [FUNCTION DOC] HELPER: Dispatcher condition for sum.IntList_out: like _cond but also requires out to be a
    # real CUDA tensor of self's dtype and the expected reduced shape. Returns bool.
    d = _base_cond_ok(self, dim, dtype)
    if d is None:
        return False
    if out.dtype != self.dtype or not out.is_cuda or is_fake_tensor(out):
        return False
    # The kernel writes ``out`` directly via a keepdim-aligned view; ``out``
    # must already have the reduced shape. Reject mis-shaped ``out`` and let
    # aten produce the proper error.
    out_kd = _out_keepdim_view(self, d, keepdim, out)
    if out_kd is None:
        return False
    return _eligibility(self, d, out_kd) is not None


def _kernel():
    # [FUNCTION DOC] HELPER: Lazily imports and returns inner_tree_sum_into from inner_tree_kernel.
    from .inner_tree_kernel import inner_tree_sum_into

    return inner_tree_sum_into


def _run(self: torch.Tensor, d: int, out_kd: torch.Tensor) -> None:
    # [FUNCTION DOC] HELPER: Launches the kernel to reduce self along d into keepdim-shaped out_kd, writing
    # in place via strided 2D/1D views. Returns early for empty problems. Raises RuntimeError if the
    # iterator rebuild fails (cond/impl disagreement).
    """Run the kernel writing the reduction of ``self`` along ``d`` into the
    keepdim-shaped ``out_kd``. Caller has validated eligibility."""
    it = _eligibility(self, d, out_kd)
    if it is None:
        raise RuntimeError("sum cutedsl: cond approved but iter rebuild failed")
    m, n, in_rs, out_rs = _geometry(self, out_kd, it)
    if m == 0 or n == 0:
        return
    in_2d = self.as_strided((m, n), (in_rs, 1))
    out_1d = out_kd.as_strided((m,), (out_rs,))
    _kernel()(out_1d, in_2d)


def _impl(self, dim, keepdim=False, *, dtype=None):
    # [FUNCTION DOC] HELPER: Implementation of sum.dim_IntList: allocates a keepdim output, runs the kernel
    # and returns it (squeezed over the reduced dim unless keepdim). The dtype argument is unused.
    d = _normalize_dim(dim[0] if not isinstance(dim, int) else dim, self.ndim)
    out_kd = torch.empty(
        _keepdim_out_shape(self, d), dtype=self.dtype, device=self.device
    )
    _run(self, d, out_kd)
    return out_kd if keepdim else out_kd.squeeze(d)


def _out_impl(self, dim, keepdim=False, *, dtype=None, out):
    # [FUNCTION DOC] HELPER: Implementation of sum.IntList_out: reduces into a keepdim view of out and returns out.
    # KNOWN ISSUE: relies on an assert (with a noqa) that the view is not None, stripped under python -O.
    d = _normalize_dim(dim[0] if not isinstance(dim, int) else dim, self.ndim)
    out_kd = _out_keepdim_view(self, d, keepdim, out)
    assert out_kd is not None  # noqa: S101  # _out_cond guarantees non-None
    _run(self, d, out_kd)
    return out


def register_to_dispatch() -> None:
    # [FUNCTION DOC] Registers the two CUDA overrides (sum.dim_IntList and sum.IntList_out) with their
    # conds and impls through cutedsl_utils.register_op_override.
    cu.register_op_override("aten", "sum.dim_IntList", "CUDA", cond=_cond, impl=_impl)
    cu.register_op_override(
        "aten", "sum.IntList_out", "CUDA", cond=_out_cond, impl=_out_impl
    )
