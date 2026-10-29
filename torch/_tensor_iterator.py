# [MODULE DOC] Python access to ATen's TensorIterator build pipeline. Defines TensorIterator (a read-only view of a built iterator)
# and factory shortcuts mirroring the C++ named constructors (binary, unary, comparison, nullary, reduce).
# Operands are registered in a fixed canonical order (outputs, inputs, const inputs). Imports the C extension types;
# no other import-time side effects.
"""Python access to ATen's TensorIterator build pipeline.

:class:`TensorIterator` builds an iterator from a set of operands and flags
that mirror ``at::TensorIteratorConfig``, then exposes the post-build
shape / dtype / device / stride information for inspection.

This is a build-pipeline-only surface: there is no ``for_each`` here. Use it
to debug shape and dtype inference, validate custom-op contracts, or inspect
how ATen would lay out a kernel's iteration.

Construction is canonical, *not* a faithful replay of arbitrary
``TensorIteratorConfig`` call sequences. Operands are always registered in
the order ``outputs -> inputs -> const_inputs`` and setters are applied in
one fixed order. Notably:

* The C++ builder distinguishes ``add_input(a); add_const_input(b)`` from
  ``add_const_input(b); add_input(a)`` -- ``input(0)`` refers to different
  operands. This Python surface cannot express that distinction: every
  ``inputs[i]`` precedes every ``const_inputs[j]``.
* Some C++ setters have order-dependent side effects (e.g.
  ``promote_inputs_to_common_dtype(true)`` also flips
  ``check_all_same_dtype`` to ``false``). The Python surface materializes
  the *final* boolean state of each knob, not the call order, so it can't
  reproduce a sequence where an intermediate setter observed a
  since-overwritten value.

Every in-tree caller of ``at::TensorIteratorConfig`` fits the canonical-
recipe shape; the lossiness is theoretical, not practical.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._C import (
    _TensorIterator as _CTensorIterator,
    _TensorIteratorSpec as _CTensorIteratorSpec,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = [
    "TensorIterator",
    "binary_op",
    "binary_float_op",
    "comparison_op",
    "unary_op",
    "unary_float_op",
    "nullary_op",
    "reduce_op",
]


class TensorIterator:
    # [CLASS DOC] Wraps a built C++ TensorIterator held in the private _impl attribute. Constructor keyword arguments mirror
    # the TensorIteratorConfig knobs; building happens once in the constructor and the object is then read-only.
    # Exposes shape, dtype, device, stride and operand-count information after coalescing and reordering.
    """A built TensorIterator. Read-only view of the build-pipeline result.

    Constructor kwargs mirror ``at::TensorIteratorConfig``; defaults match
    the C++ defaults. See module docstring for canonical-recipe caveats.

    Examples
    --------
    Build a TI for a binary op with type promotion and inspect the result::

        >>> import torch
        >>> from torch._tensor_iterator import TensorIterator
        >>> a = torch.zeros(3, 4, dtype=torch.float32)
        >>> b = torch.zeros(3, 4, dtype=torch.float64)
        >>> it = TensorIterator(
        ...     outputs=[None],
        ...     const_inputs=[a, b],
        ...     promote_inputs_to_common_dtype=True,
        ...     cast_common_dtype_to_outputs=True,
        ... )
        >>> it.common_dtype
        torch.float64
        >>> it.numel
        12
        >>> it.ndim       # contiguous (3, 4) collapses to a single dim
        1

    Declare a static shape (the iterator skips broadcast/coalesce)::

        >>> a = torch.zeros(2, 6)
        >>> out = torch.empty(2, 6)
        >>> it = TensorIterator(
        ...     outputs=[out],
        ...     const_inputs=[a],
        ...     resize_outputs=False,
        ...     static_shape=(2, 6),
        ... )

    Build a reduction TI (output must be pre-allocated to the reduced shape)::

        >>> a = torch.zeros(3, 4)
        >>> out = torch.empty(3, 1)
        >>> it = TensorIterator(
        ...     outputs=[out],
        ...     const_inputs=[a],
        ...     resize_outputs=False,
        ...     is_reduction=True,
        ... )

    Inspect post-coalesce strides (in bytes; use ``element_strides`` for
    element units)::

        >>> a = torch.zeros(3, 4)
        >>> b = torch.zeros(3, 4)
        >>> it = TensorIterator(outputs=[None], const_inputs=[a, b])
        >>> tuple(it.strides(0))           # output byte strides
        (4,)
        >>> it.element_strides(0)          # ... in elements
        (1,)
    """

    def __init__(
        self,
        *,
        outputs: list[torch.Tensor | None] | None = None,
        inputs: list[torch.Tensor] | None = None,
        const_inputs: list[torch.Tensor] | None = None,
        check_all_same_dtype: bool = True,
        check_all_same_device: bool = True,
        promote_inputs_to_common_dtype: bool = False,
        promote_integer_inputs_to_float: bool = False,
        cast_common_dtype_to_outputs: bool = False,
        enforce_safe_casting_to_output: bool = False,
        enforce_linear_iteration: bool = False,
        resize_outputs: bool = True,
        check_mem_overlap: bool = True,
        allow_cpu_scalars: bool = False,
        is_reduction: bool = False,
        static_dtype: torch.dtype | None = None,
        static_device: torch.device | None = None,
        static_shape: Sequence[int] | None = None,
        squash_dims: Sequence[int] = (),
    ) -> None:
        # [FUNCTION DOC] Builds a spec from the operand lists and boolean flags, optionally sets static dtype, device and shape
        # (squash_dims is applied only when a static shape is given), then builds the native iterator and stores it.
        # None lists become empty lists; an output of None asks the iterator to allocate it. Errors from the build
        # (shape, dtype, device or overlap checks) propagate from the C extension.
        spec = _CTensorIteratorSpec()
        spec.outputs = list(outputs) if outputs is not None else []
        spec.inputs = list(inputs) if inputs is not None else []
        spec.const_inputs = list(const_inputs) if const_inputs is not None else []

        spec.check_all_same_dtype = check_all_same_dtype
        spec.check_all_same_device = check_all_same_device
        spec.promote_inputs_to_common_dtype = promote_inputs_to_common_dtype
        spec.promote_integer_inputs_to_float = promote_integer_inputs_to_float
        spec.cast_common_dtype_to_outputs = cast_common_dtype_to_outputs
        spec.enforce_safe_casting_to_output = enforce_safe_casting_to_output
        spec.enforce_linear_iteration = enforce_linear_iteration
        spec.resize_outputs = resize_outputs
        spec.check_mem_overlap = check_mem_overlap
        spec.allow_cpu_scalars = allow_cpu_scalars
        spec.is_reduction = is_reduction

        if static_dtype is not None:
            spec.static_dtype = static_dtype
        if static_device is not None:
            spec.static_device = static_device
        if static_shape is not None:
            spec.static_shape = list(static_shape)
            spec.squash_dims = list(squash_dims)

        self._impl: _CTensorIterator = spec.build()

    @property
    def ndim(self) -> int:
        # [FUNCTION DOC] Number of iteration dimensions after coalescing.
        return self._impl.ndim

    @property
    def shape(self) -> memoryview:
        """Iterator shape (post coalesce/reorder), as a zero-copy
        ``memoryview`` of ``int64`` elements. The view holds a reference
        to this iterator and keeps it alive for as long as the view is
        reachable; copy via ``tuple(it.shape)`` if you need a snapshot
        you can outlive the iterator with. Hot-path readers (dispatch
        conditionals) get index access without an allocation."""
        # [FUNCTION DOC] Returns the post-coalesce iteration shape as a zero-copy int64 memoryview tied to this iterator's lifetime.
        return self._impl.shape

    @property
    def numel(self) -> int:
        # [FUNCTION DOC] Total number of elements iterated.
        return self._impl.numel

    @property
    def ntensors(self) -> int:
        # [FUNCTION DOC] Total number of operands (outputs plus inputs).
        return self._impl.ntensors

    @property
    def ninputs(self) -> int:
        # [FUNCTION DOC] Number of input operands.
        return self._impl.ninputs

    @property
    def noutputs(self) -> int:
        # [FUNCTION DOC] Number of output operands.
        return self._impl.noutputs

    @property
    def is_contiguous(self) -> bool:
        # [FUNCTION DOC] True if all operands are contiguous in the iteration layout.
        return self._impl.is_contiguous

    @property
    def is_trivial_1d(self) -> bool:
        # [FUNCTION DOC] True if the iteration is trivially one-dimensional.
        return self._impl.is_trivial_1d

    @property
    def common_dtype(self) -> torch.dtype | None:
        # [FUNCTION DOC] Returns the inferred common computation dtype, or None if inference produced none.
        """The inferred computation dtype, or ``None`` if no single dtype
        was inferred. Populated whenever TensorIterator can resolve a
        common dtype -- under promotion flags, or when every input
        already shares a dtype. ``None`` does not mean "promotion was
        not requested"; it means inference produced no answer."""
        return self._impl.common_dtype

    def tensor(self, index: int) -> torch.Tensor:
        # [FUNCTION DOC] Returns the operand tensor at the given index (outputs first, then inputs); may be a promoted or cast temporary.
        """Return the iterator's current operand at ``index``.

        Note: under ``promote_inputs_to_common_dtype`` /
        ``cast_common_dtype_to_outputs`` (CPU paths), this may be a
        promoted/cast kernel temporary rather than the tensor that was
        registered with the config -- it's the iterator's view of the
        operand a kernel would actually iterate over."""
        return self._impl.tensor(index)

    def input(self, index: int = 0) -> torch.Tensor:
        # [FUNCTION DOC] Returns the input operand at the given index (default 0).
        """Return the iterator's current input. See :meth:`tensor` for the
        promoted-temporary caveat."""
        return self._impl.input(index)

    def output(self, index: int = 0) -> torch.Tensor:
        # [FUNCTION DOC] Returns the output operand at the given index (default 0).
        """Return the iterator's current output. See :meth:`tensor` for the
        cast-temporary caveat."""
        return self._impl.output(index)

    def dtype(self, index: int = 0) -> torch.dtype:
        # [FUNCTION DOC] Returns the dtype of the operand at the given index (default 0).
        return self._impl.dtype(index)

    def device(self, index: int = 0) -> torch.device:
        # [FUNCTION DOC] Returns the device of the operand at the given index (default 0).
        return self._impl.device(index)

    def strides(self, index: int) -> memoryview:
        # [FUNCTION DOC] Returns the operand's byte strides as a zero-copy int64 memoryview.
        """Per-operand strides in bytes (post reorder/coalesce), as a
        zero-copy ``memoryview`` of ``int64`` elements. The view holds a
        reference to this iterator and keeps it alive for as long as the
        view is reachable; copy via ``tuple(it.strides(i))`` if you need
        a snapshot you can outlive the iterator with."""
        return self._impl.strides(index)

    def element_strides(self, index: int) -> tuple[int, ...]:
        # [FUNCTION DOC] Returns the operand's strides in element units as a freshly allocated tuple (allocates on every call).
        """Per-operand strides in elements (byte stride / element size).

        Allocates a fresh tuple on every call. Don't use on a hot path:
        cache the result, or read :meth:`strides` once and divide by
        :meth:`element_size` of the operand inline."""
        return self._impl.element_strides(index)

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns the repr of the underlying native iterator.
        return repr(self._impl)


# --- Factory shortcuts. These mirror the C++ named constructors at
# aten/src/ATen/TensorIterator.cpp:1069+. Pass ``out=None`` to ask the iterator
# to allocate a fresh output tensor of the inferred shape/dtype/device.


def binary_op(
    out: torch.Tensor | None, a: torch.Tensor, b: torch.Tensor
) -> TensorIterator:
    # [FUNCTION DOC] Builds an iterator for a binary op: CPU scalars allowed, inputs promoted to a common dtype, result cast to
    # the output dtype with safe-casting enforced. out may be None to allocate.
    """Equivalent of ``at::TensorIterator::binary_op``."""
    return TensorIterator(
        outputs=[out],
        const_inputs=[a, b],
        allow_cpu_scalars=True,
        promote_inputs_to_common_dtype=True,
        cast_common_dtype_to_outputs=True,
        enforce_safe_casting_to_output=True,
    )


def binary_float_op(
    out: torch.Tensor | None, a: torch.Tensor, b: torch.Tensor
) -> TensorIterator:
    # [FUNCTION DOC] Like binary_op but integer inputs are also promoted to floating point.
    """Equivalent of ``at::TensorIterator::binary_float_op``."""
    return TensorIterator(
        outputs=[out],
        const_inputs=[a, b],
        allow_cpu_scalars=True,
        promote_inputs_to_common_dtype=True,
        cast_common_dtype_to_outputs=True,
        enforce_safe_casting_to_output=True,
        promote_integer_inputs_to_float=True,
    )


def comparison_op(
    out: torch.Tensor | None, a: torch.Tensor, b: torch.Tensor
) -> TensorIterator:
    # [FUNCTION DOC] Builds an iterator for a comparison op. If out is None the output dtype is forced to bool; otherwise a cast
    # back to out's dtype is requested unless out is already bool.
    """Equivalent of ``at::TensorIterator::comparison_op``.

    When ``out`` is ``None``, the output dtype is forced to bool. When ``out``
    is a defined non-bool tensor, the common dtype is cast back to its dtype
    via ``cast_common_dtype_to_outputs``. The bool-output case skips that cast
    as a performance optimization.
    """
    static_dtype = torch.bool if out is None else None
    cast_to_outputs = out is not None and out.dtype != torch.bool
    return TensorIterator(
        outputs=[out],
        const_inputs=[a, b],
        allow_cpu_scalars=True,
        promote_inputs_to_common_dtype=True,
        cast_common_dtype_to_outputs=cast_to_outputs,
        static_dtype=static_dtype,
    )


def unary_op(out: torch.Tensor | None, a: torch.Tensor) -> TensorIterator:
    # [FUNCTION DOC] Builds a plain single-input iterator with default flags. out may be None to allocate.
    """Equivalent of ``at::TensorIterator::unary_op``."""
    return TensorIterator(outputs=[out], const_inputs=[a])


def unary_float_op(out: torch.Tensor | None, a: torch.Tensor) -> TensorIterator:
    # [FUNCTION DOC] Unary iterator with promotion to a common dtype, integer-to-float promotion, output cast and safe casting.
    """Equivalent of ``at::TensorIterator::unary_float_op``."""
    return TensorIterator(
        outputs=[out],
        const_inputs=[a],
        promote_inputs_to_common_dtype=True,
        cast_common_dtype_to_outputs=True,
        enforce_safe_casting_to_output=True,
        promote_integer_inputs_to_float=True,
    )


def nullary_op(out: torch.Tensor) -> TensorIterator:
    # [FUNCTION DOC] Builds an iterator with only an output (no dtype check, no resize). Raises TypeError if out is None.
    """Equivalent of ``at::TensorIterator::nullary_op``.

    Unlike the binary/unary factories, ``out`` must be a defined tensor;
    the C++ named constructor takes a non-undefined output.
    """
    if out is None:
        raise TypeError(
            "nullary_op requires a defined output tensor; None is not allowed."
        )
    return TensorIterator(
        outputs=[out],
        check_all_same_dtype=False,
        resize_outputs=False,
    )


def reduce_op(
    out: torch.Tensor,
    a: torch.Tensor,
    *,
    out2: torch.Tensor | None = None,
) -> TensorIterator:
    # [FUNCTION DOC] Builds a reduction iterator; outputs must be pre-allocated (no resize or allocation). With out2, validates that
    # out, out2 and a share a device and that out and out2 have identical dim, sizes and strides, raising RuntimeError otherwise;
    # that path skips dtype promotion. Without out2, inputs are promoted to a common dtype.
    """Equivalent of ``at::TensorIterator::reduce_op``.

    Pass ``out2`` for the two-output reduction overload (e.g. ``min`` returning
    values + indices). The output tensor(s) must be pre-allocated and shaped
    correctly: this factory does not allocate or resize. With ``out2``, both
    outputs must live on ``a``'s device and share its sizes/strides -- the
    C++ named constructor asserts this and the same checks are mirrored
    here.
    """
    # Mirror the TORCH_INTERNAL_ASSERTs in TensorIterator::reduce_op. We
    # surface them as RuntimeError up front since the binding doesn't
    # rebind the named constructor itself; without this, mismatched
    # out1/out2 would build a kernel-shape that doesn't match the
    # tensors and corrupt memory once a kernel is invoked.
    if out2 is not None:
        if not out.device == a.device == out2.device:
            raise RuntimeError(
                "reduce_op: out, out2, and the input must share a device, "
                f"got out={out.device}, out2={out2.device}, a={a.device}"
            )
        if out.dim() != out2.dim():
            raise RuntimeError(
                f"reduce_op: out and out2 must have the same dim, got "
                f"{out.dim()} and {out2.dim()}"
            )
        if out.shape != out2.shape or out.stride() != out2.stride():
            raise RuntimeError(
                "reduce_op: out and out2 must have identical sizes and strides"
            )
        return TensorIterator(
            outputs=[out, out2],
            const_inputs=[a],
            check_all_same_dtype=False,
            resize_outputs=False,
            check_mem_overlap=False,
            is_reduction=True,
        )
    return TensorIterator(
        outputs=[out],
        const_inputs=[a],
        promote_inputs_to_common_dtype=True,
        resize_outputs=False,
        check_mem_overlap=False,
        is_reduction=True,
    )
