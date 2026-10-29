# [MODULE DOC] Symbolic block-shape propagation for Inductor ops handlers. Provides broadcasting of
# shapes (tuples of ints/strings such as XBLOCK) across op arguments and an ops handler that
# computes the output shape of each op from its inputs. Import-time side effects: none.
import functools
from collections.abc import Callable, Sequence
from typing import Protocol

import sympy

import torch

from .virtualized import OpsValue, V


BlockShapeType = Sequence[int | str] | None


class ShapeVar(Protocol):
    # [CLASS DOC] Protocol for any object exposing a read-only shape property (a sequence of int or str
    # dims, or None when unknown). The property itself is a one-line stub and is not documented.
    @property
    def shape(self) -> BlockShapeType: ...


ShapeArg = ShapeVar | torch.types.Number | str | OpsValue | torch.dtype

# Inputs need to be cacheable (e.g., not a CSEVar) in order for the cache to be effective
# So first decompose CSEVars -> tuple before calling this


def get_broadcasted_shape(a: BlockShapeType, b: BlockShapeType) -> BlockShapeType:
    # [FUNCTION DOC] Broadcast two block shapes. Both must be sequences (otherwise raises an assertion
    # error); converts them to tuples and delegates to the cached helper. Returns the broadcast shape.
    if not isinstance(a, Sequence):
        raise AssertionError(f"expected a to be a Sequence, got {type(a)}")
    if not isinstance(b, Sequence):
        raise AssertionError(f"expected b to be a Sequence, got {type(b)}")
    return _get_broadcasted_shape(tuple(a), tuple(b))


@functools.lru_cache(None)
def _get_broadcasted_shape(
    a: tuple[int | str, ...], b: tuple[int | str, ...]
) -> BlockShapeType:
    # [FUNCTION DOC] HELPER: Cached shape broadcast on tuples. Left-pads the shorter shape with 1s to equal
    # rank, then merges dims pairwise. Returns the broadcast tuple. Raises an assertion error if two
    # non-1 dims differ (compared by string form).
    if len(a) > len(b):
        return _get_broadcasted_shape(a, (*[1] * (len(a) - len(b)), *b))
    elif len(a) < len(b):
        b, a = a, b
        return _get_broadcasted_shape(a, (*[1] * (len(a) - len(b)), *b))
    else:

        def _get_broadcasted_dim(d1: int | str, d2: int | str) -> int | str:
        # [FUNCTION DOC] HELPER: Merge two dims: a dim equal to 1 (by string form) yields the other; otherwise
        # both must have the same string form, else an assertion error is raised.
            if str(d1) == "1":
                return d2
            elif str(d2) == "1":
                return d1
            if str(d1) != str(d2):
                raise AssertionError(f"expected str(d1) == str(d2), got {d1} != {d2}")
            return d1

        return tuple(_get_broadcasted_dim(d1, d2) for d1, d2 in zip(a, b))


def broadcast_shapes_for_args(args: Sequence[ShapeArg]) -> BlockShapeType:
    # [FUNCTION DOC] Compute the broadcast shape of a list of op arguments. Args with a shape attribute
    # contribute it (a None shape makes the whole result None); plain numbers contribute a scalar
    # shape (empty tuple) if nothing else is known; dtypes are skipped; LoopBody, LoopBodyBlock and
    # OpsValue args yield None (marked TODO upstream); any other type raises TypeError. Returns a
    # tuple shape or None.
    result_shape: BlockShapeType = None

    for arg in args:
        if hasattr(arg, "shape"):
            shape = arg.shape
            if shape is None:
                return None
            elif result_shape is None:
                result_shape = tuple(shape)
            else:
                result_shape = get_broadcasted_shape(result_shape, tuple(shape))
        elif isinstance(arg, (int, float)):
            if result_shape is None:
                result_shape = ()
        elif isinstance(arg, torch.dtype):
            continue
        else:
            from torch._inductor.loop_body import LoopBody, LoopBodyBlock

            if isinstance(arg, (LoopBodyBlock, LoopBody, OpsValue)):
                # TODO: fix me
                return None
            raise TypeError(f"Unknown type: {type(arg)}")

    return result_shape


class ShapePropagationOpsHandler:
    # [CLASS DOC] Ops handler that, instead of generating code, returns the output block shape of each op
    # given the shapes of its arguments. Most ops are handled by the dynamic attribute fallback which
    # broadcasts all argument shapes; explicit static methods cover ops with special shape rules
    # (constant, store, reduction, dot, to_dtype, index_expr, value_expr, load_seed, indirect_indexing,
    # device_assert_async). Stateless, but constant and dot consult the current kernel.
    """
    Propagate shape from args to output
    """

    @staticmethod
    def constant(value: torch.types.Number, dtype: torch.dtype) -> BlockShapeType:
        # [FUNCTION DOC] Shape of a constant: with a Triton kernel active, a tuple of ones with the kernel's
        # tensor rank; otherwise the empty (scalar) shape.
        # TritonKernelOverrides.constant uses tl.full with shape=[1]*ndim for all types
        from torch._inductor.codegen.triton import TritonKernel

        if isinstance(V.kernel, TritonKernel):
            ndim = V.kernel.triton_tensor_ndim()
            return tuple([1] * ndim)
        else:
            return ()

    @staticmethod
    def store_reduction(name: str, index: int, value: ShapeArg) -> None:
        # [FUNCTION DOC] Store-reduction produces no value; returns None.
        return None

    @staticmethod
    def reduction(
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: str,
        value: ShapeArg | tuple[ShapeArg, ...],
    ) -> BlockShapeType | tuple[BlockShapeType, ...]:
        # [FUNCTION DOC] Reduction shape propagation is not implemented; always raises NotImplementedError.
        raise NotImplementedError

    @staticmethod
    def store(name: str, index: int, value: ShapeArg, mode: str | None = None) -> None:
        # [FUNCTION DOC] Store produces no value; returns None.
        return None

    @staticmethod
    def to_dtype(
        value: ShapeVar,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> BlockShapeType:
        # [FUNCTION DOC] dtype conversion preserves shape; returns the shape of the input value.
        return value.shape

    @staticmethod
    def dot(a: sympy.Expr, b: sympy.Expr) -> BlockShapeType:
        # [FUNCTION DOC] Shape of a dot product. Only valid under a Triton kernel (raises an assertion error
        # otherwise); always returns the 2D block shape of YBLOCK by XBLOCK.
        from torch._inductor.codegen.triton import TritonKernel

        if not isinstance(V.kernel, TritonKernel):
            raise AssertionError("dot supports Triton only")
        return ("YBLOCK", "XBLOCK")

    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> BlockShapeType:
        # [FUNCTION DOC] Index expressions embed their shape in the sympy expression; returns None (unknown).
        # shape is implicitly embedded in expr.
        return None

    @staticmethod
    def value_expr(expr: sympy.Expr, dtype: torch.dtype) -> BlockShapeType:
        # [FUNCTION DOC] Value expressions embed their shape in the sympy expression; returns None (unknown).
        # shape is implicitly embedded in expr.
        return None

    @staticmethod
    def load_seed(name: str, offset: int) -> BlockShapeType:
        # [FUNCTION DOC] Seed loads yield a scalar; returns the empty shape.
        return ()

    @staticmethod
    def indirect_indexing(
        var: ShapeArg,
        size: sympy.Expr | int,
        check: bool = True,
        wrap_neg: bool = True,
    ) -> None:
        # [FUNCTION DOC] Indirect indexing produces no shape; returns None.
        return None

    def __getattr__(self, name: str) -> Callable[..., BlockShapeType]:
    # [FUNCTION DOC] Fallback for any other op name: returns a callable that broadcasts the shapes of all
    # positional args (keyword args are ignored) and returns the resulting shape.
        return lambda *args, **kwargs: broadcast_shapes_for_args(args)

    @staticmethod
    def device_assert_async(cond: ShapeArg, msg: str) -> None:
        # [FUNCTION DOC] Async device assert produces no value; returns None.
        return None
