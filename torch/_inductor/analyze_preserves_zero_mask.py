# [MODULE DOC]
# Analyses over scheduler node bodies (prologue kernels) using custom ops handlers:
# (1) whether a masked-load prologue still stores zero when all loads are zero
# (prologue_preserves_zero_mask, via the PreservesZeros handler), and (2) whether a
# prologue can be generated without upcasting to fp32 while preserving numerics
# (can_codegen_without_upcasts, via the RecordLowPrecisionOps handler). No import-time
# side effects beyond imports.
import dataclasses
import itertools
from typing import Any, TYPE_CHECKING

import sympy

import torch
from torch._inductor import config
from torch._inductor.dtype_propagation import DtypePropagationOpsHandler
from torch._inductor.index_propagation import SymPyOps, TypedExpr

from .ops_handler import DefaultHandler
from .virtualized import StoreMode, V


if TYPE_CHECKING:
    from torch._inductor.scheduler import SchedulerNode


def construct_symbol(count: int, dtype: torch.dtype) -> sympy.Symbol:
    # [FUNCTION DOC] HELPER: Makes a fresh opaque symbol named unknown_<count>; the dtype argument is unused.
    return sympy.Symbol(f"unknown_{count}")


class PreservesZeros(SymPyOps, DefaultHandler):
    # [CLASS DOC] Ops handler that symbolically executes a prologue body with every load
    # replaced by a typed zero constant and unknown ops replaced by fresh symbols. State:
    # count (symbol counter), store_preserves_zeros (None until the single store is seen,
    # then True if the stored value is the constant zero), dtype_prop (dtype propagation).
    """
    For prologue kernels where the loads are masked, does the final store of this kernel preserve
    the zeros.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes the symbol counter, the unset store result and a dtype propagation handler.
        self.count = itertools.count(0)
        self.store_preserves_zeros: bool | None = None
        self.dtype_prop = DtypePropagationOpsHandler()

    def load(self, name: str, index: sympy.Expr) -> TypedExpr:
        # [FUNCTION DOC] Returns a zero constant (float zero for floating dtypes, else integer
        # zero) typed with the dtype propagated for the loaded buffer.
        # In prologue fusion, all loads get broadcasted
        dtype = self.dtype_prop.load(name, index)
        return TypedExpr(
            sympy.Float(0) if dtype.is_floating_point else sympy.Integer(0), dtype
        )

    def store(
        self, name: str, index: sympy.Expr, value: TypedExpr, mode: "StoreMode" = None
    ) -> None:
        # [FUNCTION DOC] Records whether the stored value is the constant zero. Raises
        # AssertionError if a store was already recorded (prologue must have one store).
        # KNOWN ISSUE: the initial isinstance check on self is vacuous.
        if not isinstance(self, PreservesZeros):
            raise AssertionError(f"expected PreservesZeros, got {type(self)}")
        # should only have a single store in prologue
        if self.store_preserves_zeros is not None:
            raise AssertionError("prologue should only have a single store")
        self.store_preserves_zeros = value.is_constant() and value.expr == 0

    def indirect_indexing(self, *args: Any, **kwargs: Any) -> sympy.Expr:
        # [FUNCTION DOC] Returns a fresh unknown symbol for any indirect index.
        return construct_symbol(next(self.count), torch.int32)

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # [FUNCTION DOC] HELPER: Fallback for ops without a specific handler: if the op has a
        # decomposition in OpDecompositions, evaluates it and returns its value; otherwise
        # returns a fresh unknown symbol typed with the dtype propagated for the op.
        from torch._inductor.codegen.common import OpDecompositions

        if hasattr(OpDecompositions, name):
            return getattr(OpDecompositions, name)(*args, **kwargs).value

        dtype = getattr(self.dtype_prop, name)(*args, **kwargs)
        return TypedExpr(construct_symbol(next(self.count), dtype), dtype)


def prologue_preserves_zero_mask(prologue: "SchedulerNode") -> bool:
    # [FUNCTION DOC]
    # Runs the prologue node body under a PreservesZeros handler (all loads are zero) and
    # returns True if its single store writes a constant zero. Raises AssertionError if
    # the body performed no store. Temporarily swaps the global ops handler.
    """
    Does this prologue preserve zero masks
    """
    preserves_zeros = PreservesZeros()
    with V.set_ops_handler(preserves_zeros):
        prologue._body(*prologue.get_ranges())

    store_preserves_zeros = preserves_zeros.store_preserves_zeros
    if not isinstance(store_preserves_zeros, bool):
        raise AssertionError(
            f"expected bool store_preserves_zeros, got {type(store_preserves_zeros)}"
        )

    return store_preserves_zeros


@dataclasses.dataclass
class DTypeContainer:
    # [CLASS DOC] HELPER: Value stand-in for dtype-only analysis: holds a dtype and whether the value is a scalar constant.
    dtype: torch.dtype
    is_scalar: bool = False


class RecordLowPrecisionOps(DefaultHandler):
    # [CLASS DOC] Ops handler that executes a body on dtype containers and sets the flag
    # low_precision_numeric_op when any numeric op involves a low-precision float (itemsize
    # below 4), or, if disallow_fp32_ops, produces a non-scalar fp32/fp64 result. Ops in
    # non_numeric_ops (dtype conversion, constant, where) never trigger the low-precision flag.
    def __init__(self, disallow_fp32_ops: bool = False) -> None:
        # [FUNCTION DOC] Stores the fp32 policy and initializes the flag, dtype propagation handler and non-numeric op names.
        self.disallow_fp32_ops = disallow_fp32_ops
        self.low_precision_numeric_op = False
        self.dtype_prop = DtypePropagationOpsHandler()
        self.non_numeric_ops = (
            "to_dtype",
            "constant",
            "where",
        )

    def load(self, name: str, index: sympy.Expr) -> DTypeContainer:
        # [FUNCTION DOC] Returns a container holding the propagated dtype of the loaded buffer.
        return DTypeContainer(self.dtype_prop.load(name, index))

    @staticmethod
    def store(
        name: str, index: sympy.Expr, value: TypedExpr, mode: "StoreMode" = None
    ) -> None:
        # [FUNCTION DOC] No-op: stores are ignored by this analysis.
        pass

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> None:
        # [FUNCTION DOC] No-op: bounds checks are ignored by this analysis.
        pass

    @staticmethod
    # pyrefly: ignore [bad-override]
    def indirect_indexing(*args: Any, **kwargs: Any) -> sympy.Expr:
        # [FUNCTION DOC] Returns the symbolic zero for any indirect index.
        return sympy.S.Zero

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # [FUNCTION DOC] HELPER: Fallback for every other op: propagates the output dtype and returns
        # it in a container (a constant yields a scalar fp32 container). Sets the
        # low_precision_numeric_op flag if the op is numeric and the output or any
        # container argument has a low-precision float dtype, or if fp32/fp64 non-scalar
        # outputs are disallowed and produced.
        out_dtype = getattr(self.dtype_prop, name)(*args, **kwargs)
        out = DTypeContainer(out_dtype, is_scalar=(name == "constant"))
        if name == "constant":
            return DTypeContainer(torch.float, is_scalar=True)

        uses_low_prec = any(
            isinstance(dtype_cont, DTypeContainer)
            and dtype_cont.dtype is not None
            and low_prec_float(dtype_cont.dtype)
            for dtype_cont in itertools.chain((out,), args, kwargs.values())
        )

        if uses_low_prec and name not in self.non_numeric_ops:
            self.low_precision_numeric_op = True

        if (
            self.disallow_fp32_ops
            and out.dtype in (torch.float32, torch.float64)
            and not out.is_scalar
        ):
            self.low_precision_numeric_op = True

        return out


def low_prec_float(dtype: torch.dtype) -> bool:
    # [FUNCTION DOC] Returns True if dtype is a floating type narrower than 4 bytes (fp16, bf16, fp8).
    return dtype.is_floating_point and dtype.itemsize < 4


def can_codegen_without_upcasts(
    prologue: "SchedulerNode",
    disallow_fp32_ops: bool = False,
) -> bool:
    # [FUNCTION DOC]
    # Returns True if the prologue can be generated without upcasting to fp32 while keeping
    # numerics: immediately True if all its operations are in the graph's low-precision
    # codegen op set; otherwise runs its body under RecordLowPrecisionOps with the
    # triton upcast config temporarily off and returns the negation of the recorded flag.
    # Side effects: temporarily patches config and swaps the global ops handler.
    """
    Can this prologue be run without `upcast_to_fp32` while preserving numerics.

    This is only true if the node only contains dtype conversions, indexing, and other non-arithmetic operators.

    If disallow_fp32_ops is True, then we also disallow ops that are explicitly computed in fp32 or fp64.
    """
    if prologue.get_operation_names() <= V.graph.low_precision_codegen_ops:
        return True

    low_prec_analysis = RecordLowPrecisionOps(disallow_fp32_ops)

    # Need to turn off upcasting to do analysis of whether we can turn it off
    with (
        config.patch("triton.codegen_upcast_to_fp32", False),
        V.set_ops_handler(low_prec_analysis),
    ):
        prologue._body(*prologue.get_ranges())

    return not low_prec_analysis.low_precision_numeric_op
