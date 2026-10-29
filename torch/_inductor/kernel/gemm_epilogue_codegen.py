# mypy: allow-untyped-defs
# [MODULE DOC] Minimal stand-ins for the kernel, body and CSE objects that Inductor's CuTeDSL op
# overrides expect, plus an op-override subclass that maps ATen-style op spellings (alpha
# arguments, _to_copy, clamp variants, scalar where) to CuTeDSL expressions, and a helper that
# builds the evaluation namespace for generated epilogue code. No import-time side effects.
"""Shared CuTeDSL emission primitives for GEMM epilogues."""

from typing import Any

import torch
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
    CuteDSLCSEVariable,
    CuteDSLOpOverrides,
)
from torch._inductor.virtualized import V
from torch.utils._sympy.value_ranges import ValueRanges


def gemm_epilogue_op_scope(cute: Any) -> dict[str, Any]:
    # [FUNCTION DOC] Build the mapping of names available to generated epilogue source: cutlass,
    # operator, the MLIR math dialect, the given cute module, and activation helpers (erf, exp,
    # gelu, relu, sigmoid, silu, tanh) implemented via cute math. Imports cutlass lazily, so it
    # fails with ImportError if cutlass is missing. Argument cute is the cutlass.cute module.
    import operator

    import cutlass
    from cutlass._mlir.dialects import math as mlir_math

    def sigmoid(x: Any) -> Any:
        # [FUNCTION DOC] HELPER: logistic function computed as one over one plus exp of negative x.
        return 1.0 / (1.0 + cute.math.exp(-x))

    return {
        "cutlass": cutlass,
        "operator": operator,
        "mlir_math": mlir_math,
        "cute": cute,
        "erf": cute.math.erf,
        "exp": cute.math.exp,
        "gelu": lambda x: 0.5 * x * (1.0 + cute.math.erf(x * 0.7071067811865476)),
        "relu": lambda x: cute.math.max(x, cute.full_like(x, 0.0)),
        "sigmoid": sigmoid,
        "silu": lambda x: x * sigmoid(x),
        "tanh": cute.math.tanh,
    }


class GemmEpilogueCuteDSLBody:
    # [CLASS DOC] HELPER: Minimal code buffer that collects generated source lines in the lines list.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Start with an empty list of lines.
        self.lines: list[str] = []

    def writeline(self, line: str) -> None:
        # [FUNCTION DOC] HELPER: Append one source line to the buffer.
        self.lines.append(line)


class GemmEpilogueCuteDSLCSE:
    # [CLASS DOC] HELPER: Trivial CSE substitute that does no deduplication; each generated
    # expression is assigned to a fresh temporary named tmp followed by a running counter.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initialize the temporary counter to zero.
        self.index = 0

    def generate(self, body, expr, *, bounds=None, dtype=None, shape=None):
        # [FUNCTION DOC] HELPER: Write an assignment of expr to a new temporary name into body and
        # return a CuteDSLCSEVariable for it, with unknown value ranges when bounds is None.
        # Side effects: increments the counter and appends a line to body.
        name = f"tmp{self.index}"
        self.index += 1
        body.writeline(f"{name} = {expr}")
        return CuteDSLCSEVariable(
            name,
            ValueRanges.unknown() if bounds is None else bounds,
            dtype=dtype,
            shape=shape,
        )


class GemmEpilogueCuteDSLKernel:
    # [CLASS DOC] HELPER: Fake kernel handler exposing the body (line buffer) and cse (temporary
    # generator) attributes that the CuTeDSL op overrides use through the virtualized kernel.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Create a fresh body buffer and CSE generator.
        self.body = GemmEpilogueCuteDSLBody()
        self.cse = GemmEpilogueCuteDSLCSE()


class GemmEpilogueCuteDSLOpOverrides(CuteDSLOpOverrides):
    # [CLASS DOC] Op handler for GEMM epilogues: all static methods that translate ATen-flavored
    # op names and keyword arguments (alpha, dtype, min/max) into the base CuteDSLOpOverrides
    # primitives. Stateless.
    """Normalize ATen spellings before emitting CuTeDSL expressions."""

    @staticmethod
    def add(a: Any, b: Any, *, alpha: Any = 1) -> Any:
        # [FUNCTION DOC] Add a and b, scaling b by alpha first unless alpha equals one.
        rhs = b if alpha == 1 else CuteDSLOpOverrides.mul(b, alpha)
        return CuteDSLOpOverrides.add(a, rhs)

    @staticmethod
    def sub(a: Any, b: Any, *, alpha: Any = 1) -> Any:
        # [FUNCTION DOC] Subtract b from a, scaling b by alpha first unless alpha equals one.
        rhs = b if alpha == 1 else CuteDSLOpOverrides.mul(b, alpha)
        return CuteDSLOpOverrides.sub(a, rhs)

    @staticmethod
    def _to_copy(x: Any, *, dtype: torch.dtype, **kwargs: Any) -> Any:
        # [FUNCTION DOC] HELPER: Cast x to dtype. Any extra keyword argument whose value is not None,
        # False or preserve_format raises NotImplementedError; otherwise defers to to_dtype.
        unsupported_kwargs = {
            key: value
            for key, value in kwargs.items()
            if value not in (None, False, torch.preserve_format)
        }
        if unsupported_kwargs:
            raise NotImplementedError(
                f"unsupported GEMM epilogue _to_copy kwargs: {unsupported_kwargs}"
            )
        return CuteDSLOpOverrides.to_dtype(x, dtype)

    @staticmethod
    def where(condition: Any, a: Any, b: Any) -> Any:
        # [FUNCTION DOC] Select a where condition holds else b. If any operand is tensor-like, use
        # the base elementwise where. For all-scalar operands, build a Python-style conditional
        # expression string; if none of the operands is a CSE variable return that string, else
        # store it in a new temporary (int32 dtype when none can be inferred), mark it as a
        # scalar expression and return it. Side effect: may emit a line into the active kernel.
        """Preserve all-scalar conditionals as scalar epilogue expressions."""
        if any(
            CuteDSLOpOverrides._is_tensor_like(value) for value in (condition, a, b)
        ):
            return CuteDSLOpOverrides.where(condition, a, b)
        result_expr = (
            f"({CuteDSLOpOverrides._as_expr(a)} if "
            f"{CuteDSLOpOverrides._as_expr(condition)} else "
            f"{CuteDSLOpOverrides._as_expr(b)})"
        )
        cse_vars = tuple(
            CuteDSLOpOverrides._get_cse_var(value) for value in (a, b, condition)
        )
        if all(value is None for value in cse_vars):
            return result_expr
        dtype, bounds = CuteDSLOpOverrides._extract_dtype_and_bounds(a, b, condition)
        result = V.kernel.cse.generate(
            V.kernel.body,
            result_expr,
            bounds=bounds,
            dtype=dtype if dtype is not None else torch.int32,
        )
        result.is_scalar_expr = True
        return result

    @staticmethod
    def clamp(x: Any, min: Any = None, max: Any = None) -> Any:
        # [FUNCTION DOC] Clamp x using maximum with min and minimum with max, skipping any bound
        # that is None.
        result = x
        if min is not None:
            result = CuteDSLOpOverrides.maximum(result, min)
        if max is not None:
            result = CuteDSLOpOverrides.minimum(result, max)
        return result

    @staticmethod
    def clamp_min(x: Any, min: Any) -> Any:
        # [FUNCTION DOC] Elementwise maximum of x and min.
        return CuteDSLOpOverrides.maximum(x, min)

    @staticmethod
    def clamp_max(x: Any, max: Any) -> Any:
        # [FUNCTION DOC] Elementwise minimum of x and max.
        return CuteDSLOpOverrides.minimum(x, max)

    @staticmethod
    def convert_element_type(x: Any, dtype: torch.dtype) -> Any:
        # [FUNCTION DOC] Alias of to_dtype: cast x to dtype.
        return CuteDSLOpOverrides.to_dtype(x, dtype)
