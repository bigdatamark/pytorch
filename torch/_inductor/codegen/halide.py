# mypy: allow-untyped-defs
# [MODULE DOC] Halide backend for Inductor codegen. Lowers fused Inductor
# scheduler nodes into Python source that builds a Halide pipeline (hl.Func,
# hl.Var, hl.RDom, ...), which is compiled via HalideCodeCache. Main contents:
# a sympy printer for index math (HalidePrinter), per-op string overrides
# (HalideOverrides), CSE variables tracking which dimensions they depend on,
# DimensionInfo for strided buffer access, HalideKernel (a SIMDKernel that
# rewrites indexing into Halide vars, emits loads/stores/reductions/scans and
# the final kernel source with scheduling), and HalideScheduling. Import-time
# side effects: creates module-level printers texpr/pexpr, a logger, and calls
# HalideOverrides._initialize_pointwise_overrides to register pointwise ops.
from __future__ import annotations

import dataclasses
import functools
import itertools
import logging
import re
import sys
from collections import defaultdict
from typing import Any, cast, TYPE_CHECKING

import sympy

import torch
import torch._logging

from ..._prims_common import is_integer_dtype
from ...utils._ordered_set import OrderedSet
from ...utils._sympy.functions import FloorDiv, Max, Min, ModularIndexing
from ...utils._sympy.symbol import symbol_is_type, SymT
from ...utils._sympy.value_ranges import ValueRanges
from .. import config, ir
from ..codecache import HalideCodeCache
from ..ir import get_reduction_combine_fn
from ..metrics import is_metric_table_enabled, log_kernel_metadata
from ..ops_handler import AddParenHandler
from ..runtime.hints import HalideInputSpec, HalideMeta
from ..utils import (
    get_bounds_index_expr,
    get_kernel_metadata,
    parallel_num_threads,
    sympy_index_symbol,
    sympy_subs,
)
from ..virtualized import _ops as ops, V
from .common import (
    BackendFeature,
    CSEVariable,
    DeferredLine,
    IndentedBuffer,
    KernelArgType,
    OpOverrides,
    PythonPrinter,
    SizeArg,
    TensorArg,
)
from .cpp import DTYPE_TO_CPP
from .cpp_utils import cexpr
from .simd import constant_repr, SIMDKernel, SIMDScheduling


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ..ops_handler import ReductionType, StoreMode
    from ..shape_propagation import BlockShapeType

log = logging.getLogger(__name__)


def halide_constant(val):
    # [FUNCTION DOC] Renders a Python scalar as Halide source text. Integers
    # outside the 32-bit range become 64-bit constants (min/max of int64 use
    # the type's min()/max() forms); floats become 64-bit float constants;
    # anything else (small ints, bools) uses its repr.
    if isinstance(val, int) and not (-2147483648 <= val <= 2147483647):
        info = torch.iinfo(torch.int64)
        if val == info.min:
            return "hl.Int(64).min()"
        if val == info.max:
            return "hl.Int(64).max()"
        return f"hl.i64({val!r})"
    if isinstance(val, float):
        return f"hl.f64({constant_repr(val)})"
    return repr(val)


class Unsupported(RuntimeError):
    # [CLASS DOC] Error raised when an operation has no Halide backend
    # implementation; message names the unsupported thing.
    def __init__(self, thing) -> None:
        # [FUNCTION DOC] Builds the error message naming the unsupported feature.
        super().__init__(f"halide backend does not support: {thing}")


class HalidePrinter(PythonPrinter):  # noqa: docstring_linter
    # [CLASS DOC] Sympy-to-source printer that renders index expressions as
    # Halide Python-binding expressions (hl.* calls). Stateless; relies on the
    # active kernel's index_dtype for integer casts. Unary math functions that
    # produce indices are wrapped in a cast to the index dtype.
    @staticmethod
    def cast_index(expr):
        # [FUNCTION DOC] Returns source text casting expr to the current kernel's index dtype.
        return f"hl.cast({V.kernel.index_dtype}, {expr})"

    @staticmethod
    def cast_float(expr):
        # [FUNCTION DOC] Returns source text casting expr to 32-bit float.
        return f"hl.cast(hl.Float(32), {expr})"

    def _print_Float(self, expr):
        # [FUNCTION DOC] HELPER: Prints a float literal as a 32-bit Halide float.
        return f"hl.f32({expr})"

    def _print_ToFloat(self, expr):
        # [FUNCTION DOC] HELPER: Prints a to-float conversion of its single argument as a 32-bit
        # float constructor. Raises AssertionError if not exactly one argument.
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"hl.f32({self._print(expr.args[0])})"

    def _print_floor(self, expr):
        # [FUNCTION DOC] HELPER: Prints floor (also used for FloorToInt) as a
        # floor call cast to the index dtype. AssertionError unless one argument.
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return self.cast_index(f"hl.floor({self._print(expr.args[0])})")

    _print_FloorToInt = _print_floor

    def _print_Trunc(self, expr):
        # [FUNCTION DOC] HELPER: Prints truncation (also TruncToInt) as a trunc
        # call cast to the index dtype. AssertionError unless one argument.
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return self.cast_index(f"hl.trunc({self._print(expr.args[0])})")

    _print_TruncToInt = _print_Trunc

    def _print_ceiling(self, expr):
        # [FUNCTION DOC] HELPER: Prints ceiling as a ceil call cast to the index
        # dtype. AssertionError unless one argument.
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return self.cast_index(f"hl.ceil({self._print(expr.args[0])})")

    def _helper_sqrt(self, expr):
        # [FUNCTION DOC] HELPER: Prints square root of the printed expr after casting it to 32-bit float.
        # pyrefly: ignore [missing-attribute]
        return f"hl.sqrt({self.cast_float(self._print(expr))})"

    def _print_Where(self, expr):
        # [FUNCTION DOC] HELPER: Prints a three-argument conditional as a Halide select.
        c =self.doprint(expr.args[0])
        p = self.doprint(expr.args[1])
        q = self.doprint(expr.args[2])
        return f"hl.select({c}, {p}, {q})"

    def _print_Min(self, expr):
        # [FUNCTION DOC] HELPER: Prints an n-ary minimum as nested binary min
        # calls, splitting the arguments in halves recursively; a single
        # argument prints as itself.
        if len(expr.args) == 1:
            # pyrefly: ignore [missing-attribute]
            return self._print(expr.args[0])

        mid = len(expr.args) // 2
        # pyrefly: ignore [missing-attribute]
        a = self._print(Min(*expr.args[:mid]))
        # pyrefly: ignore [missing-attribute]
        b = self._print(Min(*expr.args[mid:]))
        return f"hl.min({a}, {b})"

    def _print_Max(self, expr):
        # [FUNCTION DOC] HELPER: Prints an n-ary maximum as nested binary max
        # calls, splitting the arguments in halves recursively; a single
        # argument prints as itself.
        if len(expr.args) == 1:
            # pyrefly: ignore [missing-attribute]
            return self._print(expr.args[0])

        mid = len(expr.args) // 2
        # pyrefly: ignore [missing-attribute]
        a = self._print(Max(*expr.args[:mid]))
        # pyrefly: ignore [missing-attribute]
        b = self._print(Max(*expr.args[mid:]))

        return f"hl.max({a}, {b})"

    def _print_Abs(self, expr):
        # [FUNCTION DOC] HELPER: Prints absolute value as an abs call cast to the
        # index dtype. AssertionError unless one argument.
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return self.cast_index(f"hl.abs({self._print(expr.args[0])})")

    def _print_OpaqueUnaryFn_cos(self, expr):
        # [FUNCTION DOC] HELPER: Prints cosine of the single argument (AssertionError otherwise).
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"hl.cos({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_cosh(self, expr):
        # [FUNCTION DOC] HELPER: Prints hyperbolic cosine of the single argument (AssertionError otherwise).
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"hl.cosh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_acos(self, expr):
        # [FUNCTION DOC] HELPER: Prints arc cosine of the single argument (AssertionError otherwise).
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"hl.acos({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_sin(self, expr):
        # [FUNCTION DOC] HELPER: Prints sine of the single argument (AssertionError otherwise).
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"hl.sin({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_sinh(self, expr):
        # [FUNCTION DOC] HELPER: Prints hyperbolic sine of the single argument (AssertionError otherwise).
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"hl.sinh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_asin(self, expr):
        # [FUNCTION DOC] HELPER: Prints arc sine of the single argument (AssertionError otherwise).
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"hl.asin({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_tan(self, expr):
        # [FUNCTION DOC] HELPER: Prints tangent of the single argument (AssertionError otherwise).
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"hl.tan({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_tanh(self, expr):
        # [FUNCTION DOC] HELPER: Prints hyperbolic tangent of the single argument (AssertionError otherwise).
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"hl.tanh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_atan(self, expr):
        # [FUNCTION DOC] HELPER: Prints arc tangent of the single argument (AssertionError otherwise).
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"hl.atan({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_log2(self, expr):
        # [FUNCTION DOC] HELPER: Not supported; always raises NotImplementedError.
        raise NotImplementedError("log2")

    def _print_FloorDiv(self, expr):
        # [FUNCTION DOC] HELPER: Integer-valued floor division uses the base
        # printer. Otherwise both operands are cast to float, divided, floored,
        # and cast to the index dtype.
        if expr.is_integer:
            return super()._print_FloorDiv(expr)

        x, div = expr.args
        x = self.cast_float(self.doprint(x))
        div = self.cast_float(self.doprint(div))
        return self.cast_index(f"hl.floor({x} / {div})")

    def _print_Round(self, expr):
        # [FUNCTION DOC] HELPER: Prints rounding (also RoundToInt) as a round
        # call cast to the index dtype. AssertionError unless one argument.
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return self.cast_index(f"hl.round({self._print(expr.args[0])})")

    _print_RoundToInt = _print_Round

    def _print_IntTrueDiv(self, expr):
        # [FUNCTION DOC] HELPER: Prints integer true division, forcing a float
        # result by adding a float zero to the divisor.
        a, b = expr.args
        # force a cast to float
        return f"({a}) / ({b}+hl.f32(0))"

    def _print_RoundDecimal(self, expr):
        # [FUNCTION DOC] HELPER: Prints rounding to n decimal places: scale by
        # ten to the n, round, scale back (32-bit float constants). n must be
        # convertible to an integer.
        val, n = expr.args
        # pyrefly: ignore [missing-attribute]
        val = self._print(val)
        n = int(n)
        return f"hl.f32({10.0 ** (-n)!r})*hl.round(({val})*hl.f32({10.0**n!r}))"


texpr = HalidePrinter().doprint
pexpr = PythonPrinter().doprint


_halide_type = {
    torch.bool: "hl.Bool()",
    torch.bfloat16: "hl.BFloat(16)",
    torch.float16: "hl.Float(16)",
    torch.float32: "hl.Float(32)",
    torch.float64: "hl.Float(64)",
    torch.int8: "hl.Int(8)",
    torch.int16: "hl.Int(16)",
    torch.int32: "hl.Int(32)",
    torch.int64: "hl.Int(64)",
    torch.uint8: "hl.UInt(8)",
    torch.uint16: "hl.UInt(16)",
    torch.uint32: "hl.UInt(32)",
    torch.uint64: "hl.UInt(64)",
}


def halide_type(dtype):
    # [FUNCTION DOC] Maps a torch dtype to its Halide type constructor source
    # text. Raises KeyError for dtypes not in the table (e.g. complex).
    return _halide_type[dtype]


def halide_acc_type(dtype):
    # [FUNCTION DOC] Halide type text for accumulating in dtype: signed
    # integers narrower than 64 bits accumulate as 32-bit ints, half and
    # bfloat16 accumulate as 32-bit float, others keep their own type.
    if is_integer_dtype(dtype) and dtype.is_signed and dtype != torch.int64:
        dtype = torch.int32
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    return halide_type(dtype)


class HalideOverrides(OpOverrides):
    # [CLASS DOC] Op-handler overrides mapping Inductor pointwise ops to Halide
    # source-text expressions. Each static/class method receives operand
    # expressions (strings or CSE variables) and returns a source string; ops
    # Halide cannot express raise Unsupported or NotImplementedError. Remaining
    # pointwise ops are registered by an initializer call after the class body.
    @staticmethod
    def to_dtype(
        x,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types=True,
    ):
        # [FUNCTION DOC] Casts x to dtype: bool becomes a not-equal-zero test,
        # others a Halide cast to the mapped type. src_dtype and use_compute_types are ignored.
        if dtype == torch.bool:
            return f"({x} != 0)"
        return f"hl.cast({halide_type(dtype)}, {x})"

    @staticmethod
    def to_dtype_bitcast(x, dtype: torch.dtype, src_dtype: torch.dtype):
        # [FUNCTION DOC] Reinterprets the bits of x as dtype. Half-precision
        # sources are first cast to their own type (body math is fp32);
        # half-precision targets are reinterpreted then cast back up to fp32.
        if src_dtype in (torch.float16, torch.bfloat16):
            x = f"hl.cast({halide_type(src_dtype)}, {x})"  # body compute is upcast to fp32
        line = f"hl.reinterpret({halide_type(dtype)}, {x})"
        if dtype in (torch.float16, torch.bfloat16):
            line = f"hl.cast(hl.Float(32), {line})"
        return line

    @classmethod
    def constant(cls, value, dtype):
        # [FUNCTION DOC] Returns a constant expression of the given dtype built from value.
        return cls.to_dtype(halide_constant(value), dtype)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def abs(x):
        # [FUNCTION DOC] Absolute value expression.
        return f"hl.abs({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def exp(x):
        # [FUNCTION DOC] Exponential. For a named CSE variable, emits a runtime
        # choice: fast approximate exp (computed in fp32) when the value type is
        # at most 32 bits wide, else the exact exp. For plain text, exact exp.
        if not hasattr(x, "name"):
            return f"hl.exp({x})"
        return f"hl.fast_exp(hl.cast(hl.Float(32), {x})) if {x.name}.type().bits() <= 32 else hl.exp({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def sqrt(x):
        # [FUNCTION DOC] Square root expression.
        return f"hl.sqrt({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def minimum(a, b):
        # [FUNCTION DOC] Elementwise minimum that propagates NaN for float
        # operands (select on less-than or NaN), falling back to the plain
        # minimum for integer types; b is cast to a's type. Plain text operands
        # (no name) just use the plain minimum.
        # return f"hl.min({a}, {b})"  <== handles nan wrong
        if not hasattr(a, "name"):
            return f"hl.min({a}, {b})"
        b = f"hl.cast({a.name}.type(), {b})"
        return f"hl.select(({a}<{b})|hl.is_nan({a}), {a}, {b}) if {a.name}.type().is_float() else hl.min({a}, {b})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def maximum(a, b):
        # [FUNCTION DOC] Elementwise maximum that propagates NaN for float
        # operands (select on greater-than or NaN), falling back to the plain
        # maximum for integer types; b is cast to a's type. Plain text operands
        # (no name) just use the plain maximum.
        # return f"hl.max({a}, {b})"  <== handles nan wrong
        if not hasattr(a, "name"):
            return f"hl.max({a}, {b})"
        b = f"hl.cast({a.name}.type(), {b})"
        return f"hl.select(({a}>{b})|hl.is_nan({a}), {a}, {b}) if {a.name}.type().is_float() else hl.max({a}, {b})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def fmaximum(a, b):
        # [FUNCTION DOC] Maximum using Halide's plain max (no explicit NaN handling).
        return f"hl.max({a}, {b})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def where(a, b, c):
        # [FUNCTION DOC] Select b where condition a holds else c; c is cast to
        # b's type when b is a named variable.
        if hasattr(b, "name"):
            c = f"hl.cast({b.name}.type(), {c})"
        return f"hl.select({a}, {b}, {c})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def cos(x):
        # [FUNCTION DOC] Cosine expression.
        return f"hl.cos({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def sin(x):
        # [FUNCTION DOC] Sine expression.
        return f"hl.sin({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def lgamma(x):
        # [FUNCTION DOC] Not supported; always raises Unsupported.
        raise Unsupported("lgamma")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def erf(x):
        # [FUNCTION DOC] Error function expression.
        return f"hl.erf({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def cosh(x):
        # [FUNCTION DOC] Hyperbolic cosine expression.
        return f"hl.cosh({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def sinh(x):
        # [FUNCTION DOC] Hyperbolic sine expression.
        return f"hl.sinh({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def acos(x):
        # [FUNCTION DOC] Arc cosine expression.
        return f"hl.acos({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def acosh(x):
        # [FUNCTION DOC] Inverse hyperbolic cosine expression.
        return f"hl.acosh({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def asin(x):
        # [FUNCTION DOC] Arc sine expression.
        return f"hl.asin({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def asinh(x):
        # [FUNCTION DOC] Inverse hyperbolic sine expression.
        return f"hl.asinh({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def atan2(x, y):
        # [FUNCTION DOC] Two-argument arc tangent expression.
        return f"hl.atan2({x}, {y})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def atan(x):
        # [FUNCTION DOC] Arc tangent expression.
        return f"hl.atan({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def atanh(x):
        # [FUNCTION DOC] Inverse hyperbolic tangent expression.
        return f"hl.atanh({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def copysign(x, y):
        # [FUNCTION DOC] Not supported; always raises Unsupported.
        raise Unsupported("copysign")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def erfinv(x):
        # [FUNCTION DOC] Not supported; always raises Unsupported.
        raise Unsupported("erfinv")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def hypot(x, y):
        # [FUNCTION DOC] Hypotenuse (Euclidean norm of two values) expression.
        return f"hl.hypot({x}, {y})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def nextafter(x, y):
        # [FUNCTION DOC] Not supported; always raises Unsupported.
        raise Unsupported("nextafter")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def logical_and(a, b):
        # [FUNCTION DOC] Logical and, emitted as bitwise and (operands are booleans).
        return f"{a} & {b}"

    @staticmethod
    def logical_not(a):
        # [FUNCTION DOC] Logical not, emitted as an equals-zero test.
        return f"{a} == 0"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def logical_or(a, b):
        # [FUNCTION DOC] Logical or, emitted as bitwise or.
        return f"{a} | {b}"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def logical_xor(a, b):
        # [FUNCTION DOC] Logical xor, emitted as a parenthesized bitwise xor.
        return f"({a} ^ {b})"

    @staticmethod
    def bitwise_and(a, b):
        # [FUNCTION DOC] Bitwise and expression (not parenthesized).
        return f"{a} & {b}"

    @staticmethod
    def bitwise_not(a):
        # [FUNCTION DOC] Bitwise complement expression.
        return f"~{a}"

    @staticmethod
    def bitwise_or(a, b):
        # [FUNCTION DOC] Bitwise or expression (not parenthesized).
        return f"{a} | {b}"

    @staticmethod
    def bitwise_xor(a, b):
        # [FUNCTION DOC] Bitwise xor expression (not parenthesized).
        return f"{a} ^ {b}"

    @staticmethod
    def bitwise_left_shift(a, b):
        # [FUNCTION DOC] Left shift expression.
        return f"{a} << {b}"

    @staticmethod
    def bitwise_right_shift(a, b):
        # [FUNCTION DOC] Right shift expression.
        return f"{a} >> {b}"

    @staticmethod
    def rand(seed, offset):
        # [FUNCTION DOC] Uniform random expression calling the helper module's rand with seed and offset.
        return f"halide_helpers.rand({seed}, {offset})"

    @staticmethod
    def randn(seed, offset):
        # [FUNCTION DOC] Normal random expression calling the helper module's randn with seed and offset.
        return f"halide_helpers.randn({seed}, {offset})"

    @staticmethod
    def rand_eager(seed, base_offset, threads_per_round, tid, vec):
        # [FUNCTION DOC] Random expression matching eager-mode sequencing, delegating to the helper
        # module's eager-kernel function with the given seed, base offset, thread layout and vector width.
        return f"halide_helpers.rand_eager_kernel({seed}, {base_offset}, {threads_per_round}, {tid}, {vec})"

    @staticmethod
    def randint64(seed, offset, low, high):
        # [FUNCTION DOC] Random 64-bit integer in [low, high) via the helper module.
        return f"halide_helpers.randint64({seed}, {offset}, {low}, {high})"

    @staticmethod
    def load_seed(name, offset):
        # [FUNCTION DOC] Expression loading the random seed from buffer name
        # plus a kernel-argument seed offset registered on the current kernel's args (side effect).
        return f"{ops.load(name, 0)} + {V.kernel.args.seed_offset('load_seed_offset', offset)}"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def rsqrt(x):
        # [FUNCTION DOC] Reciprocal square root as one divided by sqrt (fast
        # inverse sqrt is avoided for accuracy).
        # return f"hl.fast_inverse_sqrt({x})"  <== accuracy issues
        return f"1./hl.sqrt({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def tan(x):
        # [FUNCTION DOC] Tangent expression.
        return f"hl.tan({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def tanh(x):
        # [FUNCTION DOC] Hyperbolic tangent expression.
        return f"hl.tanh({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def signbit(x):
        # [FUNCTION DOC] Sign-bit test: casts x to fp32, reinterprets as 32-bit
        # unsigned, and checks the top bit.
        return f"(hl.reinterpret(hl.UInt(32), hl.cast(hl.Float(32), {x})) >> 31) != 0"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def fmod(a, b):
        # [FUNCTION DOC] C-style remainder (sign follows dividend), computed as
        # a minus truncated quotient times b because Halide's native modulo has
        # the wrong sign convention.
        # TODO(jansel): find a better way to do this, builtin % has wrong sign
        return f"{a} - hl.trunc({a}/{b})*{b}"

    @staticmethod
    def pow(a, b):
        # [FUNCTION DOC] Power expression using the exact pow (fast pow fails accuracy tests).
        return f"hl.pow({a}, {b})"  # hl.fast_pow fails accuracy

    @staticmethod
    # pyrefly: ignore [bad-override]
    def ldexp(x, n):
        # [FUNCTION DOC] Not supported; always raises Unsupported.
        raise Unsupported("ldexp")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def log(x):
        # [FUNCTION DOC] Natural log expression using the exact log (fast log fails accuracy tests).
        return f"hl.log({x})"  # hl.fast_log fails accuracy

    @staticmethod
    def log2(x):
        # [FUNCTION DOC] Not supported; always raises NotImplementedError.
        raise NotImplementedError("log2")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def isinf(x):
        # [FUNCTION DOC] Infinity test on x after casting to fp32 (works around a Halide bug).
        # workaround https://github.com/halide/Halide/issues/8309
        return f"hl.is_inf(hl.cast(hl.Float(32), {x}))"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def isnan(x):
        # [FUNCTION DOC] NaN test on x after casting to fp32 (works around a Halide bug).
        # workaround https://github.com/halide/Halide/issues/8309
        return f"hl.is_nan(hl.cast(hl.Float(32), {x}))"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def round(x):
        # [FUNCTION DOC] Round expression.
        return f"hl.round({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def floor(x):
        # [FUNCTION DOC] Floor expression.
        return f"hl.floor({x})"

    @staticmethod
    def int_truediv(a, b):
        # [FUNCTION DOC] True division of integers: divisor is promoted to float by adding a float zero.
        return f"({a}) / ({b} + hl.f32(0))"

    @staticmethod
    def floordiv(a, b):
        # [FUNCTION DOC] Floor division computed by casting a to a float at
        # least 32 bits wide (or a's own width if larger), dividing by b, and
        # flooring. Requires a to be a named variable (uses its type).
        # KNOWN ISSUE: float division can lose precision for 64-bit integers wider than the float mantissa.
        # TODO(jansel): find a better ways to do this, the select-based trick from triton.py didn't work
        return (
            f"hl.floor(hl.cast(hl.Float(max(32, {a.name}.type().bits())), {a}) / {b})"
        )

    @classmethod
    def sign(cls, x):
        # [FUNCTION DOC] Sign of x as (0 < x) minus (x < 0), computed through
        # other ops on int8 values and cast back to x's type. Requires x to be a named variable.
        left = ops.to_dtype(ops.lt("0", x), torch.int8)
        right = ops.to_dtype(ops.lt(x, "0"), torch.int8)
        sub = ops.sub(left, right)
        return f"hl.cast({x.name}.type(), {sub})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def trunc(x):
        # [FUNCTION DOC] Truncate toward zero expression.
        return f"hl.trunc({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def truncdiv(a, b):
        # [FUNCTION DOC] Division rounding toward zero, computed in a float at
        # least 32 bits wide then truncated (Halide's round-to-zero division is
        # avoided because it crashes on zero divisors). Requires a to be a named variable.
        # this causes crashes with floating point exception, see test_div_zero_dim_cpu
        # return f"hl.div_round_to_zero({a}, {b})"
        return (
            f"hl.trunc(hl.cast(hl.Float(max(32, {a.name}.type().bits())), {a}) / {b})"
        )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def ceil(x):
        # [FUNCTION DOC] Ceiling expression.
        return f"hl.ceil({x})"

    @staticmethod
    def relu(x):
        # [FUNCTION DOC] Rectified linear unit: maximum of x and zero.
        return f"hl.max({x}, 0)"

    @classmethod
    def index_expr(cls, expr, dtype):
        # [FUNCTION DOC] Materializes a symbolic index expression as a CSE
        # function over the dimensions it uses (via the current kernel), with
        # value-range bounds. Converts to dtype unless it is already int32 or
        # int64. Returns the resulting variable.
        index = V.kernel.prepare_indexing(expr)
        var = V.kernel.genfunc(
            V.kernel.index_to_str(index),
            V.kernel.used_dims_from_index(index),
            bounds=get_bounds_index_expr(expr),
        )
        if dtype not in (torch.int32, torch.int64):
            return ops.to_dtype(var, dtype)
        return var

    @classmethod
    def value_expr(cls, expr, dtype):
        # [FUNCTION DOC] Like index_expr but always converts the result to
        # dtype (no int32/int64 shortcut). Returns the resulting variable.
        index = V.kernel.prepare_indexing(expr)
        var = V.kernel.genfunc(
            V.kernel.index_to_str(index),
            V.kernel.used_dims_from_index(index),
            bounds=get_bounds_index_expr(expr),
        )
        return ops.to_dtype(var, dtype)

    @classmethod
    def indirect_indexing(cls, index_var, size, check=True, wrap_neg=True):
        # [FUNCTION DOC] Prepares a loaded value for use as an index: converts to int32,
        # clamps to the valid range for size, records the size on the variable,
        # and returns an index symbol named after it. wrap_neg is ignored.
        # KNOWN ISSUE: negative indices are not wrapped despite the wrap_neg flag; clamp maps them to zero.
        # TODO(jansel): Halide only supports 32-bit indexing, we should error on overflow
        index_var = ops.to_dtype(index_var, torch.int32)
        index_var = ops.halide_clamp(index_var, size, check)
        index_var.indirect_indexing_size = size
        return sympy_index_symbol(str(index_var))

    @classmethod
    def halide_clamp(cls, value, size, check):
        # [FUNCTION DOC] Returns an expression clamping value into zero through
        # size minus one (cast to value's type when size is symbolic). The check
        # flag is unused and the unsafe-promise variant is deliberately
        # disabled because of a Halide bug.
        end = V.kernel.kexpr(V.kernel.rename_indexing(size) - 1)
        if not isinstance(size, (int, sympy.Integer)):
            end = f"hl.cast({value.name}.type(), {end})"
        # Skip unsafe_promise_clamped to workaround: https://github.com/halide/Halide/issues/8261#issuecomment-2148835692
        # return f"hl.unsafe_promise_clamped({value}, 0, {end})"
        return f"hl.clamp({value}, 0, {end})"

    @staticmethod
    def masked(mask, body, other):
        # [FUNCTION DOC] Evaluates the body callable with loads masked by mask
        # (fill value other), then selects the body result where the mask holds
        # else a constant other cast to the result's type (bool-ranged results
        # coerce other to bool). Returns the selected variable.
        with V.kernel.mask_loads(mask, other) as new_mask:
            result = body()

        if result.bounds.is_bool:
            other = bool(other)

        # Take dtype from result to prevent accidental promotion
        other = V.kernel.genfunc(
            f"hl.cast({result.name}.type(), {halide_constant(other)})",
            [],
            bounds=ValueRanges.wrap(other),
            shape=result.shape,
        )
        # TODO(jansel): look into removing the where in the same places triton does
        return ops.where(new_mask, result, other)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def frexp(x):
        # [FUNCTION DOC] Not supported; always raises NotImplementedError.
        raise NotImplementedError("frexp")

    @staticmethod
    def device_assert_async(cond, msg):
        # [FUNCTION DOC] Not supported; always raises NotImplementedError.
        raise NotImplementedError("device_assert_async")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def partial_accumulate(
        name: str,
        reduction_type: str,
        value: CSEVariable,
        extra_meta: dict[str, Any],
    ) -> None:
        # [FUNCTION DOC] Not supported; always raises NotImplementedError.
        raise NotImplementedError


HalideOverrides._initialize_pointwise_overrides("halide")


class HalideCSEVariable(CSEVariable):
    # [CLASS DOC] CSE variable for a Halide Func. Besides the base name/bounds/
    # dtype/shape, tracks used_dims: the ordered halide index variables the
    # value depends on (None until computed). Its string form indexes the Func
    # by those dims; a placeholder with a question mark is printed while unknown
    # and matched by undefined_re for later replacement.
    undefined_re = re.compile(r"\b(tmp\d+)\[\?\]")

    def __init__(
        self,
        name,
        bounds: ValueRanges[Any],
        dtype: torch.dtype | None = None,
        shape: BlockShapeType = None,
    ) -> None:
        # [FUNCTION DOC] Initializes the base CSE variable and sets used_dims to None (unknown).
        super().__init__(name, bounds, dtype, shape=shape)
        self.used_dims: list[sympy.Symbol] | None = None

    def update_on_args(self, name, args, kwargs):
        # [FUNCTION DOC] Recomputes used_dims as the union of the current
        # dims and the dims of every HalideCSEVariable among args and kwarg
        # values, sorted by the current kernel's ordering. Raises AssertionError
        # if an argument variable has unknown dims.
        used = OrderedSet(self.used_dims or ())
        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, HalideCSEVariable):
                if arg.used_dims is None:
                    raise AssertionError((name, arg, args))
                used.update(arg.used_dims)
        self.used_dims = V.kernel.sort_used_dims(used)

    def index_str(self, dims):
        # [FUNCTION DOC] Source text indexing this Func with the given dims: an
        # empty tuple index for scalars, else a comma-separated list of dims.
        if len(dims) == 0:
            return f"{self.name}[()]"
        # Reversed since Halide is column major
        return f"{self.name}[{', '.join(map(str, dims))}]"

    def __str__(self) -> str:
        # [FUNCTION DOC] Indexed form using used_dims, or a placeholder with a
        # question mark if dims are not yet known (replaced later in kernel codegen).
        if self.used_dims is None:
            # This will get recomputed and replaced in codegen_kernel()
            return f"{self.name}[?]"
        return self.index_str(self.used_dims)

    def subs_str(self, replacements):
        # [FUNCTION DOC] Indexed form after substituting dims through the
        # replacements mapping (dims not in it are kept). Raises AssertionError
        # if used_dims is unset or not symbolic expressions.
        if not (
            self.used_dims is not None
            and all(isinstance(x, sympy.Expr) for x in self.used_dims)
        ):
            raise AssertionError("used_dims must be set sympy expressions")
        return self.index_str([replacements.get(n, n) for n in self.used_dims])


@dataclasses.dataclass
class DimensionInfo:
    # [CLASS DOC] One dimension of a Halide buffer access: expr is the index
    # expression for the dimension (None once different accesses disagree),
    # size is its extent and stride its element stride. Strides are normalized
    # to be non-negative by flipping the sign of expr when needed.
    expr: sympy.Expr | None
    size: sympy.Expr
    stride: sympy.Expr

    def __init__(self, expr, size, stride) -> None:
        # [FUNCTION DOC] Stores the fields; if the stride is statically known
        # negative, negates both stride and expr first.
        super().__init__()
        if V.graph.sizevars.statically_known_lt(stride, 0):
            stride = -stride
            expr = -expr
        self.expr = expr
        self.size = size
        self.stride = stride

    def index_str(self, replacements=None, zero_vars=False):
        # [FUNCTION DOC] Source text for this dimension's index. A zero expr
        # prints a fresh unnamed variable when zero_vars is set. If replacements
        # is given, temp (CSE) symbols in expr are first resolved to their
        # variables' substituted index text, then all replacements applied.
        # Raises AssertionError if expr is None or a temp symbol is not a Halide CSE variable.
        if self.expr is None:
            raise AssertionError("expr must not be None")
        expr = self.expr
        if zero_vars and expr == 0:
            return "hl.Var()"
        if replacements:
            replacements = {**replacements}
            # pyrefly: ignore [missing-attribute]
            for sym in expr.free_symbols:
                if symbol_is_type(sym, SymT.TMP):
                    if not isinstance(sym, sympy.Symbol):
                        raise AssertionError(f"expected sympy.Symbol, got {type(sym)}")
                    var = V.kernel.lookup_cse_var(sym.name)
                    if not isinstance(var, HalideCSEVariable):
                        raise AssertionError(
                            f"expected HalideCSEVariable, got {type(var)}"
                        )
                    replacements[sym] = sympy_index_symbol(var.subs_str(replacements))
            expr = sympy_subs(expr, replacements)
        return V.kernel.index_to_str(expr)


def eq(left, right):
    # [FUNCTION DOC] HELPER: True only when the size variables guard machinery can prove equality (otherwise False).
    return V.graph.sizevars.guard_or_false(sympy.Eq(left, right))


def lt(left, right):
    # [FUNCTION DOC] HELPER: Conservative less-than test for non-negative
    # sizes: True if provable, else True when left divides right and they
    # differ, otherwise False.
    # KNOWN ISSUE: the divisibility fallback is unsound for unbacked symbols that may be zero (as the TODO notes).
    """Compare sizes: only use on inputs known to be >= 0."""
    if V.graph.sizevars.guard_or_false(sympy.Lt(left, right)):
        return True

    # GCD fallback: if gcd(left, right) == left then left divides right,
    # so left <= right.  Combined with left != right this gives left < right.
    #
    # TODO: This is NOT always sound for unbacked symints.
    # e.g. lt(u0, 10*u0): gcd=u0, gcd==left, u0 != 10*u0 → returns True,
    # but if u0=0 then 0 < 0 is False.  The >= 0 checks mitigate the
    # negative case but the zero case remains.
    # TODO shall we add a runtime assertion at least.
    gcd = sympy.gcd(left, right)
    if gcd == left:
        return left != right

    return False


class HalideKernel(SIMDKernel):
    # [CLASS DOC] SIMD kernel that generates a Halide pipeline as Python
    # source. It replaces the usual flat xindex/rindex indexing with a set of
    # Halide variables (halide_vars) whose products give the loop extents, with
    # index_replacements mapping range-tree symbols to expressions of them, and
    # reduction_renames mapping to reduction domain variables. Loads, compute
    # and stores all go directly to the body buffer. It tracks per-buffer
    # dimension info (buffer_dimensions/offsets/aliases), domain renames for
    # update definitions, and whether indirect indexing is used. Uses the
    # Halide overrides and the Halide expression printer.
    overrides = HalideOverrides  # type: ignore[assignment]
    kexpr: Callable[[sympy.Expr], str] = texpr

    def __init__(
        self,
        tiling: dict[str, sympy.Expr],
        **kwargs,
    ) -> None:
        # [FUNCTION DOC] Initializes the base kernel then makes compute, loads
        # and stores alias the body buffer, creates the dom-indexing buffer,
        # and empties all indexing state mappings (buffer dimensions/offsets,
        # halide vars, index replacements, reduction and dom renames, buffer
        # aliases). Dom indexing and has_reduction mirror whether inside a reduction.
        super().__init__(tiling, **kwargs)
        # For halide, we just write directly to the body
        self.compute = self.body
        self.loads = self.body
        self.stores = self.body
        self.indexing_code_dom = IndentedBuffer()
        self.needs_dom_indexing = self.inside_reduction
        self.has_reduction = self.inside_reduction
        self.buffer_dimensions: dict[str, list[DimensionInfo]] = {}
        self.buffer_offsets: dict[str, sympy.Expr] = {}
        # {h0: size1, h1: size2, ...}
        self.halide_vars: dict[sympy.Symbol, sympy.Expr] = {}
        # {x0: h0, x1: h1+10*h2, ...}
        self.index_replacements: dict[sympy.Expr, sympy.Expr] = {}
        # {h1: hr1, ...}
        self.reduction_renames: dict[sympy.Symbol, sympy.Symbol] = {}
        # {"i": {h0: hi0}, "o": ...}
        self.dom_renames: dict[str, dict[sympy.Symbol, sympy.Symbol]] = {}
        # {"in_ptr0": ["in_ptr0_view0"], ...}
        self.buffer_aliases: dict[str, list[str]] = defaultdict(list)
        self.has_indirect_indexing = False

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        # [FUNCTION DOC] Returns the Halide type text for dtype.
        return halide_type(dtype)

    # pyrefly: ignore [bad-override]
    def create_cse_var(self, name, bounds=None, dtype=None, shape=None):
        # [FUNCTION DOC] Emits a line declaring a new named Halide Func into the
        # body and returns a HalideCSEVariable for it (side effect: writes to body).
        self.body.writeline(f"{name} = hl.Func({name!r})")
        # pyrefly: ignore [bad-argument-type]
        return HalideCSEVariable(name, bounds, dtype, shape)

    def finalize_indexing(self, indices: Sequence[sympy.Expr]):
        # [FUNCTION DOC] Hook run before codegen with all indices used in the
        # fused kernel. Finds which range-tree symbols are really used (visiting
        # modular and floor-division indexing to include derived symbols), notes
        # whether indirect indexing occurs, and for each tree chooses a minimal
        # set of halide vars (sized via gcd of needed lengths and divisors, with a
        # single fallback var when sizes share no factor) stored in halide_vars,
        # marking reduction vars in reduction_renames. Then builds
        # index_replacements mapping each range-tree symbol to a combination of
        # those vars (or a modular-indexing expression in the fallback case),
        # and emits variable and reduction-domain definitions into indexing_code.
        # Raises AssertionError if state was already populated or invariants fail.
        """
        Hook called right before codegen with every index that will be
        used in the fused kernel.

        This populates self.halide_vars/index_replacements/reduction_renames which is an alternate indexing
        scheme that avoids using divide and modulus.  Instead of xindex/yindex/rindex
        we base indexing on a larger number of vars whose product combines to those.

        This function populates self.halide_vars, self.index_replacements, and self.reduction_renames
        """
        if self.index_replacements or self.halide_vars or self.reduction_renames:
            raise AssertionError(
                "indexing state must be empty before finalize_indexing"
            )
        size_hint = functools.partial(
            V.graph.sizevars.optimization_hint, fallback=sys.maxsize
        )
        # pyrefly: ignore [bad-assignment]
        indices = dict.fromkeys(map(super().prepare_indexing, indices))
        all_used_symbols = OrderedSet[Any]()
        sym_to_node = {
            n.symbol(): n
            for n in itertools.chain.from_iterable(
                [tree.nodes.values() for tree in self.range_trees]
            )
        }

        def simplify(expr):
            # [FUNCTION DOC] HELPER: Simplifies expr after removing precomputed-size replacements.
            return sympy.simplify(
                V.graph.sizevars.remove_precomputed_replacements(expr)
            )

        def visit_modular_indexing(base, divisor, modulus):
            # [FUNCTION DOC] HELPER: Replacement callback; if base is a range-tree
            # symbol, records the symbol of the derived tree node (divisor
            # multiplied, length bounded by modulus) in all_used_symbols.
            if base in sym_to_node:
                node = sym_to_node[base]
                all_used_symbols.add(
                    node.root.lookup(
                        node.divisor * divisor,
                        V.graph.sizevars.evaluate_min(
                            modulus,
                            FloorDiv(node.length, divisor),
                        ),
                    ).symbol()
                )

        def visit_floor_div(base, divisor):
            # [FUNCTION DOC] HELPER: Replacement callback; if base is a range-tree
            # symbol, records the symbol of the derived tree node (divisor
            # multiplied, length divided) in all_used_symbols.
            if base in sym_to_node:
                node = sym_to_node[base]
                all_used_symbols.add(
                    node.root.lookup(
                        node.divisor * divisor,
                        FloorDiv(node.length, divisor),
                    ).symbol()
                )

        # first figure out all_used_symbols to do dead symbol elimination
        for index in indices:
            if index.has(ModularIndexing):
                index.replace(
                    ModularIndexing(
                        sympy.Wild("base"),
                        sympy.Wild("divisor"),
                        sympy.Wild("modulus"),
                    ),
                    visit_modular_indexing,
                )
            if index.has(FloorDiv):
                index.replace(
                    FloorDiv(
                        sympy.Wild("base"),
                        sympy.Wild("divisor"),
                    ),
                    visit_floor_div,
                )
            all_used_symbols.update(super().prepare_indexing(index).free_symbols)

        self.has_indirect_indexing = any(
            symbol_is_type(sym, SymT.INDIRECT) for sym in all_used_symbols
        )

        had_fallback = False
        for tree in reversed(self.range_trees):
            nodes = [n for n in tree.nodes.values() if n.symbol() in all_used_symbols]
            nodes.sort(key=lambda n: size_hint(n.divisor))
            if not nodes:
                nodes.append(tree.lookup(1, tree.numel))
            handled_count = 0
            divisor = sympy.S.One
            added_sym_size = []
            # decide on a minimal set of symbols and put them in self.halide_vars
            while handled_count < len(nodes) and not eq(tree.numel, divisor):
                sizes_to_add = [
                    simplify(n.length) for n in nodes if eq(n.divisor, divisor)
                ]
                handled_count += len(sizes_to_add)
                if not sizes_to_add:
                    raise AssertionError(nodes)
                end = divisor * functools.reduce(
                    lambda a, b: V.graph.sizevars.evaluate_max(a, b),
                    sizes_to_add,
                )
                sizes_to_add.extend(
                    [
                        simplify(n.divisor / divisor)
                        for n in nodes
                        if lt(divisor, n.divisor) and lt(n.divisor, end)
                    ]
                )
                # pyrefly: ignore [bad-assignment]
                while sizes_to_add:
                    next_size = functools.reduce(sympy.gcd, sizes_to_add)
                    if eq(next_size, 1):
                        # sizes share no common factors, e.g [2, 21, 42, 441, 889056]
                        # TODO(jansel): we should just prevent fusion in cases that hit this
                        next_size = simplify(tree.numel / divisor)
                        if eq(next_size, 1):
                            raise AssertionError("fallback next_size must not be 1")
                        sizes_to_add = []
                        handled_count = len(nodes)
                        had_fallback = True
                    sym = sympy_index_symbol(f"h{len(self.halide_vars)}")
                    # pyrefly: ignore [missing-argument]
                    if tree.is_reduction:
                        self.reduction_renames[sym] = sympy_index_symbol(
                            f"hr{len(self.halide_vars)}"
                        )
                    self.halide_vars[sym] = next_size
                    added_sym_size.append((sym, next_size))
                    divisor *= next_size
                    new_sizes = [n.length for n in nodes if eq(n.divisor, divisor)]
                    handled_count += len(new_sizes)
                    prior_len = len(sizes_to_add)
                    sizes_to_add = [
                        sympy.simplify(s / next_size)
                        for s in sizes_to_add
                        if not eq(s, next_size)
                    ]
                    if not (len(sizes_to_add) < prior_len or prior_len == 0):
                        raise AssertionError("sizes_to_add must shrink")
                    sizes_to_add.extend(new_sizes)

            # create a mapping to the new set of symbols in self.index_replacements
            for node in nodes:
                try:
                    idx = 0
                    divisor = 1
                    while not eq(node.divisor, divisor):
                        sym, size = added_sym_size[idx]
                        idx += 1
                        divisor *= size
                    length = 1
                    expr = sympy.S.Zero
                    while not eq(node.length, length):
                        sym, size = added_sym_size[idx]
                        idx += 1
                        expr += length * sym
                        length *= size
                    self.index_replacements[node.symbol()] = expr
                except IndexError:
                    if not had_fallback:
                        raise AssertionError("IndexError without fallback") from None
                    full_index = sympy.S.Zero
                    stride = sympy.S.One
                    for sym, size in added_sym_size:
                        full_index += stride * sym
                        stride *= size
                    self.index_replacements[node.symbol()] = (
                        V.graph.sizevars.simplify_with_ranges(
                            ModularIndexing(full_index, node.divisor, node.length),
                            self.halide_vars,  # type: ignore[arg-type]
                        )
                    )

        # codegen the variable definitions
        for sym in self.halide_vars:
            self.indexing_code.writeline(f"{sym} = hl.Var({sym.name!r})")
        if self.reduction_renames:
            self.codegen_rdom(
                "rdom",
                {rv: self.halide_vars[v] for v, rv in self.reduction_renames.items()},
            )

    def setup_dom_indexing(self):
        # [FUNCTION DOC] Creates (once per prefix) a reduction domain used for
        # explicit-range Func updates. The prefix is "i" inside a reduction else
        # "o" (outputs skip reduction vars). Renames each halide var to a
        # prefixed name, emits the domain definition through codegen_rdom,
        # caches and returns the rename mapping. Raises AssertionError on an
        # unexpected variable name.
        """RDom based indexing uses explicit iteration ranges for Func updates"""
        prefix = "i" if self.inside_reduction else "o"
        if prefix in self.dom_renames:
            return self.dom_renames[prefix]

        renames = {}
        for var in self.halide_vars:
            if not self.inside_reduction and var in self.reduction_renames:
                continue
            m = re.match(r"^h(\d+)$", var.name)
            if not m:
                raise AssertionError(f"unexpected halide var name {var.name}")
            renames[var] = sympy_index_symbol(f"h{prefix}{m.group(1)}")

        self.codegen_rdom(
            f"{prefix}dom", {rv: self.halide_vars[v] for v, rv in renames.items()}
        )

        self.dom_renames[prefix] = renames
        return renames

    def codegen_rdom(self, name, vars):
        # [FUNCTION DOC] Emits indexing code defining a reduction domain called
        # name with one range per size in the vars mapping (symbol to size), then
        # binds each symbol to the corresponding domain component.
        rsizes = [
            f"hl.Range(0, {self.kexpr(self.rename_indexing(size))})"
            for size in vars.values()
        ]
        self.indexing_code.writeline(f"{name} = hl.RDom([{', '.join(rsizes)}])")
        for i, rsym in enumerate(vars.keys()):
            self.indexing_code.writeline(f"{rsym} = {name}[{i}]")

    def prepare_indexing(
        self,
        index: sympy.Expr,
    ):
        # [FUNCTION DOC] Base preparation of the index, then substitutes
        # range-tree symbols using index_replacements and simplifies using the
        # known ranges of halide_vars. Returns the new expression.
        index = super().prepare_indexing(index)
        index = sympy_subs(index, self.index_replacements)
        return V.graph.sizevars.simplify_with_ranges(index, self.halide_vars)  # type: ignore[arg-type]

    def sym_size(self, sym):
        # [FUNCTION DOC] Extent of an index symbol: the recorded indirect-index
        # size for a temp (CSE) symbol, otherwise the size in halide_vars (KeyError if unknown).
        """The size of an index symbol"""
        if symbol_is_type(sym, SymT.TMP):
            return self.lookup_cse_var(sym.name).indirect_indexing_size
        return self.halide_vars[sym]

    def indexing_to_dimensions(self, var: str, index: sympy.Expr, is_store: bool):
        # [FUNCTION DOC] Converts a flat address expression into a list of
        # DimensionInfo for buffer var. Splits the expanded index into terms by
        # the halide/temp symbols they use (constants form the offset; terms
        # sharing symbols are merged), turns each group into a strided
        # dimension (stores require a pure stride-times-symbol form, loads may
        # use an approximated extent), sorts by stride, and adds a dummy
        # dimension when the smallest stride is not one (or for scalars
        # with indirect indexing). For loads, a positive offset is folded
        # into the dimensions (reusing a prior offset if possible). Registers the
        # dims via install_dims, creating numbered view aliases of the buffer
        # when a load conflicts with earlier registrations. Returns the
        # (possibly aliased) buffer name and the dims. Raises AssertionError
        # on unsupported symbols or impossible store shapes.
        """Convert address-based indexing into dimensions using self.halide_vars"""
        symbols = []
        for sym in sorted(index.free_symbols, key=lambda x: x.name):  # type: ignore[attr-defined]
            if symbol_is_type(sym, (SymT.HALIDE, SymT.TMP)):
                symbols.append(sym)
            else:
                if not symbol_is_type(
                    sym,
                    (
                        SymT.UNBACKED_INT,
                        SymT.SIZE,
                        SymT.PRECOMPUTED_SIZE,
                    ),
                ):
                    raise AssertionError(sym)

        # group the expression by variables used
        offset = sympy.S.Zero
        split_expr = dict.fromkeys(symbols, sympy.S.Zero)
        split_failed: list[tuple[list[sympy.Symbol], sympy.Expr]] = []
        index = sympy.expand(self.rename_indexing(index))
        for part in index.args if isinstance(index, sympy.Add) else [index]:
            part_vars = [v for v in part.free_symbols if v in split_expr]
            if len(part_vars) == 0:
                offset += part
            elif len(part_vars) == 1:
                split_expr[part_vars[0]] += part
            else:
                new_split_failed = []
                for i in range(len(split_failed)):
                    if split_failed[i] is None:
                        raise AssertionError("split_failed entry must not be None")
                    other_vars, other_part = split_failed[i]
                    if OrderedSet(other_vars) & OrderedSet(part_vars):
                        part_vars.extend([v for v in other_vars if v not in part_vars])
                        part += other_part
                    else:
                        new_split_failed.append((other_vars, other_part))
                split_failed = [*new_split_failed, (part_vars, part)]

        def expr_to_dimension(expr, syms):
            # [FUNCTION DOC] HELPER: Builds a DimensionInfo for one term group.
            # A single symbol times a symbol-free stride gives an exact
            # dimension. Otherwise (loads only; AssertionError for stores) the
            # extent is the expression evaluated at maximum symbol values plus
            # one, with any integer factors pulled out into the stride.
            expr = sympy.factor(expr)
            if len(syms) == 1:
                stride_wild = sympy.Wild("wild", exclude=symbols)
                m = expr.match(stride_wild * syms[0])
                if m:
                    return DimensionInfo(
                        syms[0], self.sym_size(syms[0]), m[stride_wild]
                    )
            if is_store:
                raise AssertionError(expr)
            length = sympy.simplify(
                sympy_subs(expr, {sym: self.sym_size(sym) - 1 for sym in syms}) + 1
            )
            stride = sympy.S.One
            if isinstance(expr, sympy.Mul):
                for term in expr.args:
                    if isinstance(term, sympy.Integer):
                        stride *= term
                        expr = sympy.simplify(expr / term)
                        length = sympy.simplify(sympy.ceiling(length / term))
            return DimensionInfo(expr, length, stride)

        # try to turn each group into a strided access
        dims = []
        for syms, expr in split_failed:
            for v in syms:
                expr += split_expr.pop(v)
            dims.append(expr_to_dimension(expr, syms))
        for sym, expr in split_expr.items():
            dims.append(expr_to_dimension(expr, [sym]))
        dims.sort(
            key=lambda d: V.graph.sizevars.optimization_hint(
                d.stride, fallback=sys.maxsize
            )
        )  # type: ignore[arg-type]

        if not dims:  # scalar load/store
            if self.has_indirect_indexing:
                # workaround https://github.com/halide/Halide/issues/8338
                dims.append(DimensionInfo(sympy.S.Zero, 1, 1))
        elif not V.graph.sizevars.statically_known_equals(dims[0].stride, 1):
            # Halide assumes dimension 0 is stride == 1, so add a dummy dimension
            dims.insert(
                0, DimensionInfo(sympy.S.Zero, 1 if is_store else dims[0].stride, 1)
            )

        if dims and not is_store:
            if var in self.buffer_offsets and V.graph.sizevars.statically_known_geq(
                offset, self.buffer_offsets[var]
            ):
                # reuse the existing offset to avoid needing an input alias
                self.apply_offset_to_dimension(dims, offset - self.buffer_offsets[var])
                offset = self.buffer_offsets[var]
            elif V.graph.sizevars.statically_known_gt(
                offset, 0
            ):  # TODO(jansel): negative offsets
                # roll the offset into the dimensions for cleaner indexing
                self.apply_offset_to_dimension(dims, offset)
                offset = 0

        orig_var = var
        for i in itertools.count():
            if self.install_dims(var, dims, offset, is_store):
                return var, dims
            if is_store:
                raise AssertionError("install_dims must succeed for stores")
            var = f"{orig_var}_view{i}"
            if var not in self.buffer_aliases[orig_var]:
                self.buffer_aliases[orig_var].append(var)

    def install_dims(self, var, dims, offset, is_store):
        # [FUNCTION DOC] Registers dims/offset for buffer var. A new buffer is
        # stored as is. For an existing one, returns False on offset or rank
        # mismatch; for stores requires identical dims; for loads requires equal
        # strides and widens sizes to the maximum (clearing expr when they
        # differ). Returns True on success; mutates buffer_dimensions/offsets and shared dims.
        """Try to set self.buffer_dimensions[var], return True on success"""
        if var not in self.buffer_dimensions:
            self.buffer_dimensions[var] = dims
            self.buffer_offsets[var] = offset
            return True
        if self.buffer_offsets[var] != offset or len(
            self.buffer_dimensions[var]
        ) != len(dims):
            return False
        if is_store:
            return self.buffer_dimensions[var] == dims
        for old, new in zip(self.buffer_dimensions[var], dims):
            if old.stride != new.stride:
                return False
            if old.size != new.size or old.expr != new.expr:
                old.size = V.graph.sizevars.evaluate_max(old.size, new.size)
                old.expr = None
        return True

    def apply_offset_to_dimension(self, dims, offset):
        # [FUNCTION DOC] Folds a constant element offset into the dims in place,
        # from the largest stride down, by adding the quotient of offset by stride
        # to that dimension's expr. Raises AssertionError if a remainder is left over.
        if offset == 0:
            return
        for i in reversed(range(len(dims))):
            if dims[i].stride == 1 or V.graph.sizevars.statically_known_geq(
                offset, dims[i].stride
            ):
                part = FloorDiv(offset, dims[i].stride)
                offset -= part * dims[i].stride
                dims[i].expr += part
        if offset != 0:
            raise AssertionError(f"offset must be fully applied, got {offset}")

    def used_dims_from_index(self, index: sympy.Expr):
        # [FUNCTION DOC] Returns the ordered halide dims an index expression
        # depends on: halide symbols directly, and for temp symbols the dims of
        # the referenced CSE variable; size/index/unbacked symbols are ignored.
        # Raises AssertionError for non-symbol free symbols or CSE variables with
        # unknown dims, and NotImplementedError for other symbol kinds.
        """Detect which range trees are used to populate HalideCSEVariable.used_dims"""
        used_dims = OrderedSet[sympy.Symbol]()
        for sym in index.free_symbols:
            if not isinstance(sym, sympy.Symbol):
                raise AssertionError(f"expected sympy.Symbol, got {type(sym)}")
            if symbol_is_type(sym, SymT.TMP):
                # indirect indexing
                cse_var = self.lookup_cse_var(sym.name)
                if not (
                    isinstance(cse_var, HalideCSEVariable)
                    and cse_var.used_dims is not None
                ):
                    raise AssertionError(
                        "cse_var must be HalideCSEVariable with used_dims"
                    )
                used_dims.update(cse_var.used_dims)
            elif symbol_is_type(sym, SymT.HALIDE):
                used_dims.add(sym)
            elif symbol_is_type(
                sym, (SymT.UNBACKED_INT, SymT.SIZE, SymT.PRECOMPUTED_SIZE, SymT.INDEX)
            ):
                pass
            else:
                raise NotImplementedError(f"unhandled symbol {sym}")
        return self.sort_used_dims(used_dims)

    def sort_used_dims(self, used_dims):
        # [FUNCTION DOC] Orders the given dims by their position among halide
        # vars followed by reduction vars. Raises AssertionError if any is not a
        # symbolic expression or is not a known dim.
        if not all(isinstance(x, sympy.Expr) for x in used_dims):
            raise AssertionError("used_dims must all be sympy expressions")
        ordered = [
            sym
            for sym in itertools.chain(
                self.halide_vars, self.reduction_renames.values()
            )
            if sym in used_dims
        ]
        if len(ordered) != len(used_dims):
            raise AssertionError(
                f"expected {len(used_dims)} ordered dims, got {len(ordered)}"
            )
        return ordered

    def make_index_str(self, dims, replacements=None, zero_vars=False):
        # [FUNCTION DOC] Joins the dimension index texts into a subscript: empty
        # parentheses for no dims, and a trailing comma for one dim so it forms a tuple.
        index_str = ", ".join(d.index_str(replacements, zero_vars) for d in dims)
        if len(dims) == 0:
            index_str = "()"
        elif len(dims) == 1:
            # workaround for https://github.com/halide/Halide/issues/8299
            index_str = f"{index_str},"
        return index_str

    def load(self, name: str, index: sympy.Expr):
        # [FUNCTION DOC] Generates a load of input buffer name at index. Half
        # types are cast up to fp32. Without a load mask, returns a CSE function
        # of the buffer access. With a mask, builds a function that is the fill
        # value except where the mask holds (via a one-element reduction domain
        # with a where condition, or a select for scalars). Side effects: registers
        # the input arg and buffer dims, and writes lines into the body.
        # Raises AssertionError if the mask is not a Halide CSE variable with known dims.
        """Codegen a load from an InputBuffer"""
        var = self.args.input(name)
        index = self.prepare_indexing(index)
        var, dims = self.indexing_to_dimensions(var, index, False)
        line = f"{var}[{self.make_index_str(dims)}]"
        dtype = V.graph.get_dtype(name)
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
            line = f"hl.cast(hl.Float(32), {line})"

        if self._load_mask:
            if not (
                isinstance(self._load_mask, HalideCSEVariable)
                and self._load_mask.used_dims is not None
            ):
                raise AssertionError(
                    "_load_mask must be HalideCSEVariable with used_dims"
                )
            used_dims = OrderedSet(
                (*self.used_dims_from_index(index), *self._load_mask.used_dims)
            )
            result = self.newfunc(self.sort_used_dims(used_dims))
            if result.used_dims:
                self.body.writeline(f"{result.name}_mask = hl.RDom([hl.Range(0, 1)])")
                self.body.writeline(f"{result.name}_mask.where({self._load_mask})")
                other = self.kexpr(self._load_other or 0)  # type: ignore[arg-type]
                self.body.writeline(
                    f"{result} = hl.cast({halide_type(dtype)}, {other})"
                )
                self.body.writeline(
                    f"{result} = {line} + hl.cast({halide_type(dtype)}, {result.name}_mask)"
                )
            else:
                # scalar case
                self.body.writeline(
                    f"{result} = hl.select({self._load_mask}, {line}, hl.cast({halide_type(dtype)}, 0))"
                )
            return result
        else:
            return self.genfunc(line, self.used_dims_from_index(index))

    def lookup_cse_var(self, name: str):
        # [FUNCTION DOC] Returns the CSE variable for a name, ignoring any
        # trailing bracketed index suffix. KeyError if unknown.
        return self.cse.varname_map[re.sub(r"\[.*", "", name)]

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        # [FUNCTION DOC] Generates a store of value (a Halide CSE variable) to
        # output buffer name at index. Plain stores assign directly with zero
        # placeholders for unused dims. With indirect indexing or a store mode, uses
        # explicit-domain renames: first an undefined initialization of the buffer is
        # written, then the update. mode None assigns a cast value; "atomic_add"
        # accumulates; other modes raise NotImplementedError. Side effects:
        # registers the output arg and dims, appends deferred lines to the body.
        """Codegen a store to an OutputBuffer"""
        if not isinstance(value, HalideCSEVariable):
            raise AssertionError(f"expected HalideCSEVariable, got {type(value)}")
        var = self.args.output(name)
        index = self.prepare_indexing(index)
        var, dims = self.indexing_to_dimensions(var, index, True)
        if self.is_indirect_indexing(index) or mode is not None:
            replacements = self.setup_dom_indexing()
            index_str = self.make_index_str(dims, replacements)
            value_str = value.subs_str(replacements)
            undef_dims = (", ".join(["hl.Var()"] * len(dims))) or "()"
            self.body.writeline(
                DeferredLine(name, f"{var}[{undef_dims}] = hl.undef({var}.type())")
            )
        else:
            index_str = self.make_index_str(dims, zero_vars=True)
            value_str = str(value)

        dtype = V.graph.get_dtype(name)
        if mode is None:
            line = f"{var}[{index_str}] = hl.cast({halide_type(dtype)}, {value_str})"
        elif mode == "atomic_add":
            line = f"{var}[{index_str}] += hl.cast({halide_type(dtype)}, {value_str})"
        else:
            raise NotImplementedError(f"store mode={mode}")
        self.body.writeline(DeferredLine(name, line))

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: CSEVariable | tuple[CSEVariable, ...],
    ) -> CSEVariable | tuple[CSEVariable, ...]:
        # [FUNCTION DOC] Generates a reduction of value over the reduction
        # domain, memoized in the CSE reduction cache. A tuple input is only
        # allowed for welford combine. For argmax/argmin, emits the domain
        # arg-reduction and flattens the multi-dimensional result into a linear
        # index; welford_reduce falls back to the generic implementation; other
        # types emit a default-initialized accumulator and an update using the
        # combine function. Raises AssertionError if not inside a reduction or
        # a load mask is active, or the value has unknown dims.
        """Codegen a reduction operation"""
        if not self.inside_reduction:
            raise AssertionError("reduction called outside reduction")
        if self._load_mask:
            raise AssertionError("reduction with active load mask")
        cache_key = (src_dtype, reduction_type, value)
        if cache_key in self.cse.reduction_cache:
            return self.cse.reduction_cache[cache_key]

        if isinstance(value, tuple):
            if reduction_type != "welford_combine":
                raise AssertionError(f"expected welford_combine, got {reduction_type}")
            self.cse.reduction_cache[cache_key] = result_tuple = (
                self.welford_combine_impl(*value)
            )
            return result_tuple

        if not (isinstance(value, HalideCSEVariable) and value.used_dims is not None):
            raise AssertionError("value must be HalideCSEVariable with used_dims")
        reduction_vars = OrderedSet(self.reduction_renames)
        result_var = self.newfunc(
            [v for v in value.used_dims if v not in reduction_vars],
        )
        if reduction_vars - OrderedSet(value.used_dims):
            value = self.genfunc(
                f"{value}",
                self.sort_used_dims(OrderedSet((*value.used_dims, *reduction_vars))),
                shape=value.shape,
            )
        value_str = value.subs_str(self.reduction_renames)
        default = ir.Reduction.default_accumulator(reduction_type, src_dtype)
        acc_type = halide_acc_type(dtype)

        if reduction_type in ("argmax", "argmin"):
            index = f"{result_var.name}_{reduction_type}"
            self.body.writeline(f"{index} = hl.{reduction_type}(rdom, {value_str})")
            # turn the N-D argmax index into a 1-D one
            parts = []
            stride = 1
            for i, sym in enumerate(self.reduction_renames):
                # pyrefly: ignore [bad-argument-type]
                parts.append(f"{index}[{i}]")
                if stride != 1:
                    # pyrefly: ignore [unsupported-operation]
                    parts[-1] += f"*{stride}"
                stride *= self.halide_vars[sym]
            self.body.writeline(f"{result_var} = {' + '.join(parts)}")
        elif reduction_type == "welford_reduce":
            # TODO(jansel): implement welford_reduce without fallback
            result_var = self.welford_reduce_fallback(dtype, value)
        else:
            combine_fn = get_reduction_combine_fn(reduction_type, acc_type)
            with V.set_ops_handler(AddParenHandler(HalideOverrides())):
                combine_str = combine_fn(result_var, value_str)  # type: ignore[arg-type]
            default_str = f"hl.cast({acc_type}, {halide_constant(default)})"
            self.body.writeline(f"{result_var} = {default_str}")
            self.body.writeline(f"{result_var} = {combine_str}")

        self.cse.reduction_cache[cache_key] = result_var
        return result_var

    def welford_combine_impl(self, mean, m2, weight):
        # [FUNCTION DOC] Emits a Welford combine over the reduction domain for
        # the given mean, m2 and weight variables: a tuple-valued function
        # starting at zeros whose update merges each reduction element using the
        # parallel-variance formula (guarding zero total weight). Returns three
        # new variables unpacking the final mean, m2 and weight. Raises
        # AssertionError if an input is not a Halide CSE variable with known dims.
        if not (isinstance(mean, HalideCSEVariable) and mean.used_dims is not None):
            raise AssertionError("mean must be HalideCSEVariable with used_dims")
        if not (isinstance(m2, HalideCSEVariable) and m2.used_dims is not None):
            raise AssertionError("m2 must be HalideCSEVariable with used_dims")
        if not (isinstance(weight, HalideCSEVariable) and weight.used_dims is not None):
            raise AssertionError("weight must be HalideCSEVariable with used_dims")
        used_dims = OrderedSet(
            (*mean.used_dims, *m2.used_dims, *weight.used_dims) or self.halide_vars
        )
        used_dims -= OrderedSet(self.reduction_renames)
        result_var = self.newfunc(self.sort_used_dims(used_dims))
        default = [f"hl.cast({x.name}.type(), 0)" for x in (mean, m2, weight)]
        pfx = result_var.name
        self.body.writeline(f"{result_var} = hl.Tuple([{', '.join(default)}])")
        self.body.writeline(f"{pfx}_mean_1 = {result_var}[0]")
        self.body.writeline(f"{pfx}_m2_1 = {result_var}[1]")
        self.body.writeline(f"{pfx}_weight_1 = {result_var}[2]")
        self.body.writeline(f"{pfx}_mean_2 = {mean.subs_str(self.reduction_renames)}")
        self.body.writeline(f"{pfx}_m2_2 = {m2.subs_str(self.reduction_renames)}")
        self.body.writeline(
            f"{pfx}_weight_2 = {weight.subs_str(self.reduction_renames)}"
        )
        self.body.writeline(f"{pfx}_delta = {pfx}_mean_2 - {pfx}_mean_1")
        self.body.writeline(f"{pfx}_new_weight = {pfx}_weight_1 + {pfx}_weight_2")
        self.body.writeline(
            f"{pfx}_w2_over_w = hl.select({pfx}_new_weight == 0.0, 0.0, {pfx}_weight_2 / {pfx}_new_weight)"
        )
        update = [
            f"{pfx}_mean_1 + {pfx}_delta * {pfx}_w2_over_w",
            f"{pfx}_m2_1 + {pfx}_m2_2 + {pfx}_delta * {pfx}_delta * {pfx}_weight_1 * {pfx}_w2_over_w",
            f"{pfx}_new_weight",
        ]
        self.body.writeline(f"{result_var} = hl.Tuple([{', '.join(update)}])")

        unpacked = []
        for i in range(3):
            unpacked.append(self.newfunc(result_var.used_dims))
            self.body.writeline(f"{unpacked[-1]} = {result_var}[{i}]")
        return tuple(unpacked)

    def scan(
        self,
        dtypes: tuple[torch.dtype, ...],
        combine_fn: Callable[
            [tuple[CSEVariable, ...], tuple[CSEVariable, ...]], tuple[CSEVariable, ...]
        ],
        values_orig: tuple[CSEVariable, ...],
    ) -> tuple[CSEVariable, ...]:
        # [FUNCTION DOC] Generates an inclusive scan (prefix combine) along the
        # single reduction dimension. Initializes the result from the (accumulator
        # typed) input values, then emits an update over a domain starting at
        # index one that combines the previous element with the current one using
        # combine_fn evaluated under a non-CSE handler. Returns the result
        # variable(s); multiple values are packed in a tuple function and unpacked.
        # Raises AssertionError outside a reduction, on dtype/value count
        # mismatch, with multi-dimensional scans, or invalid variables.
        # KNOWN ISSUE: for values not depending on the reduction dim, the dims list
        # gets a nested list appended instead of the first reduction var itself.
        if not self.inside_reduction:
            raise AssertionError("scan called outside reduction")
        if len(dtypes) != len(values_orig):
            raise AssertionError(
                f"expected {len(values_orig)} dtypes, got {len(dtypes)}"
            )
        values: list[HalideCSEVariable] = []
        all_used_dims = OrderedSet[sympy.Symbol]()

        for value in values_orig:
            if not (
                isinstance(value, HalideCSEVariable) and value.used_dims is not None
            ):
                raise AssertionError("value must be HalideCSEVariable with used_dims")
            if OrderedSet(value.used_dims) & OrderedSet(self.reduction_renames):
                values.append(value)
            else:
                values.append(
                    self.genfunc(
                        f"{value}",
                        [*value.used_dims, [*self.reduction_renames][:1]],
                        shape=value.shape,
                    )
                )
            all_used_dims.update(value.used_dims)
        result_var = self.newfunc(self.sort_used_dims(all_used_dims))
        if not (
            result_var.used_dims
            and OrderedSet(result_var.used_dims) & OrderedSet(self.reduction_renames)
        ):
            raise AssertionError("result_var must use reduction dims")
        initial = [
            f"hl.cast({halide_acc_type(dtype)}, {value})"
            for dtype, value in zip(dtypes, values)
        ]

        length = self.kexpr(self.rename_indexing(self.range_trees[-1].numel))
        scan_dom = f"{result_var.name}_rdom"
        scan = f"{scan_dom}.x"
        self.body.writeline(f"{scan_dom} = hl.RDom([hl.Range(1, {length})])")

        if len(self.reduction_renames) != 1:
            raise AssertionError("multi-dimensional scan not implemented")
        (scan_var,) = [*self.reduction_renames]  # type: ignore[misc]
        scan_renames_cur = {scan_var: sympy_index_symbol(scan)}
        scan_renames_pri = {scan_var: sympy_index_symbol(scan) - 1}

        if len(values) == 1:

            def maybe_tuple(x):
                # [FUNCTION DOC] HELPER: Single-value scan case; returns the only element.
                return x[0]

            read_left = [result_var.subs_str(scan_renames_pri)]
            read_right = [result_var.subs_str(scan_renames_cur)]
        else:

            def maybe_tuple(x):
                # [FUNCTION DOC] HELPER: Multi-value scan case; wraps the elements in a Halide tuple expression.
                return f"hl.Tuple([{', '.join(x)}])"

            read_left = [
                result_var.subs_str(scan_renames_pri) + f"[{i}]"
                for i in range(len(values))
            ]
            read_right = [
                result_var.subs_str(scan_renames_cur) + f"[{i}]"
                for i in range(len(values))
            ]

        self.body.writeline(f"{result_var} = {maybe_tuple(initial)}")

        # Disable CSE for update fn
        with V.set_ops_handler(AddParenHandler(HalideOverrides())):
            combine_str = combine_fn(read_left, read_right)  # type: ignore[arg-type]
        self.body.writeline(
            f"{result_var.subs_str(scan_renames_cur)} = {maybe_tuple(combine_str)}"
        )

        if len(values) == 1:
            return (result_var,)

        unpack_vars = [self.newfunc(self.sort_used_dims(all_used_dims)) for _ in values]
        for i, v in enumerate(unpack_vars):
            self.body.writeline(f"{v} = {result_var}[{i}]")
        return tuple(unpack_vars)

    def genfunc(
        self,
        line,
        used_dims,
        *,
        bounds=ValueRanges.unknown(),
        shape: BlockShapeType = None,
    ) -> HalideCSEVariable:
        # [FUNCTION DOC] Generates a CSE variable for source text line in the
        # body (with value bounds and shape), then records used_dims on it.
        # Returns it. Raises AssertionError if CSE does not produce a Halide variable.
        var = self.cse.generate(self.body, line, bounds=bounds, shape=shape)
        if not isinstance(var, HalideCSEVariable):
            raise AssertionError(f"expected HalideCSEVariable, got {type(var)}")
        var.used_dims = used_dims
        return var

    def newfunc(self, used_dims, *, shape: BlockShapeType = None) -> HalideCSEVariable:
        # [FUNCTION DOC] Allocates a fresh CSE variable name (no code emitted)
        # and records used_dims on it. Raises AssertionError if the CSE does not
        # produce a Halide variable.
        var = self.cse.newvar(shape=shape)
        if not isinstance(var, HalideCSEVariable):
            raise AssertionError(f"expected HalideCSEVariable, got {type(var)}")
        var.used_dims = used_dims
        return var

    def halide_buffer_numel(self, name: str):
        # [FUNCTION DOC] Element count to pass to Halide for graph buffer name:
        # the layout's storage size (includes gaps), since all tensors are treated as 1D buffers.
        """
        We map all tensors to 1D buffers in Halide since Halide has trouble representing some strides that PyTorch
        supports.  If there are gaps in the underlying layout the numel we pass to Halide includes the gaps while
        PyTorch's numel excludes them.
        """
        return V.graph.get_buffer(name).get_layout().storage_size()

    def halide_argdefs(self):
        # [FUNCTION DOC] Returns the kernel arguments as (call string, arg)
        # pairs ordered inputs, then scalar sizes, then outputs (Halide wants
        # scalars before outputs). Each tensor arg is followed by extra entries
        # (no call string) for its view aliases recorded in buffer_aliases.
        # Raises AssertionError for unexpected arg names or aliased/offset tensor args.
        """
        Halide requires scalar inputs before outputs, so need to reorder args.
        """

        def arg_order(arg_tuple):
            # [FUNCTION DOC] HELPER: Sort key: tensor inputs 0, sizes 1, outputs 2.
            # Raises AssertionError if a tensor arg is neither input nor output.
            _call_str, arg = arg_tuple
            if isinstance(arg, SizeArg):
                return 1  # this would normally be at the end, move it to middle
            elif "out_ptr" in arg.name:
                return 2
            else:
                if "in_ptr" not in arg.name:
                    raise AssertionError(f"expected in_ptr in arg name, got {arg.name}")
                return 0

        result: list[tuple[str | None, KernelArgType]] = []
        _, a, b, _ = self.args.python_argdefs()
        for call_str, arg in sorted(zip(a, b), key=arg_order):
            result.append((call_str, arg))
            if isinstance(arg, TensorArg):
                if not (arg.offset == 0 and arg.alias_of is None):
                    raise AssertionError("TensorArg must have zero offset and no alias")
                result.extend(
                    (
                        None,
                        TensorArg(
                            alias,
                            arg.buffer,
                            arg.dtype,
                            arg.offset,
                            alias_of=arg.name,
                        ),
                    )
                    for alias in self.buffer_aliases.get(arg.name, ())
                )
        return result

    def halide_kernel_meta(self) -> HalideMeta:
        # [FUNCTION DOC] Builds the HalideMeta needed by the code cache: an input
        # spec per argument (shape/stride/offset strings for tensors, a long
        # scalar for sizes, alias info), plus the target string and scheduler
        # settings from config. CPU uses the CPU target and thread count; CUDA
        # (default device only) uses the GPU target, adds a compute capability
        # feature if not given, and the multiprocessor count. Always adds strict
        # float and no_runtime; optional no_asserts, debug, and large_buffers for
        # 64-bit indexing. Raises AssertionError for unsupported devices.
        """Compute metadata required by codecache.py"""
        argtypes = []
        for _, arg in self.halide_argdefs():
            if isinstance(arg, SizeArg):
                shape = None
                stride = None
                offset = None
                dtype = "long"
            else:
                shape = [
                    cexpr(self.rename_indexing(x.size))
                    for x in self.buffer_dimensions[arg.name]
                ]
                stride = [
                    cexpr(self.rename_indexing(x.stride))
                    for x in self.buffer_dimensions[arg.name]
                ]
                if len(shape) != len(stride):
                    raise AssertionError(
                        f"shape/stride length mismatch: {len(shape)} vs {len(stride)}"
                    )
                offset = cexpr(self.buffer_offsets[arg.name])
                dtype = f"{DTYPE_TO_CPP[arg.dtype]}*"
            argtypes.append(
                HalideInputSpec(
                    dtype,
                    arg.name,
                    shape=shape,
                    stride=stride,
                    offset=offset,
                    alias_of=arg.alias_of,
                )
            )

        current_device = V.graph.get_current_device_or_throw()
        if current_device.type == "cpu":
            target = [config.halide.cpu_target]
            scheduler = config.halide.scheduler_cpu
            scheduler_flags = {
                "parallelism": parallel_num_threads(),
            }
            cuda_device = None
        else:
            if current_device.type != "cuda":
                raise AssertionError("only cpu/cuda supported")
            if current_device.index > 0:
                raise AssertionError("only default device supported")
            target = [config.halide.gpu_target]
            scheduler = config.halide.scheduler_cuda
            capability = torch.cuda.get_device_properties(current_device)
            if "cuda_capability" not in target[0]:
                for major, minor in [(8, 6), (8, 0), (7, 5), (7, 0), (6, 1)]:
                    if capability.major >= major and capability.minor >= minor:
                        target.append(f"cuda_capability_{major}{minor}")
                        break
            target.append("user_context")
            scheduler_flags = {
                "parallelism": capability.multi_processor_count,
                # TODO(jansel): explore other flags, see:
                # grep parser.parse ~/Halide/src/autoschedulers/anderson2021/AutoSchedule.cpp
            }
            cuda_device = max(0, current_device.index)

        # strict_float is required for correctness
        target.append("strict_float")

        # without this we will initialize cuda once per kernel and hit errors
        target.append("no_runtime")

        if not config.halide.asserts:
            target.append("no_asserts")

        if config.halide.debug:
            target.append("debug")

        if "64" in self.index_dtype:
            # TODO(jansel): it is unclear if this does anything, since input sizes are still int32
            target.append("large_buffers")

        return HalideMeta(
            argtypes,
            target="-".join(target),
            scheduler=scheduler,
            scheduler_flags=scheduler_flags,  # type: ignore[arg-type]
            cuda_device=cuda_device,
        )

    def codegen_kernel(self, name=None):
        # [FUNCTION DOC] Produces the final kernel source string: a Halide
        # generator class with one input/output declaration per argument,
        # a generate method that binds args, applies aliases, splices indexing code
        # and the body (filling in unresolved dim placeholders), then sets size
        # estimates (and fixed min/stride/extent for inputs when integral),
        # and a main section that either builds an autoscheduled pipeline (when a
        # scheduler is configured) or compiles directly. Raises Unsupported if
        # in-place buffers are used.
        """Called at the end to generate a final kernel string"""
        if self.args.inplace_buffers:
            raise Unsupported("inplace_buffers")
        meta = self.halide_kernel_meta()  # ensure needed args are added early
        code = IndentedBuffer()
        code.splice(
            """
            import halide as hl
            from torch._inductor.runtime import halide_helpers
            from math import inf, nan

            @hl.generator(name="kernel")
            class Kernel:
        """,
            strip=True,
        )
        code.do_indent()
        for _, arg in self.halide_argdefs():
            if isinstance(arg, SizeArg):
                code.writeline(f"{arg.name} = hl.InputScalar({self.index_dtype})")
            else:
                if not arg.buffer:
                    raise AssertionError(arg)
                argcls = "hl.OutputBuffer" if "out" in arg.name else "hl.InputBuffer"
                argtype = halide_type(arg.dtype)
                ndim = len(self.buffer_dimensions[arg.name])
                code.writeline(f"{arg.name} = {argcls}({argtype}, {ndim})")
        code.splice(
            """
            def generate(g):
        """
        )
        code.do_indent()
        for _, arg in self.halide_argdefs():
            code.writeline(f"{arg.name} = g.{arg.name}")
        for old, new in self.args.aliases():
            code.writeline(f"{old} = {new}")
        code.splice(self.indexing_code)

        def update_index(m):
            # [FUNCTION DOC] HELPER: Regex replacement callback; maps a matched
            # placeholder to the final indexed string of its CSE variable.
            # Raises AssertionError if that variable's dims are still unknown.
            var = cast(HalideCSEVariable, self.cse.varname_map[m.group(1)])
            if var.used_dims is None:
                raise AssertionError(var)
            return str(var)

        for line in self.body._lines:
            if isinstance(line, str):
                # fill in missing indices
                line = HalideCSEVariable.undefined_re.sub(update_index, line)
            code.writeline(line)
        code.writeline("")
        code.writeline("assert g.using_autoscheduler()")

        for _, arg in self.halide_argdefs():
            # fallback=1 below because halide requires buffers to be at least as large as the estimates
            # This causes crashes if our estimate is greater than the vector length
            # https://github.com/halide/Halide/issues/3103
            if isinstance(arg, SizeArg):
                hint = V.graph.sizevars.optimization_hint(arg.expr, fallback=1)
                code.writeline(f"{arg.name}.set_estimate({hint})")
            else:
                dims = self.buffer_dimensions[arg.name]
                range_hints = []
                for i, dim in enumerate(dims):
                    hint = self._autoscheduler_workarounds(
                        V.graph.sizevars.optimization_hint(dim.size, fallback=1), dims
                    )
                    # pyrefly: ignore [bad-argument-type]
                    range_hints.append(f"hl.Range(0, {hint})")
                    if "out" not in arg.name:
                        code.writeline(f"{arg.name}.dim({i}).set_min(0)")
                        try:
                            code.writeline(
                                f"{arg.name}.dim({i}).set_stride({int(dim.stride)})"
                            )
                        except TypeError:
                            pass  # not integer
                        try:
                            code.writeline(
                                f"{arg.name}.dim({i}).set_extent({int(dim.size)})"
                            )
                        except TypeError:
                            pass  # not integer
                code.writeline(f"{arg.name}.set_estimates([{', '.join(range_hints)}])")

        code.do_unindent(2)
        code.splice(
            """
            if __name__ == "__main__":
                hl.main()
            """.rstrip(),
        )
        if meta.scheduler:
            code.splice(
                f"""
                else:
                    hl.load_plugin({HalideCodeCache.find_libautoschedule(meta.scheduler)!r})
                    target = hl.Target({meta.target!r})
                    autoscheduler = hl.AutoschedulerParams({meta.scheduler!r}, {meta.scheduler_flags!r})
                    with hl.GeneratorContext(target, autoscheduler):
                        gen = Kernel()
                        pipeline = gen._build_pipeline()
                        # gen.compile_to_callable() does not run the autoscheduler
                        pipeline.apply_autoscheduler(target, autoscheduler)
                        kernel = pipeline.compile_to_callable([
                                gen._get_input_parameter(a.name)._to_argument()
                                for a in gen._get_arginfos()
                                if a.dir == hl.ArgInfoDirection.Input
                            ], target)
                """,
                strip=True,
            )
        else:
            code.splice(
                f"""
                  else:
                      with hl.GeneratorContext(hl.Target({meta.target!r})):
                          kernel = Kernel().compile_to_callable()
                  """,
                strip=True,
            )
        return code.getvalue()

    @staticmethod
    def _autoscheduler_workarounds(n, dims):
        # [FUNCTION DOC] HELPER: Adjusts a dimension size estimate; for 1-D
        # buffers on CUDA with the Anderson2021 scheduler, raises the estimate to at least 2.
        if (
            len(dims) == 1
            and config.halide.scheduler_cuda == "Anderson2021"
            and V.graph.get_current_device_or_throw().type == "cuda"
        ):
            # workaround https://github.com/halide/Halide/issues/8246
            n = max(2, n)
        return n

    def call_kernel(self, name: str, node=None, deallocate_ws: bool = True):
        # [FUNCTION DOC] Emits a call to kernel name in the wrapper code,
        # passing all non-alias arguments (plus the raw CUDA stream for CUDA
        # devices). node and deallocate_ws are unused. Side effect: writes to the wrapper.
        """Codegen a call to this kernel"""
        wrapper = V.graph.wrapper_code
        call_args = [f"{n}" for n, arg in self.halide_argdefs() if arg.alias_of is None]
        current_device = V.graph.get_current_device_or_throw()
        if current_device.type == "cuda":
            stream_name = wrapper.write_get_raw_stream(
                current_device.index, V.graph.name
            )
            call_args.append(stream_name)
        wrapper.generate_kernel_call(
            name,
            call_args,
            device=current_device,
            triton=False,
        )

    def generate_assert(self, check):
        # [FUNCTION DOC] Asserts are unsupported; always returns False (nothing generated).
        return False  # TODO(jansel): support asserts

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ):
        # [FUNCTION DOC] Bounds checks are unsupported; does nothing.
        pass  # TODO(jansel): support asserts


class HalideScheduling(SIMDScheduling):
    # [CLASS DOC] Scheduling backend for Halide: reuses SIMD scheduling with
    # HalideKernel as the kernel type, advertises its feature set, and defines
    # kernels by registering Halide source with the wrapper's async compile.
    kernel_type = HalideKernel  # type: ignore[arg-type,assignment]

    @classmethod
    def get_backend_features(cls, device: torch.device) -> OrderedSet[BackendFeature]:
        # [FUNCTION DOC] Returns the set of backend features: tuple reductions,
        # store-loop-order preference, reduce-to-single-element, plus scan
        # support only when config enables Halide scan kernels. device is unused.
        result = OrderedSet(
            [
                BackendFeature.TUPLE_REDUCTION,
                BackendFeature.PREFER_STORE_LOOP_ORDER,
                BackendFeature.REDUCE_TO_SINGLE_ELEMENT,
            ]
        )
        if config.halide.scan_kernels:
            result.add(BackendFeature.SCAN)
        return result

    def define_kernel(self, src_code, node_schedule, kernel):
        # [FUNCTION DOC] Registers kernel source in the wrapper, deduplicating
        # by source text. For new source: picks a fresh kernel name, adds the
        # needed import, writes an async-compile call carrying the kernel's
        # Halide metadata and the source, defines the kernel with origin
        # metadata comments, and logs kernel metadata if that metric table is
        # on. Returns the kernel name (new or previously registered).
        """Codegen kernel definition to go in output wrapper code"""
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            kernel_name = f"halide_kernel_{wrapper.next_kernel_suffix()}"
            wrapper.src_to_kernel[src_code] = kernel_name
            wrapper.add_import_once(
                "from torch._inductor.runtime.hints import HalideMeta, HalideInputSpec"
            )

            compile_wrapper = IndentedBuffer()
            compile_wrapper.writeline(
                f"async_compile.halide({kernel.halide_kernel_meta()!r}, '''"
            )
            compile_wrapper.splice(src_code, strip=True)
            compile_wrapper.writeline("''')")

            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment = f"{origins}\n{detailed_origins}"
            wrapper.define_kernel(
                kernel_name, compile_wrapper.getvalue(), metadata_comment
            )
            if is_metric_table_enabled("kernel_metadata"):
                log_kernel_metadata(kernel_name, "", src_code)

        return kernel_name
