# [MODULE DOC] Early-prototype Inductor backend for Apple MPS: generates Metal compute shaders for pointwise and reduction kernels. Defines a sympy-to-Metal
# printer (MetalExprPrinter), op overrides (MetalOverrides), the kernel generator (MetalKernel) and scheduling (MetalScheduling). No import-time side
# effects other than initializing pointwise and special op overrides on MetalOverrides.
# This is not a feature-complete compiler backend
# Just an early prototype that shows that one can compile elementwise ops into a Metal shader
from __future__ import annotations

import functools
import itertools
import logging
import math
from pathlib import Path
from typing import Any, TYPE_CHECKING

import sympy
from sympy.printing.precedence import PRECEDENCE

import torch
from torch.utils._cpp_embed_headers import _embed_headers
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import Min
from torch.utils._sympy.printers import CppPrinter, ExprPrinter as ExprPrinter_
from torch.utils._sympy.value_ranges import ValueRanges

from ..utils import ceildiv, get_bounds_index_expr, get_kernel_metadata
from ..virtualized import ops, OpsWrapper, V
from .common import (
    CSEVariable,
    DeferredLine,
    DTYPE_TO_COMPUTATION_DTYPE,
    IndentedBuffer,
    OpOverrides,
    PythonPrinter,
)
from .simd import IterationRangesEntry, SIMDKernel, SIMDScheduling


if TYPE_CHECKING:
    from ..ops_handler import ReductionType, StoreMode
    from ..scheduler import Scheduler, SchedulerNode
    from .common import OpVarT

log = logging.getLogger(__name__)

DTYPE_TO_METAL = {
    torch.bool: "bool",
    torch.int8: "char",
    torch.int16: "short",
    torch.int32: "int",
    torch.int64: "long",
    torch.uint8: "uchar",
    torch.uint16: "ushort",
    torch.uint32: "uint",
    torch.uint64: "ulong",
    torch.float: "float",
    torch.half: "half",
    torch.bfloat16: "bfloat",
}


def value_to_metal(val: float | int | bool | str | CSEVariable) -> str:
    # [FUNCTION DOC] Renders a Python constant (or CSE variable/str) as Metal source text: infinities map to HUGE_VALF / -HUGE_VALF, NaN to NAN,
    # bools to true/false, anything else via its string form.
    if isinstance(val, float):
        if val == torch.inf:
            return "HUGE_VALF"
        elif val == -torch.inf:
            return "-HUGE_VALF"
        elif val != val:  # Only float that not equal to self is nan
            return "NAN"
        return str(val)
    elif isinstance(val, bool):
        return "true" if val else "false"
    return str(val)


class MetalExprPrinter(ExprPrinter_):
    # [CLASS DOC] Sympy expression printer that emits Metal C++ source for index/size math. Stateless; each _print_* method handles one
    # sympy function type (floor division, modular indexing, min/max, rounding, casts, conditional).
    """Converts sympy expression to Metal code snippet"""

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints floor division; integer-valued uses the c10 metal floor_divide helper, otherwise floor of the numerator divided by the divisor.
        x, div = expr.args
        x = self.doprint(x)
        div = self.doprint(div)
        if expr.is_integer:
            return f"c10::metal::floor_divide({x}, {div})"
        return f"metal::floor({x}) / ({div})"

    def _print_ModularIndexing(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints (x / div) % mod. Integer values use plain integer division, others floor first. When div is 65536 and mod is not a power of two, uses the c10 safe_mod helper to work around a Metal compiler bug.
        x, div, mod = expr.args
        # Workaround for Metal compiler bug with fused (x / A) % B, see PR 175481
        use_safe_mod = div == 65536 and (mod & (mod - 1)) != 0

        x = self.doprint(x)
        if div != 1:
            div = self.doprint(div)
            if expr.is_integer:
                x = f"({x}) / ({div})"
            else:
                x = f"metal::floor({x}) / ({div})"
        mod = self.doprint(mod)
        if use_safe_mod:
            return f"c10::metal::safe_mod({x}, {mod})"
        return f"({x}) % ({mod})"

    def _print_min_max(self, expr: sympy.Expr, fn: str) -> str:
        # [FUNCTION DOC] HELPER: Folds a variadic Min/Max into nested binary metal min/max calls via _print_binary_min_max.
        # pyrefly: ignore [missing-attribute]
        args = list(map(self._print, expr.args))
        result = args[0]
        for arg in args[1:]:
            result = self._print_binary_min_max(result, arg, fn)
        return result

    @staticmethod
    def _print_binary_min_max(a: str, b: str, fn: str) -> str:
        # [FUNCTION DOC] HELPER: Builds a metal min/max call with both operands cast to the common promoted type of their sum.
        typecast_a = f"static_cast<decltype({a}+{b})>({a})"
        typecast_b = f"static_cast<decltype({a}+{b})>({b})"
        return f"metal::{fn}({typecast_a}, {typecast_b})"

    def _print_Min(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints a sympy Min as nested metal min calls.
        return self._print_min_max(expr, "min")

    def _print_Max(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints a sympy Max as nested metal max calls.
        return self._print_min_max(expr, "max")

    def _print_Abs(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints absolute value via metal abs. Raises AssertionError unless exactly one argument.
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"metal::abs({self._print(expr.args[0])})"

    def _print_RoundToInt(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints round-to-nearest-even (metal rint) cast to long. Raises AssertionError unless exactly one argument.
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"static_cast<long>(metal::rint({self._print(expr.args[0])}))"

    def _print_RoundDecimal(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints rounding a float to ndigits via scaling by a power of ten, rint, and unscaling, cast to float. For integer inputs it raises ValueError (after an AssertionError check that ndigits is negative); AssertionError if not 2 args.
        if len(expr.args) != 2:
            raise AssertionError(f"expected 2 args, got {len(expr.args)}")
        number, ndigits = expr.args
        if number.is_integer:
            # ndigits < 0 should have been filtered by the sympy function
            if ndigits >= 0:
                raise AssertionError(f"expected ndigits < 0, got {ndigits}")
            raise ValueError(
                f"For integer inputs, only non-negative ndigits are currently supported, but got {ndigits}."
            )
        number_str = self.parenthesize(number, PRECEDENCE["Mul"])
        return f"static_cast<float>(metal::rint(1e{ndigits} * {number_str}) * 1e{-ndigits})"

    def _print_IntTrueDiv(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints true division of two integer expressions by casting both to float.
        # KNOWN ISSUE: per the in-code TODO, precision is only exact up to 2 to the power 23 because of float32.
        lhs, rhs = expr.args
        # TODO: This is only accurate up to 2**23
        # pyrefly: ignore [missing-attribute]
        return f"static_cast<float>({self._print(lhs)}) / static_cast<float>({self._print(rhs)})"

    def _print_PowByNatural(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints power by casting both operands to float and calling metal precise pow. Raises AssertionError unless 2 args.
        if len(expr.args) != 2:
            raise AssertionError(f"expected 2 args, got {len(expr.args)}")
        x, y = map(self.doprint, expr.args)
        return f"metal::precise::pow(static_cast<float>({x}), static_cast<float>({y}))"

    def _print_ToFloat(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints a cast to float. Raises AssertionError unless 1 arg.
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        x = self.doprint(expr.args[0])
        return f"static_cast<float>({x})"

    def _print_Float(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints a sympy Float; integer-valued floats are printed as plain integers because Metal rejects sympy's integer-valued float form, else the default string.
        if expr.is_integer:
            # sympy considers 0.0 to be integer, but Metal doesn't.
            # this workaround prints the float as an integer
            # xref: https://github.com/sympy/sympy/issues/26620
            return str(int(expr))
        else:
            return str(expr)

    def _print_FloorToInt(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints floor of the argument (cast to float first) converted to int. Raises AssertionError unless 1 arg. Also used as the printer for sympy floor.
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        x = self.doprint(expr.args[0])
        return f"static_cast<int>(metal::floor(static_cast<float>({x})))"

    _print_floor = _print_FloorToInt

    def _print_TruncToInt(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints truncation toward zero converted to int. Raises AssertionError unless 1 arg.
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        x = self.doprint(expr.args[0])
        return f"static_cast<int>(metal::trunc({x}))"

    def _print_OpaqueUnaryFn_log2(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints metal precise log2. Raises AssertionError unless 1 arg.
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        x = self.doprint(expr.args[0])
        return f"metal::precise::log2({x})"

    def _print_Where(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints a conditional (ternary) expression from the three sympy arguments, parenthesized as atoms.
        c, p, q = (
            self.parenthesize(arg, PRECEDENCE["Atom"] - 0.5) for arg in expr.args
        )
        return f"{c} ? {p} : {q}"


class MetalOverrides(OpOverrides):
    # [CLASS DOC] Op-handler overrides that map inductor pointwise ops to Metal source expressions (strings). Mostly static methods; they return
    # expression text, and some also emit statements into the active kernel buffers (V.kernel) or register required metal headers.
    # Special-function overrides are attached dynamically by _initialize_special_ops.
    """Implements Metal-specific overrides for ops. Base class emits Python-friendly overrides."""

    @staticmethod
    def to_dtype(
        x: CSEVariable,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> str:
        # [FUNCTION DOC] Emits a static_cast to the Metal type for dtype; float64 requests log a warning and are cast to float instead.
        if dtype == torch.double:
            log.warning(
                "float64 cast requested, probably from tensorify_python_scalars"
            )
            return f"static_cast<float>({x})"
        return f"static_cast<{DTYPE_TO_METAL[dtype]}>({x})"

    @staticmethod
    def to_dtype_bitcast(
        x: CSEVariable, dtype: torch.dtype, src_dtype: torch.dtype
    ) -> str:
        # [FUNCTION DOC] Emits an as_type bitcast to the target dtype from the source dtype.
        return f"as_type<{DTYPE_TO_METAL[dtype]}>(static_cast<{DTYPE_TO_METAL[src_dtype]}>({x}))"

    @staticmethod
    def constant(val: bool | float | int, dtype: torch.dtype) -> str:
        # [FUNCTION DOC] Renders a constant via value_to_metal; dtype is ignored.
        return value_to_metal(val)

    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> str:
        # [FUNCTION DOC] Prints the sympy index expression into a CSE variable on the kernel compute buffer (with bounds info), then casts it to dtype.
        idx_str = V.kernel.index_to_str(V.kernel.prepare_indexing(expr))
        var = V.kernel.cse.generate(
            V.kernel.compute, idx_str, bounds=get_bounds_index_expr(expr)
        )
        return ops.to_dtype(var, dtype)

    @staticmethod
    def value_expr(expr: sympy.Expr, dtype: torch.dtype) -> str:
        # [FUNCTION DOC] Same lowering as index_expr (Metal index_expr already yields the requested dtype).
        # Metal index_expr already emits the requested dtype, so value_expr has
        # the same lowering here.
        return MetalOverrides.index_expr(expr, dtype)

    @staticmethod
    def masked(mask: CSEVariable, body: sympy.Expr, other: CSEVariable) -> str:
        # [FUNCTION DOC] Emits a guarded block: runs body inside an if on mask, assigning to a fresh variable, else assigns the other value; results cached by a key of mask, generated body text and fill value. Temporarily swaps kernel buffers and mutates cse prefix/counter. Returns the CSE variable.
        # KNOWN ISSUE: the cse name_prefix is extended with a suffix and the iter_buffer_ids counter is reset, and neither is restored here; the in-code TODO also notes the annotation of other is wrong.
        # TODO: Type annotation for other is wrong, it's often float or int
        # TODO: Should it be converted to lambda on MacOS-15+?

        other_str = value_to_metal(other)
        scoped_body = IndentedBuffer()
        with V.kernel.swap_buffers(scoped_body), scoped_body.indent():
            # Reset the scoped variable counter so that each invocation of the same body
            # generates identical variable names. Without this reset, repeated calls to
            # body() would keep incrementing the counter, resulting in different cache key.
            V.kernel.cse.iter_buffer_ids = itertools.count()
            # Append "_scoped" to the current prefix so each nesting level gets unique vars
            V.kernel.cse.name_prefix += "_scoped"
            rc = body()

        # Compute cache key manually as variable name is needed to actually generate the code
        cache_key = f"{mask}:{scoped_body.getvalue()}:{other_str}"
        var = V.kernel.cse.try_get(cache_key)
        if not var:
            var = V.kernel.cse.newvar(dtype=rc.dtype)
            V.kernel.cse.put(cache_key, var)
            V.kernel.compute.writelines(
                [f"{DTYPE_TO_METAL[rc.dtype]} {var};", f"if ({mask}) {{"]
            )
            with V.kernel.compute.indent():
                V.kernel.compute.splice(scoped_body)
                V.kernel.compute.writeline(
                    f"{var} = static_cast<decltype({var})>({rc});"
                )
            V.kernel.compute.writeline(
                f"}} else {var} = static_cast<decltype({var})>({other_str});"
            )
        return var

    @staticmethod
    # pyrefly: ignore [bad-override]
    def where(a: OpVarT, b: OpVarT, c: OpVarT) -> str:
        # [FUNCTION DOC] Emits a ternary select; the else value is cast to the type of the true value.
        return f"{a} ? {b} : static_cast<decltype({b})>({value_to_metal(c)})"

    @staticmethod
    def remainder(a: OpVarT, b: OpVarT) -> str:
        # [FUNCTION DOC] Emits the c10 metal remainder helper (Python-style modulo).
        return f"c10::metal::remainder({a}, {b})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def maximum(a: CSEVariable, b: CSEVariable) -> str:
        # [FUNCTION DOC] Emits c10 metal max with both operands cast to a common promoted type.
        typecast_a = f"static_cast<decltype({a}+{b})>({a})"
        typecast_b = f"static_cast<decltype({a}+{b})>({b})"
        return f"c10::metal::max({typecast_a}, {typecast_b})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def minimum(a: CSEVariable, b: CSEVariable) -> str:
        # [FUNCTION DOC] Emits c10 metal min with both operands cast to a common promoted type.
        typecast_a = f"static_cast<decltype({a}+{b})>({a})"
        typecast_b = f"static_cast<decltype({a}+{b})>({b})"
        return f"c10::metal::min({typecast_a}, {typecast_b})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def logical_or(a: CSEVariable, b: CSEVariable) -> str:
        # [FUNCTION DOC] Returns Metal expression text for the logical or op.
        return f"{a} || {b}"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def logical_and(a: CSEVariable, b: CSEVariable) -> str:
        # [FUNCTION DOC] Returns Metal expression text for the logical and op.
        return f"{a} && {b}"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def isnan(x: CSEVariable) -> str:
        # [FUNCTION DOC] Returns Metal expression text for the is-nan op.
        return f"metal::isnan({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def isinf(x: CSEVariable) -> str:
        # [FUNCTION DOC] Returns Metal expression text for the is-inf op.
        return f"metal::isinf({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def log(x: CSEVariable) -> str:
        # [FUNCTION DOC] Returns Metal expression text for the natural log (precise) op.
        return f"metal::precise::log({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def exp(x: CSEVariable) -> str:
        # [FUNCTION DOC] Returns Metal expression text for the exponential (precise) op.
        return f"metal::precise::exp({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def abs(x: CSEVariable) -> str:
        # [FUNCTION DOC] Returns Metal expression text for the absolute value op.
        return f"metal::abs({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def signbit(x: CSEVariable) -> str:
        # [FUNCTION DOC] Returns Metal expression text for the signbit op.
        return f"metal::signbit({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def sin(x: CSEVariable) -> str:
        # [FUNCTION DOC] Returns Metal expression text for the sine (precise) op.
        return f"metal::precise::sin({x})"

    @staticmethod
    def sinc(x: CSEVariable) -> str:
        # [FUNCTION DOC] Returns Metal expression text for the sinc (c10 helper) op.
        return f"c10::metal::sinc({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def cos(x: CSEVariable) -> str:
        # [FUNCTION DOC] Returns Metal expression text for the cosine (precise) op.
        return f"metal::precise::cos({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def tan(x: CSEVariable) -> str:
        # [FUNCTION DOC] Returns Metal expression text for the tangent (precise) op.
        return f"metal::precise::tan({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def asin(x: CSEVariable) -> str:
        # [FUNCTION DOC] Returns Metal expression text for the arcsine (precise) op.
        return f"metal::precise::asin({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def acos(x: CSEVariable) -> str:
        # [FUNCTION DOC] Returns Metal expression text for the arccosine (precise) op.
        return f"metal::precise::acos({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def atan(x: CSEVariable) -> str:
        # [FUNCTION DOC] Returns Metal expression text for the arctangent (precise) op.
        return f"metal::precise::atan({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def atan2(x: CSEVariable, y: CSEVariable) -> str:
        # [FUNCTION DOC] Returns Metal expression text for the two-argument arctangent (precise) op.
        return f"::metal::precise::atan2({x}, {y})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def sqrt(x: CSEVariable) -> str:
        # [FUNCTION DOC] Returns Metal expression text for the square root (precise) op.
        return f"metal::precise::sqrt({x})"

    @staticmethod
    def neg(x: CSEVariable) -> str:
        # [FUNCTION DOC] Emits negation cast back to the operand's own type.
        # KNOWN ISSUE: per the in-code TODO this may rely on undefined behavior for unsigned types.
        # TODO: Does it rely on undefined behavior?
        # If so, add special logic for unsigned types
        return f"static_cast<decltype({x})>(-{x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def rsqrt(x: CSEVariable) -> str:
        # [FUNCTION DOC] Returns Metal expression text for the reciprocal square root (precise) op.
        return f"metal::precise::rsqrt({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def tanh(x: CSEVariable) -> str:
        # [FUNCTION DOC] Returns Metal expression text for the hyperbolic tangent (precise) op.
        return f"metal::precise::tanh({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def atanh(x: CSEVariable) -> str:
        # [FUNCTION DOC] Returns Metal expression text for the inverse hyperbolic tangent (precise) op.
        return f"metal::precise::atanh({x})"

    @staticmethod
    def floordiv(a: CSEVariable, b: CSEVariable) -> str:
        # [FUNCTION DOC] Emits the c10 floor_divide helper; operands must be integers.
        # a and b must be of integer type
        return f"c10::metal::floor_divide({a}, {b})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def floor(x: CSEVariable) -> str:
        # [FUNCTION DOC] Returns Metal expression text for the floor op.
        return f"metal::floor({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def sign(x: CSEVariable) -> str:
        # [FUNCTION DOC] Returns Metal expression text for the sign op.
        return f"metal::sign({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def fmod(a: CSEVariable, b: CSEVariable) -> str:
        # [FUNCTION DOC] Emits metal fmod with both operands cast to a common promoted type.
        typecast_a = f"static_cast<decltype({a}+{b})>({a})"
        typecast_b = f"static_cast<decltype({a}+{b})>({b})"
        return f"metal::fmod({typecast_a}, {typecast_b})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def trunc(x: CSEVariable) -> str:
        # [FUNCTION DOC] Returns Metal expression text for the truncation op.
        return f"metal::trunc({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def truncdiv(a: CSEVariable, b: CSEVariable) -> str:
        # [FUNCTION DOC] Emits division truncated toward zero: metal trunc of the quotient if either operand is floating, else plain integer division.
        quot = f"{a} / {b}"
        if (a.dtype is not None and a.dtype.is_floating_point) or (
            b.dtype is not None and b.dtype.is_floating_point
        ):
            return f"metal::trunc({quot})"
        return quot

    @staticmethod
    # pyrefly: ignore [bad-override]
    def ceil(x: CSEVariable) -> str:
        # [FUNCTION DOC] Returns Metal expression text for the ceiling op.
        return f"metal::ceil({x})"

    @staticmethod
    def rand(seed: CSEVariable, offset: CSEVariable) -> str:
        # [FUNCTION DOC] Emits the c10 metal rand call for seed/offset and registers the random header on the kernel.
        V.kernel.headers.add("random")
        return f"c10::metal::rand({seed}, {offset})"

    @staticmethod
    def randn(seed: CSEVariable, offset: CSEVariable) -> str:
        # [FUNCTION DOC] Emits the c10 metal randn call for seed/offset and registers the random header on the kernel.
        V.kernel.headers.add("random")
        return f"c10::metal::randn({seed}, {offset})"

    @staticmethod
    def randint64(
        seed: CSEVariable, offset: CSEVariable, low: CSEVariable, high: CSEVariable
    ) -> str:
        # [FUNCTION DOC] Emits the c10 metal randint64 call (seed, offset, low, high) and registers the random header on the kernel.
        V.kernel.headers.add("random")
        return f"c10::metal::randint64({seed}, {offset}, {low}, {high})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def round(x: CSEVariable) -> str:
        # [FUNCTION DOC] Emits metal rint (round half to even).
        return f"metal::rint({x})"

    @staticmethod
    def pow(a: CSEVariable, b: CSEVariable) -> str:
        # [FUNCTION DOC] Emits metal pow with both operands cast to a common promoted type.
        cast_a = f"static_cast<decltype({a}+{b})>({a})"
        cast_b = f"static_cast<decltype({a}+{b})>({b})"
        return f"metal::pow({cast_a}, {cast_b})"

    def _special_unary(self, a: CSEVariable, name: str) -> str:
        # [FUNCTION DOC] HELPER: Emits a c10 metal special-math call by name for one argument and registers the special_math header; bound via partialmethod for erf, bessel and similar ops.
        V.kernel.headers.add("special_math")
        return f"c10::metal::{name}({a})"

    def _special_binary(self, a: CSEVariable, b: CSEVariable, name: str) -> str:
        # [FUNCTION DOC] HELPER: Emits a c10 metal special-math call by name for two arguments and registers the special_math header.
        V.kernel.headers.add("special_math")
        return f"c10::metal::{name}({a}, {b})"

    @classmethod
    def _initialize_special_ops(cls) -> None:
        # [FUNCTION DOC] HELPER: Class-level setup that attaches special-math ops (erf family, bessel functions, polygamma, igamma, zeta, orthogonal polynomials, lgamma mapped to log_gamma) as partial methods on the class; names of forward-style functions get a forward suffix. Mutates the class; called once at import.
        # Unary special ops
        for name in [
            "erf",
            "erfc",
            "erfinv",
            "i0",
            "i0e",
            "i1",
            "i1e",
            "digamma",
            "spherical_bessel_j0",
        ]:
            setattr(cls, name, functools.partialmethod(cls._special_unary, name=name))

        cls.lgamma = functools.partialmethod(cls._special_unary, name="log_gamma")  # type: ignore[assignment]

        # Unary special ops with forward in method name
        for name in [
            "bessel_j0",
            "bessel_j1",
            "bessel_y0",
            "bessel_y1",
            "modified_bessel_i0",
            "modified_bessel_i1",
            "modified_bessel_k0",
            "modified_bessel_k1",
            "scaled_modified_bessel_k0",
            "scaled_modified_bessel_k1",
        ]:
            setattr(
                cls,
                name,
                functools.partialmethod(cls._special_unary, name=name + "_forward"),
            )

        # Binary special ops
        for name in [
            "polygamma",
            "igamma",
            "igammac",
            "zeta",
        ]:
            setattr(cls, name, functools.partialmethod(cls._special_binary, name=name))

        # Binary special ops with forward in method name
        for name in [
            "chebyshev_polynomial_t",
            "chebyshev_polynomial_u",
            "chebyshev_polynomial_v",
            "chebyshev_polynomial_w",
            "hermite_polynomial_h",
            "hermite_polynomial_he",
            "shifted_chebyshev_polynomial_t",
            "shifted_chebyshev_polynomial_u",
            "shifted_chebyshev_polynomial_v",
            "shifted_chebyshev_polynomial_w",
        ]:
            setattr(
                cls,
                name,
                functools.partialmethod(cls._special_binary, name=name + "_forward"),
            )


MetalOverrides._initialize_pointwise_overrides("mps")
MetalOverrides._initialize_special_ops()


class MetalKernel(SIMDKernel):
    # [CLASS DOC] SIMDKernel specialization that generates a Metal compute shader. Class attributes: Metal expression printers, max threadgroup size 1024,
    # simd group size 32, set of required c10 metal headers, and the list of multistage reduction entries. Instance state: counter for
    # threadgroup accumulator names. Key methods: load/store/reduction, codegen_body/codegen_kernel (shader source), call_kernel (launch code).
    # KNOWN ISSUE: headers and multistage_reduction_entry are class-level mutable containers shared by all kernels, so headers added by one kernel
    # (for example error or random) persist into later kernels.
    """Implement Metal codegen based on the SIMDKernel abstraction"""

    overrides = MetalOverrides  # type: ignore[assignment]
    suffix = ";"
    newvar_prefix = "auto "
    max_threadgroup_size = 1024
    simd_group_size = 32
    pexpr = PythonPrinter().doprint
    cexpr = CppPrinter().doprint
    sexpr = MetalExprPrinter().doprint
    kexpr = sexpr
    headers: OrderedSet[str] = OrderedSet(["utils"])
    multistage_reduction_entry: list[IterationRangesEntry] = []

    def __init__(
        self,
        tiling: dict[str, sympy.Expr],
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Initializes the base SIMD kernel with the tiling and keyword args, and creates the counter used to name threadgroup accumulator variables.
        super().__init__(tiling, **kwargs)
        self.acc_var_ids = itertools.count()

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        # [FUNCTION DOC] Returns the Metal type name for a torch dtype (KeyError for unsupported dtypes).
        return DTYPE_TO_METAL[dtype]

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        # [FUNCTION DOC] Generates a load from an input buffer into a CSE variable on the loads buffer; half and bfloat values are widened to float32. Returns the variable.
        """Codegen a load from an InputBuffer"""
        var = self.args.input(name)
        index = self.prepare_indexing(index)
        dtype = V.graph.get_dtype(name)
        line = f"{var}[{self.index_to_str(index)}]"
        if dtype in [torch.float16, torch.bfloat16]:
            # TODO(NS): Figure out the right balance between optype casts
            # op_math_t for half-precision floats should be float32
            # Otherwise it can lead to a correctness issues with eager
            line = f"static_cast<float>({line})"
            dtype = torch.float32
        return self.cse.generate(self.loads, line, dtype=dtype)

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        # [FUNCTION DOC] Appends a deferred store line for an output buffer: plain assignment with cast, or an atomic add (registers the atomic header) when mode is atomic_add. Raises RuntimeError for other modes. Goes to the compute buffer inside a reduction, else the stores buffer.
        var = self.args.output(name)
        index = self.prepare_indexing(index)
        dtype_str = self.dtype_to_str(V.graph.get_dtype(name))
        cast_val = f"static_cast<{dtype_str}>({value})"
        if mode is None:
            line = f"{var}[{self.index_to_str(index)}] = {cast_val};"
        elif mode == "atomic_add":
            self.headers.add("atomic")
            atomic_type = f"c10::metal::AtomicType<{dtype_str}>"
            cast_var = f"reinterpret_cast<device {atomic_type}::type *>({var})"
            line = f"{atomic_type}::atomic_add({cast_var}, {self.index_to_str(index)}, {cast_val});"
        else:
            raise RuntimeError(f"Unimplemented store mode {mode}")
        if self.inside_reduction:
            self.compute.writeline(DeferredLine(name, line))
        else:
            self.stores.writeline(DeferredLine(name, line))

    def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable) -> None:
        # [FUNCTION DOC] Appends a deferred store of a reduced value, guarded so only the thread with reduction index zero writes.
        var = self.args.output(name)
        index = self.prepare_indexing(index)
        dtype_str = self.dtype_to_str(V.graph.get_dtype(name))

        reduction_dim = next(t for t in self.range_trees if t.is_reduction)
        # Only one thread in the reduction group needs to store the results
        line = f"{var}[{self.index_to_str(index)}] = static_cast<{dtype_str}>({value});"
        line = f"if ({reduction_dim.name} == 0) {line}"
        self.stores.writeline(DeferredLine(name, line))

    def _new_idxvar(
        self,
        dtype: str | torch.dtype,
        elem_count: int | None = None,
        default_value: Any | None = None,
        is_threadgroup: bool = True,
        bounds: ValueRanges[Any] = ValueRanges.unknown(),
    ) -> CSEVariable:
        # [FUNCTION DOC] HELPER: Declares a new accumulator variable (threadgroup or thread local, optional array size and initial value) in the indexing code and returns it as a CSE variable. Raises AssertionError if a threadgroup variable is given a default value.
        if isinstance(dtype, torch.dtype):
            dtype = self.dtype_to_str(dtype)
        var_name = f"tmp_acc_{next(self.acc_var_ids)}"
        var = V.kernel.create_cse_var(var_name, bounds, dtype)
        var_def = "threadgroup " if is_threadgroup else ""
        var_def += f"{dtype} {var_name}"
        if elem_count:
            var_def += f"[{self.sexpr(elem_count)}]"
        if default_value is not None:
            if is_threadgroup:
                raise AssertionError("Thread group var can not have default value")
            var_def += f" = {default_value}"
        self.indexing_code.writeline(var_def + self.suffix)
        return var

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: CSEVariable | tuple[CSEVariable, ...],
    ) -> CSEVariable | tuple[CSEVariable, ...]:
        # [FUNCTION DOC] Caching wrapper: returns a previously generated result for the same dtype, reduction type and value, else calls _reduction_nocache and caches it.
        "Caching wrapper around _reduction_nocache"
        cache_key = (src_dtype, reduction_type, value)
        # Return cached reduction
        if cache_key in self.cse.reduction_cache:
            return self.cse.reduction_cache[cache_key]
        result = self._reduction_nocache(dtype, src_dtype, reduction_type, value)
        self.cse.reduction_cache[cache_key] = result  # type: ignore[assignment]
        return result

    def _reduction_nocache(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: CSEVariable | tuple[CSEVariable, ...],
    ) -> CSEVariable | tuple[CSEVariable, ...]:
        # [FUNCTION DOC] HELPER: Generates Metal code for a reduction (any, sum, prod, max, min, argmax, argmin, welford_reduce, welford_combine) using threadgroup shared memory and c10 helpers; uses a per-thread accumulator when the reduction is multistage (larger than the threadgroup). Registers the reduction_utils header, writes to indexing/compute/stores buffers, returns the result variable(s). Raises AssertionError if not inside a reduction or a load mask is active, NotImplementedError for other reduction types.
        # KNOWN ISSUE: for welford_combine in the multistage case the zero-initialization of the accumulator is emitted twice (redundant, harmless).
        """Codegen a reduction operation.
        Only sum and prod operations are somewhat reasonable optimized"""
        if not self.inside_reduction:
            raise AssertionError("expected to be inside reduction")
        if self._load_mask:
            raise AssertionError("expected no load mask during reduction")

        def _unwrap_helper(res3: CSEVariable) -> tuple[CSEVariable, ...]:
            # [FUNCTION DOC] HELPER: Splits a float3 result variable into its x, y, z component variables.
            # Unwraps vec3 dtype into individual components
            return OpsWrapper._unwrap(
                [CSEVariable(f"{res3}.{t}", res3.bounds, res3.dtype) for t in "xyz"]
            )

        # Establish reduction buffer size and index expression
        reduction_idx = ""
        acc_buf_size = 1
        for rd in self.range_trees:
            if not rd.is_reduction:
                continue
            if reduction_idx:
                reduction_idx += " + "
            reduction_idx += f"{rd.name} * {acc_buf_size}"

            if isinstance(rd.numel, sympy.Integer):
                acc_buf_size *= rd.numel
            else:
                acc_buf_size *= sympy.Symbol(
                    f"{rd.prefix}numel", integer=True, positive=True
                )

        acc_buf_size = Min(acc_buf_size, self.max_threadgroup_size)
        acc_buf_size_str = self.sexpr(acc_buf_size)
        # metal threadgroup arrays need a compile time constant size, so
        # fall back to the static upper bound when acc buf size is symbolic
        # happens when dynamic=True
        acc_buf_alloc_size = (
            acc_buf_size
            if isinstance(acc_buf_size, sympy.Integer)
            else self.max_threadgroup_size
        )
        shmem_buf_size = (
            ceildiv(acc_buf_size, self.simd_group_size)
            if isinstance(acc_buf_size, sympy.Integer)
            else self.simd_group_size
        )

        if reduction_type == "any":
            acc = self._new_idxvar(dtype)
            self.indexing_code.writeline(f"{acc} = false;")
            self.indexing_code.writeline(
                "threadgroup_barrier(metal::mem_flags::mem_threadgroup);"
            )
            self.compute.splice(
                f"""
                if ({value}) {{
                    {acc} = true;
                }}
            """
            )
            self.stores.writeline(
                "threadgroup_barrier(metal::mem_flags::mem_threadgroup);"
            )
            return acc

        self.headers.add("reduction_utils")

        if reduction_type in ["prod", "sum"]:
            acc_dtype = DTYPE_TO_COMPUTATION_DTYPE[src_dtype]
            acc_buf = self._new_idxvar(acc_dtype, shmem_buf_size)
            if not self.multistage_reduction_entry:
                val = value
            else:
                default_val, reduction_op = (
                    (0, "+") if reduction_type == "sum" else (1, "*")
                )
                val = self._new_idxvar(
                    acc_dtype, default_value=default_val, is_threadgroup=False
                )
                self.compute.splice(f"{val} {reduction_op}= {value};")

            return self.cse.generate(
                self.stores,
                f"c10::metal::threadgroup_{reduction_type}({acc_buf}, {val}, {reduction_idx}, {acc_buf_size_str})",
                dtype=DTYPE_TO_COMPUTATION_DTYPE[dtype],
            )
        if reduction_type in ["max", "min"]:
            acc_buf = self._new_idxvar(src_dtype, shmem_buf_size)
            src_metal_type = DTYPE_TO_METAL[src_dtype]
            cast_value = f"static_cast<{src_metal_type}>({value})"
            if not self.multistage_reduction_entry:
                val = cast_value  # type: ignore[assignment]
            else:
                op_struct = "MaxOp" if reduction_type == "max" else "MinOp"
                limit_val = f"::c10::metal::{op_struct}<{src_metal_type}>::identity()"
                val = self._new_idxvar(
                    src_dtype, default_value=limit_val, is_threadgroup=False
                )
                self.compute.splice(
                    f"{val} = ::c10::metal::{reduction_type}({val}, {cast_value});"
                )
            return self.cse.generate(
                self.stores,
                f"c10::metal::threadgroup_{reduction_type}({acc_buf}, {val}, {reduction_idx}, {acc_buf_size_str})",
                dtype=DTYPE_TO_COMPUTATION_DTYPE[dtype],
            )
        if reduction_type in ["argmin", "argmax"]:
            data_acc_buf = self._new_idxvar(src_dtype, shmem_buf_size)
            idx_acc_buf = self._new_idxvar(dtype, shmem_buf_size)
            src_metal_type = DTYPE_TO_METAL[src_dtype]
            cast_value = f"static_cast<{src_metal_type}>({value})"
            if not self.multistage_reduction_entry:
                val = cast_value  # type: ignore[assignment]
                idx_val = f"static_cast<{DTYPE_TO_METAL[dtype]}>({reduction_idx})"
            else:
                op_struct = "MaxOp" if reduction_type == "argmax" else "MinOp"
                limit_val = f"::c10::metal::{op_struct}<{src_metal_type}>::identity()"
                val = self._new_idxvar(
                    src_dtype, default_value=limit_val, is_threadgroup=False
                )
                idx_val = self._new_idxvar(dtype, default_value=0, is_threadgroup=False)  # type: ignore[assignment]
                idx_var = next(
                    t for t in self.range_tree_nodes.values() if t.is_reduction
                )
                self.compute.splice(f"""
                if (::c10::metal::{op_struct}<{src_metal_type}>::replace({cast_value}, {val})) {{
                    {val} = {cast_value};
                    {idx_val} = {idx_var.name};
                }}
                """)
            return self.cse.generate(
                self.stores,
                f"c10::metal::threadgroup_{reduction_type}({data_acc_buf}, {idx_acc_buf}, "
                f"{val}, {idx_val}, {reduction_idx}, {acc_buf_size_str})",
                dtype=dtype,
            )
        if reduction_type == "welford_reduce":
            if not self.multistage_reduction_entry:
                acc_buf = self._new_idxvar("float3", acc_buf_alloc_size)
                self.compute.splice(
                    f"{acc_buf}[{reduction_idx}] = float3({value}, 0.0, 1.0);"
                )
                wf_res = self.cse.generate(
                    self.compute,
                    f"c10::metal::threadgroup_welford_combine({acc_buf}, {reduction_idx}, {acc_buf_size_str})",
                    dtype=torch.float32,
                )
                return _unwrap_helper(wf_res)
            acc_buf = self._new_idxvar("float3", acc_buf_alloc_size)
            acc_thread_var = f"{acc_buf}[{reduction_idx}]"
            self.indexing_code.splice(f"{acc_thread_var} = 0.0;")
            self.compute.writeline(
                f"{acc_thread_var} = ::c10::metal::welford_combine({acc_thread_var}, float3({value}, 0.0, 1.0));"
            )
            wf_res = self.cse.generate(
                self.stores,
                f"c10::metal::threadgroup_welford_combine({acc_buf}, {reduction_idx}, {acc_buf_size_str})",
                dtype=torch.float32,
            )
            return _unwrap_helper(wf_res)
        if reduction_type == "welford_combine":
            if not isinstance(value, tuple):
                raise AssertionError("Input to welford combine must be tuple")
            acc_buf = self._new_idxvar("float3", acc_buf_alloc_size)
            acc_thread_var = f"{acc_buf}[{reduction_idx}]"
            inp_value = f"float3({value[0]}, {value[1]}, {value[2]})"
            self.indexing_code.splice(f"{acc_thread_var} = 0.0;")
            if self.multistage_reduction_entry:
                self.indexing_code.splice(f"{acc_thread_var} = 0.0;")
                self.compute.writeline(
                    f"{acc_thread_var} = ::c10::metal::welford_combine({acc_thread_var}, {inp_value});"
                )
            else:
                self.compute.writeline(f"{acc_thread_var} = {inp_value};")
            wf_res = self.cse.generate(
                self.stores if self.multistage_reduction_entry else self.compute,
                f"c10::metal::threadgroup_welford_combine({acc_buf}, {reduction_idx}, {acc_buf_size_str})",
                dtype=torch.float32,
            )
            return _unwrap_helper(wf_res)
        raise NotImplementedError(reduction_type)

    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry) -> None:
        # [FUNCTION DOC] Emits the definition of an index variable. Non-reduction entries or small reduction ranges become a plain assignment; large reduction ranges open a loop (once per root) striding by the max threadgroup size and compute each entry from a linear index, adding a bounds break when needed. Records entries in multistage_reduction_entry.
        index_expr = self.rename_indexing(entry.expr)
        index_str = self.sexpr(index_expr)  # type: ignore[misc]

        if not entry.is_reduction or (
            isinstance(entry.root.numel, sympy.Integer)
            and entry.root.numel <= self.max_threadgroup_size
        ):
            self.indexing_code.writeline(
                f"{self.index_dtype} {entry.name} = {index_str};"
            )
            return

        acc_size = (
            entry.root.numel
            if isinstance(entry.root.numel, sympy.Integer)
            else sympy.Symbol(f"{entry.root.prefix}numel", integer=True, positive=True)
        )

        # Check if we've already generated a loop for this reduction root
        root_already_processed = any(
            e.root is entry.root for e in self.multistage_reduction_entry
        )

        linear_idx_name = f"{entry.root.prefix}_linear_idx"

        if not root_already_processed:
            self.multistage_reduction_entry.append(entry)
            # When reducing the tensor whose size exceeds max threadgroup size
            # loop over extra indices per reduction thread and perform part of the operation
            # using values in the shared memory

            # Use floats so that it doesn't do integer division
            loop_size = (acc_size + float(self.max_threadgroup_size - 1)) // float(
                self.max_threadgroup_size
            )
            loop_size_str = self.sexpr(loop_size)

            root_name = entry.root.name

            self.body.writeline(
                f"for(auto {entry.root.prefix}_cnt = 0; {entry.root.prefix}_cnt < {loop_size_str}; ++{entry.root.prefix}_cnt) {{"
            )
            with self.body.indent():
                if isinstance(acc_size, sympy.Symbol):
                    self.body.writeline(
                        f"{self.index_dtype} {linear_idx_name} = "
                        f"{self.max_threadgroup_size} * {entry.root.prefix}_cnt + {root_name};"
                    )
                else:
                    self.body.writeline(
                        f"{self.index_dtype} {linear_idx_name} = {loop_size_str} * {root_name} + {entry.root.prefix}_cnt;"
                    )

                # Check that reduction is performed only within tensor boundary
                if (
                    isinstance(acc_size, sympy.Symbol)
                    or loop_size * self.max_threadgroup_size != acc_size
                ):
                    self.body.writeline(f"if ({linear_idx_name} >= {acc_size}) break;")

                # Compute entry value from linear index by substituting root name
                sub_index_str = index_str.replace(entry.root.name, linear_idx_name)
                self.body.writeline(
                    f"{self.index_dtype} {entry.name} = {sub_index_str};"
                )
        else:
            # root is already processed so just need to compute this entry's value inside the existing loop
            with self.body.indent():
                sub_index_str = index_str.replace(entry.root.name, linear_idx_name)
                self.body.writeline(
                    f"{self.index_dtype} {entry.name} = {sub_index_str};"
                )

    def codegen_body(self) -> None:
        # [FUNCTION DOC] Concatenates loads, compute and stores into the body. For multistage reductions closes the opened loops, invalidates CSE variables created inside (keeping reduction results), clears the entry list and range-node caches. Clears the loads, compute and stores buffers.
        """
        Concat output code from index_code, loads, compute, stores,
        suffix into self.body.

        For pointwise kernels, this is called just once at the end.

        For reduction kernels, this generates a loop over the reduction
        axis.
        """
        if self.multistage_reduction_entry:
            with self.body.indent():
                self.body.splice(self.loads)
                self.body.splice(self.compute)
            self.body.writeline("}" * len(self.multistage_reduction_entry))
            # Invalidate variables instantiated inside loop
            # But results of reduction alive. Reduction cache values can be
            # either CSEVariable or tuple of CSEVariables, in which case all
            # variables in the tuple must be preserved
            self.cse.invalidate(
                OrderedSet(
                    v
                    for item in self.cse.reduction_cache.values()
                    for v in (item if isinstance(item, tuple) else (item,))
                )
            )
            # And loop codegen
            roots = [e.root for e in self.multistage_reduction_entry]
            self.multistage_reduction_entry.clear()
            for node in self.range_tree_nodes.values():
                if any(node.root is r for r in roots):
                    node.cache_clear()
        else:
            self.body.splice(self.loads)
            self.body.splice(self.compute)
        self.body.splice(self.stores)
        self.loads.clear()
        self.compute.clear()
        self.stores.clear()

    def codegen_kernel(self, name: str = "generated_kernel") -> str:
        # [FUNCTION DOC] Returns the full shader source text wrapped for the Python wrapper (compile_mps_shader call) or the C++ wrapper (raw string with embedded headers). Emits includes, an optional max threads attribute for reductions, the kernel signature (output and input buffers, size variables, dynamic numels, optional error buffer, thread position args) and the body. Raises RuntimeError for float64 non-scalar inputs and AssertionError for more than 3 index variables.
        """Called at the end to generate a final kernel string"""
        self.codegen_body()
        code = IndentedBuffer()
        fn_name = name

        if V.graph.cpp_wrapper:
            code.writeline('(R"MTL(')
        else:
            code.writeline("compile_mps_shader('''")

        idx_vars = self.active_range_trees()
        with code.indent():
            if not V.graph.cpp_wrapper:
                for header in self.headers:
                    code.writeline(f"#include <c10/metal/{header}.h>")
            else:
                headers = [
                    f"#include <c10/metal/{header}.h>" for header in self.headers
                ]
                header_contents = _embed_headers(
                    headers,
                    [Path(__file__).parent.parent.parent / "include"],
                    OrderedSet(),  # type: ignore[arg-type]
                )
                code.writeline(header_contents)

            if self.inside_reduction:
                total_reduction_size = math.prod(
                    t.numel for t in self.range_trees if t.is_reduction
                )
                # If using dynamic shapes, set the threadgroup size to be the
                # max possible size
                threadgroup_size = (
                    min(total_reduction_size, self.max_threadgroup_size)
                    if isinstance(total_reduction_size, sympy.Integer)
                    else self.max_threadgroup_size
                )
                code.writeline(
                    f"[[max_total_threads_per_threadgroup({threadgroup_size})]]"
                )
            code.writeline(f"kernel void {fn_name}(")
            with code.indent():
                for outer, inner in self.args.output_buffers.items():
                    if outer in self.removed_buffers:
                        continue
                    dtype_str = self.dtype_to_str(V.graph.get_dtype(outer))
                    code.writeline(f"device {dtype_str}* {inner},")
                for outer, inner in self.args.input_buffers.items():
                    dtype = V.graph.get_dtype(outer)
                    # MPS does not support float64, but scalar inputs are fine
                    if dtype == torch.float64:
                        outer_buf = V.graph.try_get_buffer(outer)
                        if outer_buf is None or outer_buf.get_size() != []:
                            raise RuntimeError("float64 is not supported by MPS")
                        dtype_str = "float"
                    else:
                        dtype_str = self.dtype_to_str(dtype)
                    code.writeline(f"constant {dtype_str}* {inner},")
                for inner in self.args.sizevars.values():
                    code.writeline(f"constant long& {inner},")

                # Write dynamic values as inputs
                for idx_var in idx_vars:
                    if isinstance(idx_var.numel, sympy.Integer):
                        pass
                    else:
                        code.writeline(f"constant long& {idx_var.prefix}numel,")

                # Add error buffer parameter if error header is used
                if "error" in self.headers:
                    code.writeline("device c10::metal::ErrorMessages* error_buf,")

                if len(idx_vars) >= 4:
                    raise AssertionError("Up to 3 index variables are supported")
                thread_pos_dtype = (
                    f"uint{len(idx_vars)}" if len(idx_vars) > 1 else "uint"
                )
                thread_pos_var_name = (
                    idx_vars[0].name if len(idx_vars) == 1 else "thread_pos"
                )
                thread_pos_suffix = "," if self.inside_reduction else ""
                code.writeline(
                    f"{thread_pos_dtype} {thread_pos_var_name} [[thread_position_in_grid]]{thread_pos_suffix}"
                )
                if self.inside_reduction:
                    code.writeline(
                        f"{thread_pos_dtype} group_pos [[thread_position_in_threadgroup]]"
                    )
            code.writeline(") {")
            with code.indent():
                if len(idx_vars) > 1:
                    for idx, var in enumerate(idx_vars):
                        code.writeline(
                            f"auto {var.name} = thread_pos.{chr(120 + idx)};"
                        )
                code.splice(self.indexing_code)
                code.splice(self.body)
            code.writeline("}")

        if V.graph.cpp_wrapper:
            code.writeline(')MTL");')
        else:
            code.writeline("''')")

        return code.getvalue()

    def call_kernel(
        self, name: str, node: Any = None, deallocate_ws: bool = True
    ) -> None:
        # [FUNCTION DOC] Emits the wrapper call to launch the kernel: collects buffer, size-variable and dynamic numel arguments, computes thread and group sizes (reductions capped at the max threadgroup size), adds an error buffer index when the error header is used (Python wrapper only), and calls the wrapper's generate_kernel_call for the mps device. Raises RuntimeError in the C++ wrapper if there are no threads.
        # KNOWN ISSUE: the in-code TODO notes error buffer handling is not implemented for AOTI.
        """
        Codegens a call to this kernel
        """
        wrapper = V.graph.wrapper_code
        # Make sure sizevars has been computed
        for v in self.args.sizevars:
            wrapper.ensure_size_computed(v)

        _, call_args, _, arg_types = self.args.python_argdefs()
        arg_name_to_type = {
            str(call_arg): arg_type for call_arg, arg_type in zip(call_args, arg_types)
        }

        args = [*self.args.output_buffers.keys(), *self.args.input_buffers.keys()]
        args = [arg for arg in args if arg not in self.removed_buffers]
        args += [str(v) for v in self.args.sizevars]
        arg_types = [arg_name_to_type[arg] for arg in args]

        # Add any dynamic ints as inputs
        for tree in self.range_trees:
            if isinstance(tree.numel, (sympy.Integer, int)):
                # Don't need to pass in integers as inputs
                continue
            elif isinstance(tree.numel, sympy.Symbol):
                expr = tree.numel
            else:
                expr = V.graph.wrapper_code.generate_numel_expr(name, tree).inner

            if not tree.is_reduction or self.inside_reduction:
                args.append(str(expr))
                arg_types.append(int)

        expr_printer = self.cexpr if V.graph.cpp_wrapper else self.pexpr

        def format_threads(threads: list[str], kwarg: str) -> str:
            # [FUNCTION DOC] HELPER: Formats a list of thread extents as a braced uint64 list (C++ wrapper) or as a keyword list argument (Python wrapper).
            if V.graph.cpp_wrapper:
                threads = [f"static_cast<uint64_t>({t})" for t in threads]
                return f"{{{', '.join(threads)}}}"
            else:
                return f"{kwarg}=[{', '.join(threads)}]"

        # For reduction kernels, limit the maximum size over reduction dimensions to
        # a maximum threadgroup size
        if len(self.active_range_trees()) > 0:
            threads = [
                expr_printer(
                    Min(v.numel, self.max_threadgroup_size)  # type: ignore[misc]
                    if v.is_reduction
                    else v.numel
                )
                for v in self.active_range_trees()
            ]

            args.append(format_threads(threads, "threads"))
            arg_types.append(list)
        else:
            if V.graph.cpp_wrapper:
                raise RuntimeError("We should always have threads?")

        if self.inside_reduction:
            threads = [
                expr_printer(Min(v.numel, self.max_threadgroup_size))  # type: ignore[misc]
                if v.is_reduction
                else "1"
                for v in self.active_range_trees()
            ]
            args.append(format_threads(threads, "group_size"))
            arg_types.append(list)
        else:
            if V.graph.cpp_wrapper:
                # Add a None so that we always have a group_size in the
                # arguments. We won't use it if the value is None.
                args += [None]  # type: ignore[list-item]
                arg_types.append(None)

        # Add error buffer index if error reporting is used
        # TODO(malfet) Figure out how to do it for aoti
        if "error" in self.headers and not V.graph.cpp_wrapper:
            args.append(
                f"error_buf_idx={len([arg for arg in args if arg is not None and '=' not in arg])}"
            )

        wrapper.generate_kernel_call(
            name,
            args,
            device=torch.device("mps"),
            triton=False,
            arg_types=arg_types,
        )

    def device_assert_async(self, cond: CSEVariable, msg: str) -> None:
        # [FUNCTION DOC] Emits a device-side assertion: in the C++ wrapper an early return when the condition fails; otherwise reports the message through the error buffer (registers the error header) and returns.
        if V.graph.cpp_wrapper:
            self.cse.generate(self.compute, f"if (!{cond}) return", assignment=False)
        else:
            self.headers.add("error")
            self.compute.writelines(
                [
                    f"if (!{cond}) {{",
                    f"    TORCH_REPORT_ERROR(error_buf, {repr(msg)});",
                    "    return;",
                    "}",
                ]
            )

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> None:
        # [FUNCTION DOC] Emits an index bounds check on the compute buffer for lower and/or upper bounds, returning early on violation (C++ wrapper) or reporting an out-of-range message through the error buffer (Python wrapper). No-op if neither bound is requested.
        # KNOWN ISSUE: per the in-code TODO it is unclear whether the upper bound is inclusive or exclusive.
        if not (lower or upper):
            return

        expr_str = self.index_to_str(expr)
        size_str = self.index_to_str(size)

        # Generate bounds checking with error reporting
        # TODO(malfet): Is upper bound inclusive or exclusive?
        if lower and upper:
            # Check both lower and upper bounds
            condition = f"({expr_str} < 0 || {expr_str} >= {size_str})"
        elif lower:
            condition = f"{expr_str} < 0"
        else:
            condition = f"{expr_str} >= {size_str}"

        # Generate error reporting code
        if V.graph.cpp_wrapper:
            self.cse.generate(
                self.compute, f"if ({condition}) return", assignment=False
            )
        else:
            # Add error header for error reporting
            self.headers.add("error")
            self.compute.writelines(
                [
                    f"if ({condition}) {{",
                    f'    TORCH_REPORT_ERROR(error_buf, "Index ", {expr_str}, " out of range [0, ", {size_str}, ")");',
                    "    return;",
                    "}",
                ]
            )


class MetalScheduling(SIMDScheduling):
    # [CLASS DOC] SIMDScheduling for the MPS backend using MetalKernel as the kernel type. Defines how generated Metal sources are registered with the wrapper.
    kernel_type = MetalKernel  # type: ignore[assignment]

    def __init__(self, scheduler: Scheduler | None) -> None:
        # [FUNCTION DOC] Initializes the base SIMD scheduling with the given scheduler (may be None).
        super().__init__(scheduler)

    def define_kernel(
        self, src_code: str, node_schedule: list[SchedulerNode], kernel: MetalKernel
    ) -> str:
        # [FUNCTION DOC] Registers generated Metal source with the wrapper and returns the kernel name; reuses a cached name for identical source. C++ wrapper path: one library per kernel, defined via wrapper.define_kernel. Python path: extracts shader text from the compile_mps_shader call, strips include lines, renames the kernel function to a unique name, and writes an async_compile metal definition (with sorted header list) into the wrapper header.
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]

        if V.graph.cpp_wrapper:
            # C++ path: one library per kernel (each has a single "generated_kernel" function)
            mps_lib_name = f"mps_lib_{wrapper.next_kernel_suffix()}"
            kernel_name = mps_lib_name
            wrapper.src_to_kernel[src_code] = kernel_name
            src_code = f"const char* {mps_lib_name}_source = " + src_code
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            wrapper.define_kernel(
                mps_lib_name, src_code, f"{origins}\n{detailed_origins}", gpu=False
            )
            return kernel_name

        # Python path: register kernel with async_compile; wait() will compile all
        # accumulated Metal kernels into a single library and replace each placeholder.
        fn_name = f"generated_kernel_{wrapper.next_kernel_suffix()}"
        wrapper.src_to_kernel[src_code] = fn_name

        # Extract Metal source from compile_mps_shader('''...''') call
        metal_src_start = "compile_mps_shader('''"
        start = src_code.index(metal_src_start) + len(metal_src_start)
        end = src_code.rindex("''')")
        metal_src = src_code[start:end]

        # Strip #include lines and rename the kernel function
        body_lines = []
        for line in metal_src.split("\n"):
            if line.strip().startswith("#include"):
                continue
            body_lines.append(
                line.replace("kernel void generated_kernel(", f"kernel void {fn_name}(")
            )

        headers_repr = repr(sorted(kernel.headers))
        wrapper.header.writeline(f"{fn_name} = async_compile.metal({fn_name!r}, '''")
        for line in body_lines:
            wrapper.header.writeline(line)
        wrapper.header.writeline(f"''', {headers_repr})")

        return fn_name
