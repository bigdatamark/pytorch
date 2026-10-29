# [MODULE DOC] Sympy string printers that turn index/guard expressions into source text.
# ExprPrinter holds rules shared by C++ and Python; PythonPrinter emits Python (math/torch
# calls, and/or, floor division); CppPrinter emits C++ (std:: calls, c10 helpers, casts to the
# index type). Constants: INDEX_TYPE (int64_t name) and its max/min values. Unsupported
# functions raise NotImplementedError naming the printer class. No import-time side effects.
import sys
from typing import Literal

import sympy
from sympy.printing.precedence import PRECEDENCE, precedence
from sympy.printing.str import StrPrinter


INDEX_TYPE = "int64_t"
INDEX_TYPE_MAX = (1 << 63) - 1
INDEX_TYPE_MIN = -1 << 63


# This printer contains rules that are supposed to be generic for both C/C++ and
# Python
class ExprPrinter(StrPrinter):
    # [CLASS DOC] Base sympy StrPrinter shared by the Python and C++ printers. Sets printmethod so
    # classes supplying a torch-specific print hook are honored. Implements language-neutral
    # rules (Mul, Add, relationals, bitwise functions, Mod, float division, Float, natural Pow)
    # and declares many _print_* methods that raise NotImplementedError so subclasses must
    # provide them. Stateless apart from the base printer settings.
    # override this so that _print_FloorDiv is used
    printmethod = "_torch_sympystr"

    def _print_Mul(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints a product as factors joined by the multiplication operator.
        return self.stringify(expr.args, "*", precedence(expr))

    def _print_Not(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints a logical negation as "not" applied to the parenthesized operand.
        # pyrefly: ignore [missing-attribute]
        return f"not ({self._print(expr.args[0])})"

    def _print_Add(self, expr: sympy.Expr, order: str | None = None) -> str:
        # [FUNCTION DOC] Prints a sum as terms joined by plus signs; the order argument is ignored.
        return self.stringify(expr.args, " + ", precedence(expr))

    def _print_Relational(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints a relational by joining operands with the relation's operator text.
        return self.stringify(expr.args, f" {expr.rel_op} ", precedence(expr))

    def _print_BitwiseFn_bitwise_and(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints a bitwise-and as operands joined by the ampersand operator.
        return self.stringify(expr.args, " & ", PRECEDENCE["BitwiseAnd"])

    def _print_BitwiseFn_bitwise_or(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints a bitwise-or as operands joined by the vertical bar operator.
        return self.stringify(expr.args, " | ", PRECEDENCE["BitwiseOr"])

    def _print_BitwiseFn_bitwise_xor(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints a bitwise-xor as operands joined by the caret operator.
        return self.stringify(expr.args, " ^ ", PRECEDENCE["BitwiseXor"])

    # NB: this is OK to put here, because Mod is only defined for positive
    # numbers, and so across C/Python its behavior is consistent
    def _print_Mod(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints Mod as operands joined by the percent operator; shared by C++ and
        # Python because Mod is only used for nonnegative values.
        return self.stringify(expr.args, " % ", PRECEDENCE["Atom"] - 0.5)

    def _print_FloatTrueDiv(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints float true division as a parenthesized slash expression.
        s = self.stringify(expr.args, " / ", PRECEDENCE["Atom"] - 0.5)
        return f"({s})"

    def _print_CleanDiv(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints exact division by delegating to the floor-division printer.
        return self._print_FloorDiv(expr)

    def _print_Identity(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints the identity wrapper as just its single operand.
        # pyrefly: ignore [missing-attribute]
        return self._print(expr.args[0])

    def _print_Float(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints a float. Double-precision (53-bit) values use 17 significant digits
        # so they round-trip; other precisions use the default string.
        if expr._prec == 53:
            # IEEE-754 double precision have 53 bits. SymPy prints them with
            # 15 digits, but we need 17 for round-trip correctness
            return str(sympy.Float(expr, dps=17))
        else:
            # We don't use other precisions in pytorch
            return str(expr)

    # This must be implemented because sympy will collect x * x into Pow(x, 2), without
    # any explicit intervention.  We print it just like x * x, notably, we
    # never generate sympy.Pow with floats.
    #
    # NB: this pow by natural, you should never have used builtin sympy.pow
    # for FloatPow, and a symbolic exponent should be PowByNatural.  These
    # means exp is guaranteed to be integer.
    def _print_Pow(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints a power with a nonnegative integer exponent as repeated
        # multiplication of the base; exponent zero prints as 1. Raises an error if the exponent is
        # not an integer or is negative.
        base, exp = expr.args
        if exp != int(exp):
            raise AssertionError(exp)
        exp = int(exp)
        if exp < 0:
            raise AssertionError(f"exponent must be non-negative, got {exp}")
        if exp > 0:
            return self.stringify([base] * exp, "*", PRECEDENCE["Mul"])
        return "1"

    # Explicit NotImplemented functions are to prevent default sympy printing
    # behavior, which will just barf out ToFloat(...) to your IR.  The error
    # message is better here because it tells you which printer class it needs
    # to go in.

    def _print_ToFloat(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Raises NotImplementedError; subclasses must handle ToFloat.
        raise NotImplementedError(f"_print_ToFloat not implemented for {type(self)}")

    def _print_Infinity(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Raises NotImplementedError; subclasses must handle positive infinity.
        raise NotImplementedError(f"_print_Infinity not implemented for {type(self)}")

    def _print_NegativeInfinity(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Raises NotImplementedError; subclasses must handle negative infinity.
        raise NotImplementedError(
            f"_print_NegativeInfinity not implemented for {type(self)}"
        )

    def _print_NaN(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Raises NotImplementedError; subclasses must handle NaN.
        raise NotImplementedError(f"_print_NaN not implemented for {type(self)}")

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Raises NotImplementedError; subclasses must handle FloorDiv.
        raise NotImplementedError(f"_print_FloorDiv not implemented for {type(self)}")

    def _print_PythonMod(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Raises NotImplementedError; subclasses must handle PythonMod.
        raise NotImplementedError(f"_print_PythonMod not implemented for {type(self)}")

    def _print_IntTrueDiv(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Raises NotImplementedError; subclasses must handle IntTrueDiv.
        raise NotImplementedError(f"_print_IntTrueDiv not implemented for {type(self)}")

    def _print_PowByNatural(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Raises NotImplementedError; subclasses must handle PowByNatural.
        raise NotImplementedError(
            f"_print_PowByNatural not implemented for {type(self)}"
        )

    def _print_FloatPow(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Raises NotImplementedError; subclasses must handle FloatPow.
        raise NotImplementedError(f"_print_FloatPow not implemented for {type(self)}")

    def _print_TruncToInt(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Raises NotImplementedError; subclasses must handle TruncToInt.
        raise NotImplementedError(f"_print_TruncToInt not implemented for {type(self)}")

    def _print_RoundToInt(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Raises NotImplementedError; subclasses must handle RoundToInt.
        raise NotImplementedError(f"_print_RoundToInt not implemented for {type(self)}")

    def _print_RoundDecimal(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Raises NotImplementedError; subclasses must handle RoundDecimal.
        raise NotImplementedError(
            f"_print_RoundDecimal not implemented for {type(self)}"
        )

    # NB: Some float operations are INTENTIONALLY not implemented for
    # printers.  You can implement them as a quick unblock, but it is better
    # to ask yourself why we haven't done this computation in the Tensor
    # universe instead

    def _print_TruncToFloat(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Raises NotImplementedError; TruncToFloat is intentionally unsupported by
        # default.
        raise NotImplementedError(
            f"_print_TruncToFloat not implemented for {type(self)}"
        )


class PythonPrinter(ExprPrinter):
    # [CLASS DOC] Printer producing Python source text, used e.g. for serializing guard
    # expressions that are later evaluated (possibly with symbolic ints/floats). Maps sympy
    # functions to math module calls, torch.sym_float and torch._sym_sqrt, and/or operators,
    # floor division, and nested conditional expressions. Stateless.
    def _print_ToFloat(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints ToFloat as a call to the torch symbolic float conversion so
        # symbolic inputs propagate guards. Raises an error unless exactly one argument.
        if len(expr.args) != 1:
            raise AssertionError("ToFloat expects exactly one argument")
        # NB: We use sym_float here because the printer is used for cache
        # serialization, and cache guards get evaluated with SymInt to
        # propagate guards to the parent ShapeEnv.  However, this comes at a
        # runtime cost for guards involving float.  If this is unacceptable
        # overhead, what you want to do is have two separate printers for
        # SymInt, one for when the inputs are guaranteed to be int, and
        # another for when they could be SymInt.
        #
        # NB: sym_min/sym_max also have this problem, but I chose not to fix
        # those.
        #
        # See https://github.com/pytorch/pytorch/issues/142507 for more
        # context.
        # pyrefly: ignore [missing-attribute]
        return f"torch.sym_float({self._print(expr.args[0])})"

    def _print_And(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints a conjunction with the Python "and" keyword.
        return self.stringify(expr.args, " and ", precedence(expr))

    def _print_Or(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints a disjunction with the Python "or" keyword.
        return self.stringify(expr.args, " or ", precedence(expr))

    def _print_ModularIndexing(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints ModularIndexing (value, divisor, modulus) as the value
        # floor-divided by the divisor (skipped when the divisor prints as 1), then taken
        # modulo the modulus.
        x, div, mod = (
            self.parenthesize(arg, PRECEDENCE["Atom"] - 0.5) for arg in expr.args
        )
        if div != "1":
            x = f"({x} // {div})"
        return f"({x} % {mod})"

    def _print_Infinity(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints positive infinity as the math module infinity constant.
        return "math.inf"

    def _print_NegativeInfinity(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints negative infinity as the negated math module infinity constant.
        return "-math.inf"

    def _print_NaN(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints NaN as the math module nan constant.
        return "math.nan"

    # WARNING: this is dangerous for Triton, which has C-style modulus
    def _print_PythonMod(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints PythonMod with the percent operator (Python semantics; unsafe for
        # Triton, which uses C-style modulus).
        return self.stringify(expr.args, " % ", PRECEDENCE["Atom"] - 0.5)

    # WARNING: this is dangerous for Triton, which has C-style modulus
    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints FloorDiv with the double-slash operator and parenthesized operands
        # (unsafe for Triton, which uses C-style division).
        x, div = (self.parenthesize(arg, PRECEDENCE["Atom"] - 0.5) for arg in expr.args)
        return f"{x} // {div}"

    # WARNING: this is dangerous for Triton, when lhs, rhs > 2**53, Python
    # does a special algorithm
    def _print_IntTrueDiv(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints IntTrueDiv with the slash operator (imprecise beyond 2 to the 53
        # and differs in Triton).
        return self.stringify(expr.args, " / ", PRECEDENCE["Atom"] - 0.5)

    def _helper_sqrt(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] HELPER: Prints a square root of the given operand as a call to
        # torch._sym_sqrt, which avoids forcing specialization of symbolic floats.
        # NB: We use torch._sym_sqrt here instead of math.sqrt because the
        # guard expression may be evaluated with SymInt/SymFloat inputs (e.g.
        # during cache hit re-evaluation in evaluate_guards_expression).
        # math.sqrt on a SymFloat triggers evaluate_expr which forces
        # concretization/specialization of the symbol, creating spurious
        # guards that didn't exist in the original program.
        # torch._sym_sqrt properly propagates through the symbolic system
        # without forcing specialization.
        # See https://github.com/pytorch/pytorch/issues/152435
        # pyrefly: ignore [missing-attribute]
        return f"torch._sym_sqrt({self._print(expr)})"

    def _print_OpaqueUnaryFn_sqrt(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints the opaque sqrt function via the sqrt helper on its first argument.
        return self._helper_sqrt(expr.args[0])

    def _print_FloatPow(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints float power with the double-star operator.
        return self.stringify(expr.args, " ** ", PRECEDENCE["Pow"])

    # TODO: Not sure this works with Triton, even when base/exp are integral
    def _print_PowByNatural(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints power-by-natural with the double-star operator.
        return self.stringify(expr.args, " ** ", PRECEDENCE["Pow"])

    def _print_floor(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints floor as a math.floor call; raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("floor expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.floor({self._print(expr.args[0])})"

    def _print_FloorToInt(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints FloorToInt as math.floor; raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("FloorToInt expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.floor({self._print(expr.args[0])})"

    def _print_TruncToInt(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints TruncToInt as math.trunc; raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("TruncToInt expects exactly one argument")
        # This also could have been int(), they'll do the same thing for float
        # pyrefly: ignore [missing-attribute]
        return f"math.trunc({self._print(expr.args[0])})"

    def _print_ceiling(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints ceiling as a math.ceil call; raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("ceiling expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.ceil({self._print(expr.args[0])})"

    def _print_CeilToInt(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints CeilToInt as a math.ceil call; raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("CeilToInt expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.ceil({self._print(expr.args[0])})"

    def _print_Abs(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints absolute value as abs; raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("Abs expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"abs({self._print(expr.args[0])})"

    # NB: It's expected that we've made explicit any promotion in the sympy
    # expression, so it doesn't matter that Python max/min doesn't perform
    # promotion
    def _print_Max(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints Max as a Python max call over all arguments; raises an error if
        # fewer than two arguments.
        if len(expr.args) < 2:
            raise AssertionError("Max expects at least two arguments")
        # pyrefly: ignore [missing-attribute]
        return f"max({', '.join(map(self._print, expr.args))})"

    def _print_Min(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints Min as a Python min call over all arguments; raises an error if
        # fewer than two arguments.
        if len(expr.args) < 2:
            raise AssertionError("Min expects at least two arguments")
        # pyrefly: ignore [missing-attribute]
        return f"min({', '.join(map(self._print, expr.args))})"

    def _print_OpaqueUnaryFn_cos(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints opaque cos as math.cos; raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("cos expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.cos({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_cosh(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints opaque cosh as math.cosh; raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("cosh expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.cosh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_acos(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints opaque acos as math.acos; raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("acos expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.acos({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_sin(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints opaque sin as math.sin; raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("sin expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.sin({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_sinh(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints opaque sinh as math.sinh; raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("sinh expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.sinh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_asin(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints opaque asin as math.asin; raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("asin expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.asin({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_tan(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints opaque tan as math.tan; raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("tan expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.tan({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_tanh(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints opaque tanh as math.tanh; raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("tanh expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.tanh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_atan(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints opaque atan as math.atan; raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("atan expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.atan({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_log2(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints opaque log2 as math.log2; raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("log2 expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.log2({self._print(expr.args[0])})"

    def _print_RoundToInt(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints RoundToInt as builtin round; raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("RoundToInt expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"round({self._print(expr.args[0])})"

    def _print_RoundDecimal(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints RoundDecimal as the builtin round with a digits argument. Raises an
        # error unless two arguments, and TypeError if the digit count is not a sympy Integer.
        if len(expr.args) != 2:
            raise AssertionError("RoundDecimal expects exactly two arguments")
        number, ndigits = expr.args
        if not isinstance(ndigits, sympy.Integer):
            raise TypeError("ndigits must be an instance of sympy.Integer")
        # pyrefly: ignore [missing-attribute]
        return f"round({self._print(number)}, {ndigits})"

    def _print_Piecewise(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints a Piecewise as nested conditional expressions, built by walking
        # the (expression, condition) pairs from last to first. A true condition is the default
        # branch; an unconditional last branch is used as the fallback. Returns "0" if nothing
        # was produced.
        # Convert Piecewise(expr_cond_pairs) to nested ternary expressions
        # Piecewise((e1, c1), (e2, c2), ..., (eN, cN))
        # becomes: e1 if c1 else (e2 if c2 else (... else eN))
        result: str | None = None
        for expr_i, cond_i in reversed(expr.args):
            # pyrefly: ignore [missing-attribute]
            expr_str = self._print(expr_i)
            if cond_i == True:  # noqa: E712
                # This is the default case
                result = expr_str
            else:
                # pyrefly: ignore [missing-attribute]
                cond_str = self._print(cond_i)
                if result is None:
                    result = expr_str
                else:
                    result = f"({expr_str} if {cond_str} else {result})"
        return result if result else "0"


class CppPrinter(ExprPrinter):
    # [CLASS DOC] Printer producing C++ source text. Integers get long suffixes and are range
    # checked against the index type; conditionals use the ternary operator; floor division and
    # modular indexing use c10 helper functions; math uses std:: functions with casts to the
    # index type for integer-valued results. Stateless.
    def _print_Integer(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints an integer with an L suffix (LL on macOS and Windows). Raises
        # OverflowError outside the index type range. The minimum value is printed as a shifted
        # minus one to avoid a literal overflow.
        suffix = "LL" if sys.platform in ["darwin", "win32"] else "L"
        i = int(expr)
        if i > INDEX_TYPE_MAX or i < INDEX_TYPE_MIN:
            raise OverflowError(f"{i} too big to convert to {INDEX_TYPE}")
        elif i == INDEX_TYPE_MIN:
            if i != (-1) << 63:
                raise AssertionError("unexpected minimum index type value")
            # Writing -9223372036854775808L makes the value overflow
            # as it is parsed as -(9223372036854775808L) by the C/C++ compiler
            return f"(-1{suffix} << 63)"
        return f"{i}{suffix}"

    def _print_Where(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints Where as a C++ conditional (condition ? true-value : false-value)
        # with parenthesized operands.
        c, p, q = (
            self.parenthesize(arg, PRECEDENCE["Atom"] - 0.5) for arg in expr.args
        )
        return f"{c} ? {p} : {q}"

    def _print_Or(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints a disjunction with the C++ "||" operator.
        return self.stringify(expr.args, " || ", precedence(expr))

    def _print_Piecewise(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints a Piecewise as chained ternary operators, built from the last
        # pair to the first; a true condition is the default branch and an unconditional last
        # branch is the fallback. The result is parenthesized; returns "0" if nothing was
        # produced.
        # Convert Piecewise(expr_cond_pairs) to nested ternary operators
        # Piecewise((e1, c1), (e2, c2), ..., (eN, cN))
        # becomes: c1 ? e1 : (c2 ? e2 : (... : eN))
        result: str | None = None
        for expr_i, cond_i in reversed(expr.args):
            expr_str = self.parenthesize(expr_i, PRECEDENCE["Atom"] - 0.5)
            if cond_i == True:  # noqa: E712
                # This is the default case
                result = expr_str
            else:
                cond_str = self.parenthesize(cond_i, PRECEDENCE["Atom"] - 0.5)
                if result is None:
                    result = expr_str
                else:
                    result = f"{cond_str} ? {expr_str} : {result}"
        return f"({result})" if result else "0"

    def _print_ModularIndexing(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints ModularIndexing (value, divisor, modulus). If divisor is not 1, the
        # value is floor-divided by the c10 integer or floating helper depending on whether the
        # expression is integer-valued. Result is the value cast to the index type, modulo modulus.
        x, div, mod = expr.args
        x = self.doprint(x)
        if div != 1:
            div = self.doprint(div)
            if expr.is_integer:
                x = f"c10::div_floor_integer(static_cast<int64_t>({x}), static_cast<int64_t>({div}))"
            else:
                x = f"c10::div_floor_floating(static_cast<double>({x}), static_cast<double>({div}))"
        mod = self.doprint(mod)
        return f"(static_cast<{INDEX_TYPE}>({x}) % static_cast<{INDEX_TYPE}>({mod}))"

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints FloorDiv using the c10 floor-division helper for integers (cast to
        # int64) or floating point (cast to double), chosen by whether the expression is integer.
        x, div = expr.args
        x = self.doprint(x)
        div = self.doprint(div)
        if expr.is_integer:
            return f"c10::div_floor_integer(static_cast<int64_t>({x}), static_cast<int64_t>({div}))"
        return f"c10::div_floor_floating(static_cast<double>({x}), static_cast<double>({div}))"

    def _print_floor(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints floor as std::floor, cast to the index type if the result is
        # integer valued. Raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("floor expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        r = f"std::floor({self._print(expr.args[0])})"
        return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r

    def _print_FloorToInt(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints FloorToInt as std::floor, cast to the index type if the result is
        # integer valued. Raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("FloorToInt expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        r = f"std::floor({self._print(expr.args[0])})"
        return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r

    def _print_TruncToInt(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints TruncToInt as std::trunc cast to the index type. Raises an error
        # unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("TruncToInt expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        r = f"std::trunc({self._print(expr.args[0])})"
        return f"static_cast<{INDEX_TYPE}>({r})"

    def _print_TruncToFloat(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints TruncToFloat as std::trunc (no cast). Raises an error unless one
        # argument.
        if len(expr.args) != 1:
            raise AssertionError("TruncToFloat expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"std::trunc({self._print(expr.args[0])})"

    def _print_ToFloat(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints ToFloat as a cast to double. Raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("ToFloat expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"static_cast<double>({self._print(expr.args[0])})"

    def _print_PythonMod(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints PythonMod as a call to the c10 div_mod helper.
        x, div = expr.args
        x = self.doprint(x)
        div = self.doprint(div)
        return f"c10::div_mod({x}, {div})"

    def _print_IntTrueDiv(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints IntTrueDiv as double-cast numerator divided by double-cast
        # denominator (accurate only up to 2 to the 53).
        lhs, rhs = expr.args
        # TODO: This is only accurate up to 2**53
        # pyrefly: ignore [missing-attribute]
        return f"static_cast<double>({self._print(lhs)}) / static_cast<double>({self._print(rhs)})"

    # TODO: PowByNatural: we need to implement our own int-int pow.  Do NOT
    # use std::pow, that operates on floats
    def _print_PowByNatural(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints PowByNatural; only a base of 2 is supported (as a left shift of
        # 1 by the exponent). Any other base raises NotImplementedError.
        # Implement the special-case of 2**x for now
        base, exp = expr.args
        if base == 2:
            # pyrefly: ignore [missing-attribute]
            return f"(1 << ({self._print(exp)}))"
        raise NotImplementedError(
            f"_print_PowByNatural not implemented for {type(self)}"
        )

    def _print_FloatPow(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints float power as std::pow of the printed base and exponent.
        base, exp = expr.args
        # pyrefly: ignore [missing-attribute]
        return f"std::pow({self._print(base)}, {self._print(exp)})"

    def _print_Pow(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints a power. Exponent plus or minus one half becomes std::sqrt (or its
        # reciprocal). Integer exponents become repeated multiplication, reciprocals use float
        # one, zero gives 1.0, cast to the index type if the expression is integer-valued.
        # Other exponents become std::pow with a float exponent.
        # KNOWN ISSUE: the std::pow branch inserts the base via default string conversion
        # instead of the printer, so the base is not printed with C++ rules.
        # Uses float constants to perform FP div
        base, exp = expr.args

        if exp == 0.5 or exp == -0.5:
            # pyrefly: ignore [missing-attribute]
            base = self._print(base)
            return f"std::sqrt({base})" if exp == 0.5 else f"1.0/std::sqrt({base})"
        if exp.is_integer:
            exp = int(exp)
            if exp > 0:
                r = self.stringify([base] * exp, "*", PRECEDENCE["Mul"])
            elif exp < -1:
                r = (
                    "1.0/("
                    + self.stringify([base] * abs(exp), "*", PRECEDENCE["Mul"])
                    + ")"
                )
            elif exp == -1:
                # pyrefly: ignore [missing-attribute]
                r = "1.0/" + self._print(base)
            else:  # exp == 0
                r = "1.0"

            return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r
        else:
            # TODO: float vs double
            return f"std::pow({base}, {float(exp)})"

    def _print_Rational(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints a rational as float numerator over float denominator (or the plain
        # numerator if the denominator is 1), cast to the index type if integer-valued.
        # Uses float constants to perform FP div
        if expr.q == 1:
            r = f"{expr.p}"
        else:
            r = f"{expr.p}.0/{expr.q}.0"
        return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r

    def _print_ceiling(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints ceiling as std::ceil, cast to the index type if integer valued.
        # Raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("ceiling expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        r = f"std::ceil({self._print(expr.args[0])})"
        return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r

    def _print_CeilToInt(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints CeilToInt as std::ceil, cast to the index type if integer valued.
        # Raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("CeilToInt expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        r = f"std::ceil({self._print(expr.args[0])})"
        return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r

    def _print_min_max(self, expr: sympy.Expr, name: Literal["min", "max"]) -> str:
        # [FUNCTION DOC] HELPER: Prints min or max (per name). Non-integer results cast arguments
        # to double; integer results cast to the index type. Two arguments use the plain std
        # call, more use the initializer-list overload with an explicit element type.
        # pyrefly: ignore [missing-attribute]
        args = [self._print(a) for a in expr.args]
        if not expr.is_integer:
            args = [f"static_cast<double>({arg})" for arg in args]
            if len(args) == 2:
                return f"std::{name}({args[0]}, {args[1]})"
            # Initializer list overload
            il = "{" + ", ".join(args) + "}"
            return f"std::{name}<double>({il})"

        if len(args) == 2:
            return f"std::{name}(static_cast<{INDEX_TYPE}>({args[0]}), static_cast<{INDEX_TYPE}>({args[1]}))"
        else:
            # Initializer list overload
            il = "{" + ", ".join(args) + "}"
            return f"std::{name}<{INDEX_TYPE}>({il})"

    def _print_Min(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints Min via the shared min/max helper.
        return self._print_min_max(expr, "min")

    def _print_Max(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints Max via the shared min/max helper.
        return self._print_min_max(expr, "max")

    def _print_Abs(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints absolute value as std::abs; raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("Abs expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"std::abs({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_cos(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints opaque cos as std::cos; raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("cos expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"std::cos({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_cosh(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints opaque cosh as std::cosh; raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("cosh expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"std::cosh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_acos(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints opaque acos as std::acos; raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("acos expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"std::acos({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_sin(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints opaque sin; raises an error unless one argument.
        # KNOWN ISSUE: emits the Python math.sin call instead of std::sin, which is invalid C++.
        if len(expr.args) != 1:
            raise AssertionError("sin expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"math.sin({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_sinh(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints opaque sinh as std::sinh; raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("sinh expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"std::sinh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_asin(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints opaque asin as std::asin; raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("asin expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"std::asin({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_tan(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints opaque tan as std::tan; raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("tan expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"std::tan({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_tanh(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints opaque tanh as std::tanh; raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("tanh expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"std::tanh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_atan(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints opaque atan as std::atan; raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("atan expects exactly one argument")
        # pyrefly: ignore [missing-attribute]
        return f"std::atan({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_sqrt(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints opaque sqrt as std::sqrt of its first argument.
        # pyrefly: ignore [missing-attribute]
        return f"std::sqrt({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_log2(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints opaque log2 as std::log2 of its first argument.
        # pyrefly: ignore [missing-attribute]
        return f"std::log2({self._print(expr.args[0])})"

    def _print_RoundToInt(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints RoundToInt as std::lrint; raises an error unless one argument.
        if len(expr.args) != 1:
            raise AssertionError("RoundToInt expects exactly one argument")
        # TODO: dispatch to llrint depending on index type
        # pyrefly: ignore [missing-attribute]
        return f"std::lrint({self._print(expr.args[0])})"

    def _print_RoundDecimal(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints RoundDecimal for float numbers as a nearbyint of the scaled value,
        # rescaled back, cast to double. Integer inputs always raise: AssertionError if digits are
        # nonnegative, otherwise ValueError. Raises an error unless two arguments.
        # KNOWN ISSUE: the integer branch raises an assertion for nonnegative digits and an error
        # message claiming only nonnegative digits are supported, which are contradictory.
        if len(expr.args) != 2:
            raise AssertionError("RoundDecimal expects exactly two arguments")
        number, ndigits = expr.args
        if number.is_integer:
            # ndigits < 0 should have been filtered by the sympy function
            if ndigits >= 0:
                raise AssertionError("ndigits must be negative for integer inputs")
            raise ValueError(
                f"For integer inputs, only non-negative ndigits are currently supported, but got {ndigits}."
            )
        number_str = self.parenthesize(number, PRECEDENCE["Mul"])
        return f"static_cast<double>(std::nearbyint(1e{ndigits} * {number_str}) * 1e{-ndigits})"

    def _print_BooleanTrue(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints the boolean true constant as "true".
        return "true"

    def _print_BooleanFalse(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints the boolean false constant as "false".
        return "false"

    def _print_Infinity(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints positive infinity as the numeric_limits double infinity.
        return "std::numeric_limits<double>::infinity()"

    def _print_NegativeInfinity(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints negative infinity as the negated positive infinity text.
        return f"-{self._print_Infinity(expr)}"

    def _print_NaN(self, expr: sympy.Expr) -> str:
        # [FUNCTION DOC] Prints NaN as the numeric_limits double quiet NaN.
        return "std::numeric_limits<double>::quiet_NaN()"
