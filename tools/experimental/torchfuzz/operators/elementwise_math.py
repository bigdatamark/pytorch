# pyre-strict
# [MODULE DOC] Elementwise math operators for torchfuzz: base classes for unary, bool-output
# unary, float-only pointwise and ternary ops; a factory that builds concrete operator classes
# from torch function names; and hand-written classes for ops needing extra random arguments
# (logit, nan_to_num, round, divide, rsub, addcdiv, addcmul, clamp_max, clamp_min, pow,
# heaviside, ldexp). At import time it loops over the name tables and injects generated classes
# into the module namespace (marked as belonging to this module so the registry finds them).
"""Elementwise math Operator subclasses.

Covers unary, bool-output unary, binary, ternary, and a handful of special-case
elementwise ops (clamp_max, clamp_min, heaviside, ldexp, logcumsumexp). Op
classes that overlap upstream torchfuzz (add, sub, mul, div, clamp) are NOT
defined here — they remain registered by upstream.

Subclasses are generated via small factory helpers to avoid hundreds of lines
of near-identical boilerplate. Each generated class is assigned to a
module-level name and has ``__module__`` set so the introspection-based
registry discovers it.
"""

from __future__ import annotations

import random
from typing import Any

import torch

from torchfuzz.operators._dtypes import (
    contiguous_stride,
    FLOAT_DTYPES,
    is_float_dtype,
    random_broadcast_shape,
)
from torchfuzz.operators.base import Operator
from torchfuzz.operators.tensor_pointwise import PointwiseOperatorBase
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


# ---------------------------------------------------------------------------
# Base classes (excluded from introspection by the ``Base`` suffix)
# ---------------------------------------------------------------------------


class UnaryElementwiseOperatorBase(Operator):
    # [CLASS DOC] Base for one-input one-output elementwise ops where output and input share
    # size, stride and dtype. The class attribute requires_float (default false) restricts the
    # output to floating dtypes when set by a subclass. Bool outputs are never produced.
    """Base for one-tensor-in / one-tensor-out elementwise math (same dtype)."""

    requires_float: bool = False

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property; returns the operator name, which is the torch function path.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] True for a TensorSpec whose dtype is not bool and, when requires_float
        # is set, is a floating dtype; false otherwise.
        if not isinstance(output_spec, TensorSpec):
            return False
        if output_spec.dtype == torch.bool:
            return False
        if self.requires_float and not is_float_dtype(output_spec.dtype):
            return False
        return True

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns one input spec copying the output size, stride and dtype.
        # Raises ValueError if output_spec is not a TensorSpec.
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                f"{self.__class__.__name__} can only produce TensorSpec outputs"
            )
        return [
            TensorSpec(
                size=output_spec.size,  # pyrefly: ignore[missing-argument]
                stride=output_spec.stride,
                dtype=output_spec.dtype,
            )
        ]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Returns one line applying the torch function to the single input.
        # Raises ValueError unless exactly one input name is given.
        if len(input_names) != 1:
            raise ValueError(
                f"{self.__class__.__name__} requires exactly 1 input tensor"
            )
        return f"{output_name} = {self.torch_op_name}({input_names[0]})"


class BoolUnaryElementwiseOperatorBase(Operator):
    # [CLASS DOC] Base for predicates (isnan, isfinite, signbit, ...) that map one numeric
    # tensor to a bool tensor of the same size and stride. Stateless.
    """Base for one-numeric-tensor-in / one-bool-tensor-out (isfinite/isnan/...)."""

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property; returns the operator name, which is the torch function path.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] True only for a TensorSpec with bool dtype.
        return isinstance(output_spec, TensorSpec) and output_spec.dtype == torch.bool

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns one input spec with the output's size and stride and a dtype
        # chosen at random from the float dtypes plus int8/16/32/64 (global random module).
        # Raises ValueError if output_spec is not a TensorSpec.
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                f"{self.__class__.__name__} can only produce TensorSpec outputs"
            )
        # Pick a random float dtype at spec-fuzz time
        return [
            TensorSpec(
                size=output_spec.size,  # pyrefly: ignore[missing-argument]
                stride=output_spec.stride,
                dtype=random.choice(
                    FLOAT_DTYPES + (torch.int8, torch.int16, torch.int32, torch.int64)
                ),
            )
        ]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Returns one line applying the torch predicate to the single input.
        # Raises ValueError unless exactly one input name is given.
        if len(input_names) != 1:
            raise ValueError(
                f"{self.__class__.__name__} requires exactly 1 input tensor"
            )
        return f"{output_name} = {self.torch_op_name}({input_names[0]})"


class FloatOnlyPointwiseOperatorBase(PointwiseOperatorBase):
    # [CLASS DOC] Upstream pointwise binary base limited to floating output dtypes. By default
    # no scalar operand positions are allowed (both operands are tensors).
    """PointwiseOperatorBase restricted to float dtypes."""

    _scalar_input_positions = ()

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] True only for a TensorSpec with a floating dtype.
        if not isinstance(output_spec, TensorSpec):
            return False
        if output_spec.dtype == torch.bool:
            return False
        return is_float_dtype(output_spec.dtype)


class TernaryElementwiseOperatorBase(Operator):
    # [CLASS DOC] Base for three-input one-output elementwise ops (lerp, addcdiv, addcmul).
    # Inputs share the output dtype; requires_float (default true) limits outputs to floats.
    """Base for three-tensor-in / one-tensor-out elementwise math (float-only)."""

    requires_float: bool = True

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property; returns the operator name, which is the torch function path.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] True for a TensorSpec whose dtype is not bool and, when requires_float
        # is set, is a floating dtype; false otherwise.
        if not isinstance(output_spec, TensorSpec):
            return False
        if output_spec.dtype == torch.bool:
            return False
        if self.requires_float and not is_float_dtype(output_spec.dtype):
            return False
        return True

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns three input specs copying the output size, stride and dtype;
        # with probability 0.3 one random input is replaced by a smaller contiguous broadcast
        # shape (global random module). Raises ValueError for a non-tensor output spec.
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                f"{self.__class__.__name__} can only produce TensorSpec outputs"
            )
        spec = TensorSpec(
            size=output_spec.size,
            stride=output_spec.stride,
            dtype=output_spec.dtype,
        )
        specs: list[Spec] = [spec, spec, spec]
        if random.random() < 0.3:
            idx = random.randint(0, 2)
            bcast_size = random_broadcast_shape(tuple(output_spec.size))
            if bcast_size != tuple(output_spec.size):
                specs[idx] = TensorSpec(
                    size=bcast_size,
                    stride=contiguous_stride(bcast_size),
                    dtype=output_spec.dtype,
                )
        return specs

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Returns one line applying the torch function to the three inputs.
        # Raises ValueError unless exactly three input names are given.
        if len(input_names) != 3:
            raise ValueError(
                f"{self.__class__.__name__} requires exactly 3 input tensors"
            )
        return (
            f"{output_name} = {self.torch_op_name}("
            f"{input_names[0]}, {input_names[1]}, {input_names[2]})"
        )


# ---------------------------------------------------------------------------
# Subclass factories
# ---------------------------------------------------------------------------


def _class_name_from_op(op_name: str) -> str:
    # [FUNCTION DOC] HELPER: Derives a class name from a torch function path by dropping the
    # torch prefix, capitalizing each underscore-separated part, and appending Operator.
    """Convert ``"torch.nan_to_num"`` -> ``"NanToNumOperator"``."""
    suffix = op_name.replace("torch.", "")
    return "".join(p.capitalize() for p in suffix.split("_")) + "Operator"


def _make_subclass(
    base: type[Operator],
    op_name: str,
    extra_attrs: dict[str, Any] | None = None,
) -> type[Operator]:
    # [FUNCTION DOC] HELPER: Dynamically creates a subclass of base named from the op name,
    # whose constructor calls the base constructor with the op name. Optional extra attributes
    # (e.g. requires_float, scalar positions) are merged into the class namespace. The class is
    # relabelled as belonging to this module so registry discovery finds it. Returns the class.
    """Build a concrete Operator subclass for ``op_name`` deriving from ``base``."""
    cls_name = _class_name_from_op(op_name)

    def _init(self: Operator) -> None:
        # [FUNCTION DOC] HELPER: Constructor of generated classes; initializes the base with the
        # captured op name.
        base.__init__(self, op_name)

    attrs: dict[str, Any] = {
        "__init__": _init,
    }
    if extra_attrs:
        attrs.update(extra_attrs)
    cls = type(cls_name, (base,), attrs)
    cls.__module__ = __name__
    return cls


# ---------------------------------------------------------------------------
# Unary subclasses
# ---------------------------------------------------------------------------

# Float-only unary ops (per plan filtering rules).
_UNARY_FLOAT_OPS: tuple[str, ...] = (
    "torch.acos",
    "torch.acosh",
    "torch.angle",
    "torch.arccos",
    "torch.arccosh",
    "torch.arcsin",
    "torch.arcsinh",
    "torch.arctan",
    "torch.arctanh",
    "torch.asin",
    "torch.asinh",
    "torch.atan",
    "torch.atanh",
    "torch.cos",
    "torch.cosh",
    "torch.deg2rad",
    "torch.erf",
    "torch.erfc",
    "torch.erfinv",
    "torch.exp",
    "torch.exp2",
    "torch.expm1",
    "torch.fix",
    "torch.frac",
    "torch.log",
    "torch.log10",
    "torch.log1p",
    "torch.log2",
    "torch.rad2deg",
    "torch.reciprocal",
    "torch.rsqrt",
    "torch.sin",
    "torch.sinh",
    "torch.sqrt",
    "torch.tan",
)

# Numeric unary ops that work on int + float (no bool).
_UNARY_NUMERIC_OPS: tuple[str, ...] = (
    "torch.abs",
    "torch.absolute",
    "torch.ceil",
    "torch.floor",
    "torch.neg",
    "torch.negative",
    "torch.real",
    "torch.sgn",
    "torch.sign",
    "torch.square",
    "torch.trunc",
)


for _op in _UNARY_FLOAT_OPS:
    globals()[_class_name_from_op(_op)] = _make_subclass(
        UnaryElementwiseOperatorBase, _op, {"requires_float": True}
    )

for _op in _UNARY_NUMERIC_OPS:
    globals()[_class_name_from_op(_op)] = _make_subclass(
        UnaryElementwiseOperatorBase, _op
    )


class LogitOperator(UnaryElementwiseOperatorBase):
    # [CLASS DOC] Float-only logit with a random epsilon. State: _eps, set by
    # fuzz_inputs_specs and consumed (then cleared) by codegen.
    requires_float = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the name torch.logit and initializes the stashed epsilon.
        super().__init__("torch.logit")
        self._eps: float | None = None

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Draws epsilon uniformly from 1e-7 to 1e-2 (rounded to eight decimals),
        # stores it, then returns the base unary input specs.
        self._eps = round(random.uniform(1e-7, 1e-2), 8)
        return super().fuzz_inputs_specs(output_spec)

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Returns one line calling torch.logit on the first input with the
        # stashed epsilon, then clears the stashed value.
        eps = self._eps
        self._eps = None
        return f"{output_name} = torch.logit({input_names[0]}, eps={eps!r})"


class NanToNumOperator(UnaryElementwiseOperatorBase):
    # [CLASS DOC] Float-only nan_to_num with random replacement values. State: _nan, _posinf and
    # _neginf, set by fuzz_inputs_specs and consumed (then cleared) by codegen.
    requires_float = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the name torch.nan_to_num and initializes the three stashed
        # replacement values to None.
        super().__init__("torch.nan_to_num")
        self._nan: float | None = None
        self._posinf: float | None = None
        self._neginf: float | None = None

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Draws replacements for nan (-10..10), positive infinity (0..100) and
        # negative infinity (-100..0), each rounded to four decimals, stores them, then returns
        # the base unary input specs.
        self._nan = round(random.uniform(-10.0, 10.0), 4)
        self._posinf = round(random.uniform(0.0, 100.0), 4)
        self._neginf = round(random.uniform(-100.0, 0.0), 4)
        return super().fuzz_inputs_specs(output_spec)

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Returns one line calling torch.nan_to_num on the first input with the
        # stashed nan, posinf and neginf values, then clears them.
        nan, posinf, neginf = self._nan, self._posinf, self._neginf
        self._nan = self._posinf = self._neginf = None
        return (
            f"{output_name} = torch.nan_to_num({input_names[0]}, "
            f"nan={nan!r}, posinf={posinf!r}, neginf={neginf!r})"
        )


class RoundOperator(UnaryElementwiseOperatorBase):
    # [CLASS DOC] Float-only round with a random number of decimals. State: _decimals, set by
    # fuzz_inputs_specs and consumed (then cleared) by codegen.
    requires_float = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the name torch.round and initializes the stashed decimals.
        super().__init__("torch.round")
        self._decimals: int | None = None

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Draws an integer number of decimals from 0 to 4, stores it, then
        # returns the base unary input specs.
        self._decimals = random.randint(0, 4)
        return super().fuzz_inputs_specs(output_spec)

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Returns one line calling torch.round on the first input with the
        # stashed decimals, then clears the stashed value.
        decimals = self._decimals
        self._decimals = None
        return f"{output_name} = torch.round({input_names[0]}, decimals={decimals})"


# ---------------------------------------------------------------------------
# Bool-output unary subclasses
# ---------------------------------------------------------------------------

_BOOL_UNARY_OPS: tuple[str, ...] = (
    "torch.isfinite",
    "torch.isinf",
    "torch.isnan",
    "torch.isneginf",
    "torch.isposinf",
    "torch.signbit",
)

for _op in _BOOL_UNARY_OPS:
    globals()[_class_name_from_op(_op)] = _make_subclass(
        BoolUnaryElementwiseOperatorBase, _op
    )


# ---------------------------------------------------------------------------
# Binary subclasses
# ---------------------------------------------------------------------------

# Float-only binary ops — both args must be tensors.
_BINARY_FLOAT_TENSOR_ONLY_OPS: tuple[str, ...] = (
    "torch.arctan2",
    "torch.atan2",
    "torch.hypot",
    "torch.logaddexp",
    "torch.logaddexp2",
    "torch.nextafter",
)

# Float-only binary ops — scalar allowed in position 1 (the `other` arg).
_BINARY_FLOAT_SCALAR_OTHER_OPS: tuple[str, ...] = ("torch.copysign",)

# Float-only binary ops — scalar allowed in either position.
_BINARY_FLOAT_SCALAR_BOTH_OPS: tuple[str, ...] = ("torch.xlogy",)

# Numeric binary ops — scalar allowed in either position.
_BINARY_NUMERIC_OPS: tuple[str, ...] = (
    "torch.floor_divide",
    "torch.multiply",
    "torch.remainder",
    "torch.subtract",
)

# Numeric binary ops — both args must be tensors.
_BINARY_NUMERIC_TENSOR_ONLY_OPS: tuple[str, ...] = (
    "torch.fmax",
    "torch.fmin",
    "torch.maximum",
    "torch.minimum",
)

# Numeric binary ops — scalar allowed in position 1 only.
_BINARY_NUMERIC_SCALAR_OTHER_OPS: tuple[str, ...] = ("torch.fmod",)

for _op in _BINARY_FLOAT_TENSOR_ONLY_OPS:
    globals()[_class_name_from_op(_op)] = _make_subclass(
        FloatOnlyPointwiseOperatorBase, _op
    )

for _op in _BINARY_FLOAT_SCALAR_OTHER_OPS:
    globals()[_class_name_from_op(_op)] = _make_subclass(
        FloatOnlyPointwiseOperatorBase, _op, {"_scalar_input_positions": (1,)}
    )

for _op in _BINARY_FLOAT_SCALAR_BOTH_OPS:
    globals()[_class_name_from_op(_op)] = _make_subclass(
        FloatOnlyPointwiseOperatorBase, _op, {"_scalar_input_positions": (0, 1)}
    )

for _op in _BINARY_NUMERIC_OPS:
    globals()[_class_name_from_op(_op)] = _make_subclass(PointwiseOperatorBase, _op)

for _op in _BINARY_NUMERIC_TENSOR_ONLY_OPS:
    globals()[_class_name_from_op(_op)] = _make_subclass(
        PointwiseOperatorBase, _op, {"_scalar_input_positions": ()}
    )

for _op in _BINARY_NUMERIC_SCALAR_OTHER_OPS:
    globals()[_class_name_from_op(_op)] = _make_subclass(
        PointwiseOperatorBase, _op, {"_scalar_input_positions": (1,)}
    )


class PowOperator(PointwiseOperatorBase):
    # [CLASS DOC] Upstream pointwise binary operator specialised for torch.pow; for non-float
    # outputs the exponent is wrapped in an absolute value so integer powers are non-negative.
    """torch.pow with non-negative exponents for integer bases.

    Negative integer exponents produce fractions that cannot be
    represented in integer dtypes; PyTorch raises RuntimeError.
    Uses Python abs() instead of torch.abs() so scalar exponents work.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the operator under the name torch.pow.
        super().__init__("torch.pow")

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] For a tensor output with a non-floating dtype, returns one line calling
        # torch.pow with the second input wrapped in the builtin absolute value; otherwise defers
        # to the base class code generation.
        if (
            isinstance(output_spec, TensorSpec)
            and output_spec.dtype not in FLOAT_DTYPES
        ):
            return f"{output_name} = torch.pow({input_names[0]}, abs({input_names[1]}))"
        return super().codegen(output_name, input_names, output_spec)


class DivideOperator(PointwiseOperatorBase):
    # [CLASS DOC] Pointwise torch.divide with a random rounding mode. State: _rounding_mode,
    # set by fuzz_inputs_specs and consumed (then cleared) by codegen.
    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the name torch.divide and initializes the stashed mode.
        super().__init__("torch.divide")
        self._rounding_mode: str | None = None

    def fuzz_inputs_specs(self, output_spec: Spec, num_inputs: int = 2) -> list[Spec]:
        # [FUNCTION DOC] Picks a rounding mode at random (default, trunc or floor), stores it,
        # then returns the base class input specs for the given input count.
        self._rounding_mode = random.choice(["default", "trunc", "floor"])
        return super().fuzz_inputs_specs(output_spec, num_inputs)

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Gets the base class code line and, unless the stashed mode is the
        # default, appends a rounding-mode keyword by replacing the final closing parenthesis.
        # Clears the stashed mode.
        # KNOWN ISSUE: relies on the base code line ending with a closing parenthesis; if the
        # stashed mode is None (no prior fuzz call) a rounding mode of None is appended.
        rm = self._rounding_mode
        self._rounding_mode = None
        base = super().codegen(output_name, input_names, output_spec)
        if rm != "default":
            base = base[:-1] + f", rounding_mode={rm!r})"
        return base


class TrueDivideOperator(PointwiseOperatorBase):
    # [CLASS DOC] Plain pointwise torch.true_divide using the upstream base behavior.
    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the operator under the name torch.true_divide.
        super().__init__("torch.true_divide")


class RsubOperator(PointwiseOperatorBase):
    # [CLASS DOC] Pointwise torch.rsub (reversed subtraction) that allows a scalar in the second
    # position and sometimes adds an alpha multiplier. State: _alpha, set by fuzz_inputs_specs
    # and consumed (then cleared) by codegen.
    _scalar_input_positions = (1,)

    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the name torch.rsub and initializes the stashed alpha.
        super().__init__("torch.rsub")
        self._alpha: float | int | None = None

    def fuzz_inputs_specs(self, output_spec: Spec, num_inputs: int = 2) -> list[Spec]:
        # [FUNCTION DOC] With probability 0.3 draws alpha uniformly from -5 to 5 (four decimals;
        # truncated to an integer when the output is a non-float tensor) and stores it; otherwise
        # stores none. Then returns the base class input specs for the given input count.
        if random.random() < 0.3:
            alpha = round(random.uniform(-5.0, 5.0), 4)
            if (
                isinstance(output_spec, TensorSpec)
                and output_spec.dtype not in FLOAT_DTYPES
            ):
                alpha = int(alpha)
            self._alpha = alpha
        else:
            self._alpha = None
        return super().fuzz_inputs_specs(output_spec, num_inputs)

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Gets the base class code line and, if an alpha was stashed, appends an
        # alpha keyword by replacing the final closing parenthesis. Clears the stashed alpha.
        alpha = self._alpha
        self._alpha = None
        base = super().codegen(output_name, input_names, output_spec)
        if alpha is not None:
            base = base[:-1] + f", alpha={alpha!r})"
        return base


# ---------------------------------------------------------------------------
# Ternary subclasses (float-only)
# ---------------------------------------------------------------------------

_TERNARY_OPS: tuple[str, ...] = ("torch.lerp",)

for _op in _TERNARY_OPS:
    globals()[_class_name_from_op(_op)] = _make_subclass(
        TernaryElementwiseOperatorBase, _op
    )


class AddcdivOperator(TernaryElementwiseOperatorBase):
    # [CLASS DOC] Float-only torch.addcdiv with an optional random value multiplier. State:
    # _value, set by fuzz_inputs_specs and consumed (then cleared) by codegen.
    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the name torch.addcdiv and initializes the stashed value.
        super().__init__("torch.addcdiv")
        self._value: float | None = None

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] With probability 0.3 draws a multiplier uniformly from -5 to 5 (four
        # decimals), otherwise none; stores it, then returns the ternary base input specs.
        self._value = (
            round(random.uniform(-5.0, 5.0), 4) if random.random() < 0.3 else None
        )
        return super().fuzz_inputs_specs(output_spec)

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Returns one line calling torch.addcdiv on the three inputs, with a value
        # keyword only when a multiplier was stashed; clears the stashed value.
        # Raises ValueError unless exactly three input names are given.
        value = self._value
        self._value = None
        if len(input_names) != 3:
            raise ValueError("AddcdivOperator requires exactly 3 input tensors")
        base = (
            f"{output_name} = torch.addcdiv("
            f"{input_names[0]}, {input_names[1]}, {input_names[2]}"
        )
        if value is not None:
            return f"{base}, value={value!r})"
        return f"{base})"


class AddcmulOperator(TernaryElementwiseOperatorBase):
    # [CLASS DOC] Float-only torch.addcmul with an optional random value multiplier. State:
    # _value, set by fuzz_inputs_specs and consumed (then cleared) by codegen.
    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the name torch.addcmul and initializes the stashed value.
        super().__init__("torch.addcmul")
        self._value: float | None = None

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] With probability 0.3 draws a multiplier uniformly from -5 to 5 (four
        # decimals), otherwise none; stores it, then returns the ternary base input specs.
        self._value = (
            round(random.uniform(-5.0, 5.0), 4) if random.random() < 0.3 else None
        )
        return super().fuzz_inputs_specs(output_spec)

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Returns one line calling torch.addcmul on the three inputs, with a value
        # keyword only when a multiplier was stashed; clears the stashed value.
        # Raises ValueError unless exactly three input names are given.
        value = self._value
        self._value = None
        if len(input_names) != 3:
            raise ValueError("AddcmulOperator requires exactly 3 input tensors")
        base = (
            f"{output_name} = torch.addcmul("
            f"{input_names[0]}, {input_names[1]}, {input_names[2]}"
        )
        if value is not None:
            return f"{base}, value={value!r})"
        return f"{base})"


# ---------------------------------------------------------------------------
# Special-case operators
# ---------------------------------------------------------------------------


def _clamp_bound(output_spec: Spec) -> float | int:
    # [FUNCTION DOC] HELPER: Draws a random clamp bound uniformly from -10 to 10 (four decimals),
    # truncated to an integer when the output dtype is not floating (global random module).
    # Raises AssertionError if output_spec is not a TensorSpec.
    if not isinstance(output_spec, TensorSpec):
        raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
    bound = round(random.uniform(-10.0, 10.0), 4)
    if output_spec.dtype not in FLOAT_DTYPES:
        bound = int(bound)
    return bound


class ClampMaxOperator(Operator):
    # [CLASS DOC] torch.clamp_max with a random scalar upper bound chosen at code generation
    # time (not stored on the instance). Output: any non-bool tensor dtype; one input.
    """``torch.clamp_max(input, max)`` — random scalar bound."""

    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the operator under the name torch.clamp_max.
        super().__init__("torch.clamp_max")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property; returns the operator name, which is the torch function path.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] True for a TensorSpec whose dtype is not bool.
        if not isinstance(output_spec, TensorSpec):
            return False
        return output_spec.dtype != torch.bool

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns one input spec copying the output size, stride and dtype.
        # Raises AssertionError for a non-tensor output spec.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=output_spec.dtype,
            )
        ]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Draws a random bound and returns one line calling torch.clamp_max on
        # the first input with that bound as the max keyword.
        # Generate the bound here (not as instance state): the same operator
        # instance serves multiple clamp nodes, so stashing the bound in
        # fuzz_inputs_specs and reading it back here yielded max=None.
        bound = _clamp_bound(output_spec)
        return f"{output_name} = torch.clamp_max({input_names[0]}, max={bound!r})"


class ClampMinOperator(Operator):
    # [CLASS DOC] torch.clamp_min with a random scalar lower bound chosen at code generation
    # time (not stored on the instance). Output: any non-bool tensor dtype; one input.
    """``torch.clamp_min(input, min)`` — random scalar bound."""

    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the operator under the name torch.clamp_min.
        super().__init__("torch.clamp_min")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property; returns the operator name, which is the torch function path.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] True for a TensorSpec whose dtype is not bool.
        if not isinstance(output_spec, TensorSpec):
            return False
        return output_spec.dtype != torch.bool

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns one input spec copying the output size, stride and dtype.
        # Raises AssertionError for a non-tensor output spec.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=output_spec.dtype,
            )
        ]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Draws a random bound and returns one line calling torch.clamp_min on
        # the first input with that bound as the min keyword.
        bound = _clamp_bound(output_spec)
        return f"{output_name} = torch.clamp_min({input_names[0]}, min={bound!r})"


class HeavisideOperator(Operator):
    # [CLASS DOC] Float-only torch.heaviside; inputs are a full-shape tensor and a zero-dim
    # tensor of the same dtype giving the value at zero. Stateless.
    """``torch.heaviside(input, values)`` — second arg is a 0-d tensor."""

    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the operator under the name torch.heaviside.
        super().__init__("torch.heaviside")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property; returns the operator name, which is the torch function path.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] True only for a TensorSpec with a floating dtype.
        if not isinstance(output_spec, TensorSpec):
            return False
        if output_spec.dtype == torch.bool:
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns [full-shape spec copying the output, zero-dim spec of the same
        # dtype]. Raises ValueError for a non-tensor output spec.
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("HeavisideOperator can only produce TensorSpec outputs")
        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=output_spec.dtype,
            ),
            TensorSpec(size=(), stride=(), dtype=output_spec.dtype),
        ]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Returns one line calling torch.heaviside on the two inputs.
        # Raises ValueError unless exactly two input names are given.
        if len(input_names) != 2:
            raise ValueError("HeavisideOperator requires exactly 2 input tensors")
        return f"{output_name} = torch.heaviside({input_names[0]}, {input_names[1]})"


class LdexpOperator(Operator):
    # [CLASS DOC] Float-only torch.ldexp; the first input matches the output and the second is
    # an integer exponent tensor (signed int8 to int64, chosen randomly) of the output's size.
    """``torch.ldexp(input, other)`` — second arg is an integer exponent."""

    _EXP_DTYPES: tuple[torch.dtype, ...] = (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )

    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the operator under the name torch.ldexp.
        super().__init__("torch.ldexp")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property; returns the operator name, which is the torch function path.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] True only for a TensorSpec with a floating dtype.
        if not isinstance(output_spec, TensorSpec):
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns [spec copying the output, spec with the output's size and stride
        # and a random integer dtype]. Raises AssertionError for a non-tensor output spec.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=output_spec.dtype,
            ),
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=random.choice(self._EXP_DTYPES),
            ),
        ]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Returns one line calling torch.ldexp on the first two inputs.
        return f"{output_name} = torch.ldexp({input_names[0]}, {input_names[1]})"
