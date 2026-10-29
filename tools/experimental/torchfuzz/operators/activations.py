# pyre-strict
# [MODULE DOC] Activation-function operators for torchfuzz (all with float-only outputs):
# five plain unary activations built on the shared unary elementwise base, five operators that
# draw random hyperparameters during input-spec generation and stash them for code generation
# (celu, softshrink, hardtanh, softplus, threshold), and a prelu operator taking a
# per-channel weight tensor input. No import-time side effects.
"""Activation function Operator subclasses.

Models torch ops in the "Activation Functions" category whose user-
facing entry-point is the torch.nn.functional.X form (with two upstream
exceptions, torch.sigmoid and torch.tanh, which are already registered
upstream). All ops have float-only output dtypes.

The 7 ops in this category that ARE already registered upstream
(torch.nn.functional.{relu, leaky_relu, elu, gelu, silu}, torch.sigmoid,
torch.tanh) have NO class in this file -- they are still listed in
InductorFuzzTemplate.supported_ops, and check_exact_match.py auto-covers
them via the upstream registry.

Five plain unary activations (hardswish, logsigmoid, mish, relu6, selu)
reuse UnaryElementwiseOperatorBase from elementwise_math.py with
requires_float=True -- the base's ``torch.X(input)`` codegen template
works for ``torch.nn.functional.X`` ops because Python evaluates the
dotted name as a single attribute-lookup chain.

Six classes use the WhereOperator-style stash-on-self defense-in-depth
pattern to randomize hyperparameters per call: CeluOperator,
SoftshrinkOperator, HardtanhOperator, SoftplusOperator, ThresholdOperator.
PreluOperator does NOT stash scalars -- it draws a per-channel weight
TENSOR input via fuzz_inputs_specs.

Each stashing class initialises its stashed attribute(s) to None in
__init__, populates them in fuzz_inputs_specs, defensively asserts they
are populated in codegen using ``is not None`` (NOT truthiness -- falsy
draws like 0.0 are valid), and clears them back to None on codegen exit.
"""

from __future__ import annotations

import math
import random

from torchfuzz.operators._dtypes import FLOAT_DTYPES
from torchfuzz.operators.base import Operator
from torchfuzz.operators.elementwise_math import UnaryElementwiseOperatorBase
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


# ---------------------------------------------------------------------------
# Plain unary subclasses (5)
#
# These reuse UnaryElementwiseOperatorBase with requires_float=True.
# The base's codegen template (``torch.X(input)``) works for
# torch.nn.functional.X ops because Python evaluates
# ``torch.nn.functional.hardswish`` as a single attribute-lookup chain.
# ---------------------------------------------------------------------------


class HardswishOperator(UnaryElementwiseOperatorBase):
    # [CLASS DOC] Unary hardswish activation (torch.nn.functional.hardswish); float dtypes only.
    requires_float = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the operator under the functional hardswish name.
        super().__init__("torch.nn.functional.hardswish")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property; returns the operator name, which is the torch function path.
        return self.name


class LogsigmoidOperator(UnaryElementwiseOperatorBase):
    # [CLASS DOC] Unary log-sigmoid activation (torch.nn.functional.logsigmoid); float only.
    requires_float = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the operator under the functional logsigmoid name.
        super().__init__("torch.nn.functional.logsigmoid")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property; returns the operator name, which is the torch function path.
        return self.name


class MishOperator(UnaryElementwiseOperatorBase):
    # [CLASS DOC] Unary mish activation (torch.nn.functional.mish); float dtypes only.
    requires_float = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the operator under the functional mish name.
        super().__init__("torch.nn.functional.mish")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property; returns the operator name, which is the torch function path.
        return self.name


class Relu6Operator(UnaryElementwiseOperatorBase):
    # [CLASS DOC] Unary relu6 activation (torch.nn.functional.relu6); float dtypes only.
    requires_float = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the operator under the functional relu6 name.
        super().__init__("torch.nn.functional.relu6")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property; returns the operator name, which is the torch function path.
        return self.name


class SeluOperator(UnaryElementwiseOperatorBase):
    # [CLASS DOC] Unary selu activation (torch.nn.functional.selu); float dtypes only.
    requires_float = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the operator under the functional selu name.
        super().__init__("torch.nn.functional.selu")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property; returns the operator name, which is the torch function path.
        return self.name


# ---------------------------------------------------------------------------
# Hyperparameter-stashing subclasses (5) and per-channel-weight subclass (1)
# ---------------------------------------------------------------------------


class CeluOperator(Operator):
    # [CLASS DOC] Celu activation with a random alpha. State: _alpha, set by
    # fuzz_inputs_specs and consumed (then cleared) by codegen.
    """torch.nn.functional.celu(input, alpha) -- random alpha per call.

    NOTE: aten.celu's decomposition divides by alpha, so the
    drawn alpha is constrained to [0.01, 5.0] to avoid division by zero
    while exercising both small (steep-slope) and large (shallow-slope)
    real-world alpha values.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the operator name and initializes the stashed alpha to None.
        super().__init__("torch.nn.functional.celu")
        self._alpha: float | None = None

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property; returns the operator name, which is the torch function path.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] True only for a TensorSpec with a floating dtype.
        return isinstance(output_spec, TensorSpec) and output_spec.dtype in FLOAT_DTYPES

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Draws alpha uniformly from 0.01 to 5.0 (rounded to four decimals) and
        # stores it on the operator (global random module). Returns one input spec copying the
        # output size, stride and dtype. Raises AssertionError for a non-tensor output spec.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        self._alpha = round(random.uniform(0.01, 5.0), 4)
        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=output_spec.dtype,
            )
        ]

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        # [FUNCTION DOC] Returns one line applying celu to the first input with the stashed
        # alpha, then resets the stashed alpha to None. If no alpha was stashed the line shows
        # None for alpha.
        # is not None -- NOT truthiness -- 0.0 would be a valid float
        # if a future maintainer ever extends the random range to 0.
        out = (
            f"{output_name} = torch.nn.functional.celu("
            f"{input_names[0]}, alpha={self._alpha!r})"
        )
        self._alpha = None
        return out


class SoftshrinkOperator(Operator):
    # [CLASS DOC] Softshrink activation with a random lambda. State: _lambd, set by
    # fuzz_inputs_specs and consumed (then cleared) by codegen.
    """torch.nn.functional.softshrink(input, lambd) -- random lambd per call.

    NOTE: lambd must be >= 0 (PyTorch raises if lambd < 0). Drawn from
    [0.0, 2.0] uniformly to keep the active region within typical
    activation magnitudes.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the operator name and initializes the stashed lambda to None.
        super().__init__("torch.nn.functional.softshrink")
        self._lambd: float | None = None

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property; returns the operator name, which is the torch function path.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] True only for a TensorSpec with a floating dtype.
        return isinstance(output_spec, TensorSpec) and output_spec.dtype in FLOAT_DTYPES

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Draws lambd uniformly from 0.0 to 2.0 (rounded to four decimals) and
        # stores it on the operator (global random module). Returns one input spec copying the
        # output size, stride and dtype. Raises AssertionError for a non-tensor output spec.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        self._lambd = round(random.uniform(0.0, 2.0), 4)
        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=output_spec.dtype,
            )
        ]

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        # [FUNCTION DOC] Returns one line applying softshrink to the first input with the
        # stashed lambd, then resets the stashed value to None.
        # is not None -- NOT truthiness -- 0.0 is a valid lambd draw.
        out = (
            f"{output_name} = torch.nn.functional.softshrink("
            f"{input_names[0]}, lambd={self._lambd!r})"
        )
        self._lambd = None
        return out


class HardtanhOperator(Operator):
    # [CLASS DOC] Hardtanh activation with random bounds. State: _min_val and _max_val, set by
    # fuzz_inputs_specs (min strictly below max) and consumed (then cleared) by codegen.
    """torch.nn.functional.hardtanh(input, min_val, max_val) -- random bounds per call.

    NOTE: min_val < max_val strictly (== triggers a degenerate clamp).
    Drawn by sampling two values from [-3.0, 3.0] uniformly, sorting,
    and forcing strict inequality via ``math.nextafter(hi, math.inf)``
    if equal.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the operator name and initializes both stashed bounds to None.
        super().__init__("torch.nn.functional.hardtanh")
        self._min_val: float | None = None
        self._max_val: float | None = None

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property; returns the operator name, which is the torch function path.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] True only for a TensorSpec with a floating dtype.
        return isinstance(output_spec, TensorSpec) and output_spec.dtype in FLOAT_DTYPES

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Draws two values uniformly from -3 to 3 (rounded to four decimals),
        # orders them, nudges the upper bound to the next representable float if they are equal,
        # and stores them as the bounds (global random module). Returns one input spec copying
        # the output size, stride and dtype. Raises AssertionError for a non-tensor output spec.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        a = round(random.uniform(-3.0, 3.0), 4)
        b = round(random.uniform(-3.0, 3.0), 4)
        lo, hi = (a, b) if a < b else (b, a)
        if lo == hi:
            hi = math.nextafter(hi, math.inf)
        self._min_val = lo
        self._max_val = hi
        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=output_spec.dtype,
            )
        ]

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        # [FUNCTION DOC] Returns one line applying hardtanh to the first input with the stashed
        # min and max bounds, then resets both stashed values to None.
        # is not None on EACH attribute -- NOT truthiness -- 0.0 is valid.
        out = (
            f"{output_name} = torch.nn.functional.hardtanh("
            f"{input_names[0]}, "
            f"min_val={self._min_val!r}, max_val={self._max_val!r})"
        )
        self._min_val = None
        self._max_val = None
        return out


class SoftplusOperator(Operator):
    # [CLASS DOC] Softplus activation with random beta and threshold. State: _beta and
    # _threshold, set by fuzz_inputs_specs and consumed (then cleared) by codegen.
    """torch.nn.functional.softplus(input, beta, threshold).

    NOTE: beta must be > 0 (drawn from [0.1, 5.0]). threshold is drawn
    from [5.0, 30.0] (default 20).
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the operator name and initializes both stashed values to None.
        super().__init__("torch.nn.functional.softplus")
        self._beta: float | None = None
        self._threshold: float | None = None

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property; returns the operator name, which is the torch function path.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] True only for a TensorSpec with a floating dtype.
        return isinstance(output_spec, TensorSpec) and output_spec.dtype in FLOAT_DTYPES

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Draws beta uniformly from 0.1 to 5.0 and threshold from 5.0 to 30.0
        # (each rounded to four decimals), stores them on the operator (global random module).
        # Returns one input spec copying the output size, stride and dtype.
        # Raises AssertionError for a non-tensor output spec.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        self._beta = round(random.uniform(0.1, 5.0), 4)
        self._threshold = round(random.uniform(5.0, 30.0), 4)
        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=output_spec.dtype,
            )
        ]

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        # [FUNCTION DOC] Returns one line applying softplus to the first input with the stashed
        # beta and threshold, then resets both stashed values to None.
        out = (
            f"{output_name} = torch.nn.functional.softplus("
            f"{input_names[0]}, "
            f"beta={self._beta!r}, threshold={self._threshold!r})"
        )
        self._beta = None
        self._threshold = None
        return out


class ThresholdOperator(Operator):
    # [CLASS DOC] Threshold activation with random threshold and replacement value. State:
    # _threshold and _value, set by fuzz_inputs_specs and consumed (then cleared) by codegen.
    """torch.nn.functional.threshold(input, threshold, value).

    NOTE: threshold and value are independently drawn from [-3.0, 3.0]
    uniformly.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the operator name and initializes both stashed values to None.
        super().__init__("torch.nn.functional.threshold")
        self._threshold: float | None = None
        self._value: float | None = None

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property; returns the operator name, which is the torch function path.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] True only for a TensorSpec with a floating dtype.
        return isinstance(output_spec, TensorSpec) and output_spec.dtype in FLOAT_DTYPES

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Independently draws the threshold and the replacement value uniformly
        # from -3 to 3 (rounded to four decimals), stores them (global random module). Returns
        # one input spec copying the output size, stride and dtype.
        # Raises AssertionError for a non-tensor output spec.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        self._threshold = round(random.uniform(-3.0, 3.0), 4)
        self._value = round(random.uniform(-3.0, 3.0), 4)
        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=output_spec.dtype,
            )
        ]

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        # [FUNCTION DOC] Returns one line applying threshold to the first input with the stashed
        # threshold and value as positional arguments, then resets both to None.
        # is not None -- NOT truthiness -- 0.0 is a valid threshold/value draw.
        out = (
            f"{output_name} = torch.nn.functional.threshold("
            f"{input_names[0]}, {self._threshold!r}, {self._value!r})"
        )
        self._threshold = None
        self._value = None
        return out


class PreluOperator(Operator):
    # [CLASS DOC] PReLU activation taking two inputs: the activation input and a one-dim weight
    # tensor (per-channel when the output rank is at least 2, else a single element). Stateless.
    """torch.nn.functional.prelu(input, weight) -- random per-channel weight.

    NOTE: F.prelu is literally the same Python object as torch.prelu.
    The ``weight`` argument must be a 1-D tensor of size 1 OR of size
    input.size(1).

    For output rank >= 2, this operator emits a per-channel weight of
    shape (output_spec.size[1],). For rank 0 or 1, it falls back to a
    (1,) weight (scalar broadcast). Both forms are valid F.prelu inputs.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the operator under the functional prelu name.
        super().__init__("torch.nn.functional.prelu")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property; returns the operator name, which is the torch function path.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] True only for a TensorSpec with a floating dtype.
        return isinstance(output_spec, TensorSpec) and output_spec.dtype in FLOAT_DTYPES

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns [input spec, weight spec]. The input copies the output size,
        # stride and dtype. The weight is a contiguous one-dim tensor of the same dtype whose
        # length is the output's size at dim 1 when rank is at least 2, else 1.
        # Raises AssertionError for a non-tensor output spec.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        input_spec = TensorSpec(
            size=output_spec.size,
            stride=output_spec.stride,
            dtype=output_spec.dtype,
        )
        if len(output_spec.size) >= 2:
            weight_size = (output_spec.size[1],)
        else:
            weight_size = (1,)
        weight_spec = TensorSpec(
            size=weight_size,
            stride=(1,),
            dtype=output_spec.dtype,
        )
        return [input_spec, weight_spec]

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        # [FUNCTION DOC] Returns one line applying prelu to the first input with the second input
        # as the weight. output_spec is unused.
        return (
            f"{output_name} = torch.nn.functional.prelu("
            f"{input_names[0]}, {input_names[1]})"
        )
