# pyre-strict
# [MODULE DOC] Fuzzer operators for torch special functions on float tensors. Contains nine unary
# elementwise operators (digamma, i0, lgamma, sinc, four Bessel functions, i0e), a binary igamma
# operator that clamps inputs into its domain, and a Chebyshev polynomial base class with a T and
# a U subclass whose degree is drawn at random. Also defines a per-dtype lower clamp table for
# igamma. No import-time side effects beyond imports; operators are found by registry
# introspection.
"""Special-function Operator subclasses.

Models torch ops in the "Special Functions" category. All 12 ops
are float-only and produce a TensorSpec output of the same
shape/stride/dtype as their primary input (the `x` argument).

Three sub-groupings:

  * 9 unary float-only ops -- thin subclasses of
    UnaryElementwiseOperatorBase from elementwise_math.py with
    `requires_float = True`. The base's `codegen` template works
    for `torch.special.X` ops because Python evaluates
    `torch.special.bessel_j0` as a single attribute-lookup
    expression. (digamma, i0, lgamma, sinc, bessel_j0, bessel_j1,
    bessel_y0, bessel_y1, i0e)

  * IgammaOperator -- direct Operator subclass that emits
    `torch.clamp` on both inputs in codegen to satisfy
    `torch.igamma`'s domain constraints (a > 0, x >= 0).

  * ChebyshevPolynomialTOperator / ChebyshevPolynomialUOperator --
    share ChebyshevPolynomialOperatorBase. Pick a random small
    non-negative int n (0..5) at fuzz_inputs_specs time.
"""

from __future__ import annotations

import random

import torch

from torchfuzz.operators._dtypes import (
    contiguous_stride,
    FLOAT_DTYPES,
    random_broadcast_shape,
)
from torchfuzz.operators.base import Operator
from torchfuzz.operators.elementwise_math import UnaryElementwiseOperatorBase
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


# Per-dtype clamp min for the `a` argument of torch.igamma. 1e-6
# underflows to 0 in bfloat16, defeating the a > 0 constraint;
# bf16 uses 1e-2 instead.
_IGAMMA_A_MIN: dict[torch.dtype, float] = {
    torch.bfloat16: 1e-2,
    torch.float16: 1e-6,
    torch.float32: 1e-6,
    torch.float64: 1e-6,
}


# ---------------------------------------------------------------------------
# Group A: 9 unary float-only ops
# ---------------------------------------------------------------------------


class DigammaOperator(UnaryElementwiseOperatorBase):
    # [CLASS DOC] Unary float-only operator for torch.digamma. Codegen feeds the absolute value of
    # the input plus a small epsilon so values stay positive and away from the poles.
    """torch.digamma(x) -- logarithmic derivative of the gamma function.

    Wraps input in ``torch.abs(x) + 1e-6`` to keep values positive and
    away from the poles at non-positive integers.
    """

    requires_float = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.digamma.
        super().__init__("torch.digamma")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Return one line of source applying digamma to the absolute value of the
        # first input plus 1e-6. Extra input names and the output spec are ignored.
        return f"{output_name} = torch.digamma(torch.abs({input_names[0]}) + 1e-6)"


class I0Operator(UnaryElementwiseOperatorBase):
    # [CLASS DOC] Unary float-only operator for torch.i0; uses the base class code generation.
    """torch.i0(x) -- modified Bessel function of the first kind, order 0."""

    requires_float = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.i0.
        super().__init__("torch.i0")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name


class LgammaOperator(UnaryElementwiseOperatorBase):
    # [CLASS DOC] Unary float-only operator for torch.lgamma. Codegen feeds the absolute value of
    # the input plus a small epsilon to avoid the poles.
    """torch.lgamma(x) -- natural log of the absolute value of the gamma function.

    Wraps input in ``torch.abs(x) + 1e-6`` to keep values positive and
    away from the poles at non-positive integers.
    """

    requires_float = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.lgamma.
        super().__init__("torch.lgamma")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Return one line of source applying lgamma to the absolute value of the
        # first input plus 1e-6. Extra input names and the output spec are ignored.
        return f"{output_name} = torch.lgamma(torch.abs({input_names[0]}) + 1e-6)"


class SincOperator(UnaryElementwiseOperatorBase):
    # [CLASS DOC] Unary float-only operator for torch.sinc; uses the base class code generation.
    """torch.sinc(x) -- normalized sinc function: sin(pi*x) / (pi*x)."""

    requires_float = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.sinc.
        super().__init__("torch.sinc")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name


class _BesselOperatorBase(UnaryElementwiseOperatorBase):
    # [CLASS DOC] HELPER: Shared base for the four Bessel operators; restricts the output dtype to
    # float32 or float64 because CPU kernels support only those.
    """Shared base for bessel_j0/j1/y0/y1 -- CPU kernels only support fp32/fp64."""

    requires_float = True

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Return True only for tensor specs with dtype float32 or float64.
        if not isinstance(output_spec, TensorSpec):
            return False
        return output_spec.dtype in (torch.float32, torch.float64)


class BesselJ0Operator(_BesselOperatorBase):
    # [CLASS DOC] Unary operator for torch.special.bessel_j0 on float32/float64 tensors.
    """torch.special.bessel_j0(x) -- Bessel function of the first kind, order 0."""

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.special.bessel_j0.
        super().__init__("torch.special.bessel_j0")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name


class BesselJ1Operator(_BesselOperatorBase):
    # [CLASS DOC] Unary operator for torch.special.bessel_j1 on float32/float64 tensors.
    """torch.special.bessel_j1(x) -- Bessel function of the first kind, order 1."""

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.special.bessel_j1.
        super().__init__("torch.special.bessel_j1")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name


class BesselY0Operator(_BesselOperatorBase):
    # [CLASS DOC] Unary operator for torch.special.bessel_y0 on float32/float64 tensors. Codegen
    # feeds the absolute value of the input plus a small epsilon to avoid the singularity.
    """torch.special.bessel_y0(x) -- Bessel function of the second kind, order 0.

    Wraps input in ``torch.abs(x) + 1e-6`` because bessel_y0 returns
    -inf at x=0 and NaN for x<0.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.special.bessel_y0.
        super().__init__("torch.special.bessel_y0")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Return source applying bessel_y0 to the absolute value of the first input
        # plus 1e-6. Extra input names and the output spec are ignored.
        return (
            f"{output_name} = torch.special.bessel_y0("
            f"torch.abs({input_names[0]}) + 1e-6)"
        )


class BesselY1Operator(_BesselOperatorBase):
    # [CLASS DOC] Unary operator for torch.special.bessel_y1 on float32/float64 tensors. Codegen
    # feeds the absolute value of the input plus a small epsilon to avoid the singularity.
    """torch.special.bessel_y1(x) -- Bessel function of the second kind, order 1.

    Wraps input in ``torch.abs(x) + 1e-6`` because bessel_y1 returns
    -inf at x=0 and NaN for x<0.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.special.bessel_y1.
        super().__init__("torch.special.bessel_y1")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Return source applying bessel_y1 to the absolute value of the first input
        # plus 1e-6. Extra input names and the output spec are ignored.
        return (
            f"{output_name} = torch.special.bessel_y1("
            f"torch.abs({input_names[0]}) + 1e-6)"
        )


class I0eOperator(UnaryElementwiseOperatorBase):
    # [CLASS DOC] Unary float-only operator for torch.special.i0e; uses the base code generation.
    """torch.special.i0e(x) -- exponentially scaled modified Bessel function I0e."""

    requires_float = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.special.i0e.
        super().__init__("torch.special.i0e")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name


# ---------------------------------------------------------------------------
# Group B: IgammaOperator
# ---------------------------------------------------------------------------


class IgammaOperator(Operator):
    # [CLASS DOC] Binary operator for torch.igamma producing float tensors. Stateless; both inputs
    # match the output spec except one may be replaced by a broadcastable smaller shape. Codegen
    # clamps the first argument from below with a per-dtype minimum and the second at zero.
    """torch.igamma(a, x) -- regularized lower incomplete gamma function.

    Requires a > 0 and x >= 0. Inputs are clamped at codegen time:
      - a is clamped with a per-dtype `min` (1e-2 for bfloat16,
        1e-6 for all other float dtypes).
      - x is clamped to min=0.0 for all float dtypes.

    Both inputs are float tensors matching the output spec. No
    per-call statefulness; the clamp bounds are fixed-per-dtype
    constants.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.igamma.
        super().__init__("torch.igamma")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Return True only for tensor specs whose dtype is in the float dtype list.
        return isinstance(output_spec, TensorSpec) and output_spec.dtype in FLOAT_DTYPES

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Return two tensor specs copying the output size, stride and dtype. With
        # probability 0.3 one randomly chosen input is replaced by a contiguous spec of a randomly
        # reduced broadcastable shape (if a different shape was drawn). Raises AssertionError if
        # output_spec is not a tensor spec. Consumes global random state.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        spec = TensorSpec(
            size=output_spec.size,
            stride=output_spec.stride,
            dtype=output_spec.dtype,
        )
        specs: list[Spec] = [spec, spec]
        if random.random() < 0.3:
            idx = random.randint(0, 1)
            bcast_size = random_broadcast_shape(tuple(output_spec.size))
            if bcast_size != tuple(output_spec.size):
                specs[idx] = TensorSpec(
                    size=bcast_size,
                    stride=contiguous_stride(bcast_size),
                    dtype=output_spec.dtype,
                )
        return specs

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        # [FUNCTION DOC] Return source calling igamma on the first input clamped to the minimum
        # for the output dtype (looked up in the per-dtype table) and the second clamped at 0.0.
        # Raises AssertionError for a non-tensor spec, KeyError for a dtype missing from the
        # table (anything outside bfloat16, float16, float32, float64).
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        a_min = _IGAMMA_A_MIN[output_spec.dtype]
        return (
            f"{output_name} = torch.igamma("
            f"torch.clamp({input_names[0]}, min={a_min}), "
            f"torch.clamp({input_names[1]}, min=0.0))"
        )


# ---------------------------------------------------------------------------
# Group C: Chebyshev polynomial base + 2 subclasses
# ---------------------------------------------------------------------------


class ChebyshevPolynomialOperatorBase(Operator):
    # [CLASS DOC] Base for Chebyshev T and U operators producing float32/float64 tensors. State:
    # the pending polynomial degree (None until chosen), set while generating input specs and
    # consumed and reset by code generation, so spec generation must precede each codegen call.
    # Its name ends in Base so the registry skips it.
    """Shared base for chebyshev_polynomial_t and chebyshev_polynomial_u.

    Both ops have signature ``op(x, n)`` where x is a float tensor and
    n is the non-negative integer polynomial degree. n is picked at
    fuzz_inputs_specs time via random.randint(0, 5), stashed on
    self._n, and emitted inline at codegen time as
    `torch.tensor(n, dtype=torch.int32)`. Only x is a fuzzed input
    tensor.

    The class name suffix `Base` is excluded from the introspection-
    based registration.
    """

    def __init__(self, name: str) -> None:
        # [FUNCTION DOC] Register under the given name and start with no degree chosen.
        super().__init__(name)
        self._n: int | None = None

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Return True only for tensor specs with dtype float32 or float64.
        if not isinstance(output_spec, TensorSpec):
            return False
        return output_spec.dtype in (torch.float32, torch.float64)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Draw the polynomial degree as a random integer from 0 to 5 inclusive,
        # store it on the instance, and return one input spec copying the output size, stride and
        # dtype. Raises AssertionError if output_spec is not a tensor spec. Mutates instance state
        # and consumes global random state.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        self._n = random.randint(0, 5)
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
        # [FUNCTION DOC] Return source calling the torch op with the first input and a scalar
        # int32 tensor holding the stored degree, then reset the stored degree to None (always,
        # even on error). If no degree was stored, the generated text contains the word None as
        # the degree.
        # KNOWN ISSUE: calling codegen without a prior spec generation silently emits an invalid
        # degree instead of raising an error.
        # is not None -- NOT truthiness -- n=0 is a valid draw.
        try:
            return (
                f"{output_name} = {self.torch_op_name}("
                f"{input_names[0]}, "
                f"torch.tensor({self._n}, dtype=torch.int32))"
            )
        finally:
            self._n = None


class ChebyshevPolynomialTOperator(ChebyshevPolynomialOperatorBase):
    # [CLASS DOC] Operator for torch.special.chebyshev_polynomial_t (first kind), random degree.
    """torch.special.chebyshev_polynomial_t(x, n) -- Chebyshev T_n(x)."""

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.special.chebyshev_polynomial_t.
        super().__init__("torch.special.chebyshev_polynomial_t")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name


class ChebyshevPolynomialUOperator(ChebyshevPolynomialOperatorBase):
    # [CLASS DOC] Operator for torch.special.chebyshev_polynomial_u (second kind), random degree.
    """torch.special.chebyshev_polynomial_u(x, n) -- Chebyshev U_n(x)."""

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.special.chebyshev_polynomial_u.
        super().__init__("torch.special.chebyshev_polynomial_u")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name
