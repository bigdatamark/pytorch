# pyre-strict
# [MODULE DOC] Fuzzer operators for shape-preserving manipulation ops: conj_physical, fill, flip
# and masked_fill. Random choices (scalar value, flip dims) are drawn during input-spec
# generation, stashed on the operator and consumed (then cleared) in codegen. Also holds a
# helper producing random scalar literals per dtype. No import-time side effects.
# KNOWN ISSUE: the text above claims codegen asserts that stashed values are present; the code
# does not check this.
"""Tensor manipulation & indexing Operator subclasses.

Models torch ops in the "Tensor Manipulation & Indexing" category
that produce a TensorSpec output of the same shape/stride as their
primary input:

  - torch.conj_physical
  - torch.fill (random scalar per call)
  - torch.flip (random dims per call)
  - torch.masked_fill (random scalar per call)

torch.nonzero reuses upstream's NonzeroOperator and has no class in
this file.

Each of the three random-stash operators initialises its stashed
attribute(s) to None in __init__, populates them in fuzz_inputs_specs,
defensively asserts they are populated in codegen using `is not None`
(NOT truthiness -- falsy stash values like 0, 0.0, False are valid
draws), and clears them back to None on codegen exit. This mirrors
the per-call statefulness pattern used by WhereOperator /
LogcumsumexpOperator and is safe under upstream's CURRENT sequential
spec-then-codegen lifecycle. The clear-on-exit defends against
accidental re-entry of the same instance under that lifecycle.
"""

from __future__ import annotations

import random

import torch

from torchfuzz.operators._dtypes import (
    contiguous_stride,
    random_broadcast_shape,
    scalar_repr,
)
from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


def _random_scalar_for_dtype(dtype: torch.dtype) -> object:
    # [FUNCTION DOC] HELPER: Returns a random Python scalar suited to a dtype using the
    # global random module: a bool for bool, for float dtypes minus infinity (10 percent),
    # infinity (5 percent) or a uniform value in -10 to 10 rounded to 4 digits, a small
    # signed or unsigned integer for integer dtypes. Raises ValueError for any other dtype.
    # KNOWN ISSUE: the docstring omits the infinity outcomes for float dtypes.
    """Return a random Python scalar literal appropriate for ``dtype``.

    Uses the global ``random`` module so ``--seed`` reproduces.

    Bool                    -> random.choice([True, False])
    Float dtypes            -> round(uniform(-10.0, 10.0), 4)
    Signed integer dtypes   -> randint(-100, 100)
    Unsigned integer dtypes -> randint(0, 100)

    The narrow integer ranges keep emitted literals small, human-
    readable, and well within every supported integer dtype's
    representable range. The float branch rounds to 4 fractional
    digits to keep emitted code compact and stable across Python
    builds (repr() of a raw float can produce 16-digit literals).

    NOTE: Drift between callers' can_produce and the dtypes handled
    here raises loudly at fuzz time rather than silently emitting a
    wrong-typed literal.
    """
    if dtype == torch.bool:
        return random.choice([True, False])
    if dtype in (
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
    ):
        r = random.random()
        if r < 0.1:
            return float("-inf")
        if r < 0.15:
            return float("inf")
        return round(random.uniform(-10.0, 10.0), 4)
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        return random.randint(-100, 100)
    if dtype in (torch.uint8, torch.uint16, torch.uint32, torch.uint64):
        return random.randint(0, 100)
    raise ValueError(
        f"_random_scalar_for_dtype called with unsupported dtype {dtype}; "
        "extend can_produce of the caller and this helper together."
    )


class ConjPhysicalOperator(Operator):
    # [CLASS DOC] Elementwise conj_physical on one tensor of the same size/stride/dtype as the
    # output; excludes bool outputs. Stateless.
    """Single-tensor elementwise complex conjugate.

    For real-valued inputs this is a no-op; for complex-valued inputs
    it negates the imaginary component. Bool is excluded because
    torch.conj_physical's dispatch on bool is not defined.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Register under the name torch.conj_physical.
        super().__init__("torch.conj_physical")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Returns the registered operator name, which is the torch function
        # called in generated code.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True for tensor specs whose dtype is not bool.
        return isinstance(output_spec, TensorSpec) and output_spec.dtype != torch.bool

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns one input spec identical to the output spec (same size, stride
        # and dtype). Raises AssertionError if output_spec is not a TensorSpec.
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
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        # [FUNCTION DOC] Emits a line applying conj_physical to the single input.
        return f"{output_name} = torch.conj_physical({input_names[0]})"


class FillOperator(Operator):
    # [CLASS DOC] Operator for torch.fill: output is a copy of the input filled with one
    # random scalar. State: _value holds the scalar drawn in fuzz_inputs_specs and is cleared
    # by codegen.
    """Single-tensor torch.fill -- copy of input filled with a scalar.

    Picks a random scalar at fuzz_inputs_specs time keyed by output
    dtype; emits the literal inline at codegen time. All TensorSpec
    dtypes are supported (including bool -- handled by
    _random_scalar_for_dtype via random.choice([True, False])).
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Register under the name torch.fill and initialize the stashed
        # value to none.
        super().__init__("torch.fill")
        self._value: object | None = None

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Returns the registered operator name, which is the torch function
        # called in generated code.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True for any tensor spec.
        return isinstance(output_spec, TensorSpec)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Draws and stashes a random scalar for the output dtype, then returns
        # one input spec identical to the output spec. Raises AssertionError if output_spec is
        # not a TensorSpec.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        self._value = _random_scalar_for_dtype(output_spec.dtype)
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
        # [FUNCTION DOC] Emits a line filling the input with the stashed scalar rendered as a
        # literal, then clears the stash. Side effect: resets the stored value to none.
        # KNOWN ISSUE: the leading comment mentions an is-not-none check that is not performed.
        # is not None -- NOT truthiness -- falsy stash values
        # (0, 0.0, False) are valid draws.
        val = scalar_repr(self._value)
        self._value = None
        return f"{output_name} = torch.fill({input_names[0]}, {val})"


class FlipOperator(Operator):
    # [CLASS DOC] Operator for torch.flip: reverses element order along a random non-empty
    # subset of dims. State: _dims holds the chosen dims from fuzz_inputs_specs and is cleared
    # by codegen.
    """torch.flip(input, dims) -- reverse element order along dims.

    Picks a random non-empty subset of dims at fuzz_inputs_specs time
    using the global random module (so --seed reproduces). Output
    shape/stride/dtype match input.

    NOTE: torch.fliplr (== flip(dim=-1)) and torch.flipud (== flip(dim=0))
    are NOT modeled as separate operators. Both are Python wrappers
    that decompose to aten.flip with no extra ops, AND torch._refs.flip's
    canonicalize_dims call normalizes negative-dim forms to non-negative
    before lowering, AND Inductor registers a single decomp shared by
    all three entry points. Randomizing dims here therefore covers the
    identical compiler lowering paths. Adding the aliases as separate
    ops would only exercise additional Python frontend dispatch, which
    is not the fuzzer's target.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Register under the name torch.flip and initialize the stashed
        # dims to none.
        super().__init__("torch.flip")
        self._dims: tuple[int, ...] | None = None

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Returns the registered operator name, which is the torch function
        # called in generated code.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True for tensor specs with at least one dimension.
        return isinstance(output_spec, TensorSpec) and len(output_spec.size) >= 1

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Chooses a random count of dims and a sorted random subset, stashes it,
        # then returns one input spec identical to the output. Raises AssertionError if
        # output_spec is not a TensorSpec.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        ndim = len(output_spec.size)
        k = random.randint(1, ndim)
        self._dims = tuple(sorted(random.sample(range(ndim), k)))
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
        # [FUNCTION DOC] Emits a line flipping the input along the stashed dims, then clears the
        # stash. Side effect: resets the stored dims to none.
        # KNOWN ISSUE: the leading comment mentions an is-not-none check that is not performed;
        # if codegen runs without fuzz_inputs_specs the generated dims would be none.
        # is not None -- NOT truthiness -- empty tuples would be falsy
        # if a future maintainer ever extends the random range.
        out = f"{output_name} = torch.flip({input_names[0]}, dims={self._dims!r})"
        self._dims = None
        return out


class MaskedFillOperator(Operator):
    # [CLASS DOC] Operator for torch.masked_fill: a value tensor plus a bool mask, filled with
    # one random scalar. State: _value holds the scalar drawn in fuzz_inputs_specs and is
    # cleared by codegen.
    """torch.masked_fill(input, mask, value) -- replace masked elements.

    Two tensor inputs (the value tensor + a same-shape bool mask) and
    one random scalar literal stashed at fuzz_inputs_specs time. All
    TensorSpec dtypes are supported (including bool -- handled by
    _random_scalar_for_dtype via random.choice([True, False])).

    The mask is constructed with the same stride as the input tensor.
    TensorSpec.stride is element-count (not byte-count) and dtype-
    independent, so reusing the input stride for a bool mask correctly
    describes a contiguous bool tensor of the same shape. Verified
    against tensor_fuzzer.py: the materializer applies
    torch.as_strided(base_tensor, size, stride) where base_tensor is
    allocated with the requested dtype first, so stride is dtype-
    independent.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Register under the name torch.masked_fill and initialize the
        # stashed value to none.
        super().__init__("torch.masked_fill")
        self._value: object | None = None

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Returns the registered operator name, which is the torch function
        # called in generated code.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True for any tensor spec.
        return isinstance(output_spec, TensorSpec)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Draws and stashes a random scalar for the output dtype and returns two
        # specs: the input (same as output) and a bool mask that, with probability 0.3, gets a
        # random broadcastable shape instead of the output size; the mask is contiguous. Raises
        # AssertionError if output_spec is not a TensorSpec.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        self._value = _random_scalar_for_dtype(output_spec.dtype)
        input_spec = TensorSpec(
            size=output_spec.size,
            stride=output_spec.stride,
            dtype=output_spec.dtype,
        )
        mask_size = tuple(output_spec.size)
        if random.random() < 0.3:
            bcast_size = random_broadcast_shape(mask_size)
            if bcast_size != mask_size:
                mask_size = bcast_size
        mask_spec = TensorSpec(
            size=mask_size,
            stride=contiguous_stride(mask_size),
            dtype=torch.bool,
        )
        return [input_spec, mask_spec]

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        # [FUNCTION DOC] Emits a line calling masked_fill with the input, the mask and the
        # stashed scalar literal, then clears the stash. Side effect: resets the stored value
        # to none.
        # KNOWN ISSUE: the leading comment mentions an is-not-none check that is not performed.
        # is not None -- NOT truthiness -- falsy stash values
        # (0, 0.0, False) are valid draws.
        val = scalar_repr(self._value)
        self._value = None
        return (
            f"{output_name} = torch.masked_fill("
            f"{input_names[0]}, {input_names[1]}, {val})"
        )
