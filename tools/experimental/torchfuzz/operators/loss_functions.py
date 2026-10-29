# pyre-strict
# [MODULE DOC] Fuzzer operators for torch.nn.functional loss functions: elementwise losses (mse,
# huber, smooth_l1, bce, bce_with_logits, soft_margin, hinge_embedding), margin_ranking and the
# classification losses nll_loss and cross_entropy. All use no reduction so output shapes follow
# the inputs; range-constrained inputs are transformed inside the generated code. No import-time
# side effects.
"""Loss function Operator subclasses.

Covers elementwise losses (mse, huber, smooth_l1, binary_cross_entropy,
binary_cross_entropy_with_logits, soft_margin, hinge_embedding), a
three-input ranking loss (margin_ranking), and classification losses
(nll_loss, cross_entropy).

All operators use ``reduction='none'`` so the output shape is predictable
from the input shape and the fuzzer's top-down graph construction can
compose loss ops as interior nodes at arbitrary depth.

Subclasses that need constrained input ranges (e.g. BCE requires [0, 1])
or +1/-1 target values transform the raw fuzzer-generated tensors inline
in codegen rather than relying on the materializer to produce valid values.
"""

from __future__ import annotations

import random

import torch

from torchfuzz.operators._dtypes import FLOAT_DTYPES
from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


def _pm1_target_expr(target_name: str) -> str:
    # [FUNCTION DOC] HELPER: Builds source text for a target expression mapping values to +1
    # where the target is non-negative and -1 otherwise, using 0-dim tensors of the target's
    # dtype so the dtype is preserved. Takes the target variable name, returns the
    # expression string.
    """Emit ``torch.where(target >= 0, +1, -1)`` with ``+-1`` typed to ``target.dtype``.

    Used by ``soft_margin_loss``, ``hinge_embedding_loss``, and
    ``margin_ranking_loss``, which all require a +-1 target.

    The +1/-1 scalars are emitted as 0-dim tensors typed to ``target.dtype`` so the
    wrapped target keeps the input dtype. The naive ``torch.where(cond, 1.0, -1.0)``
    would collapse to the default float dtype (= float32) for any non-default input
    dtype, breaking the spec contract for f16/bf16/f64. ``torch.where`` broadcasts
    0-dim tensors transparently, so this matches the bare-scalar semantics at a
    fraction of the allocation cost of ``torch.ones_like(target)``.
    """
    return (
        f"torch.where({target_name} >= 0, "
        f"torch.tensor(1.0, dtype={target_name}.dtype), "
        f"torch.tensor(-1.0, dtype={target_name}.dtype))"
    )


# ---------------------------------------------------------------------------
# Base class for same-shape float losses (2 or 3 inputs)
# ---------------------------------------------------------------------------


class _SameShapeFloatLossBase(Operator):
    # [CLASS DOC] Base for losses taking N (default 2) float tensors of identical
    # size/stride/dtype and returning a same-shape float tensor. Class attribute _num_inputs
    # sets N. Subclasses override _transform_inputs (rewrite input expressions) and
    # _extra_kwargs (extra keyword args); reduction is always none.
    """N same-shape float tensors in, one same-shape float tensor out.

    Subclass hooks:
    - ``_num_inputs`` class attribute (default 2) -- number of input tensors
      requested in ``fuzz_inputs_specs``.
    - ``_transform_inputs(input_names) -> list[str]`` -- per-position expression
      transforms; default returns ``input_names`` unchanged.
    - ``_extra_kwargs() -> dict[str, object]`` -- extra keyword args appended
      to the call (joined into the source via ``f"{k}={v!r}"``); default empty.
    """

    _num_inputs: int = 2

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True for tensor specs whose dtype is one of the float dtypes.
        return isinstance(output_spec, TensorSpec) and output_spec.dtype in FLOAT_DTYPES

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns _num_inputs copies of one spec that matches the output's size,
        # stride and dtype. Raises AssertionError if output_spec is not a TensorSpec.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        spec = TensorSpec(
            size=output_spec.size,
            stride=output_spec.stride,
            dtype=output_spec.dtype,
        )
        return [spec] * self._num_inputs

    def _transform_inputs(self, input_names: list[str]) -> list[str]:
        # [FUNCTION DOC] HELPER: Hook returning the per-input source expressions; default
        # returns a copy of the names.
        return list(input_names)

    def _extra_kwargs(self) -> dict[str, object]:
        # [FUNCTION DOC] HELPER: Hook returning extra keyword arguments for the call; default is
        # an empty mapping.
        return {}

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        # [FUNCTION DOC] Emits a line calling the loss function with the (possibly transformed)
        # input expressions, keyword reduction=none, and any extra keyword arguments rendered
        # as text.
        exprs = self._transform_inputs(input_names)
        kwargs: dict[str, object] = {"reduction": "none", **self._extra_kwargs()}
        kwargs_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
        return f"{output_name} = {self.torch_op_name}({', '.join(exprs)}, {kwargs_str})"


# ---------------------------------------------------------------------------
# Group A: elementwise (2-input) loss subclasses
# ---------------------------------------------------------------------------


class MseLossOperator(_SameShapeFloatLossBase):
    # [CLASS DOC] Elementwise mean squared error loss (mse_loss) with no reduction.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Register under the name of torch.nn.functional.mse_loss.
        super().__init__("torch.nn.functional.mse_loss")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Returns the registered operator name, which is the torch function
        # called in generated code.
        return self.name


class HuberLossOperator(_SameShapeFloatLossBase):
    # [CLASS DOC] Huber loss with a random delta stashed on the instance by
    # fuzz_inputs_specs and passed at codegen time (state is shared across nodes if fuzzing
    # several before codegen).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Register the operator and set the default delta to 0.1.
        super().__init__("torch.nn.functional.huber_loss")
        self._delta: float = 0.1

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Returns the registered operator name, which is the torch function
        # called in generated code.
        return self.name

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Picks a random delta from a small list, stores it on the instance, then
        # defers to the base implementation to build the input specs.
        self._delta = random.choice([0.1, 0.5, 1.0, 2.0])
        return super().fuzz_inputs_specs(output_spec)

    def _extra_kwargs(self) -> dict[str, object]:
        # [FUNCTION DOC] Returns the mapping with the stored delta as the extra
        # keyword argument.
        # KNOWN ISSUE: the comment about is-not-None checks is stale; the value is
        # returned directly.
        # is not None -- NOT truthiness -- falsy stash values (0.0) are valid draws.
        out: dict[str, object] = {"delta": self._delta}
        return out


class SmoothL1LossOperator(_SameShapeFloatLossBase):
    # [CLASS DOC] Smooth L1 loss with a random beta (including 0) stashed on the instance by
    # fuzz_inputs_specs and passed at codegen time.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Register the operator and set the default beta to 0.
        super().__init__("torch.nn.functional.smooth_l1_loss")
        self._beta: float = 0.0

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Returns the registered operator name, which is the torch function
        # called in generated code.
        return self.name

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Picks a random beta from a small list, stores it on the instance, then
        # defers to the base implementation to build the input specs.
        # beta=0.0 reduces smooth_l1 to L1 loss; PyTorch requires beta >= 0.
        self._beta = random.choice([0.0, 0.5, 1.0, 2.0])
        return super().fuzz_inputs_specs(output_spec)

    def _extra_kwargs(self) -> dict[str, object]:
        # [FUNCTION DOC] Returns the mapping with the stored beta as the extra keyword argument.
        # KNOWN ISSUE: the comment about is-not-None checks is stale; the value is
        # returned directly.
        # is not None -- NOT truthiness -- falsy stash values (0.0) are valid draws.
        out: dict[str, object] = {"beta": self._beta}
        return out


class BinaryCrossEntropyWithLogitsOperator(_SameShapeFloatLossBase):
    # [CLASS DOC] Binary cross entropy with logits loss; inputs are used as-is, no transforms.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Register under the name of the with-logits loss function.
        super().__init__("torch.nn.functional.binary_cross_entropy_with_logits")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Returns the registered operator name, which is the torch function
        # called in generated code.
        return self.name


class BinaryCrossEntropyOperator(_SameShapeFloatLossBase):
    # [CLASS DOC] Binary cross entropy loss; both input and target are passed through sigmoid in
    # generated code so values lie in the valid zero-to-one range.
    """Wraps both input and target through sigmoid to ensure [0, 1] range."""

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Register under the name of
        # torch.nn.functional.binary_cross_entropy.
        super().__init__("torch.nn.functional.binary_cross_entropy")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Returns the registered operator name, which is the torch function
        # called in generated code.
        return self.name

    def _transform_inputs(self, input_names: list[str]) -> list[str]:
        # [FUNCTION DOC] HELPER: Returns source expressions applying sigmoid to every
        # input name.
        return [f"torch.sigmoid({n})" for n in input_names]


class SoftMarginLossOperator(_SameShapeFloatLossBase):
    # [CLASS DOC] Soft margin loss; the target is mapped to plus or minus one in generated code.
    """Wraps target to +1/-1 via :func:`_pm1_target_expr`."""

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Register under the name of
        # torch.nn.functional.soft_margin_loss.
        super().__init__("torch.nn.functional.soft_margin_loss")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Returns the registered operator name, which is the torch function
        # called in generated code.
        return self.name

    def _transform_inputs(self, input_names: list[str]) -> list[str]:
        # [FUNCTION DOC] HELPER: Returns the input name unchanged and the target name wrapped by
        # the plus/minus one helper. Expects exactly two names, otherwise unpacking fails.
        input_name, target_name = input_names
        return [input_name, _pm1_target_expr(target_name)]


class HingeEmbeddingLossOperator(_SameShapeFloatLossBase):
    # [CLASS DOC] Hinge embedding loss; the target is mapped to plus or minus one and a random
    # margin is stashed on the instance by fuzz_inputs_specs.
    """Wraps target to +1/-1 via :func:`_pm1_target_expr`; randomized margin."""

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Register the operator and set the default margin to 0.
        super().__init__("torch.nn.functional.hinge_embedding_loss")
        self._margin: float = 0.0

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Returns the registered operator name, which is the torch function
        # called in generated code.
        return self.name

    def _transform_inputs(self, input_names: list[str]) -> list[str]:
        # [FUNCTION DOC] HELPER: Returns the input name unchanged and the target name wrapped by
        # the plus/minus one helper.
        input_name, target_name = input_names
        return [input_name, _pm1_target_expr(target_name)]

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Picks a random margin from a small list, stores it, then defers to the
        # base implementation.
        self._margin = random.choice([0.0, 0.5, 1.0, 2.0])
        return super().fuzz_inputs_specs(output_spec)

    def _extra_kwargs(self) -> dict[str, object]:
        # [FUNCTION DOC] Returns the mapping with the stored margin as the extra
        # keyword argument.
        # KNOWN ISSUE: the comment about is-not-None checks is stale; the value is
        # returned directly.
        # is not None -- NOT truthiness -- falsy stash values (0.0) are valid draws.
        out: dict[str, object] = {"margin": self._margin}
        return out


# ---------------------------------------------------------------------------
# Group B: three-input ranking loss (folded into the same base via _num_inputs)
# ---------------------------------------------------------------------------


class MarginRankingLossOperator(_SameShapeFloatLossBase):
    # [CLASS DOC] Margin ranking loss over three same-shape float inputs (two inputs and a
    # target). The target is mapped to plus or minus one in generated code; a random margin is
    # stashed by fuzz_inputs_specs. _num_inputs is 3.
    """Three same-shape float inputs (input1, input2, target).

    Target is wrapped to +1/-1 via :func:`_pm1_target_expr`; randomized margin.
    """

    _num_inputs: int = 3

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Register the operator and set the default margin to 0.
        super().__init__("torch.nn.functional.margin_ranking_loss")
        self._margin: float = 0.0

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Returns the registered operator name, which is the torch function
        # called in generated code.
        return self.name

    def _transform_inputs(self, input_names: list[str]) -> list[str]:
        # [FUNCTION DOC] HELPER: Returns the two inputs unchanged and the third (target) wrapped
        # by the plus/minus one helper.
        i1, i2, target_name = input_names
        return [i1, i2, _pm1_target_expr(target_name)]

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Picks a random margin from a small list, stores it, then defers to the
        # base implementation.
        self._margin = random.choice([0.0, 0.1, 0.5, 1.0])
        return super().fuzz_inputs_specs(output_spec)

    def _extra_kwargs(self) -> dict[str, object]:
        # [FUNCTION DOC] Returns the mapping with the stored margin as the extra
        # keyword argument.
        # KNOWN ISSUE: the comment about is-not-None checks is stale; the value is
        # returned directly.
        # is not None -- NOT truthiness -- falsy stash values (0.0) are valid draws.
        out: dict[str, object] = {"margin": self._margin}
        return out


# ---------------------------------------------------------------------------
# Group C: classification losses (asymmetric input/output shapes)
# ---------------------------------------------------------------------------


class NllLossOperator(Operator):
    # [CLASS DOC] Negative log likelihood loss on a batch: input of N rows by C classes
    # (log_softmax applied in generated code), int64 target of length N reduced into valid class
    # ids, output of length N. State: the random class count is stashed by fuzz_inputs_specs and
    # cleared by codegen.
    """F.nll_loss(log_softmax(input), target % C, reduction='none').

    Input: (N, C) float -- wrapped through log_softmax to produce valid
    log-probabilities.
    Target: (N,) int64 -- wrapped with .abs() % C to produce valid class
    indices in [0, C).
    Output: (N,) float.

    C (number of classes) is randomly chosen in [2, 10] at
    fuzz_inputs_specs time and stashed for codegen.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Register the operator and initialize the stashed class count
        # to none.
        super().__init__("torch.nn.functional.nll_loss")
        self._num_classes: int | None = None

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Returns the registered operator name, which is the torch function
        # called in generated code.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True for float 1-D contiguous (stride one) tensor specs
        # with positive length. Non-contiguous strides are rejected since the result is
        # always contiguous.
        # Output of F.nll_loss(reduction='none') is always a contiguous (N,) tensor;
        # we can't honor a non-contiguous output stride without an extra
        # as_strided/copy step, so the stride == (1,) filter rejects those specs
        # instead of silently producing a contiguous result.
        return (
            isinstance(output_spec, TensorSpec)
            and output_spec.dtype in FLOAT_DTYPES
            and len(output_spec.size) == 1
            and output_spec.size[0] > 0
            and output_spec.stride == (1,)
        )

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Picks a random class count from 2 to 10, stores it, and returns the
        # input spec (N by C float, contiguous) and an int64 target spec of length N. Raises
        # AssertionError if output_spec is not a TensorSpec.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        n = output_spec.size[0]
        self._num_classes = random.randint(2, 10)
        c = self._num_classes
        input_spec = TensorSpec(size=(n, c), stride=(c, 1), dtype=output_spec.dtype)
        target_spec = TensorSpec(size=(n,), stride=(1,), dtype=torch.int64)
        return [input_spec, target_spec]

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        # [FUNCTION DOC] Emits a line computing the loss from log_softmax over dim 1 of the
        # input and the absolute value of the target modulo the stashed class count, then resets
        # the stash to none.
        # KNOWN ISSUE: if codegen runs without a preceding fuzz_inputs_specs (or twice), the
        # class count is none and the generated modulo is invalid.
        c = self._num_classes
        out = (
            f"{output_name} = torch.nn.functional.nll_loss("
            f"torch.log_softmax({input_names[0]}, dim=1), "
            f"{input_names[1]}.abs() % {c}, reduction='none')"
        )
        self._num_classes = None
        return out


class CrossEntropyOperator(Operator):
    # [CLASS DOC] Cross entropy loss on a batch: raw-logit input of N by C classes, int64 target
    # of length N reduced into valid class ids, output of length N. State: the random class
    # count is stashed by fuzz_inputs_specs and cleared by codegen.
    """F.cross_entropy(input, target % C, reduction='none').

    Input: (N, C) float -- raw logits (cross_entropy applies log_softmax
    internally).
    Target: (N,) int64 -- wrapped with .abs() % C to produce valid class
    indices in [0, C).
    Output: (N,) float.

    C (number of classes) is randomly chosen in [2, 10] at
    fuzz_inputs_specs time and stashed for codegen.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Register the operator and initialize the stashed class count
        # to none.
        super().__init__("torch.nn.functional.cross_entropy")
        self._num_classes: int | None = None

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Returns the registered operator name, which is the torch function
        # called in generated code.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True for float 1-D contiguous (stride one) tensor specs with
        # positive length.
        # Output of F.cross_entropy(reduction='none') is always a contiguous (N,)
        # tensor; we can't honor a non-contiguous output stride without an extra
        # as_strided/copy step, so the stride == (1,) filter rejects those specs
        # instead of silently producing a contiguous result.
        return (
            isinstance(output_spec, TensorSpec)
            and output_spec.dtype in FLOAT_DTYPES
            and len(output_spec.size) == 1
            and output_spec.size[0] > 0
            and output_spec.stride == (1,)
        )

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Picks a random class count from 2 to 10, stores it, and returns the
        # input spec (N by C float, contiguous) and an int64 target spec of length N. Raises
        # AssertionError if output_spec is not a TensorSpec.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        n = output_spec.size[0]
        self._num_classes = random.randint(2, 10)
        c = self._num_classes
        input_spec = TensorSpec(size=(n, c), stride=(c, 1), dtype=output_spec.dtype)
        target_spec = TensorSpec(size=(n,), stride=(1,), dtype=torch.int64)
        return [input_spec, target_spec]

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        # [FUNCTION DOC] Emits a line computing cross entropy of the input with the absolute
        # value of the target modulo the stashed class count, then resets the stash to none.
        # KNOWN ISSUE: if codegen runs without a preceding fuzz_inputs_specs (or twice), the
        # class count is none and the generated modulo is invalid.
        c = self._num_classes
        out = (
            f"{output_name} = torch.nn.functional.cross_entropy("
            f"{input_names[0]}, "
            f"{input_names[1]}.abs() % {c}, reduction='none')"
        )
        self._num_classes = None
        return out
