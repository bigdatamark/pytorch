# mypy: allow-untyped-defs
# [MODULE DOC]
# TorchScript exporter symbolic functions for ONNX opset 15. Provides handlers for identity
# comparison with none (is / is not), bernoulli, and unchecked_cast, using the Optional and
# Bernoulli operators introduced in opset 15. Import-time side effects: registers the handlers
# in the global symbolic registry for opset 15.
"""This file exports ONNX ops for opset 15.

Note [ONNX operators that are added/updated in opset 15]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/master/docs/Changelog.md#version-15-of-the-default-onnx-operator-set
New operators:
    Bernoulli
    CastLike
    Optional
    OptionalGetElement
    OptionalHasElement

Updated operators:
    BatchNormalization https://github.com/onnx/onnx/pull/3545
                        Backwards compatible
                        TODO: test coverage for mixed types inputs.
    Pow                https://github.com/onnx/onnx/pull/3412
                        Backwards compatible
                        TODO: bfloat16 support.
    Shape              https://github.com/onnx/onnx/pull/3580
                        Backwards compatible
                        TODO: optional start/end attribute.
"""

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md

import functools

import torch
from torch import _C
from torch.onnx._internal.torchscript_exporter import (
    jit_utils,
    registration,
    symbolic_helper,
    symbolic_opset9 as opset9,
)


_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=15)


@_onnx_symbolic("aten::__is_")
def aten__is_(g: jit_utils.GraphContext, self, other):
    # [FUNCTION DOC]
    # Symbolic for the identity test. If other is none: when self has an optional type, emits
    # OptionalHasElement followed by Not (true when the optional is empty); otherwise returns a
    # constant false bool tensor. If other is not none, falls back to the opset 9 equality
    # handler.
    if symbolic_helper._is_none(other):
        if isinstance(self.type(), _C.OptionalType):
            none = g.op("OptionalHasElement", self)
            return g.op("Not", none)
        else:
            return g.op("Constant", value_t=torch.BoolTensor([0]))
    return opset9.eq(g, self, other)


@_onnx_symbolic("aten::__isnot_")
@opset9.wrap_logical_op_with_negation  # type: ignore[has-type]
def aten__isnot_(g: jit_utils.GraphContext, self, other):
    # [FUNCTION DOC]
    # Symbolic for the negated identity test: the is handler's result wrapped in a logical
    # negation by the opset 9 decorator.
    return aten__is_(g, self, other)


@_onnx_symbolic("aten::bernoulli")
def bernoulli(g: jit_utils.GraphContext, input, p=None, generator=None, out=None):
    # [FUNCTION DOC]
    # Symbolic for bernoulli. Reports an unimplemented-operator error if out or generator is
    # given (not none). With no p, emits the Bernoulli op on input; with p, delegates to the
    # opset 9 implementation. Returns the resulting graph value.
    if out is not None and not symbolic_helper._is_none(out):
        symbolic_helper._unimplemented(
            "Bernoulli", "out parameter is not supported for bernoulli", input
        )
    if generator is not None and not symbolic_helper._is_none(generator):
        symbolic_helper._unimplemented(
            "Bernoulli", "generator is not supported for bernoulli", input
        )
    if p is None or symbolic_helper._is_none(p):
        return g.op("Bernoulli", input)
    return opset9.bernoulli(g, input, p, generator, out)


@_onnx_symbolic("prim::unchecked_cast")
def prim_unchecked_cast(g: jit_utils.GraphContext, self):
    # [FUNCTION DOC]
    # Symbolic for unchecked_cast: if the input has an optional type, emits OptionalGetElement to
    # unwrap it; otherwise returns the input unchanged.
    # exists to refine the type of the Value
    # if x is Optional[Tensor], unchecked_cast will cast
    # x to Tensor, so the rest of the graph knows that x is a Tensor.
    if isinstance(self.type(), _C.OptionalType):
        return g.op("OptionalGetElement", self)

    return self
