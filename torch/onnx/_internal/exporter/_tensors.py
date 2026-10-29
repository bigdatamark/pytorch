# [MODULE DOC]
# Defines SymbolicTensor, an onnx_ir Value subclass used while tracing the exporter graph.
# It overloads arithmetic and comparison operators so that torchlib op implementations can
# be written with natural operator syntax; each operator emits an ONNX node via an opset.
# No import-time side effects other than importing the lazily loaded onnx_ir module.
"""Subclass of ir.Value that supports Python operators."""

# mypy: allow-untyped-defs
from __future__ import annotations

from typing import TYPE_CHECKING

from torch.onnx._internal._lazy_import import onnx_ir as ir


if TYPE_CHECKING:
    import onnxscript


class SymbolicTensor(ir.Value):
    # [CLASS DOC]
    # An ONNX IR value that also carries an opset (the _opset attribute) so that Python
    # operators on it build new ONNX nodes (Add, Mul, Less, ...) and return their outputs.
    # Operands may be other values or Python constants; reflected variants swap the order.
    # Not all operators are implemented (indexing is a TODO; floor division, etc. are absent).
    """A subclass of ir.Value that supports Python operators."""

    def __init__(
        self,
        opset: onnxscript.values.Opset,
        name: str | None = None,
        shape: ir.Shape | None = None,
        type: ir.TypeProtocol | None = None,
        doc_string: str | None = None,
        const_value: ir.TensorProtocol | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Creates the value by forwarding name, shape, type, doc_string and const_value to
        # the base class, and stores the opset used to construct nodes for operators.
        super().__init__(
            name=name,
            shape=shape,
            type=type,
            doc_string=doc_string,
            const_value=const_value,
        )
        self._opset = opset

    @property
    def rank(self) -> int | None:
        # [FUNCTION DOC]
        # Property returning the number of dimensions, or None when the shape is unknown.
        if self.shape is None:
            return None
        return len(self.shape)

    # TODO: Implement indexing

    def __mod__(self, other):
        # [FUNCTION DOC]
        # Emits a Mod node for the remainder operator. For floating point dtypes (float,
        # double, float16, bfloat16) the fmod attribute is set so the sign follows the
        # dividend; otherwise the default integer mod is used. Returns the node output.
        if self.dtype in {
            ir.DataType.FLOAT,
            ir.DataType.DOUBLE,
            ir.DataType.FLOAT16,
            ir.DataType.BFLOAT16,
        }:
            return self._opset.Mod(self, other, fmod=1)
        return self._opset.Mod(self, other)

    def __ne__(self, other):
        # [FUNCTION DOC]
        # Emits Equal followed by Not and returns the boolean result of the inequality test.
        return self._opset.Not(self._opset.Equal(self, other))

    def __neg__(self):
        # [FUNCTION DOC]
        # Emits a Neg node and returns its output.
        return self._opset.Neg(self)

    def __add__(self, other):
        # [FUNCTION DOC]
        # Emits an Add node with self as the left operand and returns its output.
        return self._opset.Add(self, other)

    def __radd__(self, other):
        # [FUNCTION DOC]
        # Emits an Add node with self as the right operand and returns its output.
        return self._opset.Add(other, self)

    def __rand__(self, other):
        # [FUNCTION DOC]
        # Emits a logical And node with self as the right operand and returns its output.
        # Only the reflected form is defined; the forward bitwise-and operator is not.
        return self._opset.And(other, self)

    def __mul__(self, other):
        # [FUNCTION DOC]
        # Emits a Mul node with self as the left operand and returns its output.
        return self._opset.Mul(self, other)

    def __rmul__(self, other):
        # [FUNCTION DOC]
        # Emits a Mul node with self as the right operand and returns its output.
        return self._opset.Mul(other, self)

    def __matmul__(self, other):
        # [FUNCTION DOC]
        # Emits a MatMul node (matrix product) and returns its output.
        return self._opset.MatMul(self, other)

    def __pow__(self, other):
        # [FUNCTION DOC]
        # Emits a Pow node raising self to the exponent other and returns its output.
        return self._opset.Pow(self, other)

    def __sub__(self, other):
        # [FUNCTION DOC]
        # Emits a Sub node computing self minus other and returns its output.
        return self._opset.Sub(self, other)

    def __rsub__(self, other):
        # [FUNCTION DOC]
        # Emits a Sub node computing other minus self and returns its output.
        return self._opset.Sub(other, self)

    def __truediv__(self, other):
        # [FUNCTION DOC]
        # Emits a Div node computing self divided by other and returns its output.
        # KNOWN ISSUE: no reflected division operator is defined, so a constant on the
        # left side of a division is not supported.
        return self._opset.Div(self, other)

    def __lt__(self, other):
        # [FUNCTION DOC]
        # Emits a Less node and returns the boolean result of the comparison.
        return self._opset.Less(self, other)

    def __le__(self, other):
        # [FUNCTION DOC]
        # Emits a LessOrEqual node and returns the boolean result of the comparison.
        return self._opset.LessOrEqual(self, other)

    def __ge__(self, other):
        # [FUNCTION DOC]
        # Emits a GreaterOrEqual node and returns the boolean result of the comparison.
        return self._opset.GreaterOrEqual(self, other)

    def __gt__(self, other):
        # [FUNCTION DOC]
        # Emits a Greater node and returns the boolean result of the comparison.
        return self._opset.Greater(self, other)
