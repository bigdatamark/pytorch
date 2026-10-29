# [PACKAGE DOC] Operator library of torchfuzz. Each operator class knows which output specs it
# can produce, what input specs it needs, and how to emit one line of generated Python code.
# This package re-exports the operator classes (leaf, pointwise, scalar, layout, matmul,
# nn-functional) plus the registry helpers for lookup, registration and weight configuration,
# all listed in the export list. Importing it only imports the submodules.
"""Torchfuzz operators module."""

from torchfuzz.operators.arg import ArgOperator
from torchfuzz.operators.argsort import ArgsortOperator
from torchfuzz.operators.base import Operator
from torchfuzz.operators.constant import ConstantOperator
from torchfuzz.operators.gather import GatherOperator
from torchfuzz.operators.index_select import IndexSelectOperator
from torchfuzz.operators.item import ItemOperator
from torchfuzz.operators.layout import (
    CatOperator,
    ExpandOperator,
    FlattenOperator,
    ReshapeOperator,
    SplitOperator,
    SqueezeOperator,
    UnsqueezeOperator,
    ViewOperator,
)
from torchfuzz.operators.matrix_multiply import (
    AddmmOperator,
    BmmOperator,
    MatmulOperator,
    MMOperator,
)
from torchfuzz.operators.nn_functional import (
    DropoutOperator,
    EmbeddingOperator,
    LayerNormOperator,
    LinearOperator,
    MultiHeadAttentionForwardOperator,
    ReLUOperator,
    ScaledDotProductAttentionOperator,
    SoftmaxOperator,
)
from torchfuzz.operators.registry import (
    get_operator,
    list_operators,
    register_operator,
    set_operator_weight,
    set_operator_weight_by_torch_op,
    set_operator_weights,
    set_operator_weights_by_torch_op,
)
from torchfuzz.operators.scalar_pointwise import (
    ScalarAddOperator,
    ScalarDivOperator,
    ScalarMulOperator,
    ScalarPointwiseOperatorBase,
    ScalarSubOperator,
)
from torchfuzz.operators.tensor_pointwise import (
    AddOperator,
    ClampOperator,
    DivOperator,
    MulOperator,
    PointwiseOperatorBase,
    SubOperator,
)


__all__ = [
    "Operator",
    "PointwiseOperatorBase",
    "AddOperator",
    "MulOperator",
    "SubOperator",
    "DivOperator",
    "ClampOperator",
    "ScalarPointwiseOperatorBase",
    "ScalarAddOperator",
    "ScalarMulOperator",
    "ScalarSubOperator",
    "ScalarDivOperator",
    "ItemOperator",
    "ConstantOperator",
    "ArgOperator",
    "ArgsortOperator",
    "GatherOperator",
    "IndexSelectOperator",
    "ViewOperator",
    "ReshapeOperator",
    "FlattenOperator",
    "SqueezeOperator",
    "UnsqueezeOperator",
    "SplitOperator",
    "ExpandOperator",
    "CatOperator",
    "MMOperator",
    "AddmmOperator",
    "BmmOperator",
    "MatmulOperator",
    "EmbeddingOperator",
    "LinearOperator",
    "MultiHeadAttentionForwardOperator",
    "ReLUOperator",
    "ScaledDotProductAttentionOperator",
    "SoftmaxOperator",
    "DropoutOperator",
    "LayerNormOperator",
    "get_operator",
    "register_operator",
    "list_operators",
    "set_operator_weight",
    "set_operator_weights",
    "set_operator_weight_by_torch_op",
    "set_operator_weights_by_torch_op",
]
