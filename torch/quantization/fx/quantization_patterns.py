# [MODULE DOC] Backward-compatibility shim re-exporting the QuantizeHandler classes from
# torch.ao.quantization.fx.quantize_handler (the original quantization_patterns module).
# Import-time side effect: overwrites the __module__ attribute of every re-exported handler
# class to name this legacy module path (via its ao-style dotted name) for pickling and docs.
# KNOWN ISSUE: the assigned module name (ao quantization_patterns) matches no existing ao
# module (the classes live in ao quantize_handler), so module-based lookup may not resolve.
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate files under `torch/ao/quantization/fx/`, while adding an import statement
here.
"""

from torch.ao.quantization.fx.quantize_handler import (
    BatchNormQuantizeHandler,
    BinaryOpQuantizeHandler,
    CatQuantizeHandler,
    ConvReluQuantizeHandler,
    CopyNodeQuantizeHandler,
    CustomModuleQuantizeHandler,
    DefaultNodeQuantizeHandler,
    EmbeddingQuantizeHandler,
    FixedQParamsOpQuantizeHandler,
    GeneralTensorShapeOpQuantizeHandler,
    LinearReLUQuantizeHandler,
    QuantizeHandler,
    RNNDynamicQuantizeHandler,
    StandaloneModuleQuantizeHandler,
)


QuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
BinaryOpQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
CatQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
ConvReluQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
LinearReLUQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
BatchNormQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
EmbeddingQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
RNNDynamicQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
DefaultNodeQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
FixedQParamsOpQuantizeHandler.__module__ = (
    "torch.ao.quantization.fx.quantization_patterns"
)
CopyNodeQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
CustomModuleQuantizeHandler.__module__ = (
    "torch.ao.quantization.fx.quantization_patterns"
)
GeneralTensorShapeOpQuantizeHandler.__module__ = (
    "torch.ao.quantization.fx.quantization_patterns"
)
StandaloneModuleQuantizeHandler.__module__ = (
    "torch.ao.quantization.fx.quantization_patterns"
)
