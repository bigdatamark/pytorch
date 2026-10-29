# flake8: noqa: F401
# [MODULE DOC] Deprecated compatibility shim; re-exports ReferenceQuantizedModule and weight
# quantization helpers (_quantize_weight, _quantize_and_dequantize_weight, _save_weight_qparams,
# _get_weight_qparam_keys) from torch.ao.nn.quantized.reference.modules.utils.
r"""Quantized Reference Modules.

This module is in the process of migration to
`torch/ao/nn/quantized/reference`, and is kept here for
compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantized/reference`,
while adding an import statement here.
"""

from torch.ao.nn.quantized.reference.modules.utils import (
    _get_weight_qparam_keys,
    _quantize_and_dequantize_weight,
    _quantize_weight,
    _save_weight_qparams,
    ReferenceQuantizedModule,
)
