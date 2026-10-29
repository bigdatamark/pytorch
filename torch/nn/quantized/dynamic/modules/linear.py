# flake8: noqa: F401
# [MODULE DOC] Deprecated compatibility shim; re-exports the dynamically quantized Linear from
# torch.ao.nn.quantized.dynamic.modules.linear. No behavior is defined here.
r"""Quantized Dynamic Modules.

This file is in the process of migration to `torch/ao/nn/quantized/dynamic`,
and is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantized/dynamic/modules`,
while adding an import statement here.
"""

from torch.ao.nn.quantized.dynamic.modules.linear import Linear
