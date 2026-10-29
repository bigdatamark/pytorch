# flake8: noqa: F401
# [MODULE DOC] Deprecated compatibility shim; re-exports the quantizable MultiheadAttention from
# torch.ao.nn.quantizable.modules.activation. No behavior is defined here.
r"""Quantizable Modules.

This file is in the process of migration to `torch/ao/nn/quantizable`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantizable/modules`,
while adding an import statement here.
"""

from torch.ao.nn.quantizable.modules.activation import MultiheadAttention
