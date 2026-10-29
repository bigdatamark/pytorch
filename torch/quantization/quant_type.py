# flake8: noqa: F401
# [MODULE DOC] Backward-compatibility shim with no logic; re-exports from the migrated
# torch.ao.quantization location: QuantType enum and its string-mapping helper.
# Importing it only imports the ao module. Add new functionality to the ao module.
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/quantization/quant_type.py`, while adding an import statement
here.
"""

from torch.ao.quantization.quant_type import _get_quant_type_to_str, QuantType
