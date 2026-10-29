# [PACKAGE DOC] Compatibility package for FX graph-mode quantization, migrated to
# torch.ao.quantization.fx. Exports only the convert, fuse and prepare entry points,
# re-imported from the ao location. Other FX pieces (e.g. lowering) are intentionally
# omitted. Importing this package has no side effects beyond importing the ao modules.
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate files under `torch/ao/quantization/fx/`, while adding an import statement
here.
"""

from torch.ao.quantization.fx.convert import convert
from torch.ao.quantization.fx.fuse import fuse

# omitting files that's unlikely to be used right now, for example
# the newly added lower_to_fbgemm etc.
from torch.ao.quantization.fx.prepare import prepare
