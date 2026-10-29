# flake8: noqa: F401
# [MODULE DOC] Backward-compatibility shim with no logic; re-exports from the migrated
# torch.ao.quantization location: FX pattern matching helpers (find_matches, MatchAllNode, etc.).
# Importing it only imports the ao module. Add new functionality to the ao module.
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate files under `torch/ao/quantization/fx/`, while adding an import statement
here.
"""

from torch.ao.quantization.fx.match_utils import (
    _find_matches,
    _is_match,
    _MatchResult,
    MatchAllNode,
)
