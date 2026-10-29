# [MODULE DOC] Backward-compatibility shim: star-imports everything from
# torch.distributed.tensor._api so the old _tensor.api path keeps working. No logic of
# its own; no warning is emitted on import (a TODO notes it should).
"""
NOTE: torch.distributed._tensor has been moved to torch.distributed.tensor.
The imports here are purely for backward compatibility. We will remove these
imports in a few releases

TODO: throw warnings when this module imported
"""

from torch.distributed.tensor._api import *  # noqa: F403
