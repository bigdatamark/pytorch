# [MODULE DOC] Backward-compatibility shim: star-imports the contents of
# torch.distributed.tensor._dtensor_spec and torch.distributed.tensor.placement_types
# so the old _tensor.placement_types path keeps working. No logic of its own; no
# warning is emitted on import.
"""
NOTE: torch.distributed._tensor has been moved to torch.distributed.tensor.
The imports here are purely for backward compatibility. We will remove these
imports in a few releases

TODO: throw warnings when this module imported
"""

from torch.distributed.tensor._dtensor_spec import *  # noqa: F403
from torch.distributed.tensor.placement_types import *  # noqa: F403
