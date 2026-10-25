# [MODULE DOC]
# Defines the Hooks dataclass, a container for optional user callbacks invoked
# during Dynamo guard operations (export, failure, filtering).
# No import-time side effects beyond imports.
"""Hook system for Dynamo's guard functionality.

This module provides a way to register callback functions that are triggered during
guard-related operations.

The Hooks class manages two types of hook functions:
- guard_export_fn: Called when guards need to be exported, taking a GuardsSet as input
- guard_fail_fn: Called when a guard check fails, taking a GuardFail object as input
These hooks enable customization of guard export and failure handling behaviors.
"""

import dataclasses
from collections.abc import Callable, Sequence

from torch._guards import GuardsSet

from .types import GuardFail, GuardFilterEntry


@dataclasses.dataclass
class Hooks:
    # [CLASS DOC]
    # Plain dataclass holding three optional callbacks, all defaulting to none:
    # guard_export_fn (receives the set of guards when exported), guard_fail_fn
    # (receives a GuardFail record when a guard check fails), and guard_filter_fn
    # (receives a sequence of guard filter entries and returns a same-length
    # sequence of booleans deciding which guards to keep). No methods or invariants.
    guard_export_fn: Callable[[GuardsSet], None] | None = None
    guard_fail_fn: Callable[[GuardFail], None] | None = None
    guard_filter_fn: Callable[[Sequence[GuardFilterEntry]], Sequence[bool]] | None = (
        None
    )
