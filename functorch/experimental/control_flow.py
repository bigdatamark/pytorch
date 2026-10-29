# [MODULE DOC] Compatibility shim exposing control flow higher-order operators: re-exports
# cond, map and the pytree helpers _stack_pytree and _unstack_pytree from torch. No logic
# and no import-time side effects beyond the imports.
from torch import cond  # noqa: F401
from torch._higher_order_ops.map import (  # noqa: F401
    _stack_pytree,
    _unstack_pytree,
    map,
)
