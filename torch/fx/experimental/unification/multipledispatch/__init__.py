# [PACKAGE DOC] Vendored, minimal multiple-dispatch implementation (derived from
# the multipledispatch project) used by the unification package. Dispatches
# functions on the runtime types of all positional arguments, with support for
# variadic signatures and method dispatch. Re-exports the dispatch decorator,
# the Dispatcher class, MDNotImplementedError, and the halt_ordering and
# restart_ordering helpers that defer signature re-ordering.
from .core import dispatch
from .dispatcher import (
    Dispatcher,
    halt_ordering,
    MDNotImplementedError,
    restart_ordering,
)
