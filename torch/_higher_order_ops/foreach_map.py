# [MODULE DOC] Defines the foreach_map higher-order operator, a BaseHOP subclass that applies a
# pointwise callable across lists of operands as one fused unit. The public foreach_map
# function forwards to a module-level singleton instance, passing the dynamo polyfill
# foreach_map_fn as the body. No import-time side effects beyond creating the singleton.
# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
from collections.abc import Callable
from typing import Any

from torch._higher_order_ops.base_hop import BaseHOP, FunctionWithNoFreeVars


class ForeachMap(BaseHOP):
    # [CLASS DOC] BaseHOP subclass registered under the name "foreach_map". Stateless; a single
    # instance (_foreach_map) is created at module level. It wraps the body callable in
    # FunctionWithNoFreeVars (which forbids captured free variables) before dispatch.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Initializes the base HOP with the operator name "foreach_map".
        super().__init__("foreach_map")

    def __call__(self, fn, *operands, **kwargs):  # type: ignore[override]
        # [FUNCTION DOC] Wraps fn in FunctionWithNoFreeVars and invokes the base HOP call with the
        # operands and keyword arguments. Returns whatever the base call returns; errors
        # (e.g. on free variables) come from the wrapper or base class.
        fn = FunctionWithNoFreeVars(fn)
        return super().__call__(fn, *operands, **kwargs)


_foreach_map = ForeachMap()


def foreach_map(op: Callable, *operands: Any, **kwargs: dict[str, Any]):
    # [FUNCTION DOC] Public entry point: applies op across the given operand lists by calling the
    # singleton HOP with the dynamo polyfill foreach_map_fn as the body and op as its first
    # operand. Returns the HOP result. Imports the polyfill lazily to avoid a cycle.
    from torch._dynamo.polyfills import foreach_map_fn

    return _foreach_map(foreach_map_fn, op, *operands, **kwargs)
