# [MODULE DOC] Provides the public (not backward-compatible) annotate function, which attaches a
# type to the node behind a symbolic-tracing proxy. Exports only annotate; the compatibility
# decorator marks it as not backward compatible. No other import-time side effects.
from typing import Any

from torch.fx.proxy import Proxy

from ._compatibility import compatibility


__all__ = ["annotate"]


@compatibility(is_backward_compatible=False)
def annotate(val: Any, type: type) -> Any:
    # [FUNCTION DOC] If val is a Proxy, set the type of its underlying node to the given type and
    # return val; any non-proxy value is returned unchanged. Raises RuntimeError if the proxy's
    # node already has a (truthy) type, e.g. a function parameter or a repeated annotation.
    # Side effect: mutates the node's type attribute.
    """
    Annotates a Proxy object with a given type.

    This function annotates a val with a given type if a type of the val is a torch.fx.Proxy object
    Args:
        val (object): An object to be annotated if its type is torch.fx.Proxy.
        type (object): A type to be assigned to a given proxy object as val.
    Returns:
        The given val.
    Raises:
        RuntimeError: If a val already has a type in its node.
    """
    if isinstance(val, Proxy):
        if val.node.type:
            raise RuntimeError(
                f"Tried to annotate a value that already had a type on it!"
                f" Existing type is {val.node.type} "
                f"and new type is {type}. "
                f"This could happen if you tried to annotate a function parameter "
                f"value (in which case you should use the type slot "
                f"on the function signature) or you called "
                f"annotate on the same value twice"
            )
        else:
            val.node.type = type
        return val
    else:
        return val
