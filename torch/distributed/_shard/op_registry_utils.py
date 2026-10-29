# mypy: allow-untyped-defs
# [MODULE DOC] Utilities to register custom op handlers in an op table for ShardedTensor and PartialTensor: signature check and a validating wrapper.
import functools
from inspect import signature

from .common_op_utils import _basic_validation


"""
Common utilities to register ops on ShardedTensor
and PartialTensor.
"""


def _register_op(op, func, op_table):
    # [FUNCTION DOC] HELPER: Store func under op in op_table after checking that func takes exactly four parameters
    # (types, args, kwargs, process_group); raises TypeError otherwise. Mutates op_table.
    """
    Performs basic validation and registers the provided op in the given
    op_table.
    """
    if len(signature(func).parameters) != 4:
        raise TypeError(
            f"Custom sharded op function expects signature: "
            f"(types, args, kwargs, process_group), but received "
            f"signature: {signature(func)}"
        )

    op_table[op] = func


def _decorator_func(wrapped_func, op, op_table):
    # [FUNCTION DOC] HELPER: Wrap wrapped_func in a handler that first runs _basic_validation on the op and its arguments, register the wrapper in op_table and return it.
    """
    Decorator function to register the given ``op`` in the provided
    ``op_table``
    """

    @functools.wraps(wrapped_func)
    def wrapper(types, args, kwargs, process_group):
        # [FUNCTION DOC] HELPER: Validate the call, then delegate to the wrapped function and return its result.
        _basic_validation(op, args, kwargs)
        return wrapped_func(types, args, kwargs, process_group)

    _register_op(op, wrapper, op_table)
    return wrapper
