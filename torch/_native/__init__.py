# [PACKAGE DOC] Package for natively implemented (Python/DSL based) operator overrides.
# Importing it imports the DSL utility modules, the registry and the ops subpackage,
# then optionally installs a user graph-ordering function, and finally calls the
# registry to register all operator overrides (suppressing the one-time override warning).
# Exports: get_user_ordering_fn and the module-level user_order_fn.
import os
import warnings
from functools import cache
from typing import cast

# This handles collecting registration of all native ops
# Also need to import DSL utils to make sure DSL registration is ok
from . import cutedsl_utils, dsl_registry, helion_utils, ops, registry, triton_utils


@cache
def get_user_ordering_fn() -> registry.UserOrderingFn | None:
    # [FUNCTION DOC] Cached lookup of an optional user graph-ordering function.
    # Reads env var TORCH_PYTHON_NATIVE_USER_GRAPH_ORDER_FN holding a dotted path
    # "package.submodule.fn". Returns None if unset/empty; otherwise imports the module and
    # returns the named attribute. Raises ValueError (chained) if the import, lookup or
    # callable check fails. Side effect: imports the user module; result cached.
    """
    Get a user-supplied graph-ordering function if specified.

    Pass in a `package.submodule.fn` string to the env variable
    `TORCH_PYTHON_NATIVE_USER_GRAPH_ORDER_FN` that implements the
    calling API described in `torch/_native/README.md`. This function
    must be part of an importable path.

    Return either the imported function or `None`
    """
    env_var = os.getenv("TORCH_PYTHON_NATIVE_USER_GRAPH_ORDER_FN")

    if not env_var:
        return None

    try:
        import importlib

        # Split into "package.submodule.fn_name
        module_name, fn_name = env_var.rsplit(".", 1)

        module = importlib.import_module(module_name)
        fn = getattr(module, fn_name)

        if not callable(fn):
            raise TypeError(f"{env_var} does not describe a callable function")

        # Cast needed: getattr returns object, but we've verified fn is callable with correct signature
        return cast(registry.UserOrderingFn, fn)
    except Exception as e:
        raise ValueError(
            f"Could not resolve {env_var} into an importable & callable function"
        ) from e


user_order_fn = get_user_ordering_fn()
if user_order_fn:
    registry.reorder_graphs_from_user_function(user_order_fn)


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Warning only once for all operators,  other operators may also be overridden\\.",
        category=UserWarning,
    )
    registry._register_all_overrides()
