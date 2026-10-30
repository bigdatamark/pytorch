# [MODULE DOC] Test helper providing a pass-through decorator, kept in its own module so tests can
# check that names from different modules resolve correctly when scripting decorated code.
r"""
Decorator used in test_decorator.py. We define it in a
separate file on purpose to test that the names in different modules
are resolved correctly.
"""

import functools


def my_decorator(func):
    # [FUNCTION DOC] HELPER: Decorator. Returns a function that forwards all positional and
    # keyword arguments to the decorated function and preserves its metadata. The wrapper carries
    # a hook attribute returning the original function so the script compiler can strip the
    # decorator.
    """Dummy decorator that removes itself when torchscripting"""

    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Forwards all arguments to the wrapped function; returns its
        # result.
        return func(*args, **kwargs)

    # torch.jit.script() uses __prepare_scriptable__ to remove the decorator
    wrapped_func.__prepare_scriptable__ = lambda: func

    return wrapped_func
