# [MODULE DOC] Test helper with a decorated function that calls an undecorated helper defined in
# the same module, used to check name resolution across modules when scripting.
r"""
Helper function used in test_decorator.py. We define it in a
separate file on purpose to test that the names in different modules
are resolved correctly.
"""

from jit.mydecorator import my_decorator


@my_decorator
def my_function_b(x: float) -> float:
    # [FUNCTION DOC] HELPER: Returns the module-local helper applied to x, plus two. Decorated
    # with the pass-through decorator.
    return my_function_c(x) + 2


def my_function_c(x: float) -> float:
    # [FUNCTION DOC] HELPER: Returns the float argument plus three.
    return x + 3
