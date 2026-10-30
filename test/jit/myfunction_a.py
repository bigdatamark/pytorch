# [MODULE DOC] Test helper defining a decorated function that calls a function from another helper
# module (myfunction_b). Imports the decorator and function via the jit package path, so the test
# directory must be on the import path.
"""
Helper function used in test_decorator.py. We define it in a
separate file on purpose to test that the names in different modules
are resolved correctly.
"""

from jit.mydecorator import my_decorator
from jit.myfunction_b import my_function_b


@my_decorator
def my_function_a(x: float) -> float:
    # [FUNCTION DOC] HELPER: Returns the result of the other module's function applied to x, plus
    # one. Decorated with the pass-through decorator.
    return my_function_b(x) + 1
