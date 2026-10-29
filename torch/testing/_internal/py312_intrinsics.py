# [MODULE DOC] Test fixture that exercises Python 3.12 only bytecode intrinsics
# (function type parameters and annotations) inside a torch.compile region. Uses syntax
# that fails to parse on older Python, so it must only be imported on 3.12+. Defines
# class Foo; no import-time side effects beyond importing torch.
import torch


# This is a very odd way to define a test case that uses CALL_INTRINSIC_1 4 and
# CALL_INTRINSIC_2 7. The problem is those intrinsics only exists on Python
# 3.12+ and the Python code to produce them is not backwards compatible with
# previous versions.
class Foo:
    # [CLASS DOC] HELPER: Holder of a compile test case. _default_update builds a generic
    # function plus a wrapper to trigger the intrinsics; test_default_update compiles a
    # function that calls it. Stateless.
    @classmethod
    def _default_update(cls):
        # [FUNCTION DOC] HELPER: Defines a generic inner function f (with a type
        # parameter, a string annotation and docstring), sets two attributes on it
        # (attr and a deliberately bogus __wrapped__), defines an unrelated function
        # wrapper, and returns the pair (wrapper, f). No side effects outside locals.
        def f[T](a: "This is a new annotation"):  # noqa: F722
            # [FUNCTION DOC] HELPER: Empty body (docstring only); never called.
            """This is a test"""

        f.attr = "This is also a test"
        f.__wrapped__ = "This is a bald faced lie"

        def wrapper(b: "This is the prior annotation"):  # noqa: F722
            # [FUNCTION DOC] HELPER: Does nothing; exists only to be returned.
            pass

        return wrapper, f

    def test_default_update(self):
        # [FUNCTION DOC] Test: compiles (eager backend, fullgraph) a function that calls
        # _default_update and returns x plus one, then runs it on a random 2 element
        # tensor. Passes if tracing succeeds without graph breaks; returns nothing.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls _default_update, discards results, returns x + 1.
            wrapper, f = Foo._default_update()
            return x + 1

        x = torch.randn(2)
        fn(x)
