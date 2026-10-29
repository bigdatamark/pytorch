# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests that Dynamo models the tp_call protocol: callable() on builtins, types,
# bound methods and instances with or without a __call__ slot, calling objects and invoking
# __call__ directly, the TypeError messages for non-callables, the graph break for a __call__
# that is a C builtin, and polyfilled/partial callables. Defines small helper classes and two
# module-level sentinels (a plain object and a module object) at import time.
"""Tests for tp_call: callable() (PyCallable_Check) and the __call__ slot."""

import functools
import operator
import types

import torch
from torch._dynamo.exc import Unsupported
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import make_dynamo_test


_NOT_CALLABLE_OBJECT = object()
_A_MODULE = types.ModuleType("mod")


class _NoCall:
    # [CLASS DOC] HELPER: Plain class without __call__; instances are not callable, but its bound
    # method is.
    def meth(self):
        # [FUNCTION DOC] HELPER: Empty method used to obtain a callable bound method.
        pass


class _WithCall:
    # [CLASS DOC] HELPER: Class defining __call__ that returns its argument plus one.
    def __call__(self, x):
        # [FUNCTION DOC] HELPER: Return x plus 1.
        return x + 1


class _WithCallChild(_WithCall):
    # [CLASS DOC] HELPER: Subclass of the callable class that inherits its __call__ unchanged.
    pass


class _WithUntraceableCall:
    # [CLASS DOC] HELPER: Class whose __call__ class attribute is the C builtin operator.add, so it
    # is callable but Dynamo has no Python body to trace.
    # __call__ is a C builtin with no traceable Python body; the type has a
    # tp_call slot (callable) but Dynamo cannot trace the call -> graph break.
    __call__ = operator.add


class TpCallTests(TestCase):
    # [CLASS DOC] Dynamo test case covering callable() and calling behavior of objects with or
    # without a __call__ slot.
    @make_dynamo_test
    def test_callable_builtin(self):
        # [FUNCTION DOC] Under make_dynamo_test (run eagerly, then compiled with no graph breaks
        # allowed): callable of a builtin function is true.
        assert callable(len)  # noqa: S101

    @make_dynamo_test
    def test_callable_str_false(self):
        # [FUNCTION DOC] Under make_dynamo_test: callable of a str instance is false.
        assert not callable("a")  # noqa: S101

    @make_dynamo_test
    def test_callable_lambda(self):
        # [FUNCTION DOC] Under make_dynamo_test: callable of a lambda is true.
        assert callable(lambda x, y: x + y)  # noqa: S101

    @make_dynamo_test
    def test_callable_type(self):
        # [FUNCTION DOC] Under make_dynamo_test: type objects (int and a user class) are
        # callable.
        # type objects are callable via their metaclass's tp_call
        assert callable(int)  # noqa: S101
        assert callable(_NoCall)  # noqa: S101

    @make_dynamo_test
    def test_callable_bound_method(self):
        # [FUNCTION DOC] Under make_dynamo_test: a bound method of an instance is callable.
        c = _NoCall()
        assert callable(c.meth)  # noqa: S101

    @make_dynamo_test
    def test_callable_instance_without_call(self):
        # [FUNCTION DOC] Under make_dynamo_test: an instance of a class without __call__ is not
        # callable.
        # __call__ is looked up on the class slot, not the instance
        assert not callable(_NoCall())  # noqa: S101

    @make_dynamo_test
    def test_callable_instance_with_call(self):
        # [FUNCTION DOC] Under make_dynamo_test: an instance of a class with __call__ is
        # callable.
        assert callable(_WithCall())  # noqa: S101

    @make_dynamo_test
    def test_callable_subclass_instance(self):
        # [FUNCTION DOC] Under make_dynamo_test: an instance of a subclass inheriting __call__
        # is callable.
        assert callable(_WithCallChild())  # noqa: S101

    def test_callable_user_object(self):
        # [FUNCTION DOC] Compile with eager backend and fullgraph a function returning callable
        # of two user objects passed as arguments and assert the results match eager.
        # callable() on a sourced user object inside the compiled region
        def fn(x, obj, nc):
            # [FUNCTION DOC] HELPER: Return a tuple of callable(obj) and callable(nc); x is
            # unused.
            return callable(obj), callable(nc)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(
            fn(torch.randn(2), _WithCall(), _NoCall()),
            compiled(torch.randn(2), _WithCall(), _NoCall()),
        )

    def test_call_via_call_function(self):
        # [FUNCTION DOC] Compile with fullgraph a function that calls a closed-over callable
        # instance and assert the output equals eager.
        obj = _WithCall()

        def fn(x):
            # [FUNCTION DOC] HELPER: Call the closed-over instance with x.
            return obj(x)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(2)
        self.assertEqual(fn(x), compiled(x))

    def test_dunder_call(self):
        # [FUNCTION DOC] Compile with fullgraph a function that invokes __call__ explicitly on a
        # callable instance and assert it matches eager.
        obj = _WithCall()

        def fn(x):
            # [FUNCTION DOC] HELPER: Call obj.__call__ with x.
            return obj.__call__(x)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(2)
        self.assertEqual(fn(x), compiled(x))

    def test_dunder_call_matches_call(self):
        # [FUNCTION DOC] Compile with fullgraph a function summing a direct call and an explicit
        # __call__ invocation of the same object; assert it matches eager.
        obj = _WithCall()

        def fn(x):
            # [FUNCTION DOC] HELPER: Return obj(x) plus obj.__call__(x).
            return obj(x) + obj.__call__(x)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(2)
        self.assertEqual(fn(x), compiled(x))

    @make_dynamo_test
    def test_object_not_callable(self):
        # [FUNCTION DOC] Under make_dynamo_test: calling a plain object instance raises
        # TypeError with message "'object' object is not callable"; fails with AssertionError if
        # nothing is raised.
        # PyObject_Call on a type with no tp_call slot raises TypeError.
        try:
            _NOT_CALLABLE_OBJECT()
        except TypeError as e:
            assert str(e) == "'object' object is not callable"  # noqa: S101
        else:
            raise AssertionError("expected TypeError")

    @make_dynamo_test
    def test_int_not_callable(self):
        # [FUNCTION DOC] Under make_dynamo_test: calling an int raises TypeError with message
        # "'int' object is not callable"; fails with AssertionError if nothing is raised.
        n = 5
        try:
            n()
        except TypeError as e:
            assert str(e) == "'int' object is not callable"  # noqa: S101
        else:
            raise AssertionError("expected TypeError")

    @make_dynamo_test
    def test_module_not_callable(self):
        # [FUNCTION DOC] Under make_dynamo_test: calling a module object raises TypeError with
        # message "'module' object is not callable"; fails with AssertionError if nothing is
        # raised.
        try:
            _A_MODULE()
        except TypeError as e:
            assert str(e) == "'module' object is not callable"  # noqa: S101
        else:
            raise AssertionError("expected TypeError")

    def test_untraceable_dunder_call_graph_breaks(self):
        # [FUNCTION DOC] Compile with fullgraph a function calling an object whose __call__ is a
        # C builtin and assert Unsupported is raised with a message about a callable object with
        # no traceable __call__.
        obj = _WithUntraceableCall()

        def fn(x):
            # [FUNCTION DOC] HELPER: Call the closed-over untraceable callable with x.
            return obj(x)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            Unsupported, "call to a callable object with no traceable __call__"
        ):
            compiled(torch.randn(2))

    def test_polyfilled_function_dunder_call(self):
        # [FUNCTION DOC] Compile with fullgraph a function that calls __call__ of the polyfilled
        # functools.reduce with operator.add over three copies of x; asserts the result matches
        # eager (the polyfill's __call__ must route back to call_function).
        # PolyfilledFunctionVariable.__call__ must route back to call_function
        # instead of resolving the method-wrapper as a polyfill handler.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return functools.reduce.__call__(operator.add, [x, x, x]).
            return functools.reduce.__call__(operator.add, [x, x, x])

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(2)
        self.assertEqual(fn(x), compiled(x))

    def test_functools_partial_call(self):
        # [FUNCTION DOC] Compile with fullgraph a function that calls a functools.partial
        # wrapping a two-argument function and assert the output matches eager.
        def g(a, b):
            # [FUNCTION DOC] HELPER: Return a plus b.
            return a + b

        p = functools.partial(g, 10)

        def fn(x):
            # [FUNCTION DOC] HELPER: Call the closed-over partial with x.
            return p(x)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(2)
        self.assertEqual(fn(x), compiled(x))


if __name__ == "__main__":
    run_tests()
