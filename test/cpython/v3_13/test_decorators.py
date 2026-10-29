# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_decorators.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

# [MODULE DOC] Dynamo-patched copy of the CPython 3.13 test_decorators tests, which check
# decorator syntax and semantics: staticmethod and classmethod wrappers, decorator expression
# forms, application and evaluation order, and class decorators. Defines several decorator
# helper functions (attribute setting, debug checks, call counting, memoization) at module level
# and two CPythonTestCase suites. Running as a script calls run_tests.

import unittest
from types import MethodType

def funcattrs(**kwds):
    # [FUNCTION DOC] HELPER: Decorator factory that returns a decorator copying the keyword
    # arguments into the function attribute mapping and returning the same function.
    def decorate(func):
        # [FUNCTION DOC] HELPER: Updates the function attribute mapping with the captured
        # keywords and returns it.
        func.__dict__.update(kwds)
        return func
    return decorate

class MiscDecorators (object):
    # [CLASS DOC] HELPER: Namespace class holding decorator factories used by the tests.
    @staticmethod
    def author(name):
        # [FUNCTION DOC] HELPER: Decorator factory that records the given name as the author
        # attribute of a function.
        def decorate(func):
            # [FUNCTION DOC] HELPER: Stores the captured name under the author key of the
            # function attributes; returns func.
            func.__dict__['author'] = name
            return func
        return decorate

# -----------------------------------------------

class DbcheckError (Exception):
    # [CLASS DOC] HELPER: Exception raised by dbcheck wrappers when the checked expression
    # is false.
    def __init__(self, exprstr, func, args, kwds):
        # [FUNCTION DOC] HELPER: Builds the exception message from the expression text,
        # function, args and kwargs.
        # A real version of this would set attributes here
        Exception.__init__(self, "dbcheck %r failed (func=%s args=%s kwds=%s)" %
                           (exprstr, func, args, kwds))


def dbcheck(exprstr, globals=None, locals=None):
    # [FUNCTION DOC] HELPER: Decorator factory for debug assertions. Takes an expression string
    # and optional globals and locals mappings; the wrapped function raises DbcheckError when
    # the expression evaluates false at call time.
    "Decorator to implement debugging assertions"
    def decorate(func):
        # [FUNCTION DOC] HELPER: Compiles the expression once and returns a checking wrapper
        # around func.
        expr = compile(exprstr, "dbcheck-%s" % func.__name__, "eval")
        def check(*args, **kwds):
            # [FUNCTION DOC] HELPER: Evaluates the compiled expression (with args and kwds
            # visible as names in its scope); raises DbcheckError if false, otherwise calls func
            # and returns its result.
            if not eval(expr, globals, locals):
                raise DbcheckError(exprstr, func, args, kwds)
            return func(*args, **kwds)
        return check
    return decorate

# -----------------------------------------------

def countcalls(counts):
    # [FUNCTION DOC] HELPER: Decorator factory that counts calls of the decorated function in
    # the given mapping.
    "Decorator to count calls to a function"
    def decorate(func):
        # [FUNCTION DOC] HELPER: Initializes the call counter for the function name to zero and
        # returns a wrapper.
        func_name = func.__name__
        counts[func_name] = 0
        def call(*args, **kwds):
            # [FUNCTION DOC] HELPER: Increments the counter for the function name, then calls
            # func and returns the result.
            counts[func_name] += 1
            return func(*args, **kwds)
        call.__name__ = func_name
        return call
    return decorate

# -----------------------------------------------

def memoize(func):
    # [FUNCTION DOC] HELPER: Decorator that caches results by positional arguments. Unhashable
    # arguments bypass the cache and call the function directly. Preserves the function name.
    saved = {}
    def call(*args):
        # [FUNCTION DOC] HELPER: Returns the cached result for args, computing and storing it on
        # a miss; on an unhashable argument it calls the function without caching.
        try:
            return saved[args]
        except KeyError:
            res = func(*args)
            saved[args] = res
            return res
        except TypeError:
            # Unhashable argument
            return func(*args)
    call.__name__ = func.__name__
    return call

# -----------------------------------------------

class TestDecorators(CPythonTestCase):
    # [CLASS DOC] Test suite for function and method decorators, run through the Dynamo CPython
    # test harness.

    def test_single(self):
        # [FUNCTION DOC] Checks that a staticmethod-decorated function can be called on both the
        # class and an instance.
        class C(object):
            @staticmethod
            def foo(): return 42
        self.assertEqual(C.foo(), 42)
        self.assertEqual(C().foo(), 42)

    def check_wrapper_attrs(self, method_wrapper, format_str):
        # [FUNCTION DOC] HELPER: Wraps a function with the given wrapper type, asserts the
        # wrapper exposes the original function and copies its metadata attributes, checks the
        # wrapper repr against format_str, and returns the wrapper.
        def func(x):
            # [FUNCTION DOC] HELPER: Identity function used as the wrapped target.
            return x
        wrapper = method_wrapper(func)

        self.assertIs(wrapper.__func__, func)
        self.assertIs(wrapper.__wrapped__, func)

        for attr in ('__module__', '__qualname__', '__name__',
                     '__doc__', '__annotations__'):
            self.assertIs(getattr(wrapper, attr),
                          getattr(func, attr))

        self.assertEqual(repr(wrapper), format_str.format(func))
        return wrapper

    def test_staticmethod(self):
        # [FUNCTION DOC] Checks staticmethod wrapper attributes and repr, and that the wrapper
        # is directly callable.
        wrapper = self.check_wrapper_attrs(staticmethod, '<staticmethod({!r})>')

        # bpo-43682: Static methods are callable since Python 3.10
        self.assertEqual(wrapper(1), 1)

    def test_classmethod(self):
        # [FUNCTION DOC] Checks classmethod wrapper attributes and repr, and that calling it
        # directly raises a TypeError.
        wrapper = self.check_wrapper_attrs(classmethod, '<classmethod({!r})>')

        self.assertRaises(TypeError, wrapper, 1)

    def test_dotted(self):
        # [FUNCTION DOC] Checks that a decorator obtained through an attribute lookup on an
        # instance works.
        decorators = MiscDecorators()
        @decorators.author('Cleese')
        def foo(): return 42
        self.assertEqual(foo(), 42)
        self.assertEqual(foo.author, 'Cleese')

    def test_argforms(self):
        # [FUNCTION DOC] Checks decorator argument forms: star-args, keyword unpacking, mixed
        # arguments, trailing comma.
        # A few tests of argument passing, as we use restricted form
        # of expressions for decorators.

        def noteargs(*args, **kwds):
            # [FUNCTION DOC] HELPER: Decorator factory that records its arguments as an
            # attribute of the decorated function.
            def decorate(func):
                # [FUNCTION DOC] HELPER: Stores the captured args and kwds as the function
                # attribute and returns it.
                setattr(func, 'dbval', (args, kwds))
                return func
            return decorate

        args = ( 'Now', 'is', 'the', 'time' )
        kwds = dict(one=1, two=2)
        @noteargs(*args, **kwds)
        def f1(): return 42
        self.assertEqual(f1(), 42)
        self.assertEqual(f1.dbval, (args, kwds))

        @noteargs('terry', 'gilliam', eric='idle', john='cleese')
        def f2(): return 84
        self.assertEqual(f2(), 84)
        self.assertEqual(f2.dbval, (('terry', 'gilliam'),
                                     dict(eric='idle', john='cleese')))

        @noteargs(1, 2,)
        def f3(): pass
        self.assertEqual(f3.dbval, ((1, 2), {}))

    def test_dbcheck(self):
        # [FUNCTION DOC] Checks that the debug-check decorator passes valid calls and raises
        # DbcheckError otherwise.
        @dbcheck('args[1] is not None')
        def f(a, b):
            # [FUNCTION DOC] HELPER: Decorated function adding two values.
            return a + b
        self.assertEqual(f(1, 2), 3)
        self.assertRaises(DbcheckError, f, 1, None)

    def test_memoize(self):
        # [FUNCTION DOC] Checks memoization over a call-counting function: repeated hashable
        # arguments are counted once, unhashable arguments are recomputed on every call.
        counts = {}

        @memoize
        @countcalls(counts)
        def double(x):
            # [FUNCTION DOC] HELPER: Decorated function returning its argument doubled.
            return x * 2
        self.assertEqual(double.__name__, 'double')

        self.assertEqual(counts, dict(double=0))

        # Only the first call with a given argument bumps the call count:
        #
        self.assertEqual(double(2), 4)
        self.assertEqual(counts['double'], 1)
        self.assertEqual(double(2), 4)
        self.assertEqual(counts['double'], 1)
        self.assertEqual(double(3), 6)
        self.assertEqual(counts['double'], 2)

        # Unhashable arguments do not get memoized:
        #
        self.assertEqual(double([10]), [10, 10])
        self.assertEqual(counts['double'], 3)
        self.assertEqual(double([10]), [10, 10])
        self.assertEqual(counts['double'], 4)

    def test_errors(self):
        # [FUNCTION DOC] Checks error handling: invalid decorator expressions give a SyntaxError
        # at compile time, literal-like expressions give a TypeError when run, and undefined
        # names, None attribute lookups and raising decorators give NameError, TypeError,
        # AttributeError and NotImplementedError respectively.

        # Test SyntaxErrors:
        for stmt in ("x,", "x, y", "x = y", "pass", "import sys"):
            compile(stmt, "test", "exec")  # Sanity check.
            with self.assertRaises(SyntaxError):
                compile(f"@{stmt}\ndef f(): pass", "test", "exec")

        # Test TypeErrors that used to be SyntaxErrors:
        for expr in ("1.+2j", "[1, 2][-1]", "(1, 2)", "True", "...", "None"):
            compile(expr, "test", "eval")  # Sanity check.
            with self.assertRaises(TypeError):
                exec(f"@{expr}\ndef f(): pass")

        def unimp(func):
            # [FUNCTION DOC] HELPER: Decorator that always raises NotImplementedError.
            raise NotImplementedError
        context = dict(nullval=None, unimp=unimp)

        for expr, exc in [ ("undef", NameError),
                           ("nullval", TypeError),
                           ("nullval.attr", AttributeError),
                           ("unimp", NotImplementedError)]:
            codestr = "@%s\ndef f(): pass\nassert f() is None" % expr
            code = compile(codestr, "test", "exec")
            self.assertRaises(exc, eval, code, context)

    def test_expressions(self):
        # [FUNCTION DOC] Checks that a range of arbitrary expression forms compile successfully
        # as decorators.
        for expr in (
            "(x,)", "(x, y)", "x := y", "(x := y)", "x @y", "(x @ y)", "x[0]",
            "w[x].y.z", "w + x - (y + z)", "x(y)()(z)", "[w, x, y][z]", "x.y",
        ):
            compile(f"@{expr}\ndef f(): pass", "test", "exec")

    def test_double(self):
        # [FUNCTION DOC] Checks that stacking two attribute-setting decorators applies both sets
        # of attributes.
        class C(object):
            @funcattrs(abc=1, xyz="haha")
            @funcattrs(booh=42)
            def foo(self): return 42
        self.assertEqual(C().foo(), 42)
        self.assertEqual(C.foo.abc, 1)
        self.assertEqual(C.foo.xyz, "haha")
        self.assertEqual(C.foo.booh, 42)

    def test_order(self):
        # [FUNCTION DOC] Checks that stacked decorators are applied bottom-up, so the outermost
        # one wins.
        # Test that decorators are applied in the proper order to the function
        # they are decorating.
        def callnum(num):
            # [FUNCTION DOC] HELPER: Decorator factory returning a decorator that replaces the
            # function by one returning num.
            """Decorator factory that returns a decorator that replaces the
            passed-in function with one that returns the value of 'num'"""
            def deco(func):
                # [FUNCTION DOC] HELPER: Returns a zero-argument function that returns the
                # captured number.
                return lambda: num
            return deco
        @callnum(2)
        @callnum(1)
        def foo(): return 42
        self.assertEqual(foo(), 2,
                            "Application order of decorators is incorrect")

    def test_eval_order(self):
        # [FUNCTION DOC] Checks evaluation order of decorator expressions: for each decorator in
        # source order the name, arguments and factory call are evaluated, then decorators are
        # applied in reverse order. Compares the recorded action list with the explicit
        # nested-call form.
        # Evaluating a decorated function involves four steps for each
        # decorator-maker (the function that returns a decorator):
        #
        #    1: Evaluate the decorator-maker name
        #    2: Evaluate the decorator-maker arguments (if any)
        #    3: Call the decorator-maker to make a decorator
        #    4: Call the decorator
        #
        # When there are multiple decorators, these steps should be
        # performed in the above order for each decorator, but we should
        # iterate through the decorators in the reverse of the order they
        # appear in the source.

        actions = []

        def make_decorator(tag):
            # [FUNCTION DOC] HELPER: Decorator factory that records a creation action for the
            # tag and returns a decorator.
            actions.append('makedec' + tag)
            def decorate(func):
                # [FUNCTION DOC] HELPER: Records an application action for the tag and returns
                # func unchanged.
                actions.append('calldec' + tag)
                return func
            return decorate

        class NameLookupTracer (object):
            # [CLASS DOC] HELPER: Object that records which attribute is looked up, by index, in
            # the action list.
            def __init__(self, index):
                # [FUNCTION DOC] HELPER: Stores the index of this tracer.
                self.index = index

            def __getattr__(self, fname):
                # [FUNCTION DOC] HELPER: Attribute hook that records name or argument evaluation
                # for this tracer and returns the decorator factory or the index string; any
                # other attribute fails an assertion.
                if fname == 'make_decorator':
                    opname, res = ('evalname', make_decorator)
                elif fname == 'arg':
                    opname, res = ('evalargs', str(self.index))
                else:
                    assert False, "Unknown attrname %s" % fname
                actions.append('%s%d' % (opname, self.index))
                return res

        c1, c2, c3 = map(NameLookupTracer, [ 1, 2, 3 ])

        expected_actions = [ 'evalname1', 'evalargs1', 'makedec1',
                             'evalname2', 'evalargs2', 'makedec2',
                             'evalname3', 'evalargs3', 'makedec3',
                             'calldec3', 'calldec2', 'calldec1' ]

        actions = []
        @c1.make_decorator(c1.arg)
        @c2.make_decorator(c2.arg)
        @c3.make_decorator(c3.arg)
        def foo(): return 42
        self.assertEqual(foo(), 42)

        self.assertEqual(actions, expected_actions)

        # Test the equivalence claim in chapter 7 of the reference manual.
        #
        actions = []
        def bar(): return 42
        bar = c1.make_decorator(c1.arg)(c2.make_decorator(c2.arg)(c3.make_decorator(c3.arg)(bar)))
        self.assertEqual(bar(), 42)
        self.assertEqual(actions, expected_actions)

    def test_bound_function_inside_classmethod(self):
        # [FUNCTION DOC] Checks that a classmethod can wrap a bound method of another class and
        # be called on the class.
        class A:
            def foo(self, cls):
                # [FUNCTION DOC] HELPER: Method returning the string spam.
                return 'spam'

        class B:
            bar = classmethod(A().foo)

        self.assertEqual(B.bar(), 'spam')


class TestClassDecorators(CPythonTestCase):
    # [CLASS DOC] Test suite for class decorators, run through the Dynamo CPython test harness.

    def test_simple(self):
        # [FUNCTION DOC] Checks that a decorator can set an attribute on a class and return it.
        def plain(x):
            # [FUNCTION DOC] HELPER: Sets an attribute on the class and returns it.
            x.extra = 'Hello'
            return x
        @plain
        class C(object): pass
        self.assertEqual(C.extra, 'Hello')

    def test_double(self):
        # [FUNCTION DOC] Checks that stacked class decorators are applied bottom-up and
        # accumulate their effects.
        def ten(x):
            # [FUNCTION DOC] HELPER: Sets the extra attribute to ten and returns the class.
            x.extra = 10
            return x
        def add_five(x):
            # [FUNCTION DOC] HELPER: Adds five to the extra attribute and returns the class.
            x.extra += 5
            return x

        @add_five
        @ten
        class C(object): pass
        self.assertEqual(C.extra, 15)

    def test_order(self):
        # [FUNCTION DOC] Checks that the decorator closest to the class is applied first, so the
        # outer result wins.
        def applied_first(x):
            # [FUNCTION DOC] HELPER: Sets the extra attribute to the string first and returns
            # the class.
            x.extra = 'first'
            return x
        def applied_second(x):
            # [FUNCTION DOC] HELPER: Sets the extra attribute to the string second and returns
            # the class.
            x.extra = 'second'
            return x
        @applied_second
        @applied_first
        class C(object): pass
        self.assertEqual(C.extra, 'second')

if __name__ == "__main__":
    run_tests()
