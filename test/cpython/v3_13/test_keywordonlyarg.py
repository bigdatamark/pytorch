# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# [MODULE DOC] Dynamo-patched copy of the CPython 3.13 test suite for keyword-only
# arguments (PEP 3102). Defines small helper functions and a class using keyword-only
# parameters, plus one CPythonTestCase that checks compile-time syntax errors, call-time
# TypeErrors, defaults handling, name mangling and default evaluation order. Importing
# it only pulls in torch and dynamo test utilities; running the file calls run_tests.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_keywordonlyarg.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

"""Unit tests for the keyword only argument specified in PEP 3102."""

__author__ = "Jiwon Seo"
__email__ = "seojiwon at gmail dot com"

import unittest

def posonly_sum(pos_arg1, *arg, **kwarg):
    # [FUNCTION DOC] HELPER: Sums one required positional argument, any extra positional
    # arguments, and all extra keyword argument values. Returns the total.
    return pos_arg1 + sum(arg) + sum(kwarg.values())
def keywordonly_sum(*, k1=0, k2):
    # [FUNCTION DOC] HELPER: Returns the sum of two keyword-only arguments; k1 defaults
    # to 0 and k2 is required (a keyword-only parameter without default may follow one
    # with a default).
    return k1 + k2
def keywordonly_nodefaults_sum(*, k1, k2):
    # [FUNCTION DOC] HELPER: Returns k1 plus k2; both are keyword-only and neither has a
    # default, so omitting either raises TypeError.
    return k1 + k2
def keywordonly_and_kwarg_sum(*, k1, k2, **kwarg):
    # [FUNCTION DOC] HELPER: Returns k1 plus k2 plus the sum of any extra keyword
    # argument values; k1 and k2 are required keyword-only parameters.
    return k1 + k2 + sum(kwarg.values())
def mixedargs_sum(a, b=0, *arg, k1, k2=0):
    # [FUNCTION DOC] HELPER: Mixes positional parameters (a required, b default 0),
    # variadic positionals and keyword-only k1 (required) and k2 (default 0). Returns
    # the sum of all of them including the extra positionals.
    return a + b + k1 + k2 + sum(arg)
def mixedargs_sum2(a, b=0, *arg, k1, k2=0, **kwargs):
    # [FUNCTION DOC] HELPER: Same as mixedargs_sum but also accepts arbitrary extra
    # keyword arguments; their values are added to the returned total.
    return a + b + k1 + k2 + sum(arg) + sum(kwargs.values())

def sortnum(*nums, reverse=False):
    # [FUNCTION DOC] HELPER: Returns the positional arguments as a sorted list; the
    # keyword-only flag reverse selects descending order.
    return sorted(list(nums), reverse=reverse)

def sortwords(*words, reverse=False, **kwargs):
    # [FUNCTION DOC] HELPER: Like sortnum but for words; extra keyword arguments are
    # accepted and ignored. Returns the sorted list, descending when reverse is true.
    return sorted(list(words), reverse=reverse)

class Foo:
    # [CLASS DOC] HELPER: Small fixture holding two integers k1 and k2 that can only be
    # supplied by keyword. Provides a setter and a sum method; used to test keyword-only
    # arguments on constructors and methods.
    def __init__(self, *, k1, k2=0):
        # [FUNCTION DOC] HELPER: Stores the required keyword-only k1 and the optional
        # keyword-only k2 (default 0) as attributes.
        self.k1 = k1
        self.k2 = k2
    def set(self, p1, *, k1, k2):
        # [FUNCTION DOC] HELPER: Takes a positional p1 (unused) and required keyword-
        # only k1 and k2, and overwrites the stored attributes with them.
        self.k1 = k1
        self.k2 = k2
    def sum(self):
        # [FUNCTION DOC] HELPER: Returns the sum of the stored k1 and k2 attributes.
        return self.k1 + self.k2

class KeywordOnlyArgTestCase(CPythonTestCase):
    # [CLASS DOC] Test case (CPythonTestCase) covering PEP 3102 keyword-only arguments:
    # invalid definition and call syntax, error messages, successful calls through the
    # module level helpers, __kwdefaults__ handling, methods, name mangling and default
    # evaluation order. Has no state of its own.
    def assertRaisesSyntaxError(self, codestr):
        # [FUNCTION DOC] HELPER: Asserts that compiling the given source string in
        # single mode raises SyntaxError. Takes the source text; returns nothing; fails
        # the test if no SyntaxError occurs.
        def shouldRaiseSyntaxError(s):
            # [FUNCTION DOC] HELPER: Compiles the source string as a single statement;
            # the caller expects it to raise SyntaxError.
            compile(s, "<test>", "single")
        self.assertRaises(SyntaxError, shouldRaiseSyntaxError, codestr)

    def testSyntaxErrorForFunctionDefinition(self):
        # [FUNCTION DOC] Checks that invalid function definitions raise SyntaxError: a
        # bare star with nothing after it, duplicate parameter names across keyword-
        # only/variadic/positional, a star followed directly by the keyword-argument
        # collector, a non-identifier name and a tuple parameter.
        self.assertRaisesSyntaxError("def f(p, *):\n  pass\n")
        self.assertRaisesSyntaxError("def f(p1, *, p1=100):\n  pass\n")
        self.assertRaisesSyntaxError("def f(p1, *k1, k1=100):\n  pass\n")
        self.assertRaisesSyntaxError("def f(p1, *, k1, k1=100):\n  pass\n")
        self.assertRaisesSyntaxError("def f(p1, *, **k1):\n  pass\n")
        self.assertRaisesSyntaxError("def f(p1, *, k1, **k1):\n  pass\n")
        self.assertRaisesSyntaxError("def f(p1, *, None, **k1):\n  pass\n")
        self.assertRaisesSyntaxError("def f(p, *, (k1, k2), **kw):\n  pass\n")

    def testSyntaxForManyArguments(self):
        # [FUNCTION DOC] Compiles generated function definitions with 300 positional
        # parameters and with 300 keyword-only parameters and checks no limit error is
        # raised (only compile is exercised, never called).
        # more than 255 positional arguments, should compile ok
        fundef = "def f(%s):\n  pass\n" % ', '.join('i%d' % i for i in range(300))
        compile(fundef, "<test>", "single")
        # more than 255 keyword-only arguments, should compile ok
        fundef = "def f(*, %s):\n  pass\n" % ', '.join('i%d' % i for i in range(300))
        compile(fundef, "<test>", "single")

    def testTooManyPositionalErrorMessage(self):
        # [FUNCTION DOC] Calls a function with one required and one optional positional
        # plus a keyword-only parameter using three positionals and checks the TypeError
        # message names the qualified name and says it takes from 1 to 2 positional
        # arguments but 3 were given.
        def f(a, b=None, *, c=None):
            pass
        with self.assertRaises(TypeError) as exc:
            f(1, 2, 3)
        expected = (f"{f.__qualname__}() takes from 1 to 2 "
                    "positional arguments but 3 were given")
        self.assertEqual(str(exc.exception), expected)

    def testSyntaxErrorForFunctionCall(self):
        # [FUNCTION DOC] Checks that invalid call expressions (positional after keyword,
        # duplicate keyword with star-args) raise SyntaxError at compile time.
        self.assertRaisesSyntaxError("f(p, k=1, p2)")
        self.assertRaisesSyntaxError("f(p, k1=50, *(1,2), k1=100)")

    def testRaiseErrorFuncallWithUnexpectedKeywordArgument(self):
        # [FUNCTION DOC] Checks TypeError is raised for calls missing a required
        # keyword-only argument, passing a positional where only keywords are allowed
        # (including the constructor of the fixture class), or passing an unknown
        # keyword argument.
        self.assertRaises(TypeError, keywordonly_sum, ())
        self.assertRaises(TypeError, keywordonly_nodefaults_sum, ())
        self.assertRaises(TypeError, Foo, ())
        try:
            keywordonly_sum(k2=100, non_existing_arg=200)
            self.fail("should raise TypeError")
        except TypeError:
            pass
        try:
            keywordonly_nodefaults_sum(k2=2)
            self.fail("should raise TypeError")
        except TypeError:
            pass

    def testFunctionCall(self):
        # [FUNCTION DOC] Exercises successful calls: positional/star/double-star combos,
        # keyword-only with defaults, keyword-only plus collector, mixed signatures, the
        # fixture class constructor and the sorting helpers with and without the reverse
        # flag; asserts the computed sums and orderings.
        self.assertEqual(1, posonly_sum(1))
        self.assertEqual(1+2, posonly_sum(1,**{"2":2}))
        self.assertEqual(1+2+3, posonly_sum(1,*(2,3)))
        self.assertEqual(1+2+3+4, posonly_sum(1,*(2,3),**{"4":4}))

        self.assertEqual(1, keywordonly_sum(k2=1))
        self.assertEqual(1+2, keywordonly_sum(k1=1, k2=2))

        self.assertEqual(1+2, keywordonly_and_kwarg_sum(k1=1, k2=2))
        self.assertEqual(1+2+3, keywordonly_and_kwarg_sum(k1=1, k2=2, k3=3))
        self.assertEqual(1+2+3+4,
                         keywordonly_and_kwarg_sum(k1=1, k2=2,
                                                    **{"a":3,"b":4}))

        self.assertEqual(1+2, mixedargs_sum(1, k1=2))
        self.assertEqual(1+2+3, mixedargs_sum(1, 2, k1=3))
        self.assertEqual(1+2+3+4, mixedargs_sum(1, 2, k1=3, k2=4))
        self.assertEqual(1+2+3+4+5, mixedargs_sum(1, 2, 3, k1=4, k2=5))

        self.assertEqual(1+2, mixedargs_sum2(1, k1=2))
        self.assertEqual(1+2+3, mixedargs_sum2(1, 2, k1=3))
        self.assertEqual(1+2+3+4, mixedargs_sum2(1, 2, k1=3, k2=4))
        self.assertEqual(1+2+3+4+5, mixedargs_sum2(1, 2, 3, k1=4, k2=5))
        self.assertEqual(1+2+3+4+5+6,
                         mixedargs_sum2(1, 2, 3, k1=4, k2=5, k3=6))
        self.assertEqual(1+2+3+4+5+6,
                         mixedargs_sum2(1, 2, 3, k1=4, **{'k2':5, 'k3':6}))

        self.assertEqual(1, Foo(k1=1).sum())
        self.assertEqual(1+2, Foo(k1=1,k2=2).sum())

        self.assertEqual([1,2,3], sortnum(3,2,1))
        self.assertEqual([3,2,1], sortnum(1,2,3, reverse=True))

        self.assertEqual(['a','b','c'], sortwords('a','c','b'))
        self.assertEqual(['c','b','a'], sortwords('a','c','b', reverse=True))
        self.assertEqual(['c','b','a'],
                         sortwords('a','c','b', reverse=True, ignore='ignore'))

    def testKwDefaults(self):
        # [FUNCTION DOC] Verifies the keyword-only count on the code object, the
        # __kwdefaults__ mapping holding only the defaulted name, and that replacing
        # __kwdefaults__ with a mapping that lacks k2 makes a call fail with TypeError.
        def foo(p1,p2=0, *, k1, k2=0):
            return p1 + p2 + k1 + k2

        self.assertEqual(2, foo.__code__.co_kwonlyargcount)
        self.assertEqual({"k2":0}, foo.__kwdefaults__)
        foo.__kwdefaults__ = {"k1":0}
        try:
            foo(1,k1=10)
            self.fail("__kwdefaults__ is not properly changed")
        except TypeError:
            pass

    def test_kwonly_methods(self):
        # [FUNCTION DOC] Checks keyword-only parameters on a method work through an
        # instance and through the class with an explicit self, and that calling via the
        # class without self raises TypeError.
        class Example:
            def f(self, *, k1=1, k2=2):
                return k1, k2

        self.assertEqual(Example().f(k1=1, k2=2), (1, 2))
        self.assertEqual(Example.f(Example(), k1=1, k2=2), (1, 2))
        self.assertRaises(TypeError, Example.f, k1=1, k2=2)

    def test_issue13343(self):
        # [FUNCTION DOC] Regression test: a lambda whose keyword-only default refers to
        # a module name must compile without raising SystemError; the lambda is only
        # created, never called.
        # The Python compiler must scan all symbols of a function to
        # determine their scope: global, local, cell...
        # This was not done for the default values of keyword
        # arguments in a lambda definition, and the following line
        # used to fail with a SystemError.
        lambda *, k1=unittest: None

    def test_mangling(self):
        # [FUNCTION DOC] Checks that a keyword-only parameter with a double-underscore
        # name inside a class is mangled consistently so the method returns its default
        # 42.
        class X:
            def f(self, *, __a=42):
                return __a
        self.assertEqual(X().f(), 42)

    def test_default_evaluation_order(self):
        # [FUNCTION DOC] Checks defaults are evaluated left to right with keyword-only
        # ones after positional ones, for both def and lambda: an undefined name in the
        # second default raises NameError mentioning that name before later undefined
        # names.
        # See issue 16967
        a = 42
        with self.assertRaises(NameError) as err:
            def f(v=a, x=b, *, y=c, z=d):
                pass
        self.assertEqual(str(err.exception), "name 'b' is not defined")
        with self.assertRaises(NameError) as err:
            g = lambda v=a, x=b, *, y=c, z=d: None
        self.assertEqual(str(err.exception), "name 'b' is not defined")


if __name__ == "__main__":
    run_tests()
