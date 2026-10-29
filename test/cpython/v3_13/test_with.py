# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_with.py
# [MODULE DOC] CPython 3.13 unit tests for the with statement (PEP 343), adapted so the test
# cases run under the Dynamo CPython test harness. Defines mock context managers (generator-
# based and nested), and test cases for failures, normal and exceptional flow, non-local control
# flow (break, continue, return, yield, raise), assignment targets, exits that swallow
# exceptions, and multi-item with statements. Import-time side effects: none beyond class and
# function creation; run_tests is called only when run as a program.

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

__TestCase = CPythonTestCase

# ======= END DYNAMO PATCH =======

"""Unit tests for the with statement specified in PEP 343."""


__author__ = "Mike Bland"
__email__ = "mbland at acm dot org"

import sys
import traceback
import unittest
from collections import deque
from contextlib import _GeneratorContextManager, contextmanager, nullcontext


class MockContextManager(_GeneratorContextManager):
    # [CLASS DOC] HELPER: context manager built on the generator context manager class that
    # records whether it was entered, whether it was exited, and the three arguments given to
    # the exit call.
    def __init__(self, *args):
        # [FUNCTION DOC] HELPER: forwards the generator, args and kwargs to the base class
        # and sets the entered flag, the exited flag and the exit arguments to false, false
        # and none.
        super().__init__(*args)
        self.enter_called = False
        self.exit_called = False
        self.exit_args = None

    def __enter__(self):
        # [FUNCTION DOC] HELPER: sets the entered flag then delegates to the base class entry
        # and returns the generated value.
        self.enter_called = True
        return _GeneratorContextManager.__enter__(self)

    def __exit__(self, type, value, traceback):
        # [FUNCTION DOC] HELPER: sets the exited flag, stores the exception type, value and
        # traceback as exit_args, then delegates to the base class exit and returns its result.
        self.exit_called = True
        self.exit_args = (type, value, traceback)
        return _GeneratorContextManager.__exit__(self, type,
                                                 value, traceback)


def mock_contextmanager(func):
    # [FUNCTION DOC] HELPER: decorator turning a generator function into a factory of recording
    # context managers.
    def helper(*args, **kwds):
        # [FUNCTION DOC] HELPER: builds and returns a recording context manager for the wrapped
        # function using the given arguments.
        return MockContextManager(func, args, kwds)
    return helper


class MockResource(object):
    # [CLASS DOC] HELPER: plain resource object with two flags (yielded, stopped) used to
    # observe generator progress.
    def __init__(self):
        # [FUNCTION DOC] HELPER: sets both flags to false.
        self.yielded = False
        self.stopped = False


@mock_contextmanager
def mock_contextmanager_generator():
    # [FUNCTION DOC] HELPER: generator-based context manager that creates a resource, marks it
    # yielded, yields it, and in a finally clause marks it stopped, so the stopped flag shows
    # whether cleanup ran.
    mock = MockResource()
    try:
        mock.yielded = True
        yield mock
    finally:
        mock.stopped = True


class Nested(object):
    # [CLASS DOC] HELPER: re-implementation of the removed nested helper: enters several
    # managers in order and exits them in reverse order like nested with statements; keeps the
    # entered managers in a deque.

    def __init__(self, *managers):
        # [FUNCTION DOC] HELPER: stores the managers and marks the instance as not entered.
        self.managers = managers
        self.entered = None

    def __enter__(self):
        # [FUNCTION DOC] HELPER: raises RuntimeError if already entered; otherwise enters
        # each manager in order, collecting the entry results in a list that is returned. If
        # an entry fails, exits the managers entered so far and re-raises unless the exit
        # swallows the error.
        if self.entered is not None:
            raise RuntimeError("Context is not reentrant")
        self.entered = deque()
        vars = []
        try:
            for mgr in self.managers:
                vars.append(mgr.__enter__())
                self.entered.appendleft(mgr)
        except:
            if not self.__exit__(*sys.exc_info()):
                raise
        return vars

    def __exit__(self, *exc_info):
        # [FUNCTION DOC] HELPER: exits the entered managers last-in first-out, passing the
        # exception info along; a manager returning true clears the pending exception, and an
        # exception raised by a manager replaces the current one. Clears the entered state, and
        # if the final exception state differs from the original it is raised.
        # KNOWN ISSUE: the final state is a tuple of type, value and traceback (or all none),
        # and raising a tuple is not valid, so that path would fail with a TypeError instead of
        # raising the intended exception; the tests never reach it.
        # Behave like nested with statements
        # first in, last out
        # New exceptions override old ones
        ex = exc_info
        for mgr in self.entered:
            try:
                if mgr.__exit__(*ex):
                    ex = (None, None, None)
            except BaseException as e:
                ex = (type(e), e, e.__traceback__)
        self.entered = None
        if ex is not exc_info:
            raise ex


class MockNested(Nested):
    # [CLASS DOC] HELPER: recording variant of Nested that tracks entry, exit and the
    # exit arguments.
    def __init__(self, *managers):
        # [FUNCTION DOC] HELPER: initializes the base class and the entered flag, exited flag
        # and exit arguments.
        Nested.__init__(self, *managers)
        self.enter_called = False
        self.exit_called = False
        self.exit_args = None

    def __enter__(self):
        # [FUNCTION DOC] HELPER: sets the entered flag and delegates to the base class entry,
        # returning its list of results.
        self.enter_called = True
        return Nested.__enter__(self)

    def __exit__(self, *exc_info):
        # [FUNCTION DOC] HELPER: sets the exited flag, stores the exception info tuple and
        # delegates to the base class exit.
        self.exit_called = True
        self.exit_args = exc_info
        return Nested.__exit__(self, *exc_info)


class FailureTestCase(__TestCase):
    # [CLASS DOC] Tests of the errors raised by with statements: undefined names, missing
    # enter or exit methods, syntax errors for invalid targets, and exceptions raised by
    # enter and exit.
    def testNameError(self):
        # [FUNCTION DOC] A with statement on an undefined name raises NameError.
        def fooNotDeclared():
            # [FUNCTION DOC] HELPER: runs a with statement on an undeclared name.
            with foo: pass
        self.assertRaises(NameError, fooNotDeclared)

    def testEnterAttributeError1(self):
        # [FUNCTION DOC] An object with exit but no enter makes the with statement raise
        # TypeError mentioning the context manager.
        class LacksEnter(object):
            # [CLASS DOC] HELPER: has an exit method but no enter method.
            def __exit__(self, type, value, traceback):
                # [FUNCTION DOC] HELPER: does nothing.
                pass

        def fooLacksEnter():
            # [FUNCTION DOC] HELPER: uses an instance of the class lacking enter in a
            # with statement.
            foo = LacksEnter()
            with foo: pass
        self.assertRaisesRegex(TypeError, 'the context manager', fooLacksEnter)

    def testEnterAttributeError2(self):
        # [FUNCTION DOC] An object lacking both enter and exit makes the with statement raise
        # TypeError mentioning the context manager.
        class LacksEnterAndExit(object):
            # [CLASS DOC] HELPER: defines neither enter nor exit.
            pass

        def fooLacksEnterAndExit():
            # [FUNCTION DOC] HELPER: uses an instance of the class lacking both methods in a
            # with statement.
            foo = LacksEnterAndExit()
            with foo: pass
        self.assertRaisesRegex(TypeError, 'the context manager', fooLacksEnterAndExit)

    def testExitAttributeError(self):
        # [FUNCTION DOC] An object with enter but no exit makes the with statement raise
        # TypeError mentioning the context manager and exit.
        class LacksExit(object):
            # [CLASS DOC] HELPER: has an enter method but no exit method.
            def __enter__(self):
                # [FUNCTION DOC] HELPER: does nothing.
                pass

        def fooLacksExit():
            # [FUNCTION DOC] HELPER: uses an instance of the class lacking exit in a
            # with statement.
            foo = LacksExit()
            with foo: pass
        self.assertRaisesRegex(TypeError, 'the context manager.*__exit__', fooLacksExit)

    def assertRaisesSyntaxError(self, codestr):
        # [FUNCTION DOC] HELPER: asserts that compiling the given source string
        # raises SyntaxError.
        def shouldRaiseSyntaxError(s):
            # [FUNCTION DOC] HELPER: compiles the given source in single-statement mode.
            compile(s, '', 'single')
        self.assertRaises(SyntaxError, shouldRaiseSyntaxError, codestr)

    def testAssignmentToNoneError(self):
        # [FUNCTION DOC] Binding a with target to the constant none, bare or parenthesized, is a
        # syntax error.
        self.assertRaisesSyntaxError('with mock as None:\n  pass')
        self.assertRaisesSyntaxError(
            'with mock as (None):\n'
            '  pass')

    def testAssignmentToTupleOnlyContainingNoneError(self):
        # [FUNCTION DOC] Binding a with target to a tuple containing only none is a
        # syntax error.
        self.assertRaisesSyntaxError('with mock as None,:\n  pass')
        self.assertRaisesSyntaxError(
            'with mock as (None,):\n'
            '  pass')

    def testAssignmentToTupleContainingNoneError(self):
        # [FUNCTION DOC] Binding a with target to a tuple that contains none among other names
        # is a syntax error.
        self.assertRaisesSyntaxError(
            'with mock as (foo, None, bar):\n'
            '  pass')

    def testEnterThrows(self):
        # [FUNCTION DOC] If enter raises, the error propagates and the target attribute is
        # left unchanged.
        class EnterThrows(object):
            # [CLASS DOC] HELPER: manager whose enter raises RuntimeError.
            def __enter__(self):
                # [FUNCTION DOC] HELPER: raises RuntimeError.
                raise RuntimeError("Enter threw")
            def __exit__(self, *args):
                # [FUNCTION DOC] HELPER: does nothing.
                pass

        def shouldThrow():
            # [FUNCTION DOC] HELPER: sets the test attribute to none then runs a with statement
            # binding the manager to that attribute.
            ct = EnterThrows()
            self.foo = None
            # Ruff complains that we're redefining `self.foo` here,
            # but the whole point of the test is to check that `self.foo`
            # is *not* redefined (because `__enter__` raises)
            with ct as self.foo:  # noqa: F811
                pass
        self.assertRaises(RuntimeError, shouldThrow)
        self.assertEqual(self.foo, None)

    def testExitThrows(self):
        # [FUNCTION DOC] If exit raises, the with statement propagates RuntimeError.
        class ExitThrows(object):
            # [CLASS DOC] HELPER: manager whose exit raises RuntimeError.
            def __enter__(self):
                # [FUNCTION DOC] HELPER: returns none.
                return
            def __exit__(self, *args):
                # [FUNCTION DOC] HELPER: raises RuntimeError with the number 42.
                raise RuntimeError(42)
        def shouldThrow():
            # [FUNCTION DOC] HELPER: runs an empty with statement on the manager.
            with ExitThrows():
                pass
        self.assertRaises(RuntimeError, shouldThrow)

class ContextmanagerAssertionMixin(object):
    # [CLASS DOC] HELPER: mixin giving test cases a shared exception object and assertions about
    # the state of recording managers and generators before, inside and after a with statement.

    def setUp(self):
        # [FUNCTION DOC] HELPER: creates the shared test exception then calls the base
        # class setup.
        self.TEST_EXCEPTION = RuntimeError("test exception")
        super().setUp()

    def assertInWithManagerInvariants(self, mock_manager):
        # [FUNCTION DOC] HELPER: asserts the manager was entered, not yet exited, and has no
        # recorded exit arguments.
        self.assertTrue(mock_manager.enter_called)
        self.assertFalse(mock_manager.exit_called)
        self.assertEqual(mock_manager.exit_args, None)

    def assertAfterWithManagerInvariants(self, mock_manager, exit_args):
        # [FUNCTION DOC] HELPER: asserts the manager was entered and exited and that its
        # recorded exit arguments equal the given ones.
        self.assertTrue(mock_manager.enter_called)
        self.assertTrue(mock_manager.exit_called)
        self.assertEqual(mock_manager.exit_args, exit_args)

    def assertAfterWithManagerInvariantsNoError(self, mock_manager):
        # [FUNCTION DOC] HELPER: asserts the manager exited normally, with exit arguments
        # all none.
        self.assertAfterWithManagerInvariants(mock_manager,
            (None, None, None))

    def assertInWithGeneratorInvariants(self, mock_generator):
        # [FUNCTION DOC] HELPER: asserts the generator yielded and has not yet finished
        # its cleanup.
        self.assertTrue(mock_generator.yielded)
        self.assertFalse(mock_generator.stopped)

    def assertAfterWithGeneratorInvariantsNoError(self, mock_generator):
        # [FUNCTION DOC] HELPER: asserts the generator yielded and completed its cleanup.
        self.assertTrue(mock_generator.yielded)
        self.assertTrue(mock_generator.stopped)

    def raiseTestException(self):
        # [FUNCTION DOC] HELPER: raises the shared test exception.
        raise self.TEST_EXCEPTION

    def assertAfterWithManagerInvariantsWithError(self, mock_manager,
                                                  exc_type=None):
        # [FUNCTION DOC] HELPER: asserts the manager was entered and exited with an exception.
        # When no type is given, expects the shared test exception as the exit value and its
        # type as the exit type; also checks the value is an instance of that type and the
        # traceback is not none.
        self.assertTrue(mock_manager.enter_called)
        self.assertTrue(mock_manager.exit_called)
        if exc_type is None:
            self.assertEqual(mock_manager.exit_args[1], self.TEST_EXCEPTION)
            exc_type = type(self.TEST_EXCEPTION)
        self.assertEqual(mock_manager.exit_args[0], exc_type)
        # Test the __exit__ arguments. Issue #7853
        self.assertIsInstance(mock_manager.exit_args[1], exc_type)
        self.assertIsNot(mock_manager.exit_args[2], None)

    def assertAfterWithGeneratorInvariantsWithError(self, mock_generator):
        # [FUNCTION DOC] HELPER: asserts the generator yielded and completed its cleanup after
        # an error.
        self.assertTrue(mock_generator.yielded)
        self.assertTrue(mock_generator.stopped)


class NonexceptionalTestCase(__TestCase, ContextmanagerAssertionMixin):
    # [CLASS DOC] Tests of with statements that finish without raising, using recording
    # generator managers.
    def testInlineGeneratorSyntax(self):
        # [FUNCTION DOC] A with statement on a generator manager created inline works.
        with mock_contextmanager_generator():
            pass

    def testUnboundGenerator(self):
        # [FUNCTION DOC] A manager stored in a variable first is entered and exited normally.
        mock = mock_contextmanager_generator()
        with mock:
            pass
        self.assertAfterWithManagerInvariantsNoError(mock)

    def testInlineGeneratorBoundSyntax(self):
        # [FUNCTION DOC] Binding the generated resource to a name works and the name stays bound
        # after the block.
        with mock_contextmanager_generator() as foo:
            self.assertInWithGeneratorInvariants(foo)
        # FIXME: In the future, we'll try to keep the bound names from leaking
        self.assertAfterWithGeneratorInvariantsNoError(foo)

    def testInlineGeneratorBoundToExistingVariable(self):
        # [FUNCTION DOC] Same as the bound-name check, binding to a name used again after
        # the block.
        with mock_contextmanager_generator() as foo:
            self.assertInWithGeneratorInvariants(foo)
        self.assertAfterWithGeneratorInvariantsNoError(foo)

    def testInlineGeneratorBoundToDottedVariable(self):
        # [FUNCTION DOC] Binding the resource to an attribute of self works.
        with mock_contextmanager_generator() as self.foo:
            self.assertInWithGeneratorInvariants(self.foo)
        self.assertAfterWithGeneratorInvariantsNoError(self.foo)

    def testBoundGenerator(self):
        # [FUNCTION DOC] A stored manager bound to a name shows correct state inside and after
        # the block for both manager and generator.
        mock = mock_contextmanager_generator()
        with mock as foo:
            self.assertInWithGeneratorInvariants(foo)
            self.assertInWithManagerInvariants(mock)
        self.assertAfterWithGeneratorInvariantsNoError(foo)
        self.assertAfterWithManagerInvariantsNoError(mock)

    def testNestedSingleStatements(self):
        # [FUNCTION DOC] Two nested with statements enter and exit in the correct order with
        # correct state at each level.
        mock_a = mock_contextmanager_generator()
        with mock_a as foo:
            mock_b = mock_contextmanager_generator()
            with mock_b as bar:
                self.assertInWithManagerInvariants(mock_a)
                self.assertInWithManagerInvariants(mock_b)
                self.assertInWithGeneratorInvariants(foo)
                self.assertInWithGeneratorInvariants(bar)
            self.assertAfterWithManagerInvariantsNoError(mock_b)
            self.assertAfterWithGeneratorInvariantsNoError(bar)
            self.assertInWithManagerInvariants(mock_a)
            self.assertInWithGeneratorInvariants(foo)
        self.assertAfterWithManagerInvariantsNoError(mock_a)
        self.assertAfterWithGeneratorInvariantsNoError(foo)


class NestedNonexceptionalTestCase(__TestCase,
    ContextmanagerAssertionMixin):
    # [CLASS DOC] Tests of the Nested helper in normal flow, for single and multiple managers
    # and different binding forms.
    def testSingleArgInlineGeneratorSyntax(self):
        # [FUNCTION DOC] Nested with one manager runs inline.
        with Nested(mock_contextmanager_generator()):
            pass

    def testSingleArgBoundToNonTuple(self):
        # [FUNCTION DOC] Nested with one manager bound to a plain name binds the whole
        # result list.
        m = mock_contextmanager_generator()
        # This will bind all the arguments to nested() into a single list
        # assigned to foo.
        with Nested(m) as foo:
            self.assertInWithManagerInvariants(m)
        self.assertAfterWithManagerInvariantsNoError(m)

    def testSingleArgBoundToSingleElementParenthesizedList(self):
        # [FUNCTION DOC] Nested with one manager bound to a parenthesized name binds the whole
        # result list.
        m = mock_contextmanager_generator()
        # This will bind all the arguments to nested() into a single list
        # assigned to foo.
        with Nested(m) as (foo):
            self.assertInWithManagerInvariants(m)
        self.assertAfterWithManagerInvariantsNoError(m)

    def testSingleArgBoundToMultipleElementTupleError(self):
        # [FUNCTION DOC] Binding a single-manager result to a two-name tuple raises ValueError.
        def shouldThrowValueError():
            # [FUNCTION DOC] HELPER: runs the with statement that should raise.
            with Nested(mock_contextmanager_generator()) as (foo, bar):
                pass
        self.assertRaises(ValueError, shouldThrowValueError)

    def testSingleArgUnbound(self):
        # [FUNCTION DOC] A recording nested helper with one manager records both entries
        # and exits.
        mock_contextmanager = mock_contextmanager_generator()
        mock_nested = MockNested(mock_contextmanager)
        with mock_nested:
            self.assertInWithManagerInvariants(mock_contextmanager)
            self.assertInWithManagerInvariants(mock_nested)
        self.assertAfterWithManagerInvariantsNoError(mock_contextmanager)
        self.assertAfterWithManagerInvariantsNoError(mock_nested)

    def testMultipleArgUnbound(self):
        # [FUNCTION DOC] A recording nested helper with three managers enters and exits all
        # of them.
        m = mock_contextmanager_generator()
        n = mock_contextmanager_generator()
        o = mock_contextmanager_generator()
        mock_nested = MockNested(m, n, o)
        with mock_nested:
            self.assertInWithManagerInvariants(m)
            self.assertInWithManagerInvariants(n)
            self.assertInWithManagerInvariants(o)
            self.assertInWithManagerInvariants(mock_nested)
        self.assertAfterWithManagerInvariantsNoError(m)
        self.assertAfterWithManagerInvariantsNoError(n)
        self.assertAfterWithManagerInvariantsNoError(o)
        self.assertAfterWithManagerInvariantsNoError(mock_nested)

    def testMultipleArgBound(self):
        # [FUNCTION DOC] Binding the results of three managers to three names gives each a live
        # resource, completed after the block.
        mock_nested = MockNested(mock_contextmanager_generator(),
            mock_contextmanager_generator(), mock_contextmanager_generator())
        with mock_nested as (m, n, o):
            self.assertInWithGeneratorInvariants(m)
            self.assertInWithGeneratorInvariants(n)
            self.assertInWithGeneratorInvariants(o)
            self.assertInWithManagerInvariants(mock_nested)
        self.assertAfterWithGeneratorInvariantsNoError(m)
        self.assertAfterWithGeneratorInvariantsNoError(n)
        self.assertAfterWithGeneratorInvariantsNoError(o)
        self.assertAfterWithManagerInvariantsNoError(mock_nested)


class ExceptionalTestCase(ContextmanagerAssertionMixin, __TestCase):
    # [CLASS DOC] Tests of with statements whose body raises, checking both managers and
    # generators clean up and see the exception.
    def testSingleResource(self):
        # [FUNCTION DOC] The exception propagates; the manager sees the error and the resource
        # generator completes cleanup.
        cm = mock_contextmanager_generator()
        def shouldThrow():
            # [FUNCTION DOC] HELPER: runs the body that raises the test exception.
            with cm as self.resource:
                self.assertInWithManagerInvariants(cm)
                self.assertInWithGeneratorInvariants(self.resource)
                self.raiseTestException()
        self.assertRaises(RuntimeError, shouldThrow)
        self.assertAfterWithManagerInvariantsWithError(cm)
        self.assertAfterWithGeneratorInvariantsWithError(self.resource)

    def testExceptionNormalized(self):
        # [FUNCTION DOC] A zero-division error raised in the body is passed to the exit as that
        # exception type.
        cm = mock_contextmanager_generator()
        def shouldThrow():
            # [FUNCTION DOC] HELPER: runs a body dividing by zero.
            with cm as self.resource:
                # Note this relies on the fact that 1 // 0 produces an exception
                # that is not normalized immediately.
                1 // 0
        self.assertRaises(ZeroDivisionError, shouldThrow)
        self.assertAfterWithManagerInvariantsWithError(cm, ZeroDivisionError)

    def testNestedSingleStatements(self):
        # [FUNCTION DOC] An exception in the inner of two nested with statements reaches both
        # managers' exit.
        mock_a = mock_contextmanager_generator()
        mock_b = mock_contextmanager_generator()
        def shouldThrow():
            # [FUNCTION DOC] HELPER: runs the nested bodies and raises the test
            # exception inside.
            with mock_a as self.foo:
                with mock_b as self.bar:
                    self.assertInWithManagerInvariants(mock_a)
                    self.assertInWithManagerInvariants(mock_b)
                    self.assertInWithGeneratorInvariants(self.foo)
                    self.assertInWithGeneratorInvariants(self.bar)
                    self.raiseTestException()
        self.assertRaises(RuntimeError, shouldThrow)
        self.assertAfterWithManagerInvariantsWithError(mock_a)
        self.assertAfterWithManagerInvariantsWithError(mock_b)
        self.assertAfterWithGeneratorInvariantsWithError(self.foo)
        self.assertAfterWithGeneratorInvariantsWithError(self.bar)

    def testMultipleResourcesInSingleStatement(self):
        # [FUNCTION DOC] Several resources bound in a single statement through the recording
        # nested helper all see the exception.
        cm_a = mock_contextmanager_generator()
        cm_b = mock_contextmanager_generator()
        mock_nested = MockNested(cm_a, cm_b)
        def shouldThrow():
            # [FUNCTION DOC] HELPER: runs the body that raises the test exception.
            with mock_nested as (self.resource_a, self.resource_b):
                self.assertInWithManagerInvariants(cm_a)
                self.assertInWithManagerInvariants(cm_b)
                self.assertInWithManagerInvariants(mock_nested)
                self.assertInWithGeneratorInvariants(self.resource_a)
                self.assertInWithGeneratorInvariants(self.resource_b)
                self.raiseTestException()
        self.assertRaises(RuntimeError, shouldThrow)
        self.assertAfterWithManagerInvariantsWithError(cm_a)
        self.assertAfterWithManagerInvariantsWithError(cm_b)
        self.assertAfterWithManagerInvariantsWithError(mock_nested)
        self.assertAfterWithGeneratorInvariantsWithError(self.resource_a)
        self.assertAfterWithGeneratorInvariantsWithError(self.resource_b)

    def testNestedExceptionBeforeInnerStatement(self):
        # [FUNCTION DOC] When the exception occurs before the inner with statement starts, the
        # inner manager is never entered or exited and its target remains unset.
        mock_a = mock_contextmanager_generator()
        mock_b = mock_contextmanager_generator()
        self.bar = None
        def shouldThrow():
            # [FUNCTION DOC] HELPER: raises the test exception before reaching the inner
            # with statement.
            with mock_a as self.foo:
                self.assertInWithManagerInvariants(mock_a)
                self.assertInWithGeneratorInvariants(self.foo)
                self.raiseTestException()
                with mock_b as self.bar:
                    pass
        self.assertRaises(RuntimeError, shouldThrow)
        self.assertAfterWithManagerInvariantsWithError(mock_a)
        self.assertAfterWithGeneratorInvariantsWithError(self.foo)

        # The inner statement stuff should never have been touched
        self.assertEqual(self.bar, None)
        self.assertFalse(mock_b.enter_called)
        self.assertFalse(mock_b.exit_called)
        self.assertEqual(mock_b.exit_args, None)

    def testNestedExceptionAfterInnerStatement(self):
        # [FUNCTION DOC] An exception raised after the inner block finished reaches only the
        # outer manager with an error; the inner manager exited normally.
        mock_a = mock_contextmanager_generator()
        mock_b = mock_contextmanager_generator()
        def shouldThrow():
            # [FUNCTION DOC] HELPER: runs nested blocks then raises the test exception in the
            # outer block.
            with mock_a as self.foo:
                with mock_b as self.bar:
                    self.assertInWithManagerInvariants(mock_a)
                    self.assertInWithManagerInvariants(mock_b)
                    self.assertInWithGeneratorInvariants(self.foo)
                    self.assertInWithGeneratorInvariants(self.bar)
                self.raiseTestException()
        self.assertRaises(RuntimeError, shouldThrow)
        self.assertAfterWithManagerInvariantsWithError(mock_a)
        self.assertAfterWithManagerInvariantsNoError(mock_b)
        self.assertAfterWithGeneratorInvariantsWithError(self.foo)
        self.assertAfterWithGeneratorInvariantsNoError(self.bar)

    def testRaisedStopIteration1(self):
        # [FUNCTION DOC] A StopIteration raised inside a generator-based manager body
        # propagates unchanged.
        # From bug 1462485
        @contextmanager
        def cm():
            # [FUNCTION DOC] HELPER: generator context manager that just yields.
            yield

        def shouldThrow():
            # [FUNCTION DOC] HELPER: raises StopIteration inside a with block.
            with cm():
                raise StopIteration("from with")

        with self.assertRaisesRegex(StopIteration, 'from with'):
            shouldThrow()

    def testRaisedStopIteration2(self):
        # [FUNCTION DOC] A StopIteration raised inside a class-based manager body
        # propagates unchanged.
        # From bug 1462485
        class cm(object):
            # [CLASS DOC] HELPER: manager with empty enter and exit.
            def __enter__(self):
                # [FUNCTION DOC] HELPER: does nothing.
                pass
            def __exit__(self, type, value, traceback):
                # [FUNCTION DOC] HELPER: does nothing, so exceptions are not swallowed.
                pass

        def shouldThrow():
            # [FUNCTION DOC] HELPER: raises StopIteration inside a with block.
            with cm():
                raise StopIteration("from with")

        with self.assertRaisesRegex(StopIteration, 'from with'):
            shouldThrow()

    def testRaisedStopIteration3(self):
        # [FUNCTION DOC] A StopIteration that is raised by calling next on an empty iterator
        # inside a generator manager body propagates.
        # Another variant where the exception hasn't been instantiated
        # From bug 1705170
        @contextmanager
        def cm():
            # [FUNCTION DOC] HELPER: generator context manager that just yields.
            yield

        def shouldThrow():
            # [FUNCTION DOC] HELPER: raises the StopIteration produced by taking next of an
            # empty iterator.
            with cm():
                raise next(iter([]))

        with self.assertRaises(StopIteration):
            shouldThrow()

    def testRaisedGeneratorExit1(self):
        # [FUNCTION DOC] A GeneratorExit raised in a generator-based manager body propagates.
        # From bug 1462485
        @contextmanager
        def cm():
            # [FUNCTION DOC] HELPER: generator context manager that just yields.
            yield

        def shouldThrow():
            # [FUNCTION DOC] HELPER: raises GeneratorExit inside a with block.
            with cm():
                raise GeneratorExit("from with")

        self.assertRaises(GeneratorExit, shouldThrow)

    def testRaisedGeneratorExit2(self):
        # [FUNCTION DOC] A GeneratorExit raised in a class-based manager body propagates.
        # From bug 1462485
        class cm (object):
            # [CLASS DOC] HELPER: manager with empty enter and exit.
            def __enter__(self):
                # [FUNCTION DOC] HELPER: does nothing.
                pass
            def __exit__(self, type, value, traceback):
                # [FUNCTION DOC] HELPER: does nothing.
                pass

        def shouldThrow():
            # [FUNCTION DOC] HELPER: raises GeneratorExit inside a with block.
            with cm():
                raise GeneratorExit("from with")

        self.assertRaises(GeneratorExit, shouldThrow)

    def testErrorsInBool(self):
        # [FUNCTION DOC] The truth value of the exit result is evaluated: a result converting to
        # true swallows the error, to false lets it propagate, and a conversion that raises
        # propagates that error.
        # issue4589: __exit__ return code may raise an exception
        # when looking at its truth value.

        class cm(object):
            # [CLASS DOC] HELPER: manager whose exit returns an object with a configurable
            # truth conversion.
            def __init__(self, bool_conversion):
                # [FUNCTION DOC] HELPER: builds an exit result object whose truth conversion
                # calls the given function.
                class Bool:
                    # [CLASS DOC] HELPER: object whose truth conversion calls an
                    # enclosing function.
                    def __bool__(self):
                        # [FUNCTION DOC] HELPER: returns the result of calling the enclosing
                        # conversion function.
                        return bool_conversion()
                self.exit_result = Bool()
            def __enter__(self):
                # [FUNCTION DOC] HELPER: returns 3.
                return 3
            def __exit__(self, a, b, c):
                # [FUNCTION DOC] HELPER: returns the prepared exit result object.
                return self.exit_result

        def trueAsBool():
            # [FUNCTION DOC] HELPER: runs a with statement whose exit result is truthy, so the
            # failing statement in the body is swallowed.
            with cm(lambda: True):
                self.fail("Should NOT see this")
        trueAsBool()

        def falseAsBool():
            # [FUNCTION DOC] HELPER: runs a with statement whose exit result is falsy, so the
            # failure raised in the body propagates.
            with cm(lambda: False):
                self.fail("Should raise")
        self.assertRaises(AssertionError, falseAsBool)

        def failAsBool():
            # [FUNCTION DOC] HELPER: runs a with statement whose exit result conversion
            # raises ZeroDivisionError.
            with cm(lambda: 1//0):
                self.fail("Should NOT see this")
        self.assertRaises(ZeroDivisionError, failAsBool)


class NonLocalFlowControlTestCase(__TestCase):
    # [CLASS DOC] Tests that break, continue, return, yield and raise out of a with block run
    # the manager cleanup.

    def testWithBreak(self):
        # [FUNCTION DOC] Breaking out of a loop from inside a with block exits the block;
        # counter shows later code was skipped.
        counter = 0
        while True:
            counter += 1
            with mock_contextmanager_generator():
                counter += 10
                break
            counter += 100 # Not reached
        self.assertEqual(counter, 11)

    def testWithContinue(self):
        # [FUNCTION DOC] Continuing a loop from inside a with block works and code after the
        # block is skipped.
        counter = 0
        while True:
            counter += 1
            if counter > 2:
                break
            with mock_contextmanager_generator():
                counter += 10
                continue
            counter += 100 # Not reached
        self.assertEqual(counter, 12)

    def testWithReturn(self):
        # [FUNCTION DOC] Returning from inside a with block inside a loop returns the
        # counter value.
        def foo():
            # [FUNCTION DOC] HELPER: loops once, entering a manager and returning the counter
            # from inside it.
            counter = 0
            while True:
                counter += 1
                with mock_contextmanager_generator():
                    counter += 10
                    return counter
                counter += 100 # Not reached
        self.assertEqual(foo(), 11)

    def testWithYield(self):
        # [FUNCTION DOC] Yielding from inside a with block of a generator produces both values.
        def gen():
            # [FUNCTION DOC] HELPER: generator yielding twice inside a with block.
            with mock_contextmanager_generator():
                yield 12
                yield 13
        x = list(gen())
        self.assertEqual(x, [12, 13])

    def testWithRaise(self):
        # [FUNCTION DOC] Raising inside a with block propagates the error and skips later code.
        counter = 0
        try:
            counter += 1
            with mock_contextmanager_generator():
                counter += 10
                raise RuntimeError
            counter += 100 # Not reached
        except RuntimeError:
            self.assertEqual(counter, 11)
        else:
            self.fail("Didn't raise RuntimeError")


class AssignmentTargetTestCase(__TestCase):
    # [CLASS DOC] Tests of the variety of assignment targets allowed after as in a
    # with statement.

    def testSingleComplexTarget(self):
        # [FUNCTION DOC] Targets that are subscripts of dicts and lists, a computed subscript, a
        # new dict key and an attribute on an instance are all assigned.
        targets = {1: [0, 1, 2]}
        with mock_contextmanager_generator() as targets[1][0]:
            self.assertEqual(list(targets.keys()), [1])
            self.assertEqual(targets[1][0].__class__, MockResource)
        with mock_contextmanager_generator() as list(targets.values())[0][1]:
            self.assertEqual(list(targets.keys()), [1])
            self.assertEqual(targets[1][1].__class__, MockResource)
        with mock_contextmanager_generator() as targets[2]:
            keys = list(targets.keys())
            keys.sort()
            self.assertEqual(keys, [1, 2])
        # [CLASS DOC] HELPER: empty class defined on one line.
        class C: pass
        blah = C()
        with mock_contextmanager_generator() as blah.foo:
            self.assertEqual(hasattr(blah, "foo"), True)

    def testMultipleComplexTargets(self):
        # [FUNCTION DOC] Tuple unpacking of the entry result into subscript and attribute
        # targets assigns each element.
        class C:
            # [CLASS DOC] HELPER: manager whose enter returns the tuple of one, two and three
            # and whose exit does nothing.
            # [FUNCTION DOC] HELPER: returns the tuple one, two, three.
            def __enter__(self): return 1, 2, 3
            # [FUNCTION DOC] HELPER: does nothing.
            def __exit__(self, t, v, tb): pass
        targets = {1: [0, 1, 2]}
        with C() as (targets[1][0], targets[1][1], targets[1][2]):
            self.assertEqual(targets, {1: [1, 2, 3]})
        with C() as (list(targets.values())[0][2], list(targets.values())[0][1], list(targets.values())[0][0]):
            self.assertEqual(targets, {1: [3, 2, 1]})
        with C() as (targets[1], targets[2], targets[3]):
            self.assertEqual(targets, {1: 1, 2: 2, 3: 3})
        # [CLASS DOC] HELPER: empty class defined on one line.
        class B: pass
        blah = B()
        with C() as (blah.one, blah.two, blah.three):
            self.assertEqual(blah.one, 1)
            self.assertEqual(blah.two, 2)
            self.assertEqual(blah.three, 3)

    def testWithExtendedTargets(self):
        # [FUNCTION DOC] An unpacking target with a starred name gets the first, middle and last
        # parts of a range.
        with nullcontext(range(1, 5)) as (a, *b, c):
            self.assertEqual(a, 1)
            self.assertEqual(b, [2, 3])
            self.assertEqual(c, 4)


class ExitSwallowsExceptionTestCase(__TestCase):
    # [CLASS DOC] Tests of whether an exit returning true or false swallows the exception raised
    # in the body.

    def testExitTrueSwallowsException(self):
        # [FUNCTION DOC] An exit returning true swallows a zero-division error from the body.
        class AfricanSwallow:
            # [CLASS DOC] HELPER: manager whose exit returns true, swallowing exceptions.
            # [FUNCTION DOC] HELPER: does nothing.
            def __enter__(self): pass
            # [FUNCTION DOC] HELPER: returns true.
            def __exit__(self, t, v, tb): return True
        try:
            with AfricanSwallow():
                1/0
        except ZeroDivisionError:
            self.fail("ZeroDivisionError should have been swallowed")

    def testExitFalseDoesntSwallowException(self):
        # [FUNCTION DOC] An exit returning false lets a zero-division error propagate.
        class EuropeanSwallow:
            # [CLASS DOC] HELPER: manager whose exit returns false, not swallowing exceptions.
            # [FUNCTION DOC] HELPER: does nothing.
            def __enter__(self): pass
            # [FUNCTION DOC] HELPER: returns false.
            def __exit__(self, t, v, tb): return False
        try:
            with EuropeanSwallow():
                1/0
        except ZeroDivisionError:
            pass
        else:
            self.fail("ZeroDivisionError should have been raised")


class NestedWith(__TestCase):
    # [CLASS DOC] Tests of with statements having several items, using small helper manager
    # classes defined as class attributes.

    class Dummy(object):
        # [CLASS DOC] HELPER: manager recording whether it entered and exited and the exit info;
        # can return a configured value on entry and swallow exceptions.
        def __init__(self, value=None, gobble=False):
            # [FUNCTION DOC] HELPER: stores the value to return on entry (the object itself when
            # none), the swallow flag, and clears the entered and exited flags.
            if value is None:
                value = self
            self.value = value
            self.gobble = gobble
            self.enter_called = False
            self.exit_called = False

        def __enter__(self):
            # [FUNCTION DOC] HELPER: sets the entered flag and returns the configured value.
            self.enter_called = True
            return self.value

        def __exit__(self, *exc_info):
            # [FUNCTION DOC] HELPER: sets the exited flag, stores the exception info and returns
            # true if configured to swallow the exception.
            self.exit_called = True
            self.exc_info = exc_info
            if self.gobble:
                return True

    class InitRaises(object):
        # [CLASS DOC] HELPER: class whose constructor raises RuntimeError.
        # [FUNCTION DOC] HELPER: raises RuntimeError.
        def __init__(self): raise RuntimeError()

    class EnterRaises(object):
        # [CLASS DOC] HELPER: manager whose enter raises RuntimeError.
        # [FUNCTION DOC] HELPER: raises RuntimeError.
        def __enter__(self): raise RuntimeError()
        # [FUNCTION DOC] HELPER: does nothing.
        def __exit__(self, *exc_info): pass

    class ExitRaises(object):
        # [CLASS DOC] HELPER: manager whose exit raises RuntimeError.
        # [FUNCTION DOC] HELPER: does nothing.
        def __enter__(self): pass
        # [FUNCTION DOC] HELPER: raises RuntimeError.
        def __exit__(self, *exc_info): raise RuntimeError()

    def testNoExceptions(self):
        # [FUNCTION DOC] Two managers in one statement are both entered and then both exited.
        with self.Dummy() as a, self.Dummy() as b:
            self.assertTrue(a.enter_called)
            self.assertTrue(b.enter_called)
        self.assertTrue(a.exit_called)
        self.assertTrue(b.exit_called)

    def testExceptionInExprList(self):
        # [FUNCTION DOC] If the second item's constructor raises, the first manager was entered
        # and still exited.
        try:
            with self.Dummy() as a, self.InitRaises():
                pass
        except:
            pass
        self.assertTrue(a.enter_called)
        self.assertTrue(a.exit_called)

    def testExceptionInEnter(self):
        # [FUNCTION DOC] If the second item's enter raises, the body does not run, the error
        # propagates and the first manager exits.
        try:
            with self.Dummy() as a, self.EnterRaises():
                self.fail('body of bad with executed')
        except RuntimeError:
            pass
        else:
            self.fail('RuntimeError not reraised')
        self.assertTrue(a.enter_called)
        self.assertTrue(a.exit_called)

    def testExceptionInExit(self):
        # [FUNCTION DOC] If the second manager's exit raises while the first swallows, the body
        # ran and the first manager saw the exception.
        body_executed = False
        with self.Dummy(gobble=True) as a, self.ExitRaises():
            body_executed = True
        self.assertTrue(a.enter_called)
        self.assertTrue(a.exit_called)
        self.assertTrue(body_executed)
        self.assertNotEqual(a.exc_info[0], None)

    def testEnterReturnsTuple(self):
        # [FUNCTION DOC] Tuple unpacking works for each item of a multi-item statement.
        with self.Dummy(value=(1,2)) as (a1, a2), \
             self.Dummy(value=(10, 20)) as (b1, b2):
            self.assertEqual(1, a1)
            self.assertEqual(2, a2)
            self.assertEqual(10, b1)
            self.assertEqual(20, b2)

    def testExceptionLocation(self):
        # [FUNCTION DOC] For each of three nested helper functions that raise from constructor,
        # enter or exit, the traceback entry points at exactly the line and column span of the
        # failing manager expression. The test relies on exact line offsets within each
        # function, so the docs for those helpers sit above the functions.
        # The location of an exception raised from
        # __init__, __enter__ or __exit__ of a context
        # manager should be just the context manager expression,
        # pinpointing the precise context manager in case there
        # is more than one.

        # [FUNCTION DOC] HELPER: returns the exception raised when the second item's constructor
        # fails inside a multi-item with.
        def init_raises():
            try:
                with self.Dummy(), self.InitRaises() as cm, self.Dummy() as d:
                    pass
            except Exception as e:
                return e

        # [FUNCTION DOC] HELPER: returns the exception raised when the first item's enter fails
        # inside a multi-item with.
        def enter_raises():
            try:
                with self.EnterRaises(), self.Dummy() as d:
                    pass
            except Exception as e:
                return e

        # [FUNCTION DOC] HELPER: returns the exception raised when the first item's exit fails
        # inside a multi-item with.
        def exit_raises():
            try:
                with self.ExitRaises(), self.Dummy() as d:
                    pass
            except Exception as e:
                return e

        for func, expected in [(init_raises, "self.InitRaises()"),
                               (enter_raises, "self.EnterRaises()"),
                               (exit_raises, "self.ExitRaises()"),
                              ]:
            with self.subTest(func):
                exc = func()
                f = traceback.extract_tb(exc.__traceback__)[0]
                indent = 16
                co = func.__code__
                self.assertEqual(f.lineno, co.co_firstlineno + 2)
                self.assertEqual(f.end_lineno, co.co_firstlineno + 2)
                self.assertEqual(f.line[f.colno - indent : f.end_colno - indent],
                                 expected)


if __name__ == '__main__':
    run_tests()
