# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_contextlib.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests, slowTest

__TestCase = CPythonTestCase


# redirect import statements
import sys
import importlib.abc

redirect_imports = (
    "test.mapping_tests",
    "test.typinganndata",
    "test.test_grammar",
    "test.test_math",
    "test.test_iter",
    "test.typinganndata.ann_module",
)

class RedirectImportFinder(importlib.abc.MetaPathFinder):
    # [CLASS DOC] HELPER: Import-system finder that redirects imports of names in
    # redirect_imports to the same-named top-level modules. Instance is inserted into the meta
    # path at import time.
    def find_spec(self, fullname, path, target=None):
        # [FUNCTION DOC] HELPER: If the module name is in the redirect list, imports the module
        # without the "test." prefix, registers it in the module table under the requested name
        # and returns its spec; on an import failure, or for other names, returns None.
        # Check if the import is the problematic one
        if fullname in redirect_imports:
            try:
                # Attempt to import the standalone module
                name = fullname.removeprefix("test.")
                r = importlib.import_module(name)
                # Redirect the module in sys.modules
                sys.modules[fullname] = r
                # Return a module spec from the found module
                return importlib.util.find_spec(name)
            except ImportError:
                return None
        return None

# Add the custom finder to sys.meta_path
sys.meta_path.insert(0, RedirectImportFinder())


# ======= END DYNAMO PATCH =======

# [MODULE DOC] Dynamo-patched copy of the CPython 3.13 test_contextlib tests for the contextlib
# module: abstract context managers, the generator-based context manager decorator, closing,
# nullcontext, ContextDecorator, ExitStack, stream redirection, suppress and chdir. At import
# time it installs a meta path finder at the front of the import system that redirects some
# "test." module imports to top-level modules, and it star-imports contextlib. Some tests are
# marked to allow graph breaks or are skipped. Running as a script calls run_tests.

"""Unit tests for contextlib.py, and other context managers."""

import io
import os
import sys
import tempfile
import threading
import traceback
import unittest
from contextlib import *  # Tests __all__
from test import support
from test.support import os_helper
from test.support.testcase import ExceptionIsLikeMixin
import weakref


class TestAbstractContextManager(__TestCase):
    # [CLASS DOC] Tests for AbstractContextManager: default entry, slots, abstract exit,
    # structural subclassing.

    def test_enter(self):
        # [FUNCTION DOC] Checks that the default enter hook returns the manager itself.
        class DefaultEnter(AbstractContextManager):
            # [CLASS DOC] HELPER: Context manager subclass that only defines exit by delegating
            # to the base class.
            def __exit__(self, *args):
                # [FUNCTION DOC] HELPER: Delegates to the base class exit hook.
                super().__exit__(*args)

        manager = DefaultEnter()
        self.assertIs(manager.__enter__(), manager)

    def test_slots(self):
        # [FUNCTION DOC] Checks that a subclass declaring empty slots rejects new attributes
        # with an AttributeError.
        class DefaultContextManager(AbstractContextManager):
            # [CLASS DOC] HELPER: Context manager subclass with empty slots.
            __slots__ = ()

            def __exit__(self, *args):
                # [FUNCTION DOC] HELPER: Delegates to the base class exit hook.
                super().__exit__(*args)

        with self.assertRaises(AttributeError):
            DefaultContextManager().var = 42

    def test_exit_is_abstract(self):
        # [FUNCTION DOC] Checks that a subclass lacking the exit method cannot be
        # instantiated (TypeError).
        class MissingExit(AbstractContextManager):
            # [CLASS DOC] HELPER: Subclass that does not define the abstract exit method.
            pass

        with self.assertRaises(TypeError):
            MissingExit()

    def test_structural_subclassing(self):
        # [FUNCTION DOC] Checks structural subclass detection: a class with enter and exit
        # counts as a subclass, while setting either hook to None removes that status.
        class ManagerFromScratch:
            # [CLASS DOC] HELPER: Standalone class implementing both context manager hooks
            # without inheritance.
            def __enter__(self):
                # [FUNCTION DOC] HELPER: Returns the object itself.
                return self
            def __exit__(self, exc_type, exc_value, traceback):
                # [FUNCTION DOC] HELPER: Returns None, so no exception is suppressed.
                return None

        self.assertTrue(issubclass(ManagerFromScratch, AbstractContextManager))

        class DefaultEnter(AbstractContextManager):
            # [CLASS DOC] HELPER: Subclass defining only exit by delegating to the base class.
            def __exit__(self, *args):
                # [FUNCTION DOC] HELPER: Delegates to the base class exit hook.
                super().__exit__(*args)

        self.assertTrue(issubclass(DefaultEnter, AbstractContextManager))

        class NoEnter(ManagerFromScratch):
            # [CLASS DOC] HELPER: Subclass whose enter hook is disabled by setting it to None.
            __enter__ = None

        self.assertFalse(issubclass(NoEnter, AbstractContextManager))

        class NoExit(ManagerFromScratch):
            # [CLASS DOC] HELPER: Subclass whose exit hook is disabled by setting it to None.
            __exit__ = None

        self.assertFalse(issubclass(NoExit, AbstractContextManager))


class ContextManagerTestCase(__TestCase):
    # [CLASS DOC] Tests for the generator-based context manager decorator.

    def test_contextmanager_plain(self):
        # [FUNCTION DOC] Checks that code before and after the yield runs around the body and
        # the value is bound.
        state = []
        @contextmanager
        def woohoo():
            # [FUNCTION DOC] HELPER: Generator context manager recording 1, yielding 42, then
            # recording 999.
            state.append(1)
            yield 42
            state.append(999)
        with woohoo() as x:
            self.assertEqual(state, [1])
            self.assertEqual(x, 42)
            state.append(x)
        self.assertEqual(state, [1, 42, 999])

    def test_contextmanager_finally(self):
        # [FUNCTION DOC] Checks that a try-finally generator runs its cleanup when the body
        # raises, and re-raises.
        state = []
        @contextmanager
        def woohoo():
            # [FUNCTION DOC] HELPER: Generator context manager with a cleanup block that
            # records 999.
            state.append(1)
            try:
                yield 42
            finally:
                state.append(999)
        with self.assertRaises(ZeroDivisionError):
            with woohoo() as x:
                self.assertEqual(state, [1])
                self.assertEqual(x, 42)
                state.append(x)
                raise ZeroDivisionError()
        self.assertEqual(state, [1, 42, 999])

    def test_contextmanager_traceback(self):
        # [FUNCTION DOC] Checks that the traceback of an exception raised in the body (including
        # RuntimeError and StopIteration subclasses) holds a single frame, the test itself.
        @contextmanager
        def f():
            # [FUNCTION DOC] HELPER: Generator context manager that yields once.
            yield

        try:
            with f():
                1/0
        except ZeroDivisionError as e:
            frames = traceback.extract_tb(e.__traceback__)

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].name, 'test_contextmanager_traceback')
        self.assertEqual(frames[0].line, '1/0')

        # Repeat with RuntimeError (which goes through a different code path)
        class RuntimeErrorSubclass(RuntimeError):
            # [CLASS DOC] HELPER: RuntimeError subclass used to test a special exception path.
            pass

        try:
            with f():
                raise RuntimeErrorSubclass(42)
        except RuntimeErrorSubclass as e:
            frames = traceback.extract_tb(e.__traceback__)

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].name, 'test_contextmanager_traceback')
        self.assertEqual(frames[0].line, 'raise RuntimeErrorSubclass(42)')

        class StopIterationSubclass(StopIteration):
            # [CLASS DOC] HELPER: StopIteration subclass used to test exception propagation.
            pass

        for stop_exc in (
            StopIteration('spam'),
            StopIterationSubclass('spam'),
        ):
            with self.subTest(type=type(stop_exc)):
                try:
                    with f():
                        raise stop_exc
                except type(stop_exc) as e:
                    self.assertIs(e, stop_exc)
                    frames = traceback.extract_tb(e.__traceback__)
                else:
                    self.fail(f'{stop_exc} was suppressed')

                self.assertEqual(len(frames), 1)
                self.assertEqual(frames[0].name, 'test_contextmanager_traceback')
                self.assertEqual(frames[0].line, 'raise stop_exc')

    def test_contextmanager_no_reraise(self):
        # [FUNCTION DOC] Checks that exiting with an exception for a generator that finishes
        # normally returns false.
        @contextmanager
        def whee():
            # [FUNCTION DOC] HELPER: Generator context manager that yields once.
            yield
        ctx = whee()
        ctx.__enter__()
        # Calling __exit__ should not result in an exception
        self.assertFalse(ctx.__exit__(TypeError, TypeError("foo"), None))

    def test_contextmanager_trap_yield_after_throw(self):
        # [FUNCTION DOC] Checks that a generator which yields again after an exception is thrown
        # in raises RuntimeError.
        @contextmanager
        def whoo():
            # [FUNCTION DOC] HELPER: Generator that yields again in its exception handler.
            try:
                yield
            except:
                yield
        ctx = whoo()
        ctx.__enter__()
        with self.assertRaises(RuntimeError):
            ctx.__exit__(TypeError, TypeError("foo"), None)
        # if support.check_impl_detail(cpython=True):
        #     # The "gen" attribute is an implementation detail.
        #     self.assertFalse(ctx.gen.gi_suspended)

    def test_contextmanager_trap_no_yield(self):
        # [FUNCTION DOC] Checks that a generator that never yields makes entry raise
        # a RuntimeError.
        @contextmanager
        def whoo():
            # [FUNCTION DOC] HELPER: Generator function whose yield is unreachable.
            if False:
                yield
        ctx = whoo()
        with self.assertRaises(RuntimeError):
            ctx.__enter__()

    def test_contextmanager_trap_second_yield(self):
        # [FUNCTION DOC] Checks that a generator yielding twice makes a normal exit raise
        # a RuntimeError.
        @contextmanager
        def whoo():
            # [FUNCTION DOC] HELPER: Generator that yields twice.
            yield
            yield
        ctx = whoo()
        ctx.__enter__()
        with self.assertRaises(RuntimeError):
            ctx.__exit__(None, None, None)
        # if support.check_impl_detail(cpython=True):
        #     # The "gen" attribute is an implementation detail.
        #     self.assertFalse(ctx.gen.gi_suspended)

    def test_contextmanager_non_normalised(self):
        # [FUNCTION DOC] Checks that an exception different from the thrown one, raised in the
        # generator, propagates.
        @contextmanager
        def whoo():
            # [FUNCTION DOC] HELPER: Generator that converts a RuntimeError into a SyntaxError.
            try:
                yield
            except RuntimeError:
                raise SyntaxError

        ctx = whoo()
        ctx.__enter__()
        with self.assertRaises(SyntaxError):
            ctx.__exit__(RuntimeError, None, None)

    def test_contextmanager_except(self):
        # [FUNCTION DOC] Checks that a generator handler catching the body exception
        # suppresses it.
        state = []
        @contextmanager
        def woohoo():
            # [FUNCTION DOC] HELPER: Generator that catches ZeroDivisionError and records
            # its argument.
            state.append(1)
            try:
                yield 42
            except ZeroDivisionError as e:
                state.append(e.args[0])
                self.assertEqual(state, [1, 42, 999])
        with woohoo() as x:
            self.assertEqual(state, [1])
            self.assertEqual(x, 42)
            state.append(x)
            raise ZeroDivisionError(999)
        self.assertEqual(state, [1, 42, 999])

    def test_contextmanager_except_stopiter(self):
        # [FUNCTION DOC] Checks that StopIteration and its subclasses raised in the body
        # propagate unchanged.
        @contextmanager
        def woohoo():
            # [FUNCTION DOC] HELPER: Generator context manager that yields once.
            yield

        class StopIterationSubclass(StopIteration):
            # [CLASS DOC] HELPER: StopIteration subclass used to test propagation.
            pass

        for stop_exc in (StopIteration('spam'), StopIterationSubclass('spam')):
            with self.subTest(type=type(stop_exc)):
                try:
                    with woohoo():
                        raise stop_exc
                except Exception as ex:
                    self.assertIs(ex, stop_exc)
                else:
                    self.fail(f'{stop_exc} was suppressed')

    @torch._dynamo.error_on_graph_break(False)
    def test_contextmanager_except_pep479(self):
        # [FUNCTION DOC] Checks that a StopIteration raised in the body propagates for a
        # generator compiled with the stop-iteration-as-error behavior. Compiles the generator
        # source text at run time. Allowed to graph break under Dynamo.
        code = """\
from __future__ import generator_stop
from contextlib import contextmanager
@contextmanager
def woohoo():
    yield
"""
        locals = {}
        exec(code, locals, locals)
        woohoo = locals['woohoo']

        stop_exc = StopIteration('spam')
        try:
            with woohoo():
                raise stop_exc
        except Exception as ex:
            self.assertIs(ex, stop_exc)
        else:
            self.fail('StopIteration was suppressed')

    def test_contextmanager_do_not_unchain_non_stopiteration_exceptions(self):
        # [FUNCTION DOC] Checks that other exceptions re-raised in a handler keep their cause
        # chain, while StopIteration is passed through without gaining a cause.
        @contextmanager
        def test_issue29692():
            # [FUNCTION DOC] HELPER: Generator that wraps any exception in a RuntimeError
            # chained to it.
            try:
                yield
            except Exception as exc:
                raise RuntimeError('issue29692:Chained') from exc
        try:
            with test_issue29692():
                raise ZeroDivisionError
        except Exception as ex:
            self.assertIs(type(ex), RuntimeError)
            self.assertEqual(ex.args[0], 'issue29692:Chained')
            self.assertIsInstance(ex.__cause__, ZeroDivisionError)

        try:
            with test_issue29692():
                raise StopIteration('issue29692:Unchained')
        except Exception as ex:
            self.assertIs(type(ex), StopIteration)
            self.assertEqual(ex.args[0], 'issue29692:Unchained')
            self.assertIsNone(ex.__cause__)

    def test_contextmanager_wrap_runtimeerror(self):
        # [FUNCTION DOC] Checks that a RuntimeError that wraps a body exception propagates, and
        # that a wrapped StopIteration is unwrapped to a StopIteration.
        @contextmanager
        def woohoo():
            # [FUNCTION DOC] HELPER: Generator that re-raises a body exception wrapped in
            # a RuntimeError.
            try:
                yield
            except Exception as exc:
                raise RuntimeError(f'caught {exc}') from exc

        with self.assertRaises(RuntimeError):
            with woohoo():
                1 / 0

        # If the context manager wrapped StopIteration in a RuntimeError,
        # we also unwrap it, because we can't tell whether the wrapping was
        # done by the generator machinery or by the generator itself.
        with self.assertRaises(StopIteration):
            with woohoo():
                raise StopIteration

    def _create_contextmanager_attribs(self):
        # [FUNCTION DOC] HELPER: Builds a context manager function whose wrapped function has a
        # custom attribute and a docstring.
        def attribs(**kw):
            # [FUNCTION DOC] HELPER: Decorator factory that sets the given attributes on
            # a function.
            def decorate(func):
                # [FUNCTION DOC] HELPER: Sets the captured attributes on the function and
                # returns it.
                for k,v in kw.items():
                    setattr(func,k,v)
                return func
            return decorate
        @contextmanager
        @attribs(foo='bar')
        def baz(spam):
            # [FUNCTION DOC] HELPER: Documented generator function used as the context
            # manager source.
            """Whee!"""
            yield
        return baz

    def test_contextmanager_attribs(self):
        # [FUNCTION DOC] Checks that the decorated function keeps its name and custom attribute.
        baz = self._create_contextmanager_attribs()
        self.assertEqual(baz.__name__,'baz')
        self.assertEqual(baz.foo, 'bar')

    @support.requires_docstrings
    def test_contextmanager_doc_attrib(self):
        # [FUNCTION DOC] Checks that the decorated function keeps its docstring. Requires
        # docstrings enabled.
        baz = self._create_contextmanager_attribs()
        self.assertEqual(baz.__doc__, "Whee!")

    @support.requires_docstrings
    def test_instance_docstring_given_cm_docstring(self):
        # [FUNCTION DOC] Checks that a manager instance gets the docstring of the
        # generator function.
        baz = self._create_contextmanager_attribs()(None)
        self.assertEqual(baz.__doc__, "Whee!")

    def test_keywords(self):
        # [FUNCTION DOC] Checks that parameters named self, func, args and kwds can be passed
        # as keywords.
        # Ensure no keyword arguments are inhibited
        @contextmanager
        def woohoo(self, func, args, kwds):
            # [FUNCTION DOC] HELPER: Generator context manager that yields its arguments as
            # a tuple.
            yield (self, func, args, kwds)
        with woohoo(self=11, func=22, args=33, kwds=44) as target:
            self.assertEqual(target, (11, 22, 33, 44))

    def test_nokeepref(self):
        # [FUNCTION DOC] Checks that the generator context manager does not keep references to
        # its arguments.
        class A:
            # [CLASS DOC] HELPER: Empty class whose instances are weakly referenced.
            pass

        @contextmanager
        def woohoo(a, b):
            # [FUNCTION DOC] HELPER: Generator that checks both arguments are no longer alive
            # after being weakly referenced.
            a = weakref.ref(a)
            b = weakref.ref(b)
            # Allow test to work with a non-refcounted GC
            support.gc_collect()
            self.assertIsNone(a())
            self.assertIsNone(b())
            yield

        with woohoo(A(), b=A()):
            pass

    def test_param_errors(self):
        # [FUNCTION DOC] Checks that wrong arguments to a decorated function raise a TypeError
        # when called.
        @contextmanager
        def woohoo(a, *, b):
            # [FUNCTION DOC] HELPER: Generator context manager taking a positional and a
            # keyword-only argument.
            yield

        with self.assertRaises(TypeError):
            woohoo()
        with self.assertRaises(TypeError):
            woohoo(3, 5)
        with self.assertRaises(TypeError):
            woohoo(b=3)

    def test_recursive(self):
        # [FUNCTION DOC] Checks that a context manager used as a function decorator can be
        # applied recursively.
        depth = 0
        ncols = 0
        @contextmanager
        def woohoo():
            # [FUNCTION DOC] HELPER: Generator tracking nesting depth and the number of entries
            # in enclosing variables.
            nonlocal ncols
            ncols += 1
            nonlocal depth
            before = depth
            depth += 1
            yield
            depth -= 1
            self.assertEqual(depth, before)

        @woohoo()
        def recursive():
            # [FUNCTION DOC] HELPER: Function decorated with the context manager that calls
            # itself up to depth ten.
            if depth < 10:
                recursive()

        recursive()
        self.assertEqual(ncols, 10)
        self.assertEqual(depth, 0)


class ClosingTestCase(__TestCase):
    # [CLASS DOC] Tests for the closing context manager.

    @support.requires_docstrings
    def test_instance_docs(self):
        # [FUNCTION DOC] Checks that closing instances share the class docstring. Requires
        # docstrings enabled.
        # Issue 19330: ensure context manager instances have good docstrings
        cm_docstring = closing.__doc__
        obj = closing(None)
        self.assertEqual(obj.__doc__, cm_docstring)

    def test_closing(self):
        # [FUNCTION DOC] Checks that the close method of the object is called on exit and the
        # object is bound.
        state = []
        class C:
            # [CLASS DOC] HELPER: Class whose close method records a call.
            def close(self):
                # [FUNCTION DOC] HELPER: Records that close was called.
                state.append(1)
        x = C()
        self.assertEqual(state, [])
        with closing(x) as y:
            self.assertEqual(x, y)
        self.assertEqual(state, [1])

    def test_closing_error(self):
        # [FUNCTION DOC] Checks that close is called even when the body raises, and the
        # error propagates.
        state = []
        class C:
            # [CLASS DOC] HELPER: Class whose close method records a call.
            def close(self):
                # [FUNCTION DOC] HELPER: Records that close was called.
                state.append(1)
        x = C()
        self.assertEqual(state, [])
        with self.assertRaises(ZeroDivisionError):
            with closing(x) as y:
                self.assertEqual(x, y)
                1 / 0
        self.assertEqual(state, [1])


class NullcontextTestCase(__TestCase):
    # [CLASS DOC] Tests for nullcontext.
    def test_nullcontext(self):
        # [FUNCTION DOC] Checks that nullcontext returns its argument on entry.
        class C:
            # [CLASS DOC] HELPER: Empty class used as the context value.
            pass
        c = C()
        with nullcontext(c) as c_in:
            self.assertIs(c_in, c)


class FileContextTestCase(__TestCase):
    # [CLASS DOC] Tests that file objects work as context managers.

    @torch._dynamo.error_on_graph_break(False)
    def testWithOpen(self):
        # [FUNCTION DOC] Checks that a file opened for writing then reading is closed after the
        # block, also when the body raises. Uses a temporary file removed at the end. Allowed to
        # graph break.
        tfn = tempfile.mktemp()
        try:
            with open(tfn, "w", encoding="utf-8") as f:
                self.assertFalse(f.closed)
                f.write("Booh\n")
            self.assertTrue(f.closed)
            with self.assertRaises(ZeroDivisionError):
                with open(tfn, "r", encoding="utf-8") as f:
                    self.assertFalse(f.closed)
                    self.assertEqual(f.read(), "Booh\n")
                    1 / 0
            self.assertTrue(f.closed)
        finally:
            os_helper.unlink(tfn)

class LockContextTestCase(__TestCase):
    # [CLASS DOC] Tests that lock-like objects work as context managers, through a
    # shared helper.

    def boilerPlate(self, lock, locked):
        # [FUNCTION DOC] HELPER: Checks that the given lock is held inside the block and
        # released after it, also when the body raises; takes a function that reports whether
        # the lock is held.
        self.assertFalse(locked())
        with lock:
            self.assertTrue(locked())
        self.assertFalse(locked())
        with self.assertRaises(ZeroDivisionError):
            with lock:
                self.assertTrue(locked())
                1 / 0
        self.assertFalse(locked())

    def testWithLock(self):
        # [FUNCTION DOC] Checks a plain lock used as a context manager.
        lock = threading.Lock()
        self.boilerPlate(lock, lock.locked)

    def testWithRLock(self):
        # [FUNCTION DOC] Checks a reentrant lock used as a context manager.
        lock = threading.RLock()
        self.boilerPlate(lock, lock._is_owned)

    def testWithCondition(self):
        # [FUNCTION DOC] Checks a condition object used as a context manager.
        lock = threading.Condition()
        def locked():
            # [FUNCTION DOC] HELPER: Reports whether the condition lock is owned.
            return lock._is_owned()
        self.boilerPlate(lock, locked)

    def testWithSemaphore(self):
        # [FUNCTION DOC] Checks a semaphore used as a context manager.
        lock = threading.Semaphore()
        def locked():
            # [FUNCTION DOC] HELPER: Reports lock state by trying a non-blocking acquire and
            # releasing if it succeeds.
            if lock.acquire(False):
                lock.release()
                return False
            else:
                return True
        self.boilerPlate(lock, locked)

    def testWithBoundedSemaphore(self):
        # [FUNCTION DOC] Checks a bounded semaphore used as a context manager.
        lock = threading.BoundedSemaphore()
        def locked():
            # [FUNCTION DOC] HELPER: Reports lock state by trying a non-blocking acquire and
            # releasing if it succeeds.
            if lock.acquire(False):
                lock.release()
                return False
            else:
                return True
        self.boilerPlate(lock, locked)


class mycontext(ContextDecorator):
    # [CLASS DOC] HELPER: Context manager that is also a function decorator. State: started
    # flag, saved exit details, and a catch flag that decides whether exceptions are suppressed.
    """Example decoration-compatible context manager for testing"""
    started = False
    exc = None
    catch = False

    def __enter__(self):
        # [FUNCTION DOC] HELPER: Marks the manager as started and returns itself.
        self.started = True
        return self

    def __exit__(self, *exc):
        # [FUNCTION DOC] HELPER: Saves the exception details and returns the catch flag.
        self.exc = exc
        return self.catch


class TestContextDecorator(__TestCase):
    # [CLASS DOC] Tests for ContextDecorator.

    @support.requires_docstrings
    def test_instance_docs(self):
        # [FUNCTION DOC] Checks that instances share the class docstring. Requires
        # docstrings enabled.
        # Issue 19330: ensure context manager instances have good docstrings
        cm_docstring = mycontext.__doc__
        obj = mycontext()
        self.assertEqual(obj.__doc__, cm_docstring)

    def test_contextdecorator(self):
        # [FUNCTION DOC] Checks use as a plain context manager: entry starts it and exit sees
        # no exception.
        context = mycontext()
        with context as result:
            self.assertIs(result, context)
            self.assertTrue(context.started)

        self.assertEqual(context.exc, (None, None, None))


    def test_contextdecorator_with_exception(self):
        # [FUNCTION DOC] Checks that exit sees the raised exception, and that setting catch
        # suppresses it.
        context = mycontext()

        with self.assertRaisesRegex(NameError, 'foo'):
            with context:
                raise NameError('foo')
        self.assertIsNotNone(context.exc)
        self.assertIs(context.exc[0], NameError)

        context = mycontext()
        context.catch = True
        with context:
            raise NameError('foo')
        self.assertIsNotNone(context.exc)
        self.assertIs(context.exc[0], NameError)


    def test_decorator(self):
        # [FUNCTION DOC] Checks use as a function decorator for a normal call.
        context = mycontext()

        @context
        def test():
            # [FUNCTION DOC] HELPER: Function checking the manager started and recorded
            # nothing yet.
            self.assertIsNone(context.exc)
            self.assertTrue(context.started)
        test()
        self.assertEqual(context.exc, (None, None, None))


    def test_decorator_with_exception(self):
        # [FUNCTION DOC] Checks use as a decorator when the decorated function raises.
        context = mycontext()

        @context
        def test():
            # [FUNCTION DOC] HELPER: Function that checks state then raises a NameError.
            self.assertIsNone(context.exc)
            self.assertTrue(context.started)
            raise NameError('foo')

        with self.assertRaisesRegex(NameError, 'foo'):
            test()
        self.assertIsNotNone(context.exc)
        self.assertIs(context.exc[0], NameError)


    def test_decorating_method(self):
        # [FUNCTION DOC] Checks decorating a method, including positional, keyword and
        # default arguments.
        context = mycontext()

        class Test(object):
            # [CLASS DOC] HELPER: Class with a decorated method that stores its arguments.

            @context
            def method(self, a, b, c=None):
                # [FUNCTION DOC] HELPER: Stores the three arguments as attributes.
                self.a = a
                self.b = b
                self.c = c

        # these tests are for argument passing when used as a decorator
        test = Test()
        test.method(1, 2)
        self.assertEqual(test.a, 1)
        self.assertEqual(test.b, 2)
        self.assertEqual(test.c, None)

        test = Test()
        test.method('a', 'b', 'c')
        self.assertEqual(test.a, 'a')
        self.assertEqual(test.b, 'b')
        self.assertEqual(test.c, 'c')

        test = Test()
        test.method(a=1, b=2)
        self.assertEqual(test.a, 1)
        self.assertEqual(test.b, 2)


    def test_typo_enter(self):
        # [FUNCTION DOC] Checks that a misspelled enter hook raises a TypeError mentioning the
        # context manager.
        class mycontext(ContextDecorator):
            # [CLASS DOC] HELPER: ContextDecorator subclass with a misspelled enter hook.
            def __unter__(self):
                # [FUNCTION DOC] HELPER: Misspelled enter hook; does nothing.
                pass
            def __exit__(self, *exc):
                # [FUNCTION DOC] HELPER: Exit hook that does nothing.
                pass

        with self.assertRaisesRegex(TypeError, 'the context manager'):
            with mycontext():
                pass


    def test_typo_exit(self):
        # [FUNCTION DOC] Checks that a misspelled exit hook raises a TypeError mentioning the
        # exit method.
        class mycontext(ContextDecorator):
            # [CLASS DOC] HELPER: ContextDecorator subclass with a misspelled exit hook.
            def __enter__(self):
                # [FUNCTION DOC] HELPER: Enter hook that does nothing.
                pass
            def __uxit__(self, *exc):
                # [FUNCTION DOC] HELPER: Misspelled exit hook; does nothing.
                pass

        with self.assertRaisesRegex(TypeError, 'the context manager.*__exit__'):
            with mycontext():
                pass


    def test_contextdecorator_as_mixin(self):
        # [FUNCTION DOC] Checks that ContextDecorator can be mixed into a plain context
        # manager class.
        class somecontext(object):
            # [CLASS DOC] HELPER: Plain context manager class that records start and
            # exit details.
            started = False
            exc = None

            def __enter__(self):
                # [FUNCTION DOC] HELPER: Marks the manager as started and returns itself.
                self.started = True
                return self

            def __exit__(self, *exc):
                # [FUNCTION DOC] HELPER: Saves the exception details.
                self.exc = exc

        class mycontext(somecontext, ContextDecorator):
            # [CLASS DOC] HELPER: Combination of the plain manager and ContextDecorator.
            pass

        context = mycontext()
        @context
        def test():
            # [FUNCTION DOC] HELPER: Function checking that the manager started and recorded
            # nothing yet.
            self.assertIsNone(context.exc)
            self.assertTrue(context.started)
        test()
        self.assertEqual(context.exc, (None, None, None))


    def test_contextmanager_as_decorator(self):
        # [FUNCTION DOC] Checks that a generator context manager can decorate a function and
        # be reused.
        @contextmanager
        def woohoo(y):
            # [FUNCTION DOC] HELPER: Generator context manager that records its argument before
            # and 999 after.
            state.append(y)
            yield
            state.append(999)

        state = []
        @woohoo(1)
        def test(x):
            # [FUNCTION DOC] HELPER: Function that records its argument in the shared
            # state list.
            self.assertEqual(state, [1])
            state.append(x)
        test('something')
        self.assertEqual(state, [1, 'something', 999])

        # Issue #11647: Ensure the decorated function is 'reusable'
        state = []
        test('something else')
        self.assertEqual(state, [1, 'something else', 999])


class _TestBaseExitStack:
    # [CLASS DOC] HELPER: Mixin with ExitStack-style tests, to be combined with a test case. The
    # attribute exit_stack must be set by the subclass to the stack class under test.
    exit_stack = None

    @support.requires_docstrings
    def test_instance_docs(self):
        # [FUNCTION DOC] Checks that stack instances share the class docstring. Requires
        # docstrings enabled.
        # Issue 19330: ensure context manager instances have good docstrings
        cm_docstring = self.exit_stack.__doc__
        obj = self.exit_stack()
        self.assertEqual(obj.__doc__, cm_docstring)

    def test_no_resources(self):
        # [FUNCTION DOC] Checks that an empty stack can be entered and exited.
        with self.exit_stack():
            pass

    def test_callback(self):
        # [FUNCTION DOC] Checks registering callbacks with positional and keyword arguments,
        # that the registered function is returned, wrappers keep the wrapped function but not
        # its name and docstring, and bad calls raise a TypeError.
        expected = [
            ((), {}),
            ((1,), {}),
            ((1,2), {}),
            ((), dict(example=1)),
            ((1,), dict(example=1)),
            ((1,2), dict(example=1)),
            ((1,2), dict(self=3, callback=4)),
        ]
        result = []
        def _exit(*args, **kwds):
            # [FUNCTION DOC] HELPER: Callback recording its arguments in the result list.
            """Test metadata propagation"""
            result.append((args, kwds))
        with self.exit_stack() as stack:
            for args, kwds in reversed(expected):
                if args and kwds:
                    f = stack.callback(_exit, *args, **kwds)
                elif args:
                    f = stack.callback(_exit, *args)
                elif kwds:
                    f = stack.callback(_exit, **kwds)
                else:
                    f = stack.callback(_exit)
                self.assertIs(f, _exit)
            for wrapper in stack._exit_callbacks:
                self.assertIs(wrapper[1].__wrapped__, _exit)
                self.assertNotEqual(wrapper[1].__name__, _exit.__name__)
                self.assertIsNone(wrapper[1].__doc__, _exit.__doc__)
        self.assertEqual(result, expected)

        result = []
        with self.exit_stack() as stack:
            with self.assertRaises(TypeError):
                stack.callback(arg=1)
            with self.assertRaises(TypeError):
                self.exit_stack.callback(arg=2)
            with self.assertRaises(TypeError):
                stack.callback(callback=_exit, arg=3)
        self.assertEqual(result, [])

    def test_push(self):
        # [FUNCTION DOC] Checks push with exit functions and with context manager objects: the
        # right exit callback is registered and exceptions seen match expectations.
        exc_raised = ZeroDivisionError
        def _expect_exc(exc_type, exc, exc_tb):
            # [FUNCTION DOC] HELPER: Exit callback asserting the exception type is the
            # expected one.
            self.assertIs(exc_type, exc_raised)
        def _suppress_exc(*exc_details):
            # [FUNCTION DOC] HELPER: Exit callback that always suppresses the exception.
            return True
        def _expect_ok(exc_type, exc, exc_tb):
            # [FUNCTION DOC] HELPER: Exit callback asserting that no exception was passed.
            self.assertIsNone(exc_type)
            self.assertIsNone(exc)
            self.assertIsNone(exc_tb)
        class ExitCM(object):
            # [CLASS DOC] HELPER: Context manager whose exit hook runs a check; its enter hook
            # must not be called.
            def __init__(self, check_exc):
                # [FUNCTION DOC] HELPER: Stores the check function.
                self.check_exc = check_exc
            def __enter__(self):
                # [FUNCTION DOC] HELPER: Fails the test if called.
                self.fail("Should not be called!")
            def __exit__(self, *exc_details):
                # [FUNCTION DOC] HELPER: Calls the check function with the exception details.
                self.check_exc(*exc_details)
        with self.exit_stack() as stack:
            stack.push(_expect_ok)
            self.assertIs(stack._exit_callbacks[-1][1], _expect_ok)
            cm = ExitCM(_expect_ok)
            stack.push(cm)
            self.assertIs(stack._exit_callbacks[-1][1].__self__, cm)
            stack.push(_suppress_exc)
            self.assertIs(stack._exit_callbacks[-1][1], _suppress_exc)
            cm = ExitCM(_expect_exc)
            stack.push(cm)
            self.assertIs(stack._exit_callbacks[-1][1].__self__, cm)
            stack.push(_expect_exc)
            self.assertIs(stack._exit_callbacks[-1][1], _expect_exc)
            stack.push(_expect_exc)
            self.assertIs(stack._exit_callbacks[-1][1], _expect_exc)
            1/0

    def test_enter_context(self):
        # [FUNCTION DOC] Checks that enter_context calls enter, registers exit, and callbacks
        # run in reverse order.
        class TestCM(object):
            # [CLASS DOC] HELPER: Context manager recording its enter and exit calls in the
            # result list.
            def __enter__(self):
                # [FUNCTION DOC] HELPER: Records 1.
                result.append(1)
            def __exit__(self, *exc_details):
                # [FUNCTION DOC] HELPER: Records 3.
                result.append(3)

        result = []
        cm = TestCM()
        with self.exit_stack() as stack:
            @stack.callback  # Registered first => cleaned up last
            def _exit():
                # [FUNCTION DOC] HELPER: Callback recording 4 to the result list.
                result.append(4)
            self.assertIsNotNone(_exit)
            stack.enter_context(cm)
            self.assertIs(stack._exit_callbacks[-1][1].__self__, cm)
            result.append(2)
        self.assertEqual(result, [1, 2, 3, 4])

    def test_enter_context_errors(self):
        # [FUNCTION DOC] Checks that entering objects lacking enter or exit raises a TypeError
        # and registers nothing.
        class LacksEnterAndExit:
            # [CLASS DOC] HELPER: Class with neither enter nor exit.
            pass
        class LacksEnter:
            # [CLASS DOC] HELPER: Class with exit but no enter.
            def __exit__(self, *exc_info):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass
        class LacksExit:
            # [CLASS DOC] HELPER: Class with enter but no exit.
            def __enter__(self):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

        with self.exit_stack() as stack:
            with self.assertRaisesRegex(TypeError, 'the context manager'):
                stack.enter_context(LacksEnterAndExit())
            with self.assertRaisesRegex(TypeError, 'the context manager'):
                stack.enter_context(LacksEnter())
            with self.assertRaisesRegex(TypeError, 'the context manager'):
                stack.enter_context(LacksExit())
            self.assertFalse(stack._exit_callbacks)

    def test_close(self):
        # [FUNCTION DOC] Checks that close runs the registered callbacks immediately.
        result = []
        with self.exit_stack() as stack:
            @stack.callback
            def _exit():
                # [FUNCTION DOC] HELPER: Callback recording 1.
                result.append(1)
            self.assertIsNotNone(_exit)
            stack.close()
            result.append(2)
        self.assertEqual(result, [1, 2])

    def test_pop_all(self):
        # [FUNCTION DOC] Checks that pop_all transfers callbacks to a new stack, which runs them
        # when closed.
        result = []
        with self.exit_stack() as stack:
            @stack.callback
            def _exit():
                # [FUNCTION DOC] HELPER: Callback recording 3.
                result.append(3)
            self.assertIsNotNone(_exit)
            new_stack = stack.pop_all()
            result.append(1)
        result.append(2)
        new_stack.close()
        self.assertEqual(result, [1, 2, 3])

    def test_exit_raise(self):
        # [FUNCTION DOC] Checks that an exit callback returning false lets the body
        # exception propagate.
        with self.assertRaises(ZeroDivisionError):
            with self.exit_stack() as stack:
                stack.push(lambda *exc: False)
                1/0

    def test_exit_suppress(self):
        # [FUNCTION DOC] Checks that an exit callback returning true suppresses the
        # body exception.
        with self.exit_stack() as stack:
            stack.push(lambda *exc: True)
            1/0

    def test_exit_exception_traceback(self):
        # [FUNCTION DOC] Checks the traceback frames of an exception raised by a callback while
        # another is being handled, against the expected internal frames, and that the original
        # exception is its context. Captures current behavior rather than prescribing it.
        # This test captures the current behavior of ExitStack so that we know
        # if we ever unintendedly change it. It is not a statement of what the
        # desired behavior is (for instance, we may want to remove some of the
        # internal contextlib frames).

        def raise_exc(exc):
            # [FUNCTION DOC] HELPER: Raises the given exception.
            raise exc

        try:
            with self.exit_stack() as stack:
                stack.callback(raise_exc, ValueError)
                1/0
        except ValueError as e:
            exc = e

        self.assertIsInstance(exc, ValueError)
        ve_frames = traceback.extract_tb(exc.__traceback__)
        expected = \
            [('test_exit_exception_traceback', 'with self.exit_stack() as stack:')] + \
            self.callback_error_internal_frames + \
            [('_exit_wrapper', 'callback(*args, **kwds)'),
             ('raise_exc', 'raise exc')]

        self.assertEqual(
            [(f.name, f.line) for f in ve_frames], expected)

        self.assertIsInstance(exc.__context__, ZeroDivisionError)
        zde_frames = traceback.extract_tb(exc.__context__.__traceback__)
        self.assertEqual([(f.name, f.line) for f in zde_frames],
                         [('test_exit_exception_traceback', '1/0')])

    def test_exit_exception_chaining_reference(self):
        # [FUNCTION DOC] Checks that exception contexts produced by nested with statements are
        # as expected, as a reference for the stack-based chaining test.
        # Sanity check to make sure that ExitStack chaining matches
        # actual nested with statements
        class RaiseExc:
            # [CLASS DOC] HELPER: Context manager whose exit raises a given exception.
            def __init__(self, exc):
                # [FUNCTION DOC] HELPER: Stores the exception to raise.
                self.exc = exc
            def __enter__(self):
                # [FUNCTION DOC] HELPER: Returns the manager itself.
                return self
            def __exit__(self, *exc_details):
                # [FUNCTION DOC] HELPER: Raises the stored exception.
                raise self.exc

        class RaiseExcWithContext:
            # [CLASS DOC] HELPER: Context manager whose exit raises an inner exception and then
            # an outer one.
            def __init__(self, outer, inner):
                # [FUNCTION DOC] HELPER: Stores the outer and inner exceptions.
                self.outer = outer
                self.inner = inner
            def __enter__(self):
                # [FUNCTION DOC] HELPER: Returns the manager itself.
                return self
            def __exit__(self, *exc_details):
                # [FUNCTION DOC] HELPER: Raises the inner exception, catches it, then raises the
                # outer one.
                try:
                    raise self.inner
                except:
                    raise self.outer

        class SuppressExc:
            # [CLASS DOC] HELPER: Context manager that suppresses everything and saves the
            # exception details on the class.
            def __enter__(self):
                # [FUNCTION DOC] HELPER: Returns the manager itself.
                return self
            def __exit__(self, *exc_details):
                # [FUNCTION DOC] HELPER: Saves the exception details on the class and
                # returns true.
                type(self).saved_details = exc_details
                return True

        try:
            with RaiseExc(IndexError):
                with RaiseExcWithContext(KeyError, AttributeError):
                    with SuppressExc():
                        with RaiseExc(ValueError):
                            1 / 0
        except IndexError as exc:
            self.assertIsInstance(exc.__context__, KeyError)
            self.assertIsInstance(exc.__context__.__context__, AttributeError)
            # Inner exceptions were suppressed
            self.assertIsNone(exc.__context__.__context__.__context__)
        else:
            self.fail("Expected IndexError, but no exception was raised")
        # Check the inner exceptions
        inner_exc = SuppressExc.saved_details[1]
        self.assertIsInstance(inner_exc, ValueError)
        self.assertIsInstance(inner_exc.__context__, ZeroDivisionError)

    def test_exit_exception_chaining(self):
        # [FUNCTION DOC] Checks that exception chaining through a stack matches the nested with
        # reference behavior.
        # Ensure exception chaining matches the reference behaviour
        def raise_exc(exc):
            # [FUNCTION DOC] HELPER: Raises the given exception.
            raise exc

        saved_details = None
        def suppress_exc(*exc_details):
            # [FUNCTION DOC] HELPER: Saves the exception details in an enclosing variable and
            # suppresses the exception.
            nonlocal saved_details
            saved_details = exc_details
            return True

        try:
            with self.exit_stack() as stack:
                stack.callback(raise_exc, IndexError)
                stack.callback(raise_exc, KeyError)
                stack.callback(raise_exc, AttributeError)
                stack.push(suppress_exc)
                stack.callback(raise_exc, ValueError)
                1 / 0
        except IndexError as exc:
            self.assertIsInstance(exc.__context__, KeyError)
            self.assertIsInstance(exc.__context__.__context__, AttributeError)
            # Inner exceptions were suppressed
            self.assertIsNone(exc.__context__.__context__.__context__)
        else:
            self.fail("Expected IndexError, but no exception was raised")
        # Check the inner exceptions
        inner_exc = saved_details[1]
        self.assertIsInstance(inner_exc, ValueError)
        self.assertIsInstance(inner_exc.__context__, ZeroDivisionError)

    def test_exit_exception_explicit_none_context(self):
        # [FUNCTION DOC] Checks that an explicitly cleared exception context is preserved by
        # the stack.
        # Ensure ExitStack chaining matches actual nested `with` statements
        # regarding explicit __context__ = None.

        class MyException(Exception):
            # [CLASS DOC] HELPER: Simple exception class.
            pass

        @contextmanager
        def my_cm():
            # [FUNCTION DOC] HELPER: Generator that replaces any exception with one whose
            # context is cleared.
            try:
                yield
            except BaseException:
                exc = MyException()
                try:
                    raise exc
                finally:
                    exc.__context__ = None

        @contextmanager
        def my_cm_with_exit_stack():
            # [FUNCTION DOC] HELPER: Generator that enters the previous manager into an exit
            # stack and yields the stack.
            with self.exit_stack() as stack:
                stack.enter_context(my_cm())
                yield stack

        for cm in (my_cm, my_cm_with_exit_stack):
            with self.subTest():
                try:
                    with cm():
                        raise IndexError()
                except MyException as exc:
                    self.assertIsNone(exc.__context__)
                else:
                    self.fail("Expected IndexError, but no exception was raised")

    def test_exit_exception_non_suppressing(self):
        # [FUNCTION DOC] Checks that callbacks which do not suppress let the later raised
        # exceptions propagate.
        # http://bugs.python.org/issue19092
        def raise_exc(exc):
            # [FUNCTION DOC] HELPER: Raises the given exception.
            raise exc

        def suppress_exc(*exc_details):
            # [FUNCTION DOC] HELPER: Suppresses the exception.
            return True

        try:
            with self.exit_stack() as stack:
                stack.callback(lambda: None)
                stack.callback(raise_exc, IndexError)
        except Exception as exc:
            self.assertIsInstance(exc, IndexError)
        else:
            self.fail("Expected IndexError, but no exception was raised")

        try:
            with self.exit_stack() as stack:
                stack.callback(raise_exc, KeyError)
                stack.push(suppress_exc)
                stack.callback(raise_exc, IndexError)
        except Exception as exc:
            self.assertIsInstance(exc, KeyError)
        else:
            self.fail("Expected KeyError, but no exception was raised")

    def test_exit_exception_with_correct_context(self):
        # [FUNCTION DOC] Checks the context chain of exceptions raised by nested generator
        # managers entered into a stack, which must not form loops.
        # http://bugs.python.org/issue20317
        @contextmanager
        def gets_the_context_right(exc):
            # [FUNCTION DOC] HELPER: Generator that raises the given exception on cleanup.
            try:
                yield
            finally:
                raise exc

        exc1 = Exception(1)
        exc2 = Exception(2)
        exc3 = Exception(3)
        exc4 = Exception(4)

        # The contextmanager already fixes the context, so prior to the
        # fix, ExitStack would try to fix it *again* and get into an
        # infinite self-referential loop
        try:
            with self.exit_stack() as stack:
                stack.enter_context(gets_the_context_right(exc4))
                stack.enter_context(gets_the_context_right(exc3))
                stack.enter_context(gets_the_context_right(exc2))
                raise exc1
        except Exception as exc:
            self.assertIs(exc, exc4)
            self.assertIs(exc.__context__, exc3)
            self.assertIs(exc.__context__.__context__, exc2)
            self.assertIs(exc.__context__.__context__.__context__, exc1)
            self.assertIsNone(
                       exc.__context__.__context__.__context__.__context__)

    def test_exit_exception_with_existing_context(self):
        # [FUNCTION DOC] Checks the context chain when callbacks raise exceptions that already
        # have contexts.
        # Addresses a lack of test coverage discovered after checking in a
        # fix for issue 20317 that still contained debugging code.
        def raise_nested(inner_exc, outer_exc):
            # [FUNCTION DOC] HELPER: Raises the inner exception and, in a cleanup block, the
            # outer one.
            try:
                raise inner_exc
            finally:
                raise outer_exc
        exc1 = Exception(1)
        exc2 = Exception(2)
        exc3 = Exception(3)
        exc4 = Exception(4)
        exc5 = Exception(5)
        try:
            with self.exit_stack() as stack:
                stack.callback(raise_nested, exc4, exc5)
                stack.callback(raise_nested, exc2, exc3)
                raise exc1
        except Exception as exc:
            self.assertIs(exc, exc5)
            self.assertIs(exc.__context__, exc4)
            self.assertIs(exc.__context__.__context__, exc3)
            self.assertIs(exc.__context__.__context__.__context__, exc2)
            self.assertIs(
                 exc.__context__.__context__.__context__.__context__, exc1)
            self.assertIsNone(
                exc.__context__.__context__.__context__.__context__.__context__)

    def test_body_exception_suppress(self):
        # [FUNCTION DOC] Checks that a suppressing callback hides a body exception, so no error
        # is raised.
        def suppress_exc(*exc_details):
            # [FUNCTION DOC] HELPER: Suppresses the exception.
            return True
        try:
            with self.exit_stack() as stack:
                stack.push(suppress_exc)
                1/0
        except IndexError as exc:
            self.fail("Expected no exception, got IndexError")

    def test_exit_exception_chaining_suppress(self):
        # [FUNCTION DOC] Checks that an error raised by a callback after a suppressing callback
        # does not escape.
        with self.exit_stack() as stack:
            stack.push(lambda *exc: True)
            stack.push(lambda *exc: 1/0)
            stack.push(lambda *exc: {}[1])

    @slowTest
    def test_excessive_nesting(self):
        # [FUNCTION DOC] Checks that a very deep stack of callbacks (ten thousand) unwinds.
        # Marked as a slow test.
        # The original implementation would die with RecursionError here
        with self.exit_stack() as stack:
            for i in range(10000):
                stack.callback(int)

    def test_instance_bypass(self):
        # [FUNCTION DOC] Checks that attributes set on an instance are bypassed when looking up
        # enter and exit.
        class Example(object): pass
        cm = Example()
        cm.__enter__ = object()
        cm.__exit__ = object()
        stack = self.exit_stack()
        with self.assertRaisesRegex(TypeError, 'the context manager'):
            stack.enter_context(cm)
        stack.push(cm)
        self.assertIs(stack._exit_callbacks[-1][1], cm)

    def test_dont_reraise_RuntimeError(self):
        # [FUNCTION DOC] Checks that a RuntimeError subclass in a stack is not re-raised
        # infinitely and chains.
        # https://bugs.python.org/issue27122
        class UniqueException(Exception): pass
        class UniqueRuntimeError(RuntimeError): pass

        @contextmanager
        def second():
            # [FUNCTION DOC] HELPER: Generator that converts any exception to a new chained one.
            try:
                yield 1
            except Exception as exc:
                raise UniqueException("new exception") from exc

        @contextmanager
        def first():
            # [FUNCTION DOC] HELPER: Generator that re-raises any exception.
            try:
                yield 1
            except Exception as exc:
                raise exc

        # The UniqueRuntimeError should be caught by second()'s exception
        # handler which chain raised a new UniqueException.
        with self.assertRaises(UniqueException) as err_ctx:
            with self.exit_stack() as es_ctx:
                es_ctx.enter_context(second())
                es_ctx.enter_context(first())
                raise UniqueRuntimeError("please no infinite loop.")

        exc = err_ctx.exception
        self.assertIsInstance(exc, UniqueException)
        self.assertIsInstance(exc.__context__, UniqueRuntimeError)
        self.assertIsNone(exc.__context__.__context__)
        self.assertIsNone(exc.__context__.__cause__)
        self.assertIs(exc.__cause__, exc.__context__)


class TestExitStack(_TestBaseExitStack, __TestCase):
    # [CLASS DOC] Runs the shared stack tests against ExitStack and defines the internal frames
    # expected in callback tracebacks.
    exit_stack = ExitStack
    callback_error_internal_frames = [
        ('__exit__', 'raise exc'),
        ('__exit__', 'if cb(*exc_details):'),
    ]


class _TestRedirectStream:
    # [CLASS DOC] HELPER: Mixin with stream redirection tests; subclasses set the redirect class
    # and the name of the system stream attribute.

    redirect_stream = None
    orig_stream = None

    @support.requires_docstrings
    def test_instance_docs(self):
        # [FUNCTION DOC] Checks that instances share the class docstring. Requires
        # docstrings enabled.
        # Issue 19330: ensure context manager instances have good docstrings
        cm_docstring = self.redirect_stream.__doc__
        obj = self.redirect_stream(None)
        self.assertEqual(obj.__doc__, cm_docstring)

    def test_no_redirect_in_init(self):
        # [FUNCTION DOC] Checks that constructing the redirector does not change the
        # system stream.
        orig_stdout = getattr(sys, self.orig_stream)
        self.redirect_stream(None)
        self.assertIs(getattr(sys, self.orig_stream), orig_stdout)

    def test_redirect_to_string_io(self):
        # [FUNCTION DOC] Checks that output is written to the string target and the stream is
        # restored afterwards.
        f = io.StringIO()
        msg = "Consider an API like help(), which prints directly to stdout"
        orig_stdout = getattr(sys, self.orig_stream)
        with self.redirect_stream(f):
            print(msg, file=getattr(sys, self.orig_stream))
        self.assertIs(getattr(sys, self.orig_stream), orig_stdout)
        s = f.getvalue().strip()
        self.assertEqual(s, msg)

    def test_enter_result_is_target(self):
        # [FUNCTION DOC] Checks that entering returns the target stream.
        f = io.StringIO()
        with self.redirect_stream(f) as enter_result:
            self.assertIs(enter_result, f)

    def test_cm_is_reusable(self):
        # [FUNCTION DOC] Checks that the same redirector can be used for several blocks
        # in sequence.
        f = io.StringIO()
        write_to_f = self.redirect_stream(f)
        orig_stdout = getattr(sys, self.orig_stream)
        with write_to_f:
            print("Hello", end=" ", file=getattr(sys, self.orig_stream))
        with write_to_f:
            print("World!", file=getattr(sys, self.orig_stream))
        self.assertIs(getattr(sys, self.orig_stream), orig_stdout)
        s = f.getvalue()
        self.assertEqual(s, "Hello World!\n")

    def test_cm_is_reentrant(self):
        # [FUNCTION DOC] Checks that the same redirector can be used re-entrantly.
        f = io.StringIO()
        write_to_f = self.redirect_stream(f)
        orig_stdout = getattr(sys, self.orig_stream)
        with write_to_f:
            print("Hello", end=" ", file=getattr(sys, self.orig_stream))
            with write_to_f:
                print("World!", file=getattr(sys, self.orig_stream))
        self.assertIs(getattr(sys, self.orig_stream), orig_stdout)
        s = f.getvalue()
        self.assertEqual(s, "Hello World!\n")


class TestRedirectStdout(_TestRedirectStream, __TestCase):
    # [CLASS DOC] Runs the redirect tests for standard output.

    redirect_stream = redirect_stdout
    orig_stream = "stdout"


class TestRedirectStderr(_TestRedirectStream, __TestCase):
    # [CLASS DOC] Runs the redirect tests for standard error.

    redirect_stream = redirect_stderr
    orig_stream = "stderr"


class TestSuppress(ExceptionIsLikeMixin, __TestCase):
    # [CLASS DOC] Tests for suppress.

    @support.requires_docstrings
    def test_instance_docs(self):
        # [FUNCTION DOC] Checks that instances share the class docstring. Requires
        # docstrings enabled.
        # Issue 19330: ensure context manager instances have good docstrings
        cm_docstring = suppress.__doc__
        obj = suppress()
        self.assertEqual(obj.__doc__, cm_docstring)

    def test_no_result_from_enter(self):
        # [FUNCTION DOC] Checks that entering returns None.
        with suppress(ValueError) as enter_result:
            self.assertIsNone(enter_result)

    def test_no_exception(self):
        # [FUNCTION DOC] Checks that code without exceptions runs unaffected.
        with suppress(ValueError):
            self.assertEqual(pow(2, 5), 32)

    def test_exact_exception(self):
        # [FUNCTION DOC] Checks that an exception of exactly the listed type is suppressed.
        with suppress(TypeError):
            len(5)

    def test_exception_hierarchy(self):
        # [FUNCTION DOC] Checks that exceptions of a subclass of the listed type are suppressed.
        with suppress(LookupError):
            'Hello'[50]

    def test_other_exception(self):
        # [FUNCTION DOC] Checks that an unlisted exception type is not suppressed.
        with self.assertRaises(ZeroDivisionError):
            with suppress(TypeError):
                1/0

    def test_no_args(self):
        # [FUNCTION DOC] Checks that with no listed types nothing is suppressed.
        with self.assertRaises(ZeroDivisionError):
            with suppress():
                1/0

    def test_multiple_exception_args(self):
        # [FUNCTION DOC] Checks that several listed exception types can each be suppressed.
        with suppress(ZeroDivisionError, TypeError):
            1/0
        with suppress(ZeroDivisionError, TypeError):
            len(5)

    def test_cm_is_reentrant(self):
        # [FUNCTION DOC] Checks that one suppress instance can be reused and nested.
        ignore_exceptions = suppress(Exception)
        with ignore_exceptions:
            pass
        with ignore_exceptions:
            len(5)
        with ignore_exceptions:
            with ignore_exceptions: # Check nested usage
                len(5)
            outer_continued = True
            1/0
        self.assertTrue(outer_continued)

    def test_exception_groups(self):
        # [FUNCTION DOC] Checks suppression with exception groups: matching leaves are removed,
        # partial matches leave a group of the rest, and base exception groups are handled the
        # same way, becoming ordinary groups when only ordinary leaves remain.
        eg_ve = lambda: ExceptionGroup(
            "EG with ValueErrors only",
            [ValueError("ve1"), ValueError("ve2"), ValueError("ve3")],
        )
        eg_all = lambda: ExceptionGroup(
            "EG with many types of exceptions",
            [ValueError("ve1"), KeyError("ke1"), ValueError("ve2"), KeyError("ke2")],
        )
        with suppress(ValueError):
            raise eg_ve()
        with suppress(ValueError, KeyError):
            raise eg_all()
        with self.assertRaises(ExceptionGroup) as eg1:
            with suppress(ValueError):
                raise eg_all()
        self.assertExceptionIsLike(
            eg1.exception,
            ExceptionGroup(
                "EG with many types of exceptions",
                [KeyError("ke1"), KeyError("ke2")],
            ),
        )
        # Check handling of BaseExceptionGroup, using GeneratorExit so that
        # we don't accidentally discard a ctrl-c with KeyboardInterrupt.
        with suppress(GeneratorExit):
            raise BaseExceptionGroup("message", [GeneratorExit()])
        # If we raise a BaseException group, we can still suppress parts
        with self.assertRaises(BaseExceptionGroup) as eg1:
            with suppress(KeyError):
                raise BaseExceptionGroup("message", [GeneratorExit("g"), KeyError("k")])
        self.assertExceptionIsLike(
            eg1.exception, BaseExceptionGroup("message", [GeneratorExit("g")]),
        )
        # If we suppress all the leaf BaseExceptions, we get a non-base ExceptionGroup
        with self.assertRaises(ExceptionGroup) as eg1:
            with suppress(GeneratorExit):
                raise BaseExceptionGroup("message", [GeneratorExit("g"), KeyError("k")])
        self.assertExceptionIsLike(
            eg1.exception, ExceptionGroup("message", [KeyError("k")]),
        )


class TestChdir(__TestCase):
    # [CLASS DOC] Tests for the chdir context manager.
    def make_relative_path(self, *parts):
        # [FUNCTION DOC] HELPER: Builds an absolute path under the directory holding this
        # test file.
        return os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            *parts,
        )

    def test_simple(self):
        # [FUNCTION DOC] Checks that the working directory changes inside the block and is
        # restored after.
        old_cwd = os.getcwd()
        target = self.make_relative_path('data')
        self.assertNotEqual(old_cwd, target)

        with chdir(target):
            self.assertEqual(os.getcwd(), target)
        self.assertEqual(os.getcwd(), old_cwd)

    @unittest.skip("Missing archivetestdata")
    def test_reentrant(self):
        # [FUNCTION DOC] Checks nested use of the same chdir managers. Skipped because test data
        # is missing.
        old_cwd = os.getcwd()
        target1 = self.make_relative_path('data')
        target2 = self.make_relative_path('archivetestdata')
        self.assertNotIn(old_cwd, (target1, target2))
        chdir1, chdir2 = chdir(target1), chdir(target2)

        with chdir1:
            self.assertEqual(os.getcwd(), target1)
            with chdir2:
                self.assertEqual(os.getcwd(), target2)
                with chdir1:
                    self.assertEqual(os.getcwd(), target1)
                self.assertEqual(os.getcwd(), target2)
            self.assertEqual(os.getcwd(), target1)
        self.assertEqual(os.getcwd(), old_cwd)

    def test_exception(self):
        # [FUNCTION DOC] Checks that the working directory is restored when the body raises.
        old_cwd = os.getcwd()
        target = self.make_relative_path('data')
        self.assertNotEqual(old_cwd, target)

        try:
            with chdir(target):
                self.assertEqual(os.getcwd(), target)
                raise RuntimeError("boom")
        except RuntimeError as re:
            self.assertEqual(str(re), "boom")
        self.assertEqual(os.getcwd(), old_cwd)


if __name__ == "__main__":
    run_tests()
