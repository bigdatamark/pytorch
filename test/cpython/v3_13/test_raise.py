# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa
# [MODULE DOC] Dynamo-patched copy of the CPython 3.13 test_raise suite covering the raise
# statement: re-raise, exception cause (from) and context chaining, traceback
# attachment and the TracebackType constructor, and removed raise forms (tuples,
# strings). Import-time side effect: installs a RedirectImportFinder at the front of
# sys.meta_path that maps selected "test.*" module imports to top-level modules.
# The test classes derive from CPythonTestCase (aliased as __TestCase).

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_raise.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import (
    run_tests,
)

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
    # [CLASS DOC] HELPER: Import hook (meta path finder) that redirects imports of the module
    # names listed in redirect_imports from the "test." package to the same-named top-level
    # module. State: none; only find_spec is defined.
    def find_spec(self, fullname, path, target=None):
        # [FUNCTION DOC] HELPER: For a name in redirect_imports, import the module without the
        # "test." prefix, register it in sys.modules under the full name and return its spec;
        # return None if the name is not listed or the import fails with ImportError.
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

# Copyright 2007 Google, Inc. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""Tests for the raise statement."""

from test import support
import sys
import types
import unittest


def get_tb():
    # [FUNCTION DOC] HELPER: Raise and catch an OSError and return its traceback object.
    try:
        raise OSError()
    except OSError as e:
        return e.__traceback__


class Context:
    # [CLASS DOC] HELPER: Context manager that swallows any exception raised in its block.
    def __enter__(self):
        # [FUNCTION DOC] HELPER: Enter the context; returns self.
        return self
    def __exit__(self, exc_type, exc_value, exc_tb):
        # [FUNCTION DOC] HELPER: Exit the context; always
        # returns True so exceptions are suppressed.
        return True


class TestRaise(__TestCase):
    # [CLASS DOC] Tests for plain raise: bare re-raise, re-raise inside handlers, finally,
    # with blocks and generators, raise from None, bad exception construction and assert
    # messages.
    def test_invalid_reraise(self):
        # [FUNCTION DOC] A bare raise with no active exception must raise RuntimeError mentioning
        # "No active exception".
        try:
            raise
        except RuntimeError as e:
            self.assertIn("No active exception", str(e))
        else:
            self.fail("No exception raised")

    def test_reraise(self):
        # [FUNCTION DOC] A bare raise in a handler re-raises the same exception object.
        try:
            try:
                raise IndexError()
            except IndexError as e:
                exc1 = e
                raise
        except IndexError as exc2:
            self.assertIs(exc1, exc2)
        else:
            self.fail("No exception raised")

    def test_except_reraise(self):
        # [FUNCTION DOC] A bare raise after a nested handled exception re-raises the original
        # TypeError, not the inner KeyError.
        def reraise():
            # [FUNCTION DOC] HELPER: Raises TypeError, handles a KeyError inside the handler, then
            # re-raises.
            try:
                raise TypeError("foo")
            except:
                try:
                    raise KeyError("caught")
                except KeyError:
                    pass
                raise
        self.assertRaises(TypeError, reraise)

    def test_finally_reraise(self):
        # [FUNCTION DOC] A bare raise inside a finally that runs while a KeyError propagates
        # re-raises the KeyError.
        def reraise():
            # [FUNCTION DOC] HELPER: Raises TypeError, then a
            # KeyError whose finally does a bare raise.
            try:
                raise TypeError("foo")
            except:
                try:
                    raise KeyError("caught")
                finally:
                    raise
        self.assertRaises(KeyError, reraise)

    def test_nested_reraise(self):
        # [FUNCTION DOC] A bare raise in a nested function called from a handler re-raises the
        # handler's exception (TypeError).
        def nested_reraise():
            # [FUNCTION DOC] HELPER: Performs a bare raise.
            raise
        def reraise():
            # [FUNCTION DOC] HELPER: Raises TypeError and calls nested_reraise from the handler.
            try:
                raise TypeError("foo")
            except:
                nested_reraise()
        self.assertRaises(TypeError, reraise)

    def test_raise_from_None(self):
        # [FUNCTION DOC] Raising ValueError from None keeps the TypeError as context but sets the
        # cause to None.
        try:
            try:
                raise TypeError("foo")
            except:
                raise ValueError() from None
        except ValueError as e:
            self.assertIsInstance(e.__context__, TypeError)
            self.assertIsNone(e.__cause__)

    def test_with_reraise1(self):
        # [FUNCTION DOC] A bare raise after an empty with block in a handler still re-raises
        # the TypeError.
        def reraise():
            # [FUNCTION DOC] HELPER: Handler uses a Context block, then re-raises.
            try:
                raise TypeError("foo")
            except:
                with Context():
                    pass
                raise
        self.assertRaises(TypeError, reraise)

    def test_with_reraise2(self):
        # [FUNCTION DOC] A bare raise after a with block that swallowed a KeyError still
        # re-raises the TypeError.
        def reraise():
            # [FUNCTION DOC] HELPER: Handler raises a KeyError
            # inside a Context block, then re-raises.
            try:
                raise TypeError("foo")
            except:
                with Context():
                    raise KeyError("caught")
                raise
        self.assertRaises(TypeError, reraise)

    def test_yield_reraise(self):
        # [FUNCTION DOC] A generator may do a bare raise after a yield inside a handler; the
        # TypeError is raised on next resume and then the generator is exhausted.
        def reraise():
            # [FUNCTION DOC] HELPER: Generator that yields 1 inside a handler, then re-raises.
            try:
                raise TypeError("foo")
            except:
                yield 1
                raise
        g = reraise()
        next(g)
        self.assertRaises(TypeError, lambda: next(g))
        self.assertRaises(StopIteration, lambda: next(g))

    def test_erroneous_exception(self):
        # [FUNCTION DOC] If instantiating the raised exception class fails, that RuntimeError
        # is what propagates.
        class MyException(Exception):
            # [CLASS DOC] HELPER: Exception whose constructor raises RuntimeError.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Always raises RuntimeError.
                raise RuntimeError()

        try:
            raise MyException
        except RuntimeError:
            pass
        else:
            self.fail("No exception raised")

    def test_new_returns_invalid_instance(self):
        # [FUNCTION DOC] Issue 11627: if the exception class creates a non-exception object,
        # raising it must raise TypeError.
        # See issue #11627.
        class MyException(Exception):
            # [CLASS DOC] HELPER: Exception class whose __new__ returns a plain object.
            def __new__(cls, *args):
                # [FUNCTION DOC] HELPER: Ignores arguments and returns a new plain object.
                return object()

        with self.assertRaises(TypeError):
            raise MyException

    def test_assert_with_tuple_arg(self):
        # [FUNCTION DOC] An assert with a tuple message gives an AssertionError whose text is
        # the tuple's string form.
        try:
            assert False, (3,)
        except AssertionError as e:
            self.assertEqual(str(e), "(3,)")



class TestCause(__TestCase):
    # [CLASS DOC] Tests for the cause attribute set by "raise ... from ...".

    def testCauseSyntax(self):
        # [FUNCTION DOC] Checks raise from None sets cause None and suppress-context true, that
        # suppress-context can be reset, and that the context is kept when re-raised.
        try:
            try:
                try:
                    raise TypeError
                except Exception:
                    raise ValueError from None
            except ValueError as exc:
                self.assertIsNone(exc.__cause__)
                self.assertTrue(exc.__suppress_context__)
                exc.__suppress_context__ = False
                raise exc
        except ValueError as exc:
            e = exc

        self.assertIsNone(e.__cause__)
        self.assertFalse(e.__suppress_context__)
        self.assertIsInstance(e.__context__, TypeError)

    def test_invalid_cause(self):
        # [FUNCTION DOC] Raising from a non-exception (an int) must raise TypeError mentioning
        # "exception cause".
        try:
            raise IndexError from 5
        except TypeError as e:
            self.assertIn("exception cause", str(e))
        else:
            self.fail("No exception raised")

    def test_class_cause(self):
        # [FUNCTION DOC] Raising from an exception class stores an instance of it as cause.
        try:
            raise IndexError from KeyError
        except IndexError as e:
            self.assertIsInstance(e.__cause__, KeyError)
        else:
            self.fail("No exception raised")

    def test_class_cause_nonexception_result(self):
        # [FUNCTION DOC] If the cause class constructs None, raising must give TypeError saying
        # BaseException instance should have been returned.
        class ConstructsNone(BaseException):
            # [CLASS DOC] HELPER: Exception class whose constructor returns None.
            @classmethod
            def __new__(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Ignores its arguments and returns None.
                return None
        try:
            raise IndexError from ConstructsNone
        except TypeError as e:
            self.assertIn("should have returned an instance of BaseException", str(e))
        except IndexError:
            self.fail("Wrong kind of exception raised")
        else:
            self.fail("No exception raised")

    def test_instance_cause(self):
        # [FUNCTION DOC] Raising from an exception instance stores that exact object as cause.
        cause = KeyError()
        try:
            raise IndexError from cause
        except IndexError as e:
            self.assertIs(e.__cause__, cause)
        else:
            self.fail("No exception raised")

    def test_erroneous_cause(self):
        # [FUNCTION DOC] If constructing the cause class raises RuntimeError, that error
        # propagates instead of IndexError.
        class MyException(Exception):
            # [CLASS DOC] HELPER: Exception whose constructor raises RuntimeError.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Always raises RuntimeError.
                raise RuntimeError()

        try:
            raise IndexError from MyException
        except RuntimeError:
            pass
        else:
            self.fail("No exception raised")


class TestTraceback(__TestCase):
    # [CLASS DOC] Tests that raising sets or extends the traceback.

    def test_sets_traceback(self):
        # [FUNCTION DOC] A caught exception has a TracebackType traceback.
        try:
            raise IndexError()
        except IndexError as e:
            self.assertIsInstance(e.__traceback__, types.TracebackType)
        else:
            self.fail("No exception raised")

    def test_accepts_traceback(self):
        # [FUNCTION DOC] with_traceback adds the supplied traceback to the chain: the new
        # traceback differs from it and its next entry is the supplied one.
        tb = get_tb()
        try:
            raise IndexError().with_traceback(tb)
        except IndexError as e:
            self.assertNotEqual(e.__traceback__, tb)
            self.assertEqual(e.__traceback__.tb_next, tb)
        else:
            self.fail("No exception raised")


class TestTracebackType(__TestCase):
    # [CLASS DOC] Tests for traceback object attributes and construction.

    def raiser(self):
        # [FUNCTION DOC] HELPER: Always raises ValueError.
        raise ValueError

    def test_attrs(self):
        # [FUNCTION DOC] Checks traceback attributes (tb_next, tb_lineno), rejection of a non-
        # traceback tb_next (TypeError) and of loops (ValueError), and valid reassignment
        # including None.
        try:
            self.raiser()
        except Exception as exc:
            tb = exc.__traceback__

        self.assertIsInstance(tb.tb_next, types.TracebackType)
        # self.assertIs(tb.tb_frame, sys._getframe())
        # self.assertIsInstance(tb.tb_lasti, int)
        self.assertIsInstance(tb.tb_lineno, int)

        self.assertIs(tb.tb_next.tb_next, None)

        # Invalid assignments
        # with self.assertRaises(TypeError):
        #     del tb.tb_next

        with self.assertRaises(TypeError):
            tb.tb_next = "asdf"

        # Loops
        with self.assertRaises(ValueError):
            tb.tb_next = tb

        with self.assertRaises(ValueError):
            tb.tb_next.tb_next = tb

        # Valid assignments
        tb.tb_next = None
        self.assertIs(tb.tb_next, None)

        new_tb = get_tb()
        tb.tb_next = new_tb
        self.assertIs(tb.tb_next, new_tb)

    def test_constructor(self):
        # [FUNCTION DOC] Checks the TracebackType constructor stores next, frame, lasti and
        # lineno, accepts None for next, and raises TypeError for wrongly typed arguments.
        other_tb = get_tb()
        frame = sys._getframe()

        tb = types.TracebackType(other_tb, frame, 1, 2)
        self.assertEqual(tb.tb_next, other_tb)
        self.assertEqual(tb.tb_frame, frame)
        self.assertEqual(tb.tb_lasti, 1)
        self.assertEqual(tb.tb_lineno, 2)

        tb = types.TracebackType(None, frame, 1, 2)
        self.assertEqual(tb.tb_next, None)

        with self.assertRaises(TypeError):
            types.TracebackType("no", frame, 1, 2)

        with self.assertRaises(TypeError):
            types.TracebackType(other_tb, "no", 1, 2)

        with self.assertRaises(TypeError):
            types.TracebackType(other_tb, frame, "no", 2)

        with self.assertRaises(TypeError):
            types.TracebackType(other_tb, frame, 1, "nuh-uh")


class TestContext(__TestCase):
    # [CLASS DOC] Tests for implicit exception context (__context__) when raising while
    # handling another exception.
    def test_instance_context_instance_raise(self):
        # [FUNCTION DOC] Raising a new exception while handling an instance sets that instance
        # as context.
        context = IndexError()
        try:
            try:
                raise context
            except:
                raise OSError()
        except OSError as e:
            self.assertIs(e.__context__, context)
        else:
            self.fail("No exception raised")

    def test_class_context_instance_raise(self):
        # [FUNCTION DOC] Handling an exception raised from a class, then raising an instance: the
        # context is an instance of that class, not the class.
        context = IndexError
        try:
            try:
                raise context
            except:
                raise OSError()
        except OSError as e:
            self.assertIsNot(e.__context__, context)
            self.assertIsInstance(e.__context__, context)
        else:
            self.fail("No exception raised")

    def test_class_context_class_raise(self):
        # [FUNCTION DOC] Same as above but raising a class: context is an instance, not the class.
        context = IndexError
        try:
            try:
                raise context
            except:
                raise OSError
        except OSError as e:
            self.assertIsNot(e.__context__, context)
            self.assertIsInstance(e.__context__, context)
        else:
            self.fail("No exception raised")

    def test_c_exception_context(self):
        # [FUNCTION DOC] Raising OSError in a handler for a ZeroDivisionError gives that
        # error as context.
        try:
            try:
                1/0
            except:
                raise OSError
        except OSError as e:
            self.assertIsInstance(e.__context__, ZeroDivisionError)
        else:
            self.fail("No exception raised")

    def test_c_exception_raise(self):
        # [FUNCTION DOC] A NameError triggered inside a handler has ZeroDivisionError as context.
        try:
            try:
                1/0
            except:
                xyzzy
        except NameError as e:
            self.assertIsInstance(e.__context__, ZeroDivisionError)
        else:
            self.fail("No exception raised")

    def test_noraise_finally(self):
        # [FUNCTION DOC] Raising in a finally with no pending exception leaves context None.
        try:
            try:
                pass
            finally:
                raise OSError
        except OSError as e:
            self.assertIsNone(e.__context__)
        else:
            self.fail("No exception raised")

    def test_raise_finally(self):
        # [FUNCTION DOC] Raising in a finally while
        # ZeroDivisionError propagates sets it as context.
        try:
            try:
                1/0
            finally:
                raise OSError
        except OSError as e:
            self.assertIsInstance(e.__context__, ZeroDivisionError)
        else:
            self.fail("No exception raised")

    def test_context_manager(self):
        # [FUNCTION DOC] An error in a context manager exit while ZeroDivisionError propagates
        # has that error as context.
        class ContextManager:
            # [CLASS DOC] HELPER: Context manager whose exit refers to an undefined name.
            def __enter__(self):
                # [FUNCTION DOC] HELPER: Does nothing on enter.
                pass
            def __exit__(self, t, v, tb):
                # [FUNCTION DOC] HELPER: Raises NameError by using an undefined name.
                xyzzy
        try:
            with ContextManager():
                1/0
        except NameError as e:
            self.assertIsInstance(e.__context__, ZeroDivisionError)
        else:
            self.fail("No exception raised")

    def test_cycle_broken(self):
        # [FUNCTION DOC] Re-raising the caught exception itself does not make it its own
        # context.
        # Self-cycles (when re-raising a caught exception) are broken
        try:
            try:
                1/0
            except ZeroDivisionError as e:
                raise e
        except ZeroDivisionError as e:
            self.assertIsNone(e.__context__)

    def test_reraise_cycle_broken(self):
        # [FUNCTION DOC] Re-raising an earlier exception through a longer chain breaks the
        # context cycle.
        # Non-trivial context cycles (through re-raising a previous exception)
        # are broken too.
        try:
            try:
                xyzzy
            except NameError as a:
                try:
                    1/0
                except ZeroDivisionError:
                    raise a
        except NameError as e:
            self.assertIsNone(e.__context__.__context__)

    def test_not_last(self):
        # [FUNCTION DOC] The context is the exception being handled, even if other exceptions
        # were raised and caught afterwards.
        # Context is not necessarily the last exception
        context = Exception("context")
        try:
            raise context
        except Exception:
            try:
                raise Exception("caught")
            except Exception:
                pass
            try:
                raise Exception("new")
            except Exception as exc:
                raised = exc
        self.assertIs(raised.__context__, context)

    def test_3118(self):
        # [FUNCTION DOC] Issue 3118: deleting a generator in a handler must not clear the
        # context of the exception raised afterwards.
        # deleting the generator caused the __context__ to be cleared
        def gen():
            # [FUNCTION DOC] HELPER: Generator that yields 1 inside try/finally.
            try:
                yield 1
            finally:
                pass

        def f():
            # [FUNCTION DOC] HELPER: Advances the generator,
            # deletes it in a handler, raises KeyError
            # and checks its context is the ValueError.
            g = gen()
            next(g)
            try:
                try:
                    raise ValueError
                except:
                    del g
                    raise KeyError
            except Exception as e:
                self.assertIsInstance(e.__context__, ValueError)

        f()

    def test_3611(self):
        # [FUNCTION DOC] Issue 3611: an exception raised and re-raised in a __del__ triggered
        # inside a handler must not clear the context; the unraisable exception reported
        # must be ZeroDivisionError.
        import gc
        # A re-raised exception in a __del__ caused the __context__
        # to be cleared
        class C:
            # [CLASS DOC] HELPER: Class whose finalizer divides by zero and re-raises.
            def __del__(self):
                # [FUNCTION DOC] HELPER: Finalizer; raises ZeroDivisionError and re-raises it.
                try:
                    1/0
                except:
                    raise

        def f():
            # [FUNCTION DOC] HELPER: Triggers the finalizer inside a handler, then raises TypeError
            # and checks the context is the AttributeError.
            x = C()
            try:
                try:
                    f.x
                except AttributeError:
                    # make x.__del__ trigger
                    del x
                    gc.collect()  # For PyPy or other GCs.
                    raise TypeError
            except Exception as e:
                self.assertNotEqual(e.__context__, None)
                self.assertIsInstance(e.__context__, AttributeError)

        with support.catch_unraisable_exception() as cm:
            f()

            self.assertEqual(ZeroDivisionError, cm.unraisable.exc_type)


class TestRemovedFunctionality(__TestCase):
    # [CLASS DOC] Tests that old raise forms no longer work.
    def test_tuples(self):
        # [FUNCTION DOC] Raising a tuple of classes must raise TypeError.
        try:
            raise (IndexError, KeyError) # This should be a tuple!
        except TypeError:
            pass
        else:
            self.fail("No exception raised")

    def test_strings(self):
        # [FUNCTION DOC] Raising a string must raise TypeError.
        try:
            raise "foo"
        except TypeError:
            pass
        else:
            self.fail("No exception raised")


if __name__ == "__main__":
    run_tests()
