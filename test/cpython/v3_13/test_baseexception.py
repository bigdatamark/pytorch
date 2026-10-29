# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# [MODULE DOC] Exception-hierarchy tests copied from CPython 3.13
# Lib/test/test_baseexception.py, run under CPythonTestCase. Contains ExceptionClassTests
# (hierarchy matches exception_hierarchy.txt, constructor/str/repr interface, __setstate__
# refcount safety) and UsageTests (raising or catching non-exception objects fails with
# TypeError). Import-time side effect: inserts a RedirectImportFinder at the front of
# sys.meta_path so listed test.* imports resolve to same-named top-level modules.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_baseexception.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests, TEST_WITH_TORCHDYNAMO

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
    # [CLASS DOC] Meta path finder that redirects selected "test.X" imports (see redirect_imports)
    # to the top-level module "X". Stateless; installed once at import time into sys.meta_path.
    def find_spec(self, fullname, path, target=None):
        # [FUNCTION DOC] Finder hook. If fullname is in redirect_imports, imports the module
        # named without the "test." prefix, registers it in sys.modules under fullname and
        # returns that module's spec; returns None (defer to other finders) if the name is
        # not listed or the import fails with ImportError. Side effect: mutates sys.modules.
        # KNOWN ISSUE: uses importlib.util without importing it explicitly (only
        # importlib.abc is imported), relying on it being loaded elsewhere.
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

import unittest
import builtins
import os
from platform import system as platform_system


class ExceptionClassTests(CPythonTestCase):
    # [CLASS DOC] Tests about exception classes themselves: inheritance
    # hierarchy against the documented tree file, and the args/str/repr
    # interface for zero, one and many constructor arguments.

    """Tests for anything relating to exception objects themselves (e.g.,
    inheritance hierarchy)"""

    def test_builtins_new_style(self):
        # [FUNCTION DOC] Checks that Exception is a subclass of object.
        self.assertTrue(issubclass(Exception, object))

    def verify_instance_interface(self, ins):
        # [FUNCTION DOC] HELPER: Asserts the given exception instance has the attributes
        # args, __str__ and __repr__; the failure message names the instance's class.
        for attr in ("args", "__str__", "__repr__"):
            self.assertTrue(hasattr(ins, attr),
                    "%s missing %s attribute" %
                        (ins.__class__.__name__, attr))

    def test_inheritance(self):
        # [FUNCTION DOC] Reads exception_hierarchy.txt next to this file (first line is
        # the root class name, later lines are a depth-indented tree), and walks it
        # while checking that every named exception exists in builtins, is a subclass
        # of its parent in the tree, has the standard interface when constructible
        # without arguments, and that every built-in exception is listed exactly once
        # (platform-specific entries are only kept on the matching platform). Closes
        # the file in all cases. Fails if some built-in exception is not accounted for.
        # Make sure the inheritance hierarchy matches the documentation
        exc_set = set()
        for object_ in builtins.__dict__.values():
            try:
                if issubclass(object_, BaseException):
                    exc_set.add(object_.__name__)
            except TypeError:
                pass

        inheritance_tree = open(
                os.path.join(os.path.split(__file__)[0], 'exception_hierarchy.txt'),
                encoding="utf-8")
        try:
            superclass_name = inheritance_tree.readline().rstrip()
            try:
                last_exc = getattr(builtins, superclass_name)
            except AttributeError:
                self.fail("base class %s not a built-in" % superclass_name)
            self.assertIn(superclass_name, exc_set,
                          '%s not found' % superclass_name)
            exc_set.discard(superclass_name)
            superclasses = []  # Loop will insert base exception
            last_depth = 0
            for exc_line in inheritance_tree:
                exc_line = exc_line.rstrip()
                depth = exc_line.rindex('─')
                exc_name = exc_line[depth+2:]  # Slice past space
                if '(' in exc_name:
                    paren_index = exc_name.index('(')
                    platform_name = exc_name[paren_index+1:-1]
                    exc_name = exc_name[:paren_index-1]  # Slice off space
                    if platform_system() != platform_name:
                        exc_set.discard(exc_name)
                        continue
                if '[' in exc_name:
                    left_bracket = exc_name.index('[')
                    exc_name = exc_name[:left_bracket-1]  # cover space
                try:
                    exc = getattr(builtins, exc_name)
                except AttributeError:
                    self.fail("%s not a built-in exception" % exc_name)
                if last_depth < depth:
                    superclasses.append((last_depth, last_exc))
                elif last_depth > depth:
                    while superclasses[-1][0] >= depth:
                        superclasses.pop()
                self.assertTrue(issubclass(exc, superclasses[-1][1]),
                "%s is not a subclass of %s" % (exc.__name__,
                    superclasses[-1][1].__name__))
                try:  # Some exceptions require arguments; just skip them
                    self.verify_instance_interface(exc())
                except TypeError:
                    pass
                self.assertIn(exc_name, exc_set)
                exc_set.discard(exc_name)
                last_exc = exc
                last_depth = depth
        finally:
            inheritance_tree.close()
        self.assertEqual(len(exc_set), 0, "%s not accounted for" % exc_set)

    interface_tests = ("length", "args", "str", "repr")

    def interface_test_driver(self, results):
        # [FUNCTION DOC] HELPER: Compares each (given, expected) pair of
        # results against the names in interface_tests (length, args,
        # str, repr) and asserts equality, reporting the failing aspect.
        for test_name, (given, expected) in zip(self.interface_tests, results):
            self.assertEqual(given, expected, "%s: %s != %s" % (test_name,
                given, expected))

    def test_interface_single_arg(self):
        # [FUNCTION DOC] Builds an Exception with one argument and
        # checks the length of args, the argument itself, str and repr.
        # Make sure interface works properly when given a single argument
        arg = "spam"
        exc = Exception(arg)
        results = ([len(exc.args), 1], [exc.args[0], arg],
                   [str(exc), str(arg)],
            [repr(exc), '%s(%r)' % (exc.__class__.__name__, arg)])
        self.interface_test_driver(results)

    def test_interface_multi_arg(self):
        # [FUNCTION DOC] Builds an Exception with three arguments
        # and checks the length of args, args, str and repr.
        # Make sure interface correct when multiple arguments given
        arg_count = 3
        args = tuple(range(arg_count))
        exc = Exception(*args)
        results = ([len(exc.args), arg_count], [exc.args, args],
                [str(exc), str(args)],
                [repr(exc), exc.__class__.__name__ + repr(exc.args)])
        self.interface_test_driver(results)

    def test_interface_no_arg(self):
        # [FUNCTION DOC] Builds an Exception with no arguments
        # and checks the empty args, empty string and bare repr.
        # Make sure that with no args that interface is correct
        exc = Exception()
        results = ([len(exc.args), 0], [exc.args, tuple()],
                [str(exc), ''],
                [repr(exc), exc.__class__.__name__ + '()'])
        self.interface_test_driver(results)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    def test_setstate_refcount_no_crash(self):
        # [FUNCTION DOC] Regression check for a refcount bug: creates a str subclass key
        # whose hash clears the dict that holds it, then calls __setstate__ with that dict
        # and runs the garbage collector; passes if the interpreter does not crash. Skipped
        # under TorchDynamo because of unsupported class creation with closed-over objects.
        # gh-97591: Acquire strong reference before calling tp_hash slot
        # in PyObject_SetAttr.
        import gc
        d = {}
        class HashThisKeyWillClearTheDict(str):
            # [CLASS DOC] HELPER: String subclass whose hash
            # empties the enclosing dict d as a side effect.
            def __hash__(self) -> int:
                # [FUNCTION DOC] HELPER: Clears d and returns the normal string hash.
                d.clear()
                return super().__hash__()
        class Value(str):
            # [CLASS DOC] HELPER: Plain string subclass used as the dict value.
            pass
        exc = Exception()

        d[HashThisKeyWillClearTheDict()] = Value()  # refcount of Value() is 1 now

        # Exception.__setstate__ should acquire a strong reference of key and
        # value in the dict. Otherwise, Value()'s refcount would go below
        # zero in the tp_hash call in PyObject_SetAttr(), and it would cause
        # crash in GC.
        exc.__setstate__(d)  # __hash__() is called again here, clearing the dict.

        # This GC would crash if the refcount of Value() goes below zero.
        gc.collect()


class UsageTests(CPythonTestCase):
    # [CLASS DOC] Tests how exception objects behave when raised
    # or caught: non-exceptions must be rejected with TypeError.

    """Test usage of exceptions"""

    def raise_fails(self, object_):
        # [FUNCTION DOC] HELPER: Raises the given object and asserts
        # that a TypeError results; fails the test if it does not.
        """Make sure that raising 'object_' triggers a TypeError."""
        try:
            raise object_
        except TypeError:
            return  # What is expected.
        self.fail("TypeError expected for raising %s" % type(object_))

    def catch_fails(self, object_):
        # [FUNCTION DOC] HELPER: Asserts that using the given object in an except clause, alone
        # and inside a tuple, makes handling an exception raise TypeError; fails the test
        # otherwise.
        """Catching 'object_' should raise a TypeError."""
        try:
            try:
                raise Exception
            except object_:
                pass
        except TypeError:
            pass
        except Exception:
            self.fail("TypeError expected when catching %s" % type(object_))

        try:
            try:
                raise Exception
            except (object_,):
                pass
        except TypeError:
            return
        except Exception:
            self.fail("TypeError expected when catching %s as specified in a "
                        "tuple" % type(object_))

    def test_raise_new_style_non_exception(self):
        # [FUNCTION DOC] Checks that raising a new-style class or
        # instance not derived from BaseException fails with TypeError.
        # You cannot raise a new-style class that does not inherit from
        # BaseException; the ability was not possible until BaseException's
        # introduction so no need to support new-style objects that do not
        # inherit from it.
        class NewStyleClass(object):
            # [CLASS DOC] HELPER: Plain class not derived from BaseException.
            pass
        self.raise_fails(NewStyleClass)
        self.raise_fails(NewStyleClass())

    def test_raise_string(self):
        # [FUNCTION DOC] Checks that raising a string fails with TypeError.
        # Raising a string raises TypeError.
        self.raise_fails("spam")

    def test_catch_non_BaseException(self):
        # [FUNCTION DOC] Checks that catching a class or instance
        # not derived from BaseException fails with TypeError.
        # Trying to catch an object that does not inherit from BaseException
        # is not allowed.
        class NonBaseException(object):
            # [CLASS DOC] HELPER: Plain class not derived from BaseException.
            pass
        self.catch_fails(NonBaseException)
        self.catch_fails(NonBaseException())

    def test_catch_BaseException_instance(self):
        # [FUNCTION DOC] Checks that catching an instance
        # of a BaseException subclass fails with TypeError.
        # Catching an instance of a BaseException subclass won't work.
        self.catch_fails(BaseException())

    def test_catch_string(self):
        # [FUNCTION DOC] Checks that catching a string fails with TypeError.
        # Catching a string is bad.
        self.catch_fails("spam")


if __name__ == "__main__":
    run_tests()
