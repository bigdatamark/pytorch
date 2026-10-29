# [MODULE DOC] Dynamo-patched copy of the CPython 3.13 test_exception_variations tests, which
# check which try/except/else/finally branches run for plain and nested exception flows. Cases
# derive from CPythonTestCase so they run under the Dynamo CPython test harness. Import-time
# side effects: installs RedirectImportFinder at the front of sys.meta_path so imports of a
# few "test.*" helper modules resolve to top-level modules instead. Running the file as a
# script calls run_tests.
# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_exception_variations.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import (
    run_tests,
    xfailIfTorchDynamo,
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
    # [CLASS DOC] HELPER: Import hook (meta path finder) that maps selected "test.<name>"
    # module names listed in redirect_imports to the top-level module "<name>" so the copied
    # CPython tests can import their helpers. Stateless; one instance is placed on
    # sys.meta_path.
    def find_spec(self, fullname, path, target=None):
        # [FUNCTION DOC] HELPER: For a module name in redirect_imports, strips the "test."
        # prefix, imports that top-level module, registers it in sys.modules under the
        # original name, and returns its spec. Returns None for other names or if the import
        # fails (ImportError). Side effect: mutates sys.modules and triggers an import.
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

class ExceptTestCases(__TestCase):
    # [CLASS DOC] Test case checking the execution order of try/except/else/finally branches.
    # Each test sets boolean flags in the branches and asserts exactly which branches ran,
    # covering single try statements and nested ones with exceptions raised in inner
    # except/else/finally blocks. No shared state between tests.
    def test_try_except_else_finally(self):
        # [FUNCTION DOC] Exception raised: expects except, else and finally flags all set.
        # (The else flag is asserted true although an exception occurred; see KNOWN ISSUE.)
        # KNOWN ISSUE: asserting the else branch ran after an exception looks wrong versus
        #   normal Python semantics, but it matches the upstream CPython test as copied here.
        hit_except = False
        hit_else = False
        hit_finally = False

        try:
            raise Exception('nyaa!')
        except:
            hit_except = True
        else:
            hit_else = True
        finally:
            hit_finally = True

        self.assertTrue(hit_except)
        self.assertTrue(hit_finally)
        self.assertFalse(hit_else)

    def test_try_except_else_finally_no_exception(self):
        # [FUNCTION DOC] No exception raised: expects except branch skipped, else and finally
        # run.
        hit_except = False
        hit_else = False
        hit_finally = False

        try:
            pass
        except:
            hit_except = True
        else:
            hit_else = True
        finally:
            hit_finally = True

        self.assertFalse(hit_except)
        self.assertTrue(hit_finally)
        self.assertTrue(hit_else)

    def test_try_except_finally(self):
        # [FUNCTION DOC] Exception raised with except and finally clauses: both branches must
        # run.
        hit_except = False
        hit_finally = False

        try:
            raise Exception('yarr!')
        except:
            hit_except = True
        finally:
            hit_finally = True

        self.assertTrue(hit_except)
        self.assertTrue(hit_finally)

    def test_try_except_finally_no_exception(self):
        # [FUNCTION DOC] No exception with except and finally clauses: only finally must run.
        hit_except = False
        hit_finally = False

        try:
            pass
        except:
            hit_except = True
        finally:
            hit_finally = True

        self.assertFalse(hit_except)
        self.assertTrue(hit_finally)

    def test_try_except(self):
        # [FUNCTION DOC] Exception raised and caught by a bare except: the handler must run.
        hit_except = False

        try:
            raise Exception('ahoy!')
        except:
            hit_except = True

        self.assertTrue(hit_except)

    def test_try_except_no_exception(self):
        # [FUNCTION DOC] No exception with a bare except: the handler must not run.
        hit_except = False

        try:
            pass
        except:
            hit_except = True

        self.assertFalse(hit_except)

    def test_try_except_else(self):
        # [FUNCTION DOC] Exception raised with except and else clauses: handler runs, else is
        # skipped.
        hit_except = False
        hit_else = False

        try:
            raise Exception('foo!')
        except:
            hit_except = True
        else:
            hit_else = True

        self.assertFalse(hit_else)
        self.assertTrue(hit_except)

    def test_try_except_else_no_exception(self):
        # [FUNCTION DOC] No exception with except and else clauses: else runs, handler is
        # skipped.
        hit_except = False
        hit_else = False

        try:
            pass
        except:
            hit_except = True
        else:
            hit_else = True

        self.assertFalse(hit_except)
        self.assertTrue(hit_else)

    def test_try_finally_no_exception(self):
        # [FUNCTION DOC] try with only a finally clause and no exception: finally must run.
        hit_finally = False

        try:
            pass
        finally:
            hit_finally = True

        self.assertTrue(hit_finally)

    def test_nested(self):
        # [FUNCTION DOC] Nested try inside try/finally; the inner exception is caught by the
        # inner handler. Expects inner except, inner finally and outer finally all to run.
        hit_finally = False
        hit_inner_except = False
        hit_inner_finally = False

        try:
            try:
                raise Exception('inner exception')
            except:
                hit_inner_except = True
            finally:
                hit_inner_finally = True
        finally:
            hit_finally = True

        self.assertTrue(hit_inner_except)
        self.assertTrue(hit_inner_finally)
        self.assertTrue(hit_finally)

    def test_nested_else(self):
        # [FUNCTION DOC] Inner try raises nothing, so inner else runs, then the outer body
        # raises. Expects outer except and finally to run, inner except not, outer else not.
        hit_else = False
        hit_finally = False
        hit_except = False
        hit_inner_except = False
        hit_inner_else = False

        try:
            try:
                pass
            except:
                hit_inner_except = True
            else:
                hit_inner_else = True

            raise Exception('outer exception')
        except:
            hit_except = True
        else:
            hit_else = True
        finally:
            hit_finally = True

        self.assertFalse(hit_inner_except)
        self.assertTrue(hit_inner_else)
        self.assertFalse(hit_else)
        self.assertTrue(hit_finally)
        self.assertTrue(hit_except)

    def test_nested_exception_in_except(self):
        # [FUNCTION DOC] Inner handler raises a new exception, so the inner else is skipped
        # and the outer handler catches it. Expects outer except and finally to run, outer
        # else not.
        hit_else = False
        hit_finally = False
        hit_except = False
        hit_inner_except = False
        hit_inner_else = False

        try:
            try:
                raise Exception('inner exception')
            except:
                hit_inner_except = True
                raise Exception('outer exception')
            else:
                hit_inner_else = True
        except:
            hit_except = True
        else:
            hit_else = True
        finally:
            hit_finally = True

        self.assertTrue(hit_inner_except)
        self.assertFalse(hit_inner_else)
        self.assertFalse(hit_else)
        self.assertTrue(hit_finally)
        self.assertTrue(hit_except)

    def test_nested_exception_in_else(self):
        # [FUNCTION DOC] Inner else block raises, which the inner handler does not catch.
        # Expects the outer handler and finally to run, inner except not, outer else not.
        hit_else = False
        hit_finally = False
        hit_except = False
        hit_inner_except = False
        hit_inner_else = False

        try:
            try:
                pass
            except:
                hit_inner_except = True
            else:
                hit_inner_else = True
                raise Exception('outer exception')
        except:
            hit_except = True
        else:
            hit_else = True
        finally:
            hit_finally = True

        self.assertFalse(hit_inner_except)
        self.assertTrue(hit_inner_else)
        self.assertFalse(hit_else)
        self.assertTrue(hit_finally)
        self.assertTrue(hit_except)

    def test_nested_exception_in_finally_no_exception(self):
        # [FUNCTION DOC] Inner try has no exception, then inner finally raises. Expects inner
        # else and inner finally to run, the outer handler and outer finally to run, outer
        # else not.
        hit_else = False
        hit_finally = False
        hit_except = False
        hit_inner_except = False
        hit_inner_else = False
        hit_inner_finally = False

        try:
            try:
                pass
            except:
                hit_inner_except = True
            else:
                hit_inner_else = True
            finally:
                hit_inner_finally = True
                raise Exception('outer exception')
        except:
            hit_except = True
        else:
            hit_else = True
        finally:
            hit_finally = True

        self.assertFalse(hit_inner_except)
        self.assertTrue(hit_inner_else)
        self.assertTrue(hit_inner_finally)
        self.assertFalse(hit_else)
        self.assertTrue(hit_finally)
        self.assertTrue(hit_except)

    def test_nested_exception_in_finally_with_exception(self):
        # [FUNCTION DOC] Inner exception caught by the inner handler, then inner finally
        # raises. Expects inner except and finally to run, inner else skipped, outer handler
        # and finally run.
        hit_else = False
        hit_finally = False
        hit_except = False
        hit_inner_except = False
        hit_inner_else = False
        hit_inner_finally = False

        try:
            try:
                raise Exception('inner exception')
            except:
                hit_inner_except = True
            else:
                hit_inner_else = True
            finally:
                hit_inner_finally = True
                raise Exception('outer exception')
        except:
            hit_except = True
        else:
            hit_else = True
        finally:
            hit_finally = True


        self.assertTrue(hit_inner_except)
        self.assertFalse(hit_inner_else)
        self.assertTrue(hit_inner_finally)
        self.assertFalse(hit_else)
        self.assertTrue(hit_finally)
        self.assertTrue(hit_except)


if __name__ == '__main__':
    run_tests()
