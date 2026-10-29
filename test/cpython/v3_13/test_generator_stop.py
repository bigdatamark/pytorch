# [MODULE DOC] CPython 3.13 test_generator_stop adapted for the Dynamo CPython test suite.
# Enables the generator_stop future feature (PEP 479) and checks that a StopIteration
# escaping a generator body is converted to RuntimeError. Test cases derive from
# CPythonTestCase. Import-time side effect: installs RedirectImportFinder at the front of
# sys.meta_path, which maps a few "test.*" module names to their top-level equivalents.
from __future__ import generator_stop

# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_generator_stop.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

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
    # [CLASS DOC] HELPER: Meta path finder that redirects imports of names listed in
    # redirect_imports (test.X) to the standalone module X. Stateless; installed once
    # at module import time.
    def find_spec(self, fullname, path, target=None):
        # [FUNCTION DOC] HELPER: If fullname is in redirect_imports, imports the name with
        # the "test." prefix removed, registers it in sys.modules under fullname and
        # returns that module's spec. Returns None if not listed or the import fails
        # with ImportError, letting normal finders proceed.
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


class TestPEP479(__TestCase):
    # [CLASS DOC] Tests for PEP 479 behavior: StopIteration raised inside a generator
    # becomes RuntimeError. No state beyond the CPythonTestCase base.
    def test_stopiteration_wrapping(self):
        # [FUNCTION DOC] Advancing a generator whose body calls a function raising
        # StopIteration must raise RuntimeError with message "generator raised StopIteration".
        def f():
            # [FUNCTION DOC] HELPER: Always raises StopIteration.
            raise StopIteration
        def g():
            # [FUNCTION DOC] HELPER: Generator that yields the result of f, so it raises.
            yield f()
        with self.assertRaisesRegex(RuntimeError,
                                    "generator raised StopIteration"):
            next(g())

    def test_stopiteration_wrapping_context(self):
        # [FUNCTION DOC] Checks the RuntimeError produced from a generator's StopIteration
        # has the StopIteration as both __cause__ and __context__, with
        # __suppress_context__ set; fails the test otherwise.
        def f():
            # [FUNCTION DOC] HELPER: Always raises StopIteration.
            raise StopIteration
        def g():
            # [FUNCTION DOC] HELPER: Generator that yields the result of f, so it raises.
            yield f()

        try:
            next(g())
        except RuntimeError as exc:
            self.assertIs(type(exc.__cause__), StopIteration)
            self.assertIs(type(exc.__context__), StopIteration)
            self.assertTrue(exc.__suppress_context__)
        else:
            self.fail('__cause__, __context__, or __suppress_context__ '
                      'were not properly set')


if __name__ == "__main__":
    run_tests()
