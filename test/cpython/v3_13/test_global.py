# [MODULE DOC] CPython 3.13 test_global adapted for Dynamo: verifies that compiling source
# text in which a name is used or assigned before its global declaration inside a function is
# a SyntaxError at the expected line and column, while a module-level global is fine. Defines
# GlobalTests and setUpModule, which turns warnings from the compiled test string into errors
# for the whole module.
# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_global.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

"""Verify that warnings are issued for global statements following use."""

from test.support import check_syntax_error
from test.support.warnings_helper import check_warnings
import unittest
import warnings


class GlobalTests(CPythonTestCase):
    # [CLASS DOC] Compile-only tests of global statement placement rules. setUp enters a
    # check_warnings context and makes warnings from the compiled test string errors.

    def setUp(self):
        # [FUNCTION DOC] Enters check_warnings through the test's context stack and installs
        # an error filter for warnings attributed to the compiled test string.
        super().setUp()
        self.enterContext(check_warnings())
        warnings.filterwarnings("error", module="<test string>")

    def test1(self):
        # [FUNCTION DOC] Source assigns names then declares them global; expects a SyntaxError
        # at line 4, offset 5.
        prog_text_1 = """\
def wrong1():
    a = 1
    b = 2
    global a
    global b
"""
        check_syntax_error(self, prog_text_1, lineno=4, offset=5)

    def test2(self):
        # [FUNCTION DOC] Source reads a name then declares it global; expects SyntaxError at
        # line 3, offset 5.
        prog_text_2 = """\
def wrong2():
    print(x)
    global x
"""
        check_syntax_error(self, prog_text_2, lineno=3, offset=5)

    def test3(self):
        # [FUNCTION DOC] Source reads, assigns, then declares a name global; expects
        # SyntaxError at line 4, offset 5.
        prog_text_3 = """\
def wrong3():
    print(x)
    x = 2
    global x
"""
        check_syntax_error(self, prog_text_3, lineno=4, offset=5)

    def test4(self):
        # [FUNCTION DOC] Module-level global before assignment compiles without error or
        # warning.
        prog_text_4 = """\
global x
x = 2
"""
        # this should work
        compile(prog_text_4, "<test string>", "exec")


def setUpModule():
    # [FUNCTION DOC] HELPER: Module fixture entering a warnings.catch_warnings context for the
    # module lifetime and turning warnings from the compiled test string into errors.
    unittest.enterModuleContext(warnings.catch_warnings())
    warnings.filterwarnings("error", module="<test string>")


if __name__ == "__main__":
    run_tests()
