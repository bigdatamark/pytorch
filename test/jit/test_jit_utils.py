# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for JIT utility helpers: argument name extraction from callables
# (get_callable_argument_names), the script-raises-regex checker, and the
# NoTracerWarnContextManager. Not runnable directly: running the file as a script raises an
# error that points to the umbrella test runner.

import os
import sys
from textwrap import dedent

import torch
from torch.testing._internal import jit_utils


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


# Tests various JIT-related utility functions.
class TestJitUtils(JitTestCase):
    # [CLASS DOC] JitTestCase grouping tests of small JIT helper utilities.
    # Tests that POSITIONAL_OR_KEYWORD arguments are captured.
    def test_get_callable_argument_names_positional_or_keyword(self):
        # [FUNCTION DOC] Checks that for a function with only ordinary positional-or-keyword
        # parameters, the helper returns all their names in order.
        def fn_positional_or_keyword_args_only(x, y):
            # [FUNCTION DOC] HELPER: Sample function with two ordinary parameters returning
            # their sum.
            return x + y

        self.assertEqual(
            ["x", "y"],
            torch._jit_internal.get_callable_argument_names(
                fn_positional_or_keyword_args_only
            ),
        )

    # Tests that POSITIONAL_ONLY arguments are ignored.
    def test_get_callable_argument_names_positional_only(self):
        # [FUNCTION DOC] Builds a function with a positional-only parameter followed by an
        # ordinary one from source text and checks the helper returns only the ordinary
        # parameter name.
        code = dedent(
            """
            def fn_positional_only_arg(x, /, y):
                return x + y
        """
        )

        fn_positional_only_arg = jit_utils._get_py3_code(code, "fn_positional_only_arg")
        self.assertEqual(
            ["y"],
            torch._jit_internal.get_callable_argument_names(fn_positional_only_arg),
        )

    # Tests that VAR_POSITIONAL arguments are ignored.
    def test_get_callable_argument_names_var_positional(self):
        # [FUNCTION DOC] Checks that a variadic positional parameter is skipped and only the
        # ordinary parameter name is returned.
        # Tests that VAR_POSITIONAL arguments are ignored.
        def fn_var_positional_arg(x, *arg):
            # [FUNCTION DOC] HELPER: Sample function with one ordinary parameter and a
            # star-args parameter.
            return x + arg[0]

        self.assertEqual(
            ["x"],
            torch._jit_internal.get_callable_argument_names(fn_var_positional_arg),
        )

    # Tests that KEYWORD_ONLY arguments are ignored.
    def test_get_callable_argument_names_keyword_only(self):
        # [FUNCTION DOC] Checks that a keyword-only parameter is skipped and only the ordinary
        # parameter name is returned.
        def fn_keyword_only_arg(x, *, y):
            # [FUNCTION DOC] HELPER: Sample function with one ordinary parameter and one
            # keyword-only parameter.
            return x + y

        self.assertEqual(
            ["x"], torch._jit_internal.get_callable_argument_names(fn_keyword_only_arg)
        )

    # Tests that VAR_KEYWORD arguments are ignored.
    def test_get_callable_argument_names_var_keyword(self):
        # [FUNCTION DOC] Checks that a function with only a variadic keyword parameter yields an
        # empty name list.
        def fn_var_keyword_arg(**args):
            # [FUNCTION DOC] HELPER: Sample function taking only variadic keyword arguments and
            # reading two entries.
            return args["x"] + args["y"]

        self.assertEqual(
            [], torch._jit_internal.get_callable_argument_names(fn_var_keyword_arg)
        )

    # Tests that a function signature containing various different types of
    # arguments are ignored.
    def test_get_callable_argument_names_hybrid(self):
        # [FUNCTION DOC] Builds from source text a function mixing positional-only, ordinary,
        # star-args and variadic keyword parameters and checks the helper returns just the
        # ordinary parameter name.
        code = dedent(
            """
            def fn_hybrid_args(x, /, y, *args, **kwargs):
                return x + y + args[0] + kwargs['z']
        """
        )
        fn_hybrid_args = jit_utils._get_py3_code(code, "fn_hybrid_args")
        self.assertEqual(
            ["y"], torch._jit_internal.get_callable_argument_names(fn_hybrid_args)
        )

    def test_checkscriptassertraisesregex(self):
        # [FUNCTION DOC] Uses checkScriptRaisesRegex with a Python function and then with
        # function source text, both indexing a two-element tuple out of range, and expects an
        # exception whose message matches the range pattern.
        def fn():
            # [FUNCTION DOC] HELPER: Function that builds a two-element tuple and returns its
            # element at index two, which is out of range.
            tup = (1, 2)
            return tup[2]

        self.checkScriptRaisesRegex(fn, (), Exception, "range", name="fn")

        s = dedent(
            """
        def fn():
            tup = (1, 2)
            return tup[2]
        """
        )

        self.checkScriptRaisesRegex(s, (), Exception, "range", name="fn")

    def test_no_tracer_warn_context_manager(self):
        # [FUNCTION DOC] Sets the tracer-state warning flag on, checks it reads off inside the
        # no-tracer-warn context manager and on again after exit.
        torch._C._jit_set_tracer_state_warn(True)
        with jit_utils.NoTracerWarnContextManager():
            self.assertEqual(False, torch._C._jit_get_tracer_state_warn())
        self.assertEqual(True, torch._C._jit_get_tracer_state_warn())


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
