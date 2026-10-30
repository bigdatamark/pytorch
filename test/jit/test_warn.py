# Owner(s): ["oncall: jit"]
# [MODULE DOC] JIT tests for how torch.jit.script lowers Python warnings. Each test scripts a
# function that calls warnings.warn (directly, in a loop, or via helper functions), runs it while
# stderr is redirected into a buffer, and uses FileCheck to count how many times the warning text
# appears. Running the file directly raises an error pointing at test/test_jit.py.

import io
import os
import sys
import warnings
from contextlib import redirect_stderr

import torch
from torch.testing import FileCheck


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestWarn(JitTestCase):
    # [CLASS DOC] Test case for TorchScript warning semantics: a warning at a given source site is
    # emitted once per call of the scripted function, not once per loop iteration, and distinct
    # call sites or distinct callers each produce their own warning.
    def test_warn(self):
        # [FUNCTION DOC] Scripts a function that warns once, calls it with stderr captured, and
        # checks the captured text has exactly one occurrence of the warning message.
        @torch.jit.script
        def fn():
            # [FUNCTION DOC] HELPER: Scripted function issuing one fixed warning.
            warnings.warn("I am warning you")

        f = io.StringIO()
        with redirect_stderr(f):
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=1, exactly=True
        ).run(f.getvalue())

    def test_warn_only_once(self):
        # [FUNCTION DOC] Scripts a function that warns inside a ten-iteration loop, calls it once,
        # and checks the captured stderr has exactly one occurrence of the message (loop does not
        # repeat it).
        @torch.jit.script
        def fn():
            # [FUNCTION DOC] HELPER: Scripted function warning in each loop iteration.
            for _ in range(10):
                warnings.warn("I am warning you")

        f = io.StringIO()
        with redirect_stderr(f):
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=1, exactly=True
        ).run(f.getvalue())

    def test_warn_only_once_in_loop_func(self):
        # [FUNCTION DOC] Same as the loop test but the warning is raised inside a plain helper
        # function called in the scripted loop; checks that exactly one message is captured after
        # one call.
        def w():
            # [FUNCTION DOC] HELPER: Plain function (compiled recursively) that warns.
            warnings.warn("I am warning you")

        @torch.jit.script
        def fn():
            # [FUNCTION DOC] HELPER: Scripted function calling the warning helper in a
            # ten-iteration loop.
            for _ in range(10):
                w()

        f = io.StringIO()
        with redirect_stderr(f):
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=1, exactly=True
        ).run(f.getvalue())

    def test_warn_once_per_func(self):
        # [FUNCTION DOC] Scripts a function that calls two different helper functions that each
        # warn with the same text; checks the captured stderr has exactly two occurrences, one
        # per helper.
        def w1():
            # [FUNCTION DOC] HELPER: First helper that issues the warning.
            warnings.warn("I am warning you")

        def w2():
            # [FUNCTION DOC] HELPER: Second helper that issues the same warning from a
            # different site.
            warnings.warn("I am warning you")

        @torch.jit.script
        def fn():
            # [FUNCTION DOC] HELPER: Scripted function that calls both helpers once.
            w1()
            w2()

        f = io.StringIO()
        with redirect_stderr(f):
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=2, exactly=True
        ).run(f.getvalue())

    def test_warn_once_per_func_in_loop(self):
        # [FUNCTION DOC] Like the once-per-function test but both helpers are called in a
        # ten-iteration loop; checks that exactly two occurrences are captured (still one per
        # helper site).
        def w1():
            # [FUNCTION DOC] HELPER: First helper that issues the warning.
            warnings.warn("I am warning you")

        def w2():
            # [FUNCTION DOC] HELPER: Second helper that issues the same warning from a
            # different site.
            warnings.warn("I am warning you")

        @torch.jit.script
        def fn():
            # [FUNCTION DOC] HELPER: Scripted function that calls both helpers in a
            # ten-iteration loop.
            for _ in range(10):
                w1()
                w2()

        f = io.StringIO()
        with redirect_stderr(f):
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=2, exactly=True
        ).run(f.getvalue())

    def test_warn_multiple_calls_multiple_warnings(self):
        # [FUNCTION DOC] Calls the same scripted warning function twice with stderr captured and
        # checks that two occurrences of the message appear (each call warns again).
        @torch.jit.script
        def fn():
            # [FUNCTION DOC] HELPER: Scripted function issuing one fixed warning.
            warnings.warn("I am warning you")

        f = io.StringIO()
        with redirect_stderr(f):
            fn()
            fn()

        FileCheck().check_count(
            str="UserWarning: I am warning you", count=2, exactly=True
        ).run(f.getvalue())

    def test_warn_multiple_calls_same_func_diff_stack(self):
        # [FUNCTION DOC] Scripts two functions that call one shared warning helper with different
        # caller names, runs both, and checks the output contains exactly one warning mentioning
        # each caller, i.e. one helper site reached via different call stacks warns separately.
        def warn(caller: str):
            # [FUNCTION DOC] HELPER: Plain helper taking a caller name string; warns with a message
            # that ends with that name.
            warnings.warn("I am warning you from " + caller)

        @torch.jit.script
        def foo():
            # [FUNCTION DOC] HELPER: Scripted function calling the warning helper for foo.
            warn("foo")

        @torch.jit.script
        def bar():
            # [FUNCTION DOC] HELPER: Scripted function calling the warning helper for bar.
            warn("bar")

        f = io.StringIO()
        with redirect_stderr(f):
            foo()
            bar()

        FileCheck().check_count(
            str="UserWarning: I am warning you from foo",
            count=1,
            exactly=True,
        ).check_count(
            str="UserWarning: I am warning you from bar",
            count=1,
            exactly=True,
        ).run(f.getvalue())


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
