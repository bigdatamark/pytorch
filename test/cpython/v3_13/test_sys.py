# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa
# [MODULE DOC] Dynamo-adapted copy of CPython 3.13 test_sys: exercises the sys module (displayhook
# and excepthook, exit, recursion limit, switch interval, exc_info, frames, stdlib and path
# settings, flags and version info, size-of checks, monitoring, audit hooks, threads and sub-
# interpreters, stdlib module names, and similar). Import-time side effects: sets the alias
# __TestCase, installs a RedirectImportFinder at the front of sys.meta_path, and imports
# test.support helpers (interpreters may be None). Several tests spawn Python subprocesses.
# run_tests() runs under __main__.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_sys.py

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
    # [CLASS DOC] HELPER: Import hook redirecting selected "test.*" module names (see
    # redirect_imports) to same-named top-level modules. Stateless; only find_spec is used.
    def find_spec(self, fullname, path, target=None):
        # [FUNCTION DOC] HELPER: For a name in redirect_imports, imports the module without the
        # "test." prefix, registers it in sys.modules under the original name (side effect), and
        # returns its spec; returns None for other names or when that import fails with an
        # import error.
        # KNOWN ISSUE: uses importlib.util without importing it explicitly in this file.
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

import builtins
import codecs
import _datetime
import gc
import io
import locale
import operator
import os
import random
import struct
import subprocess
import sys
import sysconfig
import test.support
from test import support
from test.support import os_helper
from test.support.script_helper import assert_python_ok, assert_python_failure
from test.support import threading_helper
from test.support import import_helper
from test.support import force_not_colorized
try:
    from test.support import interpreters
except ImportError:
    interpreters = None
import textwrap
import unittest
import warnings


def requires_subinterpreters(meth):
    # [FUNCTION DOC] HELPER: Decorator that wraps a test method so it is skipped when the
    # interpreters module could not be imported. Returns the wrapped method.
    """Decorator to skip a test if subinterpreters are not supported."""
    return unittest.skipIf(interpreters is None,
                           'subinterpreters required')(meth)


DICT_KEY_STRUCT_FORMAT = 'n2BI2n'

class DisplayHookTest(__TestCase):
    # [CLASS DOC] Tests for the default display hook sys.__displayhook__ and the
    # replaceable sys.displayhook.

    def test_original_displayhook(self):
        # [FUNCTION DOC] Original display hook prints the repr of a value to stdout and stores it
        # in builtins under the underscore name; None prints nothing and leaves it unset; calling
        # without arguments raises TypeError; with sys.stdout deleted it raises RuntimeError
        # (stdout restored after).
        dh = sys.__displayhook__

        with support.captured_stdout() as out:
            dh(42)

        self.assertEqual(out.getvalue(), "42\n")
        self.assertEqual(builtins._, 42)

        del builtins._

        with support.captured_stdout() as out:
            dh(None)

        self.assertEqual(out.getvalue(), "")
        self.assertTrue(not hasattr(builtins, "_"))

        # sys.displayhook() requires arguments
        self.assertRaises(TypeError, dh)

        stdout = sys.stdout
        try:
            del sys.stdout
            self.assertRaises(RuntimeError, dh, 42)
        finally:
            sys.stdout = stdout

    @torch._dynamo.error_on_graph_break(False)
    def test_lost_displayhook(self):
        # [FUNCTION DOC] Deleting sys.displayhook then evaluating an interactive-mode compiled
        # expression raises RuntimeError; the hook is restored afterwards.
        displayhook = sys.displayhook
        try:
            del sys.displayhook
            code = compile("42", "<string>", "single")
            self.assertRaises(RuntimeError, eval, code)
        finally:
            sys.displayhook = displayhook

    @torch._dynamo.error_on_graph_break(False)
    def test_custom_displayhook(self):
        # [FUNCTION DOC] Swaps in a display hook that raises ValueError and checks the error
        # propagates out of evaluating an interactive-mode expression.
        def baddisplayhook(obj):
            # [FUNCTION DOC] HELPER: Display hook that always raises ValueError.
            raise ValueError

        with support.swap_attr(sys, 'displayhook', baddisplayhook):
            code = compile("42", "<string>", "single")
            self.assertRaises(ValueError, eval, code)


class ActiveExceptionTests(__TestCase):
    # [CLASS DOC] Tests of sys.exc_info and sys.exception inside and outside an except block.
    def test_exc_info_no_exception(self):
        # [FUNCTION DOC] Outside any handler sys.exc_info returns three None values.
        self.assertEqual(sys.exc_info(), (None, None, None))

    def test_sys_exception_no_exception(self):
        # [FUNCTION DOC] Outside any handler sys.exception returns None.
        self.assertEqual(sys.exception(), None)

    def test_exc_info_with_exception_instance(self):
        # [FUNCTION DOC] Inside an except block exc_info reports the exception type, instance
        # (raised with an argument) and the instance's traceback.
        def f():
            # [FUNCTION DOC] HELPER: Raises ValueError with the argument 42.
            raise ValueError(42)

        try:
            f()
        except Exception as e_:
            e = e_
            exc_info = sys.exc_info()

        self.assertIsInstance(e, ValueError)
        self.assertIs(exc_info[0], ValueError)
        self.assertIs(exc_info[1], e)
        self.assertIs(exc_info[2], e.__traceback__)

    def test_exc_info_with_exception_type(self):
        # [FUNCTION DOC] Same as the previous test but the exception is raised from the bare
        # class; exc_info still holds the class, a created instance and its traceback.
        def f():
            # [FUNCTION DOC] HELPER: Raises the ValueError class without an argument.
            raise ValueError

        try:
            f()
        except Exception as e_:
            e = e_
            exc_info = sys.exc_info()

        self.assertIsInstance(e, ValueError)
        self.assertIs(exc_info[0], ValueError)
        self.assertIs(exc_info[1], e)
        self.assertIs(exc_info[2], e.__traceback__)

    def test_sys_exception_with_exception_instance(self):
        # [FUNCTION DOC] Inside an except block sys.exception returns the very exception instance
        # that was raised.
        def f():
            # [FUNCTION DOC] HELPER: Raises ValueError with the argument 42.
            raise ValueError(42)

        try:
            f()
        except Exception as e_:
            e = e_
            exc = sys.exception()

        self.assertIsInstance(e, ValueError)
        self.assertIs(exc, e)

    def test_sys_exception_with_exception_type(self):
        # [FUNCTION DOC] Same as above for an exception raised from the bare class; sys.exception
        # returns the instance.
        def f():
            # [FUNCTION DOC] HELPER: Raises the ValueError class without an argument.
            raise ValueError

        try:
            f()
        except Exception as e_:
            e = e_
            exc = sys.exception()

        self.assertIsInstance(e, ValueError)
        self.assertIs(exc, e)


class ExceptHookTest(__TestCase):
    # [CLASS DOC] Tests for the default exception hook sys.__excepthook__ and sys.excepthook.

    @force_not_colorized
    def test_original_excepthook(self):
        # [FUNCTION DOC] Original excepthook prints the traceback ending in the error line to
        # captured stderr; calling it with no arguments raises TypeError.
        try:
            raise ValueError(42)
        except ValueError as exc:
            with support.captured_stderr() as err:
                sys.__excepthook__(*sys.exc_info())

        self.assertTrue(err.getvalue().endswith("ValueError: 42\n"))

        self.assertRaises(TypeError, sys.__excepthook__)

    @force_not_colorized
    def test_excepthook_bytes_filename(self):
        # [FUNCTION DOC] bpo-37467: the original excepthook must not crash on a SyntaxError whose
        # filename is a bytes object; checks the file/line, source text and final message in
        # stderr, with bytes warnings ignored.
        # bpo-37467: sys.excepthook() must not crash if a filename
        # is a bytes string
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', BytesWarning)

            try:
                raise SyntaxError("msg", (b"bytes_filename", 123, 0, "text"))
            except SyntaxError as exc:
                with support.captured_stderr() as err:
                    sys.__excepthook__(*sys.exc_info())

        err = err.getvalue()
        self.assertIn("""  File "b'bytes_filename'", line 123\n""", err)
        self.assertIn("""    text\n""", err)
        self.assertTrue(err.endswith("SyntaxError: msg\n"))

    def test_excepthook(self):
        # [FUNCTION DOC] Calling sys.excepthook with invalid arguments prints a TypeError about
        # expecting an exception to stderr; unraisable exceptions are captured and not propagated.
        with test.support.captured_output("stderr") as stderr:
            with test.support.catch_unraisable_exception():
                sys.excepthook(1, '1', 1)
        self.assertTrue("TypeError: print_exception(): Exception expected for " \
                         "value, str found" in stderr.getvalue())

    # FIXME: testing the code for a lost or replaced excepthook in
    # Python/pythonrun.c::PyErr_PrintEx() is tricky.


class SysModuleTest(__TestCase):
    # [CLASS DOC] Broad collection of tests of the sys module: exit, intervals, limits, frames,
    # flags and many other attributes. Provides tearDown that reaps child processes.

    def tearDown(self):
        # [FUNCTION DOC] Reaps any finished child processes left over by subprocess-based tests.
        test.support.reap_children()

    def test_exit(self):
        # [FUNCTION DOC] Checks sys.exit semantics: too many arguments raise TypeError; no
        # argument gives SystemExit with code None; integer, string, and tuple arguments are
        # stored on the exception (a one-element tuple is unpacked); exit codes from subprocesses
        # (0, large and overflowing values mapped to -1/0xff/0xffffffff, 47 via raising
        # SystemExit); and exit messages written to stderr after flushing, with backslash escaping
        # of surrogates and the stderr encoding.
        # call with two arguments
        self.assertRaises(TypeError, sys.exit, 42, 42)

        # call without argument
        with self.assertRaises(SystemExit) as cm:
            sys.exit()
        self.assertIsNone(cm.exception.code)

        rc, out, err = assert_python_ok('-c', 'import sys; sys.exit()')
        self.assertEqual(rc, 0)
        self.assertEqual(out, b'')
        self.assertEqual(err, b'')

        # gh-125842: Windows uses 32-bit unsigned integers for exit codes
        # so a -1 exit code is sometimes interpreted as 0xffff_ffff.
        rc, out, err = assert_python_failure('-c', 'import sys; sys.exit(0xffff_ffff)')
        self.assertIn(rc, (-1, 0xff, 0xffff_ffff))
        self.assertEqual(out, b'')
        self.assertEqual(err, b'')

        # Overflow results in a -1 exit code, which may be converted to 0xff
        # or 0xffff_ffff.
        rc, out, err = assert_python_failure('-c', 'import sys; sys.exit(2**128)')
        self.assertIn(rc, (-1, 0xff, 0xffff_ffff))
        self.assertEqual(out, b'')
        self.assertEqual(err, b'')

        # call with integer argument
        with self.assertRaises(SystemExit) as cm:
            sys.exit(42)
        self.assertEqual(cm.exception.code, 42)

        # call with tuple argument with one entry
        # entry will be unpacked
        with self.assertRaises(SystemExit) as cm:
            sys.exit((42,))
        self.assertEqual(cm.exception.code, 42)

        # call with string argument
        with self.assertRaises(SystemExit) as cm:
            sys.exit("exit")
        self.assertEqual(cm.exception.code, "exit")

        # call with tuple argument with two entries
        with self.assertRaises(SystemExit) as cm:
            sys.exit((17, 23))
        self.assertEqual(cm.exception.code, (17, 23))

        # test that the exit machinery handles SystemExits properly
        rc, out, err = assert_python_failure('-c', 'raise SystemExit(47)')
        self.assertEqual(rc, 47)
        self.assertEqual(out, b'')
        self.assertEqual(err, b'')

        def check_exit_message(code, expected, **env_vars):
            # [FUNCTION DOC] HELPER: Runs code in a failing subprocess and asserts return code 1,
            # empty stdout and stderr starting with the expected bytes; extra keyword arguments
            # become environment variables.
            rc, out, err = assert_python_failure('-c', code, **env_vars)
            self.assertEqual(rc, 1)
            self.assertEqual(out, b'')
            self.assertTrue(err.startswith(expected),
                "%s doesn't start with %s" % (ascii(err), ascii(expected)))

        # test that stderr buffer is flushed before the exit message is written
        # into stderr
        check_exit_message(
            r'import sys; sys.stderr.write("unflushed,"); sys.exit("message")',
            b"unflushed,message")

        # test that the exit message is written with backslashreplace error
        # handler to stderr
        check_exit_message(
            r'import sys; sys.exit("surrogates:\uDCFF")',
            b"surrogates:\\udcff")

        # test that the unicode message is encoded to the stderr encoding
        # instead of the default encoding (utf8)
        check_exit_message(
            r'import sys; sys.exit("h\xe9")',
            b"h\xe9", PYTHONIOENCODING='latin-1')

    @support.requires_subprocess()
    def test_exit_codes_under_repl(self):
        # [FUNCTION DOC] GH-129900: runs the interpreter reading a script from a temporary file
        # that calls exit functions or raises SystemExit with codes 0 and 123 and checks the
        # process return code matches. Requires subprocess support.
        # GH-129900: SystemExit, or things that raised it, didn't
        # get their return code propagated by the REPL
        import tempfile

        exit_ways = [
            "exit",
            "__import__('sys').exit",
            "raise SystemExit"
        ]

        for exitfunc in exit_ways:
            for return_code in (0, 123):
                with self.subTest(exitfunc=exitfunc, return_code=return_code):
                    with tempfile.TemporaryFile("w+") as stdin:
                        stdin.write(f"{exitfunc}({return_code})\n")
                        stdin.seek(0)
                        proc = subprocess.run([sys.executable], stdin=stdin)
                        self.assertEqual(proc.returncode, return_code)

    def test_getdefaultencoding(self):
        # [FUNCTION DOC] getdefaultencoding rejects arguments with TypeError and returns a str
        # (value not checked since the user could change it).
        self.assertRaises(TypeError, sys.getdefaultencoding, 42)
        # can't check more than the type, as the user might have changed it
        self.assertIsInstance(sys.getdefaultencoding(), str)

    # testing sys.settrace() is done in test_sys_settrace.py
    # testing sys.setprofile() is done in test_sys_setprofile.py

    def test_switchinterval(self):
        # [FUNCTION DOC] setswitchinterval rejects missing or non-numeric arguments (TypeError)
        # and non-positive values (ValueError); several valid values round-trip through the getter
        # and the original value is restored.
        self.assertRaises(TypeError, sys.setswitchinterval)
        self.assertRaises(TypeError, sys.setswitchinterval, "a")
        self.assertRaises(ValueError, sys.setswitchinterval, -1.0)
        self.assertRaises(ValueError, sys.setswitchinterval, 0.0)
        orig = sys.getswitchinterval()
        # sanity check
        self.assertTrue(orig < 0.5, orig)
        try:
            for n in 0.00001, 0.05, 3.0, orig:
                sys.setswitchinterval(n)
                self.assertAlmostEqual(sys.getswitchinterval(), n)
        finally:
            sys.setswitchinterval(orig)

    def test_getrecursionlimit(self):
        # [FUNCTION DOC] getrecursionlimit returns an int greater than 1 and rejects arguments
        # with TypeError.
        limit = sys.getrecursionlimit()
        self.assertIsInstance(limit, int)
        self.assertGreater(limit, 1)

        self.assertRaises(TypeError, sys.getrecursionlimit, 42)

    def test_setrecursionlimit(self):
        # [FUNCTION DOC] setrecursionlimit sets a new limit readable via the getter, raises
        # TypeError without arguments and ValueError for negative values; restores the old limit.
        old_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(10_005)
            self.assertEqual(sys.getrecursionlimit(), 10_005)

            self.assertRaises(TypeError, sys.setrecursionlimit)
            self.assertRaises(ValueError, sys.setrecursionlimit, -42)
        finally:
            sys.setrecursionlimit(old_limit)

    def test_recursionlimit_recovery(self):
        # [FUNCTION DOC] Skips under a trace function. For several small limits, sets the limit
        # and checks that infinite recursion raises RecursionError twice in a row (recovery);
        # limits too low for the current depth are skipped. Restores the old limit.
        if hasattr(sys, 'gettrace') and sys.gettrace():
            self.skipTest('fatal error if run with a trace function')

        old_limit = sys.getrecursionlimit()
        def f():
            # [FUNCTION DOC] HELPER: Recurses infinitely by calling itself.
            f()
        try:
            for depth in (50, 75, 100, 250, 1000):
                try:
                    sys.setrecursionlimit(depth)
                except RecursionError:
                    # Issue #25274: The recursion limit is too low at the
                    # current recursion depth
                    continue

                # Issue #5392: test stack overflow after hitting recursion
                # limit twice
                with self.assertRaises(RecursionError):
                    f()
                with self.assertRaises(RecursionError):
                    f()
        finally:
            sys.setrecursionlimit(old_limit)

    @test.support.cpython_only
    def test_setrecursionlimit_to_depth(self):
        # [FUNCTION DOC] CPython-only, issue 25274: setting the recursion limit to one more
        # than the current depth succeeds, but setting it to exactly the current depth
        # raises RecursionError with a message saying the limit is too low. Restores the old
        # limit afterwards.
        # Issue #25274: Setting a low recursion limit must be blocked if the
        # current recursion depth is already higher than limit.

        old_limit = sys.getrecursionlimit()
        try:
            depth = support.get_recursion_depth()
            with self.subTest(limit=sys.getrecursionlimit(), depth=depth):
                # depth + 1 is OK
                sys.setrecursionlimit(depth + 1)

                # reset the limit to be able to call self.assertRaises()
                # context manager
                sys.setrecursionlimit(old_limit)
                with self.assertRaises(RecursionError) as cm:
                    sys.setrecursionlimit(depth)
            self.assertRegex(str(cm.exception),
                             "cannot set the recursion limit to [0-9]+ "
                             "at the recursion depth [0-9]+: "
                             "the limit is too low")
        finally:
            sys.setrecursionlimit(old_limit)

    @unittest.skipUnless(support.Py_GIL_DISABLED, "only meaningful if the GIL is disabled")
    @threading_helper.requires_working_threading()
    def test_racing_recursion_limit(self):
        # [FUNCTION DOC] Free-threaded builds only: starts five threads that keep changing the
        # recursion limit and five that recurse 50 deep, and re-raises any exception captured in
        # the threads.
        from threading import Thread
        def something_recursive():
            # [FUNCTION DOC] HELPER: Thread body that runs a 50-level recursive count.
            def count(n):
                # [FUNCTION DOC] HELPER: Recursively counts down from n, returning n.
                if n > 0:
                    return count(n - 1) + 1
                return 0

            count(50)

        def set_recursion_limit():
            # [FUNCTION DOC] HELPER: Thread body that sets the recursion limit to each value from
            # 100 to 199.
            for limit in range(100, 200):
                sys.setrecursionlimit(limit)

        threads = []
        for _ in range(5):
            threads.append(Thread(target=set_recursion_limit))

        for _ in range(5):
            threads.append(Thread(target=something_recursive))

        with threading_helper.catch_threading_exception() as cm:
            with threading_helper.start_threads(threads):
                pass

            if cm.exc_value:
                raise cm.exc_value

    def test_getwindowsversion(self):
        # [FUNCTION DOC] Skips unless sys has getwindowsversion. Checks the result has five fields
        # with expected types, supports named access matching positional access, raises IndexError
        # past the end, and can still be unpacked into five values.
        # Raise SkipTest if sys doesn't have getwindowsversion attribute
        test.support.get_attribute(sys, "getwindowsversion")
        v = sys.getwindowsversion()
        self.assertEqual(len(v), 5)
        self.assertIsInstance(v[0], int)
        self.assertIsInstance(v[1], int)
        self.assertIsInstance(v[2], int)
        self.assertIsInstance(v[3], int)
        self.assertIsInstance(v[4], str)
        self.assertRaises(IndexError, operator.getitem, v, 5)
        self.assertIsInstance(v.major, int)
        self.assertIsInstance(v.minor, int)
        self.assertIsInstance(v.build, int)
        self.assertIsInstance(v.platform, int)
        self.assertIsInstance(v.service_pack, str)
        self.assertIsInstance(v.service_pack_minor, int)
        self.assertIsInstance(v.service_pack_major, int)
        self.assertIsInstance(v.suite_mask, int)
        self.assertIsInstance(v.product_type, int)
        self.assertEqual(v[0], v.major)
        self.assertEqual(v[1], v.minor)
        self.assertEqual(v[2], v.build)
        self.assertEqual(v[3], v.platform)
        self.assertEqual(v[4], v.service_pack)

        # This is how platform.py calls it. Make sure tuple
        #  still has 5 elements
        maj, min, buildno, plat, csd = sys.getwindowsversion()

    def test_call_tracing(self):
        # [FUNCTION DOC] call_tracing called with a function and a non-tuple argument
        # raises TypeError.
        self.assertRaises(TypeError, sys.call_tracing, type, 2)

    @unittest.skipUnless(hasattr(sys, "setdlopenflags"),
                         'test needs sys.setdlopenflags()')
    def test_dlopenflags(self):
        # [FUNCTION DOC] When setdlopenflags exists: getter rejects arguments, setter requires
        # one, a changed value reads back, and the original flags are restored.
        self.assertTrue(hasattr(sys, "getdlopenflags"))
        self.assertRaises(TypeError, sys.getdlopenflags, 42)
        oldflags = sys.getdlopenflags()
        self.assertRaises(TypeError, sys.setdlopenflags)
        sys.setdlopenflags(oldflags+1)
        self.assertEqual(sys.getdlopenflags(), oldflags+1)
        sys.setdlopenflags(oldflags)

    @test.support.refcount_test
    def test_refcount(self):
        # [FUNCTION DOC] Checks getrefcount requires an argument and that the count of None is
        # unchanged by binding and deleting a global; also checks gettotalrefcount returns an int
        # when present.
        # n here originally had to be a global in order for this test to pass
        # while tracing with a python function. Tracing used to call
        # PyFrame_FastToLocals, which would add a copy of any locals to the
        # frame object, causing the ref count to increase by 2 instead of 1.
        # While that no longer happens (due to PEP 667), this test case retains
        # its original global-based implementation
        # PEP 683's immortal objects also made this point moot, since the
        # refcount for None doesn't change anyway. Maybe this test should be
        # using a different constant value? (e.g. an integer)
        global n
        self.assertRaises(TypeError, sys.getrefcount)
        c = sys.getrefcount(None)
        n = None
        # Singleton refcnts don't change
        self.assertEqual(sys.getrefcount(None), c)
        del n
        self.assertEqual(sys.getrefcount(None), c)
        if hasattr(sys, "gettotalrefcount"):
            self.assertIsInstance(sys.gettotalrefcount(), int)

    def test_getframe(self):
        # [FUNCTION DOC] _getframe rejects too many arguments (TypeError) and absurd depth
        # (ValueError); the frame at depth zero has this test's code object.
        self.assertRaises(TypeError, sys._getframe, 42, 42)
        self.assertRaises(ValueError, sys._getframe, 2000000000)
        self.assertTrue(
            SysModuleTest.test_getframe.__code__ \
            is sys._getframe().f_code
        )

    def test_getframemodulename(self):
        # [FUNCTION DOC] _getframemodulename: depth zero is this module, depth one is
        # unittest.case; walks up the stack comparing each frame's module name (or main when None)
        # and checks it returns None past the stack bottom.
        # Default depth gets ourselves
        self.assertEqual(__name__, sys._getframemodulename())
        self.assertEqual("unittest.case", sys._getframemodulename(1))
        i = 0
        f = sys._getframe(i)
        while f:
            self.assertEqual(
                f.f_globals['__name__'],
                sys._getframemodulename(i) or '__main__'
            )
            i += 1
            f2 = f.f_back
            try:
                f = sys._getframe(i)
            except ValueError:
                break
            self.assertIs(f, f2)
        self.assertIsNone(sys._getframemodulename(i))

    # sys._current_frames() is a CPython-only gimmick.
    @threading_helper.reap_threads
    @threading_helper.requires_working_threading()
    def test_current_frames(self):
        # [FUNCTION DOC] Starts a thread blocked inside a nested call, then checks
        # _current_frames: keys are positive ints including the main and worker thread, the main
        # frame is the current frame, and the worker's stack shows f123 calling g456 on the
        # expected line. Always releases and joins the thread.
        import threading
        import traceback

        # Spawn a thread that blocks at a known place.  Then the main
        # thread does sys._current_frames(), and verifies that the frames
        # returned make sense.
        entered_g = threading.Event()
        leave_g = threading.Event()
        thread_info = []  # the thread's id

        def f123():
            # [FUNCTION DOC] HELPER: Thread body that calls g456.
            g456()

        def g456():
            # [FUNCTION DOC] HELPER: Records the thread id, signals the entered event, and waits
            # for the leave event.
            thread_info.append(threading.get_ident())
            entered_g.set()
            leave_g.wait()

        t = threading.Thread(target=f123)
        t.start()
        entered_g.wait()

        try:
            # At this point, t has finished its entered_g.set(), although it's
            # impossible to guess whether it's still on that line or has moved on
            # to its leave_g.wait().
            self.assertEqual(len(thread_info), 1)
            thread_id = thread_info[0]

            d = sys._current_frames()
            for tid in d:
                self.assertIsInstance(tid, int)
                self.assertGreater(tid, 0)

            main_id = threading.get_ident()
            self.assertIn(main_id, d)
            self.assertIn(thread_id, d)

            # Verify that the captured main-thread frame is _this_ frame.
            frame = d.pop(main_id)
            self.assertTrue(frame is sys._getframe())

            # Verify that the captured thread frame is blocked in g456, called
            # from f123.  This is a little tricky, since various bits of
            # threading.py are also in the thread's call stack.
            frame = d.pop(thread_id)
            stack = traceback.extract_stack(frame)
            for i, (filename, lineno, funcname, sourceline) in enumerate(stack):
                if funcname == "f123":
                    break
            else:
                self.fail("didn't find f123() on thread's call stack")

            self.assertEqual(sourceline, "g456()")

            # And the next record must be for g456().
            filename, lineno, funcname, sourceline = stack[i+1]
            self.assertEqual(funcname, "g456")
            self.assertIn(sourceline, ["leave_g.wait()", "entered_g.set()"])
        finally:
            # Reap the spawned thread.
            leave_g.set()
            t.join()

    @threading_helper.reap_threads
    @threading_helper.requires_working_threading()
    def test_current_exceptions(self):
        # [FUNCTION DOC] Like the current frames test but for _current_exceptions: a thread
        # repeatedly raises and handles an exception while waiting; checks keys are thread ids,
        # the main thread maps to None, and the worker's exception traceback frame shows f123
        # calling g456 at the raise-handler lines. Releases and joins the thread.
        import threading
        import traceback

        # Spawn a thread that blocks at a known place.  Then the main
        # thread does sys._current_frames(), and verifies that the frames
        # returned make sense.
        g_raised = threading.Event()
        leave_g = threading.Event()
        thread_info = []  # the thread's id

        def f123():
            # [FUNCTION DOC] HELPER: Thread body that calls g456.
            g456()

        def g456():
            # [FUNCTION DOC] HELPER: Records the thread id, then loops raising and catching
            # ValueError, signaling the raised event, until the leave event is set.
            thread_info.append(threading.get_ident())
            while True:
                try:
                    raise ValueError("oops")
                except ValueError:
                    g_raised.set()
                    if leave_g.wait(timeout=support.LONG_TIMEOUT):
                        break

        t = threading.Thread(target=f123)
        t.start()
        g_raised.wait(timeout=support.LONG_TIMEOUT)

        try:
            self.assertEqual(len(thread_info), 1)
            thread_id = thread_info[0]

            d = sys._current_exceptions()
            for tid in d:
                self.assertIsInstance(tid, int)
                self.assertGreater(tid, 0)

            main_id = threading.get_ident()
            self.assertIn(main_id, d)
            self.assertIn(thread_id, d)
            self.assertEqual(None, d.pop(main_id))

            # Verify that the captured thread frame is blocked in g456, called
            # from f123.  This is a little tricky, since various bits of
            # threading.py are also in the thread's call stack.
            exc_value = d.pop(thread_id)
            stack = traceback.extract_stack(exc_value.__traceback__.tb_frame)
            for i, (filename, lineno, funcname, sourceline) in enumerate(stack):
                if funcname == "f123":
                    break
            else:
                self.fail("didn't find f123() on thread's call stack")

            self.assertEqual(sourceline, "g456()")

            # And the next record must be for g456().
            filename, lineno, funcname, sourceline = stack[i+1]
            self.assertEqual(funcname, "g456")
            self.assertTrue((sourceline.startswith("if leave_g.wait(") or
                             sourceline.startswith("g_raised.set()")))
        finally:
            # Reap the spawned thread.
            leave_g.set()
            t.join()

    def test_attributes(self):
        # [FUNCTION DOC] Type and value sanity checks for many sys attributes: api_version,
        # argv, orig_argv, byteorder, float_info, int_info, hash_info (including a probable-
        # prime check of the modulus and algorithm consistency with the build config), maxsize,
        # maxunicode, prefixes, version and version_info fields, float_repr_style, and abiflags
        # on non-Windows.
        self.assertIsInstance(sys.api_version, int)
        self.assertIsInstance(sys.argv, list)
        for arg in sys.argv:
            self.assertIsInstance(arg, str)
        self.assertIsInstance(sys.orig_argv, list)
        for arg in sys.orig_argv:
            self.assertIsInstance(arg, str)
        self.assertIn(sys.byteorder, ("little", "big"))
        self.assertIsInstance(sys.builtin_module_names, tuple)
        self.assertIsInstance(sys.copyright, str)
        self.assertIsInstance(sys.exec_prefix, str)
        self.assertIsInstance(sys.base_exec_prefix, str)
        self.assertIsInstance(sys.executable, str)
        self.assertEqual(len(sys.float_info), 11)
        self.assertEqual(sys.float_info.radix, 2)
        self.assertEqual(len(sys.int_info), 4)
        self.assertTrue(sys.int_info.bits_per_digit % 5 == 0)
        self.assertTrue(sys.int_info.sizeof_digit >= 1)
        self.assertGreaterEqual(sys.int_info.default_max_str_digits, 500)
        self.assertGreaterEqual(sys.int_info.str_digits_check_threshold, 100)
        self.assertGreater(sys.int_info.default_max_str_digits,
                           sys.int_info.str_digits_check_threshold)
        self.assertEqual(type(sys.int_info.bits_per_digit), int)
        self.assertEqual(type(sys.int_info.sizeof_digit), int)
        self.assertIsInstance(sys.int_info.default_max_str_digits, int)
        self.assertIsInstance(sys.int_info.str_digits_check_threshold, int)
        self.assertIsInstance(sys.hexversion, int)

        self.assertEqual(len(sys.hash_info), 9)
        self.assertLess(sys.hash_info.modulus, 2**sys.hash_info.width)
        # sys.hash_info.modulus should be a prime; we do a quick
        # probable primality test (doesn't exclude the possibility of
        # a Carmichael number)
        for x in range(1, 100):
            self.assertEqual(
                pow(x, sys.hash_info.modulus-1, sys.hash_info.modulus),
                1,
                "sys.hash_info.modulus {} is a non-prime".format(
                    sys.hash_info.modulus)
                )
        self.assertIsInstance(sys.hash_info.inf, int)
        self.assertIsInstance(sys.hash_info.nan, int)
        self.assertIsInstance(sys.hash_info.imag, int)
        algo = sysconfig.get_config_var("Py_HASH_ALGORITHM")
        if sys.hash_info.algorithm in {"fnv", "siphash13", "siphash24"}:
            self.assertIn(sys.hash_info.hash_bits, {32, 64})
            self.assertIn(sys.hash_info.seed_bits, {32, 64, 128})

            if algo == 1:
                self.assertEqual(sys.hash_info.algorithm, "siphash24")
            elif algo == 2:
                self.assertEqual(sys.hash_info.algorithm, "fnv")
            elif algo == 3:
                self.assertEqual(sys.hash_info.algorithm, "siphash13")
            else:
                self.assertIn(sys.hash_info.algorithm, {"fnv", "siphash13", "siphash24"})
        else:
            # PY_HASH_EXTERNAL
            self.assertEqual(algo, 0)
        self.assertGreaterEqual(sys.hash_info.cutoff, 0)
        self.assertLess(sys.hash_info.cutoff, 8)

        self.assertIsInstance(sys.maxsize, int)
        self.assertIsInstance(sys.maxunicode, int)
        self.assertEqual(sys.maxunicode, 0x10FFFF)
        self.assertIsInstance(sys.platform, str)
        self.assertIsInstance(sys.prefix, str)
        self.assertIsInstance(sys.base_prefix, str)
        self.assertIsInstance(sys.platlibdir, str)
        self.assertIsInstance(sys.version, str)
        vi = sys.version_info
        self.assertIsInstance(vi[:], tuple)
        self.assertEqual(len(vi), 5)
        self.assertIsInstance(vi[0], int)
        self.assertIsInstance(vi[1], int)
        self.assertIsInstance(vi[2], int)
        self.assertIn(vi[3], ("alpha", "beta", "candidate", "final"))
        self.assertIsInstance(vi[4], int)
        self.assertIsInstance(vi.major, int)
        self.assertIsInstance(vi.minor, int)
        self.assertIsInstance(vi.micro, int)
        self.assertIn(vi.releaselevel, ("alpha", "beta", "candidate", "final"))
        self.assertIsInstance(vi.serial, int)
        self.assertEqual(vi[0], vi.major)
        self.assertEqual(vi[1], vi.minor)
        self.assertEqual(vi[2], vi.micro)
        self.assertEqual(vi[3], vi.releaselevel)
        self.assertEqual(vi[4], vi.serial)
        self.assertTrue(vi > (1,0,0))
        self.assertIsInstance(sys.float_repr_style, str)
        self.assertIn(sys.float_repr_style, ('short', 'legacy'))
        if not sys.platform.startswith('win'):
            self.assertIsInstance(sys.abiflags, str)

    def test_thread_info(self):
        # [FUNCTION DOC] thread_info has three fields; name and lock are among the allowed
        # values, and name is platform specific (pthread on Linux/Android/FreeBSD, nt on Windows,
        # and so on).
        info = sys.thread_info
        self.assertEqual(len(info), 3)
        self.assertIn(info.name, ('nt', 'pthread', 'pthread-stubs', 'solaris', None))
        self.assertIn(info.lock, ('semaphore', 'mutex+cond', None))
        if sys.platform.startswith(("linux", "android", "freebsd")):
            self.assertEqual(info.name, "pthread")
        elif sys.platform == "win32":
            self.assertEqual(info.name, "nt")
        elif sys.platform == "emscripten":
            self.assertIn(info.name, {"pthread", "pthread-stubs"})
        elif sys.platform == "wasi":
            self.assertEqual(info.name, "pthread-stubs")

    @unittest.skipUnless(support.is_emscripten, "only available on Emscripten")
    def test_emscripten_info(self):
        # [FUNCTION DOC] Emscripten only: _emscripten_info has four fields with expected types.
        self.assertEqual(len(sys._emscripten_info), 4)
        self.assertIsInstance(sys._emscripten_info.emscripten_version, tuple)
        self.assertIsInstance(sys._emscripten_info.runtime, (str, type(None)))
        self.assertIsInstance(sys._emscripten_info.pthreads, bool)
        self.assertIsInstance(sys._emscripten_info.shared_memory, bool)

    def test_43581(self):
        # [FUNCTION DOC] Issue 43581: the original stdout and stderr have the same encoding.
        # Can't use sys.stdout, as this is a StringIO object when
        # the test runs under regrtest.
        self.assertEqual(sys.__stdout__.encoding, sys.__stderr__.encoding)

    def test_intern(self):
        # [FUNCTION DOC] intern requires a str argument; a fresh string interns to itself and an
        # equal copy interns to the first object, with _is_interned checks where available. String
        # subclasses cannot be interned (TypeError) and are reported as not interned.
        has_is_interned = (test.support.check_impl_detail(cpython=True)
                           or hasattr(sys, '_is_interned'))
        self.assertRaises(TypeError, sys.intern)
        self.assertRaises(TypeError, sys.intern, b'abc')
        if has_is_interned:
            self.assertRaises(TypeError, sys._is_interned)
            self.assertRaises(TypeError, sys._is_interned, b'abc')
        s = "never interned before" + str(random.randrange(0, 10**9))
        self.assertTrue(sys.intern(s) is s)
        if has_is_interned:
            self.assertIs(sys._is_interned(s), True)
        s2 = s.swapcase().swapcase()
        if has_is_interned:
            self.assertIs(sys._is_interned(s2), False)
        self.assertTrue(sys.intern(s2) is s)
        if has_is_interned:
            self.assertIs(sys._is_interned(s2), False)

        # Subclasses of string can't be interned, because they
        # provide too much opportunity for insane things to happen.
        # We don't want them in the interned dict and if they aren't
        # actually interned, we don't want to create the appearance
        # that they are by allowing intern() to succeed.
        class S(str):
            # [CLASS DOC] HELPER: String subclass with a constant hash value of 123.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: Returns the constant 123.
                return 123

        self.assertRaises(TypeError, sys.intern, S("abc"))
        if has_is_interned:
            self.assertIs(sys._is_interned(S("abc")), False)

    @support.cpython_only
    @requires_subinterpreters
    def test_subinterp_intern_dynamically_allocated(self):
        # [FUNCTION DOC] CPython-only: a dynamically allocated string interned here gets a
        # different object in a fresh sub-interpreter; the sub-interpreter code asserts the ids
        # differ. Skipped if sub-interpreters are unavailable.
        # Implementation detail: Dynamically allocated strings
        # are distinct between interpreters
        s = "never interned before" + str(random.randrange(0, 10**9))
        t = sys.intern(s)
        self.assertIs(t, s)

        interp = interpreters.create()
        interp.exec(textwrap.dedent(f'''
            import sys

            # set `s`, avoid parser interning & constant folding
            s = str({s.encode()!r}, 'utf-8')

            t = sys.intern(s)

            assert id(t) != {id(s)}, (id(t), {id(s)})
            assert id(t) != {id(t)}, (id(t), {id(t)})
            '''))

    @support.cpython_only
    @requires_subinterpreters
    def test_subinterp_intern_statically_allocated(self):
        # [FUNCTION DOC] CPython-only: for several strings that are statically allocated,
        # interning in a fresh sub-interpreter yields the same object as here (id compared inside
        # the sub-interpreter).
        # Implementation detail: Statically allocated strings are shared
        # between interpreters.
        # See Tools/build/generate_global_objects.py for the list
        # of strings that are always statically allocated.
        for s in ('__init__', 'CANCELLED', '<module>', 'utf-8',
                  '{{', '', '\n', '_', 'x', '\0', '\N{CEDILLA}', '\xff',
                  ):
            with self.subTest(s=s):
                t = sys.intern(s)

                interp = interpreters.create()
                interp.exec(textwrap.dedent(f'''
                    import sys

                    # set `s`, avoid parser interning & constant folding
                    s = str({s.encode()!r}, 'utf-8')

                    t = sys.intern(s)
                    assert id(t) == {id(t)}, (id(t), {id(t)})
                    '''))

    @support.cpython_only
    @requires_subinterpreters
    def test_subinterp_intern_singleton(self):
        # [FUNCTION DOC] CPython-only: in a fresh sub-interpreter, zero- and one-character latin-1
        # strings are singletons, so a rebuilt string has the same id as the outer one; interning
        # is then checked in the outer interpreter with _is_interned.
        # Implementation detail: singletons are used for 0- and 1-character
        # latin1 strings.
        for s in '', '\n', '_', 'x', '\0', '\N{CEDILLA}', '\xff':
            with self.subTest(s=s):
                interp = interpreters.create()
                interp.exec(textwrap.dedent(f'''
                    import sys

                    # set `s`, avoid parser interning & constant folding
                    s = str({s.encode()!r}, 'utf-8')

                    assert id(s) == {id(s)}
                    t = sys.intern(s)
                    '''))
                self.assertTrue(sys._is_interned(s))

    def test_sys_flags(self):
        # [FUNCTION DOC] sys.flags is truthy, has a field for each expected flag with int type
        # (bool for dev_mode and safe_path), a repr, exactly that many fields, and utf8_mode is 0,
        # 1 or 2.
        self.assertTrue(sys.flags)
        attrs = ("debug",
                 "inspect", "interactive", "optimize",
                 "dont_write_bytecode", "no_user_site", "no_site",
                 "ignore_environment", "verbose", "bytes_warning", "quiet",
                 "hash_randomization", "isolated", "dev_mode", "utf8_mode",
                 "warn_default_encoding", "safe_path", "int_max_str_digits")
        for attr in attrs:
            self.assertTrue(hasattr(sys.flags, attr), attr)
            attr_type = bool if attr in ("dev_mode", "safe_path") else int
            self.assertEqual(type(getattr(sys.flags, attr)), attr_type, attr)
        self.assertTrue(repr(sys.flags))
        self.assertEqual(len(sys.flags), len(attrs))

        self.assertIn(sys.flags.utf8_mode, {0, 1, 2})

    def assert_raise_on_new_sys_type(self, sys_attr):
        # [FUNCTION DOC] HELPER: Asserts that the type of the given sys attribute cannot
        # be instantiated directly, either by calling the type or by its __new__ (both
        # raise TypeError).
        # Users are intentionally prevented from creating new instances of
        # sys.flags, sys.version_info, and sys.getwindowsversion.
        arg = sys_attr
        attr_type = type(sys_attr)
        with self.assertRaises(TypeError):
            attr_type(arg)
        with self.assertRaises(TypeError):
            attr_type.__new__(attr_type, arg)

    def test_sys_flags_no_instantiation(self):
        # [FUNCTION DOC] Direct creation of a new sys.flags object raises TypeError.
        self.assert_raise_on_new_sys_type(sys.flags)

    def test_sys_version_info_no_instantiation(self):
        # [FUNCTION DOC] Direct creation of a new sys.version_info object raises TypeError.
        self.assert_raise_on_new_sys_type(sys.version_info)

    def test_sys_getwindowsversion_no_instantiation(self):
        # [FUNCTION DOC] Skips unless on Windows; direct creation of a getwindowsversion result
        # type raises TypeError.
        # Skip if not being run on Windows.
        test.support.get_attribute(sys, "getwindowsversion")
        self.assert_raise_on_new_sys_type(sys.getwindowsversion())

    @test.support.cpython_only
    def test_clear_type_cache(self):
        # [FUNCTION DOC] CPython-only smoke test: calling _clear_type_cache runs without error.
        sys._clear_type_cache()

    @force_not_colorized
    @support.requires_subprocess()
    def test_ioencoding(self):
        # [FUNCTION DOC] Runs subprocesses printing a cent sign under different PYTHONIOENCODING
        # values: cp424 encodes it, ascii with replace prints a question mark, plain ascii (with
        # or without trailing colon) fails with a UnicodeEncodeError, and a surrogateescape-only
        # setting emits the raw byte. Requires subprocess support; colors disabled.
        env = dict(os.environ)

        # Test character: cent sign, encoded as 0x4A (ASCII J) in CP424,
        # not representable in ASCII.

        env["PYTHONIOENCODING"] = "cp424"
        p = subprocess.Popen([sys.executable, "-c", 'print(chr(0xa2))'],
                             stdout = subprocess.PIPE, env=env)
        out = p.communicate()[0].strip()
        expected = ("\xa2" + os.linesep).encode("cp424")
        self.assertEqual(out, expected)

        env["PYTHONIOENCODING"] = "ascii:replace"
        p = subprocess.Popen([sys.executable, "-c", 'print(chr(0xa2))'],
                             stdout = subprocess.PIPE, env=env)
        out = p.communicate()[0].strip()
        self.assertEqual(out, b'?')

        env["PYTHONIOENCODING"] = "ascii"
        p = subprocess.Popen([sys.executable, "-c", 'print(chr(0xa2))'],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             env=env)
        out, err = p.communicate()
        self.assertEqual(out, b'')
        self.assertIn(b'UnicodeEncodeError:', err)
        self.assertIn(rb"'\xa2'", err)

        env["PYTHONIOENCODING"] = "ascii:"
        p = subprocess.Popen([sys.executable, "-c", 'print(chr(0xa2))'],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             env=env)
        out, err = p.communicate()
        self.assertEqual(out, b'')
        self.assertIn(b'UnicodeEncodeError:', err)
        self.assertIn(rb"'\xa2'", err)

        env["PYTHONIOENCODING"] = ":surrogateescape"
        p = subprocess.Popen([sys.executable, "-c", 'print(chr(0xdcbd))'],
                             stdout=subprocess.PIPE, env=env)
        out = p.communicate()[0].strip()
        self.assertEqual(out, b'\xbd')

    @unittest.skipUnless(os_helper.FS_NONASCII,
                         'requires OS support of non-ASCII encodings')
    @unittest.skipUnless(sys.getfilesystemencoding() == locale.getpreferredencoding(False),
                         'requires FS encoding to match locale')
    @support.requires_subprocess()
    def test_ioencoding_nonascii(self):
        # [FUNCTION DOC] Where the OS supports non-ASCII file names and filesystem and locale
        # encodings match: a subprocess with empty PYTHONIOENCODING prints the non-ASCII character
        # as the filesystem bytes.
        env = dict(os.environ)

        env["PYTHONIOENCODING"] = ""
        p = subprocess.Popen([sys.executable, "-c",
                                'print(%a)' % os_helper.FS_NONASCII],
                                stdout=subprocess.PIPE, env=env)
        out = p.communicate()[0].strip()
        self.assertEqual(out, os.fsencode(os_helper.FS_NONASCII))

    @unittest.skipIf(sys.base_prefix != sys.prefix,
                     'Test is not venv-compatible')
    @support.requires_subprocess()
    def test_executable(self):
        # [FUNCTION DOC] sys.executable is an absolute path; a child process started with a
        # nonexistent argv[0] but an explicit executable reports either an empty executable or the
        # real one. Skipped in virtual environments.
        # sys.executable should be absolute
        self.assertEqual(os.path.abspath(sys.executable), sys.executable)

        # Issue #7774: Ensure that sys.executable is an empty string if argv[0]
        # has been set to a non existent program name and Python is unable to
        # retrieve the real program name

        # For a normal installation, it should work without 'cwd'
        # argument. For test runs in the build directory, see #7774.
        python_dir = os.path.dirname(os.path.realpath(sys.executable))
        p = subprocess.Popen(
            ["nonexistent", "-c",
             'import sys; print(sys.executable.encode("ascii", "backslashreplace"))'],
            executable=sys.executable, stdout=subprocess.PIPE, cwd=python_dir)
        stdout = p.communicate()[0]
        executable = stdout.strip().decode("ASCII")
        p.wait()
        self.assertIn(executable, ["b''", repr(sys.executable.encode("ascii", "backslashreplace"))])

    def check_fsencoding(self, fs_encoding, expected=None):
        # [FUNCTION DOC] HELPER: Asserts the filesystem encoding is not None, is a known codec,
        # and equals expected when expected is given.
        self.assertIsNotNone(fs_encoding)
        codecs.lookup(fs_encoding)
        if expected:
            self.assertEqual(fs_encoding, expected)

    def test_getfilesystemencoding(self):
        # [FUNCTION DOC] Checks the filesystem encoding is a valid codec, and utf-8 on macOS.
        fs_encoding = sys.getfilesystemencoding()
        if sys.platform == 'darwin':
            expected = 'utf-8'
        else:
            expected = None
        self.check_fsencoding(fs_encoding, expected)

    def c_locale_get_error_handler(self, locale, isolated=False, encoding=None):
        # [FUNCTION DOC] HELPER: Starts a child interpreter under the given locale (utf8 mode off,
        # optional isolated mode and PYTHONIOENCODING) and returns its output listing the error
        # handler of stdin, stdout and stderr. The coerce-locale env var is disabled.
        # Force the POSIX locale
        env = os.environ.copy()
        env["LC_ALL"] = locale
        env["PYTHONCOERCECLOCALE"] = "0"
        code = '\n'.join((
            'import sys',
            'def dump(name):',
            '    std = getattr(sys, name)',
            '    print("%s: %s" % (name, std.errors))',
            'dump("stdin")',
            'dump("stdout")',
            'dump("stderr")',
        ))
        args = [sys.executable, "-X", "utf8=0", "-c", code]
        if isolated:
            args.append("-I")
        if encoding is not None:
            env['PYTHONIOENCODING'] = encoding
        else:
            env.pop('PYTHONIOENCODING', None)
        p = subprocess.Popen(args,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              env=env,
                              universal_newlines=True)
        stdout, stderr = p.communicate()
        return stdout

    def check_locale_surrogateescape(self, locale):
        # [FUNCTION DOC] HELPER: Checks the standard stream error handlers under a C-like locale:
        # surrogateescape for stdin and stdout by default, overridden by an error handler in the
        # encoding variable, strict when an encoding is forced, unaffected by an empty or colon-
        # only setting; stderr is always backslashreplace.
        out = self.c_locale_get_error_handler(locale, isolated=True)
        self.assertEqual(out,
                         'stdin: surrogateescape\n'
                         'stdout: surrogateescape\n'
                         'stderr: backslashreplace\n')

        # replace the default error handler
        out = self.c_locale_get_error_handler(locale, encoding=':ignore')
        self.assertEqual(out,
                         'stdin: ignore\n'
                         'stdout: ignore\n'
                         'stderr: backslashreplace\n')

        # force the encoding
        out = self.c_locale_get_error_handler(locale, encoding='iso8859-1')
        self.assertEqual(out,
                         'stdin: strict\n'
                         'stdout: strict\n'
                         'stderr: backslashreplace\n')
        out = self.c_locale_get_error_handler(locale, encoding='iso8859-1:')
        self.assertEqual(out,
                         'stdin: strict\n'
                         'stdout: strict\n'
                         'stderr: backslashreplace\n')

        # have no any effect
        out = self.c_locale_get_error_handler(locale, encoding=':')
        self.assertEqual(out,
                         'stdin: surrogateescape\n'
                         'stdout: surrogateescape\n'
                         'stderr: backslashreplace\n')
        out = self.c_locale_get_error_handler(locale, encoding='')
        self.assertEqual(out,
                         'stdin: surrogateescape\n'
                         'stdout: surrogateescape\n'
                         'stderr: backslashreplace\n')

    @support.requires_subprocess()
    def test_c_locale_surrogateescape(self):
        # [FUNCTION DOC] Checks stream error handlers under the C locale. Requires
        # subprocess support.
        self.check_locale_surrogateescape('C')

    @support.requires_subprocess()
    def test_posix_locale_surrogateescape(self):
        # [FUNCTION DOC] Checks stream error handlers under the POSIX locale. Requires
        # subprocess support.
        self.check_locale_surrogateescape('POSIX')

    def test_implementation(self):
        # [FUNCTION DOC] sys.implementation has name, version, hexversion and cache_tag; the
        # version prefix and the hexversion value are consistent with the version fields, and the
        # name is lower case.
        # This test applies to all implementations equally.

        levels = {'alpha': 0xA, 'beta': 0xB, 'candidate': 0xC, 'final': 0xF}

        self.assertTrue(hasattr(sys.implementation, 'name'))
        self.assertTrue(hasattr(sys.implementation, 'version'))
        self.assertTrue(hasattr(sys.implementation, 'hexversion'))
        self.assertTrue(hasattr(sys.implementation, 'cache_tag'))

        version = sys.implementation.version
        self.assertEqual(version[:2], (version.major, version.minor))

        hexversion = (version.major << 24 | version.minor << 16 |
                      version.micro << 8 | levels[version.releaselevel] << 4 |
                      version.serial << 0)
        self.assertEqual(sys.implementation.hexversion, hexversion)

        # PEP 421 requires that .name be lower case.
        self.assertEqual(sys.implementation.name,
                         sys.implementation.name.lower())

    @test.support.cpython_only
    def test_debugmallocstats(self):
        # [FUNCTION DOC] CPython-only: _debugmallocstats in a subprocess writes to stderr
        # text depending on freelist and pymalloc build options (empty when neither), and
        # takes no arguments.
        # Test sys._debugmallocstats()
        from test.support.script_helper import assert_python_ok
        args = ['-c', 'import sys; sys._debugmallocstats()']
        ret, out, err = assert_python_ok(*args)

        # Output of sys._debugmallocstats() depends on configure flags.
        # The sysconfig vars are not available on Windows.
        if sys.platform != "win32":
            with_freelists = sysconfig.get_config_var("WITH_FREELISTS")
            with_pymalloc = sysconfig.get_config_var("WITH_PYMALLOC")
            if with_freelists:
                self.assertIn(b"free PyDictObjects", err)
            if with_pymalloc:
                self.assertIn(b'Small block threshold', err)
            if not with_freelists and not with_pymalloc:
                self.assertFalse(err)

        # The function has no parameter
        self.assertRaises(TypeError, sys._debugmallocstats, True)

    @unittest.skipUnless(hasattr(sys, "getallocatedblocks"),
                         "sys.getallocatedblocks unavailable on this build")
    def test_getallocatedblocks(self):
        # [FUNCTION DOC] Checks getallocatedblocks returns an int that is positive when pymalloc
        # is used, less than the total refcount when available, not increased by a collection, and
        # stable within 50 across two consecutive collections.
        try:
            import _testinternalcapi
        except ImportError:
            with_pymalloc = support.with_pymalloc()
        else:
            try:
                alloc_name = _testinternalcapi.pymem_getallocatorsname()
            except RuntimeError as exc:
                # "cannot get allocators name" (ex: tracemalloc is used)
                with_pymalloc = True
            else:
                with_pymalloc = (alloc_name in ('pymalloc', 'pymalloc_debug'))

        # Some sanity checks
        a = sys.getallocatedblocks()
        self.assertIs(type(a), int)
        if with_pymalloc:
            self.assertGreater(a, 0)
        else:
            # When WITH_PYMALLOC isn't available, we don't know anything
            # about the underlying implementation: the function might
            # return 0 or something greater.
            self.assertGreaterEqual(a, 0)
        try:
            # While we could imagine a Python session where the number of
            # multiple buffer objects would exceed the sharing of references,
            # it is unlikely to happen in a normal test run.
            self.assertLess(a, sys.gettotalrefcount())
        except AttributeError:
            # gettotalrefcount() not available
            pass
        gc.collect()
        b = sys.getallocatedblocks()
        self.assertLessEqual(b, a)
        gc.collect()
        c = sys.getallocatedblocks()
        self.assertIn(c, range(b - 50, b + 50))

    def test_is_gil_enabled(self):
        # [FUNCTION DOC] _is_gil_enabled returns a bool on free-threaded builds and True on
        # normal builds.
        if support.Py_GIL_DISABLED:
            self.assertIs(type(sys._is_gil_enabled()), bool)
        else:
            self.assertTrue(sys._is_gil_enabled())

    def test_is_finalizing(self):
        # [FUNCTION DOC] is_finalizing is False in normal run; a subprocess whose module-level
        # object prints it in a destructor at exit must print True.
        self.assertIs(sys.is_finalizing(), False)
        # Don't use the atexit module because _Py_Finalizing is only set
        # after calling atexit callbacks
        code = """if 1:
            import sys

            class AtExit:
                is_finalizing = sys.is_finalizing
                print = print

                def __del__(self):
                    self.print(self.is_finalizing(), flush=True)

            # Keep a reference in the __main__ module namespace, so the
            # AtExit destructor will be called at Python exit
            ref = AtExit()
        """
        rc, stdout, stderr = assert_python_ok('-c', code)
        self.assertEqual(stdout.rstrip(), b'True')

    def test_issue20602(self):
        # [FUNCTION DOC] Issue 20602: sys.flags and sys.float_info are still usable (printable)
        # from a destructor running at interpreter shutdown, checked in a subprocess.
        # sys.flags and sys.float_info were wiped during shutdown.
        code = """if 1:
            import sys
            class A:
                def __del__(self, sys=sys):
                    print(sys.flags)
                    print(sys.float_info)
            a = A()
            """
        rc, out, err = assert_python_ok('-c', code)
        out = out.splitlines()
        self.assertIn(b'sys.flags', out[0])
        self.assertIn(b'sys.float_info', out[1])

    def test_sys_ignores_cleaning_up_user_data(self):
        # [FUNCTION DOC] A destructor of a user object stored on sys, running at shutdown, that
        # triggers an exception must not produce any output or non-zero exit code.
        code = """if 1:
            import struct, sys

            class C:
                def __init__(self):
                    self.pack = struct.pack
                def __del__(self):
                    self.pack('I', -42)

            sys.x = C()
            """
        rc, stdout, stderr = assert_python_ok('-c', code)
        self.assertEqual(rc, 0)
        self.assertEqual(stdout.rstrip(), b"")
        self.assertEqual(stderr.rstrip(), b"")

    @unittest.skipUnless(sys.platform == "android", "Android only")
    def test_getandroidapilevel(self):
        # [FUNCTION DOC] Android only: getandroidapilevel returns a positive int.
        level = sys.getandroidapilevel()
        self.assertIsInstance(level, int)
        self.assertGreater(level, 0)

    @force_not_colorized
    @support.requires_subprocess()
    def test_sys_tracebacklimit(self):
        # [FUNCTION DOC] Runs scripts raising a ZeroDivisionError two functions deep with several
        # values of sys.tracebacklimit and checks the resulting stderr traceback. Requires
        # subprocess support.
        code = """if 1:
            import sys
            def f1():
                1 / 0
            def f2():
                f1()
            sys.tracebacklimit = %r
            f2()
        """
        def check(tracebacklimit, expected):
            # [FUNCTION DOC] HELPER: Runs the script with the given traceback limit and checks
            # stderr against expected.
            p = subprocess.Popen([sys.executable, '-c', code % tracebacklimit],
                                 stderr=subprocess.PIPE)
            out = p.communicate()[1]
            self.assertEqual(out.splitlines(), expected)

        traceback = [
            b'Traceback (most recent call last):',
            b'  File "<string>", line 8, in <module>',
            b'    f2()',
            b'    ~~^^',
            b'  File "<string>", line 6, in f2',
            b'    f1()',
            b'    ~~^^',
            b'  File "<string>", line 4, in f1',
            b'    1 / 0',
            b'    ~~^~~',
            b'ZeroDivisionError: division by zero'
        ]
        check(10, traceback)
        check(3, traceback)
        check(2, traceback[:1] + traceback[4:])
        check(1, traceback[:1] + traceback[7:])
        check(0, [traceback[-1]])
        check(-1, [traceback[-1]])
        check(1<<1000, traceback)
        check(-1<<1000, [traceback[-1]])
        check(None, traceback)

    def test_no_duplicates_in_meta_path(self):
        # [FUNCTION DOC] The list sys.meta_path contains no duplicate entries.
        self.assertEqual(len(sys.meta_path), len(set(sys.meta_path)))

    @unittest.skipUnless(hasattr(sys, "_enablelegacywindowsfsencoding"),
                         'needs sys._enablelegacywindowsfsencoding()')
    def test__enablelegacywindowsfsencoding(self):
        # [FUNCTION DOC] Windows only: in a subprocess, enabling legacy filesystem encoding makes
        # the filesystem encoding and errors mbcs and replace.
        code = ('import sys',
                'sys._enablelegacywindowsfsencoding()',
                'print(sys.getfilesystemencoding(), sys.getfilesystemencodeerrors())')
        rc, out, err = assert_python_ok('-c', '; '.join(code))
        out = out.decode('ascii', 'replace').rstrip()
        self.assertEqual(out, 'mbcs replace')

    @support.requires_subprocess()
    def test_orig_argv(self):
        # [FUNCTION DOC] Runs a child interpreter in isolated mode with extra options and an
        # argument, and checks it prints argv without the interpreter and options and orig_argv as
        # the complete command line. Requires subprocess support.
        code = textwrap.dedent('''
            import sys
            print(sys.argv)
            print(sys.orig_argv)
        ''')
        args = [sys.executable, '-I', '-X', 'utf8', '-c', code, 'arg']
        proc = subprocess.run(args, check=True, capture_output=True, text=True)
        expected = [
            repr(['-c', 'arg']),  # sys.argv
            repr(args),  # sys.orig_argv
        ]
        self.assertEqual(proc.stdout.rstrip().splitlines(), expected,
                         proc)

    def test_module_names(self):
        # [FUNCTION DOC] stdlib_module_names is a frozenset containing only strings.
        self.assertIsInstance(sys.stdlib_module_names, frozenset)
        for name in sys.stdlib_module_names:
            self.assertIsInstance(name, str)

    def test_stdlib_dir(self):
        # [FUNCTION DOC] Re-imports os fresh and checks _stdlib_dir equals the directory of os
        # after path normalization.
        # KNOWN ISSUE: if os has no usable file location the expected value is None and the
        # normalization of it would fail.
        os = import_helper.import_fresh_module('os')
        marker = getattr(os, '__file__', None)
        if marker and not os.path.exists(marker):
            marker = None
        expected = os.path.dirname(marker) if marker else None
        self.assertEqual(os.path.normpath(sys._stdlib_dir),
                         os.path.normpath(expected))

    @unittest.skipUnless(hasattr(sys, 'getobjects'), 'need sys.getobjects()')
    def test_getobjects(self):
        # [FUNCTION DOC] Needs sys.getobjects (debug builds). With zero limit returns a non-empty
        # list; with a type filter returns exactly the 100 live instances created; with limit
        # three returns three.
        # sys.getobjects(0)
        all_objects = sys.getobjects(0)
        self.assertIsInstance(all_objects, list)
        self.assertGreater(len(all_objects), 0)

        # sys.getobjects(0, MyType)
        class MyType:
            # [CLASS DOC] HELPER: Empty class whose live instances are counted.
            pass
        size = 100
        my_objects = [MyType() for _ in range(size)]
        get_objects = sys.getobjects(0, MyType)
        self.assertEqual(len(get_objects), size)
        for obj in get_objects:
            self.assertIsInstance(obj, MyType)

        # sys.getobjects(3, MyType)
        get_objects = sys.getobjects(3, MyType)
        self.assertEqual(len(get_objects), 3)

    @unittest.skipUnless(hasattr(sys, '_stats_on'), 'need Py_STATS build')
    def test_pystats(self):
        # [FUNCTION DOC] Needs a stats-enabled build. Calls the statistics on, off, clear and dump
        # functions only to ensure they do not crash; state is not saved or restored.
        # Call the functions, just check that they don't crash
        # Cannot save/restore state.
        sys._stats_on()
        sys._stats_off()
        sys._stats_clear()
        sys._stats_dump()

    @test.support.cpython_only
    @unittest.skipUnless(hasattr(sys, 'abiflags'), 'need sys.abiflags')
    def test_disable_gil_abi(self):
        # [FUNCTION DOC] CPython-only: the letter t appears in sys.abiflags exactly when the GIL
        # is disabled.
        self.assertEqual('t' in sys.abiflags, support.Py_GIL_DISABLED)


@test.support.cpython_only
class UnraisableHookTest(__TestCase):
    # [CLASS DOC] CPython-only tests of the default and custom unraisable-exception hook, using C
    # API test helpers to report unraisable errors.
    def test_original_unraisablehook(self):
        # [FUNCTION DOC] Using the original unraisablehook, checks stderr reports from
        # writeunraisable and formatunraisable with and without an object or message: first-line
        # text, traceback header, and final exception line.
        _testcapi = import_helper.import_module('_testcapi')
        from _testcapi import err_writeunraisable, err_formatunraisable
        obj = hex

        with support.swap_attr(sys, 'unraisablehook',
                                    sys.__unraisablehook__):
            with support.captured_stderr() as stderr:
                err_writeunraisable(ValueError(42), obj)
            lines = stderr.getvalue().splitlines()
            self.assertEqual(lines[0], f'Exception ignored in: {obj!r}')
            self.assertEqual(lines[1], 'Traceback (most recent call last):')
            self.assertEqual(lines[-1], 'ValueError: 42')

            with support.captured_stderr() as stderr:
                err_writeunraisable(ValueError(42), None)
            lines = stderr.getvalue().splitlines()
            self.assertEqual(lines[0], 'Traceback (most recent call last):')
            self.assertEqual(lines[-1], 'ValueError: 42')

            with support.captured_stderr() as stderr:
                err_formatunraisable(ValueError(42), 'Error in %R', obj)
            lines = stderr.getvalue().splitlines()
            self.assertEqual(lines[0], f'Error in {obj!r}:')
            self.assertEqual(lines[1], 'Traceback (most recent call last):')
            self.assertEqual(lines[-1], 'ValueError: 42')

            with support.captured_stderr() as stderr:
                err_formatunraisable(ValueError(42), None)
            lines = stderr.getvalue().splitlines()
            self.assertEqual(lines[0], 'Traceback (most recent call last):')
            self.assertEqual(lines[-1], 'ValueError: 42')

    def test_original_unraisablehook_err(self):
        # [FUNCTION DOC] bpo-22836: the original hook produces a sensible report when a destructor
        # raises, both for a plain error and when the exception's string conversion itself fails;
        # checks the report mentions the destructor, this file, the raise line, exception names,
        # and ends with newline.
        # bpo-22836: PyErr_WriteUnraisable() should give sensible reports
        class BrokenDel:
            # [CLASS DOC] HELPER: Class whose destructor raises ValueError.
            def __del__(self):
                # [FUNCTION DOC] HELPER: Raises ValueError saying del is broken.
                exc = ValueError("del is broken")
                # The following line is included in the traceback report:
                raise exc

        class BrokenStrException(Exception):
            # [CLASS DOC] HELPER: Exception whose string conversion raises another exception.
            def __str__(self):
                # [FUNCTION DOC] HELPER: Always raises an Exception saying str is broken.
                raise Exception("str() is broken")

        class BrokenExceptionDel:
            # [CLASS DOC] HELPER: Class whose destructor raises a BrokenStrException.
            def __del__(self):
                # [FUNCTION DOC] HELPER: Raises a BrokenStrException instance.
                exc = BrokenStrException()
                # The following line is included in the traceback report:
                raise exc

        for test_class in (BrokenDel, BrokenExceptionDel):
            with self.subTest(test_class):
                obj = test_class()
                with test.support.captured_stderr() as stderr, \
                     test.support.swap_attr(sys, 'unraisablehook',
                                            sys.__unraisablehook__):
                    # Trigger obj.__del__()
                    del obj

                report = stderr.getvalue()
                self.assertIn("Exception ignored", report)
                self.assertIn(test_class.__del__.__qualname__, report)
                self.assertIn("test_sys.py", report)
                self.assertIn("raise exc", report)
                if test_class is BrokenExceptionDel:
                    self.assertIn("BrokenStrException", report)
                    self.assertIn("<exception str() failed>", report)
                else:
                    self.assertIn("ValueError", report)
                    self.assertIn("del is broken", report)
                self.assertTrue(report.endswith("\n"))

    def test_original_unraisablehook_exception_qualname(self):
        # [FUNCTION DOC] The original hook prints the qualified name of a nested exception class,
        # and prefixes the module name unless it is builtins or __main__; checked for three
        # module names.
        # See bpo-41031, bpo-45083.
        # Check that the exception is printed with its qualified name
        # rather than just classname, and the module names appears
        # unless it is one of the hard-coded exclusions.
        _testcapi = import_helper.import_module('_testcapi')
        from _testcapi import err_writeunraisable
        class A:
            # [CLASS DOC] HELPER: Container class holding nested classes.
            class B:
                # [CLASS DOC] HELPER: Nested container class.
                class X(Exception):
                    # [CLASS DOC] HELPER: Innermost exception class whose module name is changed
                    # by the test.
                    pass

        for moduleName in 'builtins', '__main__', 'some_module':
            with self.subTest(moduleName=moduleName):
                A.B.X.__module__ = moduleName
                with test.support.captured_stderr() as stderr, test.support.swap_attr(
                    sys, 'unraisablehook', sys.__unraisablehook__
                ):
                    err_writeunraisable(A.B.X(), "obj")
                report = stderr.getvalue()
                self.assertIn(A.B.X.__qualname__, report)
                if moduleName in ['builtins', '__main__']:
                    self.assertNotIn(moduleName + '.', report)
                else:
                    self.assertIn(moduleName + '.', report)

    def test_original_unraisablehook_wrong_type(self):
        # [FUNCTION DOC] Calling the original unraisablehook with a bare exception instead of an
        # args object raises TypeError.
        exc = ValueError(42)
        with test.support.swap_attr(sys, 'unraisablehook',
                                    sys.__unraisablehook__):
            with self.assertRaises(TypeError):
                sys.unraisablehook(exc)

    def test_custom_unraisablehook(self):
        # [FUNCTION DOC] A custom hook receives an args object carrying the exception type, value,
        # traceback, message and object, for both writeunraisable (no message) and
        # formatunraisable (formatted message, no object). Clears references to break cycles at
        # the end.
        # KNOWN ISSUE: the cleanup assigns None to a variable named expected that is never
        # otherwise used in this test.
        _testcapi = import_helper.import_module('_testcapi')
        from _testcapi import err_writeunraisable, err_formatunraisable
        hook_args = None

        def hook_func(args):
            # [FUNCTION DOC] HELPER: Stores the received args object in the enclosing
            # hook_args variable.
            nonlocal hook_args
            hook_args = args

        obj = hex
        try:
            with test.support.swap_attr(sys, 'unraisablehook', hook_func):
                exc = ValueError(42)
                err_writeunraisable(exc, obj)
                self.assertIs(hook_args.exc_type, type(exc))
                self.assertIs(hook_args.exc_value, exc)
                self.assertIs(hook_args.exc_traceback, exc.__traceback__)
                self.assertIsNone(hook_args.err_msg)
                self.assertEqual(hook_args.object, obj)

                err_formatunraisable(exc, "custom hook %R", obj)
                self.assertIs(hook_args.exc_type, type(exc))
                self.assertIs(hook_args.exc_value, exc)
                self.assertIs(hook_args.exc_traceback, exc.__traceback__)
                self.assertEqual(hook_args.err_msg, f'custom hook {obj!r}')
                self.assertIsNone(hook_args.object)
        finally:
            # expected and hook_args contain an exception: break reference cycle
            expected = None
            hook_args = None

    def test_custom_unraisablehook_fail(self):
        # [FUNCTION DOC] If the custom hook itself raises, stderr reports that the exception was
        # ignored in the hook, with a traceback and the hook's error message.
        _testcapi = import_helper.import_module('_testcapi')
        from _testcapi import err_writeunraisable
        def hook_func(*args):
            # [FUNCTION DOC] HELPER: Hook that always raises an Exception saying hook_func failed.
            raise Exception("hook_func failed")

        with test.support.captured_output("stderr") as stderr:
            with test.support.swap_attr(sys, 'unraisablehook', hook_func):
                err_writeunraisable(ValueError(42), "custom hook fail")

        err = stderr.getvalue()
        self.assertIn(f'Exception ignored in sys.unraisablehook: '
                      f'{hook_func!r}\n',
                      err)
        self.assertIn('Traceback (most recent call last):\n', err)
        self.assertIn('Exception: hook_func failed\n', err)


@test.support.cpython_only
class SizeofTest(__TestCase):
    # [CLASS DOC] CPython-only tests of sys.getsizeof and the memory size of builtin objects.
    # setUp stores pointer size, digit size, GC header size and managed pre-header size;
    # check_sizeof is the support helper comparing getsizeof to an expected total.

    def setUp(self):
        # [FUNCTION DOC] Stores the pointer size, long digit size, GC header size and managed
        # pre-header size (from the internal C API test module) on the instance, then calls the
        # parent setUp.
        self.P = struct.calcsize('P')
        self.longdigit = sys.int_info.sizeof_digit
        _testinternalcapi = import_helper.import_module("_testinternalcapi")
        self.gc_headsize = _testinternalcapi.SIZEOF_PYGC_HEAD
        self.managed_pre_header_size = _testinternalcapi.SIZEOF_MANAGED_PRE_HEADER
        super().setUp()

    check_sizeof = test.support.check_sizeof

    def test_gc_head_size(self):
        # [FUNCTION DOC] Checks that the GC header size is included only for GC-tracked objects:
        # bool has none, an empty list does.
        # Check that the gc header size is added to objects tracked by the gc.
        vsize = test.support.calcvobjsize
        gc_header_size = self.gc_headsize
        # bool objects are not gc tracked
        self.assertEqual(sys.getsizeof(True), vsize('') + self.longdigit)
        # but lists are
        self.assertEqual(sys.getsizeof([]), vsize('Pn') + gc_header_size)

    def test_errors(self):
        # [FUNCTION DOC] getsizeof error handling: a __sizeof__ raising propagates; returning None
        # or a float raises TypeError, or returns the supplied default; boundary and negative
        # sizes follow overflow rules (largest allowed value adds GC and pre-header sizes, one
        # more raises OverflowError, negative raises ValueError or OverflowError).
        class BadSizeof:
            # [CLASS DOC] HELPER: Class whose __sizeof__ raises ValueError.
            def __sizeof__(self):
                # [FUNCTION DOC] HELPER: Raises ValueError.
                raise ValueError
        self.assertRaises(ValueError, sys.getsizeof, BadSizeof())

        class InvalidSizeof:
            # [CLASS DOC] HELPER: Class whose __sizeof__ returns None.
            def __sizeof__(self):
                # [FUNCTION DOC] HELPER: Returns None.
                return None
        self.assertRaises(TypeError, sys.getsizeof, InvalidSizeof())
        sentinel = ["sentinel"]
        self.assertIs(sys.getsizeof(InvalidSizeof(), sentinel), sentinel)

        class FloatSizeof:
            # [CLASS DOC] HELPER: Class whose __sizeof__ returns a float.
            def __sizeof__(self):
                # [FUNCTION DOC] HELPER: Returns the float 4.5.
                return 4.5
        self.assertRaises(TypeError, sys.getsizeof, FloatSizeof())
        self.assertIs(sys.getsizeof(FloatSizeof(), sentinel), sentinel)

        class OverflowSizeof(int):
            # [CLASS DOC] HELPER: Int subclass whose __sizeof__ returns its own integer value.
            def __sizeof__(self):
                # [FUNCTION DOC] HELPER: Returns the integer value of the instance.
                return int(self)
        self.assertEqual(sys.getsizeof(OverflowSizeof(sys.maxsize)),
                         sys.maxsize + self.gc_headsize + self.managed_pre_header_size)
        with self.assertRaises(OverflowError):
            sys.getsizeof(OverflowSizeof(sys.maxsize + 1))
        with self.assertRaises(ValueError):
            sys.getsizeof(OverflowSizeof(-1))
        with self.assertRaises((ValueError, OverflowError)):
            sys.getsizeof(OverflowSizeof(-sys.maxsize - 1))

    def test_default(self):
        # [FUNCTION DOC] getsizeof of True equals the base size plus one digit, with or without a
        # default argument.
        size = test.support.calcvobjsize
        self.assertEqual(sys.getsizeof(True), size('') + self.longdigit)
        self.assertEqual(sys.getsizeof(True, -1), size('') + self.longdigit)

    @torch._dynamo.error_on_graph_break(False)
    def test_objecttypes(self):
        # [FUNCTION DOC] Checks exact getsizeof values for many builtin object types defined in
        # the Objects directory (bool, builtin function, bytearray and iterator, bytes, cell, code
        # (lower bound), complex, descriptors, empty and populated dicts with string and non-
        # string keys, dict views and iterators, and further types after this point in the test).
        # check all types defined in Objects/
        calcsize = struct.calcsize
        size = test.support.calcobjsize
        vsize = test.support.calcvobjsize
        check = self.check_sizeof
        # bool
        check(True, vsize('') + self.longdigit)
        check(False, vsize('') + self.longdigit)
        # buffer
        # XXX
        # builtin_function_or_method
        check(len, size('5P'))
        # bytearray
        samples = [b'', b'u'*100000]
        for sample in samples:
            x = bytearray(sample)
            check(x, vsize('n2Pi') + x.__alloc__())
        # bytearray_iterator
        check(iter(bytearray()), size('nP'))
        # bytes
        check(b'', vsize('n') + 1)
        check(b'x' * 10, vsize('n') + 11)
        # cell
        def get_cell():
            # [FUNCTION DOC] HELPER: Returns a closure over a local variable, used to obtain a
            # cell and code object.
            x = 42
            def inner():
                # [FUNCTION DOC] HELPER: Returns the captured variable x.
                return x
            return inner
        check(get_cell().__closure__[0], size('P'))
        # code
        def check_code_size(a, expected_size):
            # [FUNCTION DOC] HELPER: Asserts the size of object a is at least expected_size.
            self.assertGreaterEqual(sys.getsizeof(a), expected_size)
        check_code_size(get_cell().__code__, size('6i13P'))
        check_code_size(get_cell.__code__, size('6i13P'))
        def get_cell2(x):
            # [FUNCTION DOC] HELPER: Like get_cell but takes x as an argument; used for a code
            # object with a parameter.
            def inner():
                # [FUNCTION DOC] HELPER: Returns the captured variable x.
                return x
            return inner
        check_code_size(get_cell2.__code__, size('6i13P') + calcsize('n'))
        # complex
        check(complex(0,1), size('2d'))
        # method_descriptor (descriptor object)
        check(str.lower, size('3PPP'))
        # classmethod_descriptor (descriptor object)
        # XXX
        # member_descriptor (descriptor object)
        import datetime
        check(datetime.timedelta.days, size('3PP'))
        # getset_descriptor (descriptor object)
        import collections
        check(collections.defaultdict.default_factory, size('3PP'))
        # wrapper_descriptor (descriptor object)
        check(int.__add__, size('3P2P'))
        # method-wrapper (descriptor object)
        check({}.__iter__, size('2P'))
        # empty dict
        check({}, size('nQ2P'))
        # dict (string key)
        check({"a": 1}, size('nQ2P') + calcsize(DICT_KEY_STRUCT_FORMAT) + 8 + (8*2//3)*calcsize('2P'))
        longdict = {str(i): i for i in range(8)}
        check(longdict, size('nQ2P') + calcsize(DICT_KEY_STRUCT_FORMAT) + 16 + (16*2//3)*calcsize('2P'))
        # dict (non-string key)
        check({1: 1}, size('nQ2P') + calcsize(DICT_KEY_STRUCT_FORMAT) + 8 + (8*2//3)*calcsize('n2P'))
        longdict = {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8}
        check(longdict, size('nQ2P') + calcsize(DICT_KEY_STRUCT_FORMAT) + 16 + (16*2//3)*calcsize('n2P'))
        # dictionary-keyview
        check({}.keys(), size('P'))
        # dictionary-valueview
        check({}.values(), size('P'))
        # dictionary-itemview
        check({}.items(), size('P'))
        # dictionary iterator
        check(iter({}), size('P2nPn'))
        # dictionary-keyiterator
        check(iter({}.keys()), size('P2nPn'))
        # dictionary-valueiterator
        check(iter({}.values()), size('P2nPn'))
        # dictionary-itemiterator
        check(iter({}.items()), size('P2nPn'))
        # dictproxy
        # [CLASS DOC] HELPER: Empty class (one-line definition); its __dict__ view
        # is size-checked.
        class C(object): pass
        check(C.__dict__, size('P'))
        # BaseException
        check(BaseException(), size('6Pb'))
        # UnicodeEncodeError
        check(UnicodeEncodeError("", "", 0, 0, ""), size('6Pb 2P2nP'))
        # UnicodeDecodeError
        check(UnicodeDecodeError("", b"", 0, 0, ""), size('6Pb 2P2nP'))
        # UnicodeTranslateError
        check(UnicodeTranslateError("", 0, 1, ""), size('6Pb 2P2nP'))
        # ellipses
        check(Ellipsis, size(''))
        # EncodingMap
        import codecs, encodings.iso8859_3
        x = codecs.charmap_build(encodings.iso8859_3.decoding_table)
        check(x, size('32B2iB'))
        # enumerate
        check(enumerate([]), size('n4P'))
        # reverse
        check(reversed(''), size('nP'))
        # float
        check(float(0), size('d'))
        # sys.floatinfo
        check(sys.float_info, vsize('') + self.P * len(sys.float_info))
        # frame
        def func():
            # [FUNCTION DOC] HELPER: Returns the current frame object.
            return sys._getframe()
        x = func()
        check(x, size('3Pi2c2P7P2ic??2P'))
        # function
        # [FUNCTION DOC] HELPER: Empty function (one-line definition) whose function object size
        # is checked.
        def func(): pass
        check(func, size('15Pi'))
        class c():
            # [CLASS DOC] HELPER: Class holding a static method and a class method, whose wrapper
            # sizes are checked.
            @staticmethod
            def foo():
                # [FUNCTION DOC] HELPER: Static method that does nothing.
                pass
            @classmethod
            def bar(cls):
                # [FUNCTION DOC] HELPER: Class method that does nothing.
                pass
            # staticmethod
            check(foo, size('PP'))
            # classmethod
            check(bar, size('PP'))
        # generator
        # [FUNCTION DOC] HELPER: Generator function (one-line definition) yielding 1, used to get
        # a generator object.
        def get_gen(): yield 1
        check(get_gen(), size('PP4P4c7P2ic??2P'))
        # iterator
        check(iter('abc'), size('lP'))
        # callable-iterator
        import re
        check(re.finditer('',''), size('2P'))
        # list
        check(list([]), vsize('Pn'))
        check(list([1]), vsize('Pn') + 2*self.P)
        check(list([1, 2]), vsize('Pn') + 2*self.P)
        check(list([1, 2, 3]), vsize('Pn') + 4*self.P)
        # sortwrapper (list)
        # XXX
        # cmpwrapper (list)
        # XXX
        # listiterator (list)
        check(iter([]), size('lP'))
        # listreverseiterator (list)
        check(reversed([]), size('nP'))
        # int
        check(0, vsize('') + self.longdigit)
        check(1, vsize('') + self.longdigit)
        check(-1, vsize('') + self.longdigit)
        PyLong_BASE = 2**sys.int_info.bits_per_digit
        check(int(PyLong_BASE), vsize('') + 2*self.longdigit)
        check(int(PyLong_BASE**2-1), vsize('') + 2*self.longdigit)
        check(int(PyLong_BASE**2), vsize('') + 3*self.longdigit)
        # module
        if support.Py_GIL_DISABLED:
            check(unittest, size('PPPPPP'))
        else:
            check(unittest, size('PPPPP'))
        # None
        check(None, size(''))
        # NotImplementedType
        check(NotImplemented, size(''))
        # object
        check(object(), size(''))
        # property (descriptor object)
        class C(object):
            # [CLASS DOC] HELPER: Class with accessor methods and a property built from them,
            # whose size is checked.
            # [FUNCTION DOC] HELPER: Property getter (one-line definition) returning the
            # private attribute.
            def getx(self): return self.__x
            # [FUNCTION DOC] HELPER: Property setter (one-line definition) storing the
            # private attribute.
            def setx(self, value): self.__x = value
            # [FUNCTION DOC] HELPER: Property deleter (one-line definition) removing the
            # private attribute.
            def delx(self): del self.__x
            x = property(getx, setx, delx, "")
            check(x, size('5Pi'))
        # PyCapsule
        check(_datetime.datetime_CAPI, size('6P'))
        # rangeiterator
        check(iter(range(1)), size('3l'))
        check(iter(range(2**65)), size('3P'))
        # reverse
        check(reversed(''), size('nP'))
        # range
        check(range(1), size('4P'))
        check(range(66000), size('4P'))
        # set
        # frozenset
        PySet_MINSIZE = 8
        samples = [[], range(10), range(50)]
        s = size('3nP' + PySet_MINSIZE*'nP' + '2nP')
        for sample in samples:
            minused = len(sample)
            if minused == 0: tmp = 1
            # the computation of minused is actually a bit more complicated
            # but this suffices for the sizeof test
            minused = minused*2
            newsize = PySet_MINSIZE
            while newsize <= minused:
                newsize = newsize << 1
            if newsize <= 8:
                check(set(sample), s)
                check(frozenset(sample), s)
            else:
                check(set(sample), s + newsize*calcsize('nP'))
                check(frozenset(sample), s + newsize*calcsize('nP'))
        # setiterator
        check(iter(set()), size('P3n'))
        # slice
        check(slice(0), size('3P'))
        # super
        check(super(int), size('3P'))
        # tuple
        check((), vsize(''))
        check((1,2,3), vsize('') + 3*self.P)
        # type
        # static type: PyTypeObject
        fmt = 'P2nPI13Pl4Pn9Pn12PIPc'
        s = vsize(fmt)
        check(int, s)
        # class
        s = vsize(fmt +                 # PyTypeObject
                  '4P'                  # PyAsyncMethods
                  '36P'                 # PyNumberMethods
                  '3P'                  # PyMappingMethods
                  '10P'                 # PySequenceMethods
                  '2P'                  # PyBufferProcs
                  '6P'
                  '1PIP'                 # Specializer cache
                  )
        # [CLASS DOC] HELPER: Empty new-style class (one-line definition) whose type and instance
        # dict sizes are checked.
        class newstyleclass(object): pass
        # Separate block for PyDictKeysObject with 8 keys and 5 entries
        check(newstyleclass, s + calcsize(DICT_KEY_STRUCT_FORMAT) + 64 + 42*calcsize("2P"))
        # dict with shared keys
        [newstyleclass() for _ in range(100)]
        check(newstyleclass().__dict__, size('nQ2P') + self.P)
        o = newstyleclass()
        o.a = o.b = o.c = o.d = o.e = o.f = o.g = o.h = 1
        # Separate block for PyDictKeysObject with 16 keys and 10 entries
        check(newstyleclass, s + calcsize(DICT_KEY_STRUCT_FORMAT) + 64 + 42*calcsize("2P"))
        # dict with shared keys
        check(newstyleclass().__dict__, size('nQ2P') + self.P)
        # unicode
        # each tuple contains a string and its expected character size
        # don't put any static strings here, as they may contain
        # wchar_t or UTF-8 representations
        samples = ['1'*100, '\xff'*50,
                   '\u0100'*40, '\uffff'*100,
                   '\U00010000'*30, '\U0010ffff'*100]
        # also update field definitions in test_unicode.test_raiseMemError
        asciifields = "nnb"
        compactfields = asciifields + "nP"
        unicodefields = compactfields + "P"
        for s in samples:
            maxchar = ord(max(s))
            if maxchar < 128:
                L = size(asciifields) + len(s) + 1
            elif maxchar < 256:
                L = size(compactfields) + len(s) + 1
            elif maxchar < 65536:
                L = size(compactfields) + 2*(len(s) + 1)
            else:
                L = size(compactfields) + 4*(len(s) + 1)
            check(s, L)
        # verify that the UTF-8 size is accounted for
        s = chr(0x4000)   # 4 bytes canonical representation
        check(s, size(compactfields) + 4)
        # compile() will trigger the generation of the UTF-8
        # representation as a side effect
        compile(s, "<stdin>", "eval")
        check(s, size(compactfields) + 4 + 4)
        # TODO: add check that forces the presence of wchar_t representation
        # TODO: add check that forces layout of unicodefields
        # weakref
        import weakref
        if support.Py_GIL_DISABLED:
            expected = size('2Pn4P')
        else:
            expected = size('2Pn3P')
        check(weakref.ref(int), expected)
        # weakproxy
        # XXX
        # weakcallableproxy
        check(weakref.proxy(int), expected)

    def check_slots(self, obj, base, extra):
        # [FUNCTION DOC] HELPER: Asserts the size of obj equals the size of its base object plus
        # the C size of the extra slot layout, adding the GC header size when obj is GC-tracked
        # but base is not.
        expected = sys.getsizeof(base) + struct.calcsize(extra)
        if gc.is_tracked(obj) and not gc.is_tracked(base):
            expected += self.gc_headsize
        self.assertEqual(sys.getsizeof(obj), expected)

    def test_slots(self):
        # [FUNCTION DOC] For subclasses of bytearray, dict, list, set, frozenset and OrderedDict
        # that add three __slots__ names, checks each instance's size is the base type's size plus
        # three pointers.
        # check all subclassable types defined in Objects/ that allow
        # non-empty __slots__
        check = self.check_slots
        class BA(bytearray):
            # [CLASS DOC] HELPER: bytearray subclass with three slots.
            __slots__ = 'a', 'b', 'c'
        check(BA(), bytearray(), '3P')
        class D(dict):
            # [CLASS DOC] HELPER: dict subclass with three slots.
            __slots__ = 'a', 'b', 'c'
        check(D(x=[]), {'x': []}, '3P')
        class L(list):
            # [CLASS DOC] HELPER: list subclass with three slots.
            __slots__ = 'a', 'b', 'c'
        check(L(), [], '3P')
        class S(set):
            # [CLASS DOC] HELPER: set subclass with three slots.
            __slots__ = 'a', 'b', 'c'
        check(S(), set(), '3P')
        class FS(frozenset):
            # [CLASS DOC] HELPER: frozenset subclass with three slots.
            __slots__ = 'a', 'b', 'c'
        check(FS(), frozenset(), '3P')
        from collections import OrderedDict
        class OD(OrderedDict):
            # [CLASS DOC] HELPER: OrderedDict subclass with three slots.
            __slots__ = 'a', 'b', 'c'
        check(OD(x=[]), OrderedDict(x=[]), '3P')

    def test_pythontypes(self):
        # [FUNCTION DOC] Size checks for types defined in the Python directory: an AST node, a
        # traceback object when one is available, and sys.flags (with a FIXME allowing one
        # extra pointer).
        # check all types defined in Python/
        size = test.support.calcobjsize
        vsize = test.support.calcvobjsize
        check = self.check_sizeof
        # _ast.AST
        import _ast
        check(_ast.AST(), size('P'))
        try:
            raise TypeError
        except TypeError as e:
            tb = e.__traceback__
            # traceback
            if tb is not None:
                check(tb, size('2P2i'))
        # symtable entry
        # XXX
        # sys.flags
        # FIXME: The +1 will not be necessary once gh-122575 is fixed
        check(sys.flags, vsize('') + self.P * (1 + len(sys.flags)))

    def test_asyncgen_hooks(self):
        # [FUNCTION DOC] Async-generator hooks: defaults are None; setting an invalid finalizer or
        # first-iteration hook raises TypeError and leaves both unchanged (gh-118473); setting
        # either hook alone keeps the other, and both access by name and by index work; restoring
        # the old pair resets to None.
        old = sys.get_asyncgen_hooks()
        self.assertIsNone(old.firstiter)
        self.assertIsNone(old.finalizer)

        firstiter = lambda *a: None
        finalizer = lambda *a: None

        with self.assertRaises(TypeError):
            sys.set_asyncgen_hooks(firstiter=firstiter, finalizer="invalid")
        cur = sys.get_asyncgen_hooks()
        self.assertIsNone(cur.firstiter)
        self.assertIsNone(cur.finalizer)

        # gh-118473
        with self.assertRaises(TypeError):
            sys.set_asyncgen_hooks(firstiter="invalid", finalizer=finalizer)
        cur = sys.get_asyncgen_hooks()
        self.assertIsNone(cur.firstiter)
        self.assertIsNone(cur.finalizer)

        sys.set_asyncgen_hooks(firstiter=firstiter)
        hooks = sys.get_asyncgen_hooks()
        self.assertIs(hooks.firstiter, firstiter)
        self.assertIs(hooks[0], firstiter)
        self.assertIs(hooks.finalizer, None)
        self.assertIs(hooks[1], None)

        sys.set_asyncgen_hooks(finalizer=finalizer)
        hooks = sys.get_asyncgen_hooks()
        self.assertIs(hooks.firstiter, firstiter)
        self.assertIs(hooks[0], firstiter)
        self.assertIs(hooks.finalizer, finalizer)
        self.assertIs(hooks[1], finalizer)

        sys.set_asyncgen_hooks(*old)
        cur = sys.get_asyncgen_hooks()
        self.assertIsNone(cur.firstiter)
        self.assertIsNone(cur.finalizer)

    def test_changing_sys_stderr_and_removing_reference(self):
        # [FUNCTION DOC] bpo-43660: a replaced stderr whose write method clears sys.stderr must
        # not crash the interpreter when an uncaught error is reported; the subprocess must fail
        # with empty stdout and stderr.
        # If the default displayhook doesn't take a strong reference
        # to sys.stderr the following code can crash. See bpo-43660
        # for more details.
        code = textwrap.dedent('''
            import sys
            class MyStderr:
                def write(self, s):
                    sys.stderr = None
            sys.stderr = MyStderr()
            1/0
        ''')
        rc, out, err = assert_python_failure('-c', code)
        self.assertEqual(out, b"")
        self.assertEqual(err, b"")

if __name__ == "__main__":
    run_tests()
