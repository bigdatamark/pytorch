# [MODULE DOC]
# Dynamo test base classes: run_tests (a guarded wrapper around the common test runner),
# TestCase (patches Dynamo config per class, resets counters and grad-mode checks per
# test) and CPythonTestCase (for CPython test-suite ports traced by Dynamo).
# No significant import-time side effects; a module logger is created.
"""Testing utilities for Dynamo, providing a specialized TestCase class and test running functionality.

This module extends PyTorch's testing framework with Dynamo-specific testing capabilities.
It includes:
- A custom TestCase class that handles Dynamo-specific setup/teardown
- Test running utilities with dependency checking
- Automatic reset of Dynamo state between tests
- Proper handling of gradient mode state
"""

import contextlib
import importlib
import inspect
import logging
import os
import re
import sys
import unittest
from collections.abc import Callable
from typing import Any

import torch
import torch.testing
from torch._dynamo import polyfills
from torch._logging._internal import trace_log
from torch.testing._internal.common_utils import (  # type: ignore[attr-defined]
    IS_WINDOWS,
    TEST_WITH_CROSSREF,
    TEST_WITH_TORCHDYNAMO,
    TestCase as TorchTestCase,
)

from . import config, utils


log = logging.getLogger(__name__)


def run_tests(needs: str | tuple[str, ...] = ()) -> None:
    # [FUNCTION DOC] Runs the test suite of the calling module unless it should be skipped.
    # needs: a module name or tuple of names that must be importable ("cuda" instead
    # requires CUDA availability). Returns silently without running anything when
    # running under TorchDynamo/crossref, on Windows without XPU unless
    # TORCHINDUCTOR_WINDOWS_TESTS is set, or when a requirement is missing.
    from torch.testing._internal.common_utils import run_tests

    if TEST_WITH_TORCHDYNAMO or TEST_WITH_CROSSREF:
        return  # skip testing

    if (
        not torch.xpu.is_available()
        and IS_WINDOWS
        and os.environ.get("TORCHINDUCTOR_WINDOWS_TESTS", "0") == "0"
    ):
        return

    if isinstance(needs, str):
        needs = (needs,)
    for need in needs:
        if need == "cuda":
            if not torch.cuda.is_available():
                return
        else:
            try:
                importlib.import_module(need)
            except ImportError:
                return

    run_tests()


class TestCase(TorchTestCase):
    # [CLASS DOC] Base class for Dynamo tests. Class-level state: _exit_stack holding a
    # config patch (strict errors, no metrics, canonical node order) active for the
    # whole class. Per test it clears counters, enables nested graph breaks, installs a
    # null trace_log handler and restores grad mode and config afterward. Also disables
    # some assertions when debug_disable_compile_counter is set.
    _exit_stack: contextlib.ExitStack

    @classmethod
    def tearDownClass(cls) -> None:
        # [FUNCTION DOC] Closes the class-wide exit stack (undoing the config patch)
        # and then calls the parent tearDownClass.
        cls._exit_stack.close()
        super().tearDownClass()

    @classmethod
    def setUpClass(cls) -> None:
        # [FUNCTION DOC] Calls the parent setUpClass, then creates the exit stack and
        # enters a Dynamo config patch (raise_on_ctx_manager_usage, no suppress_errors,
        # no compilation metrics logging, canonicalized output graph node order).
        super().setUpClass()
        cls._exit_stack = contextlib.ExitStack()  # type: ignore[attr-defined]
        cls._exit_stack.enter_context(  # type: ignore[attr-defined]
            config.patch(
                raise_on_ctx_manager_usage=True,
                suppress_errors=False,
                log_compilation_metrics=False,
                canonicalize_output_graph_node_order=True,
            ),
        )

    def setUp(self) -> None:
        # [FUNCTION DOC] Records the current grad mode and nested_graph_breaks config,
        # forces nested_graph_breaks on, calls parent setUp, clears Dynamo counters and
        # attaches a null logging handler to trace_log (so structured tracing is active).
        self._prior_is_grad_enabled = torch.is_grad_enabled()
        self._prior_nested_graph_breaks = config.nested_graph_breaks
        config.nested_graph_breaks = True
        super().setUp()
        utils.counters.clear()
        self.handler = logging.NullHandler()
        trace_log.addHandler(self.handler)

    def tearDown(self) -> None:
        # [FUNCTION DOC] Removes the trace_log handler, logs and clears counters,
        # re-enables saved tensor hooks, calls parent tearDown, restores grad mode
        # (with a warning if the test changed it) and restores nested_graph_breaks.
        trace_log.removeHandler(self.handler)
        for k, v in utils.counters.items():
            log.debug("%s %s", k, v.most_common())
        utils.counters.clear()
        torch._C._autograd._saved_tensors_hooks_enable()
        super().tearDown()
        if self._prior_is_grad_enabled is not torch.is_grad_enabled():
            log.warning("Running test changed grad mode")
            torch.set_grad_enabled(self._prior_is_grad_enabled)
        config.nested_graph_breaks = self._prior_nested_graph_breaks

    def before_cuda_memory_leak_check(self) -> None:
        # [FUNCTION DOC] Hook run before the CUDA memory leak check: calls the parent
        # hook and clears Dynamo counters.
        super().before_cuda_memory_leak_check()
        utils.counters.clear()

    def assertEqual(self, x: Any, y: Any, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        # [FUNCTION DOC] Delegates to the parent assertEqual, except that when
        # config.debug_disable_compile_counter is set it silently skips checks where
        # either side is a CompileCounterInt or where the calling line mentions
        # counters. Raises whatever the parent raises on mismatch.
        if config.debug_disable_compile_counter:
            if isinstance(x, utils.CompileCounterInt) or isinstance(
                y, utils.CompileCounterInt
            ):
                return
            # skip checks like self.assertEqual(len(counters["graph_break"]), 1)
            if (
                (cur_frame := inspect.currentframe())
                and (upper_frame := cur_frame.f_back)
                and (upper_code := inspect.getframeinfo(upper_frame).code_context)
                and "counters" in upper_code[0]
            ):
                return
        return super().assertEqual(x, y, *args, **kwargs)

    def assertExpectedInline(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        # [FUNCTION DOC] Delegates to the parent assertExpectedInline with the skip
        # frame count incremented by one (so the caller's source line is found); does
        # nothing when config.debug_disable_compile_counter is set.
        if config.debug_disable_compile_counter:
            return
        kwargs["skip"] = kwargs.get("skip", 0) + 1
        return super().assertExpectedInline(*args, **kwargs)


class CPythonTestCase(TestCase):
    # [CLASS DOC] Base class for CPython test-suite ports under test/cpython/vX_YY.
    # Restores the plain unittest assertion methods (or traceable polyfills) so Dynamo
    # can trace through them, sets dynamo_strict_nopython, and enables tracing of
    # unittest and load_build_class via a class-level _stack config patch. Tests are
    # skipped unless the running Python version matches the path's version.
    """
    Test class for CPython tests located in "test/cpython/v{Py_version}/*".

    This class enables specific features that are disabled by default, such as
    tracing through unittest methods.
    """

    _stack: contextlib.ExitStack
    dynamo_strict_nopython = True

    # Restore original unittest methods to simplify tracing CPython test cases.
    assertEqual = unittest.TestCase.assertEqual  # type: ignore[assignment]
    assertNotEqual = unittest.TestCase.assertNotEqual  # type: ignore[assignment]
    assertTrue = unittest.TestCase.assertTrue
    assertFalse = unittest.TestCase.assertFalse
    assertIs = unittest.TestCase.assertIs
    assertIsNot = unittest.TestCase.assertIsNot
    assertIsNone = unittest.TestCase.assertIsNone
    assertIsNotNone = unittest.TestCase.assertIsNotNone
    assertIn = unittest.TestCase.assertIn
    assertNotIn = unittest.TestCase.assertNotIn
    assertIsInstance = unittest.TestCase.assertIsInstance
    assertNotIsInstance = unittest.TestCase.assertNotIsInstance
    assertAlmostEqual = unittest.TestCase.assertAlmostEqual
    assertNotAlmostEqual = unittest.TestCase.assertNotAlmostEqual
    assertGreater = unittest.TestCase.assertGreater
    assertGreaterEqual = unittest.TestCase.assertGreaterEqual
    assertLess = unittest.TestCase.assertLess
    assertLessEqual = unittest.TestCase.assertLessEqual
    assertRegex = unittest.TestCase.assertRegex
    assertNotRegex = unittest.TestCase.assertNotRegex
    assertCountEqual = unittest.TestCase.assertCountEqual
    assertMultiLineEqual = polyfills.assert_multi_line_equal
    assertSequenceEqual = polyfills.assert_sequence_equal
    assertListEqual = unittest.TestCase.assertListEqual
    assertTupleEqual = unittest.TestCase.assertTupleEqual
    assertSetEqual = unittest.TestCase.assertSetEqual
    # pyrefly: ignore [bad-override]
    assertDictEqual = polyfills.assert_dict_equal
    # pyrefly: ignore [bad-override]
    assertRaises = unittest.TestCase.assertRaises
    # pyrefly: ignore [bad-override]
    assertRaisesRegex = unittest.TestCase.assertRaisesRegex
    assertWarns = unittest.TestCase.assertWarns
    assertWarnsRegex = unittest.TestCase.assertWarnsRegex
    assertLogs = unittest.TestCase.assertLogs
    fail = unittest.TestCase.fail
    failureException = unittest.TestCase.failureException

    def compile_fn(
        self,
        fn: Callable[..., Any],
        backend: str | Callable[..., Any],
        nopython: bool,
    ) -> Callable[..., Any]:
        # [FUNCTION DOC] Wraps the current test method (not setup code) with
        # torch._dynamo.optimize using the given backend and error_on_graph_break=nopython,
        # replaces the method on the instance, and returns fn unchanged. Mutates self.
        # We want to compile only the test function, excluding any setup code
        # from unittest

        method = getattr(self, self._testMethodName)
        method = torch._dynamo.optimize(backend, error_on_graph_break=nopython)(method)

        setattr(self, self._testMethodName, method)
        return fn

    def _dynamo_test_key(self) -> str:
        # [FUNCTION DOC] HELPER: Builds the test key. Returns the parent key unchanged
        # if the test file path has no version directory; otherwise returns the key
        # prefixed with "CPython", the digits of the version, and the test file name.
        suffix = super()._dynamo_test_key()
        test_cls = self.__class__
        test_file = inspect.getfile(test_cls).split(os.sep)[-1].split(".")[0]
        py_ver = re.search(r"/v([\d_]+)/", inspect.getfile(test_cls))
        if py_ver:
            py_ver = py_ver.group().strip(os.sep).replace("_", "").lstrip("v")  # type: ignore[assignment]
        else:
            return suffix
        return f"CPython{py_ver}-{test_file}-{suffix}"

    @classmethod
    def tearDownClass(cls) -> None:
        # [FUNCTION DOC] Closes the class-level config patch stack, then calls parent.
        cls._stack.close()
        super().tearDownClass()

    @classmethod
    def setUpClass(cls) -> None:
        # [FUNCTION DOC] Raises unittest.SkipTest if the test file path lacks a
        # cpython/vM_NN directory or its version differs from the running Python.
        # Otherwise calls the parent setUpClass and enters a config patch enabling
        # enable_trace_unittest and enable_trace_load_build_class on a new _stack.
        # Skip test if python versions doesn't match
        search_path = inspect.getfile(cls)

        cpython_test_regex = (
            re.escape(os.path.join("cpython") + os.path.sep) + r"v(\d)_(\d{2})"
        )

        m = re.search(cpython_test_regex, search_path)
        if m:
            test_py_ver = tuple(map(int, m.groups()))
            py_ver = sys.version_info[:2]
            if py_ver != test_py_ver:
                expected = ".".join(map(str, test_py_ver))
                got = ".".join(map(str, py_ver))
                raise unittest.SkipTest(
                    f"Test requires Python {expected} but got Python {got}"
                )
        else:
            raise unittest.SkipTest(
                f"Test requires a specific Python version but not found in path {inspect.getfile(cls)}"
            )

        super().setUpClass()
        cls._stack = contextlib.ExitStack()  # type: ignore[attr-defined]
        cls._stack.enter_context(  # type: ignore[attr-defined]
            config.patch(
                enable_trace_unittest=True,
                enable_trace_load_build_class=True,
            ),
        )

    @contextlib.contextmanager
    def subTest(self, *args, **kwargs):
        # [FUNCTION DOC] No-op context manager replacing unittest subTest; ignores all
        # arguments and just yields (avoids pytest constructs Dynamo cannot trace).
        # pytest 9.x addSubTest uses typing._GenericAlias calls that
        # Dynamo cannot trace. Use a no-op subTest instead.
        yield

    # pyrefly: ignore [implicit-any]
    def wrap_with_policy(self, method_name: str, policy: Callable) -> None:
        # [FUNCTION DOC] HELPER: Intentionally a no-op override; disables the parent's
        # policy wrapping of test methods for CPython tests.
        pass
