# Owner(s): ["module: functorch"]
# [MODULE DOC]
# Tests that with the debug_assert config enabled, the compiled wrapper's assertion function
# is codegen'd and a "debug_assert_wrapper" trace artifact is emitted exactly once per
# compile. Covers mixed, all, and partly requires_grad inputs under aot_eager.

"""
Tests for codegen'ing DebugAssertWrapper.

The codegen'd assertion function emits checks only for the specific arg
indices where requires_grad was False at compile time. Positions where
requires_grad=True are safe and generate no runtime check, replacing
the closure that iterated over all args.

Enabled via torch._functorch.config.debug_assert = True.

Tests verify that a "debug_assert_wrapper" artifact is emitted via
trace_structured.
"""

import logging
from contextlib import contextmanager

import torch
import torch._functorch.config
from torch.testing._internal.common_utils import run_tests, TestCase


trace_log = logging.getLogger("torch.__trace")


class TestCodegenDebugAssert(TestCase):
    # [CLASS DOC]
    # Test case for the debug-assert codegen, with a helper context manager that collects
    # trace artifacts by name; each test is patched to enable the debug_assert config.
    @contextmanager
    def _capture_codegen_source(self, artifact_name):
        # [FUNCTION DOC] HELPER: Context manager yielding a list of payloads of trace
        # artifacts with the given name. Attaches a handler to the trace logger at debug
        # level and restores the logger level and handler list on exit.
        """Capture codegen artifacts from the structured trace log."""
        captured: list[str] = []

        class _ArtifactHandler(logging.Handler):
        # [CLASS DOC] HELPER: Logging handler appending the payload of records whose artifact
        # metadata name matches the wanted name to the enclosing captured list.
            def emit(self, record):
            # [FUNCTION DOC] HELPER: Appends the record payload to the captured list if the
            # record metadata carries a matching artifact name and a non-missing payload.
                metadata = getattr(record, "metadata", {})
                if (
                    "artifact" in metadata
                    and metadata["artifact"].get("name") == artifact_name
                ):
                    payload = getattr(record, "payload", None)
                    if payload is not None:
                        captured.append(payload)

        handler = _ArtifactHandler()
        handler.setLevel(logging.DEBUG)
        old_level = trace_log.level
        trace_log.setLevel(logging.DEBUG)
        trace_log.addHandler(handler)
        try:
            yield captured
        finally:
            trace_log.removeHandler(handler)
            trace_log.setLevel(old_level)

    @torch._functorch.config.patch(debug_assert=True)
    def test_mixed_requires_grad(self):
        # [FUNCTION DOC] Compiles an addition of one grad-requiring and one plain tensor and
        # checks the result equals eager and exactly one debug_assert_wrapper artifact is
        # emitted.
        """
        With debug_assert=True, the wrapper should codegen assertions for
        inputs compiled without requires_grad, and skip those with
        requires_grad=True.
        """
        with self._capture_codegen_source("debug_assert_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x, y):
                return x + y

            x = torch.randn(4, requires_grad=True)
            y = torch.randn(4)
            out = f(x, y)

        self.assertEqual(out, x + y)

        self.assertEqual(len(captured), 1)

    @torch._functorch.config.patch(debug_assert=True)
    def test_all_requires_grad(self):
        # [FUNCTION DOC] Compiles a sum of three grad-requiring tensors; checks the result
        # and that the artifact is still emitted exactly once even when no position needs a
        # runtime check.
        """
        All inputs with requires_grad=True. Codegen should emit no
        assertions (all positions are safe), but the artifact should
        still be emitted.
        """
        with self._capture_codegen_source("debug_assert_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x, y, z):
                return x + y + z

            x = torch.randn(4, requires_grad=True)
            y = torch.randn(4, requires_grad=True)
            z = torch.randn(4, requires_grad=True)
            out = f(x, y, z)

        self.assertEqual(out, x + y + z)

        self.assertEqual(len(captured), 1)

    @torch._functorch.config.patch(debug_assert=True)
    def test_some_no_grad_inputs(self):
        # [FUNCTION DOC] Compiles a function over one grad-requiring and two plain tensors;
        # checks the result equals eager and exactly one debug_assert_wrapper artifact is
        # emitted.
        """
        Mix of requires_grad and non-requires_grad inputs going through
        the training path. Codegen should emit assertions only for the
        non-grad positions.
        """
        with self._capture_codegen_source("debug_assert_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x, y, z):
                return x * y + z

            x = torch.randn(4, requires_grad=True)
            y = torch.randn(4)
            z = torch.randn(4)
            out = f(x, y, z)

        self.assertEqual(out, x * y + z)

        self.assertEqual(len(captured), 1)


if __name__ == "__main__":
    run_tests()
