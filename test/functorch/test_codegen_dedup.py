# Owner(s): ["module: functorch"]
# [MODULE DOC]
# Tests the codegen'd duplicate-argument removal of the dedup wrapper (strategy 2, used when
# duplicated args are mutated). Functions are compiled with aot_function and a pass-through
# compiler, and the captured "dedup_wrapper" artifact source is inspected for which argument
# indices it selects.

"""
Tests for codegen'ing AOTDedupeWrapper.

The codegen'd remove_dupe_args emits straight-line index selections like
[args[0], args[2], args[5]] with all indices baked in as literals, replacing
the closure-based zip + filter over keep_arg_mask.

Strategy 2 (the dedup post_compile path) is triggered when duplicate args
have mutations, so strategy 1 (leafification) can't handle them. Dynamo
already deduplicates inputs, so these tests use aot_function directly.

Tests verify that a "dedup_wrapper" artifact is emitted via trace_structured.
"""

import logging
from contextlib import contextmanager

import torch
import torch._functorch.config
from torch._functorch.aot_autograd import aot_function
from torch.testing._internal.common_utils import run_tests, TestCase


trace_log = logging.getLogger("torch.__trace")


def _nop_compiler(gm, example_inputs):  # type: ignore[no-untyped-def]
    # [FUNCTION DOC] HELPER: Trivial compiler backend that returns the traced graph module's
    # forward method unchanged.
    return gm.forward


class TestCodegenDedup(TestCase):
    # [CLASS DOC]
    # Test case for dedup codegen: a helper context manager records trace artifacts by name
    # and three tests mutate duplicated inputs and check results and generated source.
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

    def test_duplicate_args_with_mutation(self):
        # [FUNCTION DOC] Passes one tensor as both args while the second is mutated in place
        # (multiplied by two). Checks the input is doubled, the output matches, and the
        # generated source mentions the first arg index but not the second.
        """
        When the same tensor is passed as two args and one position mutates,
        strategy 2 dedup kicks in. The codegen should emit straight-line
        arg selection.
        """
        with self._capture_codegen_source("dedup_wrapper") as captured:

            def f(a, b):
                # [FUNCTION DOC] HELPER: Traced function that mutates its second arg in
                # place (multiply by two) and returns the sum of both args.
                b.mul_(2)
                return a + b

            compiled_f = aot_function(f, _nop_compiler)
            x = torch.randn(4)
            x_ref = x.clone()
            out = compiled_f(x, x)

        self.assertEqual(x, x_ref * 2)
        self.assertEqual(out, x_ref * 4)

        self.assertGreaterEqual(
            len(captured),
            1,
            "Expected dedup_wrapper codegen artifact to be emitted",
        )
        source = captured[0]
        self.assertIn("args[0]", source)
        self.assertNotIn("args[1]", source)

    def test_three_way_duplicate_with_mutation(self):
        # [FUNCTION DOC] Passes one tensor as all three args with the last mutated in place
        # (add one). Checks input and output values and that the generated source selects
        # only the first arg index.
        """
        Three-way duplication where the last position mutates.
        """
        with self._capture_codegen_source("dedup_wrapper") as captured:

            def f(a, b, c):
                # [FUNCTION DOC] HELPER: Traced function that adds one to its third arg
                # in place and returns the sum of all three args.
                c.add_(1)
                return a + b + c

            compiled_f = aot_function(f, _nop_compiler)
            x = torch.randn(4)
            x_ref = x.clone()
            out = compiled_f(x, x, x)

        self.assertEqual(x, x_ref + 1)
        self.assertEqual(out, (x_ref + 1) * 3)

        self.assertGreaterEqual(
            len(captured),
            1,
            "Expected dedup_wrapper codegen artifact to be emitted",
        )
        source = captured[0]
        self.assertIn("args[0]", source)
        self.assertNotIn("args[1]", source)
        self.assertNotIn("args[2]", source)

    def test_partial_duplicate_with_mutation(self):
        # [FUNCTION DOC] Passes one tensor as the first two args and a distinct tensor as
        # the third, mutating the duplicate second arg (multiply by three). Checks values and
        # that the source selects the first and third arg indices but not the second.
        """
        Two args are duplicates (with mutation), third is distinct.
        Codegen should select [args[0], args[2]] dropping the duplicate.
        """
        with self._capture_codegen_source("dedup_wrapper") as captured:

            def f(a, b, c):
                # [FUNCTION DOC] HELPER: Traced function that multiplies its second arg
                # by three in place and returns the sum of all three args.
                b.mul_(3)
                return a + b + c

            compiled_f = aot_function(f, _nop_compiler)
            x = torch.randn(4)
            y = torch.randn(4)
            x_ref = x.clone()
            y_ref = y.clone()
            out = compiled_f(x, x, y)

        self.assertEqual(x, x_ref * 3)
        self.assertEqual(out, x_ref * 3 + x_ref * 3 + y_ref)

        self.assertGreaterEqual(
            len(captured),
            1,
            "Expected dedup_wrapper codegen artifact to be emitted",
        )
        source = captured[0]
        self.assertIn("args[0]", source)
        self.assertNotIn("args[1]", source)
        self.assertIn("args[2]", source)


if __name__ == "__main__":
    run_tests()
