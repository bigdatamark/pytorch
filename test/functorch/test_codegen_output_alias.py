# Owner(s): ["module: functorch"]
# [MODULE DOC]
# Tests the codegen'd output-alias handling of the runtime wrapper (alias regeneration of
# outputs that view inputs or intermediates) and parts of the output transform wrapper.
# Compiles small functions with aot_eager (or aot_function with nop for metadata mutation),
# checks values, aliasing via data pointers and grads, and inspects the captured source of
# the "output_alias_wrapper" or "compiled_fn_wrapper" trace artifacts.

"""
Tests for codegen'ing the output alias regeneration in
_create_runtime_wrapper.

The codegen'd output alias handler inlines each handler type's logic per
output as straight-line code: NoopAliasHandler becomes a direct fw_outs[i]
reference, IsInputHandler becomes orig_inputs[base_idx], and
AliasOfInput/IntermediateHandler become inline gen_alias_from_base calls
with baked-in indices and metadata.

Tests verify that an "output_alias_wrapper" artifact is emitted via
trace_structured.
"""

import logging
from contextlib import contextmanager
from unittest.mock import patch

import torch
import torch._functorch.config
from functorch.compile import nop
from torch._functorch.aot_autograd import aot_function
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


trace_log = logging.getLogger("torch.__trace")


class TestCodegenOutputAlias(TestCase):
    # [CLASS DOC]
    # Test case for output-alias codegen. setUp resets dynamo; a helper context manager
    # records trace artifacts by name; each test compiles a function and checks results.
    def setUp(self):
        # [FUNCTION DOC] Runs the base class setup and then resets dynamo state so each test
        # compiles from scratch.
        super().setUp()
        torch._dynamo.reset()

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

    def test_output_is_view_of_input(self):
        # [FUNCTION DOC] Returns a flattened view of the input under aot_eager. Checks values,
        # that the output shares storage with the input, and that one output_alias_wrapper
        # artifact was emitted.
        """
        Output that is a view of an input (alias_of_input). Codegen should
        emit gen_alias_from_base(orig_inputs[i], ...) inline.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Compiled function returning a flattened view of its arg.
                return x.view(-1)

            x = torch.randn(2, 3)
            out = f(x)

        self.assertEqual(out, x.view(-1))
        self.assertEqual(out.data_ptr(), x.data_ptr())

        self.assertEqual(
            len(captured),
            1,
            "Expected output_alias_wrapper codegen artifact to be emitted",
        )

    def test_output_is_input(self):
        # [FUNCTION DOC] Mutates the input in place and returns it. Checks the input was
        # doubled, the output is the very same object, and one output_alias_wrapper artifact
        # was emitted.
        """
        Output that IS the input (is_input). Codegen should emit a direct
        reference to orig_inputs[i].
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Compiled function doubling its
                # arg in place and returning the same arg.
                x.mul_(2)
                return x

            x = torch.randn(4)
            x_ref = x.clone()
            out = f(x)

        self.assertEqual(x, x_ref * 2)
        self.assertIs(out, x)

        self.assertEqual(
            len(captured),
            1,
            "Expected output_alias_wrapper codegen artifact to be emitted",
        )

    def test_mixed_alias_and_non_alias_outputs(self):
        # [FUNCTION DOC] Returns one fresh tensor and one view of the input. Checks both values,
        # that only the view shares storage with the input, and one artifact was emitted.
        """
        Multiple outputs: one aliased, one not. Codegen should emit
        gen_alias_from_base for the alias and a noop for the non-alias.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Compiled function returning a doubled
                # copy and a flattened view of its arg.
                return x * 2, x.view(-1)

            x = torch.randn(2, 3)
            out1, out2 = f(x)

        self.assertEqual(out1, x * 2)
        self.assertEqual(out2, x.view(-1))
        self.assertEqual(out2.data_ptr(), x.data_ptr())

        self.assertEqual(
            len(captured),
            1,
            "Expected output_alias_wrapper codegen artifact to be emitted",
        )

    def test_output_alias_with_mutation(self):
        # [FUNCTION DOC] Adds one to the input in place and returns a flattened view. Checks the
        # mutation reached the input, the view output values, and one artifact was emitted.
        """
        Input is mutated AND output is a view of the input. Codegen should
        handle both mutation epilogue and alias regeneration.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Compiled function adding one to
                # its arg in place and returning a flattened view.
                x.add_(1)
                return x.view(-1)

            x = torch.randn(2, 3)
            x_ref = x.clone()
            out = f(x)

        self.assertEqual(x, x_ref + 1)
        self.assertEqual(out, (x_ref + 1).view(-1))

        self.assertEqual(
            len(captured),
            1,
            "Expected output_alias_wrapper codegen artifact to be emitted",
        )

    def test_output_aliases_intermediate(self):
        # [FUNCTION DOC] Returns an intermediate and a view of it, with a grad-requiring input.
        # Checks both values and that one output_alias_wrapper artifact was emitted.
        """
        Output is a view of another output (intermediate), not of an input.
        Triggers AliasOfIntermediateHandler in the codegen.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Compiled function returning an
                # intermediate (arg plus one) and its flattened view.
                y = x + 1
                return y, y.view(-1)

            x = torch.randn(2, 3, requires_grad=True)
            out1, out2 = f(x)

        self.assertEqual(out1, x + 1)
        self.assertEqual(out2, (x + 1).view(-1))

        self.assertEqual(
            len(captured),
            1,
            "Expected output_alias_wrapper codegen artifact to be emitted",
        )

    def test_multiple_views_of_same_input(self):
        # [FUNCTION DOC] Returns two different views of the same input. Checks values, that both
        # outputs share storage with the input, and that one artifact was emitted.
        """
        Two outputs both alias the same input. Codegen should emit two
        separate gen_alias_from_base calls referencing the same base.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Compiled function returning a
                # flattened view and a reshape of its arg.
                return x.view(-1), x.reshape(6)

            x = torch.randn(2, 3)
            out1, out2 = f(x)

        self.assertEqual(out1, x.view(-1))
        self.assertEqual(out2, x.reshape(6))
        self.assertEqual(out1.data_ptr(), x.data_ptr())
        self.assertEqual(out2.data_ptr(), x.data_ptr())

        self.assertEqual(
            len(captured),
            1,
            "Expected output_alias_wrapper codegen artifact to be emitted",
        )

    def test_training_path_view_of_input(self):
        # [FUNCTION DOC] Returns a view of a grad-requiring input (training path). Checks values
        # and aliasing, runs backward and expects a gradient of ones, and one artifact emitted.
        """
        Training path (trace_joint=True): output is a view of input with
        requires_grad=True. Codegen should use _unwrap_tensoralias in the
        alias function. Also verifies backward correctness.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Compiled function returning a flattened view of its arg.
                return x.view(-1)

            x = torch.randn(2, 3, requires_grad=True)
            out = f(x)

        self.assertEqual(out, x.view(-1))
        self.assertEqual(out.data_ptr(), x.data_ptr())

        out.sum().backward()
        self.assertEqual(x.grad, torch.ones(2, 3))

        self.assertEqual(
            len(captured),
            1,
            "Expected output_alias_wrapper codegen artifact to be emitted",
        )

    def test_training_path_mixed_requires_grad(self):
        # [FUNCTION DOC] Returns one differentiable output and one view of a non-grad input.
        # Checks values and requires_grad flags, backward grads of two, and that exactly one
        # compiled_fn_wrapper artifact was captured (the alias artifact is captured but unchecked).
        """
        Training path with mixed differentiable and non-differentiable
        outputs. Exercises non-differentiable output collection in
        _transform_raw_returns codegen and backward correctness.
        """
        with (
            self._capture_codegen_source("compiled_fn_wrapper") as xform_captured,
            self._capture_codegen_source("output_alias_wrapper") as _alias_captured,
        ):

            @torch.compile(backend="aot_eager")
            def f(x, y):
                # [FUNCTION DOC] HELPER: Compiled function returning a doubled
                # first arg and a flattened view of the second.
                return x * 2, y.view(-1)

            x = torch.randn(2, 3, requires_grad=True)
            y = torch.randn(4)
            out1, out2 = f(x, y)

        self.assertEqual(out1, x * 2)
        self.assertEqual(out2, y.view(-1))
        self.assertTrue(out1.requires_grad)
        self.assertFalse(out2.requires_grad)

        out1.sum().backward()
        self.assertEqual(x.grad, torch.full((2, 3), 2.0))

        self.assertEqual(
            len(xform_captured),
            1,
            "Expected compiled_fn_wrapper codegen artifact to be emitted",
        )

    def test_training_path_mutation_and_alias(self):
        # [FUNCTION DOC] Mutates a non-leaf clone in place and returns a flattened view of it
        # (training path). Checks the mutation and the view values and one artifact emitted.
        """
        Training path: input is mutated AND returned as a view. Exercises
        both the mutation epilogue and alias codegen with trace_joint=True.
        Uses a non-leaf tensor to allow in-place mutation with grad.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Compiled function adding one to
                # its arg in place and returning a flattened view.
                x.add_(1)
                return x.view(-1)

            base = torch.randn(2, 3, requires_grad=True)
            x = base.clone()
            x_ref = x.detach().clone()
            out = f(x)

        self.assertEqual(x, x_ref + 1)
        self.assertEqual(out, (x_ref + 1).view(-1))

        self.assertEqual(
            len(captured),
            1,
            "Expected output_alias_wrapper codegen artifact to be emitted",
        )

    def test_training_path_is_input(self):
        # [FUNCTION DOC] Doubles a non-leaf clone in place and returns the same tensor (training
        # path). Checks the mutation, that the output is the input object, one artifact emitted.
        """
        Training path: output IS the input (mutation + return identity).
        Exercises IsInputHandler with trace_joint=True. Uses a non-leaf
        tensor to allow in-place mutation with grad.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Compiled function doubling its
                # arg in place and returning the same arg.
                x.mul_(2)
                return x

            base = torch.randn(4, requires_grad=True)
            x = base.clone()
            x_ref = x.detach().clone()
            out = f(x)

        self.assertEqual(x, x_ref * 2)
        self.assertIs(out, x)

        self.assertEqual(
            len(captured),
            1,
            "Expected output_alias_wrapper codegen artifact to be emitted",
        )

    def test_training_path_alias_of_intermediate_detach(self):
        # [FUNCTION DOC] Returns a detached view and a differentiable view of one intermediate.
        # Checks values, requires_grad flags, shared storage, backward grads of ones, and one
        # output_alias_wrapper artifact.
        """
        Training path: one output is a detached view of an intermediate,
        the other is a differentiable view. Exercises
        AliasOfIntermediateHandler with trace_joint=True and the
        base_is_user_output sub-path.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Compiled function returning a detached
                # copy and a flattened view of arg plus one.
                y = x + 1
                return y.detach(), y.view(-1)

            x = torch.randn(3, 3, requires_grad=True)
            out_detach, out_view = f(x)

        self.assertEqual(out_detach, x + 1)
        self.assertEqual(out_view, (x + 1).view(-1))
        self.assertFalse(out_detach.requires_grad)
        self.assertTrue(out_view.requires_grad)
        self.assertEqual(out_detach.data_ptr(), out_view.data_ptr())

        out_view.sum().backward()
        self.assertEqual(x.grad, torch.ones(3, 3))

        self.assertEqual(
            len(captured),
            1,
            "Expected output_alias_wrapper codegen artifact to be emitted",
        )

    def test_view_replay_config_false(self):
        # [FUNCTION DOC] With the view replay config patched to false, compiles a view-returning
        # function. Checks values and aliasing, one artifact, and that its source records view
        # replay as disabled.
        """
        Test that view_replay_for_aliased_outputs=False is correctly
        baked into the codegen'd alias function.
        """
        with patch(
            "torch._functorch.config.view_replay_for_aliased_outputs",
            False,
        ):
            with self._capture_codegen_source("output_alias_wrapper") as captured:

                @torch.compile(backend="aot_eager")
                def f(x):
                    # [FUNCTION DOC] HELPER: Compiled function returning
                    # a flattened view of its arg.
                    return x.view(-1)

                x = torch.randn(2, 3)
                out = f(x)

            self.assertEqual(out, x.view(-1))
            self.assertEqual(out.data_ptr(), x.data_ptr())

            self.assertEqual(len(captured), 1)
            self.assertIn("replay_views=False", captured[0])

    def test_view_replay_config_true(self):
        # [FUNCTION DOC] With the view replay config patched to true, compiles a view-returning
        # function. Checks values and aliasing, one artifact, and that its source records view
        # replay as enabled.
        """
        Test that view_replay_for_aliased_outputs=True (default) is
        correctly baked into the codegen'd alias function.
        """
        with patch(
            "torch._functorch.config.view_replay_for_aliased_outputs",
            True,
        ):
            with self._capture_codegen_source("output_alias_wrapper") as captured:

                @torch.compile(backend="aot_eager")
                def f(x):
                    # [FUNCTION DOC] HELPER: Compiled function returning
                    # a flattened view of its arg.
                    return x.view(-1)

                x = torch.randn(2, 3)
                out = f(x)

            self.assertEqual(out, x.view(-1))
            self.assertEqual(out.data_ptr(), x.data_ptr())

            self.assertEqual(len(captured), 1)
            self.assertIn("replay_views=True", captured[0])

    def test_codegen_source_contains_gen_alias(self):
        # [FUNCTION DOC] Mutates the input and returns it plus a view. Checks the single artifact
        # source mentions the alias generation helper and indexes the original inputs.
        """
        Verify the codegen'd source contains gen_alias_from_base for
        alias-of-input outputs and orig_inputs for is_input outputs.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Compiled function doubling its arg
                # in place and returning it plus a flattened view.
                x.mul_(2)
                return x, x.view(-1)

            x = torch.randn(2, 3)
            f(x)

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("gen_alias_from_base", source)
        self.assertIn("orig_inputs[", source)

    def test_codegen_source_noop_handler(self):
        # [FUNCTION DOC] Returns a fresh product and a view of the input. Checks the single
        # artifact source indexes the forward outputs and uses the alias generation helper.
        """
        Verify the codegen'd source contains fw_outs[i] for non-aliased
        (NoopAliasHandler) outputs in a mixed scenario.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Compiled function returning a doubled
                # copy and a flattened view of its arg.
                return x * 2, x.view(-1)

            x = torch.randn(2, 3)
            f(x)

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("fw_outs[", source)
        self.assertIn("gen_alias_from_base", source)

    def test_alias_of_intermediate_save_as_output(self):
        # [FUNCTION DOC] Returns two views of the same intermediate with a grad-requiring input.
        # Checks values, shared storage, backward grads of ones, one artifact whose source uses
        # the alias generation helper.
        """
        Two outputs aliasing the same intermediate (not an input). When
        multiple outputs share the same intermediate base, the first triggers
        alias_of_intermediate_save_as_output and the second triggers
        alias_of_intermediate. Both use AliasOfIntermediateHandler in codegen.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Compiled function returning two
                # differently shaped views of arg plus one.
                y = x + 1
                return y.view(-1), y.view(3, 3)

            x = torch.randn(3, 3, requires_grad=True)
            out1, out2 = f(x)

        expected = x + 1
        self.assertEqual(out1, expected.view(-1))
        self.assertEqual(out2, expected.view(3, 3))
        self.assertEqual(out1.data_ptr(), out2.data_ptr())

        out1.sum().backward()
        self.assertEqual(x.grad, torch.ones(3, 3))

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("gen_alias_from_base", source)

    def test_xform_unsafe_view_output(self):
        # [FUNCTION DOC] Returns a single view of an intermediate in the training path and
        # inspects the compiled_fn_wrapper artifact. Checks values, requires_grad, backward
        # grads, one artifact, and that its source contains an unsafe view call.
        """
        _transform_raw_returns codegen: when an output is a view of an
        intermediate and is the only output aliasing that intermediate
        (unsafe_view_alias), the codegen emits an _unsafe_view call.
        """
        with self._capture_codegen_source("compiled_fn_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Compiled function returning
                # a flattened view of arg plus one.
                return (x + 1).view(-1)

            x = torch.randn(2, 3, requires_grad=True)
            out = f(x)

        self.assertEqual(out, (x + 1).view(-1))
        self.assertTrue(out.requires_grad)

        out.sum().backward()
        self.assertEqual(x.grad, torch.ones(2, 3))

        self.assertEqual(len(captured), 1)
        self.assertIn("_unsafe_view", captured[0])

    @skipIfTorchDynamo("dynamo handles metadata mutations in-graph")
    def test_xform_metadata_only_mutation(self):
        # [FUNCTION DOC] Uses aot_function with the nop compiler on a function that transposes
        # its input in place. Checks the transposed shapes, one compiled_fn_wrapper artifact, and
        # that its source wraps the result in the tensor alias class. Skipped under dynamo.
        """
        _transform_raw_returns codegen: when an input has a metadata-only
        mutation (mutates_metadata=True, mutates_data=False), the codegen
        wraps the corresponding mutated input return in TensorAlias.
        Uses aot_function directly because dynamo handles metadata
        mutations in-graph, so they never reach the _transform_raw_returns
        codegen path.
        """
        with self._capture_codegen_source("compiled_fn_wrapper") as captured:

            def f(a, b):
                # [FUNCTION DOC] HELPER: Traced function transposing its first arg in place and
                # returning the sum with the second arg.
                a.transpose_(1, 0)
                return a + b

            a = torch.randn(3, 4, requires_grad=True).add(0)
            b = torch.randn(4, 3)
            compiled_f = aot_function(f, nop)
            out = compiled_f(a, b)

        self.assertEqual(a.shape, (4, 3))
        self.assertEqual(out.shape, (4, 3))

        self.assertEqual(len(captured), 1)
        self.assertIn("TensorAlias", captured[0])

    def test_cross_dtype_view_alias(self):
        # [FUNCTION DOC] Returns the real-view of a complex tensor (a cross-dtype view). Checks
        # values, shape, float dtype, and one artifact whose source uses the alias helper.
        """
        Output is a cross-dtype view of the input (view_as_real on a
        complex tensor). Exercises gen_alias_from_base's cross-dtype
        handling through the codegen path.
        """
        with self._capture_codegen_source("output_alias_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Compiled function returning
                # the real view of a complex arg.
                return torch.view_as_real(x)

            x = torch.randn(4, dtype=torch.complex64)
            out = f(x)

        self.assertEqual(out, torch.view_as_real(x))
        self.assertEqual(out.shape, (4, 2))
        self.assertEqual(out.dtype, torch.float32)

        self.assertEqual(len(captured), 1)
        self.assertIn("gen_alias_from_base", captured[0])

    def test_xform_aliased_output_tensoralias_wrapping(self):
        # [FUNCTION DOC] Returns a product and a view of the input with grad enabled. Checks
        # values, shared storage for the view, backward grads of two, and one compiled_fn_wrapper
        # artifact whose source contains the tensor alias wrapper.
        """
        _transform_raw_returns codegen: aliased outputs get wrapped in
        TensorAlias so autograd.Function doesn't treat them as regular
        tensors. Verifies the TensorAlias wrapping path for aliased
        outputs (distinct from the metadata-only mutation wrapping).
        Needs a non-view computation (x * 2) to force the autograd
        factory path; a pure view like x.view(-1) alone bypasses it.
        """
        with self._capture_codegen_source("compiled_fn_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                # [FUNCTION DOC] HELPER: Compiled function returning a doubled
                # copy and a flattened view of its arg.
                return x * 2, x.view(-1)

            x = torch.randn(2, 3, requires_grad=True)
            out1, out2 = f(x)

        self.assertEqual(out1, x * 2)
        self.assertEqual(out2, x.view(-1))
        self.assertEqual(out2.data_ptr(), x.data_ptr())

        out1.sum().backward()
        self.assertEqual(x.grad, torch.full((2, 3), 2.0))

        self.assertEqual(len(captured), 1)
        self.assertIn("TensorAlias", captured[0])


if __name__ == "__main__":
    run_tests()
