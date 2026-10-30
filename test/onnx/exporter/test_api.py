# Owner(s): ["module: onnx"]
# [MODULE DOC] Simple API-level tests of the dynamo based ONNX exporter entry point. Defines
# several small sample models (simple, two inputs, reduction, dynamic-shape and nested-input
# models, concat with a size one dim) and two test cases: one for export options (opset
# version, dynamic axes and shapes, input and output names, file output, deprecation warnings,
# none outputs) and one for custom translation tables. No import-time side effects other than
# imports and class definitions.
"""Simple API tests for the ONNX exporter."""

from __future__ import annotations

import io
import logging
import os

from onnxscript import FLOAT, opset18 as op

import torch
from torch.onnx._internal.exporter import _testing as onnx_testing
from torch.testing._internal import common_utils


class SampleModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Sample model returning a tuple of the input plus one and the relu of
    # that sum.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Adds one to the input and returns that sum together with its
        # relu.
        y = x + 1
        z = y.relu()
        return (y, z)


class SampleModelTwoInputs(torch.nn.Module):
    # [CLASS DOC] HELPER: Sample model with two inputs returning a tuple of their sum and its
    # relu.
    def forward(self, x, b):
        # [FUNCTION DOC] HELPER: Adds the two inputs and returns the sum together with its
        # relu.
        y = x + b
        z = y.relu()
        return (y, z)


class SampleModelReduction(torch.nn.Module):
    # [CLASS DOC] HELPER: Sample model that reduces its input to a scalar by summing all
    # elements.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the sum of all elements of the input.
        return x.sum()


class SampleModelForDynamicShapes(torch.nn.Module):
    # [CLASS DOC] HELPER: Sample model with two independent inputs used by the dynamic shape
    # tests; it applies relu to the first and sigmoid to the second.
    def forward(self, x, b):
        # [FUNCTION DOC] HELPER: Returns a tuple of the relu of x and the sigmoid of b.
        return x.relu(), b.sigmoid()


class NestedModelForDynamicShapes(torch.nn.Module):
    # [CLASS DOC] HELPER: Sample model whose inputs are nested containers: a tensor, a list of
    # two tensors, a mapping with keys a and b, and another tensor. The forward branches on
    # input sizes.
    def forward(
        self,
        x: torch.Tensor,
        ys: list[torch.Tensor],
        zs: dict[str, torch.Tensor],
        c: torch.Tensor,
    ):
        # [FUNCTION DOC] HELPER: Sums the two list tensors and the two mapping tensors into
        # one tensor, then, depending on a shape condition (first input shorter than three and
        # last input length not four), returns the first input plus or minus a constant five,
        # plus or minus the summed tensor, and the last input.
        y = ys[0] + ys[1] + zs["a"] + zs["b"]
        w = 5
        if x.shape[0] < 3 and c.shape[0] != 4:
            return x + w, x + y, c
        else:
            return x - w, x - y, c


class SampleModelForDimOne(torch.nn.Module):
    # [CLASS DOC] HELPER: Sample model that concatenates two inputs along the second axis and
    # adds a third input.
    def forward(self, x, y, z):
        # [FUNCTION DOC] HELPER: Returns the concatenation of x and y along axis one plus z.
        return torch.cat((x, y), axis=1) + z


class TestExportAPIDynamo(common_utils.TestCase):
    # [CLASS DOC] Test case for torch.onnx.export with the dynamo option on, covering export
    # options and dynamic shape handling. Uses an assert helper that exports and checks
    # numerics.
    """Tests for the ONNX exporter API when dynamo=True."""

    def assert_export(
        self, *args, strategy: str | None = "TorchExportNonStrictStrategy", **kwargs
    ):
        # [FUNCTION DOC] HELPER: Exports with the dynamo path (quiet) forwarding all
        # arguments, raises AssertionError if no program is returned, validates the program
        # numerically with the testing helper using the given capture strategy name
        # (non-strict by default, or None), and returns the program.
        onnx_program = torch.onnx.export(*args, **kwargs, dynamo=True, verbose=False)
        if onnx_program is None:
            raise AssertionError("onnx_program is None")
        onnx_testing.assert_onnx_program(onnx_program, strategy=strategy)
        return onnx_program

    def test_args_normalization_with_no_kwargs(self):
        # [FUNCTION DOC] Exports a two-input model with only positional inputs and no keyword
        # arguments and checks numerics.
        self.assert_export(
            SampleModelTwoInputs(),
            (torch.randn(1, 1, 2), torch.randn(1, 1, 2)),
        )

    def test_lower_opset_support(self):
        # [FUNCTION DOC] Exports a reduction model with opset versions 18 and 16 and asserts
        # the resulting model imports the default domain at exactly the requested opset
        # version each time.
        # First test that opset 18 (torchlib opset works)
        onnx_program = self.assert_export(
            SampleModelReduction(), (torch.randn(1, 1, 2),), opset_version=18
        )
        self.assertEqual(onnx_program.model.opset_imports[""], 18)

        onnx_program = self.assert_export(
            SampleModelReduction(), (torch.randn(1, 1, 2),), opset_version=16
        )
        self.assertEqual(onnx_program.model.opset_imports[""], 16)

    def test_symbolic_argument_user_input_is_supported_by_report_and_call(self):
        # [FUNCTION DOC] Exports a model whose first input is a Python integer with dynamic
        # shapes marking that integer and the tensor as dynamic, requesting a report in a
        # temporary artifacts directory. Meanwhile an error-level handler captures the core
        # exporter logger; the test asserts no report-saving failure or missing tensor
        # metadata error was logged. Handler and level are restored in a finally block.
        class constant_plus_tensor_inputs(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding its integer-like first input, a constant tensor
            # one, and a tensor input.
            def forward(self, a, x):
                # [FUNCTION DOC] HELPER: Returns a plus a constant one tensor plus x.
                return a + torch.tensor(1) + x

        # Capture log output
        log_capture = io.StringIO()
        log_handler = logging.StreamHandler(log_capture)
        log_handler.setLevel(logging.ERROR)
        # Get the logger used in _core.py
        logger = logging.getLogger("torch.onnx._internal.exporter._core")
        original_level = logger.level
        logger.addHandler(log_handler)
        logger.setLevel(logging.ERROR)

        try:
            with common_utils.TemporaryDirectoryName() as temp_dir:
                self.assert_export(
                    constant_plus_tensor_inputs(),
                    (
                        1,
                        torch.ones(2),
                    ),
                    dynamic_shapes=(
                        torch.export.Dim.DYNAMIC,
                        {0: torch.export.Dim.DYNAMIC},
                    ),
                    report=True,
                    artifacts_dir=temp_dir,
                )
                # Check if the expected error was logged
                log_output = log_capture.getvalue()
                self.assertNotIn("Failed to save report due to an error", log_output)
                self.assertNotIn("KeyError: 'tensor_meta'", log_output)
                # Note: We don't call assert_onnx_program here because it will fail
                # due to the input name mismatch issue mentioned in your error

        finally:
            # Clean up logging
            logger.removeHandler(log_handler)
            logger.setLevel(original_level)

    def test_constant_argument_user_input_is_omitted_in_onnx_graph(self):
        # [FUNCTION DOC] Exports a model taking a Python integer constant and a tensor with a
        # dynamic first axis and asserts the exported graph has only one input, i.e. the
        # constant argument is omitted.
        class constant_plus_tensor_inputs(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding its first input, a constant tensor one, and its
            # second input.
            def forward(self, a, x):
                # [FUNCTION DOC] HELPER: Returns a plus a constant one tensor plus x.
                return a + torch.tensor(1) + x

        onnx_program = torch.onnx.export(
            constant_plus_tensor_inputs(),
            (
                1,
                torch.ones(2),
            ),
            dynamic_shapes=(
                None,
                {0: torch.export.Dim.DYNAMIC},
            ),
            dynamo=True,
        )

        self.assertEqual(len(onnx_program.model.graph.inputs), 1)

    def test_dynamic_axes_enable_dynamic_shapes_with_fully_specified_axes(self):
        # [FUNCTION DOC] Exports with legacy dynamic axes given as mappings with custom names
        # for every axis of both inputs, validating numerics.
        self.assert_export(
            SampleModelForDynamicShapes(),
            (torch.randn(2, 2, 3), {"b": torch.randn(2, 2, 3)}),
            dynamic_axes={
                "x": {0: "customx_dim_0", 1: "customx_dim_1", 2: "customx_dim_2"},
                "b": {0: "customb_dim_0", 1: "customb_dim_1", 2: "customb_dim_2"},
            },
        )

    def test_dynamic_axes_enable_dynamic_shapes_with_default_axe_names(self):
        # [FUNCTION DOC] Exports with legacy dynamic axes given as lists of axis indices (so
        # default names are generated) for both inputs, validating numerics.
        self.assert_export(
            SampleModelForDynamicShapes(),
            (torch.randn(2, 2, 3), {"b": torch.randn(2, 2, 3)}),
            dynamic_axes={
                "x": [0, 1, 2],
                "b": [0, 1, 2],
            },
        )

    def test_dynamic_axes_supports_partial_dynamic_shapes(self):
        # [FUNCTION DOC] Exports with input names and dynamic axes given for only the second
        # input, checking partial dynamic specification works.
        self.assert_export(
            SampleModelForDynamicShapes(),
            (torch.randn(2, 2, 3), {"b": torch.randn(2, 2, 3)}),
            input_names=["x", "b"],
            dynamic_axes={
                "b": [0, 1, 2],
            },
        )

    def test_dynamic_axes_supports_output_names(self):
        # [FUNCTION DOC] First repeats the partial dynamic axes case. Then exports with input
        # and output names and dynamic axes that include an output name, checking output names
        # are supported in dynamic axes.
        self.assert_export(
            SampleModelForDynamicShapes(),
            (torch.randn(2, 2, 3), {"b": torch.randn(2, 2, 3)}),
            input_names=["x", "b"],
            dynamic_axes={
                "b": [0, 1, 2],
            },
        )
        self.assert_export(
            SampleModelForDynamicShapes(),
            (
                torch.randn(2, 2, 3),
                torch.randn(2, 2, 3),
            ),
            input_names=["x", "b"],
            output_names=["x_out", "b_out"],
            dynamic_axes={"b": [0, 1, 2], "b_out": [0, 1, 2]},
        )

    def test_from_dynamic_axes_to_dynamic_shapes_deprecation_warning(self):
        # [FUNCTION DOC] Exports with legacy dynamic axes under a check that a
        # DeprecationWarning is emitted whose message says the dynamic axes to dynamic shapes
        # conversion is deprecated and that dynamic shapes should be given directly with
        # custom names.
        with self.assertWarnsRegex(
            DeprecationWarning,
            "from_dynamic_axes_to_dynamic_shapes is deprecated and will be removed in a future release. "
            "This function converts 'dynamic_axes' format \\(including custom axis names\\) to 'dynamic_shapes' format. "
            "Instead of relying on this conversion, provide 'dynamic_shapes' directly with custom names.",
        ):
            self.assert_export(
                SampleModelForDynamicShapes(),
                (torch.randn(2, 2, 3), {"b": torch.randn(2, 2, 3)}),
                dynamic_axes={
                    "x": [0, 1, 2],
                    "b": [0, 1, 2],
                },
            )

    def test_from_dynamic_axes_to_dynamic_shapes_keeps_custom_axis_names(self):
        # [FUNCTION DOC] Exports with dynamic axes holding custom names for all inputs and
        # outputs, plus input and output names, and asserts via identity comparison that every
        # dimension of both input shapes in the ONNX graph carries exactly the custom name.
        model = SampleModelForDynamicShapes()
        input = (
            torch.randn(2, 2, 3),
            {"b": torch.randn(2, 2, 3)},
        )
        dynamic_axes = {
            "x": {0: "customx_x_0", 1: "customx_x_1", 2: "customx_x_2"},
            "b": {0: "customb_b_0", 1: "customb_b_1", 2: "customb_b_2"},
            "x_out": {0: "customx_out_x_0", 1: "customx_out_x_1", 2: "customx_out_x_2"},
            "b_out": {0: "customb_out_b_0", 1: "customb_out_b_1", 2: "customb_out_b_2"},
        }
        onnx_program = torch.onnx.export(
            model,
            input,
            dynamic_axes=dynamic_axes,
            input_names=["x", "b"],
            output_names=["x_out", "b_out"],
            dynamo=True,
        )

        # Check whether the dynamic dimension names are preserved
        self.assertIs(onnx_program.model.graph.inputs[0].shape[0].value, "customx_x_0")
        self.assertIs(onnx_program.model.graph.inputs[0].shape[1].value, "customx_x_1")
        self.assertIs(onnx_program.model.graph.inputs[0].shape[2].value, "customx_x_2")
        self.assertIs(onnx_program.model.graph.inputs[1].shape[0].value, "customb_b_0")
        self.assertIs(onnx_program.model.graph.inputs[1].shape[1].value, "customb_b_1")
        self.assertIs(onnx_program.model.graph.inputs[1].shape[2].value, "customb_b_2")

    def test_saved_f_exists_after_export(self):
        # [FUNCTION DOC] Exports to a temporary file path with the dynamo option and asserts
        # the file exists afterwards.
        with common_utils.TemporaryFileName(suffix=".onnx") as path:
            _ = torch.onnx.export(
                SampleModel(), (torch.randn(1, 1, 2),), path, dynamo=True
            )
            self.assertTrue(os.path.exists(path))

    def test_dynamic_shapes_with_fully_specified_axes(self):
        # [FUNCTION DOC] Builds an exported program of the dynamic shape model with a named
        # Dim for every axis of both inputs (strict mode), then exports that program through
        # the assert helper with no capture strategy.
        ep = torch.export.export(
            SampleModelForDynamicShapes(),
            (
                torch.randn(2, 2, 3),
                torch.randn(2, 2, 3),
            ),
            dynamic_shapes={
                "x": {
                    0: torch.export.Dim("customx_dim_0"),
                    1: torch.export.Dim("customx_dim_1"),
                    2: torch.export.Dim("customx_dim_2"),
                },
                "b": {
                    0: torch.export.Dim("customb_dim_0"),
                    1: torch.export.Dim("customb_dim_1"),
                    2: torch.export.Dim("customb_dim_2"),
                },
            },
            strict=True,
        )

        self.assert_export(ep, strategy=None)

    def test_partial_dynamic_shapes(self):
        # [FUNCTION DOC] Exports with dynamic shapes where the first input is static and the
        # second has named dims for all axes, validating numerics.
        self.assert_export(
            SampleModelForDynamicShapes(),
            (
                torch.randn(2, 2, 3),
                torch.randn(2, 2, 3),
            ),
            dynamic_shapes={
                "x": None,
                "b": {
                    0: torch.export.Dim("customb_dim_0"),
                    1: torch.export.Dim("customb_dim_1"),
                    2: torch.export.Dim("customb_dim_2"),
                },
            },
        )

    def test_dynamic_shapes_supports_nested_input_model_with_input_names_assigned(self):
        # [FUNCTION DOC] Exports the nested input model with explicit input names (renaming
        # flattened inputs in order) and dynamic axes for the first five, validating numerics.
        # Then exports again and asserts every input except the last has a named dynamic
        # dimension in the model proto graph.
        # kwargs can still be renamed as long as it's in order
        input_names = ["input_x", "input_y", "input_z", "d", "e", "f"]

        dynamic_axes = {
            "input_x": {0: "dim"},
            "input_y": {0: "dim"},
            "input_z": {0: "dim"},
            "d": {0: "dim"},
            "e": {0: "dim"},
        }

        model = NestedModelForDynamicShapes()
        input = (
            torch.ones(5),
            [torch.zeros(5), torch.ones(5)],
            {"a": torch.zeros(5), "b": torch.ones(5)},
            torch.ones(4),
        )

        self.assert_export(
            model, input, dynamic_axes=dynamic_axes, input_names=input_names
        )

        # Check whether inputs are dynamically shaped
        onnx_program = torch.onnx.export(
            model,
            input,
            dynamic_axes=dynamic_axes,
            input_names=input_names,
            dynamo=True,
        )
        self.assertTrue(
            all(
                [
                    input.type.tensor_type.shape.dim[0].dim_param
                    for input in onnx_program.model_proto.graph.input
                ][:-1]
            )
        )

    def test_upgraded_torchlib_impl(self):
        # [FUNCTION DOC] Exports a model using the tanh approximation of gelu at opset 18 and
        # at opset 20. Asserts the opset 18 graph decomposes into a tanh node without a gelu
        # node, and the opset 20 graph contains a native gelu node.
        class GeluModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model applying gelu with the tanh approximation.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the tanh-approximated gelu of the input.
                # Use GELU activation function
                return torch.nn.functional.gelu(input, approximate="tanh")

        input = (torch.randn(1, 3, 4, 4),)
        onnx_program_op18 = torch.onnx.export(
            GeluModel(),
            input,
            opset_version=18,
            dynamo=True,
        )
        all_nodes_op18 = [n.op_type for n in onnx_program_op18.model.graph]
        self.assertIn("Tanh", all_nodes_op18)
        self.assertNotIn("Gelu", all_nodes_op18)

        onnx_program_op20 = torch.onnx.export(
            GeluModel(),
            input,
            opset_version=20,
            dynamo=True,
        )
        all_nodes_op20 = [n.op_type for n in onnx_program_op20.model.graph]
        self.assertIn("Gelu", all_nodes_op20)

    def test_refine_dynamic_shapes_with_onnx_export(self):
        # [FUNCTION DOC] Exports a model whose forward asserts lower and upper size bounds
        # through a condition, with named dynamic dims for each input, and validates numerics
        # (refined dynamic shape ranges must not break export).
        # NOTE: From test/export/test_export.py

        # refine lower, upper bound
        class TestRefineDynamicShapeModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning x doubled and y plus one only when x has at
            # least six rows and y at most sixteen rows; otherwise returns None implicitly.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Under the size condition returns x times two and y
                # plus one; otherwise falls through returning None.
                if x.shape[0] >= 6 and y.shape[0] <= 16:
                    return x * 2.0, y + 1

        inps = (torch.randn(16), torch.randn(12))
        dynamic_shapes = {
            "x": (torch.export.Dim("dx"),),
            "y": (torch.export.Dim("dy"),),
        }
        self.assert_export(
            TestRefineDynamicShapeModel(), inps, dynamic_shapes=dynamic_shapes
        )

    def test_zero_output_aten_node(self):
        # [FUNCTION DOC] Exports a model that calls an aten assertion op producing no outputs,
        # and validates numerics. Note the input is passed as a bare tensor rather than a
        # one-element tuple.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model that issues an async assertion on a true tensor with a
            # message and returns x plus x.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the zero-output aten assertion op, then returns
                # the input doubled.
                torch.ops.aten._assert_async.msg(torch.tensor(True), "assertion failed")
                return x + x

        input = torch.randn(2)
        self.assert_export(Model(), (input))

    def test_export_successful_when_dynamic_dimension_is_one(self):
        # [FUNCTION DOC] Exports a concat model with inputs of size one in the batch axis,
        # sharing named dynamic batch and sequence dims, and validates numerics (a dynamic dim
        # equal to one must be accepted).
        self.assert_export(
            SampleModelForDimOne(),
            (torch.randn(1, 3), torch.randn(1, 5), torch.randn(1, 8)),
            dynamic_shapes=(
                {0: "batch", 1: "sequence"},
                {0: "batch", 1: "sequence"},
                {0: "batch", 1: "sequence"},
            ),
        )

    def test_is_in_onnx_export(self):
        # [FUNCTION DOC] Checks that the is-in-onnx-export query is false before and after
        # export, and that during the export the model took the sine branch (a sine node
        # appears in the graph).
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Model whose forward delegates to an inner function returning
            # sine of the input during ONNX export and cosine otherwise.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the inner function on the input and returns its
                # result.
                def f(x):
                    # [FUNCTION DOC] HELPER: Returns sine of x if an ONNX export is in
                    # progress, else cosine of x.
                    return x.sin() if torch.onnx.is_in_onnx_export() else x.cos()

                return f(x)

        self.assertFalse(torch.onnx.is_in_onnx_export())
        onnx_program = torch.onnx.export(
            Mod(),
            (torch.randn(3, 4),),
            dynamo=True,
        )
        self.assertFalse(torch.onnx.is_in_onnx_export())

        node_names = [n.op_type for n in onnx_program.model.graph]
        self.assertIn("Sin", node_names)

    def test_torchscript_exporter_raises_deprecation_warning(self):
        # [FUNCTION DOC] Exports with the dynamo option off (legacy TorchScript exporter) to
        # an in-memory buffer and asserts a DeprecationWarning about the legacy
        # TorchScript-based export is emitted.
        # Test that the deprecation warning is raised when using torchscript exporter
        with self.assertWarnsRegex(
            DeprecationWarning, "You are using the legacy TorchScript-based ONNX export"
        ):
            torch.onnx.export(
                SampleModel(), (torch.randn(1, 1, 2),), io.BytesIO(), dynamo=False
            )

    def test_model_output_can_be_none(self):
        # [FUNCTION DOC] Exports a model returning a tensor and None with the dynamo option
        # and validates numerics, checking None outputs are supported.
        class ModelWithNoneOutput(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning the input plus one and a None value.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a pair of x plus one and None.
                return x + 1, None

        onnx_program = torch.onnx.export(
            ModelWithNoneOutput(),
            (torch.randn(1, 1, 2),),
            dynamo=True,
        )
        onnx_testing.assert_onnx_program(onnx_program)


class TestCustomTranslationTable(common_utils.TestCase):
    # [CLASS DOC] Test case for the custom translation table option of the dynamo export,
    # which maps operator targets to replacement ONNX functions. Each test swaps an addition
    # for a subtraction and inspects the node types.
    def test_custom_translation_table_overrides_ops(self):
        # [FUNCTION DOC] Exports an addition model with a table mapping the aten add overload
        # to a function emitting an ONNX subtraction, then asserts the graph has a subtraction
        # node and no addition node.
        from onnxscript import opset18 as op

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning the sum of its two inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus y.
                return x + y

        def custom_add(self, other):
            # [FUNCTION DOC] HELPER: Replacement translation used in the table; emits an ONNX
            # subtraction of its two arguments instead of an addition.
            # Replace add with sub
            return op.Sub(self, other)

        custom_translation_table = {torch.ops.aten.add.Tensor: custom_add}

        onnx_program = torch.onnx.export(
            Model(),
            (torch.randn(2, 2), torch.randn(2, 2)),
            custom_translation_table=custom_translation_table,
            dynamo=True,
        )
        all_nodes = [n.op_type for n in onnx_program.model.graph]
        self.assertIn("Sub", all_nodes)
        self.assertNotIn("Add", all_nodes)

    def test_custom_translation_table_supports_custom_op_as_target(self):
        # [FUNCTION DOC] Registers a custom operator in the custom namespace for addition plus
        # a fake implementation, uses it in a model, and exports with a table mapping the
        # custom operator default overload to a function emitting a subtraction. Boolean
        # scalar inputs are used. Asserts a subtraction node is present and no addition node.
        # Define the custom op and use it in the model
        @torch.library.custom_op("custom::add", mutates_args=())
        def custom_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Eager implementation of the custom operator; returns the
            # sum of its two tensors.
            return a + b

        @custom_add.register_fake
        def _(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake (meta) implementation of the custom operator;
            # returns the sum of empty tensors shaped like the inputs, only providing output
            # metadata.
            return torch.empty_like(a) + torch.empty_like(b)

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model that calls the custom addition operator on its two
            # inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the custom operator applied to x and y.
                return custom_add(x, y)

        def onnx_add(self: FLOAT, other: FLOAT) -> FLOAT:
            # [FUNCTION DOC] HELPER: Replacement translation for the custom operator; emits an
            # ONNX subtraction of its two float arguments.
            # Replace add with Sub
            return op.Sub(self, other)

        custom_translation_table = {
            torch.ops.custom.add.default: onnx_add,
        }

        onnx_program = torch.onnx.export(
            Model(),
            (torch.tensor(1, dtype=torch.bool), torch.tensor(1, dtype=torch.bool)),
            custom_translation_table=custom_translation_table,
            dynamo=True,
        )
        all_nodes = [n.op_type for n in onnx_program.model.graph]
        self.assertIn("Sub", all_nodes)
        self.assertNotIn("Add", all_nodes)

    def test_custom_translation_table_supports_custom_op_with_its_decomp(self):
        # [FUNCTION DOC] Inside a scoped library defines an operator with an implicit
        # composite implementation (addition) and fake registration. First exports with a
        # table mapping the operator to a subtraction function and asserts subtraction appears
        # without addition. Then exports without a table and asserts the operator is
        # decomposed into an addition without subtraction.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor a, Tensor b) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "CompositeImplicitAutograd", lib=lib)
            @torch.library.register_fake("mylib::foo")
            def foo_impl(a, b):
                # [FUNCTION DOC] HELPER: Composite implementation and fake registration of the
                # library operator; returns the sum of a and b.
                return a + b

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Model that calls the library operator on its two inputs.
                def forward(self, x, y):
                    # [FUNCTION DOC] HELPER: Returns the library operator applied to x and y.
                    return torch.ops.mylib.foo(x, y)

            def onnx_add(self: FLOAT, other: FLOAT) -> FLOAT:
                # [FUNCTION DOC] HELPER: Replacement translation for the library operator;
                # emits an ONNX subtraction of its two float arguments.
                # Replace add with Sub
                return op.Sub(self, other)

            # With the custom op defined, we can use it in the model
            # and replace it with a custom translation table
            custom_translation_table = {
                torch.ops.mylib.foo.default: onnx_add,
            }
            onnx_program = torch.onnx.export(
                M(),
                (torch.ones(3, 3), torch.ones(3, 3)),
                custom_translation_table=custom_translation_table,
                dynamo=True,
            )
            all_nodes = [n.op_type for n in onnx_program.model.graph]
            self.assertIn("Sub", all_nodes)
            self.assertNotIn("Add", all_nodes)

            # Without the custom op defined, it's going to be decomposed
            onnx_program_decomp = torch.onnx.export(
                M(), (torch.ones(3, 3), torch.ones(3, 3)), dynamo=True
            )
            all_nodes_decomp = [n.op_type for n in onnx_program_decomp.model.graph]
            self.assertIn("Add", all_nodes_decomp)
            self.assertNotIn("Sub", all_nodes_decomp)

    def test_01_specialization_with_run_decomp_is_supported(self):
        # [FUNCTION DOC] Regression test: exports an RMS norm style model (as in Phi3) in eval
        # mode with only a keyword input having named batch and sequence dynamic dims, so that
        # shape inference is redone after decompositions run. Asserts the first output
        # dimension is not specialized to one. The local result variable shadows the imported
        # opset alias inside this test only.
        # Phi3RMSNorm changes and redo shape inference after `run_decompositions` call
        # We need this test to make sure everything we do on fx graph is covered by
        # backed_size_oblivious
        class Phi3RMSNorm(torch.nn.Module):
            # [CLASS DOC] HELPER: RMS normalization module with a learned scale of the hidden
            # size and a variance epsilon; computes the mean square over the last axis, scales
            # by its reciprocal square root in float32 and casts back to the input dtype.
            def __init__(self, hidden_size, eps=1e-6):
                # [FUNCTION DOC] HELPER: Stores the epsilon value and creates the learnable
                # weight parameter, a vector of ones with the hidden size length.
                """
                Phi3RMSNorm is equivalent to T5LayerNorm
                """
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(hidden_size))
                self.variance_epsilon = eps

            def forward(self, hidden_states):
                # [FUNCTION DOC] HELPER: Normalizes the hidden states in float32 by the root
                # mean square over the last axis, casts back to the original dtype, and
                # multiplies by the weight.
                input_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(
                    variance + self.variance_epsilon
                )
                return self.weight * hidden_states.to(input_dtype)

        op = torch.onnx.export(
            Phi3RMSNorm(256).eval(),
            args=(),
            kwargs={"hidden_states": torch.rand((1, 32, 256))},
            dynamic_shapes={
                "hidden_states": {
                    0: "batch_size",
                    1: "seq_len",
                }
            },
            dynamo=True,
        )
        # batch size is not fixed to 1
        self.assertNotEqual(op.model.graph.outputs[0].shape[0], 1)


if __name__ == "__main__":
    common_utils.run_tests()
