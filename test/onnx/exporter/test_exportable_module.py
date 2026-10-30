# Owner(s): ["module: onnx"]
# [MODULE DOC] Tests for the ExportableModule abstract base class. Defines several small
# concrete test modules (plain, with keyword arguments, with dynamic shapes, with names, with
# multiple outputs) and a test case checking abstractness, default hooks, custom hooks and the
# ONNX export entry point. No import-time side effects other than class definitions.
"""Unit tests for ExportableModule."""

from __future__ import annotations

from typing import Any

import torch
from torch.onnx._internal.exporter import _exportable_module
from torch.testing._internal import common_utils


class SimpleExportableModel(_exportable_module.ExportableModule):
    # [CLASS DOC] HELPER: Minimal concrete ExportableModule holding one linear layer from ten
    # to five features. Its example arguments are a single random batch of two rows and no
    # keyword arguments; all other hooks keep their defaults.
    """A simple concrete implementation of ExportableModule for testing."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Initializes the base module and creates the linear layer from
        # ten to five features.
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the linear layer to the input and returns the result.
        return self.linear(x)

    def example_arguments(self) -> tuple[tuple[Any], dict[str, Any] | None]:
        # [FUNCTION DOC] HELPER: Returns a pair of one positional argument, a random tensor of
        # two by ten, and no keyword arguments.
        return (torch.randn(2, 10),), None


class ExportableModelWithKwargs(_exportable_module.ExportableModule):
    # [CLASS DOC] HELPER: Concrete ExportableModule whose forward takes an input plus a scale
    # keyword with default one and multiplies the linear output by it. Its example arguments
    # include the scale keyword set to two.
    """ExportableModule with keyword arguments."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Initializes the base module and creates the linear layer from
        # ten to five features.
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x, scale: float = 1.0):
        # [FUNCTION DOC] HELPER: Returns the linear layer output multiplied by the scale
        # value.
        return self.linear(x) * scale

    def example_arguments(self) -> tuple[tuple[Any], dict[str, Any] | None]:
        # [FUNCTION DOC] HELPER: Returns a random two by ten tensor as positional arguments
        # and a keyword mapping that sets the scale to two.
        return (torch.randn(2, 10),), {"scale": 2.0}


class ExportableModelWithDynamicShapes(_exportable_module.ExportableModule):
    # [CLASS DOC] HELPER: Concrete ExportableModule that overrides the dynamic shapes hook to
    # mark the first dimension of the input as a dynamic batch axis.
    """ExportableModule with dynamic shapes."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Initializes the base module and creates the linear layer from
        # ten to five features.
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the linear layer to the input and returns the result.
        return self.linear(x)

    def example_arguments(self) -> tuple[tuple[Any], dict[str, Any] | None]:
        # [FUNCTION DOC] HELPER: Returns a random two by ten tensor as the only positional
        # argument and no keyword arguments.
        return (torch.randn(2, 10),), None

    def dynamic_shapes(self):
        # [FUNCTION DOC] HELPER: Returns a mapping from the input argument name to a mapping
        # that names axis zero as the batch size.
        return {"x": {0: "batch_size"}}


class ExportableModelWithNames(_exportable_module.ExportableModule):
    # [CLASS DOC] HELPER: Concrete ExportableModule that overrides the input names and output
    # names hooks, with one input name and one output name.
    """ExportableModule with input and output names."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Initializes the base module and creates the linear layer from
        # ten to five features.
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the linear layer to the input and returns the result.
        return self.linear(x)

    def example_arguments(self) -> tuple[tuple[Any], dict[str, Any] | None]:
        # [FUNCTION DOC] HELPER: Returns a random two by ten tensor as the only positional
        # argument and no keyword arguments.
        return (torch.randn(2, 10),), None

    def input_names(self):
        # [FUNCTION DOC] HELPER: Returns a list with the single input name input_tensor.
        return ["input_tensor"]

    def output_names(self):
        # [FUNCTION DOC] HELPER: Returns a list with the single output name output_tensor.
        return ["output_tensor"]


class ExportableModelMultipleOutputs(_exportable_module.ExportableModule):
    # [CLASS DOC] HELPER: Concrete ExportableModule with two linear layers (ten to five and
    # ten to three features) whose forward returns both results as a tuple, and with two
    # output names.
    """ExportableModule with multiple outputs."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Initializes the base module and creates the two linear
        # layers.
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 5)
        self.linear2 = torch.nn.Linear(10, 3)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the pair of results of both linear layers applied to
        # the input.
        return self.linear1(x), self.linear2(x)

    def example_arguments(self) -> tuple[tuple[Any], dict[str, Any] | None]:
        # [FUNCTION DOC] HELPER: Returns a random two by ten tensor as the only positional
        # argument and no keyword arguments.
        return (torch.randn(2, 10),), None

    def output_names(self):
        # [FUNCTION DOC] HELPER: Returns a list with the output names output1 and output2.
        return ["output1", "output2"]


class TestExportableModule(common_utils.TestCase):
    # [CLASS DOC] Test case for ExportableModule behavior using the helper modules above.
    # Stateless; each test builds its own model.
    """Tests for ExportableModule abstract class and its implementations."""

    def test_abstract_class_cannot_be_instantiated(self):
        # [FUNCTION DOC] Asserts that instantiating the abstract base class directly raises a
        # TypeError.
        """Test that ExportableModule cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            _exportable_module.ExportableModule()

    def test_simple_exportable_model(self):
        # [FUNCTION DOC] Builds the simple model, checks it is both a torch module and an
        # ExportableModule, that example arguments are a one-element tuple with no keywords,
        # and that the forward pass on them gives a two by five output.
        """Test basic ExportableModule implementation."""
        model = SimpleExportableModel()
        self.assertIsInstance(model, torch.nn.Module)
        self.assertIsInstance(model, _exportable_module.ExportableModule)

        # Test example_arguments
        args, kwargs = model.example_arguments()
        self.assertIsInstance(args, tuple)
        self.assertEqual(len(args), 1)
        self.assertIsNone(kwargs)

        # Test forward pass
        output = model(*args)
        self.assertEqual(output.shape, (2, 5))

    def test_exportable_model_with_kwargs(self):
        # [FUNCTION DOC] Builds the keyword model, checks the example arguments hold a keyword
        # mapping with scale two, and that calling forward with them gives a two by five
        # output.
        """Test ExportableModule with keyword arguments."""
        model = ExportableModelWithKwargs()

        args, kwargs = model.example_arguments()
        self.assertIsInstance(args, tuple)
        self.assertIsInstance(kwargs, dict)
        self.assertEqual(kwargs["scale"], 2.0)

        # Test forward pass with kwargs
        output = model(*args, **kwargs)
        self.assertEqual(output.shape, (2, 5))

    def test_dynamic_shapes_default(self):
        # [FUNCTION DOC] Asserts the default dynamic shapes hook of the simple model returns
        # None.
        """Test that dynamic_shapes returns None by default."""
        model = SimpleExportableModel()
        self.assertIsNone(model.dynamic_shapes())

    def test_dynamic_shapes_custom(self):
        # [FUNCTION DOC] Asserts the overriding model returns a non-empty mapping containing
        # the input name x.
        """Test custom dynamic_shapes implementation."""
        model = ExportableModelWithDynamicShapes()
        dynamic_shapes = model.dynamic_shapes()
        self.assertIsNotNone(dynamic_shapes)
        self.assertIsInstance(dynamic_shapes, dict)
        self.assertIn("x", dynamic_shapes)

    def test_input_names_default(self):
        # [FUNCTION DOC] Asserts the default input names hook of the simple model returns
        # None.
        """Test that input_names returns None by default."""
        model = SimpleExportableModel()
        self.assertIsNone(model.input_names())

    def test_output_names_default(self):
        # [FUNCTION DOC] Asserts the default output names hook of the simple model returns
        # None.
        """Test that output_names returns None by default."""
        model = SimpleExportableModel()
        self.assertIsNone(model.output_names())

    def test_input_output_names_custom(self):
        # [FUNCTION DOC] Builds the model with names and checks each hook returns a
        # one-element sequence holding the expected custom name.
        """Test custom input_names and output_names implementation."""
        model = ExportableModelWithNames()

        input_names = model.input_names()
        self.assertIsNotNone(input_names)
        self.assertEqual(len(input_names), 1)
        self.assertEqual(input_names[0], "input_tensor")

        output_names = model.output_names()
        self.assertIsNotNone(output_names)
        self.assertEqual(len(output_names), 1)
        self.assertEqual(output_names[0], "output_tensor")

    def test_multiple_outputs(self):
        # [FUNCTION DOC] Runs the two-output model on its example arguments and checks the two
        # result shapes, then checks the output names hook gives the two expected names in
        # order.
        """Test ExportableModule with multiple outputs."""
        model = ExportableModelMultipleOutputs()

        args, kwargs = model.example_arguments()
        output1, output2 = model(*args)
        self.assertEqual(output1.shape, (2, 5))
        self.assertEqual(output2.shape, (2, 3))

        output_names = model.output_names()
        self.assertEqual(len(output_names), 2)
        self.assertEqual(output_names[0], "output1")
        self.assertEqual(output_names[1], "output2")

    def test_to_onnx_calls_export(self):
        # [FUNCTION DOC] Calls the to_onnx method of the simple model and asserts the result
        # is an ONNXProgram. This performs a real export; detailed export behavior is covered
        # elsewhere.
        """Test that to_onnx correctly calls torch.onnx.export."""
        model = SimpleExportableModel()

        # This test verifies that to_onnx returns an ONNXProgram
        # The actual export functionality is tested elsewhere
        onnx_program = model.to_onnx()
        self.assertIsNotNone(onnx_program)
        self.assertIsInstance(onnx_program, torch.onnx.ONNXProgram)

    def test_exportable_module_is_nn_module(self):
        # [FUNCTION DOC] Checks the model exposes the usual module methods for parameters and
        # state dict handling, and that its parameter list is non-empty.
        """Test that ExportableModule is a proper nn.Module."""
        model = SimpleExportableModel()

        # Test that it has standard nn.Module methods
        self.assertTrue(hasattr(model, "parameters"))
        self.assertTrue(hasattr(model, "named_parameters"))
        self.assertTrue(hasattr(model, "state_dict"))
        self.assertTrue(hasattr(model, "load_state_dict"))

        # Test that parameters work correctly
        params = list(model.parameters())
        self.assertGreater(len(params), 0)

    def test_example_arguments_must_be_implemented(self):
        # [FUNCTION DOC] Defines a subclass that implements only forward and asserts that
        # instantiating it raises a TypeError because the abstract example arguments method is
        # missing.
        """Test that example_arguments must be implemented by subclasses."""

        class IncompleteExportableModel(_exportable_module.ExportableModule):
            # [CLASS DOC] HELPER: Subclass without the required example arguments method, so
            # it stays abstract.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input multiplied by two.
                return x * 2

        with self.assertRaises(TypeError):
            IncompleteExportableModel()

    def test_forward_with_complex_inputs(self):
        # [FUNCTION DOC] Defines a subclass taking two tensor inputs and implementing example
        # arguments with two positional tensors. Checks forward on those arguments returns the
        # elementwise sum with shape two by three.
        """Test ExportableModule with more complex input types."""

        class ComplexInputModel(_exportable_module.ExportableModule):
            # [CLASS DOC] HELPER: Concrete ExportableModule whose forward adds two inputs and
            # whose example arguments are two random two by three tensors.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the sum of the two inputs.
                return x + y

            def example_arguments(self):
                # [FUNCTION DOC] HELPER: Returns two random tensors of two by three as
                # positional arguments and no keyword arguments.
                return (torch.randn(2, 3), torch.randn(2, 3)), None

        model = ComplexInputModel()
        args, kwargs = model.example_arguments()
        output = model(*args)
        self.assertEqual(output.shape, (2, 3))


if __name__ == "__main__":
    common_utils.run_tests()
