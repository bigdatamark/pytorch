# Owner(s): ["module: onnx"]
# [MODULE DOC] Tests for the axis renaming feature of ONNXProgram. A helper builds a small
# program whose inputs have given symbolic shapes, and each test renames axes through a
# mapping and inspects the model input and output shapes. No import-time side effects other
# than imports.
"""Unit tests for the ONNXProgram class."""

from __future__ import annotations

import torch
from torch.onnx._internal._lazy_import import onnx_ir as ir
from torch.testing._internal import common_utils


class ONNXProgramRenameAxesTest(common_utils.TestCase):
    # [CLASS DOC] Test case for ONNXProgram.rename_axes covering string keys, symbolic
    # dimension keys, mixed keys, compound expressions, unknown axes, empty mapping and static
    # dimensions. Stateless.
    """Tests for ONNXProgram.rename_axes method."""

    def _create_onnx_program_with_dynamic_shapes(
        self, input_shapes: list[list[str | int]]
    ) -> torch.onnx.ONNXProgram:
        # [FUNCTION DOC] HELPER: Builds one float input value per given shape list, then an IR
        # model (ir version 9, producer pytorch) whose graph has these inputs, no nodes and
        # the first input as its only output. Returns an ONNXProgram wrapping the model
        # without an exported program.
        """Helper to create an ONNXProgram with specified dynamic shapes."""
        from torch.onnx._internal.exporter._onnx_program import ONNXProgram

        inputs = []
        for i, shape in enumerate(input_shapes):
            inputs.append(
                ir.Value(
                    name=f"input_{i}",
                    type=ir.DataType.FLOAT,
                    shape=ir.Shape(shape),
                )
            )

        model = ir.Model(
            ir.Graph(
                inputs=inputs,
                outputs=[inputs[0]],
                nodes=[],
            ),
            ir_version=9,
            producer_name="pytorch",
            producer_version=torch.__version__,
        )
        return ONNXProgram(model, exported_program=None)

    def test_rename_axes_with_string_keys(self):
        # [FUNCTION DOC] Renames axes of two inputs sharing symbols using a mapping with
        # string keys. Checks both inputs and the output (the first input) show the new names.
        """Test renaming axes using string keys."""
        onnx_program = self._create_onnx_program_with_dynamic_shapes(
            [["s0", "s1"], ["s0", "s2"]]
        )

        onnx_program.rename_axes(
            {
                "s0": "batch_size",
                "s1": "sequence_length",
                "s2": "hidden_size",
            }
        )

        self.assertEqual(
            onnx_program.model.graph.inputs[0].shape,
            ir.Shape(["batch_size", "sequence_length"]),
        )
        self.assertEqual(
            onnx_program.model.graph.inputs[1].shape,
            ir.Shape(["batch_size", "hidden_size"]),
        )
        # Outputs should also be renamed
        self.assertEqual(
            onnx_program.model.graph.outputs[0].shape,
            ir.Shape(["batch_size", "sequence_length"]),
        )

    def test_rename_axes_with_symbolic_dim_keys(self):
        # [FUNCTION DOC] Takes the symbolic dimension objects of the first input and uses them
        # as mapping keys. Checks the first input is renamed and that the same symbol in the
        # second input is renamed too.
        """Test renaming axes using SymbolicDim objects as keys."""
        onnx_program = self._create_onnx_program_with_dynamic_shapes(
            [["s0", "s1"], ["s0", "s2"]]
        )

        # Get SymbolicDim objects from the model
        batch_dim = onnx_program.model.graph.inputs[0].shape[0]
        seq_dim = onnx_program.model.graph.inputs[0].shape[1]

        onnx_program.rename_axes(
            {
                batch_dim: "batch_size",
                seq_dim: "sequence_length",
            }
        )

        self.assertEqual(
            onnx_program.model.graph.inputs[0].shape,
            ir.Shape(["batch_size", "sequence_length"]),
        )
        # s0 should be renamed in both inputs
        self.assertEqual(
            onnx_program.model.graph.inputs[1].shape[0],
            ir.SymbolicDim("batch_size"),
        )

    def test_rename_axes_with_mixed_string_and_symbolic_dim_keys(self):
        # [FUNCTION DOC] Renames with a mapping mixing one symbolic dimension key and string
        # keys, checking all three dimensions of the single input receive their new names.
        """Test renaming axes using a mix of string and SymbolicDim keys."""
        onnx_program = self._create_onnx_program_with_dynamic_shapes(
            [["s0", "s1", "s2"]]
        )

        # Get SymbolicDim object from the model
        batch_dim = onnx_program.model.graph.inputs[0].shape[0]

        onnx_program.rename_axes(
            {
                batch_dim: "batch_size",
                "s1": "sequence_length",
                "s2": "hidden_size",
            }
        )

        self.assertEqual(
            onnx_program.model.graph.inputs[0].shape,
            ir.Shape(["batch_size", "sequence_length", "hidden_size"]),
        )

    def test_rename_axes_with_complex_shape_expressions(self):
        # [FUNCTION DOC] Renames axes where shapes contain compound expressions. Checks a sum
        # expression is replaced as a whole by its own mapping entry while a product
        # expression has its inner name replaced.
        """Test renaming axes in complex shape expressions like 's1 + s2'."""
        onnx_program = self._create_onnx_program_with_dynamic_shapes(
            [["s0", "s1 + s2", "s1*2"]]
        )

        onnx_program.rename_axes(
            {
                "s0": "batch_size",
                "s1": "seq_len",
                "s2": "past_seq_len",
                "s1 + s2": "total_seq_len",
            }
        )

        self.assertEqual(
            onnx_program.model.graph.inputs[0].shape,
            ir.Shape(["batch_size", "total_seq_len", "seq_len*2"]),
        )

    def test_rename_axes_with_nonexistent_axis_is_ignored(self):
        # [FUNCTION DOC] Includes a mapping entry for an axis absent from the model. Checks
        # only the existing axis is renamed and no error is raised.
        """Test that renaming a non-existent axis is silently ignored."""
        onnx_program = self._create_onnx_program_with_dynamic_shapes([["s0", "s1"]])

        # Include a mapping for a non-existent axis
        onnx_program.rename_axes(
            {
                "s0": "batch_size",
                "nonexistent": "should_be_ignored",
            }
        )

        # Only s0 should be renamed
        self.assertEqual(
            onnx_program.model.graph.inputs[0].shape,
            ir.Shape(["batch_size", "s1"]),
        )

    def test_rename_axes_with_empty_mapping(self):
        # [FUNCTION DOC] Calls rename with an empty mapping and checks the first input shape
        # is unchanged.
        """Test renaming with an empty mapping does nothing."""
        onnx_program = self._create_onnx_program_with_dynamic_shapes([["s0", "s1"]])

        original_shape = onnx_program.model.graph.inputs[0].shape

        onnx_program.rename_axes({})

        self.assertEqual(onnx_program.model.graph.inputs[0].shape, original_shape)

    def test_rename_axes_with_static_dimensions_unchanged(self):
        # [FUNCTION DOC] Renames axes of a shape with an integer dimension in the middle and
        # checks the integer is preserved while symbolic ones are renamed.
        """Test that static (integer) dimensions are not affected by renaming."""
        onnx_program = self._create_onnx_program_with_dynamic_shapes([["s0", 10, "s1"]])

        onnx_program.rename_axes(
            {
                "s0": "batch_size",
                "s1": "hidden_size",
            }
        )

        self.assertEqual(
            onnx_program.model.graph.inputs[0].shape,
            ir.Shape(["batch_size", 10, "hidden_size"]),
        )


if __name__ == "__main__":
    common_utils.run_tests()
