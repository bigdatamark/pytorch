# Owner(s): ["module: onnx"]
# [MODULE DOC] Tests for the op recorder of the ONNX exporter building module. Each test
# records ONNX opset calls through an OpRecorder used as the active onnxscript evaluator and
# inspects the nodes it produced (constants created for Python values, sequence construction,
# concat insertion). No import-time side effects other than imports.
"""Unit tests for the _building module."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxscript

import torch
from torch.onnx._internal.exporter import _building, _tensors
from torch.testing._internal import common_utils


class TestOpRecorder(common_utils.TestCase):
    # [CLASS DOC] Test case for the OpRecorder tracer. setUp creates an opset 17
    # default-domain Opset, an OpRecorder over it with an empty constant farm, and an ONNX IR
    # model with an empty main graph (kept as attributes, mostly unused by the tests). Each
    # test records one opset call and checks the resulting node list.
    def setUp(self):
        # [FUNCTION DOC] Builds the per-test fixtures: the opset version 17 attribute, the
        # default-domain Opset, the OpRecorder with an empty constant farm, and an empty IR
        # model with ir version 9 whose producer is pytorch with the current torch version
        # string.
        super().setUp()
        self.opset_version = 17
        self.opset = onnxscript.values.Opset("", self.opset_version)
        self.recorder = _building.OpRecorder(opset=self.opset, constant_farm={})

        self.model = ir.Model(
            graph=ir.Graph(
                [],
                [],
                nodes=[],
                opset_imports={
                    "": self.opset_version,
                },
                name="main_graph",
            ),
            ir_version=9,
            producer_name="pytorch",
            producer_version=torch.__version__,
        )

    def test_python_constant_added_as_constant_nodes(self):
        # [FUNCTION DOC] Reshapes a symbolic tensor with a plain Python list as the shape.
        # Expects the recorder to emit two nodes: a Constant node holding the list as an array
        # value and then the Reshape node.
        input_x = _tensors.SymbolicTensor(
            opset=self.opset, name="input_x", shape=ir.Shape([2, 3, 4])
        )
        new_shape = [3, 2, 4]

        with onnxscript.evaluator.default_as(tracer := self.recorder):
            _ = self.opset.Reshape(input_x, new_shape)

        self.assertEqual(len(tracer.nodes), 2)
        self.assertEqual(tracer.nodes[0].op_type, "Constant")
        self.assertEqual(
            tracer.nodes[0].attributes["value"].value.numpy(), np.array(new_shape)
        )
        self.assertEqual(tracer.nodes[1].op_type, "Reshape")

    def test_process_python_sequence_with_allowed_sequence_type(self):
        # [FUNCTION DOC] Calls SequenceAt with a Python list of three symbolic tensors and an
        # index. Expects three nodes in total, the second being the SequenceConstruct node the
        # recorder built from the list.
        input_x = _tensors.SymbolicTensor(
            opset=self.opset, name="input_x", shape=ir.Shape([2, 3])
        )
        input_y = _tensors.SymbolicTensor(
            opset=self.opset, name="input_y", shape=ir.Shape([2, 4])
        )
        input_z = _tensors.SymbolicTensor(
            opset=self.opset, name="input_z", shape=ir.Shape([1, 3])
        )

        with onnxscript.evaluator.default_as(tracer := self.recorder):
            _ = self.opset.SequenceAt([input_x, input_y, input_z], 1)

        self.assertEqual(len(tracer.nodes), 3)
        self.assertEqual(tracer.nodes[1].op_type, "SequenceConstruct")

    def test_process_python_sequence_with_variadic_input(self):
        # [FUNCTION DOC] Calls Max, a variadic op, with symbolic tensors mixed with the Python
        # constant zero. Expects two nodes total, the first being a Constant node for the
        # Python value; no sequence construct is created for variadic inputs.
        input_x = _tensors.SymbolicTensor(
            opset=self.opset, name="input_x", shape=ir.Shape([2, 3])
        )
        input_y = _tensors.SymbolicTensor(
            opset=self.opset, name="input_y", shape=ir.Shape([2, 4])
        )
        input_z = _tensors.SymbolicTensor(
            opset=self.opset, name="input_z", shape=ir.Shape([1, 3])
        )

        with onnxscript.evaluator.default_as(tracer := self.recorder):
            _ = self.opset.Max(input_x, input_y, 0, input_z)

        self.assertEqual(len(tracer.nodes), 2)
        self.assertEqual(tracer.nodes[0].op_type, "Constant")

    def test_process_python_sequence_creates_extra_concat(self):
        # [FUNCTION DOC] Adds a Python list of two scalar symbolic tensors to another tensor.
        # Expects six nodes total and that the second to last node is a Concat along axis 0
        # that packs the scalar elements into one tensor.
        # Elements in the list must be 0D tensors
        input_x = _tensors.SymbolicTensor(
            opset=self.opset, name="input_x", shape=ir.Shape([])
        )
        input_y = _tensors.SymbolicTensor(
            opset=self.opset, name="input_y", shape=ir.Shape([])
        )
        input_z = _tensors.SymbolicTensor(
            opset=self.opset, name="input_z", shape=ir.Shape([4, 3])
        )

        with onnxscript.evaluator.default_as(tracer := self.recorder):
            _ = self.opset.Add([input_x, input_y], input_z)

        self.assertEqual(len(tracer.nodes), 6)
        self.assertEqual(tracer.nodes[-2].op_type, "Concat")
        self.assertEqual(tracer.nodes[-2].attributes["axis"].value, 0)

    def test_process_python_sequence_mix_symbolic_constant_creates_extra_concat(self):
        # [FUNCTION DOC] Same as the concat test but the list mixes a scalar symbolic tensor
        # followed by the Python constant 42. Expects five nodes and a Concat node with axis 0
        # as the second to last node.
        # Elements in the list must be 0D tensors
        input_x = _tensors.SymbolicTensor(
            opset=self.opset, name="input_x", shape=ir.Shape([])
        )
        input_z = _tensors.SymbolicTensor(
            opset=self.opset, name="input_z", shape=ir.Shape([4, 3])
        )

        with onnxscript.evaluator.default_as(tracer := self.recorder):
            _ = self.opset.Add([input_x, 42], input_z)

        self.assertEqual(len(tracer.nodes), 5)
        self.assertEqual(tracer.nodes[-2].op_type, "Concat")
        self.assertEqual(tracer.nodes[-2].attributes["axis"].value, 0)

    def test_process_python_sequence_mix_constant_symbolic_creates_extra_concat(self):
        # [FUNCTION DOC] Same as the previous mixed-list test but with the Python constant
        # first and the scalar symbolic tensor second. Expects five nodes and a Concat node
        # with axis 0 as the second to last node.
        # Elements in the list must be 0D tensors
        input_x = _tensors.SymbolicTensor(
            opset=self.opset, name="input_x", shape=ir.Shape([])
        )
        input_z = _tensors.SymbolicTensor(
            opset=self.opset, name="input_z", shape=ir.Shape([4, 3])
        )

        with onnxscript.evaluator.default_as(tracer := self.recorder):
            # Constant first
            _ = self.opset.Add([42, input_x], input_z)

        self.assertEqual(len(tracer.nodes), 5)
        self.assertEqual(tracer.nodes[-2].op_type, "Concat")
        self.assertEqual(tracer.nodes[-2].attributes["axis"].value, 0)


if __name__ == "__main__":
    common_utils.run_tests()
