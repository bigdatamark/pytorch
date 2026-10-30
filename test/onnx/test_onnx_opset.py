# Owner(s): ["module: onnx"]
# [MODULE DOC] Tests that the TorchScript-based ONNX exporter emits the expected operator sequence
# (and selected node attributes) for different opset versions. Defines two checking helpers and one
# test class that exports small modules and compares the graph node by node. Import-time side
# effect: the default opset of the first helper is read from the exporter globals when the module
# is loaded.

import io
import itertools

import onnx

import pytorch_test_common

import torch
import torch.onnx
from torch.nn import Module
from torch.onnx import producer_name, producer_version
from torch.onnx._internal.torchscript_exporter._globals import GLOBALS
from torch.testing._internal import common_utils


def check_onnx_opset_operator(
    model, ops, opset_version=GLOBALS.export_onnx_opset_version
):
    # [FUNCTION DOC] Validate a loaded ONNX model against an expected list of operators. Args:
    # model is a loaded ONNX model; ops is a list of mappings, one per graph node in order, each
    # with an op_name and optionally an attributes list of mappings from attribute field to
    # expected value; opset_version is the expected opset (default taken from exporter globals).
    # Checks producer name, producer version and first opset import, runs the ONNX checker, then
    # compares node count, op types and, when given, attribute counts and field values. Raises
    # AssertionError on any mismatch. No return value.
    # check_onnx_components
    if not (
        model.producer_name == producer_name
        and model.producer_version == producer_version
        and model.opset_import[0].version == opset_version
    ):
        raise AssertionError(
            f"Model metadata mismatch: producer_name={model.producer_name!r} (expected {producer_name!r}), "
            f"producer_version={model.producer_version!r} (expected {producer_version!r}), "
            f"opset_version={model.opset_import[0].version} (expected {opset_version})"
        )

    # check the schema with the onnx checker
    onnx.checker.check_model(model)

    # check target type and attributes
    graph = model.graph
    # ops should contain an object for each node
    # in graph.node, in the right order.
    # At least the op_name should be specified,
    # but the op's attributes can optionally be
    # specified as well
    if len(ops) != len(graph.node):
        raise AssertionError(f"Expected {len(ops)} ops, got {len(graph.node)}")
    for i in range(len(ops)):
        if graph.node[i].op_type != ops[i]["op_name"]:
            raise AssertionError(
                f"Expected op {ops[i]['op_name']}, got {graph.node[i].op_type}"
            )
        if "attributes" in ops[i]:
            attributes = ops[i]["attributes"]
            if len(attributes) != len(graph.node[i].attribute):
                raise AssertionError(
                    f"Expected {len(attributes)} attributes, got {len(graph.node[i].attribute)}"
                )
            for j in range(len(attributes)):
                for attribute_field in attributes[j]:
                    actual = getattr(graph.node[i].attribute[j], attribute_field)
                    if attributes[j][attribute_field] != actual:
                        raise AssertionError(
                            f"Attribute {attribute_field!r} mismatch on node {i}, attribute {j}: "
                            f"expected {attributes[j][attribute_field]!r}, got {actual!r}"
                        )


def check_onnx_opsets_operator(
    module,
    x,
    ops,
    opset_versions,
    training=torch.onnx.TrainingMode.EVAL,
    input_names=None,
    dynamic_axes=None,
):
    # [FUNCTION DOC] Export a module once per requested opset with the TorchScript exporter (dynamo
    # disabled) into an in-memory buffer, reload it, and validate it. Args: module and x are the
    # model and its example input(s); ops maps each opset version to its expected operator list;
    # opset_versions lists the versions to try; training sets the export mode (default eval);
    # input_names and dynamic_axes are passed to the exporter. Raises AssertionError (from the
    # checker helper) if a graph differs from expectations.
    for opset_version in opset_versions:
        f = io.BytesIO()
        torch.onnx.export(
            module,
            x,
            f,
            opset_version=opset_version,
            training=training,
            input_names=input_names,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
        model = onnx.load(io.BytesIO(f.getvalue()))
        check_onnx_opset_operator(model, ops[opset_version], opset_version)


class TestONNXOpset(pytorch_test_common.ExportTestCase):
    # [CLASS DOC] Export test case (derived from the shared ExportTestCase base) with one test per
    # operator, each asserting the exported operator sequence for specific opset versions.
    def test_opset_fallback(self):
        # [FUNCTION DOC] Checks that an elementwise NaN test exports to a single IsNaN node for
        # opset 9 and 10 (the opset 9 definition is reused for 10).
        class MyModule(Module):
            # [CLASS DOC] HELPER: Module whose forward returns the NaN mask of its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return a boolean tensor marking NaN entries of x.
                return torch.isnan(x)

        ops = [{"op_name": "IsNaN"}]
        ops = {9: ops, 10: ops}
        x = torch.tensor([1.0, float("nan"), 2.0])
        check_onnx_opsets_operator(MyModule(), x, ops, opset_versions=[9, 10])

    def test_topk(self):
        # [FUNCTION DOC] Checks top-k export: opset 9 gives one TopK node carrying k as an
        # attribute; opset 10 gives a Constant plus TopK with only the axis attribute. Also checks
        # a scripted module taking k as a tensor input (opset 10): Constant, Reshape, TopK.
        class MyModule(Module):
            # [CLASS DOC] HELPER: Module whose forward returns the top 3 values and indices of x.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return top 3 values and indices of x along last dim.
                return torch.topk(x, 3)

        ops_9 = [
            {
                "op_name": "TopK",
                "attributes": [
                    {"name": "axis", "i": -1, "type": 2},
                    {"name": "k", "i": 3, "type": 2},
                ],
            }
        ]
        ops_10 = [
            {"op_name": "Constant"},
            {"op_name": "TopK", "attributes": [{"name": "axis", "i": -1, "type": 2}]},
        ]
        ops = {9: ops_9, 10: ops_10}
        x = torch.arange(1.0, 6.0, requires_grad=True)
        check_onnx_opsets_operator(MyModule(), x, ops, opset_versions=[9, 10])

        # test with dynamic k
        class MyModuleDynamic(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: Scripted module (script method) taking the input and a tensor k.
            @torch.jit.script_method
            def forward(self, input, k):
                # [FUNCTION DOC] HELPER: Return top k values and indices of input (k is a tensor).
                return torch.topk(input, k)

        ops_10 = [
            {"op_name": "Constant", "attributes": [{"name": "value", "type": 4}]},
            {"op_name": "Reshape"},
            {"op_name": "TopK", "attributes": [{"name": "axis", "i": -1, "type": 2}]},
        ]
        ops = {10: ops_10}
        x = torch.arange(1.0, 6.0, requires_grad=True)
        k = torch.tensor(3)
        module = MyModuleDynamic()
        check_onnx_opsets_operator(module, (x, k), ops, opset_versions=[10])

    def test_maxpool(self):
        # [FUNCTION DOC] Checks 1-D max pooling export. Opset 9 emits MaxPool with kernel, pads and
        # strides; opset 10 adds ceil_mode and dilations. A second case with dilation 2 (opset 10
        # only) expects the dilations attribute to be 2.
        module = torch.nn.MaxPool1d(2, stride=1)

        ops_9 = [
            {
                "op_name": "MaxPool",
                "attributes": [
                    {"name": "kernel_shape", "ints": [2], "type": 7},
                    {"name": "pads", "ints": [0, 0], "type": 7},
                    {"name": "strides", "ints": [1], "type": 7},
                ],
            }
        ]
        ops_10 = [
            {
                "op_name": "MaxPool",
                "attributes": [
                    {"name": "ceil_mode", "i": 0, "type": 2},
                    {"name": "dilations", "ints": [1], "type": 7},
                    {"name": "kernel_shape", "ints": [2], "type": 7},
                    {"name": "pads", "ints": [0, 0], "type": 7},
                    {"name": "strides", "ints": [1], "type": 7},
                ],
            }
        ]
        ops = {9: ops_9, 10: ops_10}
        x = torch.randn(20, 16, 50)
        check_onnx_opsets_operator(module, x, ops, opset_versions=[9, 10])

        # add test with dilations
        module = torch.nn.MaxPool1d(2, stride=1, dilation=2)

        ops_10 = [
            {
                "op_name": "MaxPool",
                "attributes": [
                    {"name": "ceil_mode", "i": 0, "type": 2},
                    {"name": "dilations", "ints": [2], "type": 7},
                    {"name": "kernel_shape", "ints": [2], "type": 7},
                    {"name": "pads", "ints": [0, 0], "type": 7},
                    {"name": "strides", "ints": [1], "type": 7},
                ],
            }
        ]
        ops = {10: ops_10}
        x = torch.randn(20, 16, 50)
        check_onnx_opsets_operator(module, x, ops, opset_versions=[10])

    def test_upsample(self):
        # [FUNCTION DOC] Checks nearest-neighbor interpolation with constant sizes: opset 8 gives
        # an Upsample node with a scales attribute; opset 9 gives a Constant feeding Upsample.
        class MyModule(Module):
            # [CLASS DOC] HELPER: Module doubling the spatial size of x with nearest interpolation,
            # with sizes converted to plain integers so the exporter sees constants.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Compute twice the spatial dims of x as integers and return
                # x interpolated to that size using nearest mode.
                size = [v * 2 for v in x.size()[2:]]
                size = [int(i) for i in size]
                return torch.nn.functional.interpolate(x, size=size, mode="nearest")

        module = MyModule()
        ops8 = [
            {
                "op_name": "Upsample",
                "attributes": [
                    {"name": "mode", "s": (b"nearest"), "type": 3},
                    {"name": "scales", "floats": [1.0, 1.0, 2.0, 2.0], "type": 6},
                ],
            }
        ]
        ops9 = [
            {"op_name": "Constant"},
            {
                "op_name": "Upsample",
                "attributes": [{"name": "mode", "s": (b"nearest"), "type": 3}],
            },
        ]
        ops = {8: ops8, 9: ops9}
        x = torch.randn(2, 2, 2, 2)
        check_onnx_opsets_operator(module, x, ops, opset_versions=[8, 9])

    def test_cast_constant(self):
        # [FUNCTION DOC] Checks subtracting an integer constant from a long tensor: opset 8 inserts
        # a Cast of the constant to int64 between Constant and Sub; opset 9 needs no Cast.
        class MyModule(Module):
            # [CLASS DOC] HELPER: Module that subtracts one from its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return x minus one.
                return x - 1

        module = MyModule()
        ops_8 = [
            {"op_name": "Constant"},
            {"op_name": "Cast", "attributes": [{"name": "to", "i": 7, "type": 2}]},
            {"op_name": "Sub"},
        ]
        ops_9 = [{"op_name": "Constant"}, {"op_name": "Sub"}]
        ops = {8: ops_8, 9: ops_9}
        x = torch.ones(5, 6, dtype=torch.long)
        check_onnx_opsets_operator(module, x, ops, opset_versions=[8, 9])

    def test_slice(self):
        # [FUNCTION DOC] Checks slicing export. Static slice: opset 9 gives Slice with attributes,
        # opset 10 gives four Constants and an attribute-free Slice. Then a scripted model with a
        # slice ending at the dynamic size is checked with and without dynamic axes (opset 10).
        class MyModule(Module):
            # [CLASS DOC] HELPER: Module returning the first row of its input via a static slice.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the slice of x from index 0 up to 1 on dim 0.
                return x[0:1]

        ops_9 = [
            {
                "op_name": "Slice",
                "attributes": [
                    {"name": "axes", "ints": [0], "type": 7},
                    {"name": "ends", "ints": [1], "type": 7},
                    {"name": "starts", "ints": [0], "type": 7},
                ],
            }
        ]
        ops_10 = [
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Slice", "attributes": []},
        ]
        ops = {9: ops_9, 10: ops_10}
        x = torch.randn(3)
        check_onnx_opsets_operator(MyModule(), x, ops, opset_versions=[9, 10])

        class DynamicSliceModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: Scripted module whose slice end is the runtime size of dim 0.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return x sliced from index 1 up to its size along dim 0.
                return x[1 : x.size(0)]

        module = DynamicSliceModel()
        x = torch.rand(1, 2)
        ops_10 = [
            {"op_name": "Shape"},
            {"op_name": "Constant"},
            {"op_name": "Gather", "attributes": [{"name": "axis", "i": 0, "type": 2}]},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {
                "op_name": "Unsqueeze",
                "attributes": [{"name": "axes", "i": 0, "type": 7}],
            },
            {"op_name": "Constant"},
            {"op_name": "Slice", "attributes": []},
        ]
        ops = {10: ops_10}
        check_onnx_opsets_operator(
            module,
            x,
            ops,
            opset_versions=[10],
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
        )

        ops_10 = [
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Slice", "attributes": []},
        ]
        ops = {10: ops_10}
        check_onnx_opsets_operator(module, x, ops, opset_versions=[10])

    def test_flip(self):
        # [FUNCTION DOC] Checks that flipping a 2x3 double tensor along dim 0 exports (opset 10) as
        # four Constants followed by an attribute-free Slice.
        class MyModule(Module):
            # [CLASS DOC] HELPER: Module that flips its input along dimension 0.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return x flipped along dim 0.
                return torch.flip(x, dims=[0])

        ops_10 = [
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Slice", "attributes": []},
        ]
        ops = {10: ops_10}
        import numpy

        x = torch.tensor(numpy.arange(6.0).reshape(2, 3))
        check_onnx_opsets_operator(MyModule(), x, ops, opset_versions=[10])

    def test_dropout(self):
        # [FUNCTION DOC] Checks dropout export in both modes for opset 9 and 10: training mode
        # keeps a Dropout node with ratio 0.5; eval mode replaces it with Identity.
        class MyModule(Module):
            # [CLASS DOC] HELPER: Module holding a Dropout layer with probability 0.5.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the module and register its dropout layer.
                super().__init__()
                self.dropout = torch.nn.Dropout(0.5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply the dropout layer to x and return the result.
                return self.dropout(x)

        x = torch.randn(1, 2, 3)

        # we should only export the onnx Dropout op in training mode; test both modes

        # test training mode
        ops = [
            {
                "op_name": "Dropout",
                "attributes": [{"name": "ratio", "f": 0.5, "type": 1}],
            }
        ]
        ops = {9: ops, 10: ops}
        check_onnx_opsets_operator(
            MyModule(),
            x,
            ops,
            opset_versions=[9, 10],
            training=torch.onnx.TrainingMode.TRAINING,
        )

        # test eval mode
        ops = [{"op_name": "Identity"}]
        ops = {9: ops, 10: ops}
        check_onnx_opsets_operator(
            MyModule(),
            x,
            ops,
            opset_versions=[9, 10],
            training=torch.onnx.TrainingMode.EVAL,
        )

    def test_full(self):
        # [FUNCTION DOC] Checks that creating a 3x4 tensor filled with a scalar tensor exports as
        # Constant, ConstantOfShape and Add for opset 9 and 10.
        class MyModule(Module):
            # [CLASS DOC] HELPER: Module creating a 3x4 tensor filled with the input value.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return a 3x4 tensor filled with the scalar tensor x.
                return torch.full((3, 4), x)

        ops = [
            {"op_name": "Constant"},
            {"op_name": "ConstantOfShape"},
            {"op_name": "Add"},
        ]
        ops = {9: ops, 10: ops}
        x = torch.tensor(12.0)
        check_onnx_opsets_operator(MyModule(), x, ops, opset_versions=[9, 10])

    def test_interpolate(self):
        # [FUNCTION DOC] Checks nearest interpolation with a doubled spatial size for opset 9
        # (Upsample) and 10 (Resize). Covers a dynamic-axes export with a long Shape/Gather/Mul
        # chain, a static export where sizes are folded, and a variant converting sizes to integers
        # (Constant plus Upsample or Resize with mode attribute).
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module interpolating x to twice its spatial size (dynamic).
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return x nearest-interpolated to double its last two dims.
                size = [v * 2 for v in x.size()[2:]]
                return torch.nn.functional.interpolate(x, size=size, mode="nearest")

        ops_9 = [
            {"op_name": "Shape"},
            {"op_name": "Constant"},
            {"op_name": "Gather"},
            {"op_name": "Shape"},
            {"op_name": "Constant"},
            {"op_name": "Gather"},
            {"op_name": "Constant"},
            {"op_name": "Mul"},
            {"op_name": "Constant"},
            {"op_name": "Mul"},
            {"op_name": "Unsqueeze"},
            {"op_name": "Unsqueeze"},
            {"op_name": "Concat"},
            {"op_name": "Cast"},
            {"op_name": "Shape"},
            {"op_name": "Slice"},
            {"op_name": "Cast"},
            {"op_name": "Div"},
            {"op_name": "Constant"},
            {"op_name": "Concat"},
            {
                "op_name": "Upsample",
                "attributes": [{"name": "mode", "s": (b"nearest"), "type": 3}],
            },
        ]
        ops_10 = [
            {"op_name": "Shape"},
            {"op_name": "Constant"},
            {"op_name": "Gather"},
            {"op_name": "Shape"},
            {"op_name": "Constant"},
            {"op_name": "Gather"},
            {"op_name": "Constant"},
            {"op_name": "Mul"},
            {"op_name": "Constant"},
            {"op_name": "Mul"},
            {"op_name": "Unsqueeze"},
            {"op_name": "Unsqueeze"},
            {"op_name": "Concat"},
            {"op_name": "Cast"},
            {"op_name": "Shape"},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Slice"},
            {"op_name": "Cast"},
            {"op_name": "Div"},
            {"op_name": "Constant"},
            {"op_name": "Concat"},
            {
                "op_name": "Resize",
                "attributes": [{"name": "mode", "s": (b"nearest"), "type": 3}],
            },
        ]

        ops = {9: ops_9, 10: ops_10}
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        check_onnx_opsets_operator(
            MyModel(),
            x,
            ops,
            opset_versions=[9, 10],
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
        )

        ops_9 = [
            {"op_name": "Constant"},
            {"op_name": "Shape"},
            {"op_name": "Slice"},
            {"op_name": "Cast"},
            {"op_name": "Div"},
            {"op_name": "Constant"},
            {"op_name": "Concat"},
            {
                "op_name": "Upsample",
                "attributes": [{"name": "mode", "s": (b"nearest"), "type": 3}],
            },
        ]
        ops_10 = [
            {"op_name": "Constant"},
            {"op_name": "Shape"},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Slice"},
            {"op_name": "Cast"},
            {"op_name": "Div"},
            {"op_name": "Constant"},
            {"op_name": "Concat"},
            {"op_name": "Resize"},
        ]

        ops = {9: ops_9, 10: ops_10}
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        check_onnx_opsets_operator(MyModel(), x, ops, opset_versions=[9, 10])

        class MyDynamicModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Like the first interpolation module, sizes forced to ints.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return x nearest-interpolated to doubled constant sizes.
                size = [v * 2 for v in x.size()[2:]]
                # work around for now: turn the dynamic sizes into constant
                size = [int(i) for i in size]
                return torch.nn.functional.interpolate(x, size=size, mode="nearest")

        ops_9 = [
            {"op_name": "Constant"},
            {
                "op_name": "Upsample",
                "attributes": [{"name": "mode", "s": (b"nearest"), "type": 3}],
            },
        ]
        ops_10 = [
            {"op_name": "Constant"},
            {
                "op_name": "Resize",
                "attributes": [{"name": "mode", "s": (b"nearest"), "type": 3}],
            },
        ]
        ops = {9: ops_9, 10: ops_10}
        x = torch.randn(20, 16, 50)
        check_onnx_opsets_operator(MyDynamicModel(), x, ops, opset_versions=[9, 10])

    def test_affine_grid(self):
        # [FUNCTION DOC] Checks affine_grid export at opset 20 for 2-D and 3-D inputs and both
        # align_corners settings, in training and eval modes. Expects Constant/Unsqueeze pairs, a
        # Concat for the size and a final AffineGrid node.
        class MyModule(Module):
            # [CLASS DOC] HELPER: Module applying affine_grid with a stored align_corners flag.
            def __init__(self, align_corners):
                # [FUNCTION DOC] HELPER: Store the align_corners flag on the module.
                super().__init__()
                self.align_corners = align_corners

            def forward(self, theta, size):
                # [FUNCTION DOC] HELPER: Return the affine sampling grid for theta and size.
                return torch.nn.functional.affine_grid(
                    theta, size, align_corners=self.align_corners
                )

        opset_version = 20
        ops_2d = {
            opset_version: [
                {"op_name": "Constant"},
                {"op_name": "Unsqueeze"},
                {"op_name": "Constant"},
                {"op_name": "Unsqueeze"},
                {"op_name": "Constant"},
                {"op_name": "Unsqueeze"},
                {"op_name": "Constant"},
                {"op_name": "Unsqueeze"},
                {"op_name": "Concat"},
                {"op_name": "AffineGrid"},
            ]
        }

        ops_3d = {
            opset_version: [
                {"op_name": "Constant"},
                {"op_name": "Unsqueeze"},
                {"op_name": "Constant"},
                {"op_name": "Unsqueeze"},
                {"op_name": "Constant"},
                {"op_name": "Unsqueeze"},
                {"op_name": "Constant"},
                {"op_name": "Unsqueeze"},
                {"op_name": "Constant"},
                {"op_name": "Unsqueeze"},
                {"op_name": "Concat"},
                {"op_name": "AffineGrid"},
            ]
        }
        # 2D affine
        theta_2d = torch.empty(1, 2, 3, dtype=torch.double)
        size_2d = torch.Size([1, 1, 2, 2])
        # 3D affine
        theta_3d = torch.empty(1, 3, 4, dtype=torch.double)
        size_3d = torch.Size([1, 1, 2, 2, 2])

        for inputs, align_corners in itertools.product(
            ((theta_2d, size_2d, ops_2d), (theta_3d, size_3d, ops_3d)),
            (True, False),
        ):
            theta, size, ops = inputs
            args = (
                theta,
                size,
            )
            check_onnx_opsets_operator(
                MyModule(align_corners=align_corners),
                args,
                ops,
                opset_versions=[opset_version],
                training=torch.onnx.TrainingMode.TRAINING,
            )
            check_onnx_opsets_operator(
                MyModule(align_corners=align_corners),
                args,
                ops,
                opset_versions=[opset_version],
                training=torch.onnx.TrainingMode.EVAL,
            )

    def test_grid_sample(self):
        # [FUNCTION DOC] Checks grid_sample export for all combinations of mode, padding mode,
        # align_corners and opset 16 or 20: each must produce a single GridSample node in train and
        # eval. The 4-D case is always run; the 5-D case only for opset 20 and non-bicubic modes.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying grid_sample with stored mode, padding mode, flag.
            def __init__(self, mode, padding_mode, align_corners):
                # [FUNCTION DOC] HELPER: Store mode, padding_mode and align_corners on the module.
                super().__init__()
                self.mode = mode
                self.padding_mode = padding_mode
                self.align_corners = align_corners

            def forward(self, x, grid):
                # [FUNCTION DOC] HELPER: Return x sampled at the locations given by grid using
                # stored options.
                return torch.nn.functional.grid_sample(
                    x,
                    grid,
                    mode=self.mode,
                    padding_mode=self.padding_mode,
                    align_corners=self.align_corners,
                )

        for mode, padding_mode, align_corners, opset_version in itertools.product(
            ("bilinear", "nearest", "bicubic"),
            ("zeros", "border", "reflection"),
            (True, False),
            (16, 20),
        ):

            def test_eval_and_training(
                ops, opset_version, mode, padding_mode, align_corners, x_shape, grid
            ):
                # [FUNCTION DOC] HELPER: Export a grid-sample module for the given options with
                # random inputs of the given shapes (x_shape for the input, grid for the grid),
                # once in training mode and once in eval mode, validating against ops for
                # the opset version.
                args = (
                    torch.randn(*x_shape),  # x
                    torch.randn(grid),  # grid,
                )
                check_onnx_opsets_operator(
                    MyModule(
                        mode=mode,
                        padding_mode=padding_mode,
                        align_corners=align_corners,
                    ),
                    args,
                    ops,
                    opset_versions=[opset_version],
                    training=torch.onnx.TrainingMode.TRAINING,
                )
                check_onnx_opsets_operator(
                    MyModule(
                        mode=mode,
                        padding_mode=padding_mode,
                        align_corners=align_corners,
                    ),
                    args,
                    ops,
                    opset_versions=[opset_version],
                    training=torch.onnx.TrainingMode.EVAL,
                )

            ops = {opset_version: [{"op_name": "GridSample"}]}
            # mode = convert_grid_sample_mode(mode) if opset_version == 20 else mode
            n, c, d_in, h_in, w_in, d_out, h_out, w_out = 1, 1, 2, 3, 2, 3, 2, 4
            test_eval_and_training(
                ops,
                opset_version,
                mode,
                padding_mode,
                align_corners,
                (n, c, h_in, w_in),
                (n, h_out, w_out, 2),
            )
            if opset_version == 20 and mode != "bicubic":
                test_eval_and_training(
                    ops,
                    opset_version,
                    mode,
                    padding_mode,
                    align_corners,
                    (n, c, d_in, h_in, w_in),
                    (n, d_out, h_out, w_out, 3),
                )

    def test_flatten(self):
        # [FUNCTION DOC] Checks flatten export for opset 9 and 10: a zero-dim input gives Constant
        # plus Reshape, a 1-D input gives a single Identity.
        class MyModule(Module):
            # [CLASS DOC] HELPER: Module flattening its input completely.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return x flattened into one dimension.
                return torch.flatten(x)

        module = MyModule()

        ops_0d = [{"op_name": "Constant"}, {"op_name": "Reshape"}]
        ops_1d = [{"op_name": "Identity"}]
        for shape in ([], [3]):
            x = torch.randn(shape)
            for opset_version in [9, 10]:
                ops = {opset_version: (ops_0d if len(shape) == 0 else ops_1d)}
                check_onnx_opsets_operator(
                    module, x, ops, opset_versions=[opset_version]
                )


if __name__ == "__main__":
    common_utils.run_tests()
