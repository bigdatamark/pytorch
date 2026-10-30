# Owner(s): ["module: onnx"]
# [MODULE DOC] Tests for torch.onnx.ops: schema checks of the symbolic custom ops via opcheck,
# behavior and exportability of the symbolic and multi-output symbolic ops (attributes,
# metadata, dtype codes, dynamic shapes, lifted constants), and the native opset 23 ops rotary
# embedding and attention (eager results, opcheck, decomposition into aten ops, and ONNX export
# with node inputs and attributes verified). Uses onnx_ir, the exporter testing helpers and
# torch.library.opcheck. Runs the common test runner when executed as a script.
"""Test torch.onnx.ops."""

from __future__ import annotations

import onnx_ir as ir
import onnx_ir.passes.common as common_passes

import torch
from torch.onnx._internal.exporter import _testing as onnx_testing
from torch.onnx.ops import _impl, _symbolic_impl
from torch.testing._internal import common_utils
from torch.utils._python_dispatch import TorchDispatchMode


class SchemaTest(common_utils.TestCase):
    # [CLASS DOC] Opcheck-based schema tests for the private symbolic custom op registrations in
    # the symbolic implementation module.
    def test_symbolic_has_correct_schema(self):
        # [FUNCTION DOC] Runs the library opcheck on the single-output symbolic op once with a
        # full set of attribute lists, metadata, domain and version, and once with empty inputs
        # and empty attribute lists. Passes if opcheck finds the registered schema, fake and
        # real implementations consistent.
        torch.library.opcheck(
            _symbolic_impl._symbolic,
            ([torch.tensor(1)], "CustomOp", 1),
            dict(
                shape=[
                    1,
                ],
                attr_keys=["key"],
                attr_types=["i"],
                attr_pos=[(0, 1)],
                attr_ints=[1],
                attr_floats=[1.0],
                attr_strs=["attr"],
                metadata_props_keys=["meta_key"],
                metadata_props_values=["meta_value"],
                domain="custom_domain",
                version=42,
            ),
        )

        # Empty inputs
        torch.library.opcheck(
            _symbolic_impl._symbolic,
            ([], "CustomOp", 1),
            dict(
                shape=[
                    1,
                ],
                attr_keys=[],
                attr_types=[],
                attr_pos=[],
                attr_ints=[],
                attr_floats=[],
                attr_strs=[],
                metadata_props_keys=[],
                metadata_props_values=[],
            ),
        )

    def test_symbolic_multi_out_has_correct_schema(self):
        # [FUNCTION DOC] Runs the library opcheck on the multi-output symbolic op with three
        # output dtypes and shapes plus attributes and metadata, and again with empty inputs,
        # outputs and attribute lists. Passes if the schema and implementations are consistent.
        torch.library.opcheck(
            _symbolic_impl._symbolic_multi_out,
            ([torch.tensor(1)], "CustomMultiOutOp", [1, 2, 10]),
            dict(
                shapes=[[1, 2], [42], []],
                attr_keys=["key"],
                attr_types=["i"],
                attr_pos=[(0, 1)],
                attr_ints=[1],
                attr_floats=[1.0],
                attr_strs=["attr"],
                metadata_props_keys=["meta_key"],
                metadata_props_values=["meta_value"],
                domain="",
                version=1,
            ),
        )

        # Empty inputs
        torch.library.opcheck(
            _symbolic_impl._symbolic_multi_out,
            ([], "CustomMultiOutOp", []),
            dict(
                shapes=[],
                attr_keys=[],
                attr_types=[],
                attr_pos=[],
                attr_ints=[],
                attr_floats=[],
                attr_strs=[],
                metadata_props_keys=[],
                metadata_props_values=[],
            ),
        )


class SymbolicOpsTest(common_utils.TestCase):
    # [CLASS DOC] Tests of the public torch.onnx.ops.symbolic and symbolic_multi_out functions:
    # eager results, dtype handling, export to an ONNX graph, and argument validation.
    def test_symbolic_accepts_valid_inputs(self):
        # [FUNCTION DOC] Calls the symbolic op with attributes of every supported kind (int,
        # float, string, bool and lists of each), a float32 dtype, shape of three dimensions,
        # version and metadata, and asserts the returned placeholder tensor has that shape,
        # dtype float32 and the CPU device.
        output = torch.onnx.ops.symbolic(
            "custom_domain::CustomOp",
            (torch.tensor(1),),
            dict(
                int_key=1,
                float_key=1.0,
                str_key="attr",
                bool_key=True,
                list_int_key=[1, 2],
                list_float_key=[1.0, 2.0],
                list_str_key=["attr1", "attr2"],
                list_bool_key=[True, False],
            ),
            dtype=torch.float32,
            shape=[1, 2, 3],
            version=1,
            metadata_props={"meta_key": "meta_value"},
        )
        self.assertEqual(output.shape, torch.Size([1, 2, 3]))
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(output.device, torch.device("cpu"))

    def test_symbolic_accepts_valid_inputs_empty_shape(self):
        # [FUNCTION DOC] Calls the symbolic op with an empty shape and asserts the result is a
        # zero-dimensional tensor.
        output = torch.onnx.ops.symbolic(
            "custom_domain::CustomOp",
            (torch.tensor(1),),
            dtype=torch.float32,
            shape=[],
        )
        self.assertEqual(output.shape, torch.Size([]))

    def test_symbolic_accepts_valid_inputs_integer_types(self):
        # [FUNCTION DOC] Passes an integer ONNX dtype code (1, float) instead of a torch dtype
        # and asserts the result has torch dtype float32.
        output = torch.onnx.ops.symbolic(
            "custom_domain::CustomOp",
            (torch.tensor(1),),
            dtype=1,  # 1 is float32 in ONNX
            shape=[42],
        )
        self.assertEqual(output.dtype, torch.float32)

    def test_symbolic_accepts_valid_inputs_int4_type(self):
        # [FUNCTION DOC] Passes the ONNX dtype code for INT4 and asserts the result uses torch
        # uint8, the stand-in storage type for four-bit integers.
        output = torch.onnx.ops.symbolic(
            "custom_domain::CustomOp",
            (torch.tensor(1),),
            dtype=22,  # 22 is INT4 in ONNX
            shape=[42],
        )
        # We use torch uint8 for int4
        self.assertEqual(output.dtype, torch.uint8)

    def test_symbolic_is_exportable(self):
        # [FUNCTION DOC] Exports a model that calls the symbolic op with an extra None input,
        # all attribute kinds, version and metadata, with the dynamo exporter. Asserts the first
        # graph node has the custom op type and domain, the attributes are converted to the
        # matching ONNX IR attribute kinds (bools as 64-bit ints), metadata is kept, and the
        # output has shape 1, 2, 3 with int64 dtype. Raises an assertion error if export returns
        # nothing.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward emits one symbolic custom op node with a
            # None input and attributes.
            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Calls the symbolic op on the input and a None
                # placeholder with all attribute kinds, the input dtype, fixed shape 1, 2, 3,
                # version 1 and a metadata entry, and returns the result.
                return torch.onnx.ops.symbolic(
                    "custom_domain::CustomOp",
                    (x, None),
                    dict(
                        int_key=1,
                        float_key=1.0,
                        str_key="attr",
                        bool_key=True,
                        list_int_key=[1, 2],
                        list_float_key=[1.0, 2.0],
                        list_str_key=["attr1", "attr2"],
                        list_bool_key=[True, False],
                    ),
                    dtype=x.dtype,
                    shape=[1, 2, 3],
                    version=1,
                    metadata_props={"meta_key": "meta_value"},
                )

        onnx_program = torch.onnx.export(
            Model(), (torch.tensor(1),), dynamo=True, verbose=False
        )
        if onnx_program is None:
            raise AssertionError("onnx_program is None")
        node = onnx_program.model.graph.node(0)
        self.assertEqual(node.op_type, "CustomOp")
        self.assertEqual(node.domain, "custom_domain")
        attributes = node.attributes
        self.assertEqual(
            attributes,
            dict(
                int_key=ir.AttrInt64("int_key", 1),
                float_key=ir.AttrFloat32("float_key", 1.0),
                str_key=ir.AttrString("str_key", "attr"),
                bool_key=ir.AttrInt64("bool_key", 1),
                list_int_key=ir.AttrInt64s("list_int_key", [1, 2]),
                list_float_key=ir.AttrFloat32s("list_float_key", [1.0, 2.0]),
                list_str_key=ir.AttrStrings("list_str_key", ["attr1", "attr2"]),
                list_bool_key=ir.AttrInt64s("list_bool_key", [1, 0]),
            ),
        )
        self.assertEqual(node.metadata_props["meta_key"], "meta_value")
        outputs = node.outputs
        self.assertEqual(list(outputs[0].shape), [1, 2, 3])
        self.assertEqual(outputs[0].dtype, ir.DataType.INT64)

    def test_symbolic_export_with_dispatch_mode_lifted_constants(self):
        # [FUNCTION DOC] Exports a model under a tensor dispatch mode that wraps float32 tensor
        # arguments of every op in quantize/dequantize pairs built from buffers, then asserts
        # the exported program has at least two lifted tensor constants and that they appear
        # both in the program constants and in the ONNX graph initializers. Checks lifted
        # constants are preserved when constants are created inside a dispatch mode. Raises an
        # assertion error if export returns nothing.
        def qdq(x, scale, zero_point):
            # [FUNCTION DOC] HELPER: Emits symbolic QuantizeLinear then DequantizeLinear nodes
            # with empty attributes, output dtypes taken from the zero point and scale, and the
            # input shape; returns the dequantized value.
            q = torch.onnx.ops.symbolic(
                "QuantizeLinear",
                (x, scale, zero_point),
                attrs={},
                dtype=zero_point.dtype,
                shape=x.shape,
            )
            return torch.onnx.ops.symbolic(
                "DequantizeLinear",
                (q, scale, zero_point),
                attrs={},
                dtype=scale.dtype,
                shape=x.shape,
            )

        class Quantizer(torch.nn.Module):
            # [CLASS DOC] HELPER: Module owning a float scale buffer 0.1 and an unsigned 8-bit
            # zero point buffer 128 and applying the quantize/dequantize helper.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Registers the scale and zero point buffers.
                super().__init__()
                self.register_buffer("scale", torch.tensor(0.1, dtype=torch.float32))
                self.register_buffer("zero_point", torch.tensor(128, dtype=torch.uint8))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the quantize/dequantize helper applied to the
                # input with the stored scale and zero point.
                return qdq(x, self.scale, self.zero_point)

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model that holds the quantizer module; its forward doubles the
            # input and applies relu.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the quantizer submodule.
                super().__init__()
                self.quantizer = Quantizer()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns relu of the input plus itself.
                return torch.relu(x + x)

        class QuantMode(TorchDispatchMode):
            # [CLASS DOC] HELPER: Dispatch mode that routes each float32 tensor argument
            # (positional or keyword) of every dispatched op through the model quantizer before
            # running the op. Holds the model.
            def __init__(self, model):
                # [FUNCTION DOC] HELPER: Initializes the base mode and stores the model.
                super().__init__()
                self.model = model

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Rebuilds the positional argument list and keyword
                # mapping, replacing float32 tensors with their quantized-dequantized version,
                # then calls the original function with them and returns the result.
                args = [
                    self.model.quantizer(arg)
                    if isinstance(arg, torch.Tensor) and arg.dtype == torch.float32
                    else arg
                    for arg in args
                ]
                kwargs = {
                    key: self.model.quantizer(value)
                    if isinstance(value, torch.Tensor) and value.dtype == torch.float32
                    else value
                    for key, value in (kwargs or {}).items()
                }
                return func(*args, **kwargs)

        model = Model().eval()
        with QuantMode(model), torch.no_grad():
            onnx_program = torch.onnx.export(
                model, (torch.randn(2, 3),), dynamo=True, verbose=False
            )

        if onnx_program is None:
            raise AssertionError("onnx_program is None")
        lifted_constants = set(
            onnx_program.exported_program.graph_signature.lifted_tensor_constants
        )
        self.assertGreaterEqual(len(lifted_constants), 2)
        self.assertLessEqual(
            lifted_constants,
            set(onnx_program.exported_program.constants),
        )
        self.assertLessEqual(
            lifted_constants,
            set(onnx_program.model.graph.initializers.keys()),
        )

    def test_symbolic_preserves_dynamic_shapes(self):
        # [FUNCTION DOC] Exports a model whose symbolic op output shape is the concatenation of
        # the two input shapes, using named dynamic dimensions for the first dimension of the
        # first input and the second dimension of the second input. Asserts the graph inputs and
        # the node output keep those symbolic dimension names and the static sizes, and the
        # output dtype is float.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model whose forward emits one symbolic op with output shape
            # made of both input shapes.
            def forward(self, x: torch.Tensor, y: torch.Tensor):
                # [FUNCTION DOC] HELPER: Returns the symbolic op applied to both inputs with the
                # first input dtype, concatenated shape and version 1.
                return torch.onnx.ops.symbolic(
                    "custom_domain::CustomOp",
                    (x, y),
                    dtype=x.dtype,
                    shape=[*x.shape, *y.shape],
                    version=1,
                )

        onnx_program = torch.onnx.export(
            Model(),
            (torch.zeros(2, 3), torch.zeros(1, 2)),
            dynamic_shapes=({0: "batch"}, {1: "something_else"}),
            dynamo=True,
            verbose=False,
        )
        if onnx_program is None:
            raise AssertionError("onnx_program is None")
        node = onnx_program.model.graph.node(0)
        self.assertEqual(node.op_type, "CustomOp")
        self.assertEqual(node.domain, "custom_domain")
        inputs = onnx_program.model.graph.inputs
        self.assertEqual(str(inputs[0].shape[0]), "batch")
        self.assertEqual(inputs[0].shape[1], 3)
        self.assertEqual(inputs[1].shape[0], 1)
        self.assertEqual(str(inputs[1].shape[1]), "something_else")
        outputs = node.outputs
        self.assertEqual(str(outputs[0].shape[0]), "batch")
        self.assertEqual(outputs[0].shape[1], 3)
        self.assertEqual(outputs[0].shape[2], 1)
        self.assertEqual(str(outputs[0].shape[3]), "something_else")
        self.assertEqual(outputs[0].dtype, ir.DataType.FLOAT)

    def test_symbolic_multi_out_accepts_valid_inputs(self):
        # [FUNCTION DOC] Calls the multi-output symbolic op with all attribute kinds, three
        # dtypes (an ONNX integer code, int32 and float8) and shapes, version and metadata, and
        # asserts three CPU tensors come back with the requested shapes and dtypes.
        outputs = torch.onnx.ops.symbolic_multi_out(
            "custom_domain::CustomMultiOutOp",
            (torch.tensor(1),),
            dict(
                int_key=1,
                float_key=1.0,
                str_key="attr",
                bool_key=True,
                list_int_key=[1, 2],
                list_float_key=[1.0, 2.0],
                list_str_key=["attr1", "attr2"],
                list_bool_key=[True, False],
            ),
            dtypes=(
                1,  # 1 is float32 in ONNX
                torch.int32,
                torch.float8_e4m3fn,
            ),
            shapes=([1, 2], [42], []),
            version=1,
            metadata_props={"meta_key": "meta_value"},
        )
        self.assertEqual(len(outputs), 3)
        self.assertEqual(outputs[0].shape, torch.Size([1, 2]))
        self.assertEqual(outputs[0].dtype, torch.float32)
        self.assertEqual(outputs[1].shape, torch.Size([42]))
        self.assertEqual(outputs[1].dtype, torch.int32)
        self.assertEqual(outputs[2].shape, torch.Size([]))
        self.assertEqual(outputs[2].dtype, torch.float8_e4m3fn)
        self.assertEqual(outputs[0].device, torch.device("cpu"))
        self.assertEqual(outputs[1].device, torch.device("cpu"))
        self.assertEqual(outputs[2].device, torch.device("cpu"))

    def test_symbolic_multi_out_accepts_valid_inputs_empty_shape(self):
        # [FUNCTION DOC] Calls the multi-output symbolic op with one scalar output (empty shape)
        # and asserts the output shape is empty.
        outputs = torch.onnx.ops.symbolic_multi_out(
            "custom_domain::CustomOp",
            (torch.tensor(1),),
            dtypes=(torch.float32,),
            shapes=[[]],
        )
        self.assertEqual(outputs[0].shape, torch.Size([]))

    def test_symbolic_multi_out_accepts_valid_inputs_integer_types(self):
        # [FUNCTION DOC] Uses an integer ONNX dtype code in the dtypes list and asserts the
        # result is float32.
        outputs = torch.onnx.ops.symbolic_multi_out(
            "custom_domain::CustomOp",
            (torch.tensor(1),),
            dtypes=(1,),  # 1 is float32 in ONNX
            shapes=[[42]],
        )
        self.assertEqual(outputs[0].dtype, torch.float32)

    def test_symbolic_multi_out_accepts_valid_inputs_int4_type(self):
        # [FUNCTION DOC] Uses the ONNX INT4 dtype code and asserts the result uses torch uint8.
        outputs = torch.onnx.ops.symbolic_multi_out(
            "custom_domain::CustomOp",
            (torch.tensor(1),),
            dtypes=(22,),  # 22 is INT4 in ONNX
            shapes=[[42]],
        )
        # We use torch uint8 for int4
        self.assertEqual(outputs[0].dtype, torch.uint8)

    def test_symbolic_multi_out_is_exportable(self):
        # [FUNCTION DOC] Exports a model that calls the multi-output symbolic op with all
        # attribute kinds, three output dtypes and shapes and metadata. Asserts the first node
        # type, domain, converted attributes and metadata, and that each of three outputs has
        # the expected shape and ONNX dtype (float, int32, float8). Raises an assertion error if
        # export returns nothing.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model whose forward emits one multi-output symbolic op node.
            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Returns the three outputs of the multi-output symbolic
                # op applied to the input and a None placeholder with the given attributes.
                return torch.onnx.ops.symbolic_multi_out(
                    "custom_domain::CustomOp",
                    (x, None),
                    dict(
                        int_key=1,
                        float_key=1.0,
                        str_key="attr",
                        bool_key=True,
                        list_int_key=[1, 2],
                        list_float_key=[1.0, 2.0],
                        list_str_key=["attr1", "attr2"],
                        list_bool_key=[True, False],
                    ),
                    dtypes=(torch.float32, torch.int32, torch.float8_e4m3fn),
                    shapes=([1, 2], [42], []),
                    version=1,
                    metadata_props={"meta_key": "meta_value"},
                )

        onnx_program = torch.onnx.export(
            Model(), (torch.tensor(1),), dynamo=True, verbose=False
        )
        if onnx_program is None:
            raise AssertionError("onnx_program is None")
        node = onnx_program.model.graph.node(0)
        self.assertEqual(node.op_type, "CustomOp")
        self.assertEqual(node.domain, "custom_domain")
        attributes = node.attributes
        self.assertEqual(
            attributes,
            dict(
                int_key=ir.AttrInt64("int_key", 1),
                float_key=ir.AttrFloat32("float_key", 1.0),
                str_key=ir.AttrString("str_key", "attr"),
                bool_key=ir.AttrInt64("bool_key", 1),
                list_int_key=ir.AttrInt64s("list_int_key", [1, 2]),
                list_float_key=ir.AttrFloat32s("list_float_key", [1.0, 2.0]),
                list_str_key=ir.AttrStrings("list_str_key", ["attr1", "attr2"]),
                list_bool_key=ir.AttrInt64s("list_bool_key", [1, 0]),
            ),
        )
        self.assertEqual(node.metadata_props["meta_key"], "meta_value")
        outputs = node.outputs
        self.assertEqual(list(outputs[0].shape), [1, 2])
        self.assertEqual(outputs[0].dtype, ir.DataType.FLOAT)
        self.assertEqual(list(outputs[1].shape), [42])
        self.assertEqual(outputs[1].dtype, ir.DataType.INT32)
        self.assertEqual(list(outputs[2].shape), [])
        self.assertEqual(outputs[2].dtype, ir.DataType.FLOAT8E4M3FN)

    def test_symbolic_multi_out_preserves_dynamic_shapes(self):
        # [FUNCTION DOC] Exports a multi-output symbolic op whose first output shape
        # concatenates the two input shapes and second output is fixed size 42 with INT4 dtype,
        # using named dynamic dimensions on the inputs. Asserts the symbolic dimension names are
        # preserved on inputs and the first output, and the second output has shape 42 with INT4
        # dtype.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model whose forward emits one multi-output symbolic op with a
            # dynamic-shaped first output.
            def forward(self, x: torch.Tensor, y: torch.Tensor):
                # [FUNCTION DOC] HELPER: Returns the symbolic op applied to both inputs with
                # dtypes (input dtype and INT4 code), shapes (both input shapes concatenated,
                # and 42), and version 1.
                return torch.onnx.ops.symbolic_multi_out(
                    "custom_domain::CustomOp",
                    (x, y),
                    dtypes=(x.dtype, 22),  # 22 is INT4
                    shapes=[[*x.shape, *y.shape], [42]],
                    version=1,
                )

        onnx_program = torch.onnx.export(
            Model(),
            (torch.zeros(2, 3), torch.zeros(1, 2)),
            dynamic_shapes=({0: "batch"}, {1: "something_else"}),
            dynamo=True,
            verbose=False,
        )
        if onnx_program is None:
            raise AssertionError("onnx_program is None")
        node = onnx_program.model.graph.node(0)
        self.assertEqual(node.op_type, "CustomOp")
        self.assertEqual(node.domain, "custom_domain")
        inputs = onnx_program.model.graph.inputs
        self.assertEqual(str(inputs[0].shape[0]), "batch")
        self.assertEqual(inputs[0].shape[1], 3)
        self.assertEqual(inputs[1].shape[0], 1)
        self.assertEqual(str(inputs[1].shape[1]), "something_else")
        outputs = node.outputs
        self.assertEqual(str(outputs[0].shape[0]), "batch")
        self.assertEqual(outputs[0].shape[1], 3)
        self.assertEqual(outputs[0].shape[2], 1)
        self.assertEqual(str(outputs[0].shape[3]), "something_else")
        self.assertEqual(outputs[0].dtype, ir.DataType.FLOAT)
        self.assertEqual(list(outputs[1].shape), [42])
        self.assertEqual(outputs[1].dtype, ir.DataType.INT4)

    def test_symbolic_multi_out_raises_when_dtypes_and_shapes_differ(self):
        # [FUNCTION DOC] Asserts that the multi-output symbolic op raises a runtime error when
        # the number of dtypes does not match the number of shapes, in both directions (two
        # dtypes with three shapes, one dtype with two shapes).
        with self.assertRaises(RuntimeError):
            torch.onnx.ops.symbolic_multi_out(
                "custom_domain::CustomMultiOutOp",
                (torch.tensor(1),),
                dict(
                    int_key=1,
                    float_key=1.0,
                    str_key="attr",
                    bool_key=True,
                    list_int_key=[1, 2],
                    list_float_key=[1.0, 2.0],
                    list_str_key=["attr1", "attr2"],
                    list_bool_key=[True, False],
                ),
                dtypes=(torch.float32, torch.int32),
                shapes=([1, 2], [42], []),
                version=1,
                metadata_props={"meta_key": "meta_value"},
            )

        with self.assertRaises(RuntimeError):
            torch.onnx.ops.symbolic_multi_out(
                "custom_domain::CustomMultiOutOp",
                (torch.tensor(1),),
                dict(
                    int_key=1,
                    float_key=1.0,
                    str_key="attr",
                    bool_key=True,
                    list_int_key=[1, 2],
                    list_float_key=[1.0, 2.0],
                    list_str_key=["attr1", "attr2"],
                    list_bool_key=[True, False],
                ),
                dtypes=(torch.float32,),
                shapes=([1, 2], [42]),
                version=1,
                metadata_props={"meta_key": "meta_value"},
            )


@common_utils.instantiate_parametrized_tests
class NativeOnnxOpsTest(common_utils.TestCase):
    # [CLASS DOC] Tests for the native opset 23 ONNX ops exposed in torch.onnx.ops (rotary
    # embedding and attention). The class decorator instantiates the parametrized tests.
    # Provides an export helper used by the export tests.
    def export(self, model, args=(), kwargs=None, **options) -> torch.onnx.ONNXProgram:
        # [FUNCTION DOC] HELPER: Exports the model with the dynamo exporter (quiet) passing
        # extra keyword options such as dynamic shapes and opset version, raises an assertion
        # error if no program is returned, runs the IR checker pass on the model, and returns
        # the ONNX program.
        onnx_program = torch.onnx.export(
            model,
            args,
            kwargs=kwargs,
            dynamo=True,
            verbose=False,
            **options,
        )
        if onnx_program is None:
            raise AssertionError("onnx_program is None")
        common_passes.CheckerPass()(onnx_program.model)
        return onnx_program

    def test_onnx_ops_can_be_decomposed_to_aten(self):
        # [FUNCTION DOC] Creates random inputs and exports with torch.export a model calling
        # rotary embedding in interleaved mode. Asserts the exported graph contains the ONNX op
        # target, that running decompositions with the ONNX aten decomposition table removes it,
        # and that the decomposed module gives close results to the original model.
        input_data = torch.rand(2, 3, 4, 8)
        position_ids_data = torch.randint(0, 50, (2, 4)).long()
        sin_cache_data = torch.rand(50, 4)
        cos_cache_data = torch.rand(50, 4)

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model whose forward calls the rotary embedding op in
            # interleaved mode.
            def forward(
                self, input_data, cos_cache_data, sin_cache_data, position_ids_data
            ):
                # [FUNCTION DOC] HELPER: Returns rotary embedding of the input using the cos and
                # sin caches and position ids with interleaving on.
                return torch.onnx.ops.rotary_embedding(
                    input_data,
                    cos_cache_data,
                    sin_cache_data,
                    position_ids_data,
                    interleaved=True,
                )

        model = Model()

        ep = torch.export.export(
            model,
            (input_data, cos_cache_data, sin_cache_data, position_ids_data),
        )
        self.assertIn(
            "onnx.RotaryEmbedding.opset23",
            [str(node.target) for node in ep.graph.nodes],
        )
        # The program can be decomposed into aten ops so it is fully compatible with the PyTorch ecosystem
        aten_decomped = ep.run_decompositions(torch.onnx.ops.aten_decompositions())
        self.assertNotIn(
            "onnx.RotaryEmbedding.opset23",
            [str(node.target) for node in aten_decomped.graph.nodes],
        )
        torch.testing.assert_close(
            aten_decomped.module()(
                input_data, cos_cache_data, sin_cache_data, position_ids_data
            ),
            model(input_data, cos_cache_data, sin_cache_data, position_ids_data),
        )

    def test_rotary_embedding_opcheck(self):
        # [FUNCTION DOC] Runs library opcheck on the rotary embedding implementation with
        # four-dimensional input, caches of size 50 by 4, and position ids.
        input_data = torch.rand(2, 3, 4, 8)
        position_ids_data = torch.randint(0, 50, (2, 4)).long()
        sin_cache_data = torch.rand(50, 4)
        cos_cache_data = torch.rand(50, 4)

        torch.library.opcheck(
            _impl.rotary_embedding_23,
            (input_data, cos_cache_data, sin_cache_data, position_ids_data),
        )

    def test_rotary_embedding_opcheck_3d(self):
        # [FUNCTION DOC] Runs opcheck on the rotary embedding implementation with a
        # three-dimensional input and three heads, to verify the fake implementation output
        # stride matches the real contiguous output.
        # 3D input path must also match the real impl's contiguous output stride.
        input_data = torch.rand(2, 4, 24)
        position_ids_data = torch.randint(0, 50, (2, 4)).long()
        sin_cache_data = torch.rand(50, 4)
        cos_cache_data = torch.rand(50, 4)

        torch.library.opcheck(
            _impl.rotary_embedding_23,
            (input_data, cos_cache_data, sin_cache_data, position_ids_data),
            kwargs=dict(num_heads=3),
        )

    def test_rotary_embedding_opcheck_non_contiguous(self):
        # [FUNCTION DOC] Asserts the transposed input is non-contiguous, then runs opcheck on
        # the rotary embedding implementation to verify fake and real stride agreement for
        # non-contiguous inputs.
        # Non-contiguous inputs must not break fake/real stride alignment.
        # transpose(1, 2) on a (B, S, H, D) tensor yields the (B, H, S, D)
        # layout the op expects, with non-contiguous strides.
        input_data = torch.rand(2, 4, 3, 8).transpose(1, 2)
        self.assertFalse(input_data.is_contiguous())
        position_ids_data = torch.randint(0, 50, (2, 4)).long()
        sin_cache_data = torch.rand(50, 4)
        cos_cache_data = torch.rand(50, 4)

        torch.library.opcheck(
            _impl.rotary_embedding_23,
            (input_data, cos_cache_data, sin_cache_data, position_ids_data),
        )

    def test_rotary_embedding(self):
        # [FUNCTION DOC] Calls rotary embedding eagerly and checks the result shape equals the
        # input shape, then exports a model using it with dynamic first dimensions for the input
        # and position ids at opset 23. Asserts opset 23 import, a RotaryEmbedding first node,
        # and numerical agreement via the exporter testing helper.
        input_data = torch.rand(2, 3, 4, 8)
        position_ids_data = torch.randint(0, 50, (2, 4)).long()
        sin_cache_data = torch.rand(50, 4)
        cos_cache_data = torch.rand(50, 4)

        # Eager mode is supported. Autograd is also supported so users can choose to use the op
        # in development and production
        result = torch.onnx.ops.rotary_embedding(
            input_data, cos_cache_data, sin_cache_data, position_ids_data
        )
        self.assertEqual(result.shape, input_data.shape)

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model whose forward calls the rotary embedding op in
            # interleaved mode.
            def forward(
                self, input_data, cos_cache_data, sin_cache_data, position_ids_data
            ):
                # [FUNCTION DOC] HELPER: Returns rotary embedding of the input using the cos and
                # sin caches and position ids with interleaving on.
                return torch.onnx.ops.rotary_embedding(
                    input_data,
                    cos_cache_data,
                    sin_cache_data,
                    position_ids_data,
                    interleaved=True,
                )

        model = Model()

        # Dynamic shapes are supported
        dynamic_shapes = {
            "input_data": {0: torch.export.Dim.DYNAMIC},
            "cos_cache_data": None,
            "sin_cache_data": None,
            "position_ids_data": {0: torch.export.Dim.DYNAMIC},
        }

        onnx_program = self.export(
            model,
            (input_data, cos_cache_data, sin_cache_data, position_ids_data),
            dynamic_shapes=dynamic_shapes,
            opset_version=23,
        )
        self.assertEqual(onnx_program.model.opset_imports[""], 23)
        self.assertEqual("RotaryEmbedding", onnx_program.model.graph.node(0).op_type)
        onnx_testing.assert_onnx_program(onnx_program)

    def test_rotary_embedding_3d(self):
        # [FUNCTION DOC] Exports rotary embedding with three-dimensional input and caches and
        # two heads, with dynamic first dimensions, at opset 23. Asserts opset 23, a
        # RotaryEmbedding first node, and numerical agreement via the exporter testing helper.
        num_heads = 2
        input_data = torch.rand(2, 3, 8)
        sin_cache_data = torch.rand(2, 3, 2)
        cos_cache_data = torch.rand(2, 3, 2)

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model whose forward calls the rotary embedding op with a head
            # count and no position ids.
            def forward(self, input_data, cos_cache_data, sin_cache_data):
                # [FUNCTION DOC] HELPER: Returns rotary embedding of the input with the cos and
                # sin caches and the captured number of heads.
                return torch.onnx.ops.rotary_embedding(
                    input_data,
                    cos_cache_data,
                    sin_cache_data,
                    num_heads=num_heads,
                )

        model = Model()

        # Dynamic shapes are supported
        dynamic_shapes = {
            "input_data": {0: torch.export.Dim.DYNAMIC},
            "cos_cache_data": {0: torch.export.Dim.DYNAMIC},
            "sin_cache_data": {0: torch.export.Dim.DYNAMIC},
        }

        onnx_program = self.export(
            model,
            (input_data, cos_cache_data, sin_cache_data),
            dynamic_shapes=dynamic_shapes,
            opset_version=23,
        )
        self.assertEqual(onnx_program.model.opset_imports[""], 23)
        self.assertEqual("RotaryEmbedding", onnx_program.model.graph.node(0).op_type)
        onnx_testing.assert_onnx_program(onnx_program)

    def test_attention_without_past_kv_caches(self):
        # [FUNCTION DOC] Builds random query, key and value tensors with batch 2, 8 query heads,
        # query length 4, key length 6 and head size 64, runs opcheck on the attention
        # implementation, then calls the high-level attention and asserts the output, present
        # key, present value and qk output shapes (the present key and value equal the key and
        # value shapes when no past cache is given).
        """Test basic attention functionality."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        # Test eager mode
        torch.library.opcheck(_impl.attention_23, (Q, K, V))
        output, present_key, present_value, qk_output = torch.onnx.ops.attention(
            Q, K, V
        )

        self.assertEqual(output.shape, (batch_size, q_num_heads, q_seq_len, head_size))
        self.assertEqual(present_key.shape, K.shape)
        self.assertEqual(present_value.shape, V.shape)
        self.assertEqual(
            qk_output.shape, (batch_size, q_num_heads, q_seq_len, kv_seq_len)
        )

    def test_attention_3d_inputs(self):
        # [FUNCTION DOC] Same as the basic test but with three-dimensional query, key and value
        # (batch, sequence, heads times head size) and explicit head counts. Runs opcheck and
        # asserts the output is reshaped to three dimensions and present key and value are
        # four-dimensional.
        """Test attention with 3D inputs (requires num_heads parameters)."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_seq_len, q_num_heads * head_size)
        K = torch.rand(batch_size, kv_seq_len, kv_num_heads * head_size)
        V = torch.rand(batch_size, kv_seq_len, kv_num_heads * head_size)

        torch.library.opcheck(
            _impl.attention_23,
            (Q, K, V),
            dict(q_num_heads=q_num_heads, kv_num_heads=kv_num_heads),
        )
        output, present_key, present_value, qk_output = torch.onnx.ops.attention(
            Q, K, V, q_num_heads=q_num_heads, kv_num_heads=kv_num_heads
        )

        # Output should be reshaped back to 3D
        self.assertEqual(output.shape, (batch_size, q_seq_len, q_num_heads * head_size))
        self.assertEqual(
            present_key.shape, (batch_size, kv_num_heads, kv_seq_len, head_size)
        )
        self.assertEqual(
            present_value.shape, (batch_size, kv_num_heads, kv_seq_len, head_size)
        )

    @common_utils.parametrize(
        "name, kv_num_heads",
        [
            ("group_query_attention", 4),
            ("multi_query_attention", 1),
        ],
    )
    def test_attention_kv_num_heads(self, name: str, kv_num_heads: int):
        # [FUNCTION DOC] Parametrized over grouped-query (4 key heads) and multi-query (1 key
        # head) cases; the name parameter only labels the case. Runs opcheck, calls attention,
        # asserts output and present shapes, and checks the output is close to the scaled dot
        # product attention reference with grouped heads enabled.
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads = 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        torch.library.opcheck(_impl.attention_23, (Q, K, V))
        output, present_key, present_value, qk_output = torch.onnx.ops.attention(
            Q, K, V
        )
        expected = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, None, enable_gqa=True
        )

        self.assertEqual(output.shape, (batch_size, q_num_heads, q_seq_len, head_size))
        self.assertEqual(present_key.shape, K.shape)
        self.assertEqual(present_value.shape, V.shape)
        torch.testing.assert_close(output, expected)

    def test_attention_mqa(self):
        # [FUNCTION DOC] Multi-query attention with a single key/value head: runs opcheck, calls
        # attention, asserts the output shape and closeness to scaled dot product attention with
        # grouped heads enabled.
        """Test Multi-Query Attention (MQA)."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 1  # MQA: kv_num_heads = 1
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        torch.library.opcheck(_impl.attention_23, (Q, K, V))
        output, present_key, present_value, qk_output = torch.onnx.ops.attention(
            Q, K, V
        )
        expected = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, None, enable_gqa=True
        )

        self.assertEqual(output.shape, (batch_size, q_num_heads, q_seq_len, head_size))
        torch.testing.assert_close(output, expected)

    def test_attention_with_2d_mask(self):
        # [FUNCTION DOC] Runs opcheck and attention with a two-dimensional boolean mask and with
        # a two-dimensional float mask, and asserts both outputs have the expected shape.
        """Test attention with 2D attention mask (q_seq_len, kv_seq_len)."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        # Test with boolean mask
        bool_mask = torch.randint(0, 2, (q_seq_len, kv_seq_len), dtype=torch.bool)
        torch.library.opcheck(_impl.attention_23, (Q, K, V), dict(attn_mask=bool_mask))
        output_bool, _, _, _ = torch.onnx.ops.attention(Q, K, V, attn_mask=bool_mask)

        # Test with float mask
        float_mask = torch.randn(q_seq_len, kv_seq_len)
        torch.library.opcheck(_impl.attention_23, (Q, K, V), dict(attn_mask=float_mask))
        output_float, _, _, _ = torch.onnx.ops.attention(Q, K, V, attn_mask=float_mask)

        self.assertEqual(
            output_bool.shape, (batch_size, q_num_heads, q_seq_len, head_size)
        )
        self.assertEqual(
            output_float.shape, (batch_size, q_num_heads, q_seq_len, head_size)
        )

    def test_attention_with_4d_mask(self):
        # [FUNCTION DOC] Runs opcheck and attention with a four-dimensional boolean mask and a
        # four-dimensional float mask (batch, heads, query, key lengths), and asserts both
        # outputs have the expected shape.
        """Test attention with 4D attention mask (batch_size, num_heads, q_seq_len, kv_seq_len)."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        # Test with boolean mask
        bool_mask = torch.randint(
            0, 2, (batch_size, q_num_heads, q_seq_len, kv_seq_len), dtype=torch.bool
        )
        torch.library.opcheck(_impl.attention_23, (Q, K, V), dict(attn_mask=bool_mask))
        output_bool, _, _, _ = torch.onnx.ops.attention(Q, K, V, attn_mask=bool_mask)

        # Test with float mask
        float_mask = torch.randn(batch_size, q_num_heads, q_seq_len, kv_seq_len)
        torch.library.opcheck(_impl.attention_23, (Q, K, V), dict(attn_mask=float_mask))
        output_float, _, _, _ = torch.onnx.ops.attention(Q, K, V, attn_mask=float_mask)

        self.assertEqual(
            output_bool.shape, (batch_size, q_num_heads, q_seq_len, head_size)
        )
        self.assertEqual(
            output_float.shape, (batch_size, q_num_heads, q_seq_len, head_size)
        )

    def test_attention_with_zero_float_mask(self):
        # [FUNCTION DOC] Runs opcheck and attention with an all-zero float mask and asserts the
        # output shape.
        """Test attention with zero float mask."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        zero_mask = torch.zeros(q_seq_len, kv_seq_len)
        torch.library.opcheck(_impl.attention_23, (Q, K, V), dict(attn_mask=zero_mask))
        output, _, _, _ = torch.onnx.ops.attention(Q, K, V, attn_mask=zero_mask)

        self.assertEqual(output.shape, (batch_size, q_num_heads, q_seq_len, head_size))

    def test_attention_with_causal_mask_pattern(self):
        # [FUNCTION DOC] With equal query and key length, builds a lower triangular boolean
        # mask, runs opcheck and attention with it and asserts the output shape.
        """Test attention with lower triangular causal mask pattern."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 4  # Square for causal
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        # Create a lower triangular causal mask
        causal_mask = torch.tril(torch.ones(q_seq_len, kv_seq_len, dtype=torch.bool))
        torch.library.opcheck(
            _impl.attention_23, (Q, K, V), dict(attn_mask=causal_mask)
        )
        output, _, _, _ = torch.onnx.ops.attention(Q, K, V, attn_mask=causal_mask)

        self.assertEqual(output.shape, (batch_size, q_num_heads, q_seq_len, head_size))

    def test_attention_with_gqa_and_mask(self):
        # [FUNCTION DOC] With 8 query heads and 4 key heads, runs opcheck and attention with a
        # two-dimensional boolean mask and a four-dimensional boolean mask sized by query heads,
        # and asserts both output shapes.
        """Test attention with GQA and different mask shapes."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 4  # GQA
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        # Test 2D mask with GQA
        mask_2d = torch.randint(0, 2, (q_seq_len, kv_seq_len), dtype=torch.bool)
        torch.library.opcheck(_impl.attention_23, (Q, K, V), dict(attn_mask=mask_2d))
        output_2d, _, _, _ = torch.onnx.ops.attention(Q, K, V, attn_mask=mask_2d)

        # Test 4D mask with GQA (note: using q_num_heads for mask heads)
        mask_4d = torch.randint(
            0, 2, (batch_size, q_num_heads, q_seq_len, kv_seq_len), dtype=torch.bool
        )
        torch.library.opcheck(_impl.attention_23, (Q, K, V), dict(attn_mask=mask_4d))
        output_4d, _, _, _ = torch.onnx.ops.attention(Q, K, V, attn_mask=mask_4d)

        self.assertEqual(
            output_2d.shape, (batch_size, q_num_heads, q_seq_len, head_size)
        )
        self.assertEqual(
            output_4d.shape, (batch_size, q_num_heads, q_seq_len, head_size)
        )

    def test_attention_causal(self):
        # [FUNCTION DOC] With equal query and key length, runs opcheck and attention in causal
        # mode and asserts the output shape.
        """Test causal attention."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 4  # Square for causal
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        torch.library.opcheck(_impl.attention_23, (Q, K, V), dict(is_causal=True))
        output, _, _, _ = torch.onnx.ops.attention(Q, K, V, is_causal=True)

        self.assertEqual(output.shape, (batch_size, q_num_heads, q_seq_len, head_size))

    def test_attention_with_past_kv(self):
        # [FUNCTION DOC] Runs opcheck and attention with past key and value caches of length 3,
        # and asserts present key and value have sequence length equal to the past length plus
        # current key length.
        """Test attention with past key/value caches."""
        batch_size, q_seq_len, kv_seq_len, past_seq_len = 2, 4, 6, 3
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        past_key = torch.rand(batch_size, kv_num_heads, past_seq_len, head_size)
        past_value = torch.rand(batch_size, kv_num_heads, past_seq_len, head_size)

        torch.library.opcheck(
            _impl.attention_23,
            (Q, K, V),
            dict(past_key=past_key, past_value=past_value),
        )
        output, present_key, present_value, _ = torch.onnx.ops.attention(
            Q, K, V, past_key=past_key, past_value=past_value
        )

        # Present key/value should include past + current
        expected_total_seq_len = past_seq_len + kv_seq_len
        self.assertEqual(
            present_key.shape,
            (batch_size, kv_num_heads, expected_total_seq_len, head_size),
        )
        self.assertEqual(
            present_value.shape,
            (batch_size, kv_num_heads, expected_total_seq_len, head_size),
        )

    def test_attention_with_softcap(self):
        # [FUNCTION DOC] Runs opcheck and attention with a softcap of 30 and asserts the output
        # shape.
        """Test attention with softcap."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        torch.library.opcheck(_impl.attention_23, (Q, K, V), dict(softcap=30.0))
        output, _, _, _ = torch.onnx.ops.attention(Q, K, V, softcap=30.0)

        self.assertEqual(output.shape, (batch_size, q_num_heads, q_seq_len, head_size))

    def test_attention_qk_output_modes(self):
        # [FUNCTION DOC] For each of the four QK matmul output modes, runs opcheck and
        # attention, and asserts the output shape and the qk output shape (batch, heads, query
        # length, key length).
        """Test different QK matmul output modes."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        for mode in [0, 1, 2, 3]:
            torch.library.opcheck(
                _impl.attention_23,
                (Q, K, V),
                dict(qk_matmul_output_mode=mode),
            )
            output, _, _, qk_output = torch.onnx.ops.attention(
                Q, K, V, qk_matmul_output_mode=mode
            )

            self.assertEqual(
                output.shape, (batch_size, q_num_heads, q_seq_len, head_size)
            )
            self.assertEqual(
                qk_output.shape, (batch_size, q_num_heads, q_seq_len, kv_seq_len)
            )

    def test_attention_custom_scale(self):
        # [FUNCTION DOC] Runs opcheck and attention with scale 0.25 and asserts the output
        # shape.
        """Test attention with custom scale factor."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        custom_scale = 0.25
        torch.library.opcheck(_impl.attention_23, (Q, K, V), dict(scale=custom_scale))
        output, _, _, _ = torch.onnx.ops.attention(Q, K, V, scale=custom_scale)

        self.assertEqual(output.shape, (batch_size, q_num_heads, q_seq_len, head_size))

    def test_attention_export(self):
        # [FUNCTION DOC] Exports a model returning only the attention output; exports it with
        # the dynamo exporter at opset 23 (the export helper also runs the ONNX checker).
        # Asserts opset 23 import, an Attention first node, and numerical agreement through the
        # exporter testing helper.
        """Test that attention can be exported to ONNX."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        class AttentionModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model whose forward returns only the attention output.
            def forward(self, Q, K, V):
                # [FUNCTION DOC] HELPER: Calls attention on the three inputs and returns the
                # first of its four results.
                output, _, _, _ = torch.onnx.ops.attention(Q, K, V)
                return output

        model = AttentionModel()

        onnx_program = self.export(
            model,
            (Q, K, V),
            opset_version=23,
        )

        self.assertEqual(onnx_program.model.opset_imports[""], 23)
        self.assertEqual("Attention", onnx_program.model.graph.node(0).op_type)
        onnx_testing.assert_onnx_program(onnx_program)

    def test_attention_export_with_dynamic_shapes(self):
        # [FUNCTION DOC] Exports attention with a boolean mask using named dynamic dimensions
        # for batch and sequence lengths. Asserts the node has four inputs whose shapes carry
        # the symbolic names with static head counts and head size, that no attributes are set,
        # and numerical agreement.
        """Test attention export with dynamic shapes."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        attn_mask = torch.randint(
            0, 2, (batch_size, 1, q_seq_len, kv_seq_len), dtype=torch.bool
        )

        class AttentionModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model whose forward passes a mask to attention and returns
            # only the output.
            def forward(self, Q, K, V, attn_mask):
                # [FUNCTION DOC] HELPER: Calls attention with the mask and returns the first
                # result.
                output, _, _, _ = torch.onnx.ops.attention(Q, K, V, attn_mask=attn_mask)
                return output

        model = AttentionModel()

        dynamic_shapes = {
            "Q": {0: "batch", 2: "q_seq_len"},
            "K": {0: "batch", 2: "kv_seq_len"},
            "V": {0: "batch", 2: "kv_seq_len"},
            "attn_mask": {0: "batch", 2: "q_seq_len", 3: "kv_seq_len"},
        }

        onnx_program = self.export(
            model,
            (Q, K, V, attn_mask),
            dynamic_shapes=dynamic_shapes,
            opset_version=23,
        )

        self.assertEqual(onnx_program.model.opset_imports[""], 23)
        self.assertEqual("Attention", onnx_program.model.graph.node(0).op_type)
        node = onnx_program.model.graph.node(0)
        # Verify inputs
        self.assertEqual(len(node.inputs), 4)
        self.assertEqual(
            node.inputs[0].shape, ["batch", q_num_heads, "q_seq_len", head_size]
        )
        self.assertEqual(
            node.inputs[1].shape, ["batch", kv_num_heads, "kv_seq_len", head_size]
        )
        self.assertEqual(
            node.inputs[2].shape, ["batch", kv_num_heads, "kv_seq_len", head_size]
        )

        # Verify default attributes (should be minimal)
        self.assertEqual(len(node.attributes), 0)
        onnx_testing.assert_onnx_program(onnx_program)

    def test_attention_3d_export(self):
        # [FUNCTION DOC] Exports attention with three-dimensional inputs and head counts as
        # attributes at opset 23, then asserts opset 23, an Attention first node, and numerical
        # agreement.
        """Test attention export with 3D inputs."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_seq_len, q_num_heads * head_size)
        K = torch.rand(batch_size, kv_seq_len, kv_num_heads * head_size)
        V = torch.rand(batch_size, kv_seq_len, kv_num_heads * head_size)

        class AttentionModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model whose forward calls attention with head counts on
            # three-dimensional inputs and returns the output.
            def forward(self, Q, K, V):
                # [FUNCTION DOC] HELPER: Calls attention with the captured query and key head
                # counts and returns the first result.
                output, _, _, _ = torch.onnx.ops.attention(
                    Q, K, V, q_num_heads=q_num_heads, kv_num_heads=kv_num_heads
                )
                return output

        model = AttentionModel()

        onnx_program = self.export(
            model,
            (Q, K, V),
            opset_version=23,
        )

        self.assertEqual(onnx_program.model.opset_imports[""], 23)
        self.assertEqual("Attention", onnx_program.model.graph.node(0).op_type)
        onnx_testing.assert_onnx_program(onnx_program)

    def test_attention_decomposition(self):
        # [FUNCTION DOC] Exports attention with torch.export, asserts the ONNX attention target
        # is in the graph, decomposes with the ONNX aten decomposition table, asserts the target
        # is gone, and asserts the decomposed module result is close to the model result.
        """Test that attention can be decomposed to aten ops."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        class AttentionModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model whose forward calls attention and returns only the
            # output.
            def forward(self, Q, K, V):
                # [FUNCTION DOC] HELPER: Calls attention, unpacks four results and returns the
                # first.
                output, present_key, present_value, qk_output = (
                    torch.onnx.ops.attention(Q, K, V)
                )
                return output

        model = AttentionModel()

        ep = torch.export.export(model, (Q, K, V))
        self.assertIn(
            "onnx.Attention.opset23",
            [str(node.target) for node in ep.graph.nodes],
        )

        # The program can be decomposed into aten ops
        aten_decomped = ep.run_decompositions(torch.onnx.ops.aten_decompositions())
        self.assertNotIn(
            "onnx.Attention.opset23",
            [str(node.target) for node in aten_decomped.graph.nodes],
        )

        # Results should match
        torch.testing.assert_close(
            aten_decomped.module()(Q, K, V),
            model(Q, K, V),
        )

    def test_attention_export_with_past_key_value(self):
        # [FUNCTION DOC] Exports attention given past key and past value (passed in switched
        # keyword order with an explicit None mask) and verifies the node has six inputs in the
        # ONNX order: query, key, value, absent mask, past key, past value, with the expected
        # shapes. Checks numerical agreement.
        """Test export with past_key, past_value to ensure the optional input order is correct."""
        batch_size, q_seq_len, kv_seq_len, past_seq_len = 2, 4, 6, 3
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        past_key = torch.rand(batch_size, kv_num_heads, past_seq_len, head_size)
        past_value = torch.rand(batch_size, kv_num_heads, past_seq_len, head_size)

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model whose forward calls attention with past caches and
            # returns the output and present key and value.
            def forward(self, Q, K, V, past_key, past_value):
                # [FUNCTION DOC] HELPER: Calls attention with keywords past_key, mask None and
                # past_value in swapped order and returns the output, present key and present
                # value.
                output, present_key, present_value, _ = torch.onnx.ops.attention(
                    Q,
                    K,
                    V,
                    past_key=past_key,
                    attn_mask=None,
                    # Switched argument order
                    past_value=past_value,
                )
                return output, present_key, present_value

        model = Model()
        onnx_program = self.export(
            model, (Q, K, V, past_key, past_value), opset_version=23
        )

        node = onnx_program.model.graph.node(0)
        self.assertEqual(node.op_type, "Attention")

        # Verify all 6 inputs are present
        self.assertEqual(
            len(node.inputs), 6
        )  # Q, K, V, attn_mask, past_key, past_value
        self.assertEqual(
            node.inputs[0].shape, [batch_size, q_num_heads, q_seq_len, head_size]
        )
        self.assertEqual(
            node.inputs[1].shape, [batch_size, kv_num_heads, kv_seq_len, head_size]
        )
        self.assertEqual(
            node.inputs[2].shape, [batch_size, kv_num_heads, kv_seq_len, head_size]
        )
        self.assertIsNone(node.inputs[3])
        self.assertEqual(
            node.inputs[4].shape, [batch_size, kv_num_heads, past_seq_len, head_size]
        )
        self.assertEqual(
            node.inputs[5].shape, [batch_size, kv_num_heads, past_seq_len, head_size]
        )
        onnx_testing.assert_onnx_program(onnx_program)

    def test_attention_export_with_all_optional_inputs(self):
        # [FUNCTION DOC] Exports attention with a boolean mask covering past plus current length
        # and both past caches; verifies six inputs with the expected shapes (mask of 1, 1,
        # query, past plus key length) and numerical agreement.
        """Test export with all optional inputs: mask, past_key, past_value."""
        batch_size, q_seq_len, kv_seq_len, past_seq_len = 2, 4, 6, 3
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        attn_mask = torch.randint(
            0, 2, (1, 1, q_seq_len, kv_seq_len + past_seq_len), dtype=torch.bool
        )
        past_key = torch.rand(batch_size, kv_num_heads, past_seq_len, head_size)
        past_value = torch.rand(batch_size, kv_num_heads, past_seq_len, head_size)

        class FullAttentionModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model whose forward calls attention with mask and both past
            # caches and returns all four results.
            def forward(self, Q, K, V, attn_mask, past_key, past_value):
                # [FUNCTION DOC] HELPER: Calls attention with mask, past key and past value, and
                # returns the output, present key, present value and qk matmul result.
                output, present_key, present_value, qk_matmul = (
                    torch.onnx.ops.attention(
                        Q,
                        K,
                        V,
                        attn_mask=attn_mask,
                        past_key=past_key,
                        past_value=past_value,
                    )
                )
                return output, present_key, present_value, qk_matmul

        model = FullAttentionModel()
        onnx_program = self.export(
            model, (Q, K, V, attn_mask, past_key, past_value), opset_version=23
        )

        node = onnx_program.model.graph.node(0)
        self.assertEqual(node.op_type, "Attention")

        # Verify all 6 inputs are present
        self.assertEqual(
            len(node.inputs), 6
        )  # Q, K, V, attn_mask, past_key, past_value
        self.assertEqual(
            node.inputs[0].shape, [batch_size, q_num_heads, q_seq_len, head_size]
        )
        self.assertEqual(
            node.inputs[1].shape, [batch_size, kv_num_heads, kv_seq_len, head_size]
        )
        self.assertEqual(
            node.inputs[2].shape, [batch_size, kv_num_heads, kv_seq_len, head_size]
        )
        self.assertEqual(
            node.inputs[3].shape, [1, 1, q_seq_len, kv_seq_len + past_seq_len]
        )
        self.assertEqual(
            node.inputs[4].shape, [batch_size, kv_num_heads, past_seq_len, head_size]
        )
        self.assertEqual(
            node.inputs[5].shape, [batch_size, kv_num_heads, past_seq_len, head_size]
        )
        onnx_testing.assert_onnx_program(onnx_program)

    def test_attention_export_3d_with_num_heads_attributes(self):
        # [FUNCTION DOC] Exports attention on three-dimensional inputs with 8 query heads and 4
        # key heads. Verifies the three-dimensional input shapes, that the head count attributes
        # exist with the given values, and numerical agreement.
        """Test export with 3D inputs and explicit num_heads attributes."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 4  # GQA
        head_size = 64

        Q = torch.rand(batch_size, q_seq_len, q_num_heads * head_size)
        K = torch.rand(batch_size, kv_seq_len, kv_num_heads * head_size)
        V = torch.rand(batch_size, kv_seq_len, kv_num_heads * head_size)

        class Attention3DModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model whose forward calls attention with head counts on
            # three-dimensional inputs.
            def forward(self, Q, K, V):
                # [FUNCTION DOC] HELPER: Calls attention with the captured head counts and
                # returns the first result.
                output, _, _, _ = torch.onnx.ops.attention(
                    Q, K, V, q_num_heads=q_num_heads, kv_num_heads=kv_num_heads
                )
                return output

        model = Attention3DModel()
        onnx_program = self.export(model, (Q, K, V), opset_version=23)

        node = onnx_program.model.graph.node(0)
        self.assertEqual(node.op_type, "Attention")

        # Verify 3D input shapes
        self.assertEqual(
            node.inputs[0].shape, [batch_size, q_seq_len, q_num_heads * head_size]
        )
        self.assertEqual(
            node.inputs[1].shape, [batch_size, kv_seq_len, kv_num_heads * head_size]
        )
        self.assertEqual(
            node.inputs[2].shape, [batch_size, kv_seq_len, kv_num_heads * head_size]
        )

        # Verify num_heads attributes are set
        attrs = node.attributes
        self.assertIn("q_num_heads", attrs)
        self.assertIn("kv_num_heads", attrs)
        self.assertEqual(attrs["q_num_heads"].value, q_num_heads)
        self.assertEqual(attrs["kv_num_heads"].value, kv_num_heads)
        onnx_testing.assert_onnx_program(onnx_program)

    def test_attention_export_with_all_attributes(self):
        # [FUNCTION DOC] Exports attention with causal mode, QK output mode 2, scale 0.25,
        # softcap 30 and softmax precision 1. Verifies all five attributes exist with those
        # values (causal as integer one) and numerical agreement.
        """Test export with all possible attributes set."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        class FullAttributesModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model whose forward calls attention with all optional
            # attributes set and returns the output.
            def forward(self, Q, K, V):
                # [FUNCTION DOC] HELPER: Calls attention with causal, qk output mode 2, scale
                # 0.25, softcap 30 and softmax precision 1 and returns the first result.
                output, _, _, _ = torch.onnx.ops.attention(
                    Q,
                    K,
                    V,
                    is_causal=True,
                    qk_matmul_output_mode=2,
                    scale=0.25,
                    softcap=30.0,
                    softmax_precision=1,  # FLOAT
                )
                return output

        model = FullAttributesModel()
        onnx_program = self.export(model, (Q, K, V), opset_version=23)

        node = onnx_program.model.graph.node(0)
        self.assertEqual(node.op_type, "Attention")

        # Verify all attributes are set correctly
        attrs = node.attributes
        self.assertIn("is_causal", attrs)
        self.assertIn("qk_matmul_output_mode", attrs)
        self.assertIn("scale", attrs)
        self.assertIn("softcap", attrs)
        self.assertIn("softmax_precision", attrs)

        self.assertEqual(attrs["is_causal"].value, 1)  # True as int
        self.assertEqual(attrs["qk_matmul_output_mode"].value, 2)
        self.assertAlmostEqual(attrs["scale"].value, 0.25, places=6)
        self.assertAlmostEqual(attrs["softcap"].value, 30.0, places=6)
        self.assertEqual(attrs["softmax_precision"].value, 1)
        onnx_testing.assert_onnx_program(onnx_program)

    def test_attention_export_with_different_mask_shapes(self):
        # [FUNCTION DOC] Exports attention three times with a two-dimensional, a 4-dimensional
        # with head dimension 1 and a 4-dimensional full boolean mask; for each verifies the
        # fourth node input has the mask shape and checks numerical agreement.
        # KNOWN ISSUE: the second case is labeled as a 3D mask in comments and class names but
        #   the tensor is four-dimensional.
        """Test export with different attention mask shapes."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        # Test 2D mask
        mask_2d = torch.randint(0, 2, (q_seq_len, kv_seq_len), dtype=torch.bool)

        class Mask2DModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model whose forward passes a two-dimensional mask to attention
            # and returns the output.
            def forward(self, Q, K, V, mask):
                # [FUNCTION DOC] HELPER: Calls attention with the given mask and returns the
                # first result.
                output, _, _, _ = torch.onnx.ops.attention(Q, K, V, attn_mask=mask)
                return output

        model_2d = Mask2DModel()
        onnx_program_2d = self.export(model_2d, (Q, K, V, mask_2d), opset_version=23)

        node_2d = onnx_program_2d.model.graph.node(0)
        self.assertEqual(node_2d.inputs[3].shape, [q_seq_len, kv_seq_len])
        onnx_testing.assert_onnx_program(onnx_program_2d)

        # Test 3D mask
        mask_3d = torch.randint(
            0, 2, (batch_size, 1, q_seq_len, kv_seq_len), dtype=torch.bool
        )

        class Mask3DModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model whose forward passes the broadcast (named
            # three-dimensional but actually four-dimensional) mask to attention and returns the
            # output.
            def forward(self, Q, K, V, mask):
                # [FUNCTION DOC] HELPER: Calls attention with the given mask and returns the
                # first result.
                output, _, _, _ = torch.onnx.ops.attention(Q, K, V, attn_mask=mask)
                return output

        model_3d = Mask3DModel()
        onnx_program_3d = self.export(model_3d, (Q, K, V, mask_3d), opset_version=23)

        node_3d = onnx_program_3d.model.graph.node(0)
        self.assertEqual(
            node_3d.inputs[3].shape, [batch_size, 1, q_seq_len, kv_seq_len]
        )
        onnx_testing.assert_onnx_program(onnx_program_3d)

        # Test 4D mask
        mask_4d = torch.randint(
            0, 2, (batch_size, q_num_heads, q_seq_len, kv_seq_len), dtype=torch.bool
        )

        class Mask4DModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model whose forward passes a full four-dimensional mask to
            # attention and returns the output.
            def forward(self, Q, K, V, mask):
                # [FUNCTION DOC] HELPER: Calls attention with the given mask and returns the
                # first result.
                output, _, _, _ = torch.onnx.ops.attention(Q, K, V, attn_mask=mask)
                return output

        model_4d = Mask4DModel()
        onnx_program_4d = self.export(model_4d, (Q, K, V, mask_4d), opset_version=23)

        node_4d = onnx_program_4d.model.graph.node(0)
        self.assertEqual(
            node_4d.inputs[3].shape, [batch_size, q_num_heads, q_seq_len, kv_seq_len]
        )
        onnx_testing.assert_onnx_program(onnx_program_4d)

    def test_attention_export_with_float_mask(self):
        # [FUNCTION DOC] Exports attention with a two-dimensional float mask; verifies an
        # Attention node, the mask input shape, that the mask input dtype is float, and
        # numerical agreement.
        """Test export with float attention mask."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        float_mask = torch.randn(q_seq_len, kv_seq_len)

        class FloatMaskModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model whose forward passes a float mask to attention and
            # returns the output.
            def forward(self, Q, K, V, mask):
                # [FUNCTION DOC] HELPER: Calls attention with the given mask and returns the
                # first result.
                output, _, _, _ = torch.onnx.ops.attention(Q, K, V, attn_mask=mask)
                return output

        model = FloatMaskModel()
        onnx_program = self.export(model, (Q, K, V, float_mask), opset_version=23)

        node = onnx_program.model.graph.node(0)
        self.assertEqual(node.op_type, "Attention")
        self.assertEqual(node.inputs[3].shape, [q_seq_len, kv_seq_len])
        # Verify the mask input has float dtype in the ONNX model
        self.assertEqual(node.inputs[3].dtype, ir.DataType.FLOAT)
        onnx_testing.assert_onnx_program(onnx_program)

    def test_attention_export_qk_output_modes(self):
        # [FUNCTION DOC] For each of the four QK matmul output modes, defines a model that
        # returns the output and QK output, exports it, and verifies an Attention node, the
        # output mode attribute (checked only for nonzero modes), four node outputs, and
        # numerical agreement.
        """Test export with different QK output modes."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        for mode in [0, 1, 2, 3]:

            class QKOutputModel(torch.nn.Module):
                # [CLASS DOC] HELPER: Model, defined per loop iteration, storing a QK output
                # mode and returning the attention output and qk output.
                def __init__(self, qk_mode):
                    # [FUNCTION DOC] HELPER: Stores the QK output mode.
                    super().__init__()
                    self.qk_mode = qk_mode

                def forward(self, Q, K, V):
                    # [FUNCTION DOC] HELPER: Calls attention with the stored mode and returns
                    # the output and the QK output.
                    output, _, _, qk_output = torch.onnx.ops.attention(
                        Q, K, V, qk_matmul_output_mode=self.qk_mode
                    )
                    return output, qk_output

            model = QKOutputModel(mode)
            onnx_program = self.export(model, (Q, K, V), opset_version=23)

            node = onnx_program.model.graph.node(0)
            self.assertEqual(node.op_type, "Attention")

            # Verify qk_matmul_output_mode attribute
            attrs = node.attributes
            if mode != 0:
                self.assertIn("qk_matmul_output_mode", attrs)
                self.assertEqual(attrs["qk_matmul_output_mode"].value, mode)

            # Verify 4 outputs (output, present_key, present_value, qk_output)
            self.assertEqual(len(node.outputs), 4)
            onnx_testing.assert_onnx_program(onnx_program)

    def test_attention_export_mqa(self):
        # [FUNCTION DOC] Exports multi-query attention (one key and value head) and verifies an
        # Attention node, the query, key and value input shapes, and numerical agreement.
        """Test export with Multi-Query Attention (MQA)."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 1  # MQA
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        class MQAModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model whose forward calls attention and returns only the
            # output.
            def forward(self, Q, K, V):
                # [FUNCTION DOC] HELPER: Calls attention and returns the first result.
                output, _, _, _ = torch.onnx.ops.attention(Q, K, V)
                return output

        model = MQAModel()
        onnx_program = self.export(model, (Q, K, V), opset_version=23)

        node = onnx_program.model.graph.node(0)
        self.assertEqual(node.op_type, "Attention")

        # Verify MQA tensor shapes
        self.assertEqual(
            node.inputs[0].shape, [batch_size, q_num_heads, q_seq_len, head_size]
        )
        self.assertEqual(
            node.inputs[1].shape, [batch_size, kv_num_heads, kv_seq_len, head_size]
        )  # kv_num_heads = 1
        self.assertEqual(
            node.inputs[2].shape, [batch_size, kv_num_heads, kv_seq_len, head_size]
        )
        onnx_testing.assert_onnx_program(onnx_program)

    @common_utils.parametrize(
        "precision_enum, precision_name",
        [
            (1, "FLOAT"),
            (10, "FLOAT16"),
            (11, "DOUBLE"),
            (16, "BFLOAT16"),
        ],
    )
    def test_attention_export_with_softmax_precision(
        self, precision_enum, precision_name: str
    ):
        # [FUNCTION DOC] Parametrized over softmax precision codes (float, float16, double,
        # bfloat16) with a name label. Exports attention with that precision, verifies an
        # Attention node, the precision attribute value, and numerical agreement with loosened
        # tolerances (absolute 2e-3, relative 6e-3).
        """Test export with different softmax precision values."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        class SoftmaxPrecisionModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model storing a softmax precision code and returning the
            # attention output.
            def __init__(self, precision):
                # [FUNCTION DOC] HELPER: Stores the softmax precision.
                super().__init__()
                self.precision = precision

            def forward(self, Q, K, V):
                # [FUNCTION DOC] HELPER: Calls attention with the stored softmax precision and
                # returns the first result.
                output, _, _, _ = torch.onnx.ops.attention(
                    Q, K, V, softmax_precision=self.precision
                )
                return output

        model = SoftmaxPrecisionModel(precision_enum)
        onnx_program = self.export(model, (Q, K, V), opset_version=23)

        node = onnx_program.model.graph.node(0)
        self.assertEqual(node.op_type, "Attention")

        # Verify softmax_precision attribute
        attrs = node.attributes
        self.assertIn("softmax_precision", attrs)
        self.assertEqual(attrs["softmax_precision"].value, precision_enum)
        onnx_testing.assert_onnx_program(onnx_program, atol=2e-3, rtol=6e-3)

    def test_attention_export_gqa(self):
        # [FUNCTION DOC] Exports grouped-query attention (8 query heads, 4 key heads) and
        # verifies an Attention node and that the first graph output has shape batch, query
        # heads, query length, head size, plus numerical agreement.
        """Test export and verify output tensor shapes."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 4  # GQA
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        class AttentionOutputsModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model whose forward calls attention and returns only the
            # output.
            def forward(self, Q, K, V):
                # [FUNCTION DOC] HELPER: Calls attention and returns the first result.
                result, _, _, _ = torch.onnx.ops.attention(Q, K, V)
                return result

        model = AttentionOutputsModel()
        onnx_program = self.export(model, (Q, K, V), opset_version=23)

        node = onnx_program.model.graph.node(0)
        self.assertEqual(node.op_type, "Attention")

        graph_outputs = onnx_program.model.graph.outputs
        # output: (batch_size, q_num_heads, q_seq_len, head_size)
        self.assertEqual(
            list(graph_outputs[0].shape),
            [batch_size, q_num_heads, q_seq_len, head_size],
        )

        onnx_testing.assert_onnx_program(onnx_program)


if __name__ == "__main__":
    common_utils.run_tests()
