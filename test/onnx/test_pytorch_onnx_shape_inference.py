# Owner(s): ["module: onnx"]
# [MODULE DOC] Tests for ONNX shape and type inference in the TorchScript exporter. Defines small
# graph-building helpers (a type checker factory, a graph context builder and an op inserter) and
# two test classes: one builds tiny graphs by hand, runs the shape/type inference pass and checks
# the inferred output type of single nodes; the other exports small models with custom operator
# symbolics and checks the shape info saved in the exported model.

import io

import numpy as np

import onnx
import pytorch_test_common
from pytorch_test_common import skipIfUnsupportedMinOpsetVersion

import torch
from torch.onnx import _constants, utils
from torch.onnx._internal.torchscript_exporter import jit_utils
from torch.onnx._internal.torchscript_exporter._globals import GLOBALS
from torch.testing._internal import common_utils


def expect_tensor(scalar_type, shape=None):
    # [FUNCTION DOC] HELPER: Returns a checker function for a node output type. The checker
    # asserts (numpy equality) that the type scalar type equals the given scalar type name; when a
    # shape is given it also asserts that the varying sizes equal the shape, where unknown
    # dimensions are None.
    def verify(actual_type):
        # [FUNCTION DOC] HELPER: The returned checker. Compares scalar type, and varying sizes
        # when a shape was requested; raises an assertion error on any difference.
        np.testing.assert_equal(actual_type.scalarType(), scalar_type)
        # if shape is not None:
        #     np.testing.assert_equal(actual_type.sizes(), shape)
        if shape is not None:
            np.testing.assert_equal(actual_type.varyingSizes(), shape)

    return verify


def as_graphcontext(graph: torch.Graph) -> jit_utils.GraphContext:
    # [FUNCTION DOC] HELPER: Wraps a graph into a graph context object using the graph main block,
    # the maximum supported TorchScript exporter opset, no original node, and empty parameter
    # dictionary, environment and tracked-values set.
    return jit_utils.GraphContext(
        graph=graph,
        block=graph.block(),
        opset=_constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET,
        original_node=None,  # type: ignore[arg-type]
        params_dict={},
        env={},
        values_in_env=set(),
    )


def g_op(graph: torch.Graph, op_name: str, *args, **kwargs):
    # [FUNCTION DOC] HELPER: Inserts an operator into the graph, through a fresh graph context,
    # and returns the resulting value. Extra positional arguments are the inputs and keyword
    # arguments are the node attributes.
    return as_graphcontext(graph).op(op_name, *args, **kwargs)


class TestONNXShapeInference(pytorch_test_common.ExportTestCase):
    # [CLASS DOC] Tests that the C++ ONNX graph shape/type inference pass computes the right
    # scalar type and dimensions for individual ONNX operators. Each test builds a graph of
    # constants and inputs, adds the operator under test, runs inference and checks the output.
    def setUp(self):
        # [FUNCTION DOC] Records the maximum supported opset as the test opset and sets it as the
        # global export opset version. Does not call the base setup.
        self.opset_version = _constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET
        GLOBALS.export_onnx_opset_version = self.opset_version

    def run_test(self, g, n, type_assertion_funcs):
        # [FUNCTION DOC] HELPER: Runs the shape/type inference pass over graph g with the test
        # opset and then applies each type check function, in order, to the matching output type
        # of node n. A single function is accepted in place of a list.
        if not isinstance(type_assertion_funcs, list):
            type_assertion_funcs = [type_assertion_funcs]

        torch._C._jit_pass_onnx_graph_shape_type_inference(g, {}, self.opset_version)
        for out, type_assertion_func in zip(n.outputs(), type_assertion_funcs):
            type_assertion_func(out.type())

    def create_empty_graph(self):
        # [FUNCTION DOC] HELPER: Returns a new empty graph, after running the inference pass once
        # on it so that the constant-value map is initialized.
        g = torch._C.Graph()
        # kick off initialization for ConstantMap.
        torch._C._jit_pass_onnx_graph_shape_type_inference(g, {}, self.opset_version)
        return g

    def insert_tensor_constant(self, g, tensor):
        # [FUNCTION DOC] HELPER: Adds a Constant node holding the given tensor to graph g and
        # returns its output value.
        return g_op(g, "Constant", value_t=tensor)

    def test_cast(self):
        # [FUNCTION DOC] Casts an input of unknown type to float (target type code 1) and expects
        # a Float output scalar type without shape information.
        # Test cast with input of unknown scalar type.
        g = self.create_empty_graph()
        input = g.addInput()
        cast_out = g_op(g, "Cast", input, to_i=1)
        self.run_test(g, cast_out.node(), expect_tensor("Float"))

    def test_constant_of_shape(self):
        # [FUNCTION DOC] Builds a ConstantOfShape fed by the Shape of a 1x2x3x4 constant and
        # expects a Float output of shape 1x2x3x4.
        # Test ConstantOfShape with input of onnx::Shape node.
        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(1, 2, 3, 4))
        shape = g_op(g, "Shape", constant)
        constant_of_shape = g_op(
            g, "ConstantOfShape", shape, value_t=torch.tensor([2.0])
        )
        self.run_test(
            g, constant_of_shape.node(), expect_tensor("Float", shape=(1, 2, 3, 4))
        )

    def test_constant_of_shape_static(self):
        # [FUNCTION DOC] Builds ConstantOfShape fed by a list construct of four constant scalars
        # with values 1 to 4, typed as an int list, and expects a Float output of shape 1x2x3x4.
        # Test ConstantOfShape with input of prim::ListConstruct of static tensor
        rank = 4
        g = self.create_empty_graph()
        constants = [
            self.insert_tensor_constant(g, torch.tensor(i + 1)) for i in range(rank)
        ]
        shape = g_op(g, "prim::ListConstruct", *constants)
        shape.setType(torch._C.ListType.ofInts())
        constant_of_shape = g_op(
            g, "ConstantOfShape", shape, value_t=torch.tensor([2.0])
        )
        self.run_test(
            g, constant_of_shape.node(), expect_tensor("Float", shape=(1, 2, 3, 4))
        )

    def test_constant_of_shape_dynamic(self):
        # [FUNCTION DOC] Builds ConstantOfShape fed by a list construct of four graph inputs
        # (values unknown), typed as an int list, and expects a Float output of rank 4 with
        # all dimensions unknown.
        # Test ConstantOfShape with input of prim::ListConstruct of dynamic tensor
        rank = 4
        g = self.create_empty_graph()
        inputs = [g.addInput() for i in range(rank)]
        shape = g_op(g, "prim::ListConstruct", *inputs)
        shape.setType(torch._C.ListType.ofInts())
        constant_of_shape = g_op(
            g, "ConstantOfShape", shape, value_t=torch.tensor([2.0])
        )
        self.run_test(
            g,
            constant_of_shape.node(),
            expect_tensor("Float", shape=(None, None, None, None)),
        )

    def test_gather_dynamic_index(self):
        # [FUNCTION DOC] Gather along axis 1 of a float input of sizes unknown, 3, 16, 16 with a
        # dynamic 1-D index input; expects Float output of unknown, unknown, 16, 16.
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(
            input.type().with_dtype(torch.float).with_sizes([None, 3, 16, 16])
        )
        indices = g.addInput()
        indices.setType(indices.type().with_dtype(torch.int64).with_sizes([None]))
        output = g_op(g, "Gather", input, indices, axis_i=1)
        self.run_test(
            g, output.node(), expect_tensor("Float", shape=([None, None, 16, 16]))
        )

    def test_gather_scalar_index(self):
        # [FUNCTION DOC] Gather along axis 1 of a float input of sizes unknown, 3, 16, 16 with a
        # scalar constant index; expects the indexed axis removed, giving unknown, 16, 16.
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(
            input.type().with_dtype(torch.float).with_sizes([None, 3, 16, 16])
        )
        indices = self.insert_tensor_constant(g, torch.tensor(1))
        output = g_op(g, "Gather", input, indices, axis_i=1)
        self.run_test(g, output.node(), expect_tensor("Float", shape=([None, 16, 16])))

    def test_reshape(self):
        # [FUNCTION DOC] Reshape of constant tensors with target shapes using the zero (copy
        # this dimension) and minus one (infer) entries; expects 2x16x25, 10x16x4 and 8x16x5 for
        # three separate graphs.
        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(2, 16, 5, 5))
        constant_2 = self.insert_tensor_constant(g, torch.tensor([2, 0, -1]))
        shape = g_op(g, "Reshape", constant, constant_2)
        self.run_test(g, shape.node(), expect_tensor("Float", shape=(2, 16, 25)))

        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(2, 16, 5, 4))
        constant_2 = self.insert_tensor_constant(g, torch.tensor([-1, 0, 4]))
        shape = g_op(g, "Reshape", constant, constant_2)
        self.run_test(g, shape.node(), expect_tensor("Float", shape=(10, 16, 4)))

        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(2, 16, 5, 4))
        constant_2 = self.insert_tensor_constant(g, torch.tensor([-1, 0, 0]))
        shape = g_op(g, "Reshape", constant, constant_2)
        self.run_test(g, shape.node(), expect_tensor("Float", shape=(8, 16, 5)))

    def test_reshape_symbolic(self):
        # [FUNCTION DOC] Reshape of an input of sizes unknown, unknown, 2, 8 to a constant target
        # with two copy entries and one inferred entry; expects unknown, unknown, 16 with no
        # scalar type assertion.
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_sizes([None, None, 2, 8]))
        constant = self.insert_tensor_constant(g, torch.tensor([0, 0, -1]))
        output = g_op(g, "Reshape", input, constant)
        self.run_test(g, output.node(), expect_tensor(None, shape=(None, None, 16)))

    @skipIfUnsupportedMinOpsetVersion(14)
    def test_reshape_allowzero(self):
        # [FUNCTION DOC] Reshape with the allow-zero attribute set, of an input with sizes 3, 4, 0
        # to a target containing a zero that is taken literally; expects output sizes 0, 4, 3.
        # Needs opset 14 or newer.
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_sizes([3, 4, 0]))
        constant = self.insert_tensor_constant(g, torch.tensor([0, 4, 3]))
        output = g_op(g, "Reshape", input, constant, allowzero_i=1)
        self.run_test(g, output.node(), expect_tensor(None, shape=(0, 4, 3)))

    def test_slice(self):
        # [FUNCTION DOC] Slice of a 2-D input with unknown sizes using a dynamic start input and
        # constant end, axes and steps; expects a 2-D output with both dimensions unknown.
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_sizes([None, None]))
        start_input = g.addInput()
        start_input.setType(start_input.type().with_sizes([None]))
        end = self.insert_tensor_constant(g, torch.tensor([3]))
        axis = self.insert_tensor_constant(g, torch.tensor([0]))
        step = self.insert_tensor_constant(g, torch.tensor([1]))
        slice = g_op(g, "Slice", input, start_input, end, axis, step)
        self.run_test(g, slice.node(), expect_tensor(None, shape=(None, None)))

    def test_slice_with_dynamic_start_index(self):
        # [FUNCTION DOC] Slice of a constant 2x3x4x5 tensor with a dynamic two-element start input
        # and constant ends and axes (axis one and the last axis); expects output sizes 2,
        # unknown, 4, unknown.
        g = self.create_empty_graph()
        input = self.insert_tensor_constant(g, torch.ones(2, 3, 4, 5))
        start_input = g.addInput()
        start_input.setType(start_input.type().with_sizes([2]))
        end = self.insert_tensor_constant(g, torch.tensor([3, 4]))
        axis = self.insert_tensor_constant(g, torch.tensor([1, -1]))
        slice = g_op(g, "Slice", input, start_input, end, axis)
        self.run_test(g, slice.node(), expect_tensor(None, shape=(2, None, 4, None)))

    def test_broadcast_matmul(self):
        # [FUNCTION DOC] MatMul broadcasting on constants in four graphs: batched 5x1x2 by 3x1x2x1
        # gives 3x5x1x1; rank-1 first operand gives 3x1x1; rank-1 second operand gives 5x1; two
        # rank-1 operands give a scalar shape (empty). All expect Float.
        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(5, 1, 2))
        constant_2 = self.insert_tensor_constant(g, torch.ones(3, 1, 2, 1))
        shape = g_op(g, "MatMul", constant, constant_2)
        self.run_test(g, shape.node(), expect_tensor("Float", shape=(3, 5, 1, 1)))

        # test when first input is of rank 1
        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(2))
        constant_2 = self.insert_tensor_constant(g, torch.ones(3, 1, 2, 1))
        shape = g_op(g, "MatMul", constant, constant_2)
        self.run_test(g, shape.node(), expect_tensor("Float", shape=(3, 1, 1)))

        # test when second input is of rank 1
        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(5, 1, 2))
        constant_2 = self.insert_tensor_constant(g, torch.ones(2))
        shape = g_op(g, "MatMul", constant, constant_2)
        self.run_test(g, shape.node(), expect_tensor("Float", shape=(5, 1)))

        # test when both inputs are of rank 1
        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(2))
        constant_2 = self.insert_tensor_constant(g, torch.ones(2))
        shape = g_op(g, "MatMul", constant, constant_2)
        self.run_test(g, shape.node(), expect_tensor("Float", shape=()))

    def test_expand(self):
        # [FUNCTION DOC] Expand a constant 2x4 tensor to the Shape of a 2-D input with unknown
        # sizes; expects Float output with two unknown dimensions.
        g = self.create_empty_graph()
        input = g.addInput()
        constant = self.insert_tensor_constant(g, torch.ones(2, 4))
        input.setType(constant.type().with_sizes([None, None]))
        shape = g_op(g, "Shape", input)
        expand = g_op(g, "Expand", constant, shape)
        self.run_test(g, expand.node(), expect_tensor("Float", shape=(None, None)))

    def test_pad(self):
        # [FUNCTION DOC] Pad in constant mode on a float 3x320x100 input with a constant
        # pad list of six ones (one before and one after per dimension) and a none
        # value; expects 5x322x102.
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_dtype(torch.float).with_sizes([3, 320, 100]))
        constant = self.insert_tensor_constant(g, torch.ones(6, dtype=torch.long))
        none = g_op(g, "prim::Constant").setType(torch.NoneType.get())
        pad = g_op(g, "Pad", input, constant, none, mode_s="constant")
        self.run_test(g, pad.node(), expect_tensor("Float", shape=(5, 322, 102)))

    def test_pad_with_dynamic_input_shape(self):
        # [FUNCTION DOC] Same constant-mode Pad with an input of sizes 3, unknown, unknown;
        # expects 5, unknown, unknown (known dimension is padded, unknown stay unknown).
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_dtype(torch.float).with_sizes([3, None, None]))
        constant = self.insert_tensor_constant(g, torch.ones(6, dtype=torch.long))
        none = g_op(g, "prim::Constant").setType(torch.NoneType.get())
        pad = g_op(g, "Pad", input, constant, none, mode_s="constant")
        self.run_test(g, pad.node(), expect_tensor("Float", shape=(5, None, None)))

    def test_pad_with_dynamic_pad_size(self):
        # [FUNCTION DOC] Constant-mode Pad where the pad amounts are a graph input of length six
        # (values unknown); expects a rank-3 Float output with all dimensions unknown.
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_dtype(torch.float).with_sizes([3, 320, 100]))
        pad_size = g.addInput()
        pad_size.setType(pad_size.type().with_dtype(torch.long).with_sizes([6]))
        none = g_op(g, "prim::Constant").setType(torch.NoneType.get())
        pad = g_op(g, "Pad", input, pad_size, none, mode_s="constant")
        self.run_test(g, pad.node(), expect_tensor("Float", shape=(None, None, None)))

    def test_resize(self):
        # [FUNCTION DOC] Resize in linear mode on a float 4x32x64x64 input using constant scales
        # of 1, 1, 2, 2 and no size input; expects Float output 4x32x128x128.
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_dtype(torch.float).with_sizes([4, 32, 64, 64]))
        none = g_op(g, "prim::Constant").setType(torch.NoneType.get())
        scales = self.insert_tensor_constant(
            g, torch.tensor([1, 1, 2, 2], dtype=torch.float)
        )
        resize = g_op(
            g,
            "Resize",
            input,
            none,
            scales,
            coordinate_transformation_mode_s="align_corners",
            cubic_coeff_a_f=-0.75,
            mode_s="linear",
            nearest_mode_s="floor",
        )
        self.run_test(g, resize.node(), expect_tensor("Float", shape=(4, 32, 128, 128)))

    def test_resize_after_concat(self):
        # [FUNCTION DOC] Same Resize as the previous test but the scales come from concatenating
        # two constant tensors, relying on constant folding in the inference pass to know the
        # values; expects Float output 4x32x128x128.
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_dtype(torch.float).with_sizes([4, 32, 64, 64]))
        none = g_op(g, "prim::Constant").setType(torch.NoneType.get())
        scale_1 = self.insert_tensor_constant(
            g, torch.tensor([1, 1], dtype=torch.float)
        )
        scale_2 = self.insert_tensor_constant(
            g, torch.tensor([2, 2], dtype=torch.float)
        )
        # `scales` values should be statically known due to constant folding in shape inference.
        scales = g_op(g, "Concat", scale_1, scale_2, axis_i=0)
        resize = g_op(
            g,
            "Resize",
            input,
            none,
            scales,
            coordinate_transformation_mode_s="align_corners",
            cubic_coeff_a_f=-0.75,
            mode_s="linear",
            nearest_mode_s="floor",
        )
        self.run_test(g, resize.node(), expect_tensor("Float", shape=(4, 32, 128, 128)))

    def test_reduce_prod_with_axes(self):
        # [FUNCTION DOC] ReduceProd with the axes attribute on a Long input of size 2; expects
        # Long output of shape one (kept dimension).
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_dtype(torch.long).with_sizes([2]))
        reduce_prod = g_op(g, "ReduceProd", input, axes_i=[0])
        self.run_test(g, reduce_prod.node(), expect_tensor("Long", shape=(1,)))

    def test_reduce_prod_without_axes(self):
        # [FUNCTION DOC] ReduceProd without axes on a Long input of size 2; expects Long output of
        # shape one (all axes reduced, dimension kept).
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_dtype(torch.long).with_sizes([2]))
        reduce_prod = g_op(g, "ReduceProd", input)
        self.run_test(g, reduce_prod.node(), expect_tensor("Long", shape=(1,)))

    def test_proceeding_nodes_use_prim_pack_padded_output_dtype_correctly(self):
        # [FUNCTION DOC] Builds a two-output PackPadded node (as produced when tracing) with
        # output types set manually, then a Gather on its second output (batch sizes). Expects the
        # Gather result to use the Long type of that output with a single unknown dimension.
        g = self.create_empty_graph()
        input = g.addInput()
        input.setType(input.type().with_dtype(torch.float).with_sizes([4, 16]))
        length = g.addInput()
        length.setType(length.type().with_dtype(torch.long).with_sizes([4]))
        padded, batch_size = g_op(g, "prim::PackPadded", input, length, outputs=2)
        # `prim::PackPadded` only occurs in tracing mode. Hence its outputs inherits
        # shape and data type from traced graph.
        padded.setType(padded.type().with_dtype(torch.float).with_sizes([None, None]))
        batch_size.setType(batch_size.type().with_dtype(torch.long).with_sizes([None]))
        # `Gather` should use the data type of `batch_size` as the data type of its output.
        gather_idx = self.insert_tensor_constant(g, torch.tensor([0], dtype=torch.long))
        gather = g_op(g, "Gather", batch_size, gather_idx, axis_i=0)
        self.run_test(g, gather.node(), expect_tensor("Long", shape=(None,)))

    def test_squeeze_after_dynamic_if(self):
        # [FUNCTION DOC] Builds an If node with two blocks (one adds the input to itself, the
        # other is identity) on an int32 condition, runs fixup and node inference, then calls the
        # opset 11 squeeze symbolic on dimension zero. Raises an assertion error unless a plain
        # Squeeze node was emitted (meaning the size-one dimension was known); then expects Float
        # output of sizes unknown, 5.
        from torch.onnx.symbolic_opset11 import squeeze as squeeze11

        g = self.create_empty_graph()

        input = g.addInput()
        input.setType(input.type().with_dtype(torch.float).with_sizes([1, None, 5]))

        # Type is intentionally not bool to test that
        # the added "Cast" node doesn't stop shape inference.
        cond = g.addInput()
        cond.setType(input.type().with_dtype(torch.int32).with_sizes([1]))
        _, (if_context, else_context), new_node = jit_utils.add_op_with_blocks(
            as_graphcontext(g), "If", cond, n_blocks=2
        )
        block1_output = if_context.op("Add", input, input)
        block2_output = else_context.op("Identity", input)
        utils._add_output_to_block(if_context.block, block1_output)
        utils._add_output_to_block(else_context.block, block2_output)
        if_output = torch._C._jit_pass_fixup_onnx_controlflow_node(
            new_node, _constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET
        )[0]
        torch._C._jit_pass_onnx_node_shape_type_inference(
            new_node, {}, _constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET
        )

        # Exporter will add "If" instead of raw "Squeeze" if it does not know
        # that if the dimension it is squeezing has size 1.
        squeezed = squeeze11(as_graphcontext(g), if_output, dim=0)
        if squeezed.node().kind() != "onnx::Squeeze":
            raise AssertionError(
                f"Expected 'onnx::Squeeze' but got {squeezed.node().kind()!r}"
            )
        self.run_test(g, squeezed.node(), expect_tensor("Float", shape=(None, 5)))


class TestONNXCustomOpShapeInference(pytorch_test_common.ExportTestCase):
    # [CLASS DOC] Tests exported-model shape info for custom operators registered with symbolic
    # functions that do or do not set the output type. Exports use the legacy exporter at the
    # maximum opset with a custom opset domain, then read value info from the saved graph.
    def setUp(self):
        # [FUNCTION DOC] Calls the base setup and records the maximum supported opset
        # for the tests.
        super().setUp()
        self.opset_version = _constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET

    def test_setType_maintains_output_shape_for_single_custom_op(self):
        # [FUNCTION DOC] Registers a custom symbolic for the inverse op whose result copies the
        # input type, exports a module computing inverse plus input for a 2x3x3 input, and checks
        # that the first value info of the saved graph has only fixed dimensions equal to the
        # input sizes. The symbolic is unregistered by cleanup.
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, "::linalg_inv", 9)

        class CustomInverse(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward adds the input to its matrix inverse.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the matrix inverse of the input plus the input.
                return torch.inverse(x) + x

        def linalg_inv_settype(g, self):
            # [FUNCTION DOC] HELPER: Custom symbolic emitting a custom-domain Inverse node and
            # giving its output the type of the input.
            return g.op("com.microsoft::Inverse", self).setType(self.type())

        torch.onnx.register_custom_op_symbolic("::linalg_inv", linalg_inv_settype, 9)
        model = CustomInverse()
        x = torch.randn(2, 3, 3)
        f = io.BytesIO()
        torch.onnx.export(
            model,
            (x,),
            f,
            opset_version=self.opset_version,
            custom_opsets={"com.microsoft": 1},
            dynamo=False,
        )

        model_proto = onnx.load(io.BytesIO(f.getvalue()))
        model_value_info = model_proto.graph.value_info
        self.assertIsNotNone(model_value_info)
        if not model_value_info:
            raise AssertionError("model_value_info is empty")
        dims = model_value_info[0].type.tensor_type.shape.dim
        for i in range(len(dims)):
            # If node output has shape info, it should have dim_value
            # Otherwise, it has dim_params with dynamic shape
            self.assertTrue(dims[i].HasField("dim_value"))
        for dim, rank in zip(dims, x.size()):
            self.assertEqual(dim.dim_value, rank)

    def test_no_setType_for_single_custom_op(self):
        # [FUNCTION DOC] Registers a custom symbolic that emits the custom Inverse node
        # without setting its type, exports, and checks that dimensions of the first value
        # info are all symbolic (dim param), i.e. shape was not inferred. The symbolic is
        # unregistered by cleanup.
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, "::linalg_inv", 9)

        class CustomInverse(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward adds the input to its matrix inverse.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the matrix inverse of the input plus the input.
                return torch.inverse(x) + x

        def linalg_inv_no_settype(g, self):
            # [FUNCTION DOC] HELPER: Custom symbolic emitting a custom-domain Inverse node without
            # setting any output type.
            return g.op("com.microsoft::Inverse", self)

        torch.onnx.register_custom_op_symbolic("::linalg_inv", linalg_inv_no_settype, 9)
        model = CustomInverse()
        x = torch.randn(2, 3, 3)
        f = io.BytesIO()
        torch.onnx.export(
            model,
            (x,),
            f,
            opset_version=self.opset_version,
            custom_opsets={"com.microsoft": 1},
            dynamo=False,
        )

        model_proto = onnx.load(io.BytesIO(f.getvalue()))
        model_value_info = model_proto.graph.value_info
        self.assertIsNotNone(model_value_info)
        if not model_value_info:
            raise AssertionError("model_value_info is empty")
        dims = model_value_info[0].type.tensor_type.shape.dim
        for i in range(len(dims)):
            # If node output has shape info, it should have dim_value
            # Otherwise, it has dim_params with dynamic shape
            self.assertTrue(dims[i].HasField("dim_param"))

    def test_setType_maintains_output_shape_for_single_custom_op_with_dynamic_axes(
        self,
    ):
        # [FUNCTION DOC] Custom symbolic sets the output type to float with sizes unknown, 3, 3;
        # export uses a named input with the first axis dynamic (batch). Checks the first value
        # info has a symbolic first dimension and fixed remaining dimensions equal to the input
        # sizes. The symbolic is unregistered by cleanup.
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, "::linalg_inv", 9)

        class CustomInverse(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward adds the input to its matrix inverse.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the matrix inverse of the input plus the input.
                return torch.inverse(x) + x

        def linalg_inv_settype(g, self):
            # [FUNCTION DOC] HELPER: Custom symbolic emitting a custom-domain Inverse node whose
            # output type is float with sizes unknown, 3, 3.
            return g.op("com.microsoft::Inverse", self).setType(
                self.type().with_dtype(torch.float).with_sizes([None, 3, 3])
            )

        torch.onnx.register_custom_op_symbolic("::linalg_inv", linalg_inv_settype, 9)
        model = CustomInverse()
        x = torch.randn(2, 3, 3)
        f = io.BytesIO()
        torch.onnx.export(
            model,
            (x,),
            f,
            opset_version=self.opset_version,
            custom_opsets={"com.microsoft": 1},
            input_names=["x"],
            dynamic_axes={"x": {0: "batch"}},
            dynamo=False,
        )

        model_proto = onnx.load(io.BytesIO(f.getvalue()))
        model_value_info = model_proto.graph.value_info
        self.assertIsNotNone(model_value_info)
        if not model_value_info:
            raise AssertionError("model_value_info is empty")
        dims = model_value_info[0].type.tensor_type.shape.dim
        # The first axe should be dynamic as we defined when exporting
        self.assertTrue(dims[0].HasField("dim_param"))
        for i in range(1, len(dims)):
            # If node output has shape info, it should have dim_value
            # Otherwise, it has dim_params with dynamic shape
            self.assertTrue(dims[i].HasField("dim_value"))
            self.assertEqual(dims[i].dim_value, x.size()[i])

    def test_setType_maintains_output_shape_for_single_custom_op_with_onnx_ops(self):
        # [FUNCTION DOC] Exports a module computing the inverse of the first input plus the other
        # two (all 2x3x10x10) with a custom symbolic setting a fixed float output type. Finds the
        # Inverse node output name in the saved graph, then checks that the matching value info
        # has fixed dimensions equal to the input sizes. Raises assertion errors if the node, the
        # value info or a name is missing. The symbolic is unregistered by cleanup.
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, "::linalg_inv", 9)

        class CustomInverse(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward takes three inputs and returns the inverse
            # of the first plus the second plus the third.
            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Returns the inverse of x added to y and z.
                x = torch.inverse(x)
                return x + y + z

        def linalg_inv_settype(g, self):
            # [FUNCTION DOC] HELPER: Custom symbolic emitting a custom-domain Inverse node whose
            # output type is float with fixed sizes 2, 3, 10, 10.
            return g.op("com.microsoft::Inverse", self).setType(
                self.type().with_dtype(torch.float).with_sizes([2, 3, 10, 10])
            )

        torch.onnx.register_custom_op_symbolic("::linalg_inv", linalg_inv_settype, 9)
        model = CustomInverse()
        x = torch.randn(2, 3, 10, 10)
        y = torch.randn(2, 3, 10, 10)
        z = torch.randn(2, 3, 10, 10)
        f = io.BytesIO()
        torch.onnx.export(
            model,
            (x, y, z),
            f,
            opset_version=self.opset_version,
            custom_opsets={"com.microsoft": 1},
            dynamo=False,
        )

        model_proto = onnx.load(io.BytesIO(f.getvalue()))
        # To validate the shape of inverse Op, we need to find inverse output name,
        # and then use it to identify its value_info for the shape.
        output_name = ""
        for node in model_proto.graph.node:
            if node.op_type == "Inverse":
                output_name = node.output[0]
                break
        if not output_name:
            raise AssertionError("output_name not found")
        model_value_info = model_proto.graph.value_info
        self.assertIsNotNone(model_value_info)
        if not model_value_info:
            raise AssertionError("model_value_info is empty")
        for value_info in model_value_info:
            if not value_info.name:
                raise AssertionError("value_info.name is empty")
            if value_info.name == output_name:
                dims = value_info.type.tensor_type.shape.dim
                for i in range(len(dims)):
                    # If node output has shape info, it should have dim_value
                    # Otherwise, it has dim_params with dynamic shape
                    self.assertTrue(dims[i].HasField("dim_value"))
                for dim, rank in zip(dims, x.size()):
                    self.assertEqual(dim.dim_value, rank)


if __name__ == "__main__":
    common_utils.run_tests()
