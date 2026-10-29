# Owner(s): ["module: fx"]

# [MODULE DOC] Tests for the FX gradual type system: tensor type annotations, the consistency and
# precision relations, broadcasting of types, the GraphTypeChecker pass (type inference through
# add, reshape, transpose, flatten, batch norm, conv2d, maxpool2d and so on), and symbolic type
# inference with Refine and infer_symbolic_types. Import-time side effect: tries to import
# torchvision resnet50 and sets HAS_TORCHVISION and the skipIfNoTorchVision decorator accordingly.
# Running the file directly is refused through raise_on_run_directly in favor of test/test_fx.py.
import unittest

import sympy

import torch
from torch.fx import GraphModule, symbolic_trace
from torch.fx.annotate import annotate
from torch.fx.experimental.graph_gradual_typechecker import (
    broadcast_types,
    GraphTypeChecker,
    Refine,
)
from torch.fx.experimental.refinement_types import Equality
from torch.fx.experimental.rewriter import RewritingTracer
from torch.fx.experimental.unify_refinements import infer_symbolic_types
from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.tensor_type import Dyn, is_consistent, is_more_precise, TensorType
from torch.testing._internal.common_utils import raise_on_run_directly, TestCase


try:
    from torchvision.models import resnet50

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    # [FUNCTION DOC] HELPER: Builds and returns a bias-free 3x3 convolution layer with the given
    # input and output channels, stride, groups and dilation, using padding equal to the dilation.
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class AnnotationsTest(TestCase):
    # [CLASS DOC] Tests tensor type annotations on traced graphs plus the consistency, precision
    # and broadcasting helpers of the tensor type module. Stateless; each test builds its own
    # inputs.
    def test_annotations(self):
        # [FUNCTION DOC] Traces a module whose forward annotates three arguments (a four
        # dimensional type with an unknown last dimension, the unknown type, a three dimensional
        # type) and asserts each placeholder node carries the matching type, in order.
        """
        Test type annotations in the forward function.
        The annotation should appear in the n.graph
        where n is the corresponding node in the resulting graph.
        """

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding its first two inputs and then a third, with
            # annotated args.
            def forward(
                self, x: TensorType((1, 2, 3, Dyn)), y: Dyn, z: TensorType[Dyn, 3, Dyn]
            ):
                # [FUNCTION DOC] HELPER: Returns the sum of x and y plus z; only the annotations
                # matter.
                return torch.add(x, y) + z

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)

        expected_ph_types = [TensorType((1, 2, 3, Dyn)), Dyn, TensorType((Dyn, 3, Dyn))]
        expected_iter = iter(expected_ph_types)

        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                expected = next(expected_iter)
                if n.type != expected:
                    raise AssertionError(f"Expected n.type == {expected}, got {n.type}")

    def test_annotate(self):
        # [FUNCTION DOC] Traces a module that wraps its input with the annotate helper and a four
        # dimensional type, then asserts every placeholder node has that type. Raises
        # AssertionError on mismatch.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that annotates its input and adds it to the original.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x added to its annotated alias.
                y = annotate(x, TensorType((1, 2, 3, Dyn)))
                return torch.add(x, y)

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                if n.type != TensorType((1, 2, 3, Dyn)):
                    raise AssertionError(
                        f"Expected n.type == TensorType((1, 2, 3, Dyn)), got {n.type}"
                    )

    def test_consistency(self):
        # [FUNCTION DOC] Checks is_consistent: a concrete tensor type is consistent with one that
        # has an unknown dimension, Dyn is consistent with int, int with int, and tensor types of
        # different rank or a tensor type versus int are not.
        """
        Test the consistency relation.
        """
        self.assertTrue(is_consistent(TensorType((1, 2, 3)), TensorType((1, Dyn, 3))))
        self.assertTrue(is_consistent(int, Dyn))
        self.assertTrue(is_consistent(int, int))
        self.assertFalse(is_consistent(TensorType((1, 2, 3)), TensorType((1, 2, 3, 5))))
        self.assertFalse(is_consistent(TensorType((1, 2, 3)), int))

    def test_precision(self):
        # [FUNCTION DOC] Checks is_more_precise: a concrete tensor type is more precise than one
        # with an unknown dimension, int is more precise than Dyn and than int, and differing
        # ranks or tensor versus int are not.
        # KNOWN ISSUE: the docstring says it tests the consistency relation, but the body tests
        #     precision.
        """
        Test the consistency relation.
        """
        self.assertTrue(is_more_precise(TensorType((1, 2, 3)), TensorType((1, Dyn, 3))))
        self.assertTrue(is_more_precise(int, Dyn))
        self.assertTrue(is_more_precise(int, int))
        self.assertFalse(
            is_more_precise(TensorType((1, 2, 3)), TensorType((1, 2, 3, 5)))
        )
        self.assertFalse(is_more_precise(TensorType((1, 2, 3)), int))

    def test_broadcasting1(self):
        # [FUNCTION DOC] Checks broadcast_types on tensor types of rank four, rank four with a
        # size one dim, scalar-like empty shape, a two dimensional shape and a list-based shape:
        # both operands come back broadcast to the common shape. Raises AssertionError on
        # mismatch.
        t1 = TensorType((1, 2, 3, 4))
        t2 = TensorType((1, 2, 1, 4))
        t3 = TensorType(())
        t4 = TensorType((4, 1))
        t5 = TensorType((4, 4, 4))
        # todo switch all code to use list instead of tuple
        t6 = TensorType([1])
        result = broadcast_types(t1, t2)
        expected = (TensorType((1, 2, 3, 4)), TensorType((1, 2, 3, 4)))
        if result != expected:
            raise AssertionError(f"Expected {expected}, got {result}")
        if broadcast_types(t3, t4) != (t4, t4):
            raise AssertionError("broadcast_types(t3, t4) mismatch")
        if broadcast_types(t5, t6) != (t5, t5):
            raise AssertionError("broadcast_types(t5, t6) mismatch")

    def test_broadcasting2(self):
        # [FUNCTION DOC] Checks broadcast_types of a rank three and a rank four type, expecting
        # both to be returned as the same rank four type.
        t1 = TensorType((2, 3, 4))
        t2 = TensorType((1, 2, 1, 4))

        result = broadcast_types(t1, t2)
        expected = (TensorType((1, 2, 3, 4)), TensorType((1, 2, 3, 4)))
        if result != expected:
            raise AssertionError(f"Expected {expected}, got {result}")

    def test_broadcasting3(self):
        # [FUNCTION DOC] Checks broadcast_types when one operand has an unknown dimension: the
        # result keeps the unknown dimension for the first operand and the concrete size for the
        # second.
        t1 = TensorType((1, 2, 3, Dyn))
        t2 = TensorType((2, 3, 4))
        result = broadcast_types(t1, t2)
        expected = (TensorType((1, 2, 3, Dyn)), TensorType((1, 2, 3, 4)))
        if result != expected:
            raise AssertionError(f"Expected {expected}, got {result}")


class TypeCheckerTest(TestCase):
    # [CLASS DOC] Tests GraphTypeChecker and symbolic inference over small traced modules (add,
    # reshape, transpose, batch norm, conv2d, flatten, maxpool2d, resnet50). Many cases build a
    # module, trace it, run type checking and assert node types or a TypeError.
    def test_type_check_add_with_broadcast(self):
        # [FUNCTION DOC] Type checks adding a four dimensional type with an unknown last dim to a
        # rank three type. Every call_function node must have its broadcast metadata flag set and
        # the node types must be x, y and then the broadcast result for the add and the output.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that adds two annotated inputs.
            def forward(self, x: TensorType((1, 2, 3, Dyn)), y: TensorType((2, 3, 4))):
                # [FUNCTION DOC] HELPER: Returns the sum of x and y.
                return torch.add(x, y)

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        tc.type_check()
        expected_ph_types = [
            TensorType((1, 2, 3, Dyn)),
            TensorType((2, 3, 4)),
            TensorType((1, 2, 3, Dyn)),
            TensorType((1, 2, 3, Dyn)),
        ]
        expected_iter = iter(expected_ph_types)

        for n in symbolic_traced.graph.nodes:
            if n.op == "call_function":
                if not n.meta["broadcast"]:
                    raise AssertionError("Expected n.meta['broadcast'] to be True")
            expected = next(expected_iter)
            if n.type != expected:
                raise AssertionError(f"Expected n.type == {expected}, got {n.type}")

    def test_type_check_add_with_scalar(self):
        # [FUNCTION DOC] Type checks adding an int argument to a rank three tensor type: expects
        # node types int, the tensor type, and the tensor type for both the add and the output.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding an int input to an annotated tensor input.
            def forward(self, x: int, y: TensorType((2, 3, 4))):
                # [FUNCTION DOC] HELPER: Returns the sum of x and y.
                return torch.add(x, y)

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        tc.type_check()
        expected_ph_types = [
            int,
            TensorType((2, 3, 4)),
            TensorType((2, 3, 4)),
            TensorType((2, 3, 4)),
        ]
        expected_iter = iter(expected_ph_types)

        for n in symbolic_traced.graph.nodes:
            expected = next(expected_iter)
            if n.type != expected:
                raise AssertionError(f"Expected n.type == {expected}, got {n.type}")

    def test_type_check_add_false(self):
        # [FUNCTION DOC] Type checks adding rank four to rank three types whose dimensions
        # disagree; expects type_check to raise TypeError.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that adds two annotated inputs of incompatible shapes.
            def forward(self, x: TensorType((1, 2, 3, Dyn)), y: TensorType((1, 2, 3))):
                # [FUNCTION DOC] HELPER: Returns the sum of x and y.
                return torch.add(x, y)

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        with self.assertRaises(TypeError):
            tc.type_check()

    def test_type_check_add_true(self):
        # [FUNCTION DOC] Type checks adding two rank three types where one has an unknown
        # dimension. type_check must return true, placeholders keep their annotations and the
        # output type is the type with the unknown dimension.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that adds two compatible annotated inputs.
            def forward(self, x: TensorType((1, 2, Dyn)), y: TensorType((1, 2, 3))):
                # [FUNCTION DOC] HELPER: Returns the sum of x and y.
                return torch.add(x, y)

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        self.assertTrue(tc.type_check())

        expected_ph_types = [TensorType((1, 2, Dyn)), TensorType((1, 2, 3))]
        expected_iter = iter(expected_ph_types)

        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                expected = next(expected_iter)
                if n.type != expected:
                    raise AssertionError(f"Expected n.type == {expected}, got {n.type}")
            if n.op == "output":
                if n.type != TensorType((1, 2, Dyn)):
                    raise AssertionError(
                        f"Expected n.type == TensorType((1, 2, Dyn)), got {n.type}"
                    )

    def test_type_check_reshape_true(self):
        # [FUNCTION DOC] Type checks reshaping a one by six type to one by two by three.
        # type_check returns true and placeholder, call_function and output nodes get the expected
        # types.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that reshapes its input to one by two by three.
            def forward(self, x: TensorType((1, 6))):
                # [FUNCTION DOC] HELPER: Returns x reshaped to one by two by three.
                return torch.reshape(x, [1, 2, 3])

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        self.assertTrue(tc.type_check())

        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                if n.type != TensorType((1, 6)):
                    raise AssertionError(
                        f"Expected n.type == TensorType((1, 6)), got {n.type}"
                    )

            if n.op == "call_function":
                if n.type != TensorType((1, 2, 3)):
                    raise AssertionError(
                        f"Expected n.type == TensorType((1, 2, 3)), got {n.type}"
                    )

            if n.op == "output":
                if n.type != TensorType((1, 2, 3)):
                    raise AssertionError(
                        f"Expected n.type == TensorType((1, 2, 3)), got {n.type}"
                    )

    def test_type_check_reshape_false(self):
        # [FUNCTION DOC] Reshaping a one by five type to one by two by three has mismatched
        # element counts; expects type_check to raise TypeError.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that reshapes its input to one by two by three.
            def forward(self, x: TensorType((1, 5))):
                # [FUNCTION DOC] HELPER: Returns x reshaped to one by two by three.
                return torch.reshape(x, [1, 2, 3])

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        with self.assertRaises(TypeError):
            tc.type_check()

    def test_type_check_reshape_dyn_false(self):
        # [FUNCTION DOC] Reshaping a one by five type with an inferred dimension (minus one) to a
        # target with a fixed two cannot divide evenly; expects type_check to raise TypeError.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that reshapes its input using an inferred last dimension.
            def forward(self, x: TensorType((1, 5))):
                # [FUNCTION DOC] HELPER: Returns x reshaped to one by two by inferred.
                return torch.reshape(x, [1, 2, -1])

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        with self.assertRaises(TypeError):
            tc.type_check()

    def test_type_check_reshape_dyn_true(self):
        # [FUNCTION DOC] Reshaping a one by fifteen type to one by five by inferred is valid;
        # type_check must return true.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that reshapes its input using an inferred last dimension.
            def forward(self, x: TensorType((1, 15))):
                # [FUNCTION DOC] HELPER: Returns x reshaped to one by five by inferred.
                return torch.reshape(x, [1, 5, -1])

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        self.assertTrue(tc.type_check())

    def test_type_check_reshape_dyn_true_param_false(self):
        # [FUNCTION DOC] Reshaping an unknown by five type to one by two by inferred is rejected;
        # expects type_check to raise TypeError.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that reshapes its input using an inferred last dimension.
            def forward(self, x: TensorType((Dyn, 5))):
                # [FUNCTION DOC] HELPER: Returns x reshaped to one by two by inferred.
                return torch.reshape(x, [1, 2, -1])

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        with self.assertRaises(TypeError):
            tc.type_check()

    def test_type_check_transpose_true(self):
        # [FUNCTION DOC] Type checks swapping dims zero and one of a four dimensional type.
        # type_check returns true and call_function and output nodes are expected to have the
        # swapped type.
        # KNOWN ISSUE: the third branch tests a node op equal to the string x, which never occurs,
        #     so that placeholder check is dead code.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that transposes dims zero and one of its input.
            def forward(self, x: TensorType((1, 2, 3, 5))):
                # [FUNCTION DOC] HELPER: Returns x transposed over dims zero and one.
                return torch.transpose(x, 0, 1)

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        self.assertTrue(tc.type_check())

        for n in symbolic_traced.graph.nodes:
            if n.op == "call_function":
                if n.type != TensorType([2, 1, 3, 5]):
                    raise AssertionError(
                        f"Expected n.type == TensorType([2, 1, 3, 5]), got {n.type}"
                    )
            if n.op == "output":
                if n.type != TensorType([2, 1, 3, 5]):
                    raise AssertionError(
                        f"Expected n.type == TensorType([2, 1, 3, 5]), got {n.type}"
                    )
            if n.op == "x":
                if n.placeholder != TensorType([1, 2, 3, 5]):
                    raise AssertionError(
                        f"Expected n.placeholder == TensorType([1, 2, 3, 5]), "
                        f"got {n.placeholder}"
                    )

    def test_type_check_transpose_False(self):
        # [FUNCTION DOC] Transposing with dimension ten on a four dimensional type is out of
        # range; expects type_check to raise TypeError.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that transposes dims zero and ten of its input.
            def forward(self, x: TensorType((1, 2, 3, 5))):
                # [FUNCTION DOC] HELPER: Returns x transposed over dims zero and ten.
                return torch.transpose(x, 0, 10)

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        with self.assertRaises(TypeError):
            tc.type_check()

    def test_type_check_batch_norm_2D(self):
        # [FUNCTION DOC] Traces with RewritingTracer (so annotations in the body are kept) a block
        # applying batch norm to an input of shape two by two by five by four, with an
        # intermediate annotation having an unknown third dimension. After type_check all
        # placeholder, output, call_module and call_function nodes must have the full input shape.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: Block with a batch norm layer; forward applies it and adds the
            # input back.
            def __init__(self, inplanes, planes):
                # [FUNCTION DOC] HELPER: Stores a BatchNorm2d layer sized by planes as bn1;
                # inplanes is unused.
                super().__init__()
                norm_layer = torch.nn.BatchNorm2d
                self.bn1 = norm_layer(planes)

            def forward(self, x: TensorType((2, 2, 5, 4))):
                # [FUNCTION DOC] HELPER: Applies bn1 to x, adds x to the result in place and
                # returns it; the result carries an intermediate type annotation.
                identity = x
                out: TensorType((2, 2, Dyn, 4)) = self.bn1(x)
                out += identity
                return out

        B = BasicBlock(2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        tc = GraphTypeChecker({}, traced)
        tc.type_check()

        for n in graph.nodes:
            if n.op == "placeholder":
                if n.type != TensorType((2, 2, 5, 4)):
                    raise AssertionError(
                        f"Expected n.type == TensorType((2, 2, 5, 4)), got {n.type}"
                    )
            if n.op == "output":
                if n.type != TensorType((2, 2, 5, 4)):
                    raise AssertionError(
                        f"Expected n.type == TensorType((2, 2, 5, 4)), got {n.type}"
                    )
            if n.op == "call_module":
                if n.type != TensorType((2, 2, 5, 4)):
                    raise AssertionError(
                        f"Expected n.type == TensorType((2, 2, 5, 4)), got {n.type}"
                    )
            if n.op == "call_function":
                if n.type != TensorType((2, 2, 5, 4)):
                    raise AssertionError(
                        f"Expected n.type == TensorType((2, 2, 5, 4)), got {n.type}"
                    )

    def test_type_check_batch_norm_2D_false(self):
        # [FUNCTION DOC] Same batch norm block but the input is annotated as rank three while the
        # intermediate annotation is rank four; expects type_check to raise TypeError.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: Block with a batch norm layer; forward applies it and adds the
            # input back.
            def __init__(self, inplanes, planes):
                # [FUNCTION DOC] HELPER: Stores a BatchNorm2d layer sized by planes as bn1;
                # inplanes is unused.
                super().__init__()
                norm_layer = torch.nn.BatchNorm2d
                self.bn1 = norm_layer(planes)

            def forward(self, x: TensorType((2, 2, 5))):
                # [FUNCTION DOC] HELPER: Applies bn1 to x, adds x to the result in place and
                # returns it.
                identity = x
                out: TensorType((2, 2, Dyn, 4)) = self.bn1(x)
                out += identity
                return out

        B = BasicBlock(2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        tc = GraphTypeChecker({}, traced)
        with self.assertRaises(TypeError):
            tc.type_check()

    def test_type_check_batch_norm_2D_broadcast(self):
        # [FUNCTION DOC] Batch norm block with an unknown input type. After type_check,
        # placeholder, call_function and output nodes are expected to be a rank four type of all
        # unknown dims and the call_module node the annotated type. Then a block with one channel
        # (mismatching the annotated two) must raise TypeError.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: Block with a batch norm layer; forward applies it and adds the
            # input back.
            def __init__(self, inplanes, planes):
                # [FUNCTION DOC] HELPER: Stores a BatchNorm2d layer sized by planes as bn1;
                # inplanes is unused.
                super().__init__()
                norm_layer = torch.nn.BatchNorm2d
                self.bn1 = norm_layer(planes)

            def forward(self, x: Dyn):
                # [FUNCTION DOC] HELPER: Applies bn1 to x, adds x to the result in place and
                # returns it.
                identity = x
                out: TensorType((2, 2, Dyn, 4)) = self.bn1(x)
                out += identity
                return out

        B = BasicBlock(2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        tc = GraphTypeChecker({}, traced)
        tc.type_check()
        for n in graph.nodes:
            if n.op == "placeholder":
                if n.type != TensorType((Dyn, Dyn, Dyn, Dyn)):
                    raise AssertionError(
                        f"Expected n.type == TensorType((Dyn, Dyn, Dyn, Dyn)), "
                        f"got {n.type}"
                    )
            if n.op == "call_function":
                if n.type != TensorType((Dyn, Dyn, Dyn, Dyn)):
                    raise AssertionError(
                        f"Expected n.type == TensorType((Dyn, Dyn, Dyn, Dyn)), "
                        f"got {n.type}"
                    )
            if n.op == "output":
                if n.type != TensorType((Dyn, Dyn, Dyn, Dyn)):
                    raise AssertionError(
                        f"Expected n.type == TensorType((Dyn, Dyn, Dyn, Dyn)), "
                        f"got {n.type}"
                    )
            if n.op == "call_module":
                if n.type != TensorType((2, 2, Dyn, 4)):
                    raise AssertionError(
                        f"Expected n.type == TensorType((2, 2, Dyn, 4)), got {n.type}"
                    )

        B = BasicBlock(1, 1)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        tc = GraphTypeChecker({}, traced)
        with self.assertRaises(TypeError):
            tc.type_check()

    def test_type_check_conv2D(self):
        # [FUNCTION DOC] Same checks as the broadcast batch norm test but with a 3x3 convolution
        # layer as the annotated intermediate: nodes are all unknown rank four types except the
        # call_module node, which has the annotated type.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: Block with a 3x3 convolution and batch norm; forward uses the
            # convolution.
            def __init__(self, inplanes, planes, stride=1):
                # [FUNCTION DOC] HELPER: Builds conv1 through the conv3x3 helper and bn1 as
                # BatchNorm2d.
                super().__init__()
                norm_layer = torch.nn.BatchNorm2d
                self.conv1 = conv3x3(inplanes, planes, stride)
                self.bn1 = norm_layer(planes)

            def forward(self, x: Dyn):
                # [FUNCTION DOC] HELPER: Applies conv1 to x, adds x to the result in place and
                # returns it.
                identity = x
                out: TensorType((2, 2, Dyn, 4)) = self.conv1(x)
                out += identity
                return out

        B = BasicBlock(2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        tc = GraphTypeChecker({}, traced)
        tc.type_check()
        for n in graph.nodes:
            if n.op == "placeholder":
                if n.type != TensorType((Dyn, Dyn, Dyn, Dyn)):
                    raise AssertionError(
                        f"Expected n.type == TensorType((Dyn, Dyn, Dyn, Dyn)), "
                        f"got {n.type}"
                    )
            if n.op == "call_function":
                if n.type != TensorType((Dyn, Dyn, Dyn, Dyn)):
                    raise AssertionError(
                        f"Expected n.type == TensorType((Dyn, Dyn, Dyn, Dyn)), "
                        f"got {n.type}"
                    )
            if n.op == "output":
                if n.type != TensorType((Dyn, Dyn, Dyn, Dyn)):
                    raise AssertionError(
                        f"Expected n.type == TensorType((Dyn, Dyn, Dyn, Dyn)), "
                        f"got {n.type}"
                    )
            if n.op == "call_module":
                if n.type != TensorType((2, 2, Dyn, 4)):
                    raise AssertionError(
                        f"Expected n.type == TensorType((2, 2, Dyn, 4)), got {n.type}"
                    )

    def test_type_check_conv2D_2(self):
        # [FUNCTION DOC] Convolution block with input shape five by two by three by four. Runs the
        # real forward for the expected output shape; after type_check all placeholder,
        # call_function and call_module nodes have the input type and the output type dims equal
        # the runtime shape. A block built with mismatched channels (one in, two out) must raise
        # TypeError.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: Block with a 3x3 convolution and batch norm; forward uses the
            # convolution.
            def __init__(self, inplanes, planes, stride=1):
                # [FUNCTION DOC] HELPER: Builds conv1 through the conv3x3 helper and bn1 as
                # BatchNorm2d.
                super().__init__()
                norm_layer = torch.nn.BatchNorm2d
                self.conv1 = conv3x3(inplanes, planes, stride)
                self.bn1 = norm_layer(planes)

            def forward(self, x: TensorType((5, 2, 3, 4))):
                # [FUNCTION DOC] HELPER: Applies conv1 to x, adds x to the result in place and
                # returns it.
                identity = x
                out = self.conv1(x)
                out += identity
                return out

        B = BasicBlock(2, 2)
        b = B.forward(torch.rand(5, 2, 3, 4))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        tc = GraphTypeChecker({}, traced)
        tc.type_check()
        t = TensorType((5, 2, 3, 4))
        for n in graph.nodes:
            if n.op == "placeholder":
                if n.type != t:
                    raise AssertionError(f"Expected n.type == {t}, got {n.type}")
            if n.op == "call_function":
                if n.type != t:
                    raise AssertionError(f"Expected n.type == {t}, got {n.type}")
            if n.op == "output":
                if torch.Size(n.type.dims) != b.shape:
                    raise AssertionError(
                        f"Expected torch.Size(n.type.dims) == {b.shape}, "
                        f"got {torch.Size(n.type.dims)}"
                    )
            if n.op == "call_module":
                if n.type != t:
                    raise AssertionError(f"Expected n.type == {t}, got {n.type}")

        B = BasicBlock(1, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        tc = GraphTypeChecker({}, traced)
        with self.assertRaises(TypeError):
            tc.type_check()

    def test_type_check_conv2D_2_fully_static(self):
        # [FUNCTION DOC] Table driven check over five configurations of annotation, runtime input
        # shape, channels, stride, groups, dilation, padding and kernel size. For each, a single
        # convolution module is traced, its placeholder annotated and type checked; the output
        # type must be consistent with the real output size. A second module also gets an
        # intermediate annotation on call_module nodes and its output type must equal the expected
        # table entry and stay consistent.
        annotation_list = [
            (1, 2, 3, 5),
            (2, 5, 6, 9),
            (10, 15, 13, 14),
            (10, Dyn, 13, 14),
            (Dyn, Dyn, Dyn, 3),
        ]
        input_list = [
            (1, 2, 3, 5),
            (2, 5, 6, 9),
            (10, 15, 13, 14),
            (10, 15, 13, 14),
            (1, 2, 2, 3),
        ]
        intermediate_types = [
            (1, Dyn, Dyn, 7),
            (2, Dyn, 4, 6),
            (10, 15, Dyn, 5),
            (10, 15, 7, 7),
            (1, Dyn, Dyn, Dyn),
        ]
        in_planes_list = [2, 5, 15, 15, 2]
        stride_list = [1, 2, 3, 2, 2]
        out_planes_list = [2, 5, 15, 15, 2]
        groups_list = [1, 5, 5, 5, 2]
        dilation_list = [1, 2, 3, 3, 3]
        padding_list = [1, 2, 3, 3, 3]
        kernel_size_list = [1, 2, 3, 3, 3]
        output_types = [
            (1, 2, Dyn, 7),
            (2, 5, 4, 6),
            (10, 15, Dyn, 5),
            (10, 15, 7, 7),
            (1, 2, Dyn, Dyn),
        ]

        for i in range(5):
            annotation = annotation_list[i]
            input = input_list[i]
            in_planes = in_planes_list[i]
            stride = stride_list[i]
            out_planes = out_planes_list[i]
            groups = groups_list[i]
            dilation = dilation_list[i]
            padding = padding_list[i]
            kernel_size = kernel_size_list[i]
            intermediate_type = intermediate_types[i]

            class BasicBlock(torch.nn.Module):
                # [CLASS DOC] HELPER: Module holding one convolution configured from the loop
                # variables.
                def __init__(
                    self,
                    in_planes,
                    out_planes,
                    kernel_size,
                    stride,
                    padding,
                    groups,
                    dilation,
                ):
                    # [FUNCTION DOC] HELPER: Creates conv1 as a bias-free Conv2d from the given
                    # channel counts, kernel size, stride, padding, groups and dilation.
                    super().__init__()
                    self.conv1 = torch.nn.Conv2d(
                        in_channels=in_planes,
                        out_channels=out_planes,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        groups=groups,
                        bias=False,
                        dilation=dilation,
                    )

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Returns conv1 applied to x.
                    out = self.conv1(x)
                    return out

            B = BasicBlock(
                in_planes, out_planes, kernel_size, stride, padding, groups, dilation
            )
            ast_rewriter = RewritingTracer()
            graph = ast_rewriter.trace(B)
            traced = GraphModule(ast_rewriter.root, graph, "gm")

            # annotate our argument
            for n in graph.nodes:
                if n.op == "placeholder":
                    n.type = TensorType(annotation)

            b = B.forward(torch.rand(input))
            tc = GraphTypeChecker({}, traced)
            tc.type_check()

            for n in graph.nodes:
                if n.op == "output":
                    if not is_consistent(n.type, TensorType(b.size())):
                        raise AssertionError(
                            f"Expected n.type consistent with TensorType({b.size()})"
                        )

            # test with intermediate annotations
            class BasicBlock(torch.nn.Module):
                # [CLASS DOC] HELPER: Same single convolution module, redefined for the
                # intermediate annotation variant of the test.
                def __init__(
                    self,
                    in_planes,
                    out_planes,
                    kernel_size,
                    stride,
                    padding,
                    groups,
                    dilation,
                ):
                    # [FUNCTION DOC] HELPER: Creates conv1 as a bias-free Conv2d from the given
                    # channel counts, kernel size, stride, padding, groups and dilation.
                    super().__init__()
                    self.conv1 = torch.nn.Conv2d(
                        in_channels=in_planes,
                        out_channels=out_planes,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        groups=groups,
                        bias=False,
                        dilation=dilation,
                    )

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Returns conv1 applied to x.
                    out = self.conv1(x)
                    return out

            B = BasicBlock(
                in_planes, out_planes, kernel_size, stride, padding, groups, dilation
            )
            ast_rewriter = RewritingTracer()
            graph = ast_rewriter.trace(B)
            traced = GraphModule(ast_rewriter.root, graph, "gm")

            # populate our intermediate notes
            for n in traced.graph.nodes:
                if n.op == "call_module":
                    n.type = TensorType(intermediate_type)

            tc = GraphTypeChecker({}, traced)
            tc.type_check()

            for n in traced.graph.nodes:
                if n.op == "output":
                    if n.type != TensorType(output_types[i]):
                        raise AssertionError(
                            f"Expected n.type == TensorType({output_types[i]}), "
                            f"got {n.type}"
                        )
                    if not is_consistent(n.type, TensorType(b.size())):
                        raise AssertionError(
                            f"Expected n.type consistent with TensorType({b.size()})"
                        )

    def test_typecheck_basicblock(self):
        # [FUNCTION DOC] Type checks a ResNet style basic block on an annotated input of shape two
        # by two by four by five, then asserts the output node type has dims equal to the size of
        # the real forward result.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: ResNet basic block: two 3x3 convolutions with batch norm and
            # relu, with an optional downsample branch added to the residual. Only stride one,
            # groups one, base width 64 and dilation one are supported.
            expansion = 1

            def __init__(
                self,
                inplanes,
                planes,
                stride=1,
                downsample=None,
                groups=1,
                base_width=64,
                dilation=1,
            ):
                # [FUNCTION DOC] HELPER: Builds the convolutions, batch norms, relu, optional
                # downsample and stride. Raises ValueError if groups is not one or base_width is
                # not 64, and NotImplementedError if dilation is above one.
                super().__init__()
                norm_layer = torch.nn.BatchNorm2d
                if groups != 1 or base_width != 64:
                    raise ValueError(
                        "BasicBlock only supports groups=1 and base_width=64"
                    )
                if dilation > 1:
                    raise NotImplementedError(
                        "Dilation > 1 not supported in BasicBlock"
                    )
                # Both self.conv1 and self.downsample layers downsample the input when stride != 1
                self.conv1 = conv3x3(inplanes, planes, stride)
                self.bn1 = norm_layer(planes)
                self.relu = torch.nn.ReLU(inplace=True)
                self.conv2 = conv3x3(planes, planes)
                self.bn2 = norm_layer(planes)
                self.downsample = downsample
                self.stride = stride

            def forward(self, x: TensorType((2, 2, 4, 5))):
                # [FUNCTION DOC] HELPER: Runs conv1, bn1, relu, conv2, bn2, adds the identity (or
                # the downsample of x if present), applies relu and returns the result.
                identity = x

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity
                out = self.relu(out)

                return out

        B = BasicBlock(2, 2)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        tc = GraphTypeChecker({}, traced)
        tc.type_check()

        for n in traced.graph.nodes:
            if n.target == "output":
                if not isinstance(n.type, TensorType):
                    raise AssertionError(
                        f"Expected n.type to be TensorType, got {type(n.type)}"
                    )
                expected_size = B.forward(torch.rand(2, 2, 4, 5)).size()
                if torch.Size(n.type.dims) != expected_size:
                    raise AssertionError(
                        f"Expected torch.Size(n.type.dims) == {expected_size}, "
                        f"got {torch.Size(n.type.dims)}"
                    )

    def test_type_check_conv2D_maxpool2d_flatten(self):
        # [FUNCTION DOC] Type checks a block of convolution, max pool, convolution, max pool,
        # linear, adaptive average pool and flatten on a four by three by 32 by 32 input. After
        # eliminating dead code, every remaining node must match the expected list of tensor types
        # layer by layer.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: Block chaining conv, pool, conv, pool, a linear layer on the
            # last dim, adaptive average pooling and flatten.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates conv1, pool, conv2, fc1 and pool2 layers.
                super().__init__()

                self.conv1 = torch.nn.Conv2d(3, 6, 5)
                self.pool = torch.nn.MaxPool2d(2, 2)
                self.conv2 = torch.nn.Conv2d(6, 16, 5)
                self.fc1 = torch.nn.Linear(5, 120)
                self.pool2 = torch.nn.AdaptiveAvgPool2d((6, 7))

            def forward(self, x: TensorType((4, 3, 32, 32))):
                # [FUNCTION DOC] HELPER: Applies the layers in order and flattens from dim one;
                # returns the result.
                out = self.conv1(x)
                out = self.pool(out)
                out = self.conv2(out)
                out = self.pool(out)
                out = self.fc1(out)
                out = self.pool2(out)
                out = torch.flatten(out, 1)
                return out

        B = BasicBlock()
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        tc = GraphTypeChecker({}, traced)
        tc.type_check()

        expected_ph_types = [
            TensorType((4, 3, 32, 32)),
            TensorType((4, 6, 28, 28)),
            TensorType((4, 6, 14, 14)),
            TensorType((4, 16, 10, 10)),
            TensorType((4, 16, 5, 5)),
            TensorType((4, 16, 5, 120)),
            TensorType((4, 16, 6, 7)),
            TensorType((4, 672)),
            TensorType((4, 672)),
        ]

        expected_iter = iter(expected_ph_types)
        traced.graph.eliminate_dead_code()

        for n in traced.graph.nodes:
            expected = next(expected_iter)
            if n.type != expected:
                raise AssertionError(f"Expected n.type == {expected}, got {n.type}")

    def test_type_check_flatten(self):
        # [FUNCTION DOC] Type checks flattening dims one to two of a five dimensional type; the
        # output node type must be the merged rank four type with the product of the flattened
        # sizes.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that flattens dims one through two of its input.
            def forward(self, x: TensorType((1, 2, 3, 5, Dyn))):
                # [FUNCTION DOC] HELPER: Returns x flattened over dims one to two.
                return torch.flatten(x, 1, 2)

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        tc.type_check()
        for n in symbolic_traced.graph.nodes:
            if n.op == "output":
                if n.type != TensorType((1, 6, 5, Dyn)):
                    raise AssertionError(
                        f"Expected TensorType((1, 6, 5, Dyn)), got {n.type}"
                    )

    def test_type_check_flatten_2(self):
        # [FUNCTION DOC] Like the previous flatten test but with an unknown size inside the
        # flattened range; the merged dimension must come out as unknown.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that flattens dims one through two of its input.
            def forward(self, x: TensorType((1, Dyn, 3, 5, Dyn))):
                # [FUNCTION DOC] HELPER: Returns x flattened over dims one to two.
                return torch.flatten(x, 1, 2)

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        tc.type_check()
        for n in symbolic_traced.graph.nodes:
            if n.op == "output":
                if n.type != TensorType((1, Dyn, 5, Dyn)):
                    raise AssertionError(
                        f"Expected TensorType((1, Dyn, 5, Dyn)), got {n.type}"
                    )

    def test_type_check_flatten3(self):
        # [FUNCTION DOC] Flattening dims one to three of a two by three by four by five type must
        # give a two by sixty output type. Then runs Refine and asserts the collected constraints
        # are exactly one equality between two and two.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that flattens from start dim one to end dim three.
            def forward(self, x: TensorType((2, 3, 4, 5))):
                # [FUNCTION DOC] HELPER: Returns x flattened from dim one to dim three.
                return torch.flatten(x, start_dim=1, end_dim=3)

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        tc.type_check()
        for n in symbolic_traced.graph.nodes:
            if n.op == "output":
                if n.type != TensorType((2, 60)):
                    raise AssertionError(f"Expected TensorType((2, 60)), got {n.type}")
        r = Refine(symbolic_traced)
        r.refine()
        c = r.constraints
        if c != [Equality(2, 2)]:
            raise AssertionError(f"Expected [Equality(2, 2)], got {c}")

    def test_type_typechecl_maxpool2d_3dinput(self):
        # [FUNCTION DOC] Type checks a 2d max pool with window five and stride eight on a three
        # dimensional input of 64 by 8 by 8; the output node type must be 64 by 1 by 1.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: Block with a single max pool layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the pool layer with kernel size five and stride
                # eight.
                super().__init__()
                self.pool = torch.nn.MaxPool2d(5, 8)

            def forward(self, x: TensorType((64, 8, 8))):
                # [FUNCTION DOC] HELPER: Returns pool applied to x.
                out = self.pool(x)
                return out

        B = BasicBlock()
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        tc = GraphTypeChecker({}, traced)
        tc.type_check()

        for n in traced.graph.nodes:
            if n.target == "output":
                if n.type != TensorType((64, 1, 1)):
                    raise AssertionError(
                        f"Expected TensorType((64, 1, 1)), got {n.type}"
                    )

    def test_type_maxpool2d_fully_static(self):
        # [FUNCTION DOC] Table driven check over five max pool configurations (annotation, input
        # shape, stride, dilation, padding, kernel size). Each is traced with the placeholder
        # annotated and checked for output consistency with the real result, then redone with an
        # intermediate type set on call_module nodes and compared with the expected output type.
        annotation_list = [
            (Dyn, Dyn, 3, 5),
            (2, 5, 6, 9),
            (10, 15, 13, 14),
            (10, Dyn, 13, 14),
            (Dyn, Dyn, Dyn, 10),
        ]
        input_list = [
            (1, 2, 3, 5),
            (2, 5, 6, 9),
            (10, 15, 13, 14),
            (10, 15, 13, 14),
            (2, 2, 10, 10),
        ]
        intermediate_types = [
            (1, 2, Dyn, Dyn),
            (2, Dyn, 2, 4),
            (10, 15, Dyn, 2),
            (10, 15, 2, 3),
            (2, Dyn, Dyn, Dyn),
        ]
        stride_list = [1, 2, 3, 2, 1]
        dilation_list = [1, 2, 3, 3, 2]
        padding_list = [1, 2, 3, 3, 1]
        kernel_size_list = [2, 4, 6, 6, 3]
        output_types = [
            (1, 2, 4, 6),
            (2, 5, 2, 4),
            (10, 15, 2, 2),
            (10, 15, 2, 3),
            (2, Dyn, Dyn, 8),
        ]

        for i in range(5):
            annotation = annotation_list[i]
            input = input_list[i]
            stride = stride_list[i]
            dilation = dilation_list[i]
            padding = padding_list[i]
            kernel_size = kernel_size_list[i]
            intermediate_type = intermediate_types[i]

            class BasicBlock(torch.nn.Module):
                # [CLASS DOC] HELPER: Module holding one max pool layer built from the loop
                # variables.
                def __init__(self, kernel_size, stride, padding, dilation):
                    # [FUNCTION DOC] HELPER: Creates pool as a max pool with the given kernel
                    # size, stride, padding and dilation, without indices and without ceil mode.
                    super().__init__()
                    self.pool = torch.nn.MaxPool2d(
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        return_indices=False,
                        ceil_mode=False,
                    )

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Returns pool applied to x.
                    out = self.pool(x)
                    return out

            B = BasicBlock(kernel_size, stride, padding, dilation)
            ast_rewriter = RewritingTracer()
            graph = ast_rewriter.trace(B)
            traced = GraphModule(ast_rewriter.root, graph, "gm")

            # annotate our argument
            for n in graph.nodes:
                if n.op == "placeholder":
                    n.type = TensorType(annotation)

            b = B.forward(torch.rand(input))
            tc = GraphTypeChecker({}, traced)
            tc.type_check()

            for n in graph.nodes:
                if n.op == "output":
                    if not is_consistent(n.type, TensorType(b.size())):
                        raise AssertionError(
                            f"Expected n.type consistent with TensorType({b.size()})"
                        )

            # test with intermediate annotations
            class BasicBlock(torch.nn.Module):
                # [CLASS DOC] HELPER: Same max pool module, redefined for the intermediate
                # annotation variant.
                def __init__(self, kernel_size, stride, padding, dilation):
                    # [FUNCTION DOC] HELPER: Creates pool as a max pool with the given kernel
                    # size, stride, padding and dilation, without indices and without ceil mode.
                    super().__init__()
                    self.pool = torch.nn.MaxPool2d(
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        return_indices=False,
                        ceil_mode=False,
                    )

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Returns pool applied to x.
                    out = self.pool(x)
                    return out

            B = BasicBlock(kernel_size, stride, padding, dilation)
            ast_rewriter = RewritingTracer()
            graph = ast_rewriter.trace(B)
            traced = GraphModule(ast_rewriter.root, graph, "gm")

            # annotate our argument
            for n in graph.nodes:
                if n.op == "placeholder":
                    n.type = TensorType(annotation)

            # populate our intermediate notes
            for n in traced.graph.nodes:
                if n.op == "call_module":
                    n.type = TensorType(intermediate_type)

            tc = GraphTypeChecker({}, traced)
            tc.type_check()

            for n in traced.graph.nodes:
                if n.op == "output":
                    if n.type != TensorType(output_types[i]):
                        raise AssertionError(
                            f"Expected n.type == TensorType({output_types[i]}), "
                            f"got {n.type}"
                        )
                    if not is_consistent(n.type, TensorType(b.size())):
                        raise AssertionError(
                            f"Expected n.type consistent with TensorType({b.size()})"
                        )

    def test_flatten_fully_static(self):
        # [FUNCTION DOC] Table driven check over five flatten configurations (start and end dims,
        # with annotations from fully unknown to partly known). For each, the output type after
        # type_check must be consistent with the size of the real forward result. The intermediate
        # list in the test is unused.
        annotation_list = [
            Dyn,
            TensorType((2, 5, 6, 9)),
            TensorType((10, 15, 13, 14)),
            TensorType((10, Dyn, 13, 14)),
            TensorType((Dyn, Dyn, Dyn, 10)),
        ]
        input_list = [
            (1, 2, 3, 5),
            (2, 5, 6, 9),
            (10, 15, 13, 14),
            (10, 15, 13, 14),
            (2, 2, 10, 10),
        ]

        intermediate_list = [  # noqa: F841
            Dyn,
            (2, 5, 6, 9),
            (10, 15, 13, 14),
            (10, 15, 13, 14),
            (2, 2, 10, 10),
        ]

        start_dim = [1, 2, 1, 2, 0]
        end_dim = [1, 3, 3, 3, -2]

        for i in range(5):
            annotation = annotation_list[i]
            input = input_list[i]
            # intermediate_type = intermediate_list[i]

            class BasicBlock(torch.nn.Module):
                # [CLASS DOC] HELPER: Module that flattens its input between stored start and end
                # dims.
                def __init__(self, start, end):
                    # [FUNCTION DOC] HELPER: Stores the start and end dims.
                    super().__init__()
                    self.start = start
                    self.end = end

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Returns x flattened between the stored dims.
                    out = torch.flatten(x, self.start, self.end)
                    return out

            B = BasicBlock(start_dim[i], end_dim[i])
            ast_rewriter = RewritingTracer()
            graph = ast_rewriter.trace(B)
            traced = GraphModule(ast_rewriter.root, graph, "gm")

            # annotate our argument
            for n in graph.nodes:
                if n.op == "placeholder":
                    n.type = annotation

            b = B.forward(torch.rand(input))
            tc = GraphTypeChecker({}, traced)
            tc.type_check()

            for n in graph.nodes:
                if n.op == "output":
                    if not is_consistent(n.type, TensorType(b.size())):
                        raise AssertionError(
                            f"Expected n.type consistent with TensorType({b.size()})"
                        )

    @skipIfNoTorchVision
    def test_resnet50(self):
        # [FUNCTION DOC] Skipped without torchvision. Runs shape propagation on a traced resnet50
        # as the reference, then type checks a copy with all node types cleared and checks each
        # node type is consistent with the real shape. A second copy whose placeholder is typed to
        # the sample input must match the real shapes exactly. Finally runs symbolic inference and
        # asserts every node has a tensor type and all nodes share a single batch size entry.
        gm_run = symbolic_trace(resnet50())
        sample_input = torch.randn(1, 3, 224, 224)

        # run our nodes
        ShapeProp(gm_run).propagate(sample_input)

        gm_static = symbolic_trace(resnet50())

        for n in gm_static.graph.nodes:
            n.type = None

        g = GraphTypeChecker({}, gm_static)
        g.type_check()
        gm_static.graph.eliminate_dead_code()
        gm_run.graph.eliminate_dead_code()
        # here we are checking for consistency with fully dynamic nodes
        for n1, n2 in zip(gm_static.graph.nodes, gm_run.graph.nodes):
            if not is_consistent(n1.type, TensorType(n2.meta["tensor_meta"].shape)):
                raise AssertionError(
                    f"Expected n1.type consistent with "
                    f"TensorType({n2.meta['tensor_meta'].shape})"
                )

        # here we give the same input as to runtime
        gm_static_with_types = symbolic_trace(resnet50())

        # we initialize our placeholder
        for n in gm_static_with_types.graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType((1, 3, 224, 224))

        g = GraphTypeChecker({}, gm_static_with_types)
        g.type_check()
        for n1, n2 in zip(gm_static_with_types.graph.nodes, gm_run.graph.nodes):
            expected_type = TensorType(n2.meta["tensor_meta"].shape)
            if n1.type != expected_type:
                raise AssertionError(
                    f"Expected n1.type == {expected_type}, got {n1.type}"
                )

        # apply shape inference to graph and check
        # that the batch size is equal across all layers
        infer_symbolic_types(gm_static)

        batch_sizes = set()
        gm_static.graph.eliminate_dead_code()
        for n in gm_static.graph.nodes:
            if not isinstance(n.type, TensorType):
                raise AssertionError(
                    f"Expected n.type to be TensorType, got {type(n.type)}"
                )
            batch_sizes.add(n.type.dims[0])
        if len(batch_sizes) != 1:
            raise AssertionError(
                f"Expected len(batch_sizes) == 1, got {len(batch_sizes)}"
            )

    def test_type_check_batch_norm_symbolic(self):
        # [FUNCTION DOC] Type checks the batch norm block with an unknown input, then runs
        # symbolic type inference and asserts that each node type becomes a rank four type with
        # one shared symbolic third dimension (the test hard-codes the generated symbol name).
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: Block with a batch norm layer; forward applies it and adds the
            # input back.
            def __init__(self, inplanes, planes):
                # [FUNCTION DOC] HELPER: Stores a BatchNorm2d layer sized by planes as bn1;
                # inplanes is unused.
                super().__init__()
                norm_layer = torch.nn.BatchNorm2d
                self.bn1 = norm_layer(planes)

            def forward(self, x: Dyn):
                # [FUNCTION DOC] HELPER: Applies bn1 to x, adds x to the result in place and
                # returns it.
                identity = x
                out: TensorType((2, 2, Dyn, 4)) = self.bn1(x)
                out += identity
                return out

        B = BasicBlock(2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        tc = GraphTypeChecker({}, traced)
        tc.type_check()

        infer_symbolic_types(traced)

        my_types = iter(
            [
                TensorType[(2, 2, sympy.symbols("~7"), 4)],
                TensorType[(2, 2, sympy.symbols("~7"), 4)],
                TensorType[(2, 2, sympy.symbols("~7"), 4)],
                TensorType[(2, 2, sympy.symbols("~7"), 4)],
            ]
        )

        for n in graph.nodes:
            expected = next(my_types)
            if n.type != expected:
                raise AssertionError(f"Expected n.type == {expected}, got {n.type}")

    def test_symbolic_add_with_broadcast(self):
        # [FUNCTION DOC] Type checks an add with broadcasting, then symbolic inference and Refine.
        # The collected constraints must be equalities for the matching dims one, two and three
        # (none for the unknown vs four pair, since the unknown could be one or four). Node types
        # after inference must use shared symbolic names for the unknown dimension.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that adds two annotated inputs.
            def forward(self, x: TensorType((1, 2, 3, Dyn)), y: TensorType((2, 3, 4))):
                # [FUNCTION DOC] HELPER: Returns the sum of x and y.
                return torch.add(x, y)

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        tc.type_check()
        infer_symbolic_types(symbolic_traced)
        r = Refine(symbolic_traced)
        r.refine()

        expected_constraints = [Equality(1, 1), Equality(2, 2), Equality(3, 3)]
        if r.constraints != expected_constraints:
            raise AssertionError(
                f"Expected r.constraints == {expected_constraints}, got {r.constraints}"
            )
        # note that there is no equality constraint between dyn and 4 because
        # dyn could be 4 or 1

        infer_symbolic_types(symbolic_traced)

        expected_ph_types = [
            TensorType((1, 2, 3, sympy.symbols("~0"))),
            TensorType((2, 3, 4)),
            TensorType((1, 2, 3, sympy.symbols("~1"))),
            TensorType((1, 2, 3, sympy.symbols("~1"))),
        ]
        expected_iter = iter(expected_ph_types)

        for n in symbolic_traced.graph.nodes:
            expected = next(expected_iter)
            if n.type != expected:
                raise AssertionError(f"Expected n.type == {expected}, got {n.type}")

    def test_symbolic_add_with_broadcast_2(self):
        # [FUNCTION DOC] Symbolic inference for adding a one by two type to an unknown by two
        # type. The second input, the add and the output must all get the same symbolic first
        # dimension.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that adds two annotated inputs.
            def forward(self, x: TensorType((1, 2)), y: TensorType((Dyn, 2))):
                # [FUNCTION DOC] HELPER: Returns the sum of x and y.
                return torch.add(x, y)

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        tc.type_check()
        infer_symbolic_types(symbolic_traced)
        r = Refine(symbolic_traced)
        r.refine()

        expected_ph_types = [
            TensorType((1, 2)),
            TensorType((sympy.symbols("~1"), 2)),
            TensorType((sympy.symbols("~1"), 2)),
            TensorType((sympy.symbols("~1"), 2)),
        ]
        expected_iter = iter(expected_ph_types)

        for n in symbolic_traced.graph.nodes:
            expected = next(expected_iter)
            if n.type != expected:
                raise AssertionError(f"Expected n.type == {expected}, got {n.type}")

    def test_type_check_conv2D_types(self):
        # [FUNCTION DOC] Type checks the convolution block and infers symbolic types; the height
        # and width dims of each call_module node type must be floor expressions from sympy.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: Block with a 3x3 convolution and batch norm; forward uses the
            # convolution.
            def __init__(self, inplanes, planes, stride=1):
                # [FUNCTION DOC] HELPER: Builds conv1 through the conv3x3 helper and bn1 as
                # BatchNorm2d.
                super().__init__()
                norm_layer = torch.nn.BatchNorm2d
                self.conv1 = conv3x3(inplanes, planes, stride)
                self.bn1 = norm_layer(planes)

            def forward(self, x: Dyn):
                # [FUNCTION DOC] HELPER: Applies conv1 to x, adds x to the result in place and
                # returns it.
                identity = x
                out: TensorType((2, 2, Dyn, 4)) = self.conv1(x)
                out += identity
                return out

        B = BasicBlock(2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        tc = GraphTypeChecker({}, traced)
        tc.type_check()
        infer_symbolic_types(traced)

        for n in traced.graph.nodes:
            if n.op == "call_module":
                if not isinstance(n.type.dims[2], sympy.floor):
                    raise AssertionError(
                        f"Expected n.type.dims[2] to be sympy.floor, "
                        f"got {type(n.type.dims[2])}"
                    )
                if not isinstance(n.type.dims[3], sympy.floor):
                    raise AssertionError(
                        f"Expected n.type.dims[3] to be sympy.floor, "
                        f"got {type(n.type.dims[3])}"
                    )

    def test_type_check_symbolic_inferenceconv2D_maxpool2d_flatten(self):
        # [FUNCTION DOC] Type checks the conv, pool, conv, pool, linear, adaptive pool, flatten
        # block with unknown height and width, using plain symbolic tracing, then infers symbolic
        # types and asserts conv1 and conv2 outputs have floor expressions of symbols minus four
        # for height and width. The unused RewritingTracer variable is silenced with a lint
        # exemption.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: Block chaining conv, pool, conv, pool, a linear layer on the
            # last dim, adaptive average pooling and flatten.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates conv1, pool, conv2, fc1 and pool2 layers.
                super().__init__()

                self.conv1 = torch.nn.Conv2d(3, 6, 5)
                self.pool = torch.nn.MaxPool2d(2, 2)
                self.conv2 = torch.nn.Conv2d(6, 16, 5)
                self.fc1 = torch.nn.Linear(5, 120)
                self.pool2 = torch.nn.AdaptiveAvgPool2d((6, 7))

            def forward(self, x: TensorType((4, 3, Dyn, Dyn))):
                # [FUNCTION DOC] HELPER: Applies the layers in order and flattens from dim one;
                # returns the result.
                out = self.conv1(x)
                out = self.pool(out)
                out = self.conv2(out)
                out = self.pool(out)
                out = self.fc1(out)
                out = self.pool2(out)
                out = torch.flatten(out, 1)
                return out

        B = BasicBlock()
        ast_rewriter = RewritingTracer()  # noqa: F841
        traced = symbolic_trace(B)
        tc = GraphTypeChecker({}, traced)
        tc.type_check()
        infer_symbolic_types(traced)

        for n in traced.graph.nodes:
            if n.target == "conv1":
                expected = TensorType(
                    (
                        4,
                        6,
                        sympy.floor(sympy.symbols("~0") - 4),
                        sympy.floor(sympy.symbols("~1") - 4),
                    )
                )
                if n.type != expected:
                    raise AssertionError(f"Expected n.type == {expected}, got {n.type}")

            elif n.target == "conv2":
                expected = TensorType(
                    (
                        4,
                        16,
                        sympy.floor(sympy.symbols("~4") - 4),
                        sympy.floor(sympy.symbols("~5") - 4),
                    )
                )
                if n.type != expected:
                    raise AssertionError(f"Expected n.type == {expected}, got {n.type}")


if __name__ == "__main__":
    raise_on_run_directly("test/test_fx.py")
