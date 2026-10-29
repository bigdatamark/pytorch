# Owner(s): ["module: fx"]
# [MODULE DOC] Tests for the gradual-type constraint generation and Z3 solving
# pipeline in torch.fx.experimental.migrate_gradual_types. Each test traces a
# small module (or a torchvision model) with annotated input types (static
# TensorType shapes or Dyn), converts the graph to constraints with
# transform_all_constraints (or the ConstraintGenerator / transform_constraint
# APIs), feeds them to a z3 Solver, and asserts satisfiability plus values read
# back from the model. Test classes: TorchDynamoUseCases, HFOperations,
# ComposeOperationsGradualTypes, GradualTypes, TestSingleOperation,
# ConstraintGeneration, TestInternalConstraints, TestResNet, TestAlexNet.
# Import-time effects: z3 and torchvision are imported optionally, setting the
# flags HAS_Z3 and HAS_TORCHVISION; skipIfNoTorchVision skips the model tests
# when torchvision is missing.
# KNOWN ISSUE: HAS_Z3 is computed but never used to skip tests, so without z3
# the tests fail with a NameError instead of being skipped.
import operator
import unittest

import torch
from torch.fx import GraphModule, symbolic_trace
from torch.fx.experimental.meta_tracer import symbolic_trace as meta_symbolic_trace
from torch.fx.experimental.migrate_gradual_types.constraint import (
    BinConstraintT,
    DVar,
    T,
    TVar,
)
from torch.fx.experimental.migrate_gradual_types.constraint_generator import (
    ConstraintGenerator,
)
from torch.fx.experimental.migrate_gradual_types.constraint_transformation import (
    transform_constraint,
)
from torch.fx.experimental.migrate_gradual_types.operation import (
    op_consistency,
    op_matching,
    op_precision,
)
from torch.fx.experimental.migrate_gradual_types.transform_to_z3 import (
    evaluate_conditional_with_constraints,
    transform_all_constraints,
)
from torch.fx.experimental.migrate_gradual_types.z3_types import D, tensor_type, z3_dyn
from torch.fx.experimental.rewriter import RewritingTracer
from torch.fx.tensor_type import Dyn, TensorType


try:
    import z3  # type: ignore[import]

    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


try:
    from torchvision import models

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")


class TorchDynamoUseCases(unittest.TestCase):
    # [CLASS DOC] Small examples showing what the type constraints can prove, such
    # as the value of a dimension query and fixed shapes after a reshape.
    def test_dim(self):
        # [FUNCTION DOC] Traces a module whose input is annotated with a 1 by 2 tensor
        # type and returns the dimension count. Expects the constraints to be
        # satisfiable and the integer constant numbered 2 to be solved as the value 2.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module whose forward returns the number of dimensions of
            # its statically annotated input.
            def forward(self, x: TensorType([1, 2])):
                # [FUNCTION DOC] HELPER: returns the dimension count of the input tensor.
                y = x.dim()
                return y

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        y_res = z3.z3.Int(2)
        self.assertEqual(s.model()[y_res], 2)

    def test_reshape(self):
        # [FUNCTION DOC] Shows that a node can have a fixed shape whatever the input is.
        # The input is Dyn and is reshaped to one hundred elements; the size of the
        # result is read. Expects satisfiable constraints and the integer constant
        # numbered 4 to equal 100.
        """
        In this example, we prove that some nodes must
        always have a fixed shape regardless of the input
        """

        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module that reshapes a Dyn input to a flat tensor of 100
            # elements and returns the first entry of its size.
            def forward(self, x: Dyn):
                # [FUNCTION DOC] HELPER: flattens the input to 100 elements and returns the
                # size of the first dimension.
                y = x.view(100)
                tmp = y.size()[0]
                return tmp

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        transformed = transform_all_constraints(symbolic_traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        dim = z3.Int(4)
        self.assertEqual(s.model()[dim], 100)
        # print(s.model()[dim])


class HFOperations(unittest.TestCase):
    # [CLASS DOC] Tests for operations that appear in HuggingFace-style models:
    # shape arithmetic, indexing, embeddings, comparisons and conditionals that are
    # evaluated against the surrounding constraints.
    def test_eq_dim(self):
        # [FUNCTION DOC] Uses the RewritingTracer so that the comparison becomes a graph
        # node, finds the node whose target is the equality operator, and calls
        # evaluate_conditional_with_constraints. The input is a three-dimensional
        # tensor, so the condition on the dimension count must be satisfiable and its
        # negation unsatisfiable.
        """
        test dimensions and equalities
        """

        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module comparing the input dimension count to three.
            def forward(self, x: TensorType([32, 4, 4])):
                # [FUNCTION DOC] HELPER: returns whether the input has exactly three dimensions.
                eq = x.dim() == 3
                return eq

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())

        # The node we are considering is the gt node
        for n in graph.nodes:
            if n.target == operator.eq:
                node = n

        positive, negative = evaluate_conditional_with_constraints(
            ast_rewriter.root, graph, node
        )
        self.assertEqual(positive, z3.sat)
        self.assertEqual(negative, z3.unsat)

    def test_conditional_ne_1(self):
        # [FUNCTION DOC] Evaluates a conditional node (inequality between y and the
        # size of x taken as a tuple) against the graph constraints. Because x and y
        # are annotated identically, the inequality cannot hold: the positive branch is
        # expected to be unsat and the negative branch sat. The loop keeps the last
        # node whose target is the not-equal operator.
        """
        This test case is for the HFmodels interface.
        A function takes a node and a graph and considers
        the conditional the node represents and its negation
        and solves each formula with the remaining sets of constraints
        Returns:

        """

        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module comparing y with the three entries of the size of x.
            def forward(self, x: TensorType([32, 4, 4]), y: TensorType([32, 4, 4])):
                # [FUNCTION DOC] HELPER: reads the size of x, indexes its first three entries and
                # returns whether y differs from that tuple.
                size_5 = x.size()
                getitem_7 = size_5[0]
                getitem_8 = size_5[1]
                getitem_9 = size_5[2]
                ne_1 = y != (getitem_7, getitem_8, getitem_9)
                return ne_1

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())

        # The node we are considering is the gt node
        for n in graph.nodes:
            if n.target == operator.ne:
                node = n

        # since x and y are equal, the requirement that x != y cannot be true, so we should get unsat
        # for the positive condition and sat for the negative condition
        positive, negative = evaluate_conditional_with_constraints(
            ast_rewriter.root, graph, node
        )
        self.assertEqual(positive, z3.unsat)
        self.assertEqual(negative, z3.sat)

    def test_bmm(self):
        # [FUNCTION DOC] Batch matrix multiply of a tensor with a Dyn first dimension
        # and a 1 by 3 by 2 tensor. Compares the solved result type (constant numbered
        # 3) against the shape from running the real function on random data: the three
        # dimensions must equal that shape.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module returning the batched matrix product of two inputs.
            def forward(self, x: TensorType([Dyn, 2, 3]), y: TensorType([1, 3, 2])):
                # [FUNCTION DOC] HELPER: returns the batched matrix product of x and y.
                bmm = torch.bmm(x, y)
                return bmm

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        b = BasicBlock().forward(torch.rand(1, 2, 3), torch.rand(1, 3, 2))
        transformed = transform_all_constraints(symbolic_traced, counter=0)

        s = z3.Solver()
        s.add(transformed)

        output = z3.Const(3, tensor_type)
        self.assertEqual(s.check(), z3.sat)
        self.assertEqual(s.model()[output].arg(0).arg(1), b.shape[0])
        self.assertEqual(s.model()[output].arg(1).arg(1), b.shape[1])
        self.assertEqual(s.model()[output].arg(2).arg(1), b.shape[2])

    def test_bmm2(self):
        # [FUNCTION DOC] Like test_bmm but the first input is Dyn. Expects satisfiable
        # constraints; batch and last dimensions match the real result shape and the
        # middle dimension is left dynamic (its dynamic flag is zero).
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module returning the batched matrix product of a Dyn input
            # and a typed input.
            def forward(self, x: Dyn, y: TensorType([1, 3, 2])):
                # [FUNCTION DOC] HELPER: returns the batched matrix product of x and y.
                bmm = torch.bmm(x, y)
                return bmm

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        b = BasicBlock().forward(torch.rand(1, 2, 3), torch.rand(1, 3, 2))
        transformed = transform_all_constraints(symbolic_traced, counter=0)

        s = z3.Solver()
        s.add(transformed)

        output = z3.Const(3, tensor_type)
        self.assertEqual(s.check(), z3.sat)
        self.assertEqual(s.model()[output].arg(0).arg(1), b.shape[0])
        self.assertEqual(s.model()[output].arg(1).arg(0), 0)
        self.assertEqual(s.model()[output].arg(2).arg(1), b.shape[2])

    def test_bmm3(self):
        # [FUNCTION DOC] Batch matrix multiply where the batch dimensions (2 and 1) are
        # incompatible. Expects the constraints to be unsatisfiable.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module returning the batched matrix product of two inputs.
            def forward(self, x: TensorType([2, 3, 3]), y: TensorType([1, 3, 2])):
                # [FUNCTION DOC] HELPER: returns the batched matrix product of x and y.
                bmm = torch.bmm(x, y)
                return bmm

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        transformed = transform_all_constraints(symbolic_traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.unsat)

    def test_transpose(self):
        # [FUNCTION DOC] Transposes the first two dimensions of a four-dimensional
        # input. First checks the solved result type against the real result shape. Then
        # sets every placeholder annotation to Dyn, regenerates constraints and expects
        # them to stay satisfiable. Mutates the traced graph node types.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module swapping dimensions zero and one of its input.
            def forward(self, x: TensorType([1, 2, 3, 4])):
                # [FUNCTION DOC] HELPER: returns the input with dimensions zero and one swapped.
                transpose = x.transpose(0, 1)
                return transpose

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        b = BasicBlock().forward(torch.rand(1, 2, 3, 4))

        transformed = transform_all_constraints(symbolic_traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        output = z3.Const(2, tensor_type)
        self.assertEqual(s.check(), z3.sat)
        self.assertEqual(s.model()[output].arg(0).arg(1), b.shape[0])
        self.assertEqual(s.model()[output].arg(1).arg(1), b.shape[1])
        self.assertEqual(s.model()[output].arg(2).arg(1), b.shape[2])
        self.assertEqual(s.model()[output].arg(3).arg(1), b.shape[3])

        # change the annotation to Dyn
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = Dyn

        transformed = transform_all_constraints(symbolic_traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

    def test_index_select(self):
        # [FUNCTION DOC] Index-select along dimension zero of a 2050 by 1024 tensor with
        # a Dyn index. Checks the second result dimension equals the real one. Then,
        # with a fresh solver, forces the index vector (constant numbered 2) to Dyn and
        # expects the first result dimension to be dynamic as well.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module selecting rows of x using the index tensor y.
            def forward(self, x: TensorType([2050, 1024]), y: Dyn):
                # [FUNCTION DOC] HELPER: returns the rows of x along dimension zero picked by y.
                index_select = x.index_select(0, y)
                return index_select

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        # print(symbolic_traced)
        b = BasicBlock().forward(torch.rand(2050, 1024), torch.ones(8).int())
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        index_select = z3.Const(3, tensor_type)

        # the second dimension of the result should not be affected since
        # the index is 0
        self.assertEqual(s.model()[index_select].arg(1).arg(1), b.shape[1])

        replacement_vector = z3.Const(2, tensor_type)

        # we set the vector to Dyn
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        index_select = z3.Const(3, tensor_type)
        s.add(replacement_vector == z3_dyn)
        self.assertEqual(s.check(), z3.sat)

        # this implies that the index at 0 should be dyn
        self.assertEqual(s.model()[index_select].arg(0).arg(0), 0)

    def test_get_attr(self):
        # [FUNCTION DOC] Reads the device attribute of the input and moves the input to
        # it. Expects satisfiable constraints and a result type (constant numbered 3)
        # whose three dimensions equal the real shape. Raises AssertionError if not.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module that converts the input to its own device.
            def forward(self, x: TensorType([1, 2, 3])):
                # [FUNCTION DOC] HELPER: gets the device of x and returns x converted to it.
                getattr = x.device
                to = x.to(getattr)
                return to

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        b = BasicBlock().forward(torch.rand(1, 2, 3))
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        attr_res = z3.Const(3, tensor_type)
        if s.model()[attr_res].arg(0).arg(1) != b.shape[0]:
            raise AssertionError(
                f"Expected arg(0).arg(1) == {b.shape[0]}, "
                f"got {s.model()[attr_res].arg(0).arg(1)}"
            )
        if s.model()[attr_res].arg(1).arg(1) != b.shape[1]:
            raise AssertionError(
                f"Expected arg(1).arg(1) == {b.shape[1]}, "
                f"got {s.model()[attr_res].arg(1).arg(1)}"
            )
        if s.model()[attr_res].arg(2).arg(1) != b.shape[2]:
            raise AssertionError(
                f"Expected arg(2).arg(1) == {b.shape[2]}, "
                f"got {s.model()[attr_res].arg(2).arg(1)}"
            )

    def test_expand(self):
        # [FUNCTION DOC] Expands a 1 by 4 tensor using its last size as the target of
        # the first dimension. Checks the result type (constant numbered 4) against the
        # real shape; then sets the input to Dyn, regenerates constraints and expects
        # the last result dimension to still equal the real value. Mutates node types.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module that expands x to the shape of its last dimension
            # by four.
            def forward(self, x: TensorType([1, 4])):
                # [FUNCTION DOC] HELPER: reads the last entry of the size of x and expands x to
                # that many rows by four columns.
                size = x.size()
                getitem = size[-1]
                expand = x.expand(getitem, 4)
                return expand

        b = BasicBlock().forward(torch.rand(1, 4))

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        transformed = transform_all_constraints(symbolic_traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        expand_res = z3.Const(4, tensor_type)
        if s.model()[expand_res].arg(0).arg(1) != b.shape[0]:
            raise AssertionError(
                f"Expected arg(0).arg(1) == {b.shape[0]}, "
                f"got {s.model()[expand_res].arg(0).arg(1)}"
            )
        if s.model()[expand_res].arg(1).arg(1) != b.shape[1]:
            raise AssertionError(
                f"Expected arg(1).arg(1) == {b.shape[1]}, "
                f"got {s.model()[expand_res].arg(1).arg(1)}"
            )

        # change the annotation on the input to Dyn.
        # the last dimension should still be 4
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = Dyn

        transformed = transform_all_constraints(symbolic_traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        if s.model()[expand_res].arg(1).arg(1) != b.shape[1]:
            raise AssertionError(
                f"Expected arg(1).arg(1) == {b.shape[1]}, "
                f"got {s.model()[expand_res].arg(1).arg(1)}"
            )

    def test_getitem_tensor(self):
        # [FUNCTION DOC] Indexes a 4 by 4 tensor with two insertions of a new axis and
        # two full slices. Checks the four result dimensions against the real shape.
        # Then annotates inputs as dynamic first dimension by four, regenerates the
        # constraints and expects the third result dimension to be dynamic.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module indexing its input with new axes and full slices.
            def forward(self, x: TensorType([4, 4])):
                # [FUNCTION DOC] HELPER: returns the input indexed with two new axes in front
                # followed by two full slices.
                getitem = x[
                    (None, None, slice(None, None, None), slice(None, None, None))
                ]
                return getitem

        B = BasicBlock()
        b = B.forward(torch.rand(4, 4))

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(B)
        transformed = transform_all_constraints(symbolic_traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        get_item_res = z3.Const(2, tensor_type)
        if s.model()[get_item_res].arg(0).arg(1) != b.shape[0]:
            raise AssertionError(
                f"Expected arg(0).arg(1) == {b.shape[0]}, "
                f"got {s.model()[get_item_res].arg(0).arg(1)}"
            )
        if s.model()[get_item_res].arg(1).arg(1) != b.shape[1]:
            raise AssertionError(
                f"Expected arg(1).arg(1) == {b.shape[1]}, "
                f"got {s.model()[get_item_res].arg(1).arg(1)}"
            )
        if s.model()[get_item_res].arg(2).arg(1) != b.shape[2]:
            raise AssertionError(
                f"Expected arg(2).arg(1) == {b.shape[2]}, "
                f"got {s.model()[get_item_res].arg(2).arg(1)}"
            )
        if s.model()[get_item_res].arg(3).arg(1) != b.shape[3]:
            raise AssertionError(
                f"Expected arg(3).arg(1) == {b.shape[3]}, "
                f"got {s.model()[get_item_res].arg(3).arg(1)}"
            )

        # change the annotation on the input to make sure it propagates
        # to the output
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType([Dyn, 4])

        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        # dyn check
        if s.model()[get_item_res].arg(2).arg(0) != 0:
            raise AssertionError(
                f"Expected arg(2).arg(0) == 0, "
                f"got {s.model()[get_item_res].arg(2).arg(0)}"
            )

    def test_getitem_tensor2(self):
        # [FUNCTION DOC] Indexes a 4 by 4 tensor with two new-axis entries only and
        # checks the four solved result dimensions against the real result shape.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module indexing its input with two new-axis entries.
            def forward(self, x: TensorType([4, 4])):
                # [FUNCTION DOC] HELPER: returns the input indexed with two new axes.
                getitem = x[(None, None)]
                return getitem

        B = BasicBlock()
        b = B.forward(torch.rand(4, 4))

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(B)
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        get_item_res = z3.Const(2, tensor_type)
        if s.model()[get_item_res].arg(0).arg(1) != b.shape[0]:
            raise AssertionError(
                f"Expected arg(0).arg(1) == {b.shape[0]}, "
                f"got {s.model()[get_item_res].arg(0).arg(1)}"
            )
        if s.model()[get_item_res].arg(1).arg(1) != b.shape[1]:
            raise AssertionError(
                f"Expected arg(1).arg(1) == {b.shape[1]}, "
                f"got {s.model()[get_item_res].arg(1).arg(1)}"
            )
        if s.model()[get_item_res].arg(2).arg(1) != b.shape[2]:
            raise AssertionError(
                f"Expected arg(2).arg(1) == {b.shape[2]}, "
                f"got {s.model()[get_item_res].arg(2).arg(1)}"
            )
        if s.model()[get_item_res].arg(3).arg(1) != b.shape[3]:
            raise AssertionError(
                f"Expected arg(3).arg(1) == {b.shape[3]}, "
                f"got {s.model()[get_item_res].arg(3).arg(1)}"
            )

    def test_getitem_tensor_3(self):
        # [FUNCTION DOC] Indexes a 4 by 4 tensor with a new axis, a full slice, a new
        # axis and a full slice. Checks the four solved result dimensions against the
        # real shape. Raises AssertionError on mismatch.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module indexing its input with interleaved new axes and
            # slices.
            def forward(self, x: TensorType([4, 4])):
                # [FUNCTION DOC] HELPER: returns the input indexed with new axis, slice, new
                # axis, slice.
                getitem = x[
                    (None, slice(None, None, None), None, slice(None, None, None))
                ]
                return getitem

        B = BasicBlock()
        b = B.forward(torch.rand(4, 4))
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(B)
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        get_item_res = z3.Const(2, tensor_type)
        if s.model()[get_item_res].arg(0).arg(1) != b.shape[0]:
            raise AssertionError(
                f"Expected arg(0).arg(1) == {b.shape[0]}, "
                f"got {s.model()[get_item_res].arg(0).arg(1)}"
            )
        if s.model()[get_item_res].arg(1).arg(1) != b.shape[1]:
            raise AssertionError(
                f"Expected arg(1).arg(1) == {b.shape[1]}, "
                f"got {s.model()[get_item_res].arg(1).arg(1)}"
            )
        if s.model()[get_item_res].arg(2).arg(1) != b.shape[2]:
            raise AssertionError(
                f"Expected arg(2).arg(1) == {b.shape[2]}, "
                f"got {s.model()[get_item_res].arg(2).arg(1)}"
            )
        if s.model()[get_item_res].arg(3).arg(1) != b.shape[3]:
            raise AssertionError(
                f"Expected arg(3).arg(1) == {b.shape[3]}, "
                f"got {s.model()[get_item_res].arg(3).arg(1)}"
            )

    def test_layer_norm(self):
        # [FUNCTION DOC] Layer norm module with normalized size 1024 and a Dyn input.
        # Expects sat; forcing the output to a one-dimensional tensor of 1024 makes the
        # input migrate to the same type. Then a wrong annotation (10 by 10) must be
        # unsat and a right one (10 by 1024) must be sat with the output equal to the
        # real output shape. Mutates placeholder node types in the graph.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module holding a layer norm layer over 1024 features.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: builds the layer norm layer stored as attribute l.
                super().__init__()
                self.l = torch.nn.LayerNorm((1024,))

            def forward(self, x: Dyn):
                # [FUNCTION DOC] HELPER: applies the layer norm layer to the input.
                return self.l(x)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        # make the output a size 1 tensor which should result
        # in the migration of the input

        b = BasicBlock().forward(torch.rand(1024))
        input = z3.Const(1, tensor_type)
        output = z3.Const(2, tensor_type)
        s.add(output == tensor_type.tensor1(D(1, 1024)))
        s.check()
        self.assertEqual(s.model()[input], s.model()[output])
        # input shape = output shape
        self.assertEqual(b.shape[0], s.model()[input].arg(0).arg(1))

        # change annotation to the wrong shape
        for n in graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType([10, 10])

        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.unsat)

        # fix the annotation
        for n in graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType([10, 1024])

        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        s.check()
        b = BasicBlock().forward(torch.rand(10, 1024)).shape
        self.assertEqual(s.model()[output].arg(0).arg(1), b[0])
        self.assertEqual(s.model()[output].arg(1).arg(1), b[1])

    def test_layer_norm_functional(self):
        # [FUNCTION DOC] Same checks as the layer norm module test but through the
        # functional layer norm with normalized size 1024 on a Dyn input: sat, and a
        # forced one-dimensional 1024 output makes the input take the same type.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module applying functional layer norm over 1024 features.
            def forward(self, x: Dyn):
                # [FUNCTION DOC] HELPER: returns the functional layer norm of the input.
                return torch.nn.functional.layer_norm(x, (1024,))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        # make the output a size 1 tensor which should result
        # in the migration of the input

        b = BasicBlock().forward(torch.rand(1024))
        input = z3.Const(1, tensor_type)
        output = z3.Const(2, tensor_type)
        s.add(output == tensor_type.tensor1(D(1, 1024)))
        s.check()
        self.assertEqual(s.model()[input], s.model()[output])
        # input shape = output shape
        self.assertEqual(b.shape[0], s.model()[input].arg(0).arg(1))

    def test_ne_int_long_type_as(self):
        # [FUNCTION DOC] Not-equal comparison, cast to int, type_as and cast to long
        # on two Dyn by Dyn inputs. After forcing the first input to 2 by 4 and the
        # second to symbolic dimensions, expects sat and the solved final result
        # (constant numbered 8) to have the real 2 by 4 shape.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module chaining comparison, int cast, type_as and long
            # cast.
            def forward(self, x: TensorType([Dyn, Dyn]), y: TensorType([Dyn, Dyn])):
                # [FUNCTION DOC] HELPER: compares x and y for inequality, casts to int, matches
                # the type of y, casts to long and returns it.
                ne_int = torch.ne(x, y).int()
                type_as = ne_int.type_as(y)
                long = type_as.long()
                return long

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        # migrate one of the parameters to a fully static shape so we can compare

        input = z3.Const(1, tensor_type)
        input_2 = z3.Const(2, tensor_type)
        s1, s2 = z3.Ints("s1 s2")

        output_long = z3.Const(8, tensor_type)
        s.add(input == tensor_type.tensor2(D(1, 2), D(1, 4)))
        s.add(input_2 == tensor_type.tensor2(D(1, s1), D(1, s2)))

        self.assertEqual(s.check(), z3.sat)
        actual_shape = BasicBlock().forward(torch.rand(2, 4), torch.rand(2, 4)).shape
        self.assertEqual(s.model()[output_long].arg(0).arg(1), actual_shape[0])
        self.assertEqual(s.model()[output_long].arg(1).arg(1), actual_shape[1])

    def test_ne(self):
        # [FUNCTION DOC] Not-equal on two Dyn inputs traced with the RewritingTracer.
        # After first annotating x as 1 by 2 and y as 2 by Dyn, expects sat; then forces
        # one input to carry a dynamic second dimension and checks that the output
        # (constant numbered 3) still matches the real 2 by 2 broadcast shape.
        s1, s2 = z3.Ints("s1 s2")
        s11, s22 = z3.Ints("s11 s22")
        d1, d2 = D(s11, s1), D(0, s2)

        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module returning the elementwise not-equal of two inputs.
            def forward(self, x: Dyn, y: Dyn):
                # [FUNCTION DOC] HELPER: returns the elementwise inequality of x and y.
                return torch.ne(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        # change the annotations
        for n in graph.nodes:
            if n.name == "x":
                n.type = TensorType([1, 2])
            if n.name == "y":
                n.type = TensorType([2, Dyn])

        # resulting type should be TensorType([2, 2])
        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        # force the second dimension to be Dyn
        # output should still be TensorType([2, 2])
        input = z3.Const(2, tensor_type)
        s.add(input == tensor_type.tensor2(d1, d2))
        self.assertEqual(s.check(), z3.sat)
        B = BasicBlock().forward(torch.rand(1, 2), torch.rand(2, 1))
        output = z3.Const(3, tensor_type)
        self.assertEqual(s.model()[output].arg(0).arg(1), B.shape[0])
        self.assertEqual(s.model()[output].arg(1).arg(1), B.shape[0])

    def test_cumsum(self):
        # [FUNCTION DOC] Cumulative sum over dimension 3 using the meta tracer. The
        # annotation with three dimensions makes the dimension invalid, so unsat is
        # expected; with Dyn it is sat; with a four-dimensional static annotation it is
        # sat and the result equals the input shape with no dynamic dimension.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module applying a cumulative sum along dimension 3.
            def forward(self, x: TensorType([Dyn, 4, 3])):
                # [FUNCTION DOC] HELPER: returns the cumulative sum of x along dimension 3.
                t = torch.cumsum(x, 3)
                return t

        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(
            BasicBlock(), meta_args={}
        )
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)

        # should be unsat since the index is not valid for this annotation
        self.assertEqual(s.check(), z3.unsat)

        # modify the annotation to Dyn which should give sat
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = Dyn

        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        # # modify the annotation to the right tensor size
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType([1, 2, 3, 4])

        # verify that the input is equal to the output
        B = BasicBlock().forward(torch.rand(1, 2, 3, 4))
        res_shape = B.shape
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        # confirm the output matches the expected tensor
        result = z3.Const(2, tensor_type)
        self.assertEqual(s.model()[result].arg(0).arg(1), res_shape[0])
        self.assertEqual(s.model()[result].arg(1).arg(1), res_shape[1])
        self.assertEqual(s.model()[result].arg(2).arg(1), res_shape[2])
        self.assertEqual(s.model()[result].arg(3).arg(1), res_shape[3])

        # confirm the output is not dyn
        self.assertNotEqual(s.model()[result].arg(0).arg(0).as_long(), 0)
        self.assertNotEqual(s.model()[result].arg(1).arg(0).as_long(), 0)
        self.assertNotEqual(s.model()[result].arg(2).arg(0).as_long(), 0)
        self.assertNotEqual(s.model()[result].arg(3).arg(0).as_long(), 0)

    def test_cumsum_kwargs(self):
        # [FUNCTION DOC] Same as the cumulative sum test but passes the dimension as a
        # keyword. Expects unsat for the three-dimensional annotation and sat after the
        # placeholder annotations are changed to Dyn.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module applying a cumulative sum with a keyword dimension.
            def forward(self, x: TensorType([Dyn, 4, 3])):
                # [FUNCTION DOC] HELPER: returns the cumulative sum of x along dimension 3.
                t = torch.cumsum(x, dim=3)
                return t

        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(
            BasicBlock(), meta_args={}
        )
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)

        # should be unsat since the index is not valid for this annotation
        self.assertEqual(s.check(), z3.unsat)

        # modify the annotation to Dyn which should give sat
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = Dyn

        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

    def test_arange(self):
        # [FUNCTION DOC] Builds a range tensor whose length is the last size of a 2 by
        # 4 input. Expects sat with a static first result dimension equal to 4. Setting
        # placeholders to Dyn, then to four Dyn dimensions, must stay sat.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module producing a range of the length of the last
            # dimension.
            def forward(self, x: TensorType([2, 4])):
                # [FUNCTION DOC] HELPER: returns a range tensor sized by the last entry of the
                # size of x.
                size = x.size()
                getitem = size[-1]
                arange = torch.arange(getitem)
                return arange

        B = BasicBlock().forward(torch.rand(2, 4))

        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(
            BasicBlock(), meta_args={}
        )
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        arange_result = z3.Const(5, tensor_type)
        self.assertNotEqual(s.model()[arange_result].arg(0).arg(0).as_long(), 0)
        self.assertEqual(s.model()[arange_result].arg(0).arg(1).as_long(), B.size()[0])

        # change the annotation to Dyn. This will migrate to an arbitrary type
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = Dyn

        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType([Dyn, Dyn, Dyn, Dyn])

        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

    def test_scalar_add(self):
        # [FUNCTION DOC] Adds a scalar to a range tensor. Expects sat and the solved
        # types of the range result (constant numbered 5) and the sum (constant
        # numbered 6) to be identical.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module that adds one to a range tensor.
            def forward(self, x: TensorType([2, 4])):
                # [FUNCTION DOC] HELPER: builds a range sized by the last dimension of x and
                # returns it plus one.
                size = x.size()
                getitem = size[-1]
                arange = torch.arange(getitem)
                add = arange + 1
                return add

        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(
            BasicBlock(), meta_args={}
        )
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        arange_result = z3.Const(5, tensor_type)
        add_result = z3.Const(6, tensor_type)
        self.assertEqual(s.model()[arange_result], s.model()[add_result])

    def test_regular_add_2(self):
        # [FUNCTION DOC] Adds one to an integer size value. Expects sat and the integer
        # constant numbered 5 to equal the real result of running the module.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module returning the last dimension size plus one.
            def forward(self, x: TensorType([2, 4])):
                # [FUNCTION DOC] HELPER: converts x, reads the last size entry and returns it
                # plus one.
                to = x.to()
                size = to.size()
                getitem = size[-1]
                add = getitem + 1
                return add

        b = BasicBlock().forward(torch.rand(2, 4))

        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(
            BasicBlock(), meta_args={}
        )
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        res = z3.Int(5)
        self.assertEqual(s.model()[res], b)

    def test_regular_add_3(self):
        # [FUNCTION DOC] Like test_regular_add_2 but with the scalar on the left side.
        # Expects sat and the integer constant numbered 5 to equal the real result.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module returning one plus the last dimension size.
            def forward(self, x: TensorType([2, 4])):
                # [FUNCTION DOC] HELPER: converts x, reads the last size entry and returns one
                # plus it.
                to = x.to()
                size = to.size()
                getitem = size[-1]
                add = 1 + getitem
                return add

        b = BasicBlock().forward(torch.rand(2, 4))

        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(
            BasicBlock(), meta_args={}
        )
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        res = z3.Int(5)
        self.assertEqual(s.model()[res], b)

    def test_embedding(self):
        # [FUNCTION DOC] Embedding layer with 256008 rows and width 1024 on a 2 by 4
        # input. Checks the solved result (constant numbered 2) against the real size.
        # Then annotates the input as Dyn by Dyn: the first two dimensions become
        # dynamic and the last stays 1024. Finally a plain Dyn input must stay sat.
        # Mutates node types in the traced graph.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module with an embedding table applied to its input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates the embedding layer stored as embedding with
                # padding index one.
                super().__init__()
                self.embedding = torch.nn.Embedding(256008, 1024, padding_idx=1)

            def forward(self, x: TensorType([2, 4])):
                # [FUNCTION DOC] HELPER: returns the embedding of x.
                return self.embedding(x)

        B = BasicBlock().forward(torch.ones([2, 4], dtype=torch.long)).size()
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        embedding_result = z3.Const(2, tensor_type)

        if s.model()[embedding_result].arg(0).arg(1) != B[0]:
            raise AssertionError(
                f"Expected arg(0).arg(1) == {B[0]}, "
                f"got {s.model()[embedding_result].arg(0).arg(1)}"
            )
        if s.model()[embedding_result].arg(1).arg(1) != B[1]:
            raise AssertionError(
                f"Expected arg(1).arg(1) == {B[1]}, "
                f"got {s.model()[embedding_result].arg(1).arg(1)}"
            )
        if s.model()[embedding_result].arg(2).arg(1) != B[2]:
            raise AssertionError(
                f"Expected arg(2).arg(1) == {B[2]}, "
                f"got {s.model()[embedding_result].arg(2).arg(1)}"
            )

        # change the type. This should still be satisfiable
        for n in traced.graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType([Dyn, Dyn])

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        if s.model()[embedding_result].arg(0).arg(0) != 0:
            raise AssertionError(
                f"Expected arg(0).arg(0) == 0, "
                f"got {s.model()[embedding_result].arg(0).arg(0)}"
            )
        if s.model()[embedding_result].arg(1).arg(0) != 0:
            raise AssertionError(
                f"Expected arg(1).arg(0) == 0, "
                f"got {s.model()[embedding_result].arg(1).arg(0)}"
            )
        if s.model()[embedding_result].arg(2).arg(1) != B[2]:
            raise AssertionError(
                f"Expected arg(2).arg(1) == {B[2]}, "
                f"got {s.model()[embedding_result].arg(2).arg(1)}"
            )

        # change the type to Dyn. Here, we will get an arbitrary migration
        for n in traced.graph.nodes:
            if n.op == "placeholder":
                n.type = Dyn

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)

        self.assertEqual(s.check(), z3.sat)

    def test_embedding_2(self):
        # [FUNCTION DOC] Functional embedding with a typed index tensor and a weight
        # annotated with a Dyn first dimension and 1024 columns. Expects sat and the
        # result (constant numbered 5) to equal the real embedding output size.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module calling functional embedding on two inputs.
            def forward(self, x: TensorType([2, 4]), y: TensorType([Dyn, 1024])):
                # [FUNCTION DOC] HELPER: returns the embedding lookup of indices x in weight y.
                return torch.nn.functional.embedding(x, y)

        B = (
            BasicBlock()
            .forward(torch.ones([2, 4], dtype=torch.long), torch.rand(256008, 1024))
            .size()
        )
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        embedding_result = z3.Const(5, tensor_type)

        if s.model()[embedding_result].arg(0).arg(1) != B[0]:
            raise AssertionError(
                f"Expected arg(0).arg(1) == {B[0]}, "
                f"got {s.model()[embedding_result].arg(0).arg(1)}"
            )
        if s.model()[embedding_result].arg(1).arg(1) != B[1]:
            raise AssertionError(
                f"Expected arg(1).arg(1) == {B[1]}, "
                f"got {s.model()[embedding_result].arg(1).arg(1)}"
            )
        if s.model()[embedding_result].arg(2).arg(1) != B[2]:
            raise AssertionError(
                f"Expected arg(2).arg(1) == {B[2]}, "
                f"got {s.model()[embedding_result].arg(2).arg(1)}"
            )

    def test_size_two_args(self):
        # [FUNCTION DOC] Reads the last dimension of an input annotated Dyn by 2 by Dyn
        # using size with an argument. After requiring the integer constant numbered 39
        # to be nonzero, forces the input (constant 1) to a rank-three type with a
        # static last-but-one dimension of 2 and symbolic last dimension. Expects sat,
        # with the model value of input_d2 equal to 2 and of the constant numbered 39
        # equal to the model value of input_d1.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module returning the size of the last dimension.
            def forward(self, x: TensorType([Dyn, 2, Dyn])):
                # [FUNCTION DOC] HELPER: returns the size along the last dimension.
                size = x.size(-1)
                return size

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        d1, d2 = z3.Int(39), z3.Int(2)
        d4, d5 = z3.Int("input_d1"), z3.Int("input_d2")

        # migrate the third dimension
        s.add(d1 != 0)

        self.assertEqual(s.check(), z3.sat)
        input = z3.Const(1, tensor_type)
        s.add(input == tensor_type.tensor3(D(3, 39), D(1, 2), D(d4, d5)))

        # check if the item we got is the right one
        self.assertEqual(s.check(), z3.sat)
        self.assertEqual(s.model()[d5], s.model()[d2])
        self.assertEqual(s.model()[d1], s.model()[d4])

    def test_size_getitem(self):
        # [FUNCTION DOC] Reads the last entry of the size of a Dyn input and forces the
        # input to four dimensions. Expects sat and that the integer constants numbered
        # 23 and 3 get equal values in the model. A second module reading the entry ten
        # from the end, which is invalid for four dimensions, is traced fresh: it is sat
        # alone but unsat once the earlier input constant is required to differ from Dyn.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module returning the last entry of the size of x.
            def forward(self, x: Dyn):
                # [FUNCTION DOC] HELPER: reads the size of x and returns its last entry.
                size = x.size()
                getitem = size[-1]
                return getitem

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)

        self.assertEqual(s.check(), z3.sat)

        # force the input to be of size 4

        s1, s2, s3, s4 = z3.Ints("x1 x2 x3 x4")
        s11, s22, s33, s44 = z3.Ints("x11 x22 x33 x44")
        d1, d2, d3, d4 = (
            D(s11, s1),
            D(s22, s2),
            D(s33, s3),
            D(s44, s4),
        )

        input = z3.Const(1, tensor_type)
        s.add(input == tensor_type.tensor4(d1, d2, d3, d4))

        # check if the model is still SAT
        self.assertEqual(s.check(), z3.sat)

        s1, s2 = z3.Int(23), z3.Int(3)

        # check that the item is correct
        self.assertEqual(s.model()[s1], s.model()[s2])

        # invalid index but should still be SAT because input will be Dyn
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module (shadows the earlier class of the same name)
            # that indexes the size of x at an out-of-range position ten from the end.
            def forward(self, x: Dyn):
                # [FUNCTION DOC] HELPER: reads the size of x and returns the entry ten from the
                # end.
                size = x.size()
                getitem = size[-10]
                return getitem

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)

        self.assertEqual(s.check(), z3.sat)
        s.add(input != z3_dyn)
        self.assertEqual(s.check(), z3.unsat)

    def test_view_mul(self):
        # [FUNCTION DOC] Reshape with an inferred dimension, embedding lookup and
        # multiplication by a float. Expects sat; the embedding output (constant
        # numbered 6) has second dimension 4 and third dimension 1024, and the product
        # (constant numbered 13) has the same type as the embedding output.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module that views the input, embeds it and scales by 32.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates the embedding layer stored as embed_tokens.
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(256008, 1024, padding_idx=1)

            def forward(self, x: TensorType([2, 4])):
                # [FUNCTION DOC] HELPER: reshapes x to rows by its last size, embeds the result
                # and multiplies by 32.0.
                size = x.size()
                getitem = size[-1]
                view = x.view(-1, getitem)
                embed_tokens = self.embed_tokens(view)
                mul = embed_tokens * 32.0
                return mul

        # print(B)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # print(traced)

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        # print(s.model())

        embedding_result = z3.Const(6, tensor_type)

        # note that the view output will be: tensor3(dim(0, 0), dim(1, 4), dim(1, 1024))
        # this is due to the reshape constraints. This can be lifted
        # but would require revising the type rules accordingly so we leave it for now
        if s.model()[embedding_result].arg(1).arg(1) != 4:
            raise AssertionError(
                f"Expected arg(1).arg(1) == 4, "
                f"got {s.model()[embedding_result].arg(1).arg(1)}"
            )
        if s.model()[embedding_result].arg(2).arg(1) != 1024:
            raise AssertionError(
                f"Expected arg(2).arg(1) == 1024, "
                f"got {s.model()[embedding_result].arg(2).arg(1)}"
            )

        mul_result = z3.Const(13, tensor_type)
        if s.model()[mul_result] != s.model()[embedding_result]:
            raise AssertionError(
                f"Expected mul_result == embedding_result, "
                f"got {s.model()[mul_result]} != {s.model()[embedding_result]}"
            )

    def test_gt(self):
        # [FUNCTION DOC] Greater-than comparison of a size entry with one, traced with
        # the RewritingTracer. Expects sat and the boolean constant numbered 4 true.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module comparing the last dimension with one.
            def forward(self, x: TensorType([Dyn, 4])):
                # [FUNCTION DOC] HELPER: returns whether the last size of x is greater than one.
                size = x.size()
                getitem_1 = size[-1]
                gt = getitem_1 > 1
                return gt

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        res = z3.Bool(4)
        self.assertEqual(s.model()[res], True)

    def test_view(self):
        # [FUNCTION DOC] Views a 2 by 4 tensor with an inferred dimension and eight
        # columns. Only expects the constraints to be satisfiable.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module reshaping its input to eight columns.
            def forward(self, x: TensorType([2, 4])):
                # [FUNCTION DOC] HELPER: returns x viewed with an inferred first dimension and
                # eight columns.
                view = x.view(-1, 8)
                return view

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

    def test_lt_tensor(self):
        # [FUNCTION DOC] Tensor comparison between a 2 by 4 input and a Dyn input.
        # Only expects the constraints to be satisfiable.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module comparing two tensors with greater-than.
            def forward(self, x: TensorType([2, 4]), y: Dyn):
                # [FUNCTION DOC] HELPER: returns whether x is greater than y elementwise (the
                # variable is named lt but the operator is greater-than).
                lt = x > y
                return lt

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

    def test_conditional_wrong_assumption(self):
        # [FUNCTION DOC] Evaluates a greater-than node on a Dyn input via
        # evaluate_conditional_with_constraints. Since nothing is known about the
        # input, both the condition and its negation are expected to be satisfiable.
        """
        Test condition after making the wrong assumption about the input
        """

        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module comparing a Dyn input with one.
            def forward(self, x: Dyn):
                # [FUNCTION DOC] HELPER: returns whether x is greater than one.
                gt = x > 1
                return gt

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())

        # The node we are considering is the gt node
        for n in graph.nodes:
            if n.target == operator.gt:
                node = n

        positive, negative = evaluate_conditional_with_constraints(
            ast_rewriter.root, graph, node
        )

        self.assertEqual(positive, z3.sat)
        self.assertEqual(negative, z3.sat)

    def test_conditional(self):
        # [FUNCTION DOC] Evaluates a greater-than node on a size entry. With an input
        # of Dyn by 4 the condition is sat and its negation unsat. After changing
        # placeholders to Dyn, and then to Dyn by Dyn, both must be sat. Mutates node
        # types of the traced graph.
        """
        This test case is for the HFmodels interface.
        A function takes a node and a graph and considers
        the conditional the node represents and its negation
        and solves each formula with the remaining sets of constraints
        Returns:

        """

        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module with an embedding that returns a size comparison.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates the embedding layer stored as embed_tokens.
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(256008, 1024, padding_idx=1)

            def forward(self, x: TensorType([Dyn, 4])):
                # [FUNCTION DOC] HELPER: reshapes and embeds x (result unused) and returns
                # whether its last size is greater than one.
                size = x.size()
                getitem = size[-1]
                view = x.view(-1, getitem)
                _embed_tokens = self.embed_tokens(view)
                getitem_1 = size[-1]
                gt = getitem_1 > 1
                return gt

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())

        # The node we are considering is the gt node
        for n in graph.nodes:
            if n.target == operator.gt:
                node = n

        positive, negative = evaluate_conditional_with_constraints(
            ast_rewriter.root, graph, node
        )
        self.assertEqual(positive, z3.sat)
        self.assertEqual(negative, z3.unsat)

        # change the annotation to Dyn
        for n in graph.nodes:
            if n.op == "placeholder":
                n.type = Dyn

        # here, both should be SAT since the input is Dyn
        positive, negative = evaluate_conditional_with_constraints(
            ast_rewriter.root, graph, node
        )

        self.assertEqual(positive, z3.sat)
        self.assertEqual(negative, z3.sat)

        # change the annotation to TensorType[Dyn, Dyn]
        for n in graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType([Dyn, Dyn])

        # here, both should be SAT as well
        positive, negative = evaluate_conditional_with_constraints(
            ast_rewriter.root, graph, node
        )

        self.assertEqual(positive, z3.sat)
        self.assertEqual(negative, z3.sat)

    def test_conditional_2(self):
        # [FUNCTION DOC] Mirror of test_conditional using a less-than node: for a Dyn
        # by 4 input the condition is unsat and its negation sat.
        """
        This test case is for the HFmodels interface.
        A function takes a node and a graph and considers
        the conditional the node represents and its negation
        and solves each formula with the remaining sets of constraints
        Returns the opposite result of the above testcase

        """

        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module with an embedding that returns a size comparison.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates the embedding layer stored as embed_tokens.
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(256008, 1024, padding_idx=1)

            def forward(self, x: TensorType([Dyn, 4])):
                # [FUNCTION DOC] HELPER: reshapes and embeds x (result unused) and returns
                # whether its last size is less than one.
                size = x.size()
                getitem = size[-1]
                view = x.view(-1, getitem)
                _embed_tokens = self.embed_tokens(view)
                getitem_1 = size[-1]
                lt = getitem_1 < 1
                return lt

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())

        # The node we are considering is the gt node
        for n in graph.nodes:
            if n.target == operator.lt:
                node = n

        positive, negative = evaluate_conditional_with_constraints(
            ast_rewriter.root, graph, node
        )
        self.assertEqual(positive, z3.unsat)
        self.assertEqual(negative, z3.sat)


class ComposeOperationsGradualTypes(unittest.TestCase):
    # [CLASS DOC] Tests that combine several operations (range, comparison, masked
    # fill, reshape, convolution, addition) in a single traced module.
    def test_masked_fill(self):
        # [FUNCTION DOC] Builds a range of the last size, compares with a viewed input
        # and applies in-place masked fill. Expects sat and a result (constant numbered
        # 10) whose two dimensions equal the real shape. Annotations changed to Dyn and
        # then to four Dyn dimensions must stay sat.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module composing range, view, comparison and masked fill.
            def forward(self, x: TensorType([2, 4])):
                # [FUNCTION DOC] HELPER: builds a mask by comparing a range with a viewed x and
                # returns x after masked fill with zero.
                size = x.size()
                getitem = size[-1]
                arange = torch.arange(getitem)
                view = x.view(-1, getitem)
                lt = arange > view
                masked_fill = x.masked_fill_(lt, 0)
                return masked_fill

        B = BasicBlock().forward(torch.rand(2, 4))
        # print(B.shape)

        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(
            BasicBlock(), meta_args={}
        )
        # print(symbolic_traced)
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        masked_fill_res = z3.Const(10, tensor_type)
        self.assertEqual(
            s.model()[masked_fill_res].arg(0).arg(1).as_long(), B.size()[0]
        )
        self.assertEqual(
            s.model()[masked_fill_res].arg(1).arg(1).as_long(), B.size()[1]
        )

        # change the annotation to Dyn. This will migrate to an arbitrary type
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = Dyn

        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType([Dyn, Dyn, Dyn, Dyn])

        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

    def test_add_reshape_1(self):
        # [FUNCTION DOC] Adds two Dyn inputs reshaped to compatible 1 by 2 and 2 by 2
        # shapes. Only expects satisfiable constraints.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module adding two reshaped inputs.
            def forward(self, x: Dyn, y: Dyn):
                # [FUNCTION DOC] HELPER: returns the sum of x reshaped to 1 by 2 and y to 2 by 2.
                return torch.add(torch.reshape(x, (1, 2)), torch.reshape(y, (2, 2)))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

    def test_add_reshape_2(self):
        # [FUNCTION DOC] Adds reshapes with an inferred dimension (rows by 2) and 2 by
        # 2 by 2. Only expects satisfiable constraints.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module adding two reshaped inputs of different rank.
            def forward(self, x: Dyn, y: Dyn):
                # [FUNCTION DOC] HELPER: returns the sum of x reshaped to rows by 2 and y to
                # 2 by 2 by 2.
                return torch.add(torch.reshape(x, (-1, 2)), torch.reshape(y, (2, 2, 2)))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

    def test_conv_reshape_add_0(self):
        # [FUNCTION DOC] Convolution of a reshaped Dyn input followed by addition with
        # a Dyn input. Builds the module with fixed settings, traces it and expects the
        # constraints to be satisfiable.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module with an unbiased convolution layer.
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
                # [FUNCTION DOC] HELPER: creates a two-dimensional convolution without bias
                # from the given channel counts, kernel size, stride, padding, groups and
                # dilation, stored as conv1.
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

            def forward(self, x: Dyn, y: Dyn):
                # [FUNCTION DOC] HELPER: reshapes x to 1 by 2 by 10 by 20, applies conv1 and
                # adds y.
                return torch.add(self.conv1(torch.reshape(x, (1, 2, 10, 20))), y)

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)
        solver = z3.Solver()
        solver.add(new_transformed_c)
        self.assertEqual(solver.check(), z3.sat)

    def test_conv_reshape_add_0_2(self):
        # [FUNCTION DOC] Same module with y annotated 4 by 1. Compares the convolution
        # result type (constant numbered 4) with the real output shape, then expects the
        # add result (constant numbered 9) to take the same shape thanks to
        # broadcasting.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module with an unbiased convolution layer.
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
                # [FUNCTION DOC] HELPER: creates a two-dimensional convolution without bias
                # from the given channel counts, kernel size, stride, padding, groups and
                # dilation, stored as conv1.
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

            def forward(self, x: Dyn, y: TensorType([4, 1])):
                # [FUNCTION DOC] HELPER: reshapes x to 1 by 2 by 10 by 20, applies conv1 and
                # adds y.
                return torch.add(self.conv1(torch.reshape(x, (1, 2, 10, 20))), y)

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)

        #        4,1
        # 1, 2, 4, 8
        res = B.forward(torch.rand(20, 20), torch.rand(1, 2, 4, 8)).size()
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)
        solver = z3.Solver()
        solver.add(new_transformed_c)
        self.assertEqual(solver.check(), z3.sat)

        conv_result = z3.Const(4, tensor_type)
        add_result = z3.Const(9, tensor_type)
        input_2 = z3.Const(2, tensor_type)

        s1, s2, s3, s4 = z3.Ints("x1 x2 x3 x4")
        s11, s22, s33, s44 = z3.Ints("x11 x22 x33 x44")
        d1, d2, d3, d4 = (
            D(s11, s1),
            D(s22, s2),
            D(s33, s3),
            D(s44, s4),
        )

        solver.add(conv_result == tensor_type.tensor4(d1, d2, d3, d4))
        solver.check()
        if solver.model()[s1].as_long() != res[0]:
            raise AssertionError(
                f"Expected s1 == {res[0]}, got {solver.model()[s1].as_long()}"
            )
        if solver.model()[s2].as_long() != res[1]:
            raise AssertionError(
                f"Expected s2 == {res[1]}, got {solver.model()[s2].as_long()}"
            )
        if solver.model()[s3].as_long() != res[2]:
            raise AssertionError(
                f"Expected s3 == {res[2]}, got {solver.model()[s3].as_long()}"
            )
        if solver.model()[s4].as_long() != res[3]:
            raise AssertionError(
                f"Expected s4 == {res[3]}, got {solver.model()[s4].as_long()}"
            )

        solver.add(input_2 == tensor_type.tensor2(D(1, 4), D(1, 1)))
        self.assertEqual(solver.check(), z3.sat)
        solver.add(add_result == tensor_type.tensor4(d1, d2, d3, d4))
        self.assertEqual(solver.check(), z3.sat)

        # first dimension could be anything because we have broadcasting
        if solver.model()[s1] != res[0]:
            raise AssertionError(f"Expected s1 == {res[0]}, got {solver.model()[s1]}")
        if solver.model()[s2] != res[1]:
            raise AssertionError(f"Expected s2 == {res[1]}, got {solver.model()[s2]}")
        if solver.model()[s3] != res[2]:
            raise AssertionError(f"Expected s3 == {res[2]}, got {solver.model()[s3]}")
        if solver.model()[s4] != res[3]:
            raise AssertionError(f"Expected s4 == {res[3]}, got {solver.model()[s4]}")

    def test_conv_reshape_add_0_3(self):
        # [FUNCTION DOC] Same module with y annotated 11 by 1, which cannot broadcast
        # against the convolution output. Expects unsat.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module with an unbiased convolution layer.
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
                # [FUNCTION DOC] HELPER: creates a two-dimensional convolution without bias
                # from the given channel counts, kernel size, stride, padding, groups and
                # dilation, stored as conv1.
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

            def forward(self, x: Dyn, y: TensorType([11, 1])):
                # [FUNCTION DOC] HELPER: reshapes x to 1 by 2 by 10 by 20, applies conv1 and
                # adds y.
                return torch.add(self.conv1(torch.reshape(x, (1, 2, 10, 20))), y)

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)
        solver = z3.Solver()
        solver.add(new_transformed_c)
        self.assertEqual(solver.check(), z3.unsat)

    def test_conv_reshape_add_1(self):
        # [FUNCTION DOC] Same module with y annotated 1 by 2 by 10 by 20, which does
        # not match the convolution output shape. Expects unsat.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module with an unbiased convolution layer.
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
                # [FUNCTION DOC] HELPER: creates a two-dimensional convolution without bias
                # from the given channel counts, kernel size, stride, padding, groups and
                # dilation, stored as conv1.
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

            def forward(self, x: Dyn, y: TensorType([1, 2, 10, 20])):
                # [FUNCTION DOC] HELPER: reshapes x to 1 by 2 by 10 by 20, applies conv1 and
                # adds y.
                return torch.add(self.conv1(torch.reshape(x, (1, 2, 10, 20))), y)

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)
        solver = z3.Solver()
        solver.add(new_transformed_c)
        self.assertEqual(solver.check(), z3.unsat)


class GradualTypes(unittest.TestCase):
    # [CLASS DOC] Tests for gradual-typing behavior of convolution after reshape
    # from Dyn inputs.
    def test_conv_reshape_unsat(self):
        # [FUNCTION DOC] Reshapes a Dyn input to three dimensions and feeds it to a
        # convolution that needs four. Expects unsat.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module with an unbiased convolution layer.
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
                # [FUNCTION DOC] HELPER: creates a two-dimensional convolution without bias
                # from the given channel counts, kernel size, stride, padding, groups and
                # dilation, stored as conv1.
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

            def forward(self, x: Dyn):
                # [FUNCTION DOC] HELPER: reshapes x to 1 by 2 by 10 and applies conv1.
                return self.conv1(torch.reshape(x, (1, 2, 10)))

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)
        solver = z3.Solver()
        solver.add(new_transformed_c)
        self.assertEqual(solver.check(), z3.unsat)

    def test_conv_reshape0(self):
        # [FUNCTION DOC] Reshapes a Dyn input to 1 by 2 by 10 by 20 and convolves. Runs
        # the real module on random data to get the output size, expects sat, and
        # requires the solved convolution result (constant numbered 3) to carry those
        # sizes. Finally adds an unchecked constraint on the input type.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module with an unbiased convolution layer.
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
                # [FUNCTION DOC] HELPER: creates a two-dimensional convolution without bias
                # from the given channel counts, kernel size, stride, padding, groups and
                # dilation, stored as conv1.
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

            def forward(self, x: Dyn):
                # [FUNCTION DOC] HELPER: reshapes x to 1 by 2 by 10 by 20 and applies conv1.
                return self.conv1(torch.reshape(x, (1, 2, 10, 20)))

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        res = B.forward(torch.rand(20, 20)).size()
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)

        solver = z3.Solver()
        solver.add(new_transformed_c)
        self.assertEqual(solver.check(), z3.sat)
        conv_result = z3.Const(3, tensor_type)

        s1, s2, s3, s4 = z3.Ints("x1 x2 x3 x4")
        s11, s22, s33, s44 = z3.Ints("x11 x22 x33 x44")
        d1, d2, d3, d4 = (
            D(s11, s1),
            D(s22, s2),
            D(s33, s3),
            D(s44, s4),
        )

        solver.add(conv_result == tensor_type.tensor4(d1, d2, d3, d4))
        solver.check()
        # print(solver.model())
        # print(type(solver.model()[s1]))
        if solver.model()[s1].as_long() != res[0]:
            raise AssertionError(
                f"Expected s1 == {res[0]}, got {solver.model()[s1].as_long()}"
            )
        if solver.model()[s2].as_long() != res[1]:
            raise AssertionError(
                f"Expected s2 == {res[1]}, got {solver.model()[s2].as_long()}"
            )
        if solver.model()[s3].as_long() != res[2]:
            raise AssertionError(
                f"Expected s3 == {res[2]}, got {solver.model()[s3].as_long()}"
            )
        if solver.model()[s4].as_long() != res[3]:
            raise AssertionError(
                f"Expected s4 == {res[3]}, got {solver.model()[s4].as_long()}"
            )

        s1, s2, s3, s4 = z3.Ints("y1 y2 y3 y4")
        s11, s22, s33, s44 = z3.Ints("y11 y22 y33 y44")
        d1, d2, d3, d4 = (
            D(s11, s1),
            D(s22, s2),
            D(s33, s3),
            D(s44, s4),
        )

        input = z3.Const(1, tensor_type)
        solver.add(input == tensor_type.tensor4(d1, d2, d3, d4))

        # assert solver.check() == sat
        # solver.add(s11 == 1)
        # solver.add(s22 == 1)
        # solver.add(s33 == 1)
        # solver.add(s44 == 1)
        #
        # print(solver.check())
        # print(solver.model())

    def test_conv_reshape1(self):
        # [FUNCTION DOC] Reshapes a 20 by 20 input to 1 by an inferred dimension by 10
        # by 20 and convolves. Expects sat and the solved convolution result (constant
        # numbered 3) to match the real output size.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module with an unbiased convolution layer.
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
                # [FUNCTION DOC] HELPER: creates a two-dimensional convolution without bias
                # from the given channel counts, kernel size, stride, padding, groups and
                # dilation, stored as conv1.
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

            def forward(self, x: TensorType([20, 20])):
                # [FUNCTION DOC] HELPER: reshapes x to 1 by an inferred dimension by 10 by 20
                # and applies conv1.
                return self.conv1(torch.reshape(x, (1, -1, 10, 20)))

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        res = B.forward(torch.rand(20, 20)).size()
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)

        solver = z3.Solver()
        solver.add(new_transformed_c)
        self.assertEqual(solver.check(), z3.sat)
        conv_result = z3.Const(3, tensor_type)

        s1, s2, s3, s4 = z3.Ints("x1 x2 x3 x4")
        s11, s22, s33, s44 = z3.Ints("x11 x22 x33 x44")
        d1, d2, d3, d4 = (
            D(s11, s1),
            D(s22, s2),
            D(s33, s3),
            D(s44, s4),
        )

        solver.add(conv_result == tensor_type.tensor4(d1, d2, d3, d4))
        solver.check()
        # print(solver.model())
        if solver.model()[s1].as_long() != res[0]:
            raise AssertionError(
                f"Expected s1 == {res[0]}, got {solver.model()[s1].as_long()}"
            )
        if solver.model()[s2].as_long() != res[1]:
            raise AssertionError(
                f"Expected s2 == {res[1]}, got {solver.model()[s2].as_long()}"
            )
        if solver.model()[s3].as_long() != res[2]:
            raise AssertionError(
                f"Expected s3 == {res[2]}, got {solver.model()[s3].as_long()}"
            )
        if solver.model()[s4].as_long() != res[3]:
            raise AssertionError(
                f"Expected s4 == {res[3]}, got {solver.model()[s4].as_long()}"
            )


class TestSingleOperation(unittest.TestCase):
    # [CLASS DOC] Tests that exercise one operation at a time (convolution,
    # addition with broadcasting and padding, reshape, flatten and pooling).
    def test_conv_wrong_example(self):
        # [FUNCTION DOC] Two convolutions on the same Dyn input whose input channel
        # counts disagree (2 versus 4). Prints the solver result, expects sat, then
        # forcing the input to a four-dimensional tensor stays sat, and requiring a
        # non-dynamic second dimension makes it unsat.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module with two convolutions and an in-place ReLU.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates conv1 (2 input channels) and conv2 (4 input
                # channels), both unbiased, plus a ReLU stored as relu.
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    in_channels=2,
                    out_channels=2,
                    kernel_size=2,
                    stride=2,
                    padding=2,
                    groups=2,
                    bias=False,
                    dilation=2,
                )

                self.conv2 = torch.nn.Conv2d(
                    in_channels=4,
                    out_channels=2,
                    kernel_size=2,
                    stride=2,
                    padding=2,
                    groups=2,
                    bias=False,
                    dilation=2,
                )

                self.relu = torch.nn.ReLU(inplace=True)

            def forward(self, x: Dyn):
                # [FUNCTION DOC] HELPER: applies the first convolution and ReLU (result
                # discarded) then returns ReLU of the second convolution.
                y = self.relu(self.conv1(x))  # noqa: F841
                z = self.relu(self.conv2(x))
                return z

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced)

        solver3 = z3.Solver()
        solver3.add(transformed)
        print(solver3.check())
        if solver3.check() != z3.sat:
            raise AssertionError(
                f"Expected solver3.check() == z3.sat, got {solver3.check()}"
            )

        s1, s2, s3, s4 = z3.Ints("s1 s2 s3 s4")
        s11, s22, s33, s44 = z3.Ints("s11 s22 s33 s44")
        d1, d2, d3, d4 = (
            D(s11, s1),
            D(s22, s2),
            D(s33, s3),
            D(s44, s4),
        )
        x = z3.Const(1, tensor_type)
        solver3.add(x == tensor_type.tensor4(d1, d2, d3, d4))
        if solver3.check() != z3.sat:
            raise AssertionError(
                f"Expected solver3.check() == z3.sat, got {solver3.check()}"
            )

        solver3.add(s22 != 0)
        if solver3.check() != z3.unsat:
            raise AssertionError(
                f"Expected solver3.check() == z3.unsat, got {solver3.check()}"
            )

    def test_conv_dyn(self):
        # [FUNCTION DOC] Convolution on a Dyn input. Constrains the input (constant 1)
        # and output (constant 2) to rank four with symbolic dimensions; expects sat
        # and the batch dimension value and flag of input and output to be equal.
        # After requiring the second input dimension value to differ from 2, the
        # model must still be sat with the second input flag zero (dynamic); requiring
        # that flag to be nonzero then gives unsat. A second solver where the input is
        # forced to rank three must be unsat.
        s1, s2, s3, s4 = z3.Ints("s1 s2 s3 s4")
        e1, e2, e3, e4 = z3.Ints("e1 e2 e3 e4")
        s11, s22, s33, s44 = z3.Ints("s11 s22 s33 s44")
        e11, e22, e33, e44 = z3.Ints("e11 e22 e33 e44")
        d1, d2, d3, d4 = (
            D(s11, s1),
            D(s22, s2),
            D(s33, s3),
            D(s44, s4),
        )
        b1, b2, b3, b4 = D(e11, e1), D(e22, e2), D(e33, e3), D(e44, e4)

        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module with an unbiased convolution layer.
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
                # [FUNCTION DOC] HELPER: creates a two-dimensional convolution without bias
                # from the given channel counts, kernel size, stride, padding, groups and
                # dilation, stored as conv1.
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

            def forward(self, x: Dyn):
                # [FUNCTION DOC] HELPER: applies conv1 to x.
                return self.conv1(x)

        BasicBlock(2, 2, 2, 2, 2, 2, 2).forward(torch.rand(4, 2, 3, 4))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock(2, 2, 2, 2, 2, 2, 2))
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced)

        solver3 = z3.Solver()
        solver3.add(transformed)
        if solver3.check() != z3.sat:
            raise AssertionError(
                f"Expected solver3.check() == z3.sat, got {solver3.check()}"
            )

        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        solver3.add(
            x == tensor_type.tensor4(d1, d2, d3, d4),
            y == tensor_type.tensor4(b1, b2, b3, b4),
        )

        if solver3.check() != z3.sat:
            raise AssertionError(
                f"Expected solver3.check() == z3.sat, got {solver3.check()}"
            )
        if solver3.model()[s1].as_long() != solver3.model()[e1].as_long():
            raise AssertionError(
                f"Expected s1 == e1, got {solver3.model()[s1].as_long()} != "
                f"{solver3.model()[e1].as_long()}"
            )
        if solver3.model()[s11].as_long() != solver3.model()[e11].as_long():
            raise AssertionError(
                f"Expected s11 == e11, got {solver3.model()[s11].as_long()} != "
                f"{solver3.model()[e11].as_long()}"
            )

        solver3.add(s2 != 2)
        if solver3.check() != z3.sat:
            raise AssertionError(
                f"Expected solver3.check() == z3.sat, got {solver3.check()}"
            )
        if solver3.model()[s22].as_long() != 0:
            raise AssertionError(
                f"Expected s22 == 0, got {solver3.model()[s22].as_long()}"
            )

        solver3.add(s22 != 0)
        self.assertEqual(solver3.check(), z3.unsat)

        solver2 = z3.Solver()
        solver2.add(transformed)
        if solver2.check() != z3.sat:
            raise AssertionError(
                f"Expected solver2.check() == z3.sat, got {solver2.check()}"
            )
        solver2.add(x == tensor_type.tensor3(d1, d2, d3))
        self.assertEqual(solver2.check(), z3.unsat)

    def test_add(self):
        # [FUNCTION DOC] Adds two Dyn inputs after forcing them to one-element tensors
        # of sizes one and two (sat). A second module with a typed Dyn-length first
        # input and sizes four and five is unsat since they differ. A third module
        # whose first input is forced to rank two is unsat.
        s1, s2, s3, s4 = z3.Ints("s1 s2 s3 s4")
        s11, s22, s33, s44 = z3.Ints("s11 s22 s33 s44")
        d1, d2 = D(s11, s1), D(s22, s2)

        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module adding two Dyn inputs.
            def forward(self, x: Dyn, y: Dyn):
                # [FUNCTION DOC] HELPER: returns the sum of x and y.
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        # make the tensor be of size 1
        x = z3.Const(1, tensor_type)
        s.add(x == tensor_type.tensor1(D(1, s11)))
        self.assertEqual(s.check(), z3.sat)

        y = z3.Const(2, tensor_type)
        s.add(y == tensor_type.tensor1(D(1, s22)))
        self.assertEqual(s.check(), z3.sat)

        s.add(s11 == 1)  # tensor[1]
        s.add(s22 == 2)  # tensor[2]
        self.assertEqual(s.check(), z3.sat)

        class BasicBlock2(torch.nn.Module):
            # [CLASS DOC] HELPER: module adding a one-dimensional input and a Dyn input.
            def forward(self, x: TensorType((Dyn,)), y: Dyn):
                # [FUNCTION DOC] HELPER: returns the sum of x and y.
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock2())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        # make the tensor be of size 1
        x = z3.Const(1, tensor_type)
        s.add(x == tensor_type.tensor1(D(1, s11)))
        self.assertEqual(s.check(), z3.sat)
        y = z3.Const(2, tensor_type)
        s.add(y == tensor_type.tensor1(D(1, s22)))
        self.assertEqual(s.check(), z3.sat)
        s.add(s11 == 4)  # tensor[4]
        s.add(s22 == 5)  # tensor[5]
        self.assertEqual(s.check(), z3.unsat)

        class BasicBlock3(torch.nn.Module):
            # [CLASS DOC] HELPER: module adding a one-dimensional input and a Dyn input.
            def forward(self, x: TensorType((Dyn,)), y: Dyn):
                # [FUNCTION DOC] HELPER: returns the sum of x and y.
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock3())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced)
        s = z3.Solver()
        s.add(transformed)
        x = z3.Const(1, tensor_type)
        s.add(x == tensor_type.tensor2(d1, d2))
        self.assertEqual(s.check(), z3.unsat)

    def test_add_padding(self):
        # [FUNCTION DOC] Adds a rank-one input and a rank-two input (broadcast
        # padding). Forces the first to a one-element tensor and expects sat.
        s1, s2, s3, s4 = z3.Ints("s1 s2 s3 s4")

        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module adding a vector and a matrix.
            def forward(self, x: TensorType((Dyn,)), y: TensorType((Dyn, Dyn))):
                # [FUNCTION DOC] HELPER: returns the sum of x and y.
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        x = z3.Const(1, tensor_type)
        s.add(x == tensor_type.tensor1(D(1, s1)))

        self.assertEqual(s.check(), z3.sat)

        # print(s.model())

    def test_add_padding_2(self):
        # [FUNCTION DOC] Adds a matrix and a vector. Forces x to a matrix and y to a
        # vector with a dynamic dimension, checks the padded broadcast intermediate
        # results and the sum are rank two while y stays rank one, then requires the
        # second dimension to exceed one so no broadcasting happens and the sum's
        # second dimension must not be dynamic.
        s1, s2, s3, s4 = z3.Ints("s1 s2 s3 s4")

        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module adding a matrix and a vector.
            def forward(self, x: TensorType([Dyn, Dyn]), y: TensorType([Dyn])):
                # [FUNCTION DOC] HELPER: returns the sum of x and y.
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        # print(s.model())

        x = z3.Const(1, tensor_type)
        s.add(x == tensor_type.tensor2(D(1, s1), D(1, s2)))
        self.assertEqual(s.check(), z3.sat)

        y = z3.Const(2, tensor_type)
        s.add(y == tensor_type.tensor1(D(0, s3)))
        self.assertEqual(s.check(), z3.sat)

        add_result = z3.Const(3, tensor_type)
        broadcast_res1, broadcast_res2 = (
            z3.Const(4, tensor_type),
            z3.Const(5, tensor_type),
        )

        # print(s.model())

        if s.model()[broadcast_res1].decl() != tensor_type.tensor2:
            raise AssertionError(
                f"Expected broadcast_res1.decl() == tensor_type.tensor2, "
                f"got {s.model()[broadcast_res1].decl()}"
            )
        if s.model()[broadcast_res2].decl() != tensor_type.tensor2:
            raise AssertionError(
                f"Expected broadcast_res2.decl() == tensor_type.tensor2, "
                f"got {s.model()[broadcast_res2].decl()}"
            )
        if s.model()[add_result].decl() != tensor_type.tensor2:
            raise AssertionError(
                f"Expected add_result.decl() == tensor_type.tensor2, "
                f"got {s.model()[add_result].decl()}"
            )
        if s.model()[y].decl() != tensor_type.tensor1:
            raise AssertionError(
                f"Expected y.decl() == tensor_type.tensor1, got {s.model()[y].decl()}"
            )

        # print(s.model())

        # prevent broadcasting for that dimension
        s.add(s2 > 1)

        if not s.check():
            raise AssertionError("Expected s.check() to be truthy")

        # the second dimension of the result is a number, not Dyn.
        # however if the first input dimension had been 1, we would
        # have had dyn in the result, as seen in the next test case
        if s.model()[add_result].arg(1).arg(0).as_long() == 0:
            raise AssertionError(
                f"Expected arg(1).arg(0) != 0, "
                f"got {s.model()[add_result].arg(1).arg(0).as_long()}"
            )

    def test_add_padding_3(self):
        # [FUNCTION DOC] Adds a Dyn by 1 matrix and a vector whose dimension is marked
        # dynamic. With a nonzero second dimension on x, expects sat and both
        # dimensions of the sum (constant numbered 3) to be dynamic.
        s1, s2, s3, s4 = z3.Ints("s1 s2 s3 s4")

        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module adding a matrix and a vector.
            def forward(self, x: TensorType([Dyn, 1]), y: TensorType([Dyn])):
                # [FUNCTION DOC] HELPER: returns the sum of x and y.
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        # print(transformed)
        self.assertEqual(s.check(), z3.sat)

        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        s.add(s2 != 0)
        s.add(x == tensor_type.tensor2(D(0, s1), D(s2, 1)))
        s.add(y == tensor_type.tensor1(D(0, s3)))

        self.assertEqual(s.check(), z3.sat)

        # print(s.model())

        add_result = z3.Const(3, tensor_type)
        if s.model()[add_result].arg(0).arg(0).as_long() != 0:
            raise AssertionError(
                f"Expected arg(0).arg(0) == 0, "
                f"got {s.model()[add_result].arg(0).arg(0).as_long()}"
            )
        if s.model()[add_result].arg(1).arg(0).as_long() != 0:
            raise AssertionError(
                f"Expected arg(1).arg(0) == 0, "
                f"got {s.model()[add_result].arg(1).arg(0).as_long()}"
            )

    def test_add_padding_4(self):
        # [FUNCTION DOC] Adds a 2 by 1 matrix and a vector of three. Expects sat and
        # the sum (constant numbered 3) to be the static two by three tensor type.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module adding a matrix and a vector.
            def forward(self, x: TensorType([2, 1]), y: TensorType([3])):
                # [FUNCTION DOC] HELPER: returns the sum of x and y.
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)

        self.assertEqual(s.check(), z3.sat)

        add_result = z3.Const(3, tensor_type)
        expected = tensor_type.tensor2(D(1, 2), D(1, 3))
        if s.model()[add_result] != expected:
            raise AssertionError(
                f"Expected add_result == {expected}, got {s.model()[add_result]}"
            )

    def test_add_padding_5(self):
        # [FUNCTION DOC] Adds a 2 by 2 matrix and a vector of three. The last
        # dimensions conflict (two versus three, neither one). Expects unsat.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module adding a matrix and a vector.
            def forward(self, x: TensorType([2, 2]), y: TensorType([3])):
                # [FUNCTION DOC] HELPER: returns the sum of x and y.
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.unsat)

    def test_add_size_3(self):
        # [FUNCTION DOC] Adds two Dyn-shaped rank-three tensors. Forces the dimensions
        # so that broadcast between a one and other sizes is possible: sat, still sat
        # after fixing one dimension to five, and unsat once the matching dimension of
        # the other input is six.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module adding two rank-three inputs.
            def forward(
                self, x: TensorType([Dyn, Dyn, Dyn]), y: TensorType([Dyn, Dyn, Dyn])
            ):
                # [FUNCTION DOC] HELPER: returns the sum of x and y.
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        s1, s2, s3, s4, s5 = z3.Ints("s1 s2 s3 s4 s5")

        s.add(x == tensor_type.tensor3(D(1, s1), D(1, 1), D(1, s2)))
        s.add(y == tensor_type.tensor3(D(1, s3), D(1, s4), D(1, s5)))

        self.assertEqual(s.check(), z3.sat)
        s.add(s2 == 5)
        self.assertEqual(s.check(), z3.sat)
        s.add(s5 == 6)
        self.assertEqual(s.check(), z3.unsat)

    def test_add_padding_6(self):
        # [FUNCTION DOC] Adds a rank-one and a rank-three tensor. Forcing the vector
        # length to four and the last dimension of the other input to five is unsat.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module adding a vector and a rank-three tensor.
            def forward(self, x: TensorType([Dyn]), y: TensorType([Dyn, Dyn, Dyn])):
                # [FUNCTION DOC] HELPER: returns the sum of x and y.
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        s1, s2, s3, s4, s5 = z3.Ints("s1 s2 s3 s4 s5")

        s.add(x == tensor_type.tensor1(D(1, s1)))
        s.add(y == tensor_type.tensor3(D(1, s2), D(1, s3), D(1, s4)))

        self.assertEqual(s.check(), z3.sat)

        s.add(s1 == 4)
        s.add(s4 == 5)

        self.assertEqual(s.check(), z3.unsat)

    def test_add_padding_7(self):
        # [FUNCTION DOC] Adds a vector and a rank-four tensor, then forces the first
        # input to rank two, which is inconsistent with its annotation. Expects unsat.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module adding a vector and a rank-four tensor.
            def forward(
                self, x: TensorType([Dyn]), y: TensorType([Dyn, Dyn, Dyn, Dyn])
            ):
                # [FUNCTION DOC] HELPER: returns the sum of x and y.
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        x = z3.Const(1, tensor_type)
        s1, s2, s3, s4, s5 = z3.Ints("s1 s2 s3 s4 s5")
        s.add(x == tensor_type.tensor2(D(s1, s2), D(s2, s3)))
        self.assertEqual(s.check(), z3.unsat)

    def test_add_padding_8(self):
        # [FUNCTION DOC] Adds a vector and a rank-four tensor. Forces x to a vector
        # with flag variable and nonnegative flag, then y to four dynamic dimensions.
        # Expects sat in both steps.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module adding a vector and a rank-four tensor.
            def forward(
                self, x: TensorType([Dyn]), y: TensorType([Dyn, Dyn, Dyn, Dyn])
            ):
                # [FUNCTION DOC] HELPER: returns the sum of x and y.
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        s1, s2, s3, s4, s5 = z3.Ints("s1 s2 s3 s4 s5")
        s.add(x == tensor_type.tensor1(D(s1, 1)))
        s.add(s1 >= 0)

        self.assertEqual(s.check(), z3.sat)

        s.add(y == tensor_type.tensor4(D(0, s2), D(0, s3), D(0, s4), D(0, s5)))
        self.assertEqual(s.check(), z3.sat)

    def test_add_padding_9(self):
        # [FUNCTION DOC] Adds a Dyn input and a rank-four tensor. Forces x to a vector
        # and y to four dimensions, fixes some flags and finally forces two dimensions
        # to differ from one; expects sat and the model to give equal values for those
        # two dimensions (broadcasting against the last one).
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module adding a Dyn input and a rank-four tensor.
            def forward(self, x: Dyn, y: TensorType([Dyn, Dyn, Dyn, Dyn])):
                # [FUNCTION DOC] HELPER: returns the sum of x and y.
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)

        self.assertEqual(s.check(), z3.sat)
        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        s1, s2, s3, s4, s5, s6, s7 = z3.Ints("s1 s2 s3 s4 s5 s6 s7")
        s.add(x == tensor_type.tensor1(D(s1, s7)))
        s.add(s1 == 1)
        self.assertEqual(s.check(), z3.sat)

        s.add(y == tensor_type.tensor4(D(0, s2), D(0, s3), D(0, s4), D(s6, s5)))
        self.assertEqual(s.check(), z3.sat)

        s.add(s6 == 1)

        self.assertEqual(s.check(), z3.sat)
        s.add(s5 != 1, s7 != 1)
        if not s.check():
            raise AssertionError("Expected s.check() to be truthy")

        if s.model()[s5].as_long() != s.model()[s7].as_long():
            raise AssertionError(
                f"Expected s5 == s7, got {s.model()[s5].as_long()} != "
                f"{s.model()[s7].as_long()}"
            )

    def test_conv_static(self):
        # [FUNCTION DOC] Convolution on a fully static 1 by 2 by 10 by 20 input. Solves
        # for the output dimension variables (last two) and compares them with the real
        # output size. Repeats with a second module configuration and trace and checks
        # again. Raises AssertionError on mismatch.
        s1, s2, s3, s4 = z3.Ints("s1 s2 s3 s4")
        e1, e2, e3, e4 = z3.Ints("e1 e2 e3 e4")
        s11, s22, s33, s44 = z3.Ints("s11 s22 s33 s44")
        e11, e22, e33, e44 = z3.Ints("e11 e22 e33 e44")
        d1, d2, d3, d4 = (
            D(s11, s1),
            D(s22, s2),
            D(s33, s3),
            D(s44, s4),
        )
        b1, b2, b3, b4 = D(e11, e1), D(e22, e2), D(e33, e3), D(e44, e4)

        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module with a convolution layer (note the groups
            # argument is accepted but not forwarded, and bias is left at its default).
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
                # [FUNCTION DOC] HELPER: creates a two-dimensional convolution from the given
                # channel counts, kernel size, stride, padding and dilation, stored as conv1;
                # the groups argument is ignored.
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )

            def forward(self, x: TensorType((1, 2, 10, 20))):
                # [FUNCTION DOC] HELPER: applies conv1 to x.
                return self.conv1(x)

        ast_rewriter = RewritingTracer()

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        res = B.forward(torch.rand(1, 2, 10, 20)).size()

        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)
        solver = z3.Solver()
        solver.add(new_transformed_c)
        self.assertEqual(solver.check(), z3.sat)

        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        solver.add(x == tensor_type.tensor4(d1, d2, d3, d4))
        solver.add(y == tensor_type.tensor4(b1, b2, b3, b4))
        self.assertEqual(solver.check(), z3.sat)
        # print(solver.model())
        if solver.model()[e3].as_long() != res[2]:
            raise AssertionError(
                f"Expected e3 == {res[2]}, got {solver.model()[e3].as_long()}"
            )
        if solver.model()[e4].as_long() != res[3]:
            raise AssertionError(
                f"Expected e4 == {res[3]}, got {solver.model()[e4].as_long()}"
            )

        B2 = BasicBlock(2, 4, 5, 2, 9, 2, 2)
        res2 = B2.forward(torch.rand(1, 2, 10, 20)).size()

        graph2 = ast_rewriter.trace(B2)
        traced2 = GraphModule(ast_rewriter.root, graph2, "gm")
        new_transformed_c = transform_all_constraints(traced2)
        solver = z3.Solver()
        solver.add(new_transformed_c)

        solver.add(x == tensor_type.tensor4(d1, d2, d3, d4))
        solver.add(y == tensor_type.tensor4(b1, b2, b3, b4))

        self.assertEqual(solver.check(), z3.sat)
        if solver.model()[e3].as_long() != res2[2]:
            raise AssertionError(
                f"Expected e3 == {res2[2]}, got {solver.model()[e3].as_long()}"
            )
        if solver.model()[e4].as_long() != res2[3]:
            raise AssertionError(
                f"Expected e4 == {res2[3]}, got {solver.model()[e4].as_long()}"
            )

    def test_reshape_dyn(self):
        # [FUNCTION DOC] Reshapes a Dyn input to two rows with an inferred dimension.
        # After forcing a rank-one input, constrains its size to be 2, 4 or 9 (sat), and
        # adds the constraint that it equals 9 (unsat, since 9 is not divisible by 2).
        s11, s22, s33, s44 = z3.Ints("s11 s22 s33 s44")

        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module reshaping x to two rows.
            def forward(self, x: Dyn):
                # [FUNCTION DOC] HELPER: returns x reshaped to two rows with an inferred width.
                return torch.reshape(x, (2, -1))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        x = z3.Const(1, tensor_type)
        s.add(x == tensor_type.tensor1(D(1, s11)))
        self.assertEqual(s.check(), z3.sat)
        s.add(z3.Or([s11 == 2, s11 == 4, s11 == 9]))
        self.assertEqual(s.check(), z3.sat)
        s.add(s11 == 9)
        self.assertEqual(s.check(), z3.unsat)

    def test_reshape_annotated(self):
        # [FUNCTION DOC] Reshapes a rank-one input to two rows with an inferred width;
        # forcing the input to rank two is expected to be unsat.
        s1, s2, s3, s4 = z3.Ints("s1 s2 s3 s4")
        s11, s22, s33, s44 = z3.Ints("s11 s22 s33 s44")
        d1, d2 = D(s11, s1), D(s22, s2)

        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module reshaping a vector to two rows.
            def forward(self, x: TensorType([Dyn])):
                # [FUNCTION DOC] HELPER: returns x reshaped to two rows with an inferred width.
                return torch.reshape(x, (2, -1))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        x = z3.Const(1, tensor_type)
        s.add(x == tensor_type.tensor2(d1, d2))
        self.assertEqual(s.check(), z3.unsat)

    def test_reshape_static_target(self):
        # [FUNCTION DOC] Reshapes a vector to a static 2 by 3 target. Forcing the input
        # to a vector gives a size of six; requiring a different size is unsat.
        s11, s22, s33, s44 = z3.Ints("s11 s22 s33 s44")

        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module reshaping a vector to 2 by 3.
            def forward(self, x: TensorType([Dyn])):
                # [FUNCTION DOC] HELPER: returns x reshaped to 2 by 3.
                return torch.reshape(x, (2, 3))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced)
        # print(transformed)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        x = z3.Const(1, tensor_type)
        s.add(x == tensor_type.tensor1(D(1, s11)))
        s.check()
        if s.model()[s11].as_long() != 6:
            raise AssertionError(f"Expected s11 == 6, got {s.model()[s11].as_long()}")
        s.add(s11 != 6)
        self.assertEqual(s.check(), z3.unsat)

    def test_reshape_static_target2(self):
        # [FUNCTION DOC] Reshapes a Dyn input to 2 by 3 by 1 by 1; forcing a vector
        # input gives size six and any other size is unsat.
        s11, s22, s33, s44 = z3.Ints("s11 s22 s33 s44")

        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module reshaping x to 2 by 3 by 1 by 1.
            def forward(self, x: Dyn):
                # [FUNCTION DOC] HELPER: returns x reshaped to 2 by 3 by 1 by 1.
                return torch.reshape(x, (2, 3, 1, 1))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        x = z3.Const(1, tensor_type)
        s.add(x == tensor_type.tensor1(D(1, s11)))
        s.check()
        if s.model()[s11].as_long() != 6:
            raise AssertionError(f"Expected s11 == 6, got {s.model()[s11].as_long()}")
        s.add(s11 != 6)
        self.assertEqual(s.check(), z3.unsat)

    def test_conv2D_maxpool2d_flatten(self):
        # [FUNCTION DOC] Small convolutional network (conv, pool, conv, pool, linear,
        # adaptive pool, flatten) on a static 4 by 3 by 32 by 32 input. Compares the
        # solved first two output dimensions (constant numbered 48) with the real
        # output shape.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module with two convolutions, pooling layers, a linear
            # layer and adaptive average pooling.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates the layers conv1, pool, conv2, fc1 and pool2.
                super().__init__()

                self.conv1 = torch.nn.Conv2d(3, 6, 5)
                self.pool = torch.nn.MaxPool2d(2, 2)
                self.conv2 = torch.nn.Conv2d(6, 16, 5)
                self.fc1 = torch.nn.Linear(5, 120)
                self.pool2 = torch.nn.AdaptiveAvgPool2d((6, 7))

            def forward(self, x: TensorType((4, 3, 32, 32))):
                # [FUNCTION DOC] HELPER: applies conv1, pool, conv2, pool, fc1, pool2 and
                # flattens from dimension one.
                out = self.conv1(x)
                out = self.pool(out)
                out = self.conv2(out)
                out = self.pool(out)
                out = self.fc1(out)
                out = self.pool2(out)
                out = torch.flatten(out, 1)
                return out

        B = BasicBlock()
        res = B.forward(torch.rand(4, 3, 32, 32)).shape
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        constraints = transform_all_constraints(traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        solver.check()
        input = z3.Const(1, tensor_type)
        solver.add(input == tensor_type.tensor4(D(1, 4), D(1, 3), D(1, 32), D(1, 32)))
        solver.check()
        output = z3.Const(48, tensor_type)
        if solver.model()[output].arg(0).arg(1) != res[0]:
            raise AssertionError(
                f"Expected arg(0).arg(1) == {res[0]}, "
                f"got {solver.model()[output].arg(0).arg(1)}"
            )
        if solver.model()[output].arg(1).arg(1) != res[1]:
            raise AssertionError(
                f"Expected arg(1).arg(1) == {res[1]}, "
                f"got {solver.model()[output].arg(1).arg(1)}"
            )

    def test_conv2D_maxpool2d_flatten_unsat(self):
        # [FUNCTION DOC] Same network but the input is forced to have last dimension 45
        # instead of 32. Expects unsat.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module with two convolutions, pooling layers, a linear
            # layer and adaptive average pooling.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates the layers conv1, pool, conv2, fc1 and pool2.
                super().__init__()

                self.conv1 = torch.nn.Conv2d(3, 6, 5)
                self.pool = torch.nn.MaxPool2d(2, 2)
                self.conv2 = torch.nn.Conv2d(6, 16, 5)
                self.fc1 = torch.nn.Linear(5, 120)
                self.pool2 = torch.nn.AdaptiveAvgPool2d((6, 7))

            def forward(self, x: TensorType((4, 3, 32, 32))):
                # [FUNCTION DOC] HELPER: applies conv1, pool, conv2, pool, fc1, pool2 and
                # flattens from dimension one.
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
        constraints = transform_all_constraints(traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        solver.check()
        input = z3.Const(1, tensor_type)
        solver.add(input == tensor_type.tensor4(D(1, 4), D(1, 3), D(1, 32), D(1, 45)))
        self.assertEqual(solver.check(), z3.unsat)

    def test_conv2D_maxpool2d_flatten_dyn(self):
        # [FUNCTION DOC] Same network with the batch dimension Dyn. Expects sat.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module with two convolutions, pooling layers, a linear
            # layer and adaptive average pooling.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates the layers conv1, pool, conv2, fc1 and pool2.
                super().__init__()

                self.conv1 = torch.nn.Conv2d(3, 6, 5)
                self.pool = torch.nn.MaxPool2d(2, 2)
                self.conv2 = torch.nn.Conv2d(6, 16, 5)
                self.fc1 = torch.nn.Linear(5, 120)
                self.pool2 = torch.nn.AdaptiveAvgPool2d((6, 7))

            def forward(self, x: TensorType((Dyn, 3, 32, 32))):
                # [FUNCTION DOC] HELPER: applies conv1, pool, conv2, pool, fc1, pool2 and
                # flattens from dimension one.
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
        constraints = transform_all_constraints(traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        self.assertEqual(solver.check(), z3.sat)

    def test_type_check_flatten(self):
        # [FUNCTION DOC] Type-checks flatten over dimensions one to three. A static 2 by
        # 3 by 4 by 5 input is sat with solved result dimensions equal to the real
        # shape. A Dyn third dimension is sat and the flattened dimension is dynamic. A
        # rank-three input with start 10 and end 0 is unsat. Several module classes are
        # defined in turn with the same name.
        s1, s2, s3, s4 = z3.Ints("s1 s2 s3 s4")

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module flattening dimensions one to three.
            def forward(self, x: TensorType([2, 3, 4, 5])):
                # [FUNCTION DOC] HELPER: returns x flattened from dimension one to three.
                return torch.flatten(x, start_dim=1, end_dim=3)

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        constraints = transform_all_constraints(symbolic_traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        self.assertEqual(solver.check(), z3.sat)
        flatten = z3.Const(2, tensor_type)

        res = M().forward(torch.rand(2, 3, 4, 5)).size()
        if solver.model()[flatten].arg(0).arg(1) != res[0]:
            raise AssertionError(
                f"Expected arg(0).arg(1) == {res[0]}, "
                f"got {solver.model()[flatten].arg(0).arg(1)}"
            )
        if solver.model()[flatten].arg(1).arg(1) != res[1]:
            raise AssertionError(
                f"Expected arg(1).arg(1) == {res[1]}, "
                f"got {solver.model()[flatten].arg(1).arg(1)}"
            )

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module flattening dimensions one to three of an input with
            # a Dyn dimension.
            def forward(self, x: TensorType([2, 3, Dyn, 5])):
                # [FUNCTION DOC] HELPER: returns x flattened from dimension one to three.
                return torch.flatten(x, start_dim=1, end_dim=3)

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        constraints = transform_all_constraints(symbolic_traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        self.assertEqual(solver.check(), z3.sat)
        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        solver.add(x == tensor_type.tensor4(D(1, 2), D(1, 3), D(0, s1), D(1, 5)))
        self.assertEqual(solver.check(), z3.sat)
        if solver.model()[y].arg(1).arg(0) != 0:
            raise AssertionError(
                f"Expected arg(1).arg(0) == 0, got {solver.model()[y].arg(1).arg(0)}"
            )

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module flattening with an invalid start and end.
            def forward(self, x: TensorType([2, 3, Dyn])):
                # [FUNCTION DOC] HELPER: returns x flattened from dimension ten to zero.
                return torch.flatten(x, 10, 0)

        module = M()
        # print(module.forward(torch.rand(2,3,5)).shape)
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        constraints = transform_all_constraints(symbolic_traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        self.assertEqual(solver.check(), z3.unsat)


class ConstraintGeneration(unittest.TestCase):
    # [CLASS DOC] Tests that count generated constraints.
    def test_add_reshape(self):
        # [FUNCTION DOC] Generates constraints for an addition of two reshaped Dyn
        # inputs with the ConstraintGenerator and expects the conjunction to contain
        # exactly 11 conjuncts. Raises AssertionError otherwise.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module adding two reshaped inputs.
            def forward(self, x: Dyn, y: Dyn):
                # [FUNCTION DOC] HELPER: returns the sum of x reshaped to 1 by 2 and y to 2 by 2.
                return torch.add(torch.reshape(x, (1, 2)), torch.reshape(y, (2, 2)))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        generator = ConstraintGenerator(traced)
        new_constraints, counter = generator.generate_constraints(0)
        if len(new_constraints.conjucts) != 11:
            raise AssertionError(
                f"Expected len(conjucts) == 11, got {len(new_constraints.conjucts)}"
            )

    def test_conv_reshape_add(self):
        # [FUNCTION DOC] Generates constraints for a convolution of a reshaped Dyn input
        # plus another Dyn input and expects exactly 16 conjuncts.
        class BasicBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: module with an unbiased convolution layer.
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
                # [FUNCTION DOC] HELPER: creates a two-dimensional convolution without bias
                # from the given channel counts, kernel size, stride, padding, groups and
                # dilation, stored as conv1.
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

            def forward(self, x: Dyn, y: Dyn):
                # [FUNCTION DOC] HELPER: reshapes x to 1 by 2 by 10 by 20, applies conv1 and
                # adds y.
                return torch.add(self.conv1(torch.reshape(x, (1, 2, 10, 20))), y)

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        generator = ConstraintGenerator(traced)
        new_constraints, counter = generator.generate_constraints(0)
        if len(new_constraints.conjucts) != 16:
            raise AssertionError(
                f"Expected len(conjucts) == 16, got {len(new_constraints.conjucts)}"
            )


class TestInternalConstraints(unittest.TestCase):
    # [CLASS DOC] Tests of how single constraints are transformed into Z3-ready
    # form with transform_constraint.
    def test_precision(self):
        # [FUNCTION DOC] A precision constraint with Dyn becomes the true constraint.
        # A precision constraint with a three-dimensional type containing one Dyn
        # becomes a conjunction of seven constraints.
        c1 = BinConstraintT(Dyn, TVar("x"), op_precision)
        transformed, _ = transform_constraint(c1, 0)
        if transformed != T():
            raise AssertionError(f"Expected transformed == T(), got {transformed}")

        c2 = BinConstraintT(TensorType([1, Dyn, 3]), TVar("x"), op_precision)
        transformed, counter = transform_constraint(c2, 0)
        if len(transformed.conjucts) != 7:
            raise AssertionError(
                f"Expected len(conjucts) == 7, got {len(transformed.conjucts)}"
            )

    def test_matching(self):
        # [FUNCTION DOC] A matching constraint between a tensor variable and a rank-four
        # tensor type of dimension variables is expected to turn into a disjunction of
        # two constraints.
        c1 = BinConstraintT(
            TVar("x"),
            TensorType([DVar("a"), DVar("b"), DVar("c"), DVar("d")]),
            op_matching,
        )
        transformed, _ = transform_constraint(c1, 0)
        if len(transformed.disjuncts) != 2:
            raise AssertionError(
                f"Expected len(disjuncts) == 2, got {len(transformed.disjuncts)}"
            )

    def test_consistency(self):
        # [FUNCTION DOC] A consistency constraint against a rank-two type with dimension
        # variables becomes a disjunction of five. Transforming the result again keeps
        # five disjuncts.
        c1 = BinConstraintT(
            TVar("x"), TensorType([DVar("a"), DVar("b")]), op_consistency
        )
        transformed, count = transform_constraint(c1, 0)

        if len(transformed.disjuncts) != 5:
            raise AssertionError(
                f"Expected len(disjuncts) == 5, got {len(transformed.disjuncts)}"
            )
        transformed, count = transform_constraint(transformed, count)
        if len(transformed.disjuncts) != 5:
            raise AssertionError(
                f"Expected len(disjuncts) == 5, got {len(transformed.disjuncts)}"
            )

    # def test_apply_broadcasting(self):
    #     c1 = ApplyBroadcasting(TVar(1), TVar(2), TVar(3), TVar(4))
    #     transformed, count = transform_apply_broadcasting(c1, 5)
    #     assert len(transformed.conjucts) == 41


@skipIfNoTorchVision
class TestResNet(unittest.TestCase):
    # [CLASS DOC] Tests running the constraints on a traced torchvision resnet50;
    # skipped when torchvision is missing.
    def test_resnet50_unsat(self):
        # [FUNCTION DOC] Sets all nodes to Dyn and forces a rank-three input; the
        # network needs rank four, so expects unsat.
        traced = symbolic_trace(models.resnet50())
        for n in traced.graph.nodes:
            n.type = Dyn

        constraints = transform_all_constraints(traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        input = z3.Const(1, tensor_type)
        # input with 3 dimensions
        solver.add(input == tensor_type.tensor3(D(1, 1), D(1, 3), D(1, 224)))
        self.assertEqual(solver.check(), z3.unsat)

    def test_resnet50(self):
        # [FUNCTION DOC] Sets all nodes to Dyn, forces a 1 by 3 by 224 by 224 input and
        # expects the final linear output (constant numbered 650) to be the two
        # dimensional type from the real model output size.
        traced = symbolic_trace(models.resnet50())
        for n in traced.graph.nodes:
            n.type = Dyn

        sample_input = torch.randn(1, 3, 224, 224)
        res = models.resnet50().forward(sample_input).size()
        constraints = transform_all_constraints(traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        self.assertEqual(solver.check(), z3.sat)
        linear = z3.Const(650, tensor_type)

        input = z3.Const(1, tensor_type)
        solver.add(input == tensor_type.tensor4(D(1, 1), D(1, 3), D(1, 224), D(1, 224)))
        self.assertEqual(solver.check(), z3.sat)
        expected = tensor_type.tensor2(D(1, res[0]), D(1, res[1]))
        if solver.model()[linear] != expected:
            raise AssertionError(
                f"Expected linear == {expected}, got {solver.model()[linear]}"
            )

    def test_resnet502(self):
        # [FUNCTION DOC] Forces a symbolic batch greater than four with a 3 by 224 by
        # 224 image and checks the batch variable equals the first dimension of the
        # linear output (constant numbered 650).
        traced = symbolic_trace(models.resnet50())
        for n in traced.graph.nodes:
            n.type = Dyn

        constraints = transform_all_constraints(traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        linear = z3.Const(650, tensor_type)
        input = z3.Const(1, tensor_type)
        batch = z3.Int("b")
        solver.add(
            input == tensor_type.tensor4(D(1, batch), D(1, 3), D(1, 224), D(1, 224))
        )
        solver.add(batch > 4)
        solver.check()
        if solver.model()[batch] != solver.model()[linear].arg(0).arg(1):
            raise AssertionError(
                f"Expected batch == linear.arg(0).arg(1), "
                f"got {solver.model()[batch]} != {solver.model()[linear].arg(0).arg(1)}"
            )

    def test_resnet503(self):
        # [FUNCTION DOC] Constrains the input with symbolic batch and the linear output
        # to two symbolic dimensions: sat; requiring the batch to differ from the first
        # output dimension is unsat.
        traced = symbolic_trace(models.resnet50())
        for n in traced.graph.nodes:
            n.type = Dyn

        constraints = transform_all_constraints(traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        linear = z3.Const(650, tensor_type)
        input = z3.Const(1, tensor_type)
        batch, d1, d2 = z3.Ints("b d1 d2")
        solver.add(
            input == tensor_type.tensor4(D(1, batch), D(1, 3), D(1, 224), D(1, 224))
        )
        solver.add(linear == tensor_type.tensor2(D(1, d1), D(1, d2)))
        self.assertEqual(solver.check(), z3.sat)
        solver.add(batch != d1)
        self.assertEqual(solver.check(), z3.unsat)


@skipIfNoTorchVision
class TestAlexNet(unittest.TestCase):
    # [CLASS DOC] Tests running the constraints on a traced torchvision alexnet;
    # skipped when torchvision is missing.
    def test_alexnet1(self):
        # [FUNCTION DOC] Sets all nodes to Dyn, forces a 10 by 3 by 227 by 227 input
        # and checks the solved types of the convolution, relu, pooling, flatten and
        # linear nodes (numbered constants) against known expected shapes and the real
        # final output size. Raises AssertionError on mismatch.
        alexnet = models.alexnet()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(alexnet)

        for n in symbolic_traced.graph.nodes:
            n.type = Dyn

        # print(symbolic_traced)

        res = alexnet.forward(torch.rand(10, 3, 227, 227)).size()
        constraints = transform_all_constraints(symbolic_traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        self.assertEqual(solver.check(), z3.sat)
        input = z3.Const(1, tensor_type)
        conv = z3.Const(2, tensor_type)
        solver.add(
            input == tensor_type.tensor4(D(1, 10), D(1, 3), D(1, 227), D(1, 227))
        )
        self.assertEqual(solver.check(), z3.sat)
        expected_conv = tensor_type.tensor4(D(1, 10), D(1, 64), D(1, 56), D(1, 56))
        if solver.model()[conv] != expected_conv:
            raise AssertionError(
                f"Expected conv == {expected_conv}, got {solver.model()[conv]}"
            )

        relu = z3.Const(7, tensor_type)
        expected_relu = tensor_type.tensor4(D(1, 10), D(1, 64), D(1, 56), D(1, 56))
        if solver.model()[relu] != expected_relu:
            raise AssertionError(
                f"Expected relu == {expected_relu}, got {solver.model()[relu]}"
            )

        maxpool = z3.Const(8, tensor_type)
        expected_maxpool = tensor_type.tensor4(D(1, 10), D(1, 64), D(1, 27), D(1, 27))
        if solver.model()[maxpool] != expected_maxpool:
            raise AssertionError(
                f"Expected maxpool == {expected_maxpool}, got {solver.model()[maxpool]}"
            )

        maxpool2 = z3.Const(42, tensor_type)
        expected_maxpool2 = tensor_type.tensor4(D(1, 10), D(1, 256), D(1, 6), D(1, 6))
        if solver.model()[maxpool2] != expected_maxpool2:
            raise AssertionError(
                f"Expected maxpool2 == {expected_maxpool2}, "
                f"got {solver.model()[maxpool2]}"
            )

        flatten = z3.Const(52, tensor_type)
        expected_flatten = tensor_type.tensor2(D(1, 10), D(1, 9216))
        if solver.model()[flatten] != expected_flatten:
            raise AssertionError(
                f"Expected flatten == {expected_flatten}, got {solver.model()[flatten]}"
            )

        linear = z3.Const(64, tensor_type)
        expected_linear = tensor_type.tensor2(D(1, 10), D(1, 4096))
        if solver.model()[linear] != expected_linear:
            raise AssertionError(
                f"Expected linear == {expected_linear}, got {solver.model()[linear]}"
            )

        linear2 = z3.Const(109, tensor_type)
        expected_linear2 = tensor_type.tensor2(D(1, res[0]), D(1, res[1]))
        if solver.model()[linear2] != expected_linear2:
            raise AssertionError(
                f"Expected linear2 == {expected_linear2}, got {solver.model()[linear2]}"
            )

    def test_alexnet2(self):
        # [FUNCTION DOC] Annotates the placeholder as Dyn by 4 by 227 by 227 (wrong
        # channel count). Expects unsat.
        alexnet = models.alexnet()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(alexnet)

        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType([Dyn, 4, 227, 227])

        constraints = transform_all_constraints(symbolic_traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        self.assertEqual(solver.check(), z3.unsat)

    def test_alexnet3(self):
        # [FUNCTION DOC] Annotates the placeholder as Dyn by Dyn by 227 by 227. Expects
        # sat.
        alexnet = models.alexnet()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(alexnet)

        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType([Dyn, Dyn, 227, 227])

        constraints = transform_all_constraints(symbolic_traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        self.assertEqual(solver.check(), z3.sat)

    def test_alexnet4(self):
        # [FUNCTION DOC] Annotates the placeholder as a rank-three type; alexnet needs
        # rank four. Expects unsat.
        alexnet = models.alexnet()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(alexnet)

        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType([Dyn, Dyn, 227])

        constraints = transform_all_constraints(symbolic_traced, counter=0)
        solver = z3.Solver()
        solver.add(constraints)
        self.assertEqual(solver.check(), z3.unsat)


if __name__ == "__main__":
    unittest.main()
