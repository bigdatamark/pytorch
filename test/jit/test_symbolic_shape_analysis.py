# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for the prototype symbolic shape analysis passes of TorchScript graphs.
# Tests set input tensor types with concrete, unknown or fresh symbolic dimensions, run the
# shape propagation passes (optionally building and partially evaluating the shared shape
# compute graph), and compare inferred symbolic sizes with eager results. Imports an OpInfo
# sample generator and a few testing utilities. No import-time side effects other than imports.

import operator
import unittest
from textwrap import dedent
from typing import Any, List

import torch
from torch import nn, Tensor
from torch.testing import FileCheck
from torch.testing._internal.common_methods_invocations import sample_inputs_cat_concat
from torch.testing._internal.common_utils import make_tensor, raise_on_run_directly
from torch.testing._internal.jit_utils import execWrapper, JitTestCase


# XXX: still in prototype
class TestSymbolicShapeAnalysis(JitTestCase):
    # [CLASS DOC] JitTestCase suite for shape propagation on TorchScript graphs, covering
    # broadcasting, symbolic dim symbols, shared shape graphs, control flow, unary and binary
    # ops (including in-place), convolutions, pooling, arange, embedding, cat, convolution
    # backward and the stitching of per-node shape graphs into one partially evaluated graph.
    # State: setUp saves the process-wide symbolic shapes test mode flag in an attribute and
    # enables it; tearDown restores it.
    def setUp(self):
        # [FUNCTION DOC] Saves the current global symbolic shapes test mode flag in an instance
        # attribute and then turns the mode on. It calls the parent of JitTestCase setUp,
        # skipping JitTestCase setUp itself. Side effect: mutates global JIT state until
        # tearDown.
        super(JitTestCase, self).setUp()
        self.prev_symbolic_shapes_test_enabled = (
            torch._C._jit_symbolic_shapes_test_mode_enabled()
        )
        torch._C._jit_set_symbolic_shapes_test_mode(True)

    def tearDown(self):
        # [FUNCTION DOC] Restores the symbolic shapes test mode flag saved by setUp. It does not
        # call a parent tearDown.
        torch._C._jit_set_symbolic_shapes_test_mode(
            self.prev_symbolic_shapes_test_enabled
        )

    def test_shape_analysis(self):
        # [FUNCTION DOC] Scripts a multiplication of two tensors and propagates shapes after
        # assigning input sizes. Concrete sizes of 1 by 6 by 5 and 1 by 7 by 1 by 5 must
        # broadcast to 1 by 7 by 6 by 5 in the graph text. With all dimensions unknown, the
        # output first dim must equal the second input first dim (longer rank) and the other
        # output dims must be new symbols not in either input. With explicit fresh symbols the
        # output dims must be the broadcast of those symbols.
        @torch.jit.script
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Scripted function returning the product of its two inputs.
            return x * y

        inputs = list(foo.graph.inputs())

        def prop_shapes_on_graph(inp0, inp1):
            # [FUNCTION DOC] HELPER: Sets the sizes of both graph inputs of the scripted
            # function to the given lists (entries may be None or symbols) and runs shape
            # propagation on its graph.
            inputs[0].setType(inputs[0].type().with_sizes(inp0))
            inputs[1].setType(inputs[1].type().with_sizes(inp1))
            torch._C._jit_pass_propagate_shapes_on_graph(foo.graph)

        prop_shapes_on_graph([1, 6, 5], [1, 7, 1, 5])
        FileCheck().check("1, 7, 6, 5").run(foo.graph)

        # None implicitly creates a new symbolic symbol
        prop_shapes_on_graph([None, None], [None, None, None])
        output_shape = foo.graph.findNode("aten::mul").output().type().symbolic_sizes()
        inp0_shape = inputs[0].type().symbolic_sizes()
        inp1_shape = inputs[1].type().symbolic_sizes()

        # output shape dim 0 should be taken from the second inp dim0
        # other two dims we cannot infer and are given a new symbolic shape
        self.assertEqual(output_shape[0], inp1_shape[0])
        self.assertFalse(output_shape[1] in inp0_shape + inp1_shape)
        self.assertFalse(output_shape[2] in inp0_shape + inp1_shape)

        # XXX: symbolic shapes are represented with an increasing counter of unique
        # values, use `_new_symbolic_shape_symbol` api instead of specifying negative
        # dimensions directly so there is no chance of collision between manual number
        # and current counter value.
        sym1 = torch._C._new_symbolic_shape_symbol()
        sym2 = torch._C._new_symbolic_shape_symbol()
        sym3 = torch._C._new_symbolic_shape_symbol()
        prop_shapes_on_graph([sym1, 1, sym3], [1, sym2, sym3])
        output_shape = foo.graph.findNode("aten::mul").output().type().symbolic_sizes()
        self.assertEqual(output_shape[0], sym1)
        self.assertEqual(output_shape[1], sym2)
        self.assertEqual(output_shape[2], sym3)

    def test_shared_shape_graph(self):
        # [FUNCTION DOC] Checks that a multiplication node and a division node in one scripted
        # graph get the same shape compute graph object, and that it is not None.
        @torch.jit.script
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Scripted function returning the product and the quotient of
            # its two inputs.
            return x * y, x / y

        mul_node = foo.graph.findNode("aten::mul")
        div_node = foo.graph.findNode("aten::div")

        mul_graph = torch._C._jit_shape_compute_graph_for_node(mul_node)
        div_graph = torch._C._jit_shape_compute_graph_for_node(div_node)
        self.assertIsNotNone(mul_graph)
        self.assertIs(mul_graph, div_graph)

    def test_write(self):
        # [FUNCTION DOC] Checks that propagation bails out when shapes cannot be computed
        # safely: with unknown inputs of a multiplication, the output stays a plain Tensor type
        # in the graph; and when a view is given a size list that was mutated in place, the view
        # output also stays a plain Tensor.
        @torch.jit.script
        def foo(a, b):
            # [FUNCTION DOC] HELPER: Scripted function returning the product of two inputs.
            return a * b

        # broadcast appends can't be removed, so we bail on propagation
        torch._C._jit_pass_propagate_shapes_on_graph(foo.graph)
        FileCheck().check("Tensor = aten::mul").run(foo.graph)

        @torch.jit.script
        def foo(y):
            # [FUNCTION DOC] HELPER: Scripted function that builds a list of four ints,
            # overwrites its first element, and uses the list as the target shape of a view of
            # the input.
            x = [1, 2, 3, 4]
            x[0] = 5
            return y.view(x)

        torch._C._jit_pass_propagate_shapes_on_graph(foo.graph)
        FileCheck().check("Tensor = aten::view").run(foo.graph)

    def test_if_propagation(self):
        # [FUNCTION DOC] Checks shape propagation through a conditional. After constant
        # propagation, the view node output must have symbolic sizes of unknown, 3, 2, unknown
        # (unknown dims normalized to minus one), and the conditional output, which merges a 2
        # by 3 by 4 by 5 constant tensor with the view result, must keep only the agreed
        # dimension 3: unknown, 3, unknown, unknown.
        @torch.jit.script
        def foo(i: int, z):
            # [FUNCTION DOC] HELPER: Scripted function that creates a ones tensor and a view of
            # the input sized by the integer argument, and returns one or the other depending on
            # whether the integer equals 4.
            x = torch.ones([2, 3, 4, 5])
            y = z.view([z.size(i), 3, 2, z.size(i)])
            if i == 4:
                return x
            else:
                return y

        torch._C._jit_pass_constant_propagation(foo.graph)
        torch._C._jit_pass_propagate_shapes_on_graph(foo.graph)
        view = foo.graph.findNode("aten::view")

        def neg_to_one(li):
            # [FUNCTION DOC] HELPER: Maps a list of dimension values to a list where every
            # negative value is replaced by minus one.
            return [elem if elem >= 0 else -1 for elem in li]

        self.assertEqual(
            neg_to_one(view.output().type().symbolic_sizes()), [-1, 3, 2, -1]
        )
        if_out = next(foo.graph.findNode("prim::If").outputs())
        self.assertEqual(neg_to_one(if_out.type().symbolic_sizes()), [-1, 3, -1, -1])

    def test_unary_shape_functions(self):
        # [FUNCTION DOC] Traces each function in a list of unary ops (currently only hardtanh)
        # on a random 4 by 4 tensor, sets the input size to 2 by 2, propagates shapes, and
        # asserts the output symbolic sizes are 2 by 2.
        unary_ops = [
            torch.nn.functional.hardtanh,
        ]
        for fn in unary_ops:
            t = torch.jit.trace(fn, (torch.rand([4, 4])))
            ten_input = next(t.graph.inputs())
            ten_input.setType(ten_input.type().with_sizes([2, 2]))
            torch._C._jit_pass_propagate_shapes_on_graph(t.graph)
            self.assertEqual(next(t.graph.outputs()).type().symbolic_sizes(), [2, 2])

    def test_unary_shape_fns_inplace(self):
        # [FUNCTION DOC] Same as the unary shape function test but for a scripted function
        # applying an in-place multiplication by 2; the output sizes must be 2 by 2 when the
        # input is set to 2 by 2.
        def mul_inplace(x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Multiplies the input tensor in place by 2 and returns it.
            y = x.mul_(2)
            return y

        unary_ops = [mul_inplace]
        for fn in unary_ops:
            # t = torch.jit.trace(fn, torch.rand([4, 4]))  # For some reason tracing is erroring out.
            t = torch.jit.script(fn)
            ten_input = next(t.graph.inputs())
            ten_input.setType(ten_input.type().with_sizes([2, 2]))
            torch._C._jit_pass_propagate_shapes_on_graph(t.graph)
            self.assertEqual(next(t.graph.outputs()).type().symbolic_sizes(), [2, 2])

    def test_binary_shape_functions(self):
        # [FUNCTION DOC] For each of multiplication, true division, greater-than and addition
        # (operator module functions), traces on 1-D inputs, sets input sizes 1 by 4 by 8 and 4
        # by 1 by 8, propagates shapes, and asserts the output sizes are the broadcast result 4
        # by 4 by 8.
        binary_ops = [
            operator.__mul__,
            operator.__truediv__,
            operator.__gt__,
            operator.__add__,
        ]

        for fn in binary_ops:
            size_1 = [1, 4, 8]
            size_2 = [4, 1, 8]
            t = torch.jit.trace(fn, (torch.rand([4]), torch.rand([4])))
            inputs = list(t.graph.inputs())
            inputs[0].setType(inputs[0].type().with_sizes(size_1))
            inputs[1].setType(inputs[1].type().with_sizes(size_2))
            torch._C._jit_pass_propagate_shapes_on_graph(t.graph)
            self.assertEqual(next(t.graph.outputs()).type().symbolic_sizes(), [4, 4, 8])

    def test_binary_shape_fns_inplace(self):
        # [FUNCTION DOC] Scripts in-place division and addition of two tensors. Only the first
        # input gets sizes 4 by 4 by 8 (in-place ops cannot broadcast); the second input type is
        # deliberately left untouched. Asserts the output sizes are 4 by 4 by 8 after
        # propagation.
        def div_inplace_tensor(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: Divides the first tensor in place by the second and returns
            # the result.
            z = x.div_(y)
            return z

        def add_inplace_tensor(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: Adds the second tensor in place to the first and returns
            # the result.
            z = x.add_(y)
            return z

        binary_ops = [
            div_inplace_tensor,
            add_inplace_tensor,
        ]

        for fn in binary_ops:
            size_1 = [4, 4, 8]  # x (can't broadcast because it's an inplace op)
            t = torch.jit.script(fn)
            inputs = list(t.graph.inputs())
            inputs[0].setType(inputs[0].type().with_sizes(size_1))
            # Intentionally not populate the type of inputs[1]
            torch._C._jit_pass_propagate_shapes_on_graph(t.graph)
            self.assertEqual(next(t.graph.outputs()).type().symbolic_sizes(), [4, 4, 8])

    def test_size_and_sizes(self):
        # [FUNCTION DOC] Checks that sizes read from the second input (individual dims and the
        # whole size list) propagate to the view output. With the second input set to 5, 8 and a
        # fresh symbol, both scripted graphs must produce output sizes 5, 8 and that symbol.
        @torch.jit.script
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Views the first input using a mix of the second input first
            # dim, the constant 8 and the second input last dim.
            return x.view(y.size(0), 8, y.size(-1))

        @torch.jit.script
        def foo2(x, y):
            # [FUNCTION DOC] HELPER: Views the first input using the whole size of the second
            # input.
            return x.view(y.size())

        for graph in [foo.graph, foo2.graph]:
            inputs = list(graph.inputs())
            sym1 = torch._C._new_symbolic_shape_symbol()

            inputs[1].setType(inputs[1].type().with_sizes([5, 8, sym1]))
            torch._C._jit_pass_propagate_shapes_on_graph(graph)
            self.assertEqual(
                next(graph.outputs()).type().symbolic_sizes(), [5, 8, sym1]
            )

    def test_adaptive_avg_pool2d(self):
        # [FUNCTION DOC] For several input shapes and output size specs (ints, tuples, None
        # entries), computes the eager adaptive average pooling output size, traces the pooling,
        # erases non-input shape information, runs peephole and constant propagation, and calls
        # checkShapeAnalysis requiring exact propagation.
        inps = [
            [(1, 64, 8, 9), (5, 7)],
            [(1, 64, 10, 9), (7)],
            [(1, 64, 10, 9), (5, None)],
            [(1, 8, 4, 3), (None, None)],
            [(1, 8, 4, 3), (None, 5)],
        ]

        for inp in inps:
            t = torch.randn(*inp[0])
            out_size = torch.nn.functional.adaptive_avg_pool2d(t, inp[1]).size()

            def foo(x):
                # [FUNCTION DOC] HELPER: Applies adaptive average pooling with the current
                # output size spec of the loop to the input.
                return torch.nn.functional.adaptive_avg_pool2d(x, inp[1])

            fn = torch.jit.trace(foo, (t,))
            torch._C._jit_erase_non_input_shape_information(fn.graph)
            torch._C._jit_pass_peephole(fn.graph)
            torch._C._jit_pass_constant_propagation(fn.graph)
            self.checkShapeAnalysis(out_size, fn.graph, assert_propagation=True)

    def test_conv_deconv(self):
        # [FUNCTION DOC] Iterates over six configurations of 1-D, 2-D and 3-D convolution and
        # transposed convolution (input shape, weight shape, stride, padding, output padding,
        # dilation, groups, function). Computes the eager output size, traces a wrapper, erases
        # non-input shape info, and runs checkShapeAnalysis with exact propagation required.
        for (
            inp_shape,
            weight_shape,
            bias,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            mod,
        ) in [
            ([32, 6, 10], [16, 3, 3], None, 2, 2, 1, 1, 2, torch.nn.functional.conv1d),
            (
                [32, 16, 10],
                [16, 3, 3],
                None,
                2,
                2,
                1,
                1,
                2,
                torch.nn.functional.conv_transpose1d,
            ),
            (
                [1, 32, 5, 10],
                [30, 16, 3, 3],
                None,
                [2, 2],
                [0, 0],
                0,
                1,
                2,
                torch.nn.functional.conv2d,
            ),
            (
                [1, 30, 5, 10],
                [30, 16, 3, 3],
                None,
                [2, 2],
                [0, 0],
                0,
                1,
                2,
                torch.nn.functional.conv_transpose2d,
            ),
            (
                [3, 14, 10, 66, 55],
                [2, 7, 7, 4, 4],
                None,
                1,
                1,
                2,
                1,
                2,
                torch.nn.functional.conv3d,
            ),
            (
                [3, 2, 10, 66, 55],
                [2, 7, 7, 4, 4],
                None,
                1,
                1,
                0,
                1,
                2,
                torch.nn.functional.conv_transpose3d,
            ),
        ]:
            inp = torch.rand(inp_shape)
            weight = torch.rand(weight_shape)
            if mod in [
                torch.nn.functional.conv1d,
                torch.nn.functional.conv2d,
                torch.nn.functional.conv3d,
            ]:
                res = mod(inp, weight, bias, stride, padding, dilation, groups).size()
            else:
                res = mod(
                    inp, weight, bias, stride, padding, output_padding, dilation, groups
                ).size()

            def foo(inp, weight):
                # [FUNCTION DOC] HELPER: Calls the current convolution function with the loop
                # settings; transposed variants also receive output padding, plain convolutions
                # do not.
                if mod in [
                    torch.nn.functional.conv1d,
                    torch.nn.functional.conv2d,
                    torch.nn.functional.conv3d,
                ]:
                    return mod(inp, weight, bias, stride, padding, dilation, groups)
                else:
                    return mod(
                        inp,
                        weight,
                        bias,
                        stride,
                        padding,
                        output_padding,
                        dilation,
                        groups,
                    )

            fn = torch.jit.trace(foo, (inp, weight))
            torch._C._jit_erase_non_input_shape_information(fn.graph)
            torch._C._jit_pass_peephole(fn.graph)
            torch._C._jit_pass_constant_propagation(fn.graph)
            self.checkShapeAnalysis(res, fn.graph, assert_propagation=True)

    def test_arange_shape(self):
        # [FUNCTION DOC] For many argument tuples for arange (ints, floats, negative steps, edge
        # values), generates the source text of a function returning the arange result, compiles
        # it into a CompilationUnit, runs it for the eager size, and calls checkShapeAnalysis
        # with exact propagation and without constant propagation. Tensor constructors have no
        # OpInfo, hence the explicit list.
        # no opinfo for tensor constructors
        inps = [
            (10,),
            (10, 10),
            (0, 10),
            (0, 1000),
            (1, -1, -1),
            (1, 0, -1),
            (1, 2, 1),
            (0.6, 0.89, 0.1),
            (1, 10, 0.3),
            (1, 10, 4),
            (0.6, 0.7, 0.8),
            (1, 10, 0.3),
            # (True,),  TODO: https://github.com/pytorch/pytorch/issues/63405
            # (False,), TODO: https://github.com/pytorch/pytorch/issues/63405
            (0, 5),
            (0, 5, 2),
            (0, 5 + 1e-6),
            (0, 5 - 1e-6),
            (10, -1 + 1e-6, -1),
            (10, -1, -1),
            (10, -1 - 1e-6, -1),
        ]

        for inp in inps:
            funcs_template = dedent(
                """
            def func():
                return torch.arange({args})
            """
            )

            inp_s = str(inp)[1:-1]  # remove tuple parens
            funcs_str = funcs_template.format(args=inp_s)
            scope = {}
            execWrapper(funcs_str, globals(), scope)
            cu = torch.jit.CompilationUnit(funcs_str)
            self.checkShapeAnalysis(
                list(cu.func().size()),
                cu.func.graph,
                assert_propagation=True,
                constant_prop=False,
            )

    def test_shape_embedding_bag(self):
        # [FUNCTION DOC] Runs under no-grad. For pairs of int64 index tensors and Embedding
        # modules (various sparse, max norm and empty settings), computes the eager size of an
        # embedding lookup with the module weight and options, traces a lookup, and calls
        # checkShapeAnalysis with exact propagation and no constant propagation.
        # KNOWN ISSUE: the test name says embedding bag, but the code exercises the plain
        #     embedding function only.
        # TODO: merge into opinfos, having difficulties there
        with torch.no_grad():

            def make_arg(shape, low=None, high=None):
                # [FUNCTION DOC] HELPER: Creates an int64 CPU tensor of the given shape with
                # optional low and high bounds and no gradient.
                return make_tensor(
                    shape,
                    device="cpu",
                    dtype=torch.int64,
                    low=low,
                    high=high,
                    requires_grad=False,
                )

            nn_inps = (
                (
                    make_arg((40,), 0, 9),
                    torch.nn.Embedding(20, embedding_dim=64, max_norm=1.0),
                ),
                (make_arg((2, 4), 0, 9), torch.nn.Embedding(10, 20, sparse=True)),
                (make_arg((0,)), torch.nn.Embedding(0, 0, sparse=True)),
                (make_arg((2, 4), 0, 9), torch.nn.Embedding(10, 0, sparse=True)),
                (make_arg((4,), 0, 21), torch.nn.Embedding(22, 5, max_norm=1.0)),
                (
                    make_arg((2,), 0, 1),
                    torch.nn.Embedding.from_pretrained(
                        torch.arange(6.0).view(2, 3),
                        max_norm=2.0,
                        norm_type=0.5,
                        scale_grad_by_freq=False,
                        sparse=True,
                    ),
                ),
            )

            for inp, module in nn_inps:
                kwargs = {
                    "weight": module.weight.detach(),
                    "padding_idx": module.padding_idx,
                    "max_norm": module.max_norm,
                    "norm_type": module.norm_type,
                    "scale_grad_by_freq": module.scale_grad_by_freq,
                    "sparse": module.sparse,
                }

                out_size = torch.nn.functional.embedding(inp, **kwargs).size()

                def foo(x):
                    # [FUNCTION DOC] HELPER: Ignores its argument and returns the embedding
                    # lookup of the enclosing loop index tensor using the keyword options built
                    # for the current module.
                    return torch.nn.functional.embedding(inp, **kwargs)

                fn = torch.jit.trace(foo, (inp.detach(),), check_trace=False)

                self.checkShapeAnalysis(
                    out_size, fn.graph, assert_propagation=True, constant_prop=False
                )

    def test_shape_concat(self):
        # [FUNCTION DOC] Takes cat sample inputs (CPU float, no grad). For samples with exactly
        # two tensors, scripts a module that concatenates the two inputs along the sample dim,
        # sets the graph input sizes from the tensors, and calls checkShapeAnalysis with exact
        # propagation. Other samples are skipped.
        # TODO: unify with opinfo tests, traces of lists dont preserve sizes in IR
        sample_inputs = sample_inputs_cat_concat(None, "cpu", torch.float, False)

        class CatMod(nn.Module):
            # [CLASS DOC] HELPER: Module that concatenates two tensors along a dim kept as a
            # constant attribute.
            __constants__ = ["dim"]

            def __init__(self, dim=0):
                # [FUNCTION DOC] HELPER: Stores the concatenation dim, default zero.
                super().__init__()
                self.dim = dim

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the concatenation of the two inputs along the
                # stored dim.
                return torch.cat([x, y], dim=self.dim)

        for inp in sample_inputs:
            mod = torch.jit.script(CatMod(**inp.kwargs).eval())

            args = inp.input

            # This test is hard-coded only to work with two sample inputs
            # but the OpInfo may have more/less
            if len(args) != 2:
                continue

            out_size = mod(*args).size()
            inps = list(mod.graph.inputs())
            inps[1].setType(inps[1].type().with_sizes(args[0].size()))
            inps[2].setType(inps[2].type().with_sizes(args[1].size()))
            self.checkShapeAnalysis(out_size, mod.graph, assert_propagation=True)

    def assert_shape_equal_scripted(self, script_fn, given_ins):
        # [FUNCTION DOC] HELPER: Runs a scripted function on given tensors, sets each graph
        # input type to the size of the matching tensor, and calls checkShapeAnalysis with the
        # list of sizes of the outputs and exact propagation required. Asserts the number of
        # inputs matches the graph input count.
        expected_res = script_fn(*given_ins)
        g = script_fn.graph
        graph_ins = list(g.inputs())
        self.assertEqual(len(given_ins), len(graph_ins))
        for inp, graph_in in zip(given_ins, graph_ins):
            graph_in.setType(graph_in.type().with_sizes(inp.size()))

        out_sizes = [out.size() for out in expected_res]
        self.checkShapeAnalysis(out_sizes, g, assert_propagation=True)

    def test_convolution_backward(self):
        # [FUNCTION DOC] Checks shape analysis of the convolution backward op, which is not
        # exposed in the Python API and has no OpInfo. Two scripted functions, one with an
        # explicit bias size list and one with no bias sizes, are compared with eager output
        # sizes (grad input, weight and bias shapes) via the scripted helper.
        # No opinfos for ops that are not part of the Python API
        # Also, as the return shapes are the input, weight, and bias shape, there is no point
        # in a really complicated test

        input = torch.randn(
            (16, 16, 8, 8), dtype=torch.float32, device="cpu", requires_grad=True
        )
        weight = torch.randn(
            (8, 4, 3, 3), dtype=torch.float32, device="cpu", requires_grad=True
        )
        out_grad = torch.randn((16, 8, 8, 8), dtype=torch.float32, device="cpu")

        @torch.jit.script
        def conv_bwd(input, weight, grad):
            # [FUNCTION DOC] HELPER: Calls the convolution backward op with a fixed bias size
            # list of one element and fixed stride, padding, dilation and group arguments.
            bias_sizes = [
                8,
            ]
            args = ([1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, True])
            return torch.ops.aten.convolution_backward(
                grad, input, weight, bias_sizes, *args
            )

        self.assert_shape_equal_scripted(conv_bwd, (input, weight, out_grad))

        @torch.jit.script
        def conv_bwd_2(input, weight, grad):
            # [FUNCTION DOC] HELPER: Same as the other scripted function but with no bias sizes.
            bias_sizes = None
            args = ([1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, True])
            return torch.ops.aten.convolution_backward(
                grad, input, weight, bias_sizes, *args
            )

        self.assert_shape_equal_scripted(conv_bwd_2, (input, weight, out_grad))

    def test_returning_input_symbolic_shapes(self):
        # [FUNCTION DOC] Builds a frozen scripted Conv2d with unknown input sizes, builds the
        # shape compute graph, partially evaluates it, and runs it as a function on the dims 20,
        # 16, 5, 10. The first four results must be the input dims; the last two must equal the
        # spatial dims of the eager output.
        mm = torch.jit.freeze(torch.jit.script(nn.Conv2d(16, 33, 3, stride=2).eval()))
        inps = list(mm.graph.inputs())
        inps[1].setType(inps[1].type().with_sizes([None, None, None, None]))
        shape_compute_graph = (
            torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mm.graph)
        )
        g = shape_compute_graph.partial_eval_shape_graph()
        # to make into a jit function can't have multiple outputs
        g.makeMultiOutputIntoTuple()
        func = torch._C._create_function_from_graph("partial_eval_graph", g)
        out = func([20, 16, 5, 10])
        # first four outputs should be unknown symbolic shapes from input
        self.assertEqual(out[0:4], [20, 16, 5, 10])
        # last two are two new symbolic dims - height and width
        self.assertEqual(out[4:], list(mm(torch.rand([20, 16, 5, 10])).size()[2:]))

    def test_partial_eval_graph_conv(self):
        # [FUNCTION DOC] Builds a frozen scripted Conv2d and its shape compute graph. Checks the
        # output has symbolic (negative) dims at indices 0, 2 and 3 and a concrete channel dim
        # at index 1, then evaluates the partially evaluated graph on a concrete input shape and
        # compares results with the eager output dims other than the channel.
        mm = torch.jit.freeze(torch.jit.script(nn.Conv2d(16, 33, 3, stride=2).eval()))
        shape_compute_graph = (
            torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mm.graph)
        )
        output_sizes = (
            mm.graph.findNode("aten::conv2d").output().type().symbolic_sizes()
        )
        # calculating 0, 2 and 3 index
        for i in [0, 2, 3]:
            self.assertTrue(output_sizes[i] < 0)
        self.assertTrue(output_sizes[1] >= 0)
        g = shape_compute_graph.partial_eval_shape_graph()
        # to make into a jit function can't have multiple outputs
        g.makeMultiOutputIntoTuple()
        func = torch._C._create_function_from_graph("partial_eval_graph", g)
        inp = torch.randn(20, 16, 5, 10)
        output = func([20, 16, 5, 10])
        output_eager = list(mm(inp).size())
        for o, oe in zip(output, output_eager[0:1] + output_eager[2:]):
            self.assertEqual(o, oe)

    def checkSymShapeCompute(
        self, shape_compute_graph, nodes, node_output_sizes, shape_inputs
    ):
        # [FUNCTION DOC] HELPER: Verifies a partially evaluated shape graph. Asserts its input
        # count equals the number of given shape inputs, maps each graph output to its symbolic
        # dim, runs the graph on the shape inputs, and for every node checks each symbolic dim
        # of the output type against the expected size: non-negative dims must equal it
        # directly, negative ones are looked up in the graph results.
        g = shape_compute_graph.partial_eval_shape_graph()
        self.assertTrue(len(list(g.inputs())) == len(shape_inputs))
        output_sym_map = shape_compute_graph.graph_output_to_symbolic_shape_dim()
        # map from sym shape -> index
        sym_shape_to_index = {}
        for index, output in enumerate(g.outputs()):
            sym_shape_to_index[output_sym_map[output]] = index

        g.makeMultiOutputIntoTuple()
        func = torch._C._create_function_from_graph("partial_eval_graph", g)
        sym_outputs = func(*shape_inputs)

        for node, output_shape in zip(nodes, node_output_sizes):
            output_type_sizes = node.output().type().symbolic_sizes()
            for i, sym_shape in enumerate(output_type_sizes):
                if sym_shape >= 0:
                    self.assertEqual(sym_shape, output_shape[i])
                else:
                    sym_shape_index = sym_shape_to_index[sym_shape]
                    self.assertEqual(sym_outputs[sym_shape_index], output_shape[i])

    def test_partial_eval_stitching(self):
        # [FUNCTION DOC] Freezes a scripted sequential model of convolution, max pooling and
        # convolution, builds the stitched shape compute graph, and checks symbolic shape
        # evaluation for the max pool node and both convolution nodes against the eager
        # intermediate output sizes for a 1 by 3 by 224 by 224 input.
        conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        max_pool = torch.nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )
        conv2 = nn.Conv2d(
            64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

        mod = torch.jit.freeze(
            torch.jit.script(nn.Sequential(conv1, max_pool, conv2).eval())
        )

        conv1_output = conv1(torch.rand(1, 3, 224, 224))
        max_pool_output = max_pool(conv1_output)
        conv2_output = conv2(max_pool_output)

        shape_compute_graph = (
            torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mod.graph)
        )
        nodes = [mod.graph.findNode("aten::max_pool2d")] + list(
            mod.graph.findAllNodes("aten::conv2d")
        )
        output_shapes = [
            max_pool_output.size(),
            conv1_output.size(),
            conv2_output.size(),
        ]
        self.checkSymShapeCompute(
            shape_compute_graph, nodes, output_shapes, ([1, 3, 224, 224],)
        )

    def test_refinement_through_graph_stitching(self):
        # [FUNCTION DOC] Builds a module applying two identical convolutions to the same input
        # and adding them. After plain propagation with unknown input dims the two outputs have
        # different spatial symbols; after building the compute graph the symbols are equal,
        # showing the stitched graphs refine equivalence.
        class TwoConvs(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two Conv2d layers, applied to the same input and
            # summed.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two identical Conv2d layers with 3 input
                # channels and 64 outputs.
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
                )
                self.conv2 = torch.nn.Conv2d(
                    3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the sum of the two convolution outputs for the
                # same input.
                a = self.conv1(x)
                b = self.conv2(x)
                return a + b

        mod = torch.jit.freeze(torch.jit.script(TwoConvs()).eval())
        inp_tensor = list(mod.graph.inputs())[1]
        inp_tensor.setType(inp_tensor.type().with_sizes([None, None, None, None]))
        torch._C._jit_pass_propagate_shapes_on_graph(mod.graph)
        outs = list(next(mod.graph.outputs()).node().inputs())
        out1 = outs[0].type().symbolic_sizes()
        out2 = outs[1].type().symbolic_sizes()
        self.assertTrue(out1[2] != out2[2])
        self.assertTrue(out1[3] != out2[3])
        # by joining partial eval graphs of both convs we are able to recognize the output shapes
        # are equivalent
        torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mod.graph)
        out1 = outs[0].type().symbolic_sizes()
        out2 = outs[1].type().symbolic_sizes()
        self.assertEqual(out1, out2)

    def test_stitching_multi_output(self):
        # [FUNCTION DOC] Traces and freezes a max pool returning indices, sets unknown input
        # dims, lowers tuples, builds the compute graph, and checks both outputs of the pooling
        # node share symbolic sizes. Runs the partially evaluated graph on the real size: first
        # four results equal the input dims and the rest equal the eager spatial dims.
        max_pool = torch.nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            ceil_mode=False,
            return_indices=True,
        )
        tensor = torch.rand(1, 3, 224, 224)
        mod = torch.jit.trace(max_pool, (tensor,))
        mod = torch.jit.freeze(mod.eval())
        inp = list(mod.graph.inputs())[1]
        inp.setType(inp.type().with_sizes([None, None, None, None]))
        output_tensor = list(mod(tensor)[0].size())
        self.run_pass("lower_all_tuples", mod.graph)
        shape_compute_graph = (
            torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mod.graph)
        )
        max_pool_node = mod.graph.findNode("aten::max_pool2d_with_indices")
        outs = list(max_pool_node.outputs())
        self.assertEqual(
            outs[0].type().symbolic_sizes(), outs[1].type().symbolic_sizes()
        )
        g = shape_compute_graph.partial_eval_shape_graph()
        # to make into a jit function can't have multiple outputs
        g.makeMultiOutputIntoTuple()
        func = torch._C._create_function_from_graph("partial_eval_graph", g)
        mapping = shape_compute_graph.graph_output_to_symbolic_shape_dim()  # noqa: F841
        output_shape = func(tensor.size())
        # the first 4 dims are input sym dimensions, then the ,
        self.assertEqual(list(output_shape[0:4]), list(tensor.size()))
        self.assertEqual(list(output_shape[4:]), output_tensor[2:])

    def test_sym_ir_parsing(self):
        # [FUNCTION DOC] Parses a graph from text whose input type has symbolic dims, and checks
        # the symbolic sizes of the input come back as the two given negative symbols.
        graph_str1 = """graph(%x.1 : Float(SS(-2), SS(-3))):
                        %3 : int = prim::Constant[value=1]()
                        %4 : Tensor = aten::add(%x.1, %x.1, %3)
                        return (%4)"""
        g = torch._C.parse_ir(graph_str1)
        inp = next(g.inputs())
        out = inp.type().symbolic_sizes()
        self.assertEqual(out, [-2, -3])

    def test_stitching_concat(self):
        # [FUNCTION DOC] For scripted functions adding a quotient to a concatenation (default
        # dim and dim minus 2), sets unknown 2-D input sizes, builds the compute graph, and
        # checks symbolic shape evaluation for the division, addition and cat nodes against the
        # expected size lists for four given input shapes.
        @torch.jit.script
        def foo1(a, b, x, y):
            # [FUNCTION DOC] HELPER: Returns the quotient of the first two inputs plus the
            # concatenation of the last two along the default dim.
            return (a / b) + torch.cat([x, y])

        @torch.jit.script
        def foo2(a, b, x, y):
            # [FUNCTION DOC] HELPER: Returns the quotient of the first two inputs plus the
            # concatenation of the last two along dim minus 2.
            return (a / b) + torch.cat([x, y], dim=-2)

        for foo in [foo1, foo2]:
            g = foo.graph
            for inp in foo.graph.inputs():
                inp.setType(inp.type().with_sizes([None, None]))

            shape_compute_graph = (
                torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(
                    foo.graph
                )
            )
            nodes = (
                [g.findNode("aten::div")]
                + [g.findNode("aten::add")]
                + [g.findNode("aten::cat")]
            )

            inps = [1, 10], [20, 10], [15, 1], [5, 1]
            output_shapes = [[20, 10], [20, 10], [20, 1]]

            self.checkSymShapeCompute(shape_compute_graph, nodes, output_shapes, inps)

    @unittest.skipIf(
        not hasattr(torch.jit, "_shapes"), "shape functions not loaded in python"
    )
    def test_shape_function_includes(self):
        # [FUNCTION DOC] Skipped if torch.jit has no shape function module. Calls the Python
        # shape functions for conv2d and matmul directly and checks results 1 by 33 by 2 by 4
        # and 10 by 10.
        inp_shape = [1, 16, 5, 10]
        weight_shape = [33, 16, 3, 3]
        bias = None
        stride = [2, 2]
        padding = [0, 0]
        dilation = [1, 1]
        groups = 1
        res = torch.jit._shapes.conv2d(
            inp_shape, weight_shape, bias, stride, padding, dilation, groups
        )
        self.assertEqual(res, [1, 33, 2, 4])

        m1_shape = [10, 20]
        m2_shape = [20, 10]
        res = torch.jit._shapes.matmul(m1_shape, m2_shape)
        self.assertEqual(res, [10, 10])

    def test_register_function_error_checking(self):
        # [FUNCTION DOC] Checks that registering an invalid shape compute graph for an add node
        # raises a RuntimeError: input types that are not supertypes of the schema (message
        # mentions expected supertype of int), a wrong output type (message mentions graph
        # type), and too many inputs (message mentions fewer arguments than schema).
        # Registration fails before touching the global map.
        # this will error before registering on global map, so
        # no issue in overwriting schema mappings
        @torch.jit.script
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Scripted add of two tensors, used to obtain the add node.
            return x + y

        node = foo.graph.findNode("aten::add")

        @torch.jit.script
        def wrong_input_types(x, y):
            # [FUNCTION DOC] HELPER: Scripted function returning an empty int list; its inputs
            # do not match the schema.
            x: List[int] = []
            return x

        with self.assertRaisesRegex(RuntimeError, "Expected supertype of int"):
            torch._C._jit_register_shape_compute_graph_for_node(
                node, wrong_input_types.graph
            )

        @torch.jit.script
        def wrong_output_types(x: List[int], y: List[int]):
            # [FUNCTION DOC] HELPER: Scripted function with two int list inputs returning an
            # empty tensor list; the output type is wrong.
            x: List[Tensor] = []
            return x

        with self.assertRaisesRegex(RuntimeError, "but got graph_type"):
            torch._C._jit_register_shape_compute_graph_for_node(
                node, wrong_output_types.graph
            )

        @torch.jit.script
        def too_many_inputs(x: List[int], y: List[int], z: Any, z2: Any):
            # [FUNCTION DOC] HELPER: Scripted function with four inputs, more than the add
            # schema allows, returning an empty int list.
            x: List[int] = []
            return x

        with self.assertRaises(RuntimeError) as error:
            torch._C._jit_register_shape_compute_graph_for_node(
                node, too_many_inputs.graph
            )

        self.assertTrue("fewer arguments than schema" in str(error.exception))

    def test_cross_entropy_loss(self):
        # [FUNCTION DOC] Sets input sizes 8 by 2 and 8 for the inputs of a scripted unreduced
        # cross entropy loss, propagates shapes, and asserts the output sizes are 8.
        @torch.jit.script
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Calls the cross entropy loss op with reduction mode zero
            # (no reduction).
            return torch.ops.aten.cross_entropy_loss(x, y, reduction=0)

        inputs = list(foo.graph.inputs())
        inputs[0].setType(inputs[0].type().with_sizes([8, 2]))
        inputs[1].setType(
            inputs[1]
            .type()
            .with_sizes(
                [
                    8,
                ]
            )
        )
        torch._C._jit_pass_propagate_shapes_on_graph(foo.graph)
        self.assertEqual(
            next(foo.graph.outputs()).type().sizes(),
            [
                8,
            ],
        )

    def test_squeeze_dims(self):
        # [FUNCTION DOC] Sets input sizes 1 by 5 by 8 for a scripted squeeze of dim 0,
        # propagates shapes, and asserts the output sizes are 5 by 8.
        @torch.jit.script
        def foo(x):
            # [FUNCTION DOC] HELPER: Squeezes dim zero of the input via the aten op.
            return torch.ops.aten.squeeze(x, dim=0)

        input = next(foo.graph.inputs())
        input.setType(input.type().with_sizes([1, 5, 8]))
        torch._C._jit_pass_propagate_shapes_on_graph(foo.graph)
        self.assertEqual(next(foo.graph.outputs()).type().symbolic_sizes(), [5, 8])


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
