# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for the TorchScript peephole optimization passes and related graph cleanups:
# type_as removal, list length and index folding, short-circuit boolean simplification, conv dim
# folding, normalization of rsub and identity comparisons, integer refinement, adding zero,
# dictionary lookups and length, and list slicing. Tests script or trace small functions, run a
# named pass, and use FileCheck to assert on the resulting graph text. No import-time side
# effects; running the file directly raises an error pointing at test_jit.

import unittest
from typing import Callable, List

import torch
from torch import nn
from torch.testing import FileCheck
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import _inline_everything, JitTestCase, RUN_CUDA


class TestPeephole(JitTestCase):
    # [CLASS DOC] JitTestCase suite for peephole and related passes. Each test builds its own
    # scripted or traced function and checks both the optimized graph and, where possible, the
    # numerical result.
    def test_peephole_with_writes(self):
        # [FUNCTION DOC] Checks that a function accumulating its input into a local with in-place
        # adds scripts correctly with the peephole pass active.
        def test_write(x):
            # [FUNCTION DOC] HELPER: Starts from the integer zero, adds the input to it twice with
            # augmented assignment, and returns the sum.
            s = 0
            s += x
            s += x
            return s

        self.checkScript(test_write, (torch.ones(4, 4),))

    def test_peephole_with_non_output_writes(self):
        # [FUNCTION DOC] Checks that peephole does not remove an in-place write to a value only
        # consumed by an ignored Python function call.
        @torch.jit.ignore
        def nomnom(x):
            # [FUNCTION DOC] HELPER: Ignored (not compiled) Python function with an empty body; it
            # only gives the graph a side-effecting use of its argument.
            pass

        def test_write(x):
            # [FUNCTION DOC] HELPER: Clones the input, adds zero to produce y, mutates the clone
            # in place, passes the clone to the ignored function, and returns y plus y.
            t = torch.ones_like(x)
            z = x.clone()
            y = z + 0
            z.add_(t)
            # this makes sure z isn't blasted out of existence
            # because it isn't returned or used in a side-effectful
            # way
            nomnom(z)
            return y + y

        a = torch.ones(4, 4)
        self.checkScript(test_write, (a,))

    def test_peephole_no_output_aliasing(self):
        # [FUNCTION DOC] Checks that when a function returns its input and the input plus zero,
        # the compiled result gives two separate tensors, verified by differing data pointers.
        def test_peephole(x):
            # [FUNCTION DOC] HELPER: Returns the input and a value computed by adding zero to it.
            y = x + 0
            return x, y

        a = torch.ones(4, 4)
        j = self.checkScript(test_peephole, (a,))
        r1, r2 = j(a)
        self.assertNotEqual(r1.data_ptr(), r2.data_ptr())

    def test_peephole(self):
        # [FUNCTION DOC] Traces a type-cast-to-other function with same-dtype inputs, checks the
        # cast op appears and is removed by the peephole pass; then traces with differing dtypes
        # and runs peephole again.
        # KNOWN ISSUE: the final assertion compares the saved graph string with the string of that
        # string, so it is always true and does not test that the graph is unchanged.
        a = torch.tensor([0.4])
        b = torch.tensor([0.7])
        c = torch.tensor([0], dtype=torch.int32)

        def f(x, y):
            # [FUNCTION DOC] HELPER: Casts the first input to the dtype of the second.
            return x.type_as(y)

        tf = torch.jit.trace(f, (a, b))
        FileCheck().check("type_as").run(str(tf.graph))
        self.run_pass("peephole", tf.graph)
        FileCheck().check_not("type_as").run(str(tf.graph))
        tf2 = torch.jit.trace(f, (a, c))
        s = str(tf2.graph)
        self.run_pass("peephole", tf2.graph)
        self.assertEqual(s, str(s))

    def test_peephole_dynamic(self):
        # [FUNCTION DOC] Scripts the cast-to-other-tensor-type function, whose input dtypes are
        # unknown, and checks the peephole pass leaves the graph text unchanged.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Casts the first input to the dtype of the second.
            return x.type_as(y)

        fn = torch.jit.script(f)
        s = str(fn.graph)
        torch._C._jit_pass_peephole(fn.graph)
        self.assertEqual(s, str(fn.graph))

    def test_peephole_list_ops(self):
        # [FUNCTION DOC] Checks peephole folding of list operations in five scripted functions:
        # the length of a literal list becomes a constant; a list grown in a loop keeps its length
        # call; constant in-range indices into a literal list are folded; an out-of-range negative
        # index is kept; and a grown list keeps its index operation.
        @torch.jit.script
        def foo(x, y, z):
            # [FUNCTION DOC] HELPER: Returns the length of a literal list of its three inputs.
            return len([x, y, z])

        self.run_pass("peephole", foo.graph)
        FileCheck().check("value=3").check_next("return").run(foo.graph)

        @torch.jit.script
        def foo(x, y, z):
            # [FUNCTION DOC] HELPER: Builds a three-element list, appends the first input in a
            # loop, and returns the length of a fresh literal list.
            li = [x, y, z]
            for _ in range(len(x)):
                li.append(x)
            return len([x, y, z])

        self.run_pass("peephole", foo.graph)
        FileCheck().check_not("aten::len").run(foo.graph)

        @torch.jit.script
        def foo(x, y, z):
            # [FUNCTION DOC] HELPER: Builds a three-element list and returns its entries at index
            # one and at index minus two.
            li = [x, y, z]
            return li[1], li[-2]

        FileCheck().check("aten::__getitem__").run(foo.graph)
        self.run_pass("peephole", foo.graph)
        FileCheck().check_not("aten::__getitem__").run(foo.graph)

        @torch.jit.script
        def foo(x, y, z):
            # [FUNCTION DOC] HELPER: Builds a three-element list and indexes it at the
            # out-of-range index minus seven.
            li = [x, y, z]
            return li[-7]

        self.run_pass("peephole", foo.graph)
        FileCheck().check("aten::__getitem__").run(foo.graph)

        @torch.jit.script
        def foo(x, y, z):
            # [FUNCTION DOC] HELPER: Builds a list, appends in a loop so its length is not static,
            # and returns the entry at index minus two.
            li = [x, y, z]
            for _ in range(len(x)):
                li.append(x)
            return li[-2]

        self.run_pass("peephole", foo.graph)
        FileCheck().check("aten::__getitem__").run(foo.graph)

    @unittest.skipIf(not RUN_CUDA, "cpp tests require CUDA")
    def test_peephole_cuda(self):
        # [FUNCTION DOC] Skipped without CUDA. Traces a cast-to-other-tensor-type function with a
        # CPU input and a CUDA target: peephole must leave the graph unchanged. With both inputs
        # on CUDA the cast is removed after peephole and dead code elimination.
        a = torch.tensor([0.4], device="cpu")
        b = torch.tensor([0.7], device="cuda")
        c = torch.tensor([0.7], device="cuda")

        def f(x, y):
            # [FUNCTION DOC] HELPER: Casts the first input to the dtype and device of the second.
            return x.type_as(y)

        trace = torch.jit.trace(f, (a, c))
        s = str(trace.graph)
        self.run_pass("peephole", trace.graph)
        self.assertEqual(s, str(trace.graph))
        trace = torch.jit.trace(f, (b, c))
        self.run_pass("peephole", trace.graph)
        self.run_pass("dce", trace.graph)
        FileCheck().check_not("type_as").run(str(trace.graph))

    @_inline_everything
    def test_peephole_type_refinements(self):
        # [FUNCTION DOC] Run with all calls inlined. Checks that an optional-refinement cast is
        # removed by peephole when the argument is known non-None, but kept when refinement comes
        # from an isinstance check on a scalar from the tensor, in which case both int and float
        # inputs still script correctly.
        def refine(x):
            # [FUNCTION DOC] HELPER: Returns the input if it is not None, otherwise a default
            # tensor. Takes an optional tensor, per the legacy comment-style signature below.
            # type: (Optional[Tensor]) -> Tensor
            return x if x is not None else torch.tensor(3)

        @torch.jit.script
        def test():
            # [FUNCTION DOC] HELPER: Scripted function calling the refine helper with a tensor.
            return refine(torch.tensor(4))

        FileCheck().check("prim::unchecked_cast").run(test.graph)
        self.run_pass("peephole", test.graph)
        FileCheck().check_not("prim::unchecked_cast").run(test.graph)

        # refinement not optimized out
        def is_int_tensor(x):
            # [FUNCTION DOC] HELPER: Extracts a Python scalar from a tensor; returns it plus three
            # if it is an int, otherwise eight.
            scalar = x.item()
            if isinstance(scalar, int):
                return scalar + 3
            else:
                return 8

        self.checkScript(is_int_tensor, (torch.tensor(2),))
        self.checkScript(is_int_tensor, (torch.tensor(2.5),))
        graph = torch.jit.script(is_int_tensor).graph
        self.run_pass("peephole", graph)
        FileCheck().check("prim::unchecked_cast").run(graph)

    def test_short_circuit_optimization(self):
        # [FUNCTION DOC] Checks boolean short-circuit simplification: after constant propagation
        # an expression and-ed with false or or-ed with true has no branch or equality node, and
        # after peephole and-ing true or or-ing false drops the branch while keeping the
        # comparison. Also checks results.
        @torch.jit.script
        def const_expressions(x):
            # [FUNCTION DOC] HELPER: Scripted function returning a pair: an equality test and-ed
            # with false, and an equality test or-ed with true; the signature is given by the
            # legacy comment on the next line.
            # type: (int) -> Tuple[bool, bool]
            return x == 1 and False, x == 1 or True

        self.run_pass("constant_propagation", const_expressions.graph)
        FileCheck().check_not("prim::If").check_not("aten::eq").run(
            const_expressions.graph
        )
        self.assertEqual(const_expressions(1), (False, True))

        @torch.jit.script
        def redundant_expressions(x):
            # [FUNCTION DOC] HELPER: Scripted function returning a pair: an equality test and-ed
            # with true, and an equality test or-ed with false; the signature is given by the
            # legacy comment on the next line.
            # type: (int) -> Tuple[bool, bool]
            return x == 1 and True, x == 1 or False

        self.run_pass("peephole", redundant_expressions.graph)
        self.assertEqual(redundant_expressions(1), (True, True))
        self.assertEqual(redundant_expressions(0), (False, False))
        # and True / or False are removed from graph
        FileCheck().check("aten::eq").check_not("prim::If").run(
            redundant_expressions.graph
        )

    def test_conv_dim_folding(self):
        # [FUNCTION DOC] For each of the 1d, 2d and 3d convolution classes, scripts a module that
        # applies the convolution and returns the rank of the output, runs inline and peephole,
        # and expects no conv or dim ops (rank folded to a constant). A second module that resizes
        # the output in place must keep both ops.
        modules = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
        for mod in modules:

            class ConvDim(torch.nn.Module):
                # [CLASS DOC] HELPER: Module with one bias-free convolution of the loop-selected
                # class; forward returns the rank of the convolved input.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Builds the convolution layer with 3 input channels,
                    # 32 output channels, kernel 3 and stride 2.
                    super().__init__()
                    self.conv = mod(3, 32, kernel_size=3, stride=2, bias=False)

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Applies the convolution and returns the number of
                    # dimensions of the result.
                    x = self.conv(x)
                    return x.dim()

            conv_dim = torch.jit.script(ConvDim())
            self.run_pass("inline", conv_dim.graph)
            self.run_pass("peephole", conv_dim.graph)
            FileCheck().check_not("conv").check_not("dim").run(conv_dim.graph)

            class ConvDimMutate(torch.nn.Module):
                # [CLASS DOC] HELPER: Like the module above but forward also resizes the
                # convolution output in place before returning its rank, which blocks the folding.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Builds the convolution layer with 3 input channels,
                    # 32 output channels, kernel 3 and stride 2.
                    super().__init__()
                    self.conv = mod(3, 32, kernel_size=3, stride=2, bias=False)

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Convolves, resizes the result in place to a 4 by 4
                    # shape, and returns its number of dimensions.
                    x = self.conv(x)
                    x.resize_([4, 4])
                    return x.dim()

            conv_dim = torch.jit.script(ConvDimMutate())
            self.run_pass("inline", conv_dim.graph)
            self.run_pass("peephole", conv_dim.graph)
            FileCheck().check("conv").check("dim").run(conv_dim.graph)

    def test_normalized_rsub(self):
        # [FUNCTION DOC] Checks the reverse subtraction op is normalized to ordinary subtraction
        # when scripted: the graph has exactly two subtraction ops and no reverse subtraction, and
        # results match eager.
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])

        def convertible_rsub(x, y):
            # [FUNCTION DOC] HELPER: Returns the plain difference of the inputs and the reverse
            # subtraction of them in swapped order.
            return (x - y), torch.rsub(y, x)

        self.checkScript(convertible_rsub, (a, b))
        op_graph = torch.jit.script(convertible_rsub).graph
        FileCheck().check_count("aten::sub", 2, exactly=True).run(op_graph)
        FileCheck().check_count("aten::rsub", 0, exactly=True).run(op_graph)

    def test_normalized_is_op(self):
        # [FUNCTION DOC] Checks that identity comparisons among booleans script to three equality
        # ops with no identity-comparison op left, and match eager results.
        def convertible_is_op(x: bool, y: bool):
            # [FUNCTION DOC] HELPER: Returns three identity comparisons: input with true, false
            # with input, and the two inputs.
            return x is True, False is x, x is y

        self.checkScript(convertible_is_op, (True, False))

        op_graph = torch.jit.script(convertible_is_op).graph
        FileCheck().check_count("aten::eq", 3, exactly=True).run(op_graph)
        FileCheck().check_count("aten::__is__", 0, exactly=True).run(op_graph)

    def test_normalized_isnot_op(self):
        # [FUNCTION DOC] Checks that negated identity comparisons among booleans script to three
        # not-equal ops with no negated identity op left, and match eager results.
        def convertible_isnot_op(x: bool, y: bool):
            # [FUNCTION DOC] HELPER: Returns three negated identity comparisons: input with true,
            # false with input, and the two inputs.
            return x is not True, False is not x, x is not y

        self.checkScript(convertible_isnot_op, (True, False))

        op_graph = torch.jit.script(convertible_isnot_op).graph
        FileCheck().check_count("aten::ne", 3, exactly=True).run(op_graph)
        FileCheck().check_count("aten::__isnot__", 0, exactly=True).run(op_graph)

    def test_peephole_list_len(self):
        # [FUNCTION DOC] Checks the list-idioms peephole pass with list length refinement: guards
        # that raise unless lengths match constants let later length calls fold to constants,
        # which constant propagation then reduces to a single returned constant. Also checks which
        # tuple outputs become constant for several assert and branch patterns, and that nothing
        # folds when the length is mutated or only one of two conditions must hold.
        def run_peephole_and_check_const_value(graph, const_string):
            # [FUNCTION DOC] HELPER: Runs the list-idioms pass with length refinement, then
            # constant propagation on the graph, and checks that the given constant text appears
            # followed immediately by the return.
            torch._C._jit_pass_peephole_list_idioms(graph, refine_list_len=True)
            self.run_pass("constant_propagation", graph)
            FileCheck().check(const_string).check_next("return").run(graph)

        def gen_li(inp_len: int):
            # [FUNCTION DOC] HELPER: Returns a list of zeros with the requested length.
            return [0 for i in range(inp_len)]

        @torch.jit.script
        def foo(x: List[int], y: List[int]):
            # [FUNCTION DOC] HELPER: Raises an exception unless the first list has four items and
            # the second five; returns the sum of the lengths.
            if len(x) != 4 or len(y) != 5:
                raise Exception("")  # noqa: TRY002

            return len(x) + len(y)

        run_peephole_and_check_const_value(foo.graph, "value=9")
        self.assertEqual(foo(gen_li(4), gen_li(5)), 9)
        with self.assertRaises(Exception):
            foo(2, 4)

        @torch.jit.script
        def foo(x: List[int], y: List[int]):
            # [FUNCTION DOC] HELPER: Like the previous function but the check is phrased as an
            # equality test with a pass branch and an else branch that raises.
            if len(x) == 4 and len(y) == 5:
                pass
            else:
                raise Exception("hi")  # noqa: TRY002

            return len(x) + len(y)

        run_peephole_and_check_const_value(foo.graph, "value=9")
        self.assertEqual(foo(gen_li(4), gen_li(5)), 9)
        with self.assertRaises(Exception):
            foo(2, 4)

        @torch.jit.script
        def foo(x: List[int], y: List[int], z: List[int]):
            # [FUNCTION DOC] HELPER: Nested checks that raise unless the three lists have lengths
            # four, eight and three; returns the first length plus the product of the other two.
            if len(x) != 4:
                raise Exception("..")  # noqa: TRY002
            else:
                if len(y) != 8:
                    raise Exception("...")  # noqa: TRY002
                else:
                    if len(z) == 3:
                        pass
                    else:
                        raise Exception("...")  # noqa: TRY002

            return len(x) + len(y) * len(z)

        run_peephole_and_check_const_value(foo.graph, "value=28")
        self.assertEqual(foo(gen_li(4), gen_li(8), gen_li(3)), 28)
        with self.assertRaises(Exception):
            foo(1, 2, 3)

        # refinement should persist in second len(x) call

        @torch.jit.script
        def foo(x: List[int], cond: bool):
            # [FUNCTION DOC] HELPER: Returns the length of the list inside nested checks on the
            # length and a flag; all paths return four.
            if len(x) == 4:
                if cond:
                    return len(x)
                return 4

            return 4

        run_peephole_and_check_const_value(foo.graph, "value=4")

        def test_const_tuple_output(graph, const_inputs):
            # [FUNCTION DOC] HELPER: Finds the tuple construction node in the graph and asserts
            # that inputs at the listed positions are constants and all others are not.
            tup = graph.findNode("prim::TupleConstruct")
            for i, elem in enumerate(tup.inputs()):
                if i in const_inputs:
                    self.assertIsNotNone(elem.toIValue())
                else:
                    self.assertIsNone(elem.toIValue())

        # testing combinations of x1 : {True, False} x
        # {then/else branch} x assert {True/False}

        @torch.jit.script
        def foo(x: List[int], b: List[int]):
            # [FUNCTION DOC] HELPER: Sets a flag in a branch on the first length, asserts the flag
            # is false, and returns both lengths; only the second length can be inferred (as four).
            if len(x) == 5:
                x1 = True
            else:
                x1 = len(b) != 4
            assert x1 == False  # noqa: E712, S101 TODO: canonicalize x is False to aten::eq
            return len(x), len(b)

        torch._C._jit_pass_peephole_list_idioms(foo.graph, refine_list_len=True)
        torch._C._jit_pass_constant_propagation(foo.graph)
        # we can only infer len(b) == 4 here
        test_const_tuple_output(foo.graph, [1])

        @torch.jit.script
        def foo(x: List[int], b: List[int]):
            # [FUNCTION DOC] HELPER: Like the previous function but the then-branch sets the flag
            # false; nothing can be inferred.
            if len(x) == 5:
                x1 = False
            else:
                x1 = len(b) != 4
            assert x1 == False  # noqa: E712, S101 TODO: canonicalize x is False to aten::eq
            return len(x), len(b)

        torch._C._jit_pass_peephole_list_idioms(foo.graph, refine_list_len=True)
        torch._C._jit_pass_constant_propagation(foo.graph)
        # can't infer anything
        test_const_tuple_output(foo.graph, [])

        @torch.jit.script
        def foo(x: List[int], b: List[int]):
            # [FUNCTION DOC] HELPER: Like the earlier flag function but the else-branch tests
            # equality with four; nothing can be inferred.
            if len(x) == 5:
                x1 = True
            else:
                x1 = len(b) == 4
            assert x1 == False  # noqa: E712, S101 TODO: canonicalize x is False to aten::eq
            return len(x), len(b)

        torch._C._jit_pass_peephole_list_idioms(foo.graph, refine_list_len=True)
        torch._C._jit_pass_constant_propagation(foo.graph)
        # we can't infer anything, only len(b) != 4
        test_const_tuple_output(foo.graph, [])

        @torch.jit.script
        def foo(x: List[int], b: List[int]):
            # [FUNCTION DOC] HELPER: Same shape as the first flag function: the second length can
            # be inferred to be four.
            if len(x) == 5:
                x1 = True
            else:
                x1 = len(b) != 4
            assert x1 == False  # noqa: E712, S101 TODO: canonicalize x is False to aten::eq
            return len(x), len(b)

        torch._C._jit_pass_peephole_list_idioms(foo.graph, refine_list_len=True)
        torch._C._jit_pass_constant_propagation(foo.graph)
        # can infer len(b) == 4
        test_const_tuple_output(foo.graph, [1])

        # swap branches
        @torch.jit.script
        def foo(x: List[int], b: List[int]):
            # [FUNCTION DOC] HELPER: Same as above with the branch condition inverted and the
            # branches swapped; the second length can be inferred.
            if len(x) != 5:
                x1 = len(b) != 4
            else:
                x1 = True
            assert x1 == False  # noqa: E712, S101 TODO: canonicalize x is False to aten::eq
            return len(x), len(b)

        torch._C._jit_pass_peephole_list_idioms(foo.graph, refine_list_len=True)
        torch._C._jit_pass_constant_propagation(foo.graph)
        # can infer len(b) == 4
        test_const_tuple_output(foo.graph, [1])

        # use __not__
        @torch.jit.script
        def foo(x: List[int], b: List[int]):
            # [FUNCTION DOC] HELPER: Same as above but the assert uses logical negation of the
            # flag; the second length can be inferred.
            if len(x) != 5:
                x1 = len(b) != 4
            else:
                x1 = True
            assert not x1  # noqa: S101
            return len(x), len(b)

        torch._C._jit_pass_peephole_list_idioms(foo.graph, refine_list_len=True)
        torch._C._jit_pass_constant_propagation(foo.graph)
        # can infer len(b) == 4
        test_const_tuple_output(foo.graph, [1])

        # Test unsuccessful optimizations

        @torch.jit.script
        def foo(x: List[int]):
            # [FUNCTION DOC] HELPER: Asserts the length is four, then appends to the list and
            # returns the length; the mutation prevents folding so two length calls remain.
            assert len(x) == 4  # noqa: S101
            x.append(3)
            return len(x)

        torch._C._jit_pass_peephole_list_idioms(foo.graph, refine_list_len=True)
        self.run_pass("constant_propagation", foo.graph)
        FileCheck().check_count("aten::len", 2).run(foo.graph)

        @torch.jit.script
        def foo(x: List[int], y: List[int]):
            # [FUNCTION DOC] HELPER: Asserts an or of two length conditions, which proves neither
            # alone; returns the sum of the lengths, so four length calls remain.
            assert len(x) == 4 or len(y) == 5  # noqa: S101
            return len(x) + len(y)

        torch._C._jit_pass_peephole_list_idioms(foo.graph, refine_list_len=True)
        self.run_pass("constant_propagation", foo.graph)
        FileCheck().check_count("aten::len", 4).run(foo.graph)

    def test_integer_refinement(self):
        # [FUNCTION DOC] Checks the integer refinement pass on integer arguments: guards that
        # raise unless an argument equals a constant let later uses fold to constants. Also checks
        # results and the raising path, and that an or-ed assert does not fold.
        def run_peephole_and_check_const_value(graph, const_string):
            # [FUNCTION DOC] HELPER: Runs refine-integer-values, constant propagation and dead
            # code elimination on the graph, then checks the given constant text appears followed
            # immediately by the return.
            self.run_pass("refine_integer_values", graph)
            self.run_pass("constant_propagation", graph)
            self.run_pass("dce", graph)
            FileCheck().check(const_string).check_next("return").run(graph)

        @torch.jit.script
        def foo(x: int, y: int):
            # [FUNCTION DOC] HELPER: Raises unless the first argument is four and the second five;
            # returns their sum.
            if x != 4 or y != 5:
                raise Exception("")  # noqa: TRY002

            return x + y

        graph = foo.graph
        self.run_pass("refine_integer_values", graph)
        self.run_pass("constant_propagation", graph)
        self.run_pass("dce", graph)

        run_peephole_and_check_const_value(foo.graph, "value=9")
        self.assertEqual(foo(4, 5), 9)
        with self.assertRaises(Exception):
            foo(2, 4)

        @torch.jit.script
        def foo(x: int, y: int):
            # [FUNCTION DOC] HELPER: Like the previous function but written as an equality test
            # with a pass branch and a raising else branch.
            if x == 4 and y == 5:
                pass
            else:
                raise Exception("hi")  # noqa: TRY002

            return x + y

        run_peephole_and_check_const_value(foo.graph, "value=9")
        self.assertEqual(foo(4, 5), 9)
        with self.assertRaises(Exception):
            foo(2, 4)

        @torch.jit.script
        def foo(x: int, y: int, z: int):
            # [FUNCTION DOC] HELPER: Nested checks that raise unless the arguments are four, eight
            # and three; returns the first plus the product of the others.
            if x != 4:
                raise Exception("..")  # noqa: TRY002
            else:
                if y != 8:
                    raise Exception("...")  # noqa: TRY002
                else:
                    if z == 3:
                        pass
                    else:
                        raise Exception("...")  # noqa: TRY002

            return x + y * z

        run_peephole_and_check_const_value(foo.graph, "value=28")
        self.assertEqual(foo(4, 8, 3), 28)
        with self.assertRaises(Exception):
            foo(1, 2, 3)

        # refinement should persist in second len(x) call

        @torch.jit.script
        def foo(x: int, cond: bool):
            # [FUNCTION DOC] HELPER: Returns the first argument inside nested checks on its value
            # and a flag; all paths return four.
            if x == 4:
                if cond:
                    return x
                return 4

            return 4

        run_peephole_and_check_const_value(foo.graph, "value=4")

        @torch.jit.script
        def foo(x: int, y: int):
            # [FUNCTION DOC] HELPER: Asserts an or of two equalities, which proves neither;
            # returns the sum. The test expects the add to remain after a list-idioms pass.
            assert x == 4 or y == 5  # noqa: S101
            return x + y

        torch._C._jit_pass_peephole_list_idioms(foo.graph, refine_list_len=True)
        self.run_pass("constant_propagation", foo.graph)
        FileCheck().check("aten::add").run(foo.graph)

    def test_optimize_out_comparison_same_value(self):
        # [FUNCTION DOC] Checks that self-comparisons for equality and inequality are optimized
        # away by peephole for an int and for a list of ints, and that results match eager.
        # KNOWN ISSUE: the check_not looks for the op name aten::neq, but the actual inequality op
        # is named aten::ne, so that half of the check cannot fail.
        def foo(x: int):
            # [FUNCTION DOC] HELPER: Returns the equality and inequality of an int with itself.
            return x == x, x != x

        def foo2(x: List[int]):
            # [FUNCTION DOC] HELPER: Returns equality and inequality of an int list with itself.
            return x == x, x != x

        for func, inp in zip([foo, foo2], [1, [2, 3]]):
            func_s = torch.jit.script(func)
            self.run_pass("peephole", func_s.graph)
            FileCheck().check_not("aten::eq").check_not("aten::neq").run(func_s.graph)
            self.assertEqual(func(inp), func_s(inp))

    def test_peephole_add_zero(self):
        # [FUNCTION DOC] Scripts a function adding zero on either side of an int, runs peephole
        # and checks the result is still the pair of the input.
        # KNOWN ISSUE: the FileCheck for no add op is built but never run, so the graph is not
        # actually checked.
        @torch.jit.script
        def foo(x: int):
            # [FUNCTION DOC] HELPER: Returns a pair: the input plus zero and zero plus the input.
            return x + 0, 0 + x

        self.run_pass("peephole", foo.graph)
        FileCheck().check_not("aten::add")
        self.assertEqual(foo(3), (3, 3))

    def test_noop_peephole(self):
        # [FUNCTION DOC] Checks four functions where adding zero must stay (plain input, mutated
        # tensor, tensor also returned, float zero), each keeping exactly one add. Then checks
        # removal: a chain of add zero then multiply by one needs two peephole runs, and a float
        # zero is removed only when shape peepholes are enabled and the input has known shape.
        # test unsuccessful
        def foo1(x):
            # [FUNCTION DOC] HELPER: Returns the input plus zero.
            return x + 0

        def foo2():
            # [FUNCTION DOC] HELPER: Makes a zero tensor, subtracts three in place, and returns it
            # plus zero.
            x = torch.zeros([2, 2])
            x.sub_(3)
            return x + 0

        def foo3():
            # [FUNCTION DOC] HELPER: Returns a zero tensor together with that tensor plus zero.
            x = torch.zeros([2, 2])
            return x, x + 0

        def foo4():
            # [FUNCTION DOC] HELPER: Returns a zero tensor plus the float zero.
            x = torch.zeros([2, 2])
            return x + 0.0

        funcs = foo1, foo2, foo3, foo4
        inps = (torch.ones([2]),), (), (), ()
        for func, inp in zip(funcs, inps):
            foo_s = torch.jit.script(func)
            self.run_pass("peephole", foo_s.graph)
            FileCheck().check_count("aten::add", 1, exactly=True).run(foo_s.graph)
            self.assertEqual(func(*inp), foo_s(*inp))

        # successful
        def func(x):
            # [FUNCTION DOC] HELPER: Returns the input plus zero, times one, minus five.
            return (x + 0) * 1 - 5

        func_s = torch.jit.script(func)
        self.run_pass("peephole", func_s.graph)
        # bail on modified value first
        FileCheck().check_not("aten::add").check("aten::mul").run(func_s.graph)
        # second run it should succeed
        self.run_pass("peephole", func_s.graph)
        FileCheck().check_not("aten::add").check_not("aten::mul").run(func_s.graph)
        self.assertEqual(func(torch.ones([2, 2])), func_s(torch.ones([2, 2])))

        def func(x):
            # [FUNCTION DOC] HELPER: Returns the input plus the float zero, minus five.
            return (x + 0.0) - 5

        func_s = torch.jit.script(func)
        inp = next(func_s.graph.inputs())
        inp.setType(torch._C.TensorType.create_from_tensor(torch.rand([2, 2])))
        torch._C._jit_pass_peephole(func_s.graph, disable_shape_peepholes=True)
        FileCheck().check("aten::add").run(func_s.graph)
        torch._C._jit_pass_peephole(func_s.graph, disable_shape_peepholes=False)
        FileCheck().check_not("aten::add").run(func_s.graph)

    def test_refine_integer_values(self):
        # [FUNCTION DOC] Checks that after refining integer values, constant propagation and dead
        # code elimination, a function returning a value equal to the argument has a graph whose
        # body is only the return, and that results are correct.
        @torch.jit.script
        def foo(x: int):
            # [FUNCTION DOC] HELPER: Returns the constant one when the input equals one, otherwise
            # returns the input.
            y = 1
            if x == 1:
                return y
            else:
                return x

        self.run_pass("refine_integer_values", foo.graph)
        self.run_pass("constant_propagation", foo.graph)
        self.run_pass("dce", foo.graph)
        FileCheck().check("graph").check_next("return").run(foo.graph)
        self.assertEqual(foo(2), 2)
        self.assertEqual(foo(1), 1)

    def test_peephole_len_list(self):
        # [FUNCTION DOC] Checks length folding of a size list: the length call over a size list is
        # replaced once the input has a known rank of two, but is kept when the size list is
        # mutated by an append.
        @torch.jit.script
        def foo(x):
            # [FUNCTION DOC] HELPER: Returns the length of the size of its input.
            return len(x.size())

        self.run_pass("peephole", foo.graph)
        FileCheck().check("aten::len").run(foo.graph)
        inputs = list(foo.graph.inputs())
        inputs[0].setType(inputs[0].type().with_sizes([None, None]))
        self.run_pass("peephole", foo.graph)
        FileCheck().check_not("aten::len").run(foo.graph)
        self.assertEqual(2, foo(torch.rand([3, 1])))

        @torch.jit.script
        def foo(x):
            # [FUNCTION DOC] HELPER: Takes the size list of the input, appends four, and returns
            # its length.
            li = x.size()
            li.append(4)
            return len(li)

        inputs = list(foo.graph.inputs())
        inputs[0].setType(inputs[0].type().with_sizes([None, None]))
        self.run_pass("peephole", foo.graph)
        FileCheck().check("aten::len").run(foo.graph)
        self.assertEqual(3, foo(torch.rand([3, 1])))

    def test_peephole_optional_refine(self):
        # [FUNCTION DOC] Sets the type of the branch output of a scripted function to an optional
        # int and checks that peephole refines it so that no optional int type remains.
        @torch.jit.script
        def foo(z: int, z2: int, cond: bool):
            # [FUNCTION DOC] HELPER: Returns one of two int arguments depending on a boolean.
            if cond:
                return z
            else:
                return z2

        out = next(foo.graph.findNode("prim::If").outputs())
        out.setType(torch._C.OptionalType(torch._C.IntType.get()))
        self.run_pass("peephole", foo.graph)
        FileCheck().check_not("int?").run(foo.graph)

    def test_peephole_int(self):
        # [FUNCTION DOC] Checks that the scalar-to-int conversion op is removed by peephole once
        # the input type is changed to int.
        @torch.jit.script
        def foo(x):
            # [FUNCTION DOC] HELPER: Converts the input number to an int; the signature is given
            # by the legacy comment on the next line.
            # type: (number)
            return int(x)

        FileCheck().check("aten::Int").run(foo.graph)
        next(foo.graph.inputs()).setType(torch._C.IntType.get())
        self.run_pass("peephole", foo.graph)
        FileCheck().check_not("aten::Int").run(foo.graph)

    def test_peephole_arith(self):
        # [FUNCTION DOC] Checks that peephole simplifies arithmetic with constants (adding or
        # subtracting a constant, multiplying by one, floor division by one, division by one) so
        # the graph keeps only the final list construction and return, and results are correct.
        @torch.jit.script
        def foo(input0: int, input1: int, input2: int, input3: int):
            # [FUNCTION DOC] HELPER: Computes two chains of constant arithmetic on the int inputs
            # and returns a list holding one integer and one converted value.
            _1 = torch.add(input1, 2)
            _3 = torch.add(input3, 2)
            _5 = torch.add(1, torch.sub(_1, 3) // 1)
            _6 = torch.add(1 * torch.sub(_3, 3) // 1, 1) / 1
            return [_5, int(_6)]

        FileCheck().check("aten::add").check("aten::sub").check("aten::mul").check(
            "aten::floordiv"
        ).check("aten::div").run(foo.graph)
        self.run_pass("peephole", foo.graph)
        FileCheck().check("graph").check("):").check_next("ListConstruct").check_next(
            "return"
        ).run(foo.graph)
        self.assertEqual(foo(0, 1, 2, 3), [1, 3])

    def test_peephole_dict_getitem_simple(self):
        # [FUNCTION DOC] Checks that peephole replaces lookups with constant keys in a dictionary
        # literal by the stored values, for int, string and float keys, so no dictionary
        # construction or lookup remains.
        @torch.jit.script
        def foo(a: int, b: int):
            # [FUNCTION DOC] HELPER: Builds a dictionary with int keys zero and one and returns
            # the values for keys one and zero.
            d = {0: a, 1: b}
            x = d[1]
            y = d[0]
            return x, y

        self.run_pass("peephole", foo.graph)
        FileCheck().check_not("DictConstruct").check_not("__getitem__").run(foo.graph)
        self.assertEqual(foo(0, 1), (1, 0))

        @torch.jit.script
        def foo(a: int, b: int):
            # [FUNCTION DOC] HELPER: Same with string keys.
            d = {"0": a, "1": b}
            x = d["1"]
            y = d["0"]
            return x, y

        self.run_pass("peephole", foo.graph)
        FileCheck().check_not("DictConstruct").check_not("__getitem__").run(foo.graph)
        self.assertEqual(foo(0, 1), (1, 0))

        @torch.jit.script
        def foo(a: int, b: int):
            # [FUNCTION DOC] HELPER: Same with float keys.
            d = {0.0: a, 1.0: b}
            x = d[1.0]
            y = d[0.0]
            return x, y

        self.run_pass("peephole", foo.graph)
        FileCheck().check_not("DictConstruct").check_not("__getitem__").run(foo.graph)
        self.assertEqual(foo(0, 1), (1, 0))

    def test_peephole_dict_getitem_no_optimization_missing_key(self):
        # [FUNCTION DOC] Checks that a lookup of a key absent from a dictionary literal is not
        # optimized away.
        @torch.jit.script
        def foo():
            # [FUNCTION DOC] HELPER: Builds a one-entry dictionary and looks up a missing key.
            d = {0: 1}
            return d[2]

        self.run_pass("peephole", foo.graph)
        FileCheck().check("DictConstruct").check("__getitem__").run(foo.graph)

    def test_peephole_dict_getitem_no_optimization_get_input_arg(self):
        # [FUNCTION DOC] Checks that a lookup with a non-constant key is not folded, since the key
        # might not be present; the result for key zero is checked.
        # Here we don't know if the input arg is in the dict, so we can't
        # make the optimization.
        @torch.jit.script
        def foo(a: int):
            # [FUNCTION DOC] HELPER: Builds a one-entry dictionary and looks up the key given by
            # the argument.
            d = {0: 1}
            return d[a]

        self.run_pass("peephole", foo.graph)
        FileCheck().check("DictConstruct").check("__getitem__").run(foo.graph)
        self.assertEqual(foo(0), 1)

    def test_peephole_dict_getitem_no_optimization_dict_modified(self):
        # [FUNCTION DOC] Checks that a lookup in a dictionary mutated by assignment is not folded,
        # and the result reflects the assignment.
        @torch.jit.script
        def foo():
            # [FUNCTION DOC] HELPER: Builds a one-entry dictionary, overwrites its entry, and
            # returns the entry.
            d = {0: 1}
            d[0] = 2
            return d[0]

        self.run_pass("peephole", foo.graph)
        FileCheck().check("DictConstruct").check("__getitem__").run(foo.graph)
        self.assertEqual(foo(), 2)

    def test_peephole_dict_getitem_no_optimization_overlapping_keys(self):
        # [FUNCTION DOC] Checks that a lookup is not folded when the dictionary literal
        # repeats a key.
        @torch.jit.script
        def foo():
            # [FUNCTION DOC] HELPER: Builds a dictionary literal where the key zero appears twice
            # and looks it up.
            d = {0: 1, 0: 2}  # noqa: F601
            return d[0]

        self.run_pass("peephole", foo.graph)
        FileCheck().check("DictConstruct").check("__getitem__").run(foo.graph)

    def test_peephole_dict_getitem_no_optimization_keys_might_overlap(self):
        # [FUNCTION DOC] Checks that a lookup is not folded when a runtime key might equal a
        # constant key.
        @torch.jit.script
        def foo(x: int):
            # [FUNCTION DOC] HELPER: Builds a dictionary with constant key zero and a runtime key
            # and looks up the runtime key.
            d = {0: 1, x: 2}
            return d[x]

        self.run_pass("peephole", foo.graph)
        FileCheck().check("DictConstruct").check("__getitem__").run(foo.graph)

    def test_peephole_dict_getitem_no_optimization_unsupported_type(self):
        # [FUNCTION DOC] Checks that a lookup is not folded when keys are tensors, an unsupported
        # key type for this optimization.
        @torch.jit.script
        def foo():
            # [FUNCTION DOC] HELPER: Builds a dictionary keyed by a random tensor and looks it up.
            a = torch.rand((2, 2))
            d = {a: 1}
            return d[a]

        self.run_pass("peephole", foo.graph)
        FileCheck().check("DictConstruct").check("__getitem__").run(foo.graph)
        self.assertEqual(foo(), 1)

    def test_peephole_dict_len(self):
        # [FUNCTION DOC] Checks that peephole replaces the length of a dictionary literal with
        # distinct constant keys by its constant size, and results are correct.
        @torch.jit.script
        def foo():
            # [FUNCTION DOC] HELPER: Builds a two-entry dictionary and returns its length.
            d = {0: 1, 1: 2}
            return len(d)

        self.run_pass("peephole", foo.graph)
        FileCheck().check_not("DictConstruct").check_not("len").run(foo.graph)
        self.assertEqual(foo(), 2)

    def test_peephole_dict_len_no_optimization_overlapping_keys(self):
        # [FUNCTION DOC] Checks that the length of a dictionary literal with a repeated key is not
        # folded and equals one.
        @torch.jit.script
        def foo():
            # [FUNCTION DOC] HELPER: Builds a dictionary literal where key zero appears twice and
            # returns its length.
            d = {0: 1, 0: 2}  # noqa: F601
            return len(d)

        self.run_pass("peephole", foo.graph)
        FileCheck().check("DictConstruct").check("len").run(foo.graph)
        self.assertEqual(foo(), 1)

    def test_peephole_dict_len_no_optimization_keys_might_overlap(self):
        # [FUNCTION DOC] Checks that the length is not folded when a runtime key might equal a
        # constant key.
        @torch.jit.script
        def foo(x: int):
            # [FUNCTION DOC] HELPER: Builds a dictionary with constant key zero and a runtime key
            # and returns its length.
            d = {0: 1, x: 2}
            return len(d)

        self.run_pass("peephole", foo.graph)
        FileCheck().check("DictConstruct").check("len").run(foo.graph)

    def test_peephole_dict_len_no_optimization_unsupported_type(self):
        # [FUNCTION DOC] Checks that the length of a dictionary keyed by tensors is not folded,
        # and the length is one.
        @torch.jit.script
        def foo():
            # [FUNCTION DOC] HELPER: Builds a dictionary keyed by a random tensor and returns
            # its length.
            a = torch.rand((2, 2))
            d = {a: 1}
            return len(d)

        self.run_pass("peephole", foo.graph)
        FileCheck().check("DictConstruct").check("len").run(foo.graph)
        self.assertEqual(foo(), 1)

    def test_peephole_slice_all_three_args(self):
        # [FUNCTION DOC] Checks that slicing a list literal with start, stop and step constants is
        # folded by peephole so no slice op remains, and results match eager.
        def foo(x: int):
            # [FUNCTION DOC] HELPER: Slices a seven-element list literal containing the input with
            # start minus five, stop six, step two.
            return [1, 2, x, 4, 5, 6, 7][-5:6:2]

        graph = torch.jit.script(foo).graph
        self.run_pass("peephole", graph)
        FileCheck().check_not("aten::slice").run(graph)
        self.checkScript(foo, (3,))

    def test_peephole_slice_one_empty_arg(self):
        # [FUNCTION DOC] Checks that list-literal slices with exactly one omitted bound or step
        # (three forms) are folded so no slice op remains.
        def check_helper(fn: Callable[[int], None]) -> None:
            # [FUNCTION DOC] HELPER: Scripts the given function, runs peephole, checks no slice op
            # remains, and checks the scripted result matches eager for the input three.
            graph = torch.jit.script(fn).graph
            self.run_pass("peephole", graph)
            FileCheck().check_not("aten::slice").run(graph)
            self.checkScript(fn, (3,))

        def foo(x: int):
            # [FUNCTION DOC] HELPER: Slices a list literal from index one with step two.
            return [1, 2, x, 4, 5, 6, 7][1::2]

        check_helper(foo)

        def foo(x: int):
            # [FUNCTION DOC] HELPER: Slices a list literal up to index five with step three.
            return [1, 2, x, 4, 5, 6, 7][:5:3]

        check_helper(foo)

        def foo(x: int):
            # [FUNCTION DOC] HELPER: Slices a list literal from zero to four.
            return [1, 2, x, 4, 5, 6, 7][0:4]

        check_helper(foo)

    def test_peephole_slice_two_empty_args(self):
        # [FUNCTION DOC] Checks that list-literal slices with two omitted parts (three forms) are
        # folded so no slice op remains.
        def check_helper(fn: Callable[[int], None]) -> None:
            # [FUNCTION DOC] HELPER: Scripts the given function, runs peephole, checks no slice op
            # remains, and checks the scripted result matches eager for the input three.
            graph = torch.jit.script(fn).graph
            self.run_pass("peephole", graph)
            FileCheck().check_not("aten::slice").run(graph)
            self.checkScript(fn, (3,))

        def foo(x: int):
            # [FUNCTION DOC] HELPER: Slices a list literal with step two only.
            return [1, 2, x, 4, 5, 6, 7][::2]

        check_helper(foo)

        def foo(x: int):
            # [FUNCTION DOC] HELPER: Slices a list literal up to index five.
            return [1, 2, x, 4, 5, 6, 7][:5]

        check_helper(foo)

        def foo(x: int):
            # [FUNCTION DOC] HELPER: Slices a list literal from index one.
            return [1, 2, x, 4, 5, 6, 7][1:]

        check_helper(foo)

    def test_peephole_slice_optimization_not_applied_list_modified(self):
        # [FUNCTION DOC] Checks that the slice is kept when the list was changed by item
        # assignment.
        @torch.jit.script
        def foo():
            # [FUNCTION DOC] HELPER: Builds a seven-element list, sets its first item to zero, and
            # returns a slice of it.
            li = [1, 2, 3, 4, 5, 6, 7]
            li[0] = 0
            return li[2:5]

        self.run_pass("peephole", foo.graph)
        FileCheck().check("aten::slice").run(foo.graph)

    def test_peephole_slice_optimization_not_applied_non_const_args(self):
        # [FUNCTION DOC] Checks that the slice is kept when its bounds are runtime arguments.
        @torch.jit.script
        def foo(x: int, y: int):
            # [FUNCTION DOC] HELPER: Builds a list literal and slices it with the two int
            # arguments as bounds.
            li = [1, 2, 3, 4, 5, 6, 7]
            return li[x:y]

        self.run_pass("peephole", foo.graph)
        FileCheck().check("aten::slice").run(foo.graph)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
