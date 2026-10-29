# Owner(s): ["module: fx"]
# [MODULE DOC] Tests for torch.fx.subgraph_rewriter: replace_pattern and
# replace_pattern_with_filters. Each test traces a small module, rewrites matched
# subgraphs with a replacement function or traced module, lints the graph, and compares
# the outputs against a reference function traced from the expected result. Later tests
# cover multiple matches, filters, placeholder and attribute matching, wrapped functions,
# annotations, replaced submodules and call_method patterns.
# Importing the file directly is guarded: running it as the main program raises a RuntimeError
# telling the user to run through test/test_fx.py. Two torch.fx.wrap decorated helpers are
# defined at import time.

import os
import sys

import torch
from torch.fx import subgraph_rewriter, symbolic_trace
from torch.fx.annotate import annotate

# Make the helper files in test/ importable
from torch.fx.experimental.rewriter import RewritingTracer


pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase


if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_fx.py TESTNAME\n\n"
        "instead."
    )


@torch.fx.wrap
def wrapped_gemm_bias_mul(a, b, bias):
    # [FUNCTION DOC] HELPER: Wrapped (traced as a leaf call) function that applies a linear
    # layer using the second argument as weight and the third as bias to the first argument,
    # multiplies the linear result by the first argument and returns both the linear result and
    # the product as a tuple.
    lin_res = torch.nn.functional.linear(a, b, bias=bias)
    mul_res = lin_res * a
    return lin_res, mul_res


@torch.fx.wrap
def wrapped_gemm_bias_mul_with_c(a, b, bias, c):
    # [FUNCTION DOC] HELPER: Wrapped function like the previous one but multiplies the linear
    # result by a fourth argument c instead of the first argument; returns the linear result and
    # product as a tuple.
    lin_res = torch.nn.functional.linear(a, b, bias=bias)
    mul_res = lin_res * c
    return lin_res, mul_res


class TestSubgraphRewriter(JitTestCase):
    # [CLASS DOC] JitTestCase containing the subgraph rewriter tests. Test methods build
    # local modules and pattern, replacement and comparison functions, rewrite a
    # symbolically traced module in place and check results with assertEqual on outputs or
    # on the returned match lists.
    def test_subgraph_rewriter_preserves_logic(self):
        # [FUNCTION DOC] Replaces a negate plus relu pattern with itself and
        # checks the rewritten graph still lints and computes the same output as a
        # traced reference function.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing negate plus relu of the input into a value,
            # then adding the value to itself.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns val added to itself where val is the negation
                # of x plus the relu of x.
                val = torch.neg(x) + torch.relu(x)
                return torch.add(val, val)

        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern function returning negation of x plus relu of x.
            return torch.neg(x) + torch.relu(x)

        def comparison(x):
            # [FUNCTION DOC] HELPER: Reference function with the same logic as
            # the module forward.
            val = torch.neg(x) + torch.relu(x)
            return torch.add(val, val)

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.rand(1, 3)

        # Replace `pattern` with the same pattern (shouldn't change
        # the underlying logic)
        subgraph_rewriter.replace_pattern(traced, pattern, pattern)

        traced.graph.lint()

        ref_output = comparison_fn(x)
        test_output = traced.forward(x)
        self.assertEqual(ref_output, test_output)

    def test_subgraph_rewriter_with_oneliner_pattern(self):
        # [FUNCTION DOC] Replaces a single negate operation with relu using replace_pattern and
        # checks the output matches a reference where relu replaces negate.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module negating x then adding the result to itself.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Negates x and returns the value added to itself.
                val = torch.neg(x)
                return torch.add(val, val)

        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern function returning the negation of x.
            return torch.neg(x)

        def replacement(x):
            # [FUNCTION DOC] HELPER: Replacement function returning relu of x.
            return torch.relu(x)

        def comparison(x):
            # [FUNCTION DOC] HELPER: Reference function applying relu then adding the
            # result to itself.
            val = torch.relu(x)
            return torch.add(val, val)

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.rand(1, 3)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_output = comparison_fn(x)
        test_output = traced.forward(x)
        self.assertEqual(ref_output, test_output)

    def test_subgraph_rewriter_with_trivial_replacement(self):
        # [FUNCTION DOC] Uses replace_pattern_with_filters with an empty filter list to replace
        # an add-to-itself pattern with identity. Two matches are expected, the second of which
        # has no replacements recorded; the output must equal a single negate of the input.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module negating x then applying add-to-itself twice.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Negates x, doubles it by self addition,
                # and doubles again.
                val = torch.neg(x)
                val = torch.add(val, val)
                return torch.add(val, val)

        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern function adding x to itself.
            return torch.add(x, x)

        def replacement(x):
            # [FUNCTION DOC] HELPER: Replacement function returning x unchanged.
            return x

        def comparison(x):
            # [FUNCTION DOC] HELPER: Reference function returning the negation of x.
            return torch.neg(x)

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.randn(1, 5)

        matches = subgraph_rewriter.replace_pattern_with_filters(
            traced, pattern, replacement, []
        )

        traced.graph.lint()

        ref_output = comparison_fn(x)
        test_output = traced.forward(x)
        no_replacements = len(matches) == 2 and len(matches[1].replacements) == 0
        self.assertEqual(ref_output, test_output)
        self.assertTrue(no_replacements)

    def test_subgraph_rewriter_single_pattern_match(self):
        # [FUNCTION DOC] Replaces a single occurrence of the negate plus relu pattern with a
        # relu and checks output equality with the reference.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with negate plus relu followed by an add-to-itself.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Computes val as negation of x plus relu of x and
                # returns it added to itself.
                val = torch.neg(x) + torch.relu(x)
                return torch.add(val, val)

        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern function returning negation of x plus relu of x.
            return torch.neg(x) + torch.relu(x)

        def replacement(x):
            # [FUNCTION DOC] HELPER: Replacement function returning relu of x.
            return torch.relu(x)

        def comparison(x):
            # [FUNCTION DOC] HELPER: Reference function returning relu of x added to itself.
            val = torch.relu(x)
            return torch.add(val, val)

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.rand(1, 3)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_output = comparison_fn(x)
        test_output = traced.forward(x)
        self.assertEqual(ref_output, test_output)

    def test_subgraph_rewriter_multiple_pattern_match(self):
        # [FUNCTION DOC] Replaces two separate occurrences of a concatenate-then-sum pattern
        # with a stack of the two inputs and checks the result against a reference using stack
        # in both places.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing two identical concatenate-then-sum values and
            # returning x plus the max of each.
            def forward(self, x, w1, w2):
                # [FUNCTION DOC] HELPER: Computes two sums of the concatenation of w1 and w2 and
                # returns x plus the max of both.
                m1 = torch.cat([w1, w2]).sum()
                m2 = torch.cat([w1, w2]).sum()
                return x + torch.max(m1) + torch.max(m2)

        def pattern(w1, w2):
            # [FUNCTION DOC] HELPER: Pattern function returning the sum of the concatenation of
            # w1 and w2.
            return torch.cat([w1, w2]).sum()

        def replacement(w1, w2):
            # [FUNCTION DOC] HELPER: Replacement function returning the stack of w1 and w2.
            return torch.stack([w1, w2])

        def comparison(x, w1, w2):
            # [FUNCTION DOC] HELPER: Reference function with stack in
            # place of concatenate-then-sum.
            m1 = torch.stack([w1, w2])
            m2 = torch.stack([w1, w2])
            return x + torch.max(m1) + torch.max(m2)

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.rand(1, 3)
        w1 = torch.rand(1, 3)
        w2 = torch.rand(1, 3)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x, w1, w2)
        test_outs = traced.forward(x, w1, w2)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_graph_argument_order(self):
        # [FUNCTION DOC] Rewrites a matrix multiply pattern with itself to check the argument
        # order of the graph is preserved; outputs must equal the reference matrix multiply.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the matrix product of x and y.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the matrix multiplication of x and y.
                return torch.mm(x, y)

        def pattern(x, y):
            # [FUNCTION DOC] HELPER: Pattern function returning the matrix product of x and y.
            return torch.mm(x, y)

        def comparison(x, y):
            # [FUNCTION DOC] HELPER: Reference function returning the matrix product of x and y.
            return torch.mm(x, y)

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.randn(3, 4)
        y = torch.randn(4, 5)

        subgraph_rewriter.replace_pattern(traced, pattern, pattern)

        traced.graph.lint()

        ref_outs = comparison_fn(x, y)
        test_outs = traced.forward(x, y)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_correct_output_replacement(self):
        # [FUNCTION DOC] Replaces relu of x with the negation of x inside a graph where the relu
        # output feeds an addition with the negation of y; verifies the replacement output is
        # wired to the right user.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding negate of y to relu of x then doubling
            # by self addition.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Computes val from the negation of y plus relu of x and
                # returns val added to itself.
                val = torch.neg(y) + torch.relu(x)
                return torch.add(val, val)

        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern function returning relu of x.
            return torch.relu(x)

        def replacement(x):
            # [FUNCTION DOC] HELPER: Replacement function returning negation of x.
            return torch.neg(x)

        def comparison(x, y):
            # [FUNCTION DOC] HELPER: Reference function with negation replacing relu.
            val = torch.neg(y) + torch.neg(x)
            return torch.add(val, val)

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x, y)
        test_outs = traced.forward(x, y)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_traced_as_callable(self):
        # [FUNCTION DOC] Passes already traced GraphModules for the pattern and the replacement
        # to replace_pattern (instead of plain functions) and checks the result against a
        # reference using sigmoid.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Target module with negate plus relu then add-to-itself.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Computes negate plus relu and returns the result
                # added to itself.
                val = torch.neg(x) + torch.relu(x)
                return torch.add(val, val)

        class Pattern(torch.nn.Module):
            # [CLASS DOC] HELPER: Module serving as the pattern: negate plus relu of the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns negation of x plus relu of x.
                return torch.neg(x) + torch.relu(x)

        class Replacement(torch.nn.Module):
            # [CLASS DOC] HELPER: Module serving as the replacement: sigmoid of the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns sigmoid of x.
                return torch.sigmoid(x)

        def comparison(x):
            # [FUNCTION DOC] HELPER: Reference function with sigmoid then add-to-itself.
            val = torch.sigmoid(x)
            return torch.add(val, val)

        traced = symbolic_trace(M())
        traced_pattern = symbolic_trace(Pattern())
        traced_replacement = symbolic_trace(Replacement())
        comparison_fn = symbolic_trace(comparison)

        x = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, traced_pattern, traced_replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_pattern_is_entire_graph(self):
        # [FUNCTION DOC] Uses a pattern that spans the entire graph (negate then add-to-itself)
        # and a replacement of sigmoid plus concatenation; checks the output equals the
        # replacement function run directly.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module negating x and adding it to itself.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Negates x and returns it added to itself.
                a = torch.neg(x)
                return torch.add(a, a)

        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern function matching the whole module logic.
            a = torch.neg(x)
            return torch.add(a, a)

        def replacement(x):
            # [FUNCTION DOC] HELPER: Replacement function concatenating the sigmoid of
            # x with itself.
            a = torch.sigmoid(x)
            return torch.cat([a, a])

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(replacement)

        x = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_pattern_output_pattern_node_can_have_users_that_are_not_matched(
        self,
    ):
        # [FUNCTION DOC] Checks a node that is the output of the pattern may have users outside
        # the match: relu is replaced by sigmoid where the relu result is used by both a negate
        # and a subtraction. Output must match the reference.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing relu of x as y and returning negate of
            # y minus y.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies relu, then returns negation of the result
                # minus the result.
                y = torch.relu(x)
                return torch.neg(y) - y

        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern function returning relu of x.
            return torch.relu(x)

        def replacement(x):
            # [FUNCTION DOC] HELPER: Replacement function returning sigmoid of x.
            return torch.sigmoid(x)

        def comparison(x):
            # [FUNCTION DOC] HELPER: Reference function with sigmoid in place of relu.
            y = torch.sigmoid(x)
            return torch.neg(y) - y

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_internal_pattern_nodes_cannot_have_users_that_are_not_matched(
        self,
    ):
        # [FUNCTION DOC] Checks that a pattern whose internal (non-output) nodes are used
        # outside the matched region does not match: replace_pattern must return an empty list.
        # The module has duplicated concatenations and addmm calls so internal intermediate
        # values are used by unmatched nodes.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module producing several concatenations and addmm calls
            # (including unused ones) and returning sums of two of the results.
            def forward(self, x, w1, w2, b1, b2):
                # [FUNCTION DOC] HELPER: Builds two unused values and several used
                # intermediates, then returns the sum of one addmm product path and the
                # sum of another.
                m0 = torch.cat([w1, w2])  # noqa: F841
                m1 = torch.cat([w1, w2])
                m2 = torch.cat([x, b2])
                t0 = torch.addmm(b1, m1, m2.t())  # noqa: F841
                t1 = torch.sum(w1, 1)
                t2 = torch.addmm(b1, m1, m2.t())
                return torch.sum(t1), torch.sum(t2)

        def pattern(x, w1, w2, b1, b2):
            # [FUNCTION DOC] HELPER: Pattern function concatenating w1 and w2, concatenating
            # x and b2, and returning addmm of b1, the first concatenation and the transpose
            # of the second.
            m1 = torch.cat([w1, w2])
            m2 = torch.cat([x, b2])
            return torch.addmm(b1, m1, m2.t())

        def replacement(x, w1, w2, b1, b2):
            # [FUNCTION DOC] HELPER: Replacement function returning the concatenation of x,
            # w1 and w2.
            return torch.cat([x, w1, w2])

        traced = symbolic_trace(M())

        # Result should be [] since no matches can be found
        res = subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        self.assertEqual(res, [])

    def test_subgraph_rewriter_placeholder_matching(self):
        # [FUNCTION DOC] Documents with the docstring that a placeholder in the pattern can
        # match a node that has its own inputs. Traces a module that adds three in place,
        # dequantizes, applies sigmoid and converts to float16, and a pattern starting from the
        # dequantize call; uses an identity replacement and compares to a reference that only
        # adds three to the input.
        """
        This tests that a placeholder Node can be matched to a Node with
        a different number of input Nodes. In the example below, the
        original traced Module looks like this:

            opcode         target                                                      args                      kwargs
            -------------  ----------------------------------------------------------  ------------------------  --------
            placeholder    x                                                           ()                        {}
            call_function  <built-in function add>                                     (x, 3)                    {}
            call_method    dequantize                                                  (add,)                    {}
            call_function  <built-in method sigmoid of type object at 0x7f7c1f440fe0>  (dequantize,)             {}
            call_method    to                                                          (sigmoid, torch.float16)  {}
            output         output                                                      (to,)                     {}

        while the pattern we want to match looks like this:

            opcode         target                                                      args                      kwargs
            -------------  ----------------------------------------------------------  ------------------------  --------
            placeholder    x                                                           ()                        {}
            call_method    dequantize                                                  (x,)                      {}
            call_function  <built-in method sigmoid of type object at 0x7f7c1f440fe0>  (dequantize,)             {}
            call_method    to                                                          (sigmoid, torch.float16)  {}
            output         output                                                      (to,)                     {}

        Here, we want to be able to match the original graph's
        `call_function.add` Node with the pattern graph's
        `placeholder.x` Node.

        Credit to Jerry Zhang (GitHub: jerryzh168) for this test case
        """

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing a dtype attribute of float16 and applying in
            # place add of three, dequantize, sigmoid and conversion to that dtype.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores float16 as the dtype attribute.
                super().__init__()
                self.dtype = torch.float16

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds three to x in place, dequantizes, applies sigmoid,
                # converts to the stored dtype and returns it.
                x += 3
                x = x.dequantize()
                x = torch.sigmoid(x)
                dtype = self.dtype
                x = x.to(dtype)
                return x

        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern function dequantizing x, applying sigmoid and
            # converting to float16.
            x = x.dequantize()
            x = torch.sigmoid(x)
            x = x.to(torch.float16)
            return x

        def replacement(x):
            # [FUNCTION DOC] HELPER: Replacement function returning x unchanged.
            return x

        def comparison(x):
            # [FUNCTION DOC] HELPER: Reference function returning x plus three.
            return x + 3

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_replaces_referenced_submodules(self):
        # [FUNCTION DOC] Replaces a pattern module (sigmoid then relu submodules) with a
        # replacement module (tanh then relu) in a traced module. After rewriting, checks
        # the output equals a reference module, that the tanh submodule now exists, that
        # fetching the sigmoid submodule raises AttributeError, and that the submod
        # attribute is still a ReLU.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding one to x then applying its sigmoid and then
            # submod (ReLU) submodules.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a Sigmoid submodule named sigmoid and a ReLU
                # submodule named submod.
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()
                self.submod = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds one to x and returns submod applied to
                # sigmoid of x.
                x = x + 1
                return self.submod(self.sigmoid(x))

        class Pattern(torch.nn.Module):
            # [CLASS DOC] HELPER: Pattern module applying the sigmoid submodule then
            # the ReLU submodule.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates sigmoid and submod (ReLU) submodules.
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()
                self.submod = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies sigmoid then submod to x.
                return self.submod(self.sigmoid(x))

        class Replacement(torch.nn.Module):
            # [CLASS DOC] HELPER: Replacement module applying a tanh submodule then
            # a ReLU submodule.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates tanh and submod (ReLU) submodules.
                super().__init__()
                self.tanh = torch.nn.Tanh()
                self.submod = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies tanh then submod to x.
                return self.submod(self.tanh(x))

        class Comparison(torch.nn.Module):
            # [CLASS DOC] HELPER: Reference module: add one, then tanh and ReLU submodules.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates tanh and submod (ReLU) submodules.
                super().__init__()
                self.tanh = torch.nn.Tanh()
                self.submod = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds one to x, then applies tanh and then submod.
                x = x + 1
                return self.submod(self.tanh(x))

        traced = symbolic_trace(M())
        comparison = Comparison()

        x = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, Pattern(), Replacement())

        traced.graph.lint()

        ref_outs = comparison(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

        traced.get_submodule("tanh")
        with self.assertRaisesRegex(AttributeError, "has no attribute"):
            traced.get_submodule("sigmoid")

        submod = traced.get_submodule("submod")
        self.assertEqual(type(submod), torch.nn.ReLU)

    def test_subgraph_rewriter_annotations_int(self):
        # [FUNCTION DOC] Compares type annotations on placeholders: M1 is traced with the AST
        # rewriting tracer where a local integer annotation is used, M2 uses the annotate helper
        # with int and is traced with symbolic_trace. Raises AssertionError if the placeholder
        # type of either traced graph is not int. It never runs the rewriter, despite its name.
        class M1(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an int annotated local assigned from x, returning
            # the sum of x and it.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Assigns x to an integer annotated variable and
                # returns the sum.
                y: int = x
                return torch.add(x, y)

        class M2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module using the annotate helper to mark x as an int and
            # adding it to x.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Annotates x as int and returns x added to
                # the annotated value.
                y = annotate(x, int)
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(M1())

        module = M2()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        for n, m in zip(symbolic_traced.graph.nodes, graph.nodes):
            if n.op == "placeholder":
                if n.type is not int:
                    raise AssertionError(f"Expected n.type to be int, got {n.type}")
                if m.type is not int:
                    raise AssertionError(f"Expected m.type to be int, got {m.type}")

    def test_subgraph_rewriter_replace_consecutive_submodules(self):
        # [FUNCTION DOC] Replaces each of three consecutive sigmoid calls with exp
        # using replace_pattern and checks the output equals the reference that applies
        # exp three times.
        def f(x):
            # [FUNCTION DOC] HELPER: Function applying sigmoid three times in a row.
            x = torch.sigmoid(x)
            x = torch.sigmoid(x)
            return torch.sigmoid(x)

        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern function returning sigmoid of x.
            return torch.sigmoid(x)

        def replacement(x):
            # [FUNCTION DOC] HELPER: Replacement function returning exp of x.
            return torch.exp(x)

        def comparison(x):
            # [FUNCTION DOC] HELPER: Reference function applying exp three times.
            x = torch.exp(x)
            x = torch.exp(x)
            return torch.exp(x)

        traced = symbolic_trace(f)
        comparison_fn = symbolic_trace(comparison)

        x = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_with_overlapping_matches(self):
        # [FUNCTION DOC] Matches the two-sigmoid pattern against four chained sigmoids; the
        # overlapping matches must be resolved so two non-overlapping replacements by negation
        # are made. Output must equal two successive negations.
        def f(x):
            # [FUNCTION DOC] HELPER: Function applying sigmoid four times in a row.
            x = torch.sigmoid(x)
            x = torch.sigmoid(x)
            x = torch.sigmoid(x)
            return torch.sigmoid(x)

        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern function applying sigmoid twice.
            x = torch.sigmoid(x)
            x = torch.sigmoid(x)
            return x

        def replacement(x):
            # [FUNCTION DOC] HELPER: Replacement function returning negation of x.
            return torch.neg(x)

        def comparison(x):
            # [FUNCTION DOC] HELPER: Reference function negating x twice.
            x = torch.neg(x)
            return torch.neg(x)

        traced = symbolic_trace(f)
        comparison_fn = symbolic_trace(comparison)

        x = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_replace_with_multiple_outputs(self):
        # [FUNCTION DOC] Replaces a pattern with two outputs (sigmoid and relu of the same
        # input) by a replacement returning exp and abs; checks the output against a reference
        # with exp and abs.
        def f(x):
            # [FUNCTION DOC] HELPER: Function returning the sum of sigmoid of x and relu of x.
            y = torch.sigmoid(x)
            z = torch.relu(x)
            return y + z

        def pattern(a):
            # [FUNCTION DOC] HELPER: Pattern function returning the sigmoid and the relu of its
            # input as a pair.
            b = torch.sigmoid(a)
            c = torch.relu(a)
            return b, c

        def replacement(x):
            # [FUNCTION DOC] HELPER: Replacement function returning the exp and the abs of x
            # as a pair.
            return torch.exp(x), torch.abs(x)

        def comparison(x):
            # [FUNCTION DOC] HELPER: Reference function returning exp of x plus abs of x.
            y = torch.exp(x)
            z = torch.abs(x)
            return y + z

        traced = symbolic_trace(f)
        comparison_fn = symbolic_trace(comparison)

        x = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_replace_with_duplicated_outputs(self):
        # [FUNCTION DOC] Pattern returns three outputs including the intermediate difference
        # value; the replacement returns the same abs result twice plus an exp. Verifies
        # duplicated replacement outputs are wired correctly: the result equals twice the abs of
        # the second input.
        def f(x1, x2):
            # [FUNCTION DOC] HELPER: Function subtracting x2 from x1 and returning sigmoid plus
            # relu of the difference.
            x = x1 - x2
            y = torch.sigmoid(x)
            z = torch.relu(x)
            return y + z

        def pattern(a1, a2):
            # [FUNCTION DOC] HELPER: Pattern function computing the difference, its sigmoid and
            # its relu and returning all three.
            a = a1 - a2
            b = torch.sigmoid(a)
            c = torch.relu(a)
            return b, c, a

        def replacement(x1, x2):
            # [FUNCTION DOC] HELPER: Replacement function returning the abs of x2 twice and the
            # exp of x1 as the third output.
            y1 = torch.exp(x1)
            y2 = torch.abs(x2)
            return y2, y2, y1

        def comparison(x1, x2):
            # [FUNCTION DOC] HELPER: Reference function returning the abs of x2 added to itself.
            y2 = torch.abs(x2)
            return y2 + y2

        traced = symbolic_trace(f)
        comparison_fn = symbolic_trace(comparison)

        x1 = torch.randn(3, 4)
        x2 = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x1, x2)
        test_outs = traced.forward(x1, x2)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_with_unused_args(self):
        # [FUNCTION DOC] Rewrites an addition into a subtraction in a module with an unused
        # third argument. Asserts the graph still has three placeholders after lint, and the
        # output equals x1 minus x2.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with three inputs that returns only x plus y.
            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Returns x plus y ignoring z.
                return x + y

        def pattern(x, y):
            # [FUNCTION DOC] HELPER: Pattern function adding x and y.
            return x + y

        def replacement(x, y):
            # [FUNCTION DOC] HELPER: Replacement function subtracting y from x.
            return x - y

        def comparison(x1, x2, x3):
            # [FUNCTION DOC] HELPER: Reference function returning x1 minus x2 and ignoring x3.
            return x1 - x2

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x1 = torch.randn(3, 4)
        x2 = torch.randn(3, 4)
        x3 = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()
        placeholder_nodes = [n for n in traced.graph.nodes if n.op == "placeholder"]
        if len(placeholder_nodes) != 3:
            raise AssertionError(
                f"Expected 3 placeholder nodes, got {len(placeholder_nodes)}"
            )

        ref_outs = comparison_fn(x1, x2, x3)
        test_outs = traced.forward(x1, x2, x3)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_with_unused_results(self):
        # [FUNCTION DOC] Pattern is a copy followed by an in place copy into the same tensor;
        # replacement is a single in place copy. The module has an in place update of the
        # cache followed by relu, so the test checks the cache mutation still happens before
        # relu after rewriting. Compares output of the rewritten graph (on cloned inputs)
        # with a reference.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module multiplying x and y, index-copying into the cache,
            # copying and in place copying into the cache, then returning the relu of the cache.
            def forward(self, x, y, cache):
                # [FUNCTION DOC] HELPER: Mutates the cache through aten copy operations and
                # returns relu of the cache; the in place copy result is unused.
                m = torch.mul(x, y)
                n = cache.index_copy(0, torch.tensor([0]), m)
                p = torch.ops.aten.copy.default(cache, n)
                q = torch.ops.aten.copy_.default(cache, p)  # noqa: F841
                u = torch.relu(cache)
                # check the result to ensure cache is updated before relu op
                return u

        def pattern(self_tensor, src_tensor):
            # [FUNCTION DOC] HELPER: Pattern function applying aten copy then aten in place copy
            # on the first tensor and returning the in place result.
            p = torch.ops.aten.copy.default(self_tensor, src_tensor)
            q = torch.ops.aten.copy_.default(self_tensor, p)
            return q

        def replacement(self_tensor, src_tensor):
            # [FUNCTION DOC] HELPER: Replacement function applying a single aten in place copy
            # and returning it.
            q = torch.ops.aten.copy_.default(self_tensor, src_tensor)
            return q

        def comparison(x, y, cache):
            # [FUNCTION DOC] HELPER: Reference function with a single in place copy of n into
            # the cache followed by relu.
            m = torch.mul(x, y)
            n = cache.index_copy(0, torch.tensor([0]), m)
            q = torch.ops.aten.copy_.default(cache, n)  # noqa: F841
            u = torch.relu(cache)
            return u

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        x = torch.randn(1, 8)
        y = torch.randn(1, 8)
        cache = torch.randn(2, 8)
        x_clone = x.clone()
        y_clone = y.clone()
        cache_clone = cache.clone()

        ref_outs = comparison_fn(x, y, cache)
        test_outs = traced.forward(x_clone, y_clone, cache_clone)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_call_method(self):
        # [FUNCTION DOC] Replaces a method call chain (dequantize, sigmoid, conversion to
        # float16) with the identity; the rewritten module must return its input unchanged.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying dequantize, sigmoid and conversion to float16
            # as method calls.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls dequantize, sigmoid and to-float16 on x in order
                # and returns the result.
                x = x.dequantize()
                x = x.sigmoid()
                x = x.to(torch.float16)
                return x

        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern function with the same method call chain.
            x = x.dequantize()
            x = x.sigmoid()
            x = x.to(torch.float16)
            return x

        def replacement(x):
            # [FUNCTION DOC] HELPER: Replacement function returning x unchanged.
            return x

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(replacement)

        x1 = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x1)
        test_outs = traced.forward(x1)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_nodes_with_kwargs(self):
        # [FUNCTION DOC] Rewrites a linear plus multiply pattern (linear called with a keyword
        # bias) into a call of the wrapped function wrapped_gemm_bias_mul. Asserts exactly one
        # match and that a node targeting the wrapped function exists in the rewritten graph.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a 128 by 128 weight and 128 bias parameter
            # computing the multiply of input by the linear result plus the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates uninitialized parameters w0 and b0.
                super().__init__()
                self.w0 = torch.nn.Parameter(torch.empty([128, 128]))
                self.b0 = torch.nn.Parameter(torch.empty([128]))

            def forward(self, in0):
                # [FUNCTION DOC] HELPER: Applies linear with bias keyword, multiplies the input
                # by the result, adds the input and returns the sum.
                lin_res = torch.nn.functional.linear(in0, self.w0, bias=self.b0)
                mul_res = in0 * lin_res
                sum_res = mul_res + in0
                return sum_res

        def pattern(a, b, bias):
            # [FUNCTION DOC] HELPER: Pattern function applying linear with bias keyword and
            # multiplying a by the result; returns both values.
            lin_res = torch.nn.functional.linear(a, b, bias=bias)
            mul_res = a * lin_res
            return lin_res, mul_res

        def replacement(a, b, bias):
            # [FUNCTION DOC] HELPER: Replacement function calling the wrapped gemm bias mul
            # helper and returning both of its results.
            lin_res, mul_res = wrapped_gemm_bias_mul(a, b, bias)
            return lin_res, mul_res

        traced = symbolic_trace(M())
        matches = subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        self.assertEqual(len(matches), 1)

        found_repalcement_node = False
        for node in traced.graph.nodes:
            if node.target is wrapped_gemm_bias_mul:
                found_repalcement_node = True
                break

        self.assertTrue(found_repalcement_node)

    def test_subgraph_rewriter_local_revert(self):
        # [FUNCTION DOC] Checks that a failed partial match is reverted. The model has three
        # candidate anchor positions for the pattern (linear, then multiply of c by the linear
        # result), where the first and third match and the second does not. Asserts
        # replace_pattern returns two matches and that exactly two nodes target the wrapped
        # function with the extra c argument.
        # Following model will have 3 anchors as the matching candidate with the given pattern
        # Anchor 1 and 3 is a real match, but anchor 2 is not.
        # The subgraph rewriter should be able to revert the changes made while matching anchor 2.
        # Final match with anchor 3 should be successful.

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with five weight and bias parameter pairs, three chains
            # of linear and elementwise operations, a sigmoid branch which is a near match that
            # should fail, and a final concatenation along dimension one.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates uninitialized parameters w0 to w4 (128 by 128)
                # and b0 to b4 (128).
                super().__init__()
                self.w0 = torch.nn.Parameter(torch.empty([128, 128]))
                self.b0 = torch.nn.Parameter(torch.empty([128]))
                self.w1 = torch.nn.Parameter(torch.empty([128, 128]))
                self.b1 = torch.nn.Parameter(torch.empty([128]))
                self.w2 = torch.nn.Parameter(torch.empty([128, 128]))
                self.b2 = torch.nn.Parameter(torch.empty([128]))
                self.w3 = torch.nn.Parameter(torch.empty([128, 128]))
                self.b3 = torch.nn.Parameter(torch.empty([128]))
                self.w4 = torch.nn.Parameter(torch.empty([128, 128]))
                self.b4 = torch.nn.Parameter(torch.empty([128]))

            def forward(self, in0, in1):
                # [FUNCTION DOC] HELPER: Runs linear chains on both inputs, builds the multiply,
                # add and sigmoid based values, and returns the concatenation of the
                # second-anchor and third-anchor results along dimension one.
                lin_res_1 = torch.nn.functional.linear(in1, self.w0, bias=self.b0)
                lin_res_2 = torch.nn.functional.linear(lin_res_1, self.w1, bias=self.b1)
                # potential match at anchor 1
                mul_res_1 = in1 * lin_res_2
                sum_res_1 = mul_res_1 + in1
                lin_res_3 = torch.nn.functional.linear(sum_res_1, self.w2, bias=self.b2)
                sigmoid_res_1 = torch.sigmoid(lin_res_3)
                # potential match at anchor 2
                mul_res_2 = lin_res_3 * sigmoid_res_1
                lin_res_4 = torch.nn.functional.linear(in0, self.w3, bias=self.b3)
                lin_res_5 = torch.nn.functional.linear(lin_res_4, self.w4, bias=self.b4)
                # potential match at anchor 3
                mul_res_3 = in0 * lin_res_5
                sum_res_2 = mul_res_3 + in0
                cat_res = torch.cat(
                    [mul_res_2, sum_res_2],
                    dim=1,
                )
                return cat_res

        def gemm_bias_mul_pattern_with_c(a, b, bias, c):
            # [FUNCTION DOC] HELPER: Pattern function: linear with bias keyword then multiply of
            # c by the linear result; returns both.
            lin_res = torch.nn.functional.linear(a, b, bias=bias)
            mul_res = c * lin_res
            return lin_res, mul_res

        def gemm_bias_mul_replacement_with_c(a, b, bias, c):
            # [FUNCTION DOC] HELPER: Replacement function calling the wrapped variant that takes
            # c and returning its two results.
            lin_res, mul_res = wrapped_gemm_bias_mul_with_c(a, b, bias, c)
            return lin_res, mul_res

        traced = symbolic_trace(M())
        matches = subgraph_rewriter.replace_pattern(
            traced, gemm_bias_mul_pattern_with_c, gemm_bias_mul_replacement_with_c
        )

        self.assertEqual(len(matches), 2)

        repalcement_node_found = 0
        for node in traced.graph.nodes:
            if node.target is wrapped_gemm_bias_mul_with_c:
                repalcement_node_found += 1

        self.assertEqual(repalcement_node_found, 2)

    def test_replace_pattern_with_filters(self):
        # [FUNCTION DOC] Uses replace_pattern_with_filters with a quantized add and relu
        # pattern. Without filters, two matches are found and two multiply replacement nodes
        # appear. With a filter that requires the second pattern input to match a plain number,
        # only the first (scalar add) is matched. Uses the local helper to compare the
        # replacement nodes in the graph with those reported in the matches.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two dequantize, add, relu, quantize blocks; the
            # first adds a scalar two, the second adds a tensor and therefore should not match
            # the scalar filter.
            def forward(self, x, scale, zero_point):
                # [FUNCTION DOC] HELPER: Executes the two blocks and returns the
                # final quantized tensor.
                # Match, second input to add is a scalar
                x = x.dequantize()
                x = torch.add(x, 2)
                x = x.relu()
                x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)

                y = x + 1
                # NOT a match, second input to add is NOT a scalar
                x = x.dequantize()
                x = torch.add(x, y)
                x = x.relu()
                x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)

                return x

        def BinaryOpScalarReLUPattern(x, num, scale, zero_point):
            # [FUNCTION DOC] HELPER: Pattern function: dequantize, add num, relu, quantize per
            # tensor with scale and zero point.
            x = x.dequantize()
            x = torch.add(x, num)
            x = x.relu()
            x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
            return x

        def BinaryOpScalarReLUReplacement(x, num, scale, zero_point):
            # [FUNCTION DOC] HELPER: Replacement function multiplying x by num and ignoring
            # scale and zero point.
            x = torch.mul(x, num)
            return x

        def second_input_is_scalar(match, original_graph, pattern_graph):
            # [FUNCTION DOC] HELPER: Match filter. Finds the pattern placeholder at index one
            # and returns true only if the node matched to it in the original graph is a plain
            # int or float number.
            """check the node that's matched to the second input of the pattern graph
            is a scalar number
            """
            input_idx = 0
            for node in pattern_graph.nodes:
                if node.op == "placeholder":
                    if input_idx == 1:
                        num_node = node
                    input_idx += 1
            return isinstance(match.nodes_map[num_node], (int, float))

        def check_replacement_nodes(self, traced, matches):
            # [FUNCTION DOC] HELPER: Collects multiply nodes in the traced graph and all
            # replacement nodes from the matches, asserts both lists have the same length and
            # are equal, and returns the count.
            replacement_nodes_in_graph = [
                node for node in traced.graph.nodes if node.target == torch.mul
            ]
            replacement_nodes_in_res = [r for m in matches for r in m.replacements]
            self.assertEqual(
                len(replacement_nodes_in_graph), len(replacement_nodes_in_res)
            )
            self.assertEqual(replacement_nodes_in_graph, replacement_nodes_in_res)
            return len(replacement_nodes_in_graph)

        # match without filter, should find 2 match
        traced = symbolic_trace(M())
        matches = subgraph_rewriter.replace_pattern_with_filters(
            traced, BinaryOpScalarReLUPattern, BinaryOpScalarReLUReplacement, None
        )
        self.assertEqual(len(matches), 2)
        self.assertEqual(check_replacement_nodes(self, traced, matches), 2)

        # match with filter, should find 1 match
        traced = symbolic_trace(M())
        matches = subgraph_rewriter.replace_pattern_with_filters(
            traced,
            BinaryOpScalarReLUPattern,
            BinaryOpScalarReLUReplacement,
            [second_input_is_scalar],
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(check_replacement_nodes(self, traced, matches), 1)

    def test_matching_pattern_with_list_type_arg(self):
        # [FUNCTION DOC] Rewrites a reshape-alias-copy call with list arguments, swapping the
        # order of the two list arguments in the replacement. Asserts one match and checks the
        # generated code string after rewriting using assertExpectedInline.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the aten reshape alias copy operator with size
            # [1, 2] and stride [3, 4].
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the reshape alias copy of x with
                # the fixed lists.
                return torch.ops.aten._reshape_alias_copy.default(x, [1, 2], [3, 4])

        def pattern(x, arg0, arg1):
            # [FUNCTION DOC] HELPER: Pattern function passing x and the two arguments to the
            # reshape alias copy operator in order.
            return torch.ops.aten._reshape_alias_copy.default(x, arg0, arg1)

        def replacement(x, arg0, arg1):
            # [FUNCTION DOC] HELPER: Replacement function calling the same operator with the two
            # list arguments swapped.
            return torch.ops.aten._reshape_alias_copy.default(x, arg1, arg0)

        traced = symbolic_trace(M())
        matches = subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        self.assertEqual(len(matches), 1)

        self.assertExpectedInline(
            traced.code.strip(),
            """\
def forward(self, x):
    _reshape_alias_copy_default_1 = torch.ops.aten._reshape_alias_copy.default(x, [3, 4], [1, 2]);  x = None
    return _reshape_alias_copy_default_1""",
        )

    def test_replacement_with_attrs(self):
        # [FUNCTION DOC] Pattern and replacement are modules that hold tensor attributes (a for
        # the pattern, c for the replacement), while the traced module has attributes a and b.
        # Asserts replace_pattern finds exactly one match.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with tensor attributes a and b returning x plus
            # a minus b.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates single-element tensors a (one) and b (two).
                super().__init__()
                self.a = torch.tensor([1])
                self.b = torch.tensor([2])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus a minus b.
                return x + self.a - self.b

        class Pattern(torch.nn.Module):
            # [CLASS DOC] HELPER: Pattern module holding a tensor attribute a and adding
            # it to x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates tensor attribute a (one).
                super().__init__()
                self.a = torch.tensor([1])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus a.
                return x + self.a

        class Replacement(torch.nn.Module):
            # [CLASS DOC] HELPER: Replacement module holding a tensor attribute c and
            # subtracting it from x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates tensor attribute c (three).
                super().__init__()
                self.c = torch.tensor([3])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x minus c.
                return x - self.c

        traced = symbolic_trace(M())
        matches = subgraph_rewriter.replace_pattern(traced, Pattern(), Replacement())
        self.assertEqual(len(matches), 1)

    def test_matching_variable_arguments(self):
        # [FUNCTION DOC] Matches max_pool2d_with_indices where the traced call omits the padding
        # argument but the pattern passes padding as keyword set to zeros (the default).
        # Replacing the pattern with itself must yield one match.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling max_pool2d_with_indices with kernel size
            # and stride lists.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the max pool with indices result for x with
                # kernel size [2, 2] and stride [2, 2].
                return torch.ops.aten.max_pool2d_with_indices.default(
                    x, [2, 2], stride=[2, 2]
                )

        def pattern(x, kernel_size, stride):
            # [FUNCTION DOC] HELPER: Pattern function calling the same operator with
            # kernel_size, stride and an explicit zero padding keyword.
            # default padding is [0, 0]
            return torch.ops.aten.max_pool2d_with_indices.default(
                x, kernel_size, stride, padding=[0, 0]
            )

        traced = symbolic_trace(M())
        matches = subgraph_rewriter.replace_pattern(traced, pattern, pattern)

        self.assertEqual(len(matches), 1)

    def test_replaced_nodes(self):
        # [FUNCTION DOC] Replaces an add with a multiply followed by a subtract using
        # replace_pattern_with_filters, then uses a local helper to assert the multiply and
        # subtract nodes in the graph equal the replacement nodes reported by the matches, and
        # that there are two of them.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the sum of two inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x added to y.
                return torch.add(x, y)

        def pattern(x, y):
            # [FUNCTION DOC] HELPER: Pattern function adding x and y.
            return torch.add(x, y)

        def replacement(x, y):
            # [FUNCTION DOC] HELPER: Replacement function returning the product of x and
            # y minus y.
            return torch.sub(torch.mul(x, y), y)

        traced = symbolic_trace(M())
        matches = subgraph_rewriter.replace_pattern_with_filters(
            traced, pattern, replacement
        )

        def check_replacement_nodes(self, traced, matches):
            # [FUNCTION DOC] HELPER: Collects multiply and subtract nodes in the traced graph
            # and the replacement nodes from the matches, asserts both lists are equal in length
            # and content, and returns the count.
            replacement_nodes_in_graph = [
                node
                for node in traced.graph.nodes
                if node.target in {torch.sub, torch.mul}
            ]
            replacement_nodes_in_res = [r for m in matches for r in m.replacements]
            self.assertEqual(
                len(replacement_nodes_in_graph), len(replacement_nodes_in_res)
            )
            self.assertEqual(replacement_nodes_in_graph, replacement_nodes_in_res)
            return len(replacement_nodes_in_graph)

        self.assertEqual(check_replacement_nodes(self, traced, matches), 2)

    def test_replace_pattern_with_callback(self) -> None:
        # [FUNCTION DOC] Calls replace_pattern_with_filters with a replacement callback (a
        # lambda tracing the replacement function into a fresh graph each time) instead of a
        # replacement argument. Checks, using a local helper, the replacement nodes in the graph
        # (multiply and subtract) match those in the returned matches and total two.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the sum of two inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x added to y.
                return torch.add(x, y)

        def pattern(x, y):
            # [FUNCTION DOC] HELPER: Pattern function adding x and y.
            return torch.add(x, y)

        def replacement(x, y):
            # [FUNCTION DOC] HELPER: Replacement function returning the product of x and
            # y minus y.
            return torch.sub(torch.mul(x, y), y)

        traced = symbolic_trace(M())
        # Return the same replacement graph for all matches, but have it be a unique
        # object each time.
        matches = subgraph_rewriter.replace_pattern_with_filters(
            traced,
            pattern,
            replacement_callback=lambda *args: symbolic_trace(replacement).graph,
        )

        def check_replacement_nodes(self, traced, matches):
            # [FUNCTION DOC] HELPER: Collects multiply and subtract nodes in the traced graph
            # and the replacement nodes from the matches, asserts both lists are equal in length
            # and content, and returns the count.
            replacement_nodes_in_graph = [
                node
                for node in traced.graph.nodes
                if node.target in {torch.sub, torch.mul}
            ]
            replacement_nodes_in_res = [r for m in matches for r in m.replacements]
            self.assertEqual(
                len(replacement_nodes_in_graph), len(replacement_nodes_in_res)
            )
            self.assertEqual(replacement_nodes_in_graph, replacement_nodes_in_res)
            return len(replacement_nodes_in_graph)

        self.assertEqual(check_replacement_nodes(self, traced, matches), 2)
