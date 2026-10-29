# Owner(s): ["oncall: fx"]
# [MODULE DOC] Tests for the FX common subexpression elimination (CSE) pass. Builds a
# default pass instance with the banned-ops list from the library at import time and defines
# a shared check helper that traces a function, runs the pass and verifies node-count
# change, the modified flag, idempotence on a second run and numerical equivalence. The test
# class exercises duplicate ops, argument variants, list arguments, banned ops and random
# ops. Running the file directly raises an error pointing to the main FX test file.

import random

import torch
from torch.fx import symbolic_trace
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.dialect.common.cse_pass import CSEPass, get_CSE_banned_ops
from torch.testing._internal.common_utils import raise_on_run_directly, TestCase


banned_ops = get_CSE_banned_ops()
P_default = CSEPass(banned_ops=banned_ops)


def check(self, f, t, delta, check_val=True, graph_input=False, P=None):
    # [FUNCTION DOC] HELPER: Shared assertion helper. Gets a graph module (the given one, or
    # the function traced with make_fx on the tensor), runs the pass (default pass if none
    # given) and raises AssertionError unless the modified flag equals "node count
    # decreased". Then asserts the node count dropped by exactly delta, or merely did not
    # grow when delta is -1; asserts a second run of the pass does not shrink the graph
    # further; and if check_val is set, compares outputs of the original and new modules on
    # the tensor (both must be None, or all elements must be equal). Failure messages
    # include the graphs.
    """
    check if the CSE modified graph of ``f``
    1) has delta less nodes, and
    2) do not reduce the number of nodes further on a second pass, and
    3) modified returned is true only if the number of nodes decreases.

    Args:
        f: function to be checked
        t: tensor to be passed to f
        delta: an integer >= -1.
               If delta = -1, it only checks if the new graph has less or equal number of nodes
        check_val: if True, check if the output of f is correct
        graph_input: True is f is type GraphModule
        P: the pass to use. If None, use P_default
    """
    if graph_input:
        fx_g = f
    else:
        fx_g = make_fx(f)(t)

    if P is None:
        P = P_default

    res = P(fx_g)
    new_g = res.graph_module
    new_graph = new_g.graph
    modified = res.modified

    # the number of nodes decrease/ or stay the same
    old_num_nodes = len(fx_g.graph.nodes)
    new_num_nodes = len(new_graph.nodes)

    if (new_num_nodes < old_num_nodes) != modified:
        raise AssertionError("modified should be True if the number of nodes decrease")

    if delta == -1:
        self.assertTrue(
            old_num_nodes >= new_num_nodes,
            (
                lambda msg: f"{msg}\nnumber of nodes increased {old_num_nodes}, {new_num_nodes}"
            ),
        )
    else:
        self.assertTrue(
            old_num_nodes == new_num_nodes + delta,
            (
                lambda msg: f"{msg}\nnumber of nodes not the same {old_num_nodes - delta}, {new_num_nodes}\n {fx_g.graph} \n {new_graph}"
            ),
        )

    # a second pass should not reduce more nodes
    res = P(new_g)
    pass_2_graph = res.graph_module.graph
    pass_2_num_nodes = len(pass_2_graph.nodes)
    self.assertTrue(
        pass_2_num_nodes == new_num_nodes,
        (
            lambda msg: f"{msg}\nsecond pass graph has less node {pass_2_num_nodes}, {new_num_nodes}\n {new_graph} \n {pass_2_graph}"
        ),
    )

    # check correctness
    if check_val:
        true_result = fx_g(t)
        our_result = new_g(t)
        if true_result is None:  # both return None
            self.assertTrue(
                our_result is None,
                lambda msg: f"{msg}\ntrue result is None, CSE result is {our_result}",
            )
        else:  # results returned are the same
            self.assertTrue(
                torch.all(true_result == our_result),
                (
                    lambda msg: f"{msg}\nresults are different {true_result}, {our_result}"
                ),
            )  # check results are the same


class TestCSEPass(TestCase):
    # [CLASS DOC] Tests for the CSE pass. Each test defines a small function, traces it and
    # calls the shared check helper with the expected number of nodes the pass should
    # remove. No state.
    def test_nochange(self):
        # [FUNCTION DOC] Verifies that a function with no duplicate computations is left
        # unchanged (zero nodes removed).
        def f(x):
            # [FUNCTION DOC] HELPER: Traced function. Computes input plus 1, adds the input
            # to it, then the input plus itself, and returns the sum of the two; no repeated
            # subexpressions.
            a = x + 1
            b = x + a
            a = x
            d = x + a
            return b + d

        t = torch.randn(2, 2)
        check(self, f, t, 0)

    def test_empty(self):
        # [FUNCTION DOC] Verifies that an empty function (returning nothing) is traced and
        # passed through with no change; the output-is-None path of the check is used.
        def f(x):
            # [FUNCTION DOC] HELPER: Traced function that does nothing and returns None.
            pass

        t = torch.randn(2, 2)
        check(self, f, t, 0)

    def test_immutable_list_type(self):
        # [FUNCTION DOC] Verifies that identical reductions with a dimension list argument
        # are merged, as are identical full reductions; expects two nodes removed.
        def f(x):
            # [FUNCTION DOC] HELPER: Traced function. Sums over dimension 1 twice and over
            # all elements twice, and returns the sum of the four results.
            a = x.sum(dim=1)
            b = x.sum(dim=1)
            c = x.sum()
            d = x.sum()
            return a + b + c + d

        t = torch.randn(2, 2)
        check(self, f, t, 2)

    def test_immutable_list_multiple_entries(self):
        # [FUNCTION DOC] Verifies that duplicate reductions whose dimension argument is a
        # two element list are merged, along with duplicate reductions over one dimension;
        # expects two nodes removed.
        def f(x):
            # [FUNCTION DOC] HELPER: Traced function. Two sums over both dimensions given as
            # a list, two sums over dimension 1, all four added together.
            a = x.sum(dim=[0, 1])
            b = x.sum(dim=[0, 1])
            c = x.sum(dim=1)
            d = x.sum(dim=1)
            return a + b + c + d

        t = torch.randn(2, 2)
        check(self, f, t, 2)

    def test_simple(self):
        # [FUNCTION DOC] Verifies that two identical cosine ops are merged and that the
        # resulting duplicate additions then collapse too; expects two nodes removed.
        def f(x):
            # [FUNCTION DOC] HELPER: Traced function. Two cosines of the input, each added
            # to itself, and the two sums added together.
            a = x.cos()
            b = x.cos()
            c = a + a
            d = b + b
            return c + d

        t = torch.randn(2, 2)
        check(self, f, t, 2)

    def test_simple_2(self):
        # [FUNCTION DOC] Like the simple test but with a chained cosine then sine computed
        # twice on a one element input; expects three nodes removed.
        def f(x):
            # [FUNCTION DOC] HELPER: Traced function. Computes cosine then sine of the input
            # twice, doubles each by adding it to itself, and returns the sum.
            a = x.cos().sin()
            b = x.cos().sin()
            c = a + a
            d = b + b
            return c + d

        t = torch.randn(1)
        check(self, f, t, 3)

    def test_two_args_default(self):
        # [FUNCTION DOC] Verifies that a reduction with an explicit default keepdim value is
        # treated as the same as one without it, so all four sums collapse into one; expects
        # three nodes removed.
        def f(x):
            # [FUNCTION DOC] HELPER: Traced function. Four sums over dimension 1, two of
            # which pass the keepdim flag as false explicitly; returns their total.
            a = x.sum(dim=1)
            b = x.sum(dim=1, keepdim=False)
            c = x.sum(dim=1, keepdim=False)
            d = x.sum(dim=1)
            return a + b + c + d

        t = torch.randn(2, 2)
        check(self, f, t, 3)

    def test_two_args(self):
        # [FUNCTION DOC] Verifies that reductions with keepdim true are distinct from those
        # without, so only two duplicates are removed (one per group).
        def f(x):
            # [FUNCTION DOC] HELPER: Traced function. Two sums over dimension 1 with default
            # keepdim and two with keepdim true; returns their total.
            a = x.sum(dim=1)
            b = x.sum(dim=1, keepdim=True)
            c = x.sum(dim=1, keepdim=True)
            d = x.sum(dim=1)
            return a + b + c + d

        t = torch.randn(2, 2)
        check(self, f, t, 2)

    def test_simple_multiple_same_ops(self):
        # [FUNCTION DOC] Verifies that four identical full reductions collapse into one;
        # expects three nodes removed.
        def f(x):
            # [FUNCTION DOC] HELPER: Traced function. Four sums of the whole input added
            # together.
            a = x.sum()
            b = x.sum()
            c = x.sum()
            d = x.sum()
            return a + b + c + d

        t = torch.randn(2, 2)
        check(self, f, t, 3)

    def test_nested_immutable_list_type(self):
        # [FUNCTION DOC] Verifies that duplicate concatenations whose argument is a tuple of
        # tensors are merged; expects one node removed.
        def f(x):
            # [FUNCTION DOC] HELPER: Traced function. Concatenates a tuple of the input
            # twice, two times, and returns the sum of the two results.
            a = torch.cat((x, x))
            b = torch.cat((x, x))
            return a + b

        t = torch.randn(2, 2)
        check(self, f, t, 1)

    def test_kwarg(self):
        # [FUNCTION DOC] Verifies that duplicate ones_like calls on the same tensor are
        # merged; expects one node removed.
        def f(x):
            # [FUNCTION DOC] HELPER: Traced function. Two ones-like tensors of the input,
            # added together.
            a = torch.ones_like(x)
            b = torch.ones_like(x)
            return a + b

        t = torch.randn(2, 2)
        check(self, f, t, 1)

    """
    Generate function with random ops and check if the result is the same
    """

    def test_random(self):
        # [FUNCTION DOC] Randomized robustness test. Traces (symbolically) a function that
        # applies 100 randomly chosen unary ops to randomly chosen earlier values, removes
        # dead code and recompiles, then runs the check helper 30 times on the same graph
        # with delta -1, so the pass may remove any number of nodes but must preserve
        # results and be idempotent. The random choices are made once at trace time.
        def f(x):
            # [FUNCTION DOC] HELPER: Traced function. Starting from a list holding the
            # input, repeats 100 times: pick a random op among clone, cosine, tanh, gelu,
            # apply it to a randomly picked prior value and append the result. Returns the
            # last value.
            vals = [x]
            ops = [torch.clone, torch.cos, torch.tanh, torch.nn.functional.gelu]
            for _ in range(100):
                new_val = random.choice(ops)(random.choice(vals))
                vals.append(new_val)
            return vals[-1]

        fx_g = symbolic_trace(f)
        fx_g.graph.eliminate_dead_code()
        fx_g.recompile()
        t = torch.randn(2, 2)

        for _ in range(30):
            check(self, fx_g, t, -1, graph_input=True)

    """
    Test that banned list ban ops as expected.
    """

    def test_banned_list(self):
        # [FUNCTION DOC] Verifies the banned-ops option. With add banned the pass removes
        # nothing; with the default instance the duplicate add is merged, removing one node.
        def f(x):
            # [FUNCTION DOC] HELPER: Traced function. Two identical additions of 1 to the
            # input, added together.
            a = x + 1
            b = x + 1
            return a + b

        t = torch.randn(2, 2)
        P_ban_add = CSEPass(banned_ops=[torch.ops.aten.add])
        check(self, f, t, 0, P=P_ban_add)  # check that add is banned
        check(self, f, t, 1)  # check that add is not banned by default

    def test_rand_like(self):
        # [FUNCTION DOC] Verifies that random-number ops with the same arguments (rand_like)
        # are not merged; expects zero nodes removed, and skips value comparison since
        # results are random.
        def f(x):
            # [FUNCTION DOC] HELPER: Traced function. Two rand_like tensors of the input,
            # added together.
            a = torch.rand_like(x)
            b = torch.rand_like(x)
            return a + b

        t = torch.randn(2, 2)
        check(self, f, t, 0, check_val=False)

    def test_rand_n(self):
        # [FUNCTION DOC] Verifies that two random normal factory calls of identical size are
        # not merged; zero nodes removed, value comparison skipped.
        def f(x):
            # [FUNCTION DOC] HELPER: Traced function. Two random normal tensors of four
            # elements, added together; ignores its input.
            a = torch.randn(4)
            b = torch.randn(4)
            return a + b

        t = torch.randn(2, 2)
        check(self, f, t, 0, check_val=False)


if __name__ == "__main__":
    raise_on_run_directly("test/test_fx.py")
