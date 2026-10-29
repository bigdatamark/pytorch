# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the pre-grad FX fusion passes in torch._inductor.fx_passes.pre_grad:
# sink_cat_after_pointwise, linear_permute_fusion, permute_linear_fusion and
# permute_matmul_fusion. Each test symbolically traces a function or module, runs ShapeProp and
# the pass, checks numerics against eager, and counts call nodes. No import-time side effects
# other than defining the PassFunc alias and the helpers below.
from collections.abc import Callable
from typing import Any

import torch
from torch._inductor.fx_passes.pre_grad import (
    linear_permute_fusion,
    linear_transpose,
    permute_linear_fusion,
    permute_matmul_fusion,
    sink_cat_after_pointwise,
    transpose_linear,
    transpose_matmul,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.fx.passes.shape_prop import ShapeProp


PassFunc = Callable[[torch.fx.GraphModule, Any], torch.fx.GraphModule]


def chain_passes(*passes: PassFunc) -> PassFunc:
    # [FUNCTION DOC] HELPER: Builds a combined pass from the given pass callables. The returned
    # function takes a module (or callable) and example inputs, then applies each pass in order.
    # Before each pass, if the current object is a GraphModule, ShapeProp is run on it with the
    # inputs so shape metadata is available; the pass result feeds the next pass. Returns the final
    # module.
    def parent_pass(module: torch.fx.GraphModule, input: Any) -> torch.fx.GraphModule:
        # [FUNCTION DOC] HELPER: Applies the chained passes to the module in order. Runs shape
        # propagation with the unpacked inputs whenever the current object is a GraphModule
        # (mutating node metadata), and returns the module produced by the last pass.
        for pass_ in passes:
            if isinstance(module, torch.fx.GraphModule):
                ShapeProp(module).propagate(*input)
            module = pass_(module)
        return module

    return parent_pass


def count_call(module: torch.fx.GraphModule, op: str, target_op: Any) -> int:
    # [FUNCTION DOC] HELPER: Counts graph nodes whose op kind equals the given op string and whose
    # target equals the given target. Returns the integer count; does not mutate the module.
    return sum(
        1 if (n.op == op and n.target == target_op) else 0 for n in module.graph.nodes
    )


def count_call_function(module: torch.fx.GraphModule, target_op: Any) -> int:
    # [FUNCTION DOC] HELPER: Counts call_function nodes in the module graph with the given target.
    return count_call(module, "call_function", target_op)


def count_call_method(module: torch.fx.GraphModule, target_op: Any) -> int:
    # [FUNCTION DOC] HELPER: Counts call_method nodes in the module graph with the given method
    # name.
    return count_call(module, "call_method", target_op)


class TestFxFusion(TestCase):
    # [CLASS DOC] Test case checking that the pre-grad FX passes rewrite the expected patterns and
    # preserve numerics. All tests are CPU-only and stateless.
    def test_sink_cat_after_pointwise(self):
        # [FUNCTION DOC] Verifies sink_cat_after_pointwise on several spellings of the cat call
        # (keyword dim, positional dim, default dim, tensors keyword, axis keyword, out set to
        # none). For each, the traced and transformed result must match eager and the graph must
        # contain two tanh method calls (the pointwise op is pushed onto each cat input).
        def test_kwarg(x, y):
            # [FUNCTION DOC] HELPER: Cat of two inputs with dim keyword, two views, then tanh.
            return torch.cat([x, y], dim=-1).view(-1).view(128).tanh()

        def test_arg(x, y):
            # [FUNCTION DOC] HELPER: Same as the keyword variant but dim is passed positionally.
            return torch.cat([x, y], -1).view(-1).view(128).tanh()

        def test_arg2(x, y):
            # [FUNCTION DOC] HELPER: Cat with the default dim, two views, then tanh.
            return torch.cat([x, y]).view(-1).view(128).tanh()

        def test_kwarg2(x, y):
            # [FUNCTION DOC] HELPER: Cat using both tensors and dim keywords, then tanh.
            return torch.cat(tensors=[x, y], dim=0).tanh()

        def test_kwarg3(x, y):
            # [FUNCTION DOC] HELPER: Cat using tensors and dim keywords, a view to 128, then tanh.
            return torch.cat(tensors=[x, y], dim=0).view(128).tanh()

        def test_axis(x, y):
            # [FUNCTION DOC] HELPER: Cat using the axis keyword alias for dim, then tanh.
            return torch.cat([x, y], axis=-1).tanh()

        def test_out_none(x, y):
            # [FUNCTION DOC] HELPER: Cat with an explicit out argument of none, then tanh.
            return torch.cat([x, y], dim=-1, out=None).tanh()

        trace_func = chain_passes(torch.fx.symbolic_trace, sink_cat_after_pointwise)
        inputs = [
            torch.randn(8, 8),
            torch.randn(8, 8),
        ]
        for f in [
            test_kwarg,
            test_arg,
            test_arg2,
            test_kwarg2,
            test_kwarg3,
            test_axis,
            test_out_none,
        ]:
            traced = trace_func(f, inputs)
            torch.testing.assert_close(f(*inputs), traced(*inputs))
            self.assertEqual(count_call_method(traced, "tanh"), 2)

    def test_sink_cat_after_pointwise_out_kwarg(self):
        # [FUNCTION DOC] Verifies that sink_cat_after_pointwise leaves a cat with a real out
        # tensor alone: results match eager, and the graph still has exactly one cat call and
        # one tanh method call (no duplication of the pointwise op).
        out = torch.empty(8, 16)

        def test_out(x, y):
            # [FUNCTION DOC] HELPER: Cat of two inputs written into the captured out tensor,
            # followed by tanh.
            return torch.cat([x, y], dim=-1, out=out).tanh()

        trace_func = chain_passes(torch.fx.symbolic_trace, sink_cat_after_pointwise)
        inputs = [
            torch.randn(8, 8),
            torch.randn(8, 8),
        ]
        traced = trace_func(test_out, inputs)
        torch.testing.assert_close(test_out(*inputs), traced(*inputs))
        self.assertEqual(count_call_function(traced, torch.cat), 1)
        self.assertEqual(count_call_method(traced, "tanh"), 1)

    def test_linear_permute_fusion(self):
        # [FUNCTION DOC] Verifies linear_permute_fusion, with and without bias. After the pass
        # no plain linear calls remain and exactly one linear_transpose call exists; the
        # module output must match eager.
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a weight (and optional bias) parameter that applies
            # a linear layer then permutes the last two dimensions of a three-dimensional result.
            def __init__(self, k: int, n: int, has_bias: bool):
                # [FUNCTION DOC] HELPER: Creates a random weight of shape n by k and, if has_bias
                # is set, a random bias of length n; records has_bias.
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(n, k))
                self.has_bias = has_bias
                if has_bias:
                    self.bias = torch.nn.Parameter(torch.randn(n))

            def forward(self, input: torch.Tensor):
                # [FUNCTION DOC] HELPER: Applies linear (with bias only if configured), then
                # swaps dimensions 1 and 2 of the result and returns it.
                if self.has_bias:
                    a0 = torch.nn.functional.linear(input, self.weight, self.bias)
                else:
                    a0 = torch.nn.functional.linear(input, self.weight)
                b0 = a0.permute(0, 2, 1)
                return b0

        m, k, n = 16, 8, 4
        trace_func = chain_passes(torch.fx.symbolic_trace, linear_permute_fusion)
        for has_bias in [True, False]:
            module = TestModule(k, n, has_bias).eval()
            input = torch.randn(6, m, k)
            traced = trace_func(module, [input])
            num_linear = count_call_function(traced, torch.nn.functional.linear)
            num_linear_transpose = count_call_function(traced, linear_transpose)
            self.assertEqual(num_linear, 0)
            self.assertEqual(num_linear_transpose, 1)

            torch.testing.assert_close(module(input), traced(input))

    def test_permute_linear_fusion(self):
        # [FUNCTION DOC] Verifies permute_linear_fusion, with and without bias. After the pass
        # no plain linear calls remain and exactly one transpose_linear call exists; the
        # module output must match eager.
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that permutes the last two dimensions of its input and
            # then applies a linear layer with a stored weight and optional bias.
            def __init__(self, k: int, n: int, has_bias: bool):
                # [FUNCTION DOC] HELPER: Creates a random weight of shape n by k and, if has_bias
                # is set, a random bias of length n; records has_bias.
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(n, k))
                self.has_bias = has_bias
                if has_bias:
                    self.bias = torch.nn.Parameter(torch.randn(n))

            def forward(self, input: torch.Tensor):
                # [FUNCTION DOC] HELPER: Swaps dimensions 1 and 2 of the input, then applies
                # linear with the bias only if configured, and returns the result.
                input1 = input.permute(0, 2, 1)
                if self.has_bias:
                    return torch.nn.functional.linear(input1, self.weight, self.bias)
                return torch.nn.functional.linear(input1, self.weight)

        m, k, n = 16, 8, 4

        trace_func = chain_passes(torch.fx.symbolic_trace, permute_linear_fusion)
        for has_bias in [True, False]:
            module = TestModule(k, n, has_bias).eval()
            input = torch.randn(6, k, m)
            traced = trace_func(module, [input])
            num_linear = count_call_function(traced, torch.nn.functional.linear)
            num_transpose_linear = count_call_function(traced, transpose_linear)
            self.assertEqual(num_linear, 0)
            self.assertEqual(num_transpose_linear, 1)

            torch.testing.assert_close(module(input), traced(input))

    def test_permute_bmm_fusion(self):
        # [FUNCTION DOC] Verifies permute_matmul_fusion for a batched matmul. After the pass no
        # bmm calls remain and exactly one transpose_matmul call exists; output matches eager.
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a fixed random tensor of shape batch by k by n
            # (a plain attribute, not a parameter) and multiplying the permuted input by it.
            def __init__(self, batch: int, k: int, n: int):
                # [FUNCTION DOC] HELPER: Stores a random tensor of shape batch by k by n as the
                # other operand.
                super().__init__()
                self.other = torch.randn(batch, k, n)

            def forward(self, input: torch.Tensor):
                # [FUNCTION DOC] HELPER: Swaps dimensions 1 and 2 of the input, then runs bmm
                # against the stored tensor and returns the result.
                input1 = input.permute(0, 2, 1)
                output = torch.bmm(input1, self.other)
                return output

        batch, m, k, n = 6, 16, 8, 4

        trace_func = chain_passes(torch.fx.symbolic_trace, permute_matmul_fusion)
        module = TestModule(batch, k, n).eval()
        input = torch.randn(batch, k, m)
        traced = trace_func(module, [input])
        num_bmm = count_call_function(traced, torch.bmm)
        num_transpose_matmul = count_call_function(traced, transpose_matmul)
        self.assertEqual(num_bmm, 0)
        self.assertEqual(num_transpose_matmul, 1)

        torch.testing.assert_close(module(input), traced(input))


if __name__ == "__main__":
    run_tests()
