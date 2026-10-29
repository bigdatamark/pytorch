# Owner(s): ["module: functorch"]
# [MODULE DOC] Tests for functorch memory-efficient fusion and for the FX common subexpression
# elimination pass (fx_graph_cse). Defines several activation functions, a helper that compares
# eager and fused results (forward and gradients) on CUDA, and a helper that checks the CSE pass
# node counts, idempotence and numerics. Test classes cover activations, no-change cases,
# reductions and random-op graphs. Import-time side effect: sets the HAS_CUDA flag.

import inspect
import random
import unittest
from collections.abc import Callable

import torch
import torch.fx as fx
import torch.nn as nn
from functorch import make_fx
from functorch.compile import memory_efficient_fusion
from torch._functorch.compile_utils import fx_graph_cse
from torch.nn import functional as F
from torch.testing._internal.common_utils import run_tests, TestCase


HAS_CUDA = torch.cuda.is_available()


def _num_args(fn: Callable):
    # [FUNCTION DOC] HELPER: Returns how many parameters the given callable's signature has.
    return len(inspect.signature(fn).parameters)


def gelu_bias(bias, y):
    # [FUNCTION DOC] HELPER: Bias-add followed by tanh-approximated GELU; returns a tensor computed
    # from the sum of the two inputs.
    x = bias + y
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


def swish(x):
    # [FUNCTION DOC] HELPER: Swish activation; multiplies the input by its sigmoid.
    return x * torch.sigmoid(x)


def mish(x):
    # [FUNCTION DOC] HELPER: Mish activation; multiplies the input by tanh of its softplus.
    return x.mul(torch.tanh(F.softplus(x)))


def hard_sigmoid(x):
    # [FUNCTION DOC] HELPER: Hard sigmoid; adds 3, clamps to the range 0 to 6, divides by 6.
    return (x + 3.0).clamp(min=0.0, max=6.0).div(6.0)


def hard_swish(x):
    # [FUNCTION DOC] HELPER: Hard swish; input times the hard sigmoid of the input.
    return x * (x + 3.0).clamp(min=0.0, max=6.0).div(6.0)


def hard_mish(x):
    # [FUNCTION DOC] HELPER: Hard mish; half the input times the input plus 2 clamped to 0..2.
    # Currently unused (its test is commented out with a TODO about an assertion failure).
    return 0.5 * x * (x + 2.0).clamp(min=0.0, max=2.0)


# todo: convert these into tests
# def group_std(x, groups: int = 32, eps: float = 1e-5, flatten: bool = False):
#     B, C, H, W = x.shape
#     x_dtype = x.dtype
#     if flatten:
#         x = x.reshape(B, groups, -1)  # FIXME simpler shape causing TPU / XLA issues
#         std = x.float().var(dim=2, unbiased=False, keepdim=True).add(eps).sqrt().to(x_dtype)
#     else:
#         x = x.reshape(B, groups, C // groups, H, W)
#         std = x.float().var(dim=(2, 3, 4), unbiased=False, keepdim=True).add(eps).sqrt().to(x_dtype)
#     return std.expand(x.shape).reshape(B, C, H, W)

# class EvoNorm2dS0(nn.Module):
#     def __init__(self, num_features, groups=32, group_size=None, apply_act=True, eps=1e-5, **_):
#         super().__init__()
#         self.apply_act = apply_act  # apply activation (non-linearity)
#         if group_size:
#             assert num_features % group_size == 0
#             self.groups = num_features // group_size
#         else:
#             self.groups = groups
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(num_features))
#         self.bias = nn.Parameter(torch.zeros(num_features))
#         self.v = nn.Parameter(torch.ones(num_features)) if apply_act else None
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.ones_(self.weight)
#         nn.init.zeros_(self.bias)
#         if self.v is not None:
#             nn.init.ones_(self.v)

#     def forward(self, x):
#         x_dtype = x.dtype
#         v_shape = (1, -1, 1, 1)
#         if self.v is not None:
#             v = self.v.view(v_shape).to(dtype=x_dtype)
#             x = x * (x * v).sigmoid() / group_std(x, self.groups, self.eps)
#         return x * self.weight.view(v_shape).to(dtype=x_dtype) + self.bias.view(v_shape).to(dtype=x_dtype)


# device = "cuda"
# dtype = torch.float

# evo_norm = EvoNorm2dS0(2048)
# evo_norm_inp = [(128, 2048, 8, 8)]


def run_and_compare_activation(self, fn, inps):
    # [FUNCTION DOC] HELPER: Compares eager execution of fn against its memory_efficient_fusion
    # version on CUDA in float32 under the fuser1 JIT fuser. Builds random inputs of the given
    # shapes (requiring grad), runs the reference forward and backward, then runs the fused
    # function five times, clearing gradients on each pass. Asserts via the test case self that
    # outputs and input gradients match. Modules are moved to CUDA first. Needs CUDA.
    with torch.jit.fuser("fuser1"):
        device = "cuda"
        dtype = torch.float
        if isinstance(fn, nn.Module):
            fn = fn.to(device=device, dtype=dtype)

        ref_args = [
            torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
            for shape in inps
        ]
        res_args = [i.detach().clone().requires_grad_(True) for i in ref_args]

        ref = fn(*ref_args)
        ref.sum().backward()

        mem_optimized_fn = memory_efficient_fusion(fn)
        for _ in range(5):
            for i in res_args:
                i.grad = None
            res = mem_optimized_fn(*res_args)
            res.sum().backward()

        self.assertEqual(ref, res)
        for ref_arg, res_arg in zip(ref_args, res_args):
            self.assertEqual(ref_arg.grad, res_arg.grad)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
class TestMemoryEfficientOpAuthoring(TestCase):
    # [CLASS DOC] CUDA-only test case comparing memory-efficient fusion against eager mode for
    # several activation and normalization functions. Stateless; each test calls the shared compare
    # helper with a function and input shapes. Skipped when CUDA is unavailable.
    def test_gelu_bias(self):
        # [FUNCTION DOC] Checks fusion of the bias-add GELU on two 1024-element inputs.
        run_and_compare_activation(self, gelu_bias, [(1024,), (1024,)])

    def test_mish(self):
        # [FUNCTION DOC] Checks fusion of mish on one 1024-element input.
        run_and_compare_activation(self, mish, [(1024,)])

    def test_swish(self):
        # [FUNCTION DOC] Checks fusion of swish on one 1024-element input.
        run_and_compare_activation(self, swish, [(1024,)])

    def test_hard_sigmoid(self):
        # [FUNCTION DOC] Checks fusion of hard sigmoid on one 1024-element input.
        run_and_compare_activation(self, hard_sigmoid, [(1024,)])

    def test_hard_swish(self):
        # [FUNCTION DOC] Checks fusion of hard swish on one 1024-element input.
        run_and_compare_activation(self, hard_swish, [(1024,)])

    def test_layer_norm(self):
        # [FUNCTION DOC] Checks fusion of a hand-written layer norm over inputs of shape 10 by 16,
        # a weight of 16 and a bias of 16.
        def layer_norm(x, weight, bias):
            # [FUNCTION DOC] HELPER: Layer norm over the last dimension with epsilon 1e-5; computes
            # mean and variance manually, normalizes, then applies weight and bias.
            dim = -1
            eps = 1e-5
            mean = torch.mean(x, dim, keepdim=True)
            centered = x - mean
            var = torch.sum(centered * centered, dim, keepdim=True) / x.size(-1)
            rvar = 1.0 / torch.sqrt(var + eps)
            normed = (x - mean) * rvar
            return normed * weight + bias

        bs = 10
        ln_size = 16
        layer_norm_inps = [(bs, ln_size), (ln_size,), (ln_size,)]
        run_and_compare_activation(self, layer_norm, layer_norm_inps)

    def test_rmsnorm(self):
        # [FUNCTION DOC] Checks fusion of a T5-style RMS norm module on an input of shape 256 by
        # 256 by 1024.
        class T5LayerNorm(nn.Module):
            # [CLASS DOC] HELPER: Module with a learnable weight of ones and an epsilon attribute;
            # normalizes by the root mean square over the last dimension without mean subtraction.
            def __init__(self, hidden_size, eps=1e-6):
                # [FUNCTION DOC] HELPER: Stores a ones-initialized weight parameter of the given
                # hidden size and the variance epsilon.
                """
                Construct a layernorm module in the T5 style No bias and no subtraction of mean.
                """
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))
                self.variance_epsilon = eps

            def forward(self, hidden_states):
                # [FUNCTION DOC] HELPER: Computes variance as the mean of squares in float32,
                # scales the input by the reciprocal square root of variance plus epsilon, casts to
                # the weight dtype if that is half or bfloat16, and returns weight times result.
                # layer norm should always be calculated in float32
                variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(
                    variance + self.variance_epsilon
                )

                # convert into half-precision if necessary
                if self.weight.dtype in [torch.float16, torch.bfloat16]:
                    hidden_states = hidden_states.to(self.weight.dtype)

                return self.weight * hidden_states

        bs = 256
        seq = 256
        hidden = 1024
        t5_norm = T5LayerNorm(hidden)
        t5_norm_inputs = [(bs, seq, hidden)]
        run_and_compare_activation(self, t5_norm, t5_norm_inputs)

    # TODO - Assertion failure
    # def test_hard_mish(self):
    #   for compiler in compilers:
    #     run_and_compare_activation(hard_mish, 1024)


# check if the CSE modified graph of f has delta less nodes, and do not reduce the number of nodes further on a second pass.
# delta is an integer >= -1. If delta = -1, only check if the new graph
#   has less or equal number of nodes
def check(f, t, delta, check_val=True, graph_input=False):
    # [FUNCTION DOC] HELPER: Runs the CSE pass on a graph and validates it. The graph comes
    # from tracing f with make_fx on t, or is f itself when graph_input is true. A delta of -1
    # only requires that node count not grow; otherwise old count must equal new count plus
    # delta. A second pass must not remove further nodes. If check_val, the original and
    # rewritten modules are run on t and results compared (both None, or equal). Raises
    # AssertionError on any violation.
    if graph_input:
        fx_g = f
    else:
        fx_g = make_fx(f)(t)
    new_graph = fx_graph_cse(fx_g.graph)
    new_g = fx.GraphModule(fx_g, new_graph)

    # the number of nodes decrease/ or stay the same
    old_num_nodes = len(fx_g.graph.nodes)
    new_num_nodes = len(new_graph.nodes)
    if delta == -1:
        if old_num_nodes < new_num_nodes:
            raise AssertionError(
                f"number of nodes increased {old_num_nodes}, {new_num_nodes}"
            )
    else:
        if old_num_nodes != new_num_nodes + delta:
            raise AssertionError(
                f"number of nodes not the same {old_num_nodes - delta}, {new_num_nodes}\n {fx_g.graph} \n {new_graph}"
            )

    # a second pass should not reduce more nodes
    pass_2_graph = fx_graph_cse(new_graph)
    pass_2_num_nodes = len(pass_2_graph.nodes)
    if pass_2_num_nodes != new_num_nodes:
        raise AssertionError(
            f"second pass graph has less node {pass_2_num_nodes}, {new_num_nodes}\n {new_graph} \n {pass_2_graph}"
        )

    # check correctness
    if check_val:
        true_result = fx_g(t)
        our_result = new_g(t)
        if true_result is None:  # both return None
            if our_result is not None:
                raise AssertionError(f"true result is None, CSE result is {our_result}")
        else:  # results returned are the same
            if not torch.all(true_result == our_result):
                raise AssertionError(
                    f"results are different {true_result}, {our_result}"
                )


class NoChangeTestCase(TestCase):
    # [CLASS DOC] Test case where CSE must not merge anything (or only match as stated), such as
    # random ops, signed zeros and differing NaN signs. Stateless.
    def test_nochange(self):
        # [FUNCTION DOC] Graph with no duplicate computations; expects no node reduction.
        def f(x):
            # [FUNCTION DOC] HELPER: Traced function reusing the input in sums; no duplicates.
            a = x + 1
            b = x + a
            a = x
            d = x + a
            return b + d

        t = torch.randn(2, 2)
        check(f, t, 0)

    def test_empty(self):
        # [FUNCTION DOC] Function with an empty body returning None; expects no change.
        def f(x):
            # [FUNCTION DOC] HELPER: Does nothing and returns None.
            pass

        t = torch.randn(2, 2)
        check(f, t, 0)

    def test_rand_like(self):
        # [FUNCTION DOC] Two rand_like calls must not be merged; value check is skipped.
        def f(x):
            # [FUNCTION DOC] HELPER: Adds two independent random tensors shaped like the input.
            a = torch.rand_like(x)
            b = torch.rand_like(x)
            return a + b

        t = torch.randn(2, 2)
        check(f, t, 0, check_val=False)

    def test_rand_n(self):
        # [FUNCTION DOC] Two randn calls must not be merged; value check is skipped.
        def f(x):
            # [FUNCTION DOC] HELPER: Adds two independent random vectors of length 4.
            a = torch.randn(4)
            b = torch.randn(4)
            return a + b

        t = torch.randn(2, 2)
        check(f, t, 0, check_val=False)

    def test_hash_with_numbers(self):
        # [FUNCTION DOC] Regression test for hash collisions between an integer and a float
        # constant. Skipped when running compiled. Compiles a function with a dynamic-shape compile
        # backend that records the graph, asserts exactly one graph was captured, and expects CSE
        # to change nothing in it (graph passed directly, no value check).
        # Test to repro issue with fx_graph_cse when
        # hash((primals_2, 1.0)) == hash((primals_2, 1))

        if torch._dynamo.is_compiling():
            self.skipTest("Unsupported if test run is compiled")

        def f(inpt, osize):
            # [FUNCTION DOC] HELPER: Computes a scale from the last dimension size (as an integer
            # and as a float) and the output size, clamps the input and multiplies by the scale.
            size = inpt.shape[-1]
            s1 = size - 1
            s2 = size - 1.0
            scale = s2 / (osize - 1.0)
            inpt = torch.clamp(inpt, 0, s1)
            return scale * inpt

        # Fetch dynamic graph
        gms = []

        def toy_backend(gm, _):
            # [FUNCTION DOC] HELPER: Backend that records the graph module in the list and returns
            # its forward method.
            gms.append(gm)
            return gm.forward

        torch._dynamo.reset()
        fn = torch.compile(backend=toy_backend, dynamic=True)(f)

        t = torch.rand(3, 100)
        _ = fn(t, 50)
        if len(gms) != 1:
            raise AssertionError(f"Expected 1 graph module, got {len(gms)}: {gms}")
        fx_g = gms[0]
        check(fx_g, None, 0, check_val=False, graph_input=True)

    def test_neg_nan_not_merged(self):
        # [FUNCTION DOC] Fills with NaN and with negative NaN; the two must not be merged.
        def f(x):
            # [FUNCTION DOC] HELPER: Adds a NaN-filled and a negative-NaN-filled tensor.
            a = torch.full_like(x, float("nan"))
            b = torch.full_like(x, -float("nan"))
            return a + b

        t = torch.randn(2, 2)
        check(f, t, 0, check_val=False)

    def test_neg_zero_not_merged(self):
        # [FUNCTION DOC] Fills with 0.0 and -0.0; they must not be merged since reciprocals differ.
        def f(x):
            # [FUNCTION DOC] HELPER: Stacks reciprocals of tensors filled with positive
            # and negative zero.
            a = torch.full_like(x, 0.0)
            b = torch.full_like(x, -0.0)
            return torch.stack([a.reciprocal(), b.reciprocal()])

        t = torch.randn(2, 2)
        check(f, t, 0)

    def test_complex_neg_zero_not_merged(self):
        # [FUNCTION DOC] Complex tensors filled with positive and negative zero real parts must
        # not be merged.
        def f(x):
            # [FUNCTION DOC] HELPER: Casts input to complex, stacks reciprocals of the real parts
            # of two tensors filled with complex values differing in the sign of zero.
            y = x.to(torch.cfloat)
            a = torch.full_like(y, complex(0.0, 0.0))
            b = torch.full_like(y, complex(-0.0, 0.0))
            return torch.stack([a.real.reciprocal(), b.real.reciprocal()])

        t = torch.randn(2, 2)
        check(f, t, 0)


class ReduceTestCase(TestCase):
    # [CLASS DOC] Test case where CSE must reduce the node count by a stated amount: duplicated ops
    # with list arguments, keyword arguments and NaN fills. Stateless.
    def test_immutable_list_type(self):
        # [FUNCTION DOC] Duplicate sums with list-typed and default dim arguments; expects
        # 2 fewer nodes.
        def f(x):
            # [FUNCTION DOC] HELPER: Sums over dim 1 twice and over all dims twice, then adds all.
            a = x.sum(dim=1)
            b = x.sum(dim=1)
            c = x.sum()
            d = x.sum()
            return a + b + c + d

        t = torch.randn(2, 2)
        check(f, t, 2)

    def test_immutable_list_multiple_entries(self):
        # [FUNCTION DOC] Duplicate sums over a two-element dim list and over dim 1; expects
        # 2 fewer nodes.
        def f(x):
            # [FUNCTION DOC] HELPER: Sums over dims 0 and 1 twice and over dim 1 twice, adds all.
            a = x.sum(dim=[0, 1])
            b = x.sum(dim=[0, 1])
            c = x.sum(dim=1)
            d = x.sum(dim=1)
            return a + b + c + d

        t = torch.randn(2, 2)
        check(f, t, 2)

    def test_simple(self):
        # [FUNCTION DOC] Duplicate cos calls; expects 2 fewer nodes.
        def f(x):
            # [FUNCTION DOC] HELPER: Computes cos twice, doubles each by self-addition, adds them.
            a = x.cos()
            b = x.cos()
            c = a + a
            d = b + b
            return c + d

        t = torch.randn(2, 2)
        check(f, t, 2)

    def test_simple_2(self):
        # [FUNCTION DOC] Duplicate chains of cos then sin; expects 3 fewer nodes.
        def f(x):
            # [FUNCTION DOC] HELPER: Computes cos then sin twice, doubles each, adds them.
            a = x.cos().sin()
            b = x.cos().sin()
            c = a + a
            d = b + b
            return c + d

        t = torch.randn(1)
        check(f, t, 3)

    def test_two_args_default(self):
        # [FUNCTION DOC] Sums with an explicit default keepdim equal those without; expects
        # 3 fewer nodes.
        def f(x):
            # [FUNCTION DOC] HELPER: Four sums over dim 1, two with explicit keepdim false.
            a = x.sum(dim=1)
            b = x.sum(dim=1, keepdim=False)
            c = x.sum(dim=1, keepdim=False)
            d = x.sum(dim=1)
            return a + b + c + d

        t = torch.randn(2, 2)
        check(f, t, 3)

    def test_two_args(self):
        # [FUNCTION DOC] Sums with keepdim true are distinct from those without; expects
        # 2 fewer nodes.
        def f(x):
            # [FUNCTION DOC] HELPER: Two sums over dim 1 and two with keepdim true, all added.
            a = x.sum(dim=1)
            b = x.sum(dim=1, keepdim=True)
            c = x.sum(dim=1, keepdim=True)
            d = x.sum(dim=1)
            return a + b + c + d

        t = torch.randn(2, 2)
        check(f, t, 2)

    def test_simple_multiple_same_ops(self):
        # [FUNCTION DOC] Four identical full sums collapse to one; expects 3 fewer nodes.
        def f(x):
            # [FUNCTION DOC] HELPER: Adds four identical full-tensor sums.
            a = x.sum()
            b = x.sum()
            c = x.sum()
            d = x.sum()
            return a + b + c + d

        t = torch.randn(2, 2)
        check(f, t, 3)

    def test_nested_immutable_list_type(self):
        # [FUNCTION DOC] Duplicate concatenations of a tuple of tensors; expects 1 fewer node.
        def f(x):
            # [FUNCTION DOC] HELPER: Concatenates the input with itself twice and adds the results.
            a = torch.cat((x, x))
            b = torch.cat((x, x))
            return a + b

        t = torch.randn(2, 2)
        check(f, t, 1)

    def test_kwarg(self):
        # [FUNCTION DOC] Duplicate ones_like calls (keyword/factory style); expects 1 fewer node.
        def f(x):
            # [FUNCTION DOC] HELPER: Adds two ones tensors shaped like the input.
            a = torch.ones_like(x)
            b = torch.ones_like(x)
            return a + b

        t = torch.randn(2, 2)
        check(f, t, 1)

    def test_nan_full_deduplication(self):
        # [FUNCTION DOC] Two identical NaN-filled tensors are merged; expects 1 fewer node,
        # no value check.
        def f(x):
            # [FUNCTION DOC] HELPER: Adds two identical NaN-filled tensors shaped like the input.
            a = torch.full_like(x, float("nan"))
            b = torch.full_like(x, float("nan"))
            return a + b

        t = torch.randn(2, 2)
        check(f, t, 1, check_val=False)

    def test_nan_dedup_non_factory_op(self):
        # [FUNCTION DOC] Two identical clamps with a NaN minimum are merged; expects 1 fewer node.
        def f(x):
            # [FUNCTION DOC] HELPER: Adds two identical clamps of the input with a NaN minimum.
            a = x.clamp(min=float("nan"))
            b = x.clamp(min=float("nan"))
            return a + b

        t = torch.randn(2, 2)
        check(f, t, 1, check_val=False)

    def test_nan_dedup_constant_pad(self):
        # [FUNCTION DOC] Two identical constant pads with a NaN value are merged; expects
        # 1 fewer node.
        def f(x):
            # [FUNCTION DOC] HELPER: Adds two identical NaN-valued pads of the input.
            a = torch.nn.functional.pad(x, (1, 1, 1, 1), value=float("nan"))
            b = torch.nn.functional.pad(x, (1, 1, 1, 1), value=float("nan"))
            return a + b

        t = torch.randn(2, 2)
        check(f, t, 1, check_val=False)

    def test_complex_nan_full_deduplication(self):
        # [FUNCTION DOC] Two identical complex NaN fills are merged; expects 1 fewer node.
        def f(x):
            # [FUNCTION DOC] HELPER: Casts input to complex and adds two identical
            # complex NaN fills.
            y = x.to(torch.cfloat)
            a = torch.full_like(y, complex(float("nan"), 0.0))
            b = torch.full_like(y, complex(float("nan"), 0.0))
            return a + b

        t = torch.randn(2, 2)
        check(f, t, 1, check_val=False)


class RandomOpTestCase(TestCase):
    # [CLASS DOC] Test case running CSE on randomly generated graphs. Stateless.
    def test_random(self):
        # [FUNCTION DOC] Builds a function applying 100 randomly chosen unary ops (clone, cos,
        # tanh, gelu) to randomly chosen earlier values, symbolically traces it, removes dead code,
        # recompiles, then 30 times runs the check helper requiring only that node count not grow.
        # Uses the global random module, so the traced graph varies between runs.
        def f(x):
            # [FUNCTION DOC] HELPER: Function with the random op chain described above; returns
            # the last value.
            vals = [x]
            ops = [torch.clone, torch.cos, torch.tanh, torch.nn.functional.gelu]
            for _ in range(100):
                new_val = random.choice(ops)(random.choice(vals))
                vals.append(new_val)
            return vals[-1]

        fx_g = fx.symbolic_trace(f)
        fx_g.graph.eliminate_dead_code()
        fx_g.recompile()
        t = torch.randn(2, 2)

        for _ in range(30):
            check(fx_g, t, -1, graph_input=True)


if __name__ == "__main__":
    run_tests()
