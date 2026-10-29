# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for inductor's group and batch fusion FX passes (group_batch_fusion): group
# linear, batch layernorm, batch linear (lhs, pre-grad, post-grad), pointwise, math and dropout
# batching, the cat-then-linear rewrite, and the greedy independent-subset search used to pick
# fusion candidates. Defines small fixture modules, then test cases that compile them and check
# the pass counters from the dynamo counters registry plus numerics versus eager. Import-time
# side effect: tries to import an fbgemm inductor lowerings module and records availability in
# has_fbgemm; no other global state is touched.

import collections
import operator
import unittest

import torch
import torch._dynamo
import torch._inductor
import torch._inductor.fx_passes.group_batch_fusion
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE, requires_gpu


try:
    # importing this will register fbgemm lowerings for inductor
    import deeplearning.fbgemm.fbgemm_gpu.fb.inductor_lowerings  # noqa: F401

    has_fbgemm = True
except Exception:
    has_fbgemm = False


class _TestHighwaySelfGating(torch.nn.Module):
    # [CLASS DOC] HELPER: Fixture module with a highway-style gate. See forward for the
    # computation.
    def __init__(
        self,
        d_model: int,
        size: int,
        device="cuda",
        # [FUNCTION DOC] HELPER: Stores the list length and device, builds two linear layers of
        # equal d_model size (gating and transform) and a sigmoid, all moved to the device.
    ) -> None:
        super().__init__()
        self.size = size
        self.device = device
        self.gating_proj = torch.nn.Linear(d_model, d_model).to(self.device)
        self.transform_proj = torch.nn.Linear(d_model, d_model).to(self.device)
        self.gating_func = torch.nn.Sigmoid().to(self.device)

        self.d_model = d_model

    def forward(
        self,
        inputs: list[torch.Tensor],
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: For each of the first size tensors in the input list,
        # multiplies the gating projection by the sigmoid of the transform projection, and
        # returns all results joined along the last dimension.
        results = []
        for i in range(self.size):
            x = inputs[i]
            gating_proj = self.gating_proj(x)
            transform_proj = self.transform_proj(x)
            x = gating_proj * self.gating_func(transform_proj)
            results.append(x)

        return torch.cat(results, dim=-1)


class MyModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Fixture with three groups of ten square linear layers kept in plain
    # Python lists.
    # KNOWN ISSUE: the layers sit in plain lists, not registered submodules, so named_parameters
    # of this module is empty and the parameter and gradient comparisons are vacuous.
    def __init__(self, z: int, has_bias: bool, device="cuda") -> None:
        # [FUNCTION DOC] HELPER: Records size z and device, fixes the sequence length at ten,
        # and builds three lists of ten z by z linear layers (with or without bias) on the
        # device.
        super().__init__()
        self.z = z
        self.device = device
        self.seq_len = 10
        self.seq1 = [
            torch.nn.Linear(z, z, has_bias).to(self.device) for _ in range(self.seq_len)
        ]
        self.seq2 = [
            torch.nn.Linear(z, z, has_bias).to(self.device) for _ in range(self.seq_len)
        ]
        self.seq3 = [
            torch.nn.Linear(z, z, has_bias).to(self.device) for _ in range(self.seq_len)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Builds ten shifted copies of the input, passes them through the
        # first linear group, shifts back, mixes some copies from the earlier stage with later
        # ones, applies the second group, shifts again, picks another mixed set of ten tensors,
        # applies the third group and returns the results concatenated along dimension one.
        x1 = [x + 0.1 * i for i in range(self.seq_len)]
        x2 = [self.seq1[i](x1[i]) for i in range(self.seq_len)]
        x3 = [x2[i] - 0.1 * i for i in range(self.seq_len)]
        x4 = [x1[i] for i in range(3)] + [x3[i] for i in range(3, self.seq_len)]
        x5 = [self.seq2[i](x4[i]) for i in range(self.seq_len)]
        x6 = [x5[i] + 0.1 * (self.seq_len - i) for i in range(self.seq_len)]
        x7 = (
            [x1[i] for i in range(4)]
            + [x3[i] for i in range(6, 8)]
            + [x6[i] for i in range(4)]
        )
        x8 = [self.seq3[i](x7[i]) for i in range(self.seq_len)]
        x9 = torch.cat(x8, dim=1)
        return x9


class MyModule2(torch.nn.Module):
    # [CLASS DOC] HELPER: Fixture with six linear layers of differing input widths (six, eight,
    # ten) and three batch norm layers, used to exercise group linear fusion with different
    # shapes.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates linear0 to linear5 (inputs 6, 8, 10, 6, 8, 10; output
        # 8) and batch norms bn0 to bn2 with eight features.
        super().__init__()
        self.linear0 = torch.nn.Linear(6, 8)
        self.linear1 = torch.nn.Linear(8, 8)
        self.linear2 = torch.nn.Linear(10, 8)
        self.linear3 = torch.nn.Linear(6, 8)
        self.linear4 = torch.nn.Linear(8, 8)
        self.linear5 = torch.nn.Linear(10, 8)
        self.bn0 = torch.nn.BatchNorm1d(8)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Splits the input along dimension one into widths six, eight,
        # ten. The three pieces get a constant offset, a linear and a batch norm each; the same
        # pieces also go through sin, cos and a scaled sin followed by the remaining linears.
        # All six outputs are concatenated along dimension zero and passed through a sigmoid.
        t = torch.split(x, [6, 8, 10], dim=1)
        a0 = self.bn0(self.linear0(t[0] + 0.1))
        a1 = self.bn1(self.linear1(t[1] + 0.2))
        a2 = self.bn2(self.linear2(t[2] + 0.3))
        a3 = self.linear3(torch.sin(t[0]))
        a4 = self.linear4(torch.cos(t[1]))
        a5 = self.linear5(torch.sin(t[2] * 0.5))

        b = torch.cat([a0, a1, a2, a3, a4, a5])
        return torch.sigmoid(b)


class MyModule3(torch.nn.Module):
    # [CLASS DOC] HELPER: Fixture applying two rounds of layer norm over split chunks, with
    # optional per-chunk weights and biases. Used for the batch layernorm fusion test.
    def __init__(self, device, has_weight=True, has_bias=True):
        # [FUNCTION DOC] HELPER: Stores the device. Creates two parameter lists of five
        # one-dimensional size-ten tensors (scale0, bias0) and two lists of five tensors of
        # shape five by ten (scale1, bias1). The scale1 and bias1 entries are replaced by lists
        # of None when has_weight or has_bias is false. Parameter lists are moved to the device.
        super().__init__()
        self.device = device
        self.scale0 = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(10)) for _ in range(5)]
        ).to(self.device)
        self.bias0 = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(10)) for _ in range(5)]
        ).to(self.device)
        self.scale1 = (
            torch.nn.ParameterList(
                [torch.nn.Parameter(torch.randn(5, 10)) for _ in range(5)]
            ).to(self.device)
            if has_weight
            else [None for _ in range(5)]
        )
        self.bias1 = (
            torch.nn.ParameterList(
                [torch.nn.Parameter(torch.randn(5, 10)) for _ in range(5)]
            ).to(self.device)
            if has_bias
            else [None for _ in range(5)]
        )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Moves the input to the device, splits the last dimension into
        # chunks of ten, layer-normalizes each over size ten with scale0 and bias0,
        # concatenates, splits again, layer-normalizes each chunk over the trailing five by ten
        # shape with the optional scale1 and bias1, and returns the chunks concatenated on
        # dimension two.
        l1_out = torch.split(x.to(self.device), 10, dim=2)
        post_l1 = [
            torch.nn.functional.layer_norm(
                l1_out[i], (10,), weight=self.scale0[i], bias=self.bias0[i]
            )
            for i in range(len(l1_out))
        ]
        l1_out = torch.cat(post_l1, dim=2)

        l2_out = torch.split(l1_out, 10, dim=2)
        post_l2 = [
            torch.nn.functional.layer_norm(
                l2_out[i], (5, 10), weight=self.scale1[i], bias=self.bias1[i]
            )
            for i in range(len(l2_out))
        ]

        return torch.cat(post_l2, dim=2)


class MyModule4(torch.nn.Module):
    # [CLASS DOC] HELPER: Fixture applying ten linears with a shared input (varying output
    # sizes) twice, with a concat and split between, used for batch_linear_lhs fusion.
    # KNOWN ISSUE: the weights and biases are plain Python lists, not registered parameters, so
    # parameter and gradient comparisons see an empty set.
    def __init__(self, z, device, has_bias):
        # [FUNCTION DOC] HELPER: Stores size, device, bias flag and sequence length ten. Builds
        # two lists of ten weight tensors with row counts that decrease by index modulo five,
        # and, when has_bias is set, matching bias lists, all moved to the device.
        super().__init__()
        self.z = z
        self.device = device
        self.has_bias = has_bias
        self.seq_len = 10
        self.weights1 = [
            torch.nn.Parameter(torch.randn(z - i % 5, z)).to(self.device)
            for i in range(self.seq_len)
        ]
        self.weights2 = [
            torch.nn.Parameter(torch.randn(z - i % 5, z)).to(self.device)
            for i in range(self.seq_len)
        ]

        if has_bias:
            self.biases1 = [
                torch.nn.Parameter(torch.randn(z - i % 5)).to(self.device)
                for i in range(self.seq_len)
            ]
            self.biases2 = [
                torch.nn.Parameter(torch.randn(z - i % 5)).to(self.device)
                for i in range(self.seq_len)
            ]

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Adds a constant to the input, applies ten functional linears
        # sharing it as the left operand, concatenates on dimension one, splits into chunks of
        # ten and concatenates them on dimension zero, applies ten more linears to that result,
        # concatenates and returns the sigmoid.
        x = x + 1.2
        x1 = [
            torch.nn.functional.linear(
                x, self.weights1[i], self.biases1[i] if self.has_bias else None
            )
            for i in range(self.seq_len)
        ]
        x2 = torch.cat(x1, dim=1)
        x3 = torch.split(x2, 10, dim=1)
        x4 = torch.cat(x3)
        x5 = [
            torch.nn.functional.linear(
                x4, self.weights2[i], self.biases2[i] if self.has_bias else None
            )
            for i in range(self.seq_len)
        ]
        x6 = torch.cat(x5, dim=1)
        return torch.sigmoid(x6)


class MyModule5(torch.nn.Module):
    # [CLASS DOC] HELPER: Fixture with five parallel linear layers over input slices, for
    # batch_linear (pre-grad) fusion.
    def __init__(self, device, has_bias=True):
        # [FUNCTION DOC] HELPER: Stores the device, builds a parameter list of five 50 by 100
        # weights and either five bias vectors of size 50 or a list of None values when has_bias
        # is false.
        super().__init__()
        self.device = device

        self.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(50, 100)).to(self.device) for _ in range(5)]
        )

        self.biases = (
            ([torch.nn.Parameter(torch.randn(50)).to(self.device) for _ in range(5)])
            if has_bias
            else [None for _ in range(5)]
        )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Splits the input into slices of width 100, applies one
        # functional linear per slice with its own weight and bias, concatenates on dimension
        # one and returns the sin.
        l1_out = torch.split(x.to(self.device), 100, dim=1)
        l1_linear = [
            torch.nn.functional.linear(l1_out[i], self.weights[i], self.biases[i])
            for i in range(len(l1_out))
        ]
        l1_out = torch.cat(l1_linear, dim=1)
        return torch.sin(l1_out)


class _TestPointwiseOps(torch.nn.Module):
    # [CLASS DOC] HELPER: Fixture with parallel pointwise ops (sigmoid, relu, add, mul, sub,
    # div) on split chunks, for pre-grad and post-grad pointwise batching tests.
    def __init__(self, device, has_bias=True):
        # [FUNCTION DOC] HELPER: Stores the device; the has_bias argument is unused.
        super().__init__()
        self.device = device

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Splits the input in two halves of width 500, then each half
        # into chunks of 50, applies sigmoid then relu to each chunk of both halves, adds the
        # two branches pairwise, then multiplies, subtracts and divides each result with itself.
        # Returns the chunks concatenated on dimension one.
        inputs = torch.split(x.to(self.device), 500, dim=1)
        x_split = torch.split(inputs[0].to(self.device), 50, dim=1)
        y_split = torch.split(inputs[1].to(self.device), 50, dim=1)
        sigmoid_1 = [torch.sigmoid(x_split[i]) for i in range(len(x_split))]
        sigmoid_2 = [torch.sigmoid(y_split[i]) for i in range(len(y_split))]
        relu_1 = [torch.nn.functional.relu(sigmoid_1[i]) for i in range(len(sigmoid_1))]
        relu_2 = [torch.nn.functional.relu(sigmoid_2[i]) for i in range(len(sigmoid_2))]
        add = [torch.add(relu_1[i], relu_2[i]) for i in range(len(relu_1))]
        mul = [torch.mul(add[i], add[i]) for i in range(len(add))]
        sub = [torch.sub(mul[i], mul[i]) for i in range(len(mul))]
        div = [torch.div(sub[i], sub[i]) for i in range(len(sub))]
        return torch.cat(div, dim=1)


class _TestPointwiseOpsPostGrad(torch.nn.Module):
    # [CLASS DOC] HELPER: Fixture using aten-level ops (tanh, sigmoid, relu, add) on split
    # chunks, for post-grad batching passes.
    def __init__(self, device):
        # [FUNCTION DOC] HELPER: Stores the device.
        super().__init__()
        self.device = device

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Splits the input into two halves and then chunks of 50 using
        # aten ops, applies tanh, sigmoid and relu per chunk in each half, adds the halves
        # pairwise and returns the sums concatenated on dimension one.
        inputs = torch.ops.aten.split(x.to(self.device), 500, dim=1)
        x_split = torch.ops.aten.split(inputs[0].to(self.device), 50, dim=1)
        y_split = torch.ops.aten.split(inputs[1].to(self.device), 50, dim=1)
        tanh_1 = [torch.ops.aten.tanh(x_split[i]) for i in range(len(x_split))]
        tanh_2 = [torch.ops.aten.tanh(y_split[i]) for i in range(len(y_split))]
        sigmoid_1 = [torch.ops.aten.sigmoid(tanh_1[i]) for i in range(len(tanh_1))]
        sigmoid_2 = [torch.ops.aten.sigmoid(tanh_2[i]) for i in range(len(tanh_2))]
        relu_1 = [torch.ops.aten.relu(sigmoid_1[i]) for i in range(len(sigmoid_1))]
        relu_2 = [torch.ops.aten.relu(sigmoid_2[i]) for i in range(len(sigmoid_2))]
        add = [torch.ops.aten.add(relu_1[i], relu_2[i]) for i in range(len(relu_1))]
        return torch.cat(add, dim=1)


class _TestMathOps(torch.nn.Module):
    # [CLASS DOC] HELPER: Fixture exercising clamp, nan_to_num, detach and stack in lists of
    # ten, for the math op batching passes.
    def __init__(self, device):
        # [FUNCTION DOC] HELPER: Stores the device.
        super().__init__()
        self.device = device

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Makes two lists of ten device copies of the input, clamps each
        # to a symmetric range, replaces non-finite values with a constant, detaches, stacks
        # each list on dimension zero, then stacks the two stacks together and returns the
        # result.
        inputs = [x.to(self.device) for i in range(10)]
        others = [x.to(self.device) for i in range(10)]
        clamp_input = [x.clamp(min=-1000.1, max=1000.1) for x in inputs]
        clamp_other = [x.clamp(min=-1000.1, max=1000.1) for x in others]
        nan_to_num_input = [torch.nan_to_num(x, 7.0) for x in clamp_input]
        nan_to_num_other = [torch.nan_to_num(x, 7.0) for x in clamp_other]
        detach_input = [x.detach() for x in nan_to_num_input]
        detach_other = [x.detach() for x in nan_to_num_other]
        stack_input = torch.stack(detach_input, dim=0)
        stack_other = torch.stack(detach_other, dim=0)
        return torch.stack((stack_input, stack_other), dim=0)


class _TestDropout(torch.nn.Module):
    # [CLASS DOC] HELPER: Fixture applying five independent training-mode dropouts to equal
    # slices.
    def __init__(self, device):
        # [FUNCTION DOC] HELPER: Stores the device.
        super().__init__()
        self.device = device

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC] HELPER: Splits the input into five slices of width 20 along dimension
        # one, applies dropout with probability 0.05 in training mode (not in place) to each and
        # returns the five results as a tuple.
        split = x.split([20, 20, 20, 20, 20], 1)
        getitem_1 = split[0]
        getitem_2 = split[1]
        getitem_3 = split[2]
        getitem_4 = split[3]
        getitem_5 = split[4]
        dropout = torch.nn.functional.dropout(
            getitem_1, p=0.05, training=True, inplace=False
        )
        dropout_1 = torch.nn.functional.dropout(
            getitem_2, p=0.05, training=True, inplace=False
        )
        dropout_2 = torch.nn.functional.dropout(
            getitem_3, p=0.05, training=True, inplace=False
        )
        dropout_3 = torch.nn.functional.dropout(
            getitem_4, p=0.05, training=True, inplace=False
        )
        dropout_4 = torch.nn.functional.dropout(
            getitem_5, p=0.05, training=True, inplace=False
        )
        return (dropout, dropout_1, dropout_2, dropout_3, dropout_4)


class TestGroupBatchFusion(TestCase):
    # [CLASS DOC] Tests for group and batch fusion passes driven through torch.compile. Most
    # tests set the fusion option config, clear the dynamo counters, compile a fixture module,
    # compare results, parameters and gradients with eager, and assert how many times each pass
    # fired. Helper comparison methods are defined first.
    def compare_dict_tensors(self, ref_dict, res_dict, rtol=1e-3, atol=1e-3):
        # [FUNCTION DOC] HELPER: Compares two mappings of tensors. Returns False if sizes differ
        # or any tensor pair is not close within the tolerances. The compiled side's keys must
        # carry the compiled-module prefix on each eager key; raises AssertionError if one is
        # missing. Returns True when all match.
        if len(set(ref_dict.keys())) != len(set(res_dict.keys())):
            return False
        for key1 in ref_dict:
            key2 = "_orig_mod." + key1
            if key2 not in res_dict:
                raise AssertionError(f"{key1} does not exist in traced module")
            if not torch.allclose(ref_dict[key1], res_dict[key2], rtol=rtol, atol=atol):
                return False
        return True

    def compare_pred(self, module, traced, input, rtol=1e-3, atol=1e-3):
        # [FUNCTION DOC] HELPER: Runs both the eager and compiled module on the inputs and
        # asserts the outputs are equal within the tolerances.
        ref = module(*input)
        res = traced(*input)
        self.assertEqual(ref, res, rtol=rtol, atol=atol)

    def compare_parameters(self, module, traced, rtol=1e-3, atol=1e-3):
        # [FUNCTION DOC] HELPER: Gathers named parameters of both modules and asserts they match
        # through compare_dict_tensors.
        ref_params = dict(module.named_parameters())
        res_params = dict(traced.named_parameters())
        self.assertTrue(self.compare_dict_tensors(ref_params, res_params, rtol, atol))

    def compare_gradients(self, module, traced, rtol=1e-3, atol=1e-3):
        # [FUNCTION DOC] HELPER: Gathers the gradient of every named parameter of both modules
        # and asserts they match through compare_dict_tensors.
        ref_grad = {key: param.grad for key, param in module.named_parameters()}
        res_grad = {key: param.grad for key, param in traced.named_parameters()}
        self.assertTrue(
            self.compare_dict_tensors(ref_grad, res_grad, rtol=rtol, atol=atol)
        )

    @requires_gpu()
    @unittest.skipIf(not has_fbgemm, "requires fbgemm")
    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "group_linear": {"require_fbgemm": True},
        },
    )
    def test_group_linear_fusion(self):
        # [FUNCTION DOC] Needs GPU and fbgemm. Compiles the grouped linear fixture with and
        # without bias and checks group_linear fires twice after the forward pass and four times
        # after the backward, with outputs, parameters and gradients matching eager.
        z = 10
        for has_bias in [True, False]:
            counters.clear()
            module = MyModule(z, has_bias).to(GPU_TYPE)
            input = [torch.randn(z, z, device=GPU_TYPE)]
            traced = torch.compile(module)
            ref = module(*input)
            res = traced(*input)
            self.compare_pred(module, traced, input)
            self.assertEqual(
                counters["inductor"]["group_linear"],
                2,
            )
            ref.sum().backward()
            res.sum().backward()
            self.compare_parameters(module, traced)
            self.compare_gradients(module, traced)
            self.assertEqual(
                counters["inductor"]["group_linear"],
                4,
            )
            counters.clear()

    @requires_gpu()
    @unittest.skipIf(not has_fbgemm, "requires fbgemm")
    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "group_linear": {"require_fbgemm": True},
        },
    )
    def test_group_linear_fusion_different_shapes(self):
        # [FUNCTION DOC] Needs GPU and fbgemm. Compiles the fixture with differently shaped
        # linears and asserts group_linear fires once after forward and twice after backward,
        # batch_fusion never fires, and outputs, parameters and gradients match eager.
        counters.clear()
        module = MyModule2().eval().to(GPU_TYPE)
        input = [torch.rand(4, 24, device=GPU_TYPE)]
        traced = torch.compile(module)
        ref = module(*input)
        res = traced(*input)
        self.compare_pred(module, traced, input)
        self.assertEqual(
            counters["inductor"]["group_linear"],
            1,
        )
        self.assertEqual(
            counters["inductor"]["batch_fusion"],
            0,
        )
        ref.sum().backward()
        res.sum().backward()
        self.compare_parameters(module, traced)
        self.compare_gradients(module, traced)
        self.assertEqual(
            counters["inductor"]["group_linear"],
            2,
        )
        counters.clear()

    @requires_gpu()
    @unittest.skipIf(GPU_TYPE == "mps", "welford_reduce is yet not implemented for MPS")
    @torch._inductor.config.patch(
        pre_grad_fusion_options={"batch_layernorm": {}},
        post_grad_fusion_options={},
    )
    def test_batch_layer_norm_fusion(self):
        # [FUNCTION DOC] Needs GPU (skipped on MPS). Runs the layer norm fixture over all weight
        # and bias presence combinations with batch layernorm enabled and expects two firings
        # per case, with outputs, parameters and gradients matching eager at tight tolerance.
        for has_weight in [True, False]:
            for has_bias in [True, False]:
                counters.clear()
                module = MyModule3(GPU_TYPE, has_weight, has_bias).to(GPU_TYPE)
                input = [torch.randn(2, 5, 50, device=GPU_TYPE)]
                traced = torch.compile(module)
                ref = module(*input)
                res = traced(*input)
                self.compare_pred(module, traced, input)
                self.assertEqual(counters["inductor"]["batch_layernorm"], 2)
                ref.sum().backward()
                res.sum().backward()
                self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
                self.compare_gradients(module, traced, rtol=1e-8, atol=1e-8)
                counters.clear()

    @requires_gpu()
    @torch._inductor.config.patch(
        pre_grad_fusion_options={"batch_linear_lhs": {}},
        post_grad_fusion_options={},
    )
    def test_batch_linear_lhs_fusion(self):
        # [FUNCTION DOC] Needs GPU. With batch_linear_lhs enabled, compiles the lhs fixture with
        # and without bias and expects two firings, with matching outputs, parameters and
        # gradients.
        z = 10
        for has_bias in [True, False]:
            counters.clear()
            module = MyModule4(z, GPU_TYPE, has_bias)
            input = [torch.randn(20, z, device=GPU_TYPE)]
            traced = torch.compile(module)
            ref = module(*input)
            res = traced(*input)
            self.compare_pred(module, traced, input)
            self.assertEqual(counters["inductor"]["batch_linear_lhs"], 2)
            ref.sum().backward()
            res.sum().backward()
            self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
            self.compare_gradients(module, traced, rtol=1e-8, atol=1e-8)
            counters.clear()

    @requires_gpu()
    def test_as_strided_storage_offset_after_mm_fusion(self):
        # [FUNCTION DOC] Needs GPU. Regression test for storage offsets after the post-grad
        # batched matmul rewrite: builds a custom backend that fuses the first three matched mm
        # nodes with the batch linear rule, compiles a small QKV attention model through it and
        # checks that exactly three nodes were fused and that outputs match eager. Resets dynamo
        # state before each run.
        """
        Post-grad batch linear fusion rewrites parallel mm nodes into a
        batched bmm followed by select views. The select outputs preserve the
        original row stride, but they inherit a non-zero storage offset.
        Downstream as_strided must inherit that offset instead of resetting to
        the base storage offset.
        """
        import copy

        from torch._dynamo.backends.common import aot_autograd
        from torch._inductor.compile_fx import compile_fx_inner
        from torch._inductor.decomposition import select_decomp_table
        from torch._inductor.fx_passes.group_batch_fusion import (
            graph_search_options,
            PostGradBatchLinearFusion,
        )
        from torch._inductor.pattern_matcher import stable_topological_sort

        fused_counts = []

        def fusing_compiler(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Backend compiler. Copies the default graph search options
            # with a minimum fuse set size of three, finds nodes matched by the post-grad batch
            # linear rule, records how many matched, and if at least three matched fuses the
            # first three, re-sorts topologically, lints and recompiles the graph. Then compiles
            # with the standard inductor inner compiler.
            opts = graph_search_options.copy()
            opts["min_fuse_set_size"] = 3
            rule = PostGradBatchLinearFusion(graph_search_options=opts)
            mm_nodes = [n for n in gm.graph.nodes if rule.match(n) is not None]
            fused_counts.append(len(mm_nodes))
            if len(mm_nodes) >= 3:
                rule.fuse(gm.graph, mm_nodes[:3])
                stable_topological_sort(gm.graph)
                gm.graph.lint()
                gm.recompile()
            return compile_fx_inner(gm, example_inputs)

        fusing_backend = aot_autograd(
            fw_compiler=fusing_compiler,
            decompositions=select_decomp_table(),
        )

        class QKVModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Small QKV projection model whose forward reshapes and
            # re-strides q, k, v so that downstream strides depend on the storage offsets
            # produced by the fused matmuls.
            def __init__(self, hidden_size=64, num_heads=4):
                # [FUNCTION DOC] HELPER: Stores hidden size, head count and head dim; creates
                # three bias-free linear layers for q, k, v and the attention scale (square root
                # of head dim).
                super().__init__()
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                self.head_dim = hidden_size // num_heads
                self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
                self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
                self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
                self.scale = self.head_dim**0.5

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Permutes the input, projects it with the three linears,
                # scales q, reshapes each to heads by flattened-tokens by head dim and re-views
                # it with explicit strides, transposes k and v, and returns the batched product
                # of q and k together with k and v.
                # Reduced-size version of the issue author's QKV repro.
                x = x.permute(1, 0, 2)
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)
                seq_len, batch_size, _ = q.shape

                q = q / self.scale
                q = q.view(seq_len, batch_size, self.num_heads, self.head_dim)
                q = q.permute(2, 1, 0, 3)
                q = q.reshape(self.num_heads, batch_size * seq_len, self.head_dim)
                q = q.as_strided(
                    (self.num_heads, batch_size * seq_len, self.head_dim),
                    (self.head_dim, self.num_heads * self.head_dim, 1),
                )

                k = k.view(seq_len, batch_size, self.num_heads, self.head_dim)
                k = k.permute(2, 1, 0, 3)
                k = k.reshape(self.num_heads, batch_size * seq_len, self.head_dim)
                k = k.as_strided(
                    (self.num_heads, batch_size * seq_len, self.head_dim),
                    (self.head_dim, self.num_heads * self.head_dim, 1),
                )
                k = k.permute(0, 2, 1)

                v = v.view(seq_len, batch_size, self.num_heads, self.head_dim)
                v = v.permute(2, 1, 0, 3)
                v = v.reshape(self.num_heads, batch_size * seq_len, self.head_dim)
                v = v.as_strided(
                    (self.num_heads, batch_size * seq_len, self.head_dim),
                    (self.head_dim, self.num_heads * self.head_dim, 1),
                )
                v = v.permute(0, 2, 1)

                return torch.bmm(q, k), k, v

        torch.manual_seed(42)
        model = QKVModel().to(GPU_TYPE).eval()
        x = torch.randn(1, 8, 64, device=GPU_TYPE)

        torch._dynamo.reset()
        with torch.no_grad():
            ref = model(x)

        torch._dynamo.reset()
        compiled = torch.compile(copy.deepcopy(model), backend=fusing_backend)
        with torch.no_grad():
            res = compiled(x)

        self.assertEqual(fused_counts, [3])
        for ref_t, res_t in zip(ref, res):
            self.assertEqual(ref_t, res_t, rtol=1e-3, atol=1e-3)

    @requires_gpu()
    @torch._inductor.config.patch(
        pre_grad_fusion_options={"batch_linear_lhs": {}},
        post_grad_fusion_options={},
    )
    def test_batch_linear_lhs_fusion_nn_linear_inlined(self):
        # [FUNCTION DOC] Needs GPU. Like the lhs fusion test but the linears come from nn.Linear
        # modules that dynamo inlines into the low-level linear call; expects one firing with
        # matching outputs, parameters and gradients.
        # Same shape pattern as test_batch_linear_lhs_fusion, but the linears
        # are produced through nn.Linear modules. Dynamo inlines those to
        # torch._C._nn.linear, which the upstream matcher used to miss.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a list of n registered linear layers of shrinking
            # output size.
            def __init__(self, z, n, has_bias):
                # [FUNCTION DOC] HELPER: Builds a module list of n linears from z inputs,
                # outputs z minus index modulo five, optional bias.
                super().__init__()
                self.linears = torch.nn.ModuleList(
                    [torch.nn.Linear(z, z - i % 5, bias=has_bias) for i in range(n)]
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds a constant to the input, applies every linear,
                # concatenates on dimension one and returns the sigmoid.
                x = x + 1.2
                outs = [lin(x) for lin in self.linears]
                return torch.sigmoid(torch.cat(outs, dim=1))

        z, n = 10, 10
        for has_bias in [True, False]:
            counters.clear()
            module = M(z, n, has_bias).to(GPU_TYPE)
            input = [torch.randn(20, z, device=GPU_TYPE)]
            traced = torch.compile(module)
            ref = module(*input)
            res = traced(*input)
            self.compare_pred(module, traced, input)
            self.assertEqual(counters["inductor"]["batch_linear_lhs"], 1)
            ref.sum().backward()
            res.sum().backward()
            self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
            self.compare_gradients(module, traced, rtol=1e-8, atol=1e-8)
            counters.clear()

    @requires_gpu()
    @torch._inductor.config.patch(
        pre_grad_fusion_options={"batch_linear": {}},
        post_grad_fusion_options={},
    )
    def test_batch_linear_pre_grad_fusion(self):
        # [FUNCTION DOC] Needs GPU. With batch_linear enabled in the pre-grad options, compiles
        # the five-linear fixture with and without bias and expects one firing; outputs,
        # parameters and gradients must match eager.
        for has_bias in [True, False]:
            counters.clear()
            module = MyModule5(GPU_TYPE, has_bias)
            input = [torch.randn(50, 500, device=GPU_TYPE)]
            traced = torch.compile(module)
            ref = module(*input)
            res = traced(*input)
            self.compare_pred(module, traced, input)
            self.assertEqual(counters["inductor"]["batch_linear"], 1)
            ref.sum().backward()
            res.sum().backward()
            self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
            self.compare_gradients(module, traced, rtol=1e-8, atol=1e-8)
            counters.clear()

    @requires_gpu()
    @torch._inductor.config.patch(
        pre_grad_fusion_options={
            "batch_relu": {},
            "batch_sigmoid": {},
        },
        post_grad_fusion_options={
            "batch_aten_add": {},
            "batch_aten_mul": {},
            "batch_aten_sub": {},
            "batch_aten_div": {},
        },
    )
    def test_pointwise_op_fusion(self):
        # [FUNCTION DOC] Needs GPU. Enables pre-grad relu and sigmoid batching and post-grad
        # add, mul, sub and div batching, compiles the pointwise fixture and expects one firing
        # for each, with outputs, parameters and gradients matching eager.
        counters.clear()
        module = _TestPointwiseOps(GPU_TYPE)
        input = [torch.randn(50, 1000, requires_grad=True, device=GPU_TYPE)]
        traced = torch.compile(module)
        ref = module(*input)
        res = traced(*input)
        self.compare_pred(module, traced, input)
        self.assertEqual(counters["inductor"]["batch_relu"], 1)
        self.assertEqual(counters["inductor"]["batch_sigmoid"], 1)
        self.assertEqual(counters["inductor"]["batch_aten_add"], 1)
        self.assertEqual(counters["inductor"]["batch_aten_mul"], 1)
        self.assertEqual(counters["inductor"]["batch_aten_sub"], 1)
        self.assertEqual(counters["inductor"]["batch_aten_div"], 1)
        ref.sum().backward()
        res.sum().backward()
        self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
        self.compare_gradients(module, traced, rtol=1e-8, atol=1e-8)
        counters.clear()

    @requires_gpu()
    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "batch_aten_relu": {},
            "batch_aten_sigmoid": {},
            "batch_aten_tanh": {},
            "unbind_stack_aten_pass": {},
        },
    )
    def test_pointwise_op_fusion_post_grad(self):
        # [FUNCTION DOC] Needs GPU. Enables post-grad relu, sigmoid, tanh and unbind-stack
        # batching, compiles the aten pointwise fixture and expects one firing each for tanh,
        # relu and sigmoid and two for the unbind stack pass; results and gradients must match
        # eager.
        counters.clear()
        module = _TestPointwiseOpsPostGrad(GPU_TYPE)
        input = [torch.randn(50, 1000, requires_grad=True, device=GPU_TYPE)]
        traced = torch.compile(module)
        ref = module(*input)
        res = traced(*input)
        self.compare_pred(module, traced, input)
        self.assertEqual(counters["inductor"]["batch_aten_tanh"], 1)
        self.assertEqual(counters["inductor"]["batch_aten_relu"], 1)
        self.assertEqual(counters["inductor"]["batch_aten_sigmoid"], 1)
        self.assertEqual(counters["inductor"]["unbind_stack_aten_pass"], 2)
        ref.sum().backward()
        res.sum().backward()
        self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
        self.compare_gradients(module, traced, rtol=1e-8, atol=1e-8)
        counters.clear()

    @requires_gpu()
    @torch._dynamo.config.patch(canonicalize_output_graph_node_order=False)
    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "batch_linear_post_grad": {
                "shape_broadcast_batch_linear": True,
                "fuse_nodes_with_same_users": True,
            },
            "batch_aten_mul": {"fuse_nodes_with_same_parent": False},
            "batch_aten_sigmoid": {"fuse_nodes_with_same_parent": True},
            "batch_aten_add": {"fuse_nodes_with_same_parent": True},
            "normalization_aten_pass": {},
            "unbind_stack_aten_pass": {},
        },
    )
    def test_gate_fusion_post_grad(self):
        # [FUNCTION DOC] Needs GPU, with canonical output node ordering disabled in dynamo
        # config. Compiles the highway gating fixture on 20 inputs with several post-grad
        # batching passes enabled and asserts the exact firing counts (linear 2, sigmoid 1, mul
        # 1, add 2, normalization 1, unbind stack 5), then that outputs, parameters and
        # gradients match eager.
        counters.clear()
        size = 20
        module = _TestHighwaySelfGating(d_model=10, size=size, device=GPU_TYPE)
        input = [
            [
                torch.randn(10, 10, requires_grad=True, device=GPU_TYPE)
                for i in range(size)
            ]
        ]
        traced = torch.compile(module)
        ref = module(*input)
        res = traced(*input)
        self.compare_pred(module, traced, input)
        self.assertEqual(counters["inductor"]["batch_linear_post_grad"], 2)
        self.assertEqual(counters["inductor"]["batch_aten_sigmoid"], 1)
        self.assertEqual(counters["inductor"]["batch_aten_mul"], 1)
        self.assertEqual(counters["inductor"]["batch_aten_add"], 2)
        self.assertEqual(counters["inductor"]["normalization_aten_pass"], 1)
        self.assertEqual(counters["inductor"]["unbind_stack_aten_pass"], 5)
        ref.sum().backward()
        res.sum().backward()
        self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
        self.compare_gradients(module, traced, rtol=1e-8, atol=1e-8)
        counters.clear()

    @requires_gpu()
    @torch._inductor.config.patch(
        pre_grad_fusion_options={
            "normalization_pass": {},
            "batch_detach": {},
            "batch_nan_to_num": {},
            "batch_clamp": {},
            "unbind_stack_pass": {},
            "unbind_stack_to_slices_pass": {},
        },
        post_grad_fusion_options={},
    )
    def test_math_op_fusion(self):
        # [FUNCTION DOC] Needs GPU. Enables normalization, detach, nan_to_num, clamp and
        # unbind-stack passes, compiles the math op fixture on a vector with NaN and infinities,
        # asserts the per-pass firing counts and that eager and compiled results are close.
        counters.clear()
        module = _TestMathOps(GPU_TYPE)
        input = [
            torch.tensor(
                [float("nan"), float("inf"), -float("inf"), 3.14], device=GPU_TYPE
            )
        ]
        traced = torch.compile(module)
        ref = module(*input)
        res = traced(*input)
        self.compare_pred(module, traced, input)
        self.assertEqual(counters["inductor"]["normalization_pass"], 3)
        self.assertEqual(counters["inductor"]["batch_clamp"], 1)
        self.assertEqual(counters["inductor"]["batch_detach"], 1)
        self.assertEqual(counters["inductor"]["batch_nan_to_num"], 1)
        self.assertEqual(counters["inductor"]["unbind_stack_to_slices_pass"], 2)
        self.assertEqual(counters["inductor"]["unbind_stack_pass"], 2)
        self.assertTrue(torch.allclose(ref, res))
        counters.clear()

    @config.patch(
        is_predispatch=True,
        pre_grad_fusion_options={"batch_clamp": {}},
    )
    def test_math_op_fusion_predispatch_positional_args(self):
        # [FUNCTION DOC] Config enables the predispatch path and batch clamp. Compiles a
        # function stacking clamps whose bounds are passed positionally, in two groups with
        # different bounds, with full graph capture, and expects results equal to eager and two
        # batch_clamp firings.
        counters.clear()

        def fn(x):
            # [FUNCTION DOC] HELPER: Stacks ten clamped shifted copies of the input; the first
            # five use one pair of positional bounds and the last five another.
            return torch.stack(
                [torch.clamp(x + i, -1.0, 1.0) for i in range(5)]
                + [torch.clamp(x + i, 0.0, 2.0) for i in range(5, 10)]
            )

        x = torch.randn(8)
        self.assertEqual(fn(x), torch.compile(fn, fullgraph=True)(x))
        self.assertEqual(counters["inductor"]["batch_clamp"], 2)
        counters.clear()

    @requires_gpu()
    @torch._inductor.config.patch(
        pre_grad_fusion_options={
            "normalization_pass": {},
            "batch_dropout": {},
        }
    )
    def test_batch_dropout_pre_grad_fusion(self):
        # [FUNCTION DOC] Needs GPU. Enables normalization and batch dropout, runs eager and
        # compiled dropout fixtures on one input and asserts each pass fired once. Outputs are
        # not compared because dropout is random.
        counters.clear()
        module = _TestDropout(GPU_TYPE)
        input = [torch.randn(10, 100, requires_grad=True, device=GPU_TYPE)]
        traced = torch.compile(module)
        module(*input)
        traced(*input)
        self.assertEqual(counters["inductor"]["normalization_pass"], 1)
        self.assertEqual(counters["inductor"]["batch_dropout"], 1)
        counters.clear()

    @unittest.skipUnless(torch.xpu.is_available(), "batch_linear_lhs is XPU-only")
    @torch._inductor.config.patch(
        pre_grad_fusion_options={
            "batch_linear_lhs": {"devices": ("xpu",), "min_fuse_set_size": 2},
        },
    )
    def test_xpu_batch_linear_lhs(self):
        # [FUNCTION DOC] Runs only when XPU is available, with batch_linear_lhs restricted to
        # xpu devices. Asserts the default option restricts devices to xpu, compiles the lhs
        # fixture on xpu with and without bias, expects at least one firing, matching results,
        # parameters and gradients, and that the global pre-grad option mapping is not mutated
        # by compilation.
        # batch_linear_lhs is disabled by default; enabling it for XPU via mock
        # config must make the fusion fire on XPU tensors.
        default_options = config.pre_grad_fusion_options
        self.assertIn("batch_linear_lhs", default_options)
        self.assertEqual(default_options["batch_linear_lhs"]["devices"], ("xpu",))
        z = 10
        for has_bias in [True, False]:
            orig_fusion_options = dict(config.pre_grad_fusion_options)
            counters.clear()
            module = MyModule4(z, "xpu", has_bias)
            input = [torch.randn(20, z, device="xpu")]
            traced = torch.compile(module)
            ref = module(*input)
            res = traced(*input)
            self.compare_pred(module, traced, input)
            self.assertGreater(counters["inductor"]["batch_linear_lhs"], 0)
            self.assertEqual(ref, res)
            ref.sum().backward()
            res.sum().backward()
            self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
            self.compare_gradients(module, traced, rtol=1e-8, atol=1e-8)
            self.assertEqual(
                orig_fusion_options,
                dict(config.pre_grad_fusion_options),
                "config.pre_grad_fusion_options should not be mutated",
            )
            counters.clear()

    @requires_gpu()
    @torch._inductor.config.patch(
        is_predispatch=True,
        pre_grad_fusion_options={
            "batch_linear_lhs": {"devices": (GPU_TYPE,), "min_fuse_set_size": 2},
        },
    )
    def test_predispatch_device_aware_batch_linear_lhs(self):
        # [FUNCTION DOC] Needs GPU. With the predispatch path enabled and devices restricted to
        # the current GPU type, compiles the lhs fixture with full graph capture and asserts the
        # pass fires at least once and results, parameters and gradients match eager. Verifies
        # option resolution and device filtering in the predispatch path.
        # Verify that the predispatch path (_run_pre_dispatch_passes) routes
        # through _resolve_pre_grad_fusion_options and applies device filtering.
        # The default config has devices=("xpu",) which was previously bypassed
        # in the predispatch path (review #181854#issuecomment-4705071437).
        # This test uses devices=(GPU_TYPE,) to confirm the wrapper closure
        # correctly resolves and filters fusion options.
        z = 10
        module = MyModule4(z, GPU_TYPE, has_bias=True)
        input = [torch.randn(20, z, device=GPU_TYPE)]

        counters.clear()
        traced = torch.compile(module, fullgraph=True)
        ref = module(*input)
        res = traced(*input)
        self.compare_pred(module, traced, input)
        self.assertGreater(
            counters["inductor"]["batch_linear_lhs"],
            0,
            "batch_linear_lhs should fire in predispatch mode when devices match GPU_TYPE",
        )
        self.assertEqual(ref, res)
        ref.sum().backward()
        res.sum().backward()
        self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
        self.compare_gradients(module, traced, rtol=1e-8, atol=1e-8)
        counters.clear()

    @torch._inductor.config.patch(
        is_predispatch=True,
        pre_grad_fusion_options={
            "batch_linear_lhs": {"min_fuse_set_size": 2},
        },
    )
    def test_predispatch_cpu_device_agnostic_batch_linear_lhs(self):
        # [FUNCTION DOC] Runs on CPU. With the predispatch path and batch_linear_lhs enabled
        # without any devices restriction, compiles the lhs fixture with full graph capture and
        # asserts the pass fires at least once and results, parameters and gradients match
        # eager.
        # Verify that the predispatch path correctly enables batch_linear_lhs
        # when no "devices" key is present (user explicitly opts in).
        # This test runs on CPU so it can be verified in local dev without GPU.
        z = 10
        module = MyModule4(z, "cpu", has_bias=True)
        input = [torch.randn(20, z, device="cpu")]

        counters.clear()
        traced = torch.compile(module, fullgraph=True)
        ref = module(*input)
        res = traced(*input)
        self.compare_pred(module, traced, input)
        self.assertGreater(
            counters["inductor"]["batch_linear_lhs"],
            0,
            "batch_linear_lhs should fire in predispatch mode when no devices key restricts it",
        )
        self.assertEqual(ref, res)
        ref.sum().backward()
        res.sum().backward()
        self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
        self.compare_gradients(module, traced, rtol=1e-8, atol=1e-8)
        counters.clear()

    def test_batch_linear_lhs_skips_tensor_subclass_weights(self):
        # [FUNCTION DOC] Builds a tiny FX graph with a linear call whose weight example value is
        # a tensor subclass that raises if concatenation is dispatched. Asserts the match method
        # of the batch linear lhs rule returns None so that the weight subclass is never
        # concatenated. Only the matcher is exercised; no compilation happens.
        """batch_linear_lhs match() must return None when weight is a tensor subclass.

        Regression test for incompatibility with torchao W8A8 quantization:
        fuse() calls torch.cat on the weight example_values, which raises
        NotImplementedError for subclasses that don't implement aten.cat
        (e.g. torchao Int8Tensor). The fix guards match() to skip subclasses.
        See: https://github.com/pytorch/ao/pull/4560
        """
        from torch._inductor.fx_passes.group_batch_fusion import BatchLinearLHSFusion

        class _SubclassWeight(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass whose dispatch hook raises AssertionError when
            # the default aten concatenation is called and otherwise forwards the call
            # unchanged.
            """Minimal subclass that raises on aten.cat to detect the bug."""

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Dispatch hook; raises AssertionError on the aten cat
                # op, otherwise runs the op on the given arguments, with empty keyword arguments
                # if none were given.
                if func == torch.ops.aten.cat.default:
                    raise AssertionError(
                        "aten.cat called on _SubclassWeight — "
                        "batch_linear_lhs should have been skipped"
                    )
                return func(*args, **(kwargs or {}))

        fusion = BatchLinearLHSFusion()
        z = 4

        # Build a minimal FX graph: F.linear(x, w) with a subclass weight.
        graph = torch.fx.Graph()
        x_node = graph.placeholder("x")
        w_node = graph.placeholder("w")
        x_node.meta["example_value"] = torch.randn(2, z)
        w_node.meta["example_value"] = torch.randn(z, z).as_subclass(_SubclassWeight)
        linear_node = graph.call_function(
            torch.nn.functional.linear, args=(x_node, w_node)
        )
        linear_node.meta["example_value"] = torch.randn(2, z)
        graph.output(linear_node)

        # match() must return None (skip) for a subclass weight.
        result = fusion.match(linear_node)
        self.assertIsNone(
            result,
            "batch_linear_lhs should skip (return None) when weight is a tensor subclass",
        )


class _TestBMMFusionModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Fixture holding ten 10 by 10 linear layers; forward sums their
    # outputs.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates a module list of ten 10 by 10 linear layers.
        super().__init__()
        self.my_modules = torch.nn.ModuleList()
        for _ in range(10):
            self.my_modules.append(torch.nn.Linear(10, 10))

    def forward(self, inputs):
        # [FUNCTION DOC] HELPER: Pairs each linear layer with an input, and returns the sum of
        # all outputs (the first one seeds the accumulator, the rest are added in place).
        output = None
        for linear, input in zip(self.my_modules, inputs):
            if output is None:
                output = linear(input)
            else:
                output += linear(input)
        return output


@requires_gpu()
@torch._inductor.config.patch(
    post_grad_fusion_options={"batch_linear_post_grad": {"require_fbgemm": False}}
)
class TestPostGradBatchLinearFusion(TestCase):
    # [CLASS DOC] GPU-only test class (class decorators require a GPU and set post-grad batch
    # linear fusion without fbgemm) checking the post-grad batch linear pass on ten summed
    # linears.
    def test_batch_linear_post_grad_fusion(self):
        # [FUNCTION DOC] Compiles the BMM fixture on ten random inputs, asserts the result is
        # close to eager and that the batch_linear_post_grad counter equals two. The counters
        # are not cleared first.
        pt1_module = _TestBMMFusionModule().to(GPU_TYPE)
        inputs = []
        for _ in range(10):
            inputs.append(torch.randn(10, 10).to(GPU_TYPE))
        eager_output = pt1_module(inputs)
        pt2_module = torch.compile(pt1_module)
        pt2_output = pt2_module(inputs)
        self.assertTrue(torch.allclose(eager_output, pt2_output))
        self.assertEqual(
            counters["inductor"]["batch_linear_post_grad"],
            2,
        )


class TestFindIndependentSubsetGreedy(TestCase):
    # [CLASS DOC] Tests for the greedy independent-subset iterator used by fusion. Graphs are
    # described by name-to-argument-names mappings and built as FX placeholder nodes.
    # Helper function to build a Graph from a data description.
    def build_graph(self, desc):
        # [FUNCTION DOC] HELPER: Builds an FX graph from a mapping of node name to the names of
        # its inputs. Nodes are created once all their inputs exist; others are deferred to the
        # back of a queue. Raises AssertionError if a full pass over the queue makes no progress
        # (cycle or unknown name). Returns the graph and a mapping from name to node.
        # desc: {
        #   "n1": ["n2", "n3"],
        #   "n2": ["n3"],
        #   "n3": [],
        # }
        #
        g = torch.fx.Graph()
        lookup = {}
        desc = collections.deque((k, v) for k, v in desc.items())
        unsatisfied = 0
        while desc:
            unsatisfied += 1
            if unsatisfied > len(desc):
                raise AssertionError("cycle or bad input?")
            name, v = desc.popleft()
            args = tuple(lookup.get(n) for n in v)
            if None in args:
                desc.append((name, v))
                continue
            node = g.create_node("placeholder", "target", name=name, args=args)
            lookup[name] = node
            unsatisfied = 0
        return g, lookup

    def verify(self, tree, subnodes, min_fuse, max_fuse, expected):
        # [FUNCTION DOC] HELPER: Builds the graph, maps names to nodes, runs the greedy subset
        # iterator with min and max fuse set sizes and asserts the produced groups equal the
        # expected groups of nodes.
        _, lookup = self.build_graph(tree)
        subnodes = [lookup[n] for n in subnodes]
        expected = [[lookup[n] for n in sub] for sub in expected]
        opts = {
            "min_fuse_set_size": min_fuse,
            "max_fuse_set_size": max_fuse,
        }
        result = list(
            torch._inductor.fx_passes.group_batch_fusion.find_independent_subset_greedy(
                subnodes, opts
            )
        )
        self.assertEqual(expected, result)

    def test_find_independent_subset_greedy(self):
        # [FUNCTION DOC] Table-driven check of the independent subset iterator with many graphs
        # (some randomly generated) and various min and max set sizes; each case lists the
        # expected ordered groups. Only calls verify.
        # First some randomly generated tests.
        self.verify({"n0": (), "n1": ()}, ["n0"], 0, 100, [["n0"]])
        self.verify(
            {"n0": (), "n1": (), "n2": ("n0",)}, ["n1", "n2"], 0, 100, [["n1", "n2"]]
        )
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": ("n0",),
                "n3": (),
                "n4": ("n0", "n1", "n2"),
                "n5": ("n0", "n2", "n4"),
                "n6": ("n3",),
                "n7": ("n4", "n5", "n6", "n1", "n3"),
                "n8": ("n7", "n1", "n3", "n5", "n0"),
                "n9": ("n3", "n4", "n8", "n6", "n5", "n2", "n0", "n7"),
                "n10": ("n0",),
                "n11": ("n4", "n0", "n2", "n3", "n1", "n9"),
                "n12": ("n2", "n3", "n10", "n6", "n9"),
            },
            ["n10", "n5", "n3", "n4", "n9"],
            0,
            100,
            [["n10", "n5", "n3"], ["n4"], ["n9"]],
        )
        self.verify({"n0": (), "n1": (), "n2": ("n0",)}, ["n2"], 0, 100, [["n2"]])
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": (),
                "n3": (),
                "n4": ("n3", "n1", "n0"),
                "n5": ("n1", "n2", "n4", "n0"),
                "n6": ("n0", "n3", "n2"),
                "n7": ("n6", "n1", "n5", "n4", "n3", "n0"),
                "n8": ("n2", "n7", "n3"),
                "n9": ("n3", "n5", "n6", "n7", "n2", "n1"),
                "n10": ("n8", "n0", "n2", "n4", "n6", "n3"),
                "n11": ("n6", "n5", "n8", "n1", "n3", "n10", "n2"),
                "n12": ("n7", "n4"),
            },
            ["n7"],
            0,
            100,
            [["n7"]],
        )
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": (),
                "n3": ("n1", "n2"),
                "n4": ("n1",),
                "n5": (),
                "n6": ("n5",),
                "n7": ("n1", "n6", "n5", "n2", "n3", "n0"),
                "n8": ("n5", "n7", "n2", "n6"),
                "n9": ("n1",),
                "n10": ("n9",),
                "n11": ("n3", "n4", "n0", "n2"),
                "n12": ("n8", "n9", "n5", "n1"),
                "n13": ("n11", "n4", "n12", "n1", "n9", "n3", "n0"),
            },
            ["n9", "n2", "n8", "n10", "n5", "n6", "n13", "n7", "n3", "n0", "n4"],
            0,
            100,
            [
                ["n9", "n2", "n5", "n0", "n4"],
                ["n8", "n10"],
                ["n6", "n3"],
                ["n13"],
                ["n7"],
            ],
        )
        self.verify({"n0": ()}, ["n0"], 0, 100, [["n0"]])
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": (),
                "n3": (),
                "n4": ("n1", "n2"),
                "n5": ("n0", "n4", "n1"),
                "n6": ("n1", "n5"),
                "n7": (),
                "n8": ("n7", "n1", "n3", "n5", "n6"),
                "n9": ("n2", "n1", "n8", "n0", "n4", "n7", "n6", "n5"),
                "n10": ("n4", "n7", "n2", "n3", "n8"),
                "n11": (),
                "n12": ("n9", "n7", "n5", "n11", "n8"),
                "n13": (
                    "n5",
                    "n6",
                    "n12",
                    "n3",
                    "n9",
                    "n8",
                    "n4",
                    "n11",
                    "n2",
                    "n10",
                    "n1",
                ),
                "n14": ("n7", "n3", "n12", "n10", "n2", "n0", "n4", "n5"),
                "n15": ("n9", "n5", "n1", "n13", "n8", "n10", "n12", "n7", "n11", "n3"),
                "n16": (
                    "n2",
                    "n4",
                    "n15",
                    "n5",
                    "n0",
                    "n6",
                    "n3",
                    "n8",
                    "n14",
                    "n12",
                    "n9",
                    "n10",
                    "n7",
                    "n13",
                ),
            },
            ["n0", "n3", "n2", "n11", "n1", "n6", "n12", "n5", "n4", "n15", "n8"],
            0,
            100,
            [
                ["n0", "n3", "n2", "n11", "n1"],
                ["n6"],
                ["n12"],
                ["n5"],
                ["n4"],
                ["n15"],
                ["n8"],
            ],
        )
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": (),
                "n3": ("n2", "n1"),
                "n4": ("n2", "n3", "n1"),
                "n5": ("n3", "n1"),
                "n6": ("n1",),
                "n7": ("n5", "n4"),
                "n8": ("n6", "n2"),
            },
            ["n4", "n3", "n1", "n8", "n5", "n6", "n2"],
            0,
            100,
            [["n4", "n8", "n5"], ["n3", "n6"], ["n1", "n2"]],
        )
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": (),
                "n3": ("n1", "n0"),
                "n4": ("n0",),
                "n5": ("n1", "n4"),
                "n6": ("n2", "n1", "n4"),
                "n7": ("n0", "n3"),
                "n8": ("n5", "n0", "n6", "n1", "n4", "n2", "n3"),
                "n9": ("n1", "n4", "n8", "n7", "n5"),
                "n10": ("n9", "n8", "n0", "n2", "n7", "n1", "n3", "n5"),
                "n11": ("n9", "n2", "n6", "n0", "n3"),
                "n12": ("n1", "n4", "n7", "n10", "n5", "n2", "n11", "n6"),
                "n13": ("n9", "n2", "n3", "n0", "n7", "n5", "n10", "n11"),
                "n14": (
                    "n8",
                    "n0",
                    "n3",
                    "n6",
                    "n10",
                    "n1",
                    "n5",
                    "n9",
                    "n12",
                    "n11",
                    "n4",
                ),
                "n15": (
                    "n3",
                    "n10",
                    "n0",
                    "n4",
                    "n9",
                    "n11",
                    "n2",
                    "n13",
                    "n12",
                    "n8",
                    "n5",
                    "n14",
                ),
                "n16": ("n6",),
                "n17": (
                    "n4",
                    "n3",
                    "n14",
                    "n8",
                    "n15",
                    "n16",
                    "n2",
                    "n5",
                    "n7",
                    "n12",
                    "n1",
                    "n0",
                    "n11",
                ),
            },
            ["n17", "n16", "n10", "n4", "n8", "n12", "n6", "n1"],
            0,
            100,
            [["n17"], ["n16", "n10"], ["n4", "n1"], ["n8"], ["n12"], ["n6"]],
        )
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": ("n0",),
                "n3": ("n0", "n1"),
                "n4": ("n0",),
                "n5": ("n0",),
                "n6": ("n5", "n3", "n0", "n2"),
                "n7": (),
                "n8": ("n2", "n5", "n3", "n1", "n7", "n6", "n0"),
                "n9": ("n4",),
                "n10": ("n4", "n5", "n1", "n2", "n0", "n6", "n8", "n9", "n7"),
                "n11": ("n3", "n0", "n9", "n10", "n5", "n1", "n2", "n7", "n4", "n6"),
                "n12": ("n9", "n5"),
            },
            ["n8", "n3", "n1", "n12", "n2", "n5", "n11", "n4", "n10", "n6", "n0"],
            0,
            100,
            [
                ["n8", "n12"],
                ["n3", "n2", "n5", "n4"],
                ["n1", "n0"],
                ["n11"],
                ["n10"],
                ["n6"],
            ],
        )
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": (),
                "n3": (),
                "n4": ("n2", "n3"),
                "n5": ("n1", "n3", "n2", "n4"),
                "n6": ("n5", "n4", "n1", "n3"),
                "n7": ("n5",),
                "n8": ("n5", "n4", "n1"),
                "n9": ("n2", "n3", "n1", "n5", "n7", "n0", "n8"),
                "n10": ("n5", "n3", "n1", "n7", "n8", "n9"),
                "n11": ("n1", "n4", "n2", "n0", "n8", "n9"),
                "n12": ("n4", "n3", "n9"),
                "n13": (
                    "n6",
                    "n10",
                    "n4",
                    "n8",
                    "n0",
                    "n11",
                    "n12",
                    "n7",
                    "n3",
                    "n2",
                    "n1",
                ),
                "n14": ("n4", "n13", "n2"),
                "n15": ("n11", "n7", "n6", "n10", "n14"),
                "n16": ("n15", "n3"),
                "n17": ("n10", "n2", "n7", "n0", "n5", "n6", "n9"),
                "n18": (
                    "n16",
                    "n8",
                    "n6",
                    "n9",
                    "n11",
                    "n12",
                    "n14",
                    "n5",
                    "n13",
                    "n4",
                    "n1",
                ),
            },
            [
                "n1",
                "n0",
                "n16",
                "n6",
                "n15",
                "n9",
                "n7",
                "n4",
                "n3",
                "n11",
                "n13",
                "n17",
                "n12",
                "n18",
            ],
            0,
            100,
            [
                ["n1", "n0", "n4"],
                ["n16", "n17"],
                ["n6", "n9"],
                ["n15"],
                ["n7"],
                ["n3"],
                ["n11", "n12"],
                ["n13"],
                ["n18"],
            ],
        )
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": (),
                "n3": ("n2",),
                "n4": ("n1",),
                "n5": (),
                "n6": ("n1", "n4"),
                "n7": ("n5", "n1"),
                "n8": ("n6",),
                "n9": ("n6", "n1", "n2", "n0"),
                "n10": ("n0", "n7"),
                "n11": ("n0", "n4", "n3", "n5"),
                "n12": ("n9", "n8", "n7", "n4", "n0"),
            },
            ["n8", "n9", "n11", "n2", "n4", "n0", "n7", "n5", "n1"],
            0,
            100,
            [["n8", "n9", "n11", "n7"], ["n2", "n4", "n0", "n5"], ["n1"]],
        )
        self.verify(
            {"n0": (), "n1": (), "n2": (), "n3": ("n0",), "n4": ("n3",)},
            ["n1", "n2", "n4"],
            0,
            100,
            [["n1", "n2", "n4"]],
        )
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": ("n1",),
                "n3": ("n2", "n1"),
                "n4": ("n3",),
                "n5": (),
                "n6": ("n1", "n5"),
                "n7": (),
                "n8": ("n4", "n5"),
                "n9": ("n0", "n3", "n6", "n4", "n5", "n8", "n7", "n1"),
                "n10": ("n3", "n0", "n6", "n9", "n7"),
                "n11": (),
                "n12": ("n1", "n8", "n3", "n6", "n7", "n0", "n10", "n5", "n9", "n11"),
                "n13": ("n9", "n11", "n4"),
                "n14": (),
                "n15": ("n6", "n12"),
                "n16": (
                    "n1",
                    "n7",
                    "n10",
                    "n3",
                    "n9",
                    "n0",
                    "n2",
                    "n5",
                    "n8",
                    "n13",
                    "n14",
                    "n15",
                    "n4",
                    "n6",
                ),
            },
            [
                "n11",
                "n16",
                "n5",
                "n12",
                "n7",
                "n2",
                "n0",
                "n6",
                "n3",
                "n9",
                "n8",
                "n15",
                "n14",
                "n4",
                "n13",
                "n1",
            ],
            0,
            100,
            [
                ["n11", "n5", "n7", "n2", "n0", "n14"],
                ["n16"],
                ["n12", "n13"],
                ["n6", "n3"],
                ["n9"],
                ["n8"],
                ["n15"],
                ["n4"],
                ["n1"],
            ],
        )
        self.verify({"n0": (), "n1": ()}, ["n1"], 0, 100, [["n1"]])
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": ("n1",),
                "n3": (),
                "n4": ("n0", "n2", "n3"),
                "n5": ("n2", "n3"),
                "n6": ("n3",),
            },
            ["n6", "n2", "n3", "n1"],
            0,
            100,
            [["n6", "n2"], ["n3", "n1"]],
        )
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": (),
                "n3": ("n2",),
                "n4": ("n0",),
                "n5": ("n1", "n2"),
                "n6": ("n2", "n3", "n1", "n0", "n5"),
                "n7": ("n6", "n2", "n0", "n4", "n5", "n1"),
                "n8": ("n4",),
                "n9": ("n4", "n6", "n7", "n1", "n2"),
            },
            ["n8", "n6", "n2", "n4", "n7", "n5", "n3", "n9"],
            0,
            100,
            [["n8", "n6"], ["n2", "n4"], ["n7"], ["n5", "n3"], ["n9"]],
        )
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": (),
                "n3": ("n1", "n2"),
                "n4": ("n0",),
                "n5": ("n2", "n3", "n0", "n1"),
                "n6": ("n4", "n1"),
                "n7": ("n5",),
                "n8": ("n7", "n1", "n5", "n6", "n3", "n4", "n0"),
                "n9": ("n2", "n8"),
            },
            ["n1", "n7", "n4", "n2", "n0", "n8", "n3", "n5"],
            0,
            100,
            [["n1", "n4", "n2"], ["n7"], ["n0", "n3"], ["n8"], ["n5"]],
        )
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": ("n0",),
                "n3": ("n1",),
                "n4": ("n2", "n1"),
                "n5": (),
                "n6": ("n0",),
                "n7": ("n6", "n3", "n2", "n1", "n0"),
                "n8": ("n0", "n2"),
                "n9": ("n6", "n5", "n8", "n4", "n0"),
                "n10": ("n1", "n7", "n5", "n8", "n6", "n2", "n4", "n9"),
            },
            ["n0"],
            0,
            100,
            [["n0"]],
        )

        # trivial test of min_fuse
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": (),
                "n3": ("n1", "n2"),
                "n4": ("n1",),
                "n5": (),
                "n6": ("n5",),
                "n7": ("n1", "n6", "n5", "n2", "n3", "n0"),
                "n8": ("n5", "n7", "n2", "n6"),
                "n9": ("n1",),
                "n10": ("n9",),
                "n11": ("n3", "n4", "n0", "n2"),
                "n12": ("n8", "n9", "n5", "n1"),
                "n13": ("n11", "n4", "n12", "n1", "n9", "n3", "n0"),
            },
            ["n9", "n2", "n8", "n10", "n5", "n6", "n13", "n7", "n3", "n0", "n4"],
            2,
            10,
            [["n9", "n2", "n5", "n0", "n4"], ["n8", "n10"], ["n6", "n3"]],
        )

        # trivial test of max_fuse
        self.verify(
            {
                "n0": (),
                "n1": (),
                "n2": (),
                "n3": ("n1", "n2"),
                "n4": ("n1",),
                "n5": (),
                "n6": ("n5",),
                "n7": ("n1", "n6", "n5", "n2", "n3", "n0"),
                "n8": ("n5", "n7", "n2", "n6"),
                "n9": ("n1",),
                "n10": ("n9",),
                "n11": ("n3", "n4", "n0", "n2"),
                "n12": ("n8", "n9", "n5", "n1"),
                "n13": ("n11", "n4", "n12", "n1", "n9", "n3", "n0"),
            },
            ["n9", "n2", "n8", "n10", "n5", "n6", "n13", "n7", "n3", "n0", "n4"],
            0,
            3,
            [
                ["n9", "n2", "n5"],
                ["n8", "n10", "n4"],
                ["n6", "n3", "n0"],
                ["n13"],
                ["n7"],
            ],
        )

    def test_find_independent_subset_greedy_fuse(self):
        # [FUNCTION DOC] Checks that fusing nodes while iterating changes later groups: after
        # the first group of n2, n3 and n5 is yielded, n2 and n3 are replaced by one fused node,
        # so the remaining nodes come out as n4 alone, then n0 with n1, then the iterator ends.
        # ensure that fusing the sets during iteration results in the correct
        # iteration results. In the example graph after we merge n2 and n3,
        # n4 is no longer independent from n1.
        g, lookup = self.build_graph(
            {
                "n0": (),
                "n1": (),
                "n2": ("n0",),
                "n3": ("n1",),
                "n4": ("n2",),
                "n5": (),
            }
        )
        opts = {
            "min_fuse_set_size": 0,
            "max_fuse_set_size": 100,
        }
        subnodes = ["n2", "n3", "n4", "n0", "n1", "n5"]
        subnodes = [lookup[n] for n in subnodes]
        i = torch._inductor.fx_passes.group_batch_fusion.find_independent_subset_greedy(
            subnodes, opts
        )
        self.assertEqual(next(i), [lookup[n] for n in ["n2", "n3", "n5"]])

        # fuse n2 and n3 which makes n4 now dependent on n1.
        args = tuple(lookup[n] for n in ["n0", "n1"])
        fused = g.create_node("placeholder", "target", name="n2+n3", args=args)
        lookup["n2"].replace_all_uses_with(fused)
        g.erase_node(lookup["n2"])
        lookup["n3"].replace_all_uses_with(fused)
        g.erase_node(lookup["n3"])

        self.assertEqual(next(i), [lookup[n] for n in ["n4"]])
        self.assertEqual(next(i), [lookup[n] for n in ["n0", "n1"]])
        self.assertRaises(StopIteration, lambda: next(i))


class CatLinearMod(torch.nn.Module):
    # [CLASS DOC] HELPER: Fixture applying a linear layer to a concatenation of a list of tensors
    # along the chosen dimension (default last). The input width is the sum of widths if
    # concatenating on the last dimension, otherwise the first width.
    # linear(cat(parts, dim=cat_dim)). cat_dim=-1 is the matchable shape.
    def __init__(self, widths, out_features, has_bias=True, cat_dim=-1, device="cpu"):
        # [FUNCTION DOC] HELPER: Stores the cat dimension and builds the linear layer on the
        # given device.
        super().__init__()
        self.cat_dim = cat_dim
        # when the cat is not on the last dim the feature width is unchanged,
        # so the linear contracts a single part's width
        in_features = sum(widths) if cat_dim == -1 else widths[0]
        self.lin = torch.nn.Linear(in_features, out_features, bias=has_bias).to(device)

    def forward(self, parts):
        # [FUNCTION DOC] HELPER: Concatenates the list of tensors on the stored dimension and
        # returns the linear layer applied to it.
        x = torch.cat(parts, dim=self.cat_dim)
        return self.lin(x)


class CatLinearTwoUserMod(torch.nn.Module):
    # [CLASS DOC] HELPER: Fixture whose concatenation feeds both a linear layer and a sum, so the
    # cat has two users.
    # the cat feeds the linear *and* a second consumer, so it can't be erased.
    def __init__(self, widths, out_features, device="cpu"):
        # [FUNCTION DOC] HELPER: Builds a linear layer over the summed widths with bias on the
        # given device.
        super().__init__()
        self.lin = torch.nn.Linear(sum(widths), out_features).to(device)

    def forward(self, parts):
        # [FUNCTION DOC] HELPER: Concatenates on the last dimension and returns the linear
        # output together with the sum of the concatenated tensor.
        x = torch.cat(parts, dim=-1)
        return self.lin(x), x.sum()


@torch._inductor.config.patch(
    pre_grad_fusion_options={"cat_linear": {}},
    post_grad_fusion_options={},
)
class TestCatLinearFusion(TestCase):
    # [CLASS DOC] Tests for the cat_linear pre-grad pass (config patched to enable only it) that
    # rewrites a linear over a concatenation into per-part linears plus adds. Helpers report
    # whether the pass fires or return the rewritten graph; tests cover accepted and rejected
    # shapes.
    def _fires(self, module, parts, traffic_floor=0):
        # [FUNCTION DOC] HELPER: Clears counters and dynamo state, compiles the module on the
        # inputs with the traffic floor constant of the pass replaced temporarily by the given
        # value, then reads the cat_linear counter, clears counters and returns that count.
        # unit-test tensors never reach the real 512 MiB floor, so unless a test
        # is exercising the floor, drop it to 0 and let the shape gates decide.
        from unittest import mock

        from torch._inductor.fx_passes import group_batch_fusion as gbf

        counters.clear()
        torch._dynamo.reset()
        with mock.patch.object(
            gbf, "CAT_LINEAR_CAT_TRAFFIC_FLOOR_BYTES", traffic_floor
        ):
            torch.compile(module)(parts)
        n = counters["inductor"]["cat_linear"]
        counters.clear()
        return n

    def _fuse_and_get_graph(self, module, parts):
        # [FUNCTION DOC] HELPER: Captures the dynamo graph with a custom backend that runs the
        # pre-grad fusion passes, eliminates dead code, lints and recompiles, with the traffic
        # floor set to zero. Returns the rewritten graph module. Counters are left set for the
        # caller to inspect.
        # run the pre-grad pass on the dynamo graph and hand the rewritten
        # GraphModule back so we can check the cat is really gone, not just that
        # the counter bumped (the counter increments inside fuse, before DCE).
        from unittest import mock

        from torch._inductor.fx_passes import group_batch_fusion as gbf
        from torch._inductor.fx_passes.group_batch_fusion import (
            group_batch_fusion_passes,
        )

        captured = {}

        def backend(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Backend that applies the pre-grad fusion passes to the
            # graph, removes dead nodes, lints, recompiles, stores the module in the enclosing
            # mapping and returns its forward.
            group_batch_fusion_passes(gm.graph, pre_grad=True)
            # the rewrite drops the linear's only use of the cat; DCE is what
            # actually deletes the now-dead cat, same as the real pre-grad path.
            gm.graph.eliminate_dead_code()
            gm.graph.lint()
            gm.recompile()
            captured["gm"] = gm
            return gm.forward

        counters.clear()
        torch._dynamo.reset()
        with mock.patch.object(gbf, "CAT_LINEAR_CAT_TRAFFIC_FLOOR_BYTES", 0):
            torch.compile(module, backend=backend)(parts)
        return captured["gm"]

    def _node_counts(self, gm):
        # [FUNCTION DOC] HELPER: Counts call_function nodes that are concatenations, linears
        # (either the functional or the low-level variant) and python-operator additions.
        # Returns the three counts.
        cats = linears = adds = 0
        for n in gm.graph.nodes:
            if n.op != "call_function":
                continue
            if n.target is torch.cat:
                cats += 1
            elif n.target in (torch.nn.functional.linear, torch._C._nn.linear):
                linears += 1
            elif n.target is operator.add:
                adds += 1
        return cats, linears, adds

    def test_cat_linear_removes_cat_two_part(self):
        # [FUNCTION DOC] Two-part cat-linear case: asserts the pass fires once and the rewritten
        # graph has no cat, two linears and one add.
        m = CatLinearMod([16, 16], 32)
        parts = [torch.randn(4, 16), torch.randn(4, 16)]
        gm = self._fuse_and_get_graph(m, parts)
        self.assertEqual(counters["inductor"]["cat_linear"], 1)
        cats, linears, adds = self._node_counts(gm)
        self.assertEqual(cats, 0)
        self.assertEqual(linears, 2)
        self.assertEqual(adds, 1)
        counters.clear()

    def test_cat_linear_removes_cat_three_part(self):
        # [FUNCTION DOC] Three-part case: asserts the pass fires once and the graph has no cat,
        # three linears and two adds.
        m = CatLinearMod([16, 16, 16], 32)
        parts = [torch.randn(4, 16) for _ in range(3)]
        gm = self._fuse_and_get_graph(m, parts)
        self.assertEqual(counters["inductor"]["cat_linear"], 1)
        cats, linears, adds = self._node_counts(gm)
        self.assertEqual(cats, 0)
        self.assertEqual(linears, 3)
        self.assertEqual(adds, 2)
        counters.clear()

    def test_cat_linear_keeps_cat_when_rejected(self):
        # [FUNCTION DOC] Four parts exceed the allowed count, so the pass must not fire and the
        # graph must keep its single cat.
        # a rejected case (too many parts) must leave the cat untouched
        m = CatLinearMod([16, 16, 16, 16], 32)
        parts = [torch.randn(4, 16) for _ in range(4)]
        gm = self._fuse_and_get_graph(m, parts)
        self.assertEqual(counters["inductor"]["cat_linear"], 0)
        cats, _, _ = self._node_counts(gm)
        self.assertEqual(cats, 1)
        counters.clear()

    def test_cat_linear_two_part(self):
        # [FUNCTION DOC] Asserts the pass fires once for a two-part concatenation.
        m = CatLinearMod([16, 16], 32)
        parts = [torch.randn(4, 16), torch.randn(4, 16)]
        self.assertEqual(self._fires(m, parts), 1)

    def test_cat_linear_three_part(self):
        # [FUNCTION DOC] Asserts the pass fires once for a three-part concatenation.
        m = CatLinearMod([16, 16, 16], 32)
        parts = [torch.randn(4, 16) for _ in range(3)]
        self.assertEqual(self._fires(m, parts), 1)

    def test_cat_linear_rejects_non_last_dim(self):
        # [FUNCTION DOC] Concatenating on dimension zero must be rejected; the pass must not
        # fire.
        m = CatLinearMod([16, 16], 16, cat_dim=0)
        parts = [torch.randn(4, 16), torch.randn(4, 16)]
        self.assertEqual(self._fires(m, parts), 0)

    def test_cat_linear_rejects_too_many_parts(self):
        # [FUNCTION DOC] Four parts must be rejected; the pass must not fire.
        m = CatLinearMod([16, 16, 16, 16], 32)
        parts = [torch.randn(4, 16) for _ in range(4)]
        self.assertEqual(self._fires(m, parts), 0)

    def test_cat_linear_rejects_wide_output(self):
        # [FUNCTION DOC] An output width of 128 exceeds the allowed ceiling, so the pass must
        # not fire.
        # N=128 is past the N<=96 ceiling; a wide output is compute-bound.
        m = CatLinearMod([64, 64], 128)
        parts = [torch.randn(4, 64), torch.randn(4, 64)]
        self.assertEqual(self._fires(m, parts), 0)

    def test_cat_linear_rejects_small_traffic(self):
        # [FUNCTION DOC] With the real traffic floor in effect, a small concatenation is below
        # it, so the pass must not fire even though shape checks would pass.
        # all shape gates pass; at the real 512 MiB floor this concat is too small.
        from torch._inductor.fx_passes import group_batch_fusion as gbf

        m = CatLinearMod([16, 16], 32)
        parts = [torch.randn(4, 16), torch.randn(4, 16)]
        self.assertEqual(
            self._fires(m, parts, traffic_floor=gbf.CAT_LINEAR_CAT_TRAFFIC_FLOOR_BYTES),
            0,
        )

    def test_cat_linear_rejects_thin_piece(self):
        # [FUNCTION DOC] A part of width 4 is too thin; the pass must not fire.
        m = CatLinearMod([4, 16], 32)
        parts = [torch.randn(4, 4), torch.randn(4, 16)]
        self.assertEqual(self._fires(m, parts), 0)

    def test_cat_linear_rejects_multi_user_cat(self):
        # [FUNCTION DOC] A concatenation also used by a sum has two users; the pass must not
        # fire.
        m = CatLinearTwoUserMod([16, 16], 32)
        parts = [torch.randn(4, 16), torch.randn(4, 16)]
        self.assertEqual(self._fires(m, parts), 0)

    @requires_gpu()
    def test_cat_linear_numerics(self):
        # [FUNCTION DOC] Needs GPU. For both bias settings, compares eager and compiled bfloat16
        # output of a two-part cat-linear (traffic floor patched to zero), expects one firing
        # and a close match at loose bfloat16 tolerance.
        from unittest import mock

        from torch._inductor.fx_passes import group_batch_fusion as gbf

        for has_bias in [True, False]:
            counters.clear()
            torch._dynamo.reset()
            module = CatLinearMod(
                [176, 176], 64, has_bias=has_bias, device=GPU_TYPE
            ).to(torch.bfloat16)
            parts = [
                torch.randn(64, 176, device=GPU_TYPE, dtype=torch.bfloat16),
                torch.randn(64, 176, device=GPU_TYPE, dtype=torch.bfloat16),
            ]
            ref = module(parts)
            with mock.patch.object(gbf, "CAT_LINEAR_CAT_TRAFFIC_FLOOR_BYTES", 0):
                res = torch.compile(module)(parts)
            self.assertEqual(counters["inductor"]["cat_linear"], 1)
            self.assertEqual(ref, res, rtol=1.6e-2, atol=1e-2)
            counters.clear()

    @requires_gpu()
    def test_cat_linear_numerics_backward(self):
        # [FUNCTION DOC] Needs GPU. Same setup as the forward numerics test in bfloat16, but
        # runs backward on the summed output and compares the weight grad, bias grad (if
        # present) and per-input grads of eager versus compiled at loose tolerance; asserts the
        # pass fired once.
        # forward is not enough: the rewrite has to keep the backward correct too.
        # grads reach the shared weight through the differentiable slices (one
        # slice per piece), and the bias grad flows only through out_0 since bias
        # rides on the first piece. Compare W.grad, bias.grad, and the per-part
        # input grads eager-vs-compiled at the same bf16 tol as the fwd test.
        from unittest import mock

        from torch._inductor.fx_passes import group_batch_fusion as gbf

        for has_bias in [True, False]:
            counters.clear()
            torch._dynamo.reset()
            module = CatLinearMod(
                [176, 176], 64, has_bias=has_bias, device=GPU_TYPE
            ).to(torch.bfloat16)
            base = [
                torch.randn(64, 176, device=GPU_TYPE, dtype=torch.bfloat16),
                torch.randn(64, 176, device=GPU_TYPE, dtype=torch.bfloat16),
            ]

            def run(compiled):
                # [FUNCTION DOC] HELPER: Zeroes module grads, clones the base inputs as leaf
                # tensors that need grad, runs eager or compiled forward, backward on the sum,
                # and returns clones of the weight grad, bias grad (None without bias) and the
                # list of input grads.
                module.zero_grad(set_to_none=True)
                parts = [p.detach().clone().requires_grad_(True) for p in base]
                fn = torch.compile(module) if compiled else module
                fn(parts).sum().backward()
                return (
                    module.lin.weight.grad.detach().clone(),
                    module.lin.bias.grad.detach().clone() if has_bias else None,
                    [p.grad.detach().clone() for p in parts],
                )

            w_ref, b_ref, in_ref = run(False)
            with mock.patch.object(gbf, "CAT_LINEAR_CAT_TRAFFIC_FLOOR_BYTES", 0):
                w_res, b_res, in_res = run(True)

            self.assertEqual(counters["inductor"]["cat_linear"], 1)
            self.assertEqual(w_ref, w_res, rtol=1.6e-2, atol=1e-2)
            if has_bias:
                self.assertEqual(b_ref, b_res, rtol=1.6e-2, atol=1e-2)
            for gr, gs in zip(in_ref, in_res):
                self.assertEqual(gr, gs, rtol=1.6e-2, atol=1e-2)
            counters.clear()

    def test_cat_linear_numerics_cpu(self):
        # [FUNCTION DOC] CPU float32 forward check of the two-part cat-linear with and without
        # bias, traffic floor patched to zero: expects one firing and equal outputs.
        from unittest import mock

        from torch._inductor.fx_passes import group_batch_fusion as gbf

        for has_bias in [True, False]:
            counters.clear()
            torch._dynamo.reset()
            module = CatLinearMod([24, 24], 32, has_bias=has_bias)
            parts = [torch.randn(8, 24), torch.randn(8, 24)]
            ref = module(parts)
            with mock.patch.object(gbf, "CAT_LINEAR_CAT_TRAFFIC_FLOOR_BYTES", 0):
                res = torch.compile(module)(parts)
            self.assertEqual(counters["inductor"]["cat_linear"], 1)
            self.assertEqual(ref, res)
            counters.clear()

    def test_cat_linear_numerics_backward_cpu(self):
        # [FUNCTION DOC] CPU float32 backward check at default tight tolerances, for both bias
        # settings: weight, bias and input gradients of compiled versus eager must match, and
        # the pass must fire once. Guards against slice offset or bias placement errors that
        # forward checks cannot see.
        # fp32 pin for the backward. forward alone can't catch a wrong weight
        # column offset or a misplaced bias, both only show up in the grads:
        # the shared weight grad comes through the differentiable slices (one
        # per piece) and the bias grad flows only through out_0 since the bias
        # rides the first piece. fp32 at the default (tight) tol so a subtle
        # slice/bias error can't hide under a loose bf16 tolerance.
        from unittest import mock

        from torch._inductor.fx_passes import group_batch_fusion as gbf

        for has_bias in [True, False]:
            counters.clear()
            torch._dynamo.reset()
            module = CatLinearMod([24, 24], 32, has_bias=has_bias)
            base = [torch.randn(8, 24), torch.randn(8, 24)]

            def run(compiled):
                # [FUNCTION DOC] HELPER: Zeroes module grads, clones the base inputs as leaf
                # tensors that need grad, runs eager or compiled forward, backward on the sum,
                # and returns clones of the weight grad, bias grad (None without bias) and the
                # list of input grads.
                module.zero_grad(set_to_none=True)
                parts = [p.detach().clone().requires_grad_(True) for p in base]
                fn = torch.compile(module) if compiled else module
                fn(parts).sum().backward()
                return (
                    module.lin.weight.grad.detach().clone(),
                    module.lin.bias.grad.detach().clone() if has_bias else None,
                    [p.grad.detach().clone() for p in parts],
                )

            w_ref, b_ref, in_ref = run(False)
            with mock.patch.object(gbf, "CAT_LINEAR_CAT_TRAFFIC_FLOOR_BYTES", 0):
                w_res, b_res, in_res = run(True)

            self.assertEqual(counters["inductor"]["cat_linear"], 1)
            self.assertEqual(w_ref, w_res)
            if has_bias:
                self.assertEqual(b_ref, b_res)
            for gr, gs in zip(in_ref, in_res):
                self.assertEqual(gr, gs)
            counters.clear()


if __name__ == "__main__":
    run_tests()
