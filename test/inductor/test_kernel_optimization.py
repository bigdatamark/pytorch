# Owner(s): ["module: inductor"]
# [MODULE DOC] Inductor test for kernel-level graph optimization passes. Defines a small einsum
# module and one GPU test checking that the einsum-to-pointwise pre-grad pass fires and that
# compiled results, parameters and gradients match eager. No import-time side effects.

import torch
import torch._inductor
from torch._dynamo.utils import counters
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import serialTest
from torch.testing._internal.inductor_utils import GPU_TYPE, requires_gpu


class _TestEinsumtoPointwise(torch.nn.Module):
    # [CLASS DOC] HELPER: Test module with two einsum calls (shared-weight and per-batch weight
    # forms), each followed by a bias add, whose results are summed. Holds no parameters; all
    # tensors are passed to the forward method.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Constructor; only calls the base module constructor.
        super().__init__()

    def forward(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        bias: torch.Tensor,
        input2: torch.Tensor,
        weights2: torch.Tensor,
        bias2: torch.Tensor,
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Takes input, weights, bias and a second input, weights, bias
        # triple. Computes the first einsum (batch/n/i with n/i/o) plus bias, the second einsum
        # (batch/n/i with batch/n/i/o) plus bias2, and returns the elementwise sum of both
        # results.
        output = torch.functional.einsum("bni, nio -> bno", input, weights)
        add1 = output.add(bias)
        output2 = torch.functional.einsum("bni, bnio -> bno", input2, weights2)
        add2 = output2 + bias2
        return add1 + add2


class TestKernelOptimization(TestCase):
    # [CLASS DOC] Test case that compiles the einsum module with torch.compile and compares
    # against eager. Provides comparison helpers for predictions, parameters and gradients.
    def compare_dict_tensors(self, ref_dict, res_dict, rtol=1e-3, atol=1e-3):
        # [FUNCTION DOC] HELPER: Compares two mappings of name to tensor, where the second has
        # keys prefixed with the compiled-module wrapper prefix. Returns False when key counts
        # differ or any pair is not close within rtol/atol; raises AssertionError if an expected
        # key is missing from the second mapping. Returns True otherwise.
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
        # [FUNCTION DOC] HELPER: Runs the eager and traced modules on the same inputs and asserts
        # the outputs are equal within the given tolerances.
        ref = module(*input)
        res = traced(*input)
        self.assertEqual(ref, res, rtol=rtol, atol=atol)

    def compare_parameters(self, module, traced, rtol=1e-3, atol=1e-3):
        # [FUNCTION DOC] HELPER: Collects named parameters of both modules and asserts they match
        # via the dictionary comparison helper with the given tolerances.
        ref_params = dict(module.named_parameters())
        res_params = dict(traced.named_parameters())
        self.assertTrue(self.compare_dict_tensors(ref_params, res_params, rtol, atol))

    def compare_gradients(self, module, traced, rtol=1e-3, atol=1e-3):
        # [FUNCTION DOC] HELPER: Collects the gradient of every named parameter of both modules
        # and asserts they match via the dictionary comparison helper. Parameters without grads
        # yield None entries, which the underlying close-check would not accept.
        ref_grad = {key: param.grad for key, param in module.named_parameters()}
        res_grad = {key: param.grad for key, param in traced.named_parameters()}
        self.assertTrue(
            self.compare_dict_tensors(ref_grad, res_grad, rtol=rtol, atol=atol)
        )

    @requires_gpu()
    @torch._inductor.config.patch(
        pre_grad_fusion_options={
            "einsum_to_pointwise_pass": {},
        },
        post_grad_fusion_options={},
    )
    @serialTest()  # Needs slightly more memory on GPUs
    def test_einsum_to_pointwise(self):
        # [FUNCTION DOC] GPU-only, serial test. Enables only the einsum-to-pointwise pre-grad
        # pass, clears the dynamo counters, runs the eager and compiled module on six random GPU
        # inputs, runs backward on both, then compares predictions, parameters and gradients.
        # Asserts the inductor counter for the pass equals one, and clears counters again at the
        # end.
        counters.clear()
        module = _TestEinsumtoPointwise().to(GPU_TYPE)
        input = [
            torch.randn(4096, 9, 512, device=GPU_TYPE, requires_grad=True),
            torch.randn(9, 512, 96, device=GPU_TYPE, requires_grad=True),
            torch.randn(9, 96, device=GPU_TYPE, requires_grad=True),
            torch.randn(4096, 9, 160, device=GPU_TYPE, requires_grad=True),
            torch.randn(4096, 9, 160, 96, device=GPU_TYPE, requires_grad=True),
            torch.randn(4096, 9, 96, device=GPU_TYPE, requires_grad=True),
        ]
        traced = torch.compile(module)
        ref = module(*input)
        res = traced(*input)
        ref.sum().backward()
        res.sum().backward()
        self.compare_pred(module, traced, input)
        self.compare_parameters(module, traced)
        self.compare_gradients(module, traced)
        self.assertEqual(
            counters["inductor"]["einsum_to_pointwise_pass"],
            1,
        )
        counters.clear()


if __name__ == "__main__":
    run_tests()
