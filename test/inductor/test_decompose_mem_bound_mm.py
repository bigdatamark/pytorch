# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the Inductor post-grad pass that decomposes memory-bound matrix
# multiplications (bmm, mm, addmm, linear) into cheaper pointwise/reduction forms. Each test
# compiles a small module, compares outputs, parameters and gradients with eager, and checks the
# inductor counters for the decompose bmm, mm and addmm entries against whether the shape should
# trigger the pass. Also tests the device-check helper and that inputs are realized. Most tests
# need a GPU.

import logging
import unittest

import torch
import torch._inductor
from torch._dynamo.utils import counters
from torch._inductor.fx_passes.decompose_mem_bound_mm import check_device
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    is_navi3_arch,
    parametrize,
    patch_test_members,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU_AND_TRITON
from torch.testing._internal.triton_utils import requires_gpu


class MyModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Module wrapping a single linear layer built from input and output
    # feature counts and a bias flag; the forward pass applies the layer.
    def __init__(
        self, n_input: int, n_output: int, has_bias: bool, device=GPU_TYPE
    ) -> None:
        # [FUNCTION DOC] HELPER: Creates the linear layer with the given sizes and bias flag
        # (the device argument is accepted but unused).
        super().__init__()
        self.linear = torch.nn.Linear(n_input, n_output, bias=has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Applies the linear layer to the input and returns the result.
        return self.linear(x)


class MyModule2(torch.nn.Module):
    # [CLASS DOC] HELPER: Module whose forward pass is a batched matrix multiply of two inputs.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Calls the base module initializer only.
        super().__init__()

    def forward(self, input1, input2):
        # [FUNCTION DOC] HELPER: Returns the batched matrix product of the two inputs.
        output = torch.bmm(input1, input2)
        return output


class MyModule3(torch.nn.Module):
    # [CLASS DOC] HELPER: Module whose forward pass is a plain matrix multiply of two inputs.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Calls the base module initializer only.
        super().__init__()

    def forward(self, input1, input2):
        # [FUNCTION DOC] HELPER: Returns the matrix product of the two inputs.
        output = torch.mm(input1, input2)
        return output


class _TestDecomposeAddMM(torch.nn.Module):
    # [CLASS DOC] HELPER: Module whose forward pass calls the aten addmm op directly, adding the
    # product of the last two inputs to the first.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Calls the base module initializer only.
        super().__init__()

    def forward(
        self, z: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Returns the bias-add matrix multiply of the three inputs
        # through the aten op overload, which the decompose pass can match.
        return torch.ops.aten.addmm.default(z, x, y)


@requires_gpu
@torch._inductor.config.patch(
    post_grad_fusion_options={
        "decompose_mm_pass": {},
    }
)
@instantiate_parametrized_tests
class TestDecomposeMemMM(TestCase):
    # [CLASS DOC] GPU test case, parametrized, with the decompose mm pass enabled through post
    # grad fusion options. Holds comparison tolerances (absolute and relative, default 1e-3)
    # used by the helper compare methods; tests may override them through patched members.
    def __init__(self, method_name="runTest", methodName="runTest"):
        # [FUNCTION DOC] Forwards both name arguments to the base test case and sets the default
        # absolute and relative tolerance to 1e-3.
        super().__init__(method_name, methodName)
        self.atol = 1e-3
        self.rtol = 1e-3

    def setup_tolerance(self, rtol=None, atol=None):
        # [FUNCTION DOC] HELPER: Resolves the tolerances: a missing relative or absolute
        # value falls back to the current instance value, and the result is stored back on
        # the instance.
        if rtol is None:
            rtol = self.rtol
        if atol is None:
            atol = self.atol
        self.rtol = rtol
        self.atol = atol

    def compare_dict_tensors(self, ref_dict, res_dict, rtol=None, atol=None):
        # [FUNCTION DOC] HELPER: Compares two mappings of tensors. The compiled-module mapping
        # is looked up with an original-module key prefix added to each reference key. Returns
        # False if the number of keys differs or any pair is not close within tolerance; raises
        # an assertion error if an expected prefixed key is missing from the result mapping;
        # returns True otherwise.
        self.setup_tolerance(rtol, atol)
        if len(set(ref_dict.keys())) != len(set(res_dict.keys())):
            return False
        for key1 in ref_dict:
            key2 = "_orig_mod." + key1
            if key2 not in res_dict:
                raise AssertionError(f"{key1} does not exist in traced module")
            if not torch.allclose(
                ref_dict[key1], res_dict[key2], rtol=self.rtol, atol=self.atol
            ):
                return False
        return True

    def compare_pred(self, module, traced, input, rtol=None, atol=None):
        # [FUNCTION DOC] HELPER: Runs the eager module and the compiled module on the same
        # inputs and asserts the outputs are equal within the tolerances.
        self.setup_tolerance(rtol, atol)
        ref = module(*input)
        res = traced(*input)
        self.assertEqual(ref, res, rtol=self.rtol, atol=self.atol)

    def compare_parameters(self, module, traced, rtol=None, atol=None):
        # [FUNCTION DOC] HELPER: Collects named parameters of the eager and compiled modules and
        # asserts they are close within tolerance, using the prefixed key comparison.
        self.setup_tolerance(rtol, atol)
        ref_params = dict(module.named_parameters())
        res_params = dict(traced.named_parameters())
        self.assertTrue(
            self.compare_dict_tensors(
                ref_params, res_params, rtol=self.rtol, atol=self.atol
            )
        )

    def compare_gradients(self, module, traced, rtol=None, atol=None):
        # [FUNCTION DOC] HELPER: Collects the gradient of each named parameter of the eager
        # and compiled modules and asserts they are close within tolerance, using the prefixed
        # key comparison.
        self.setup_tolerance(rtol, atol)
        ref_grad = {key: param.grad for key, param in module.named_parameters()}
        res_grad = {key: param.grad for key, param in traced.named_parameters()}
        self.assertTrue(
            self.compare_dict_tensors(
                ref_grad, res_grad, rtol=self.rtol, atol=self.atol
            )
        )

    @parametrize(
        "b,m,k,n,should_decompose",
        [(10240, 2, 2, 2, True), (10240, 2, 32, 32, False), (2000, 2, 2, 2, False)],
    )
    def test_decompose_bmm(self, b, m, n, k, should_decompose):
        # [FUNCTION DOC] Parametrized over batch, row, inner and column sizes plus whether the
        # pass should apply. Sets debug logging, compiles a batched matrix multiply module on
        # the GPU, checks outputs, then the decompose bmm counter is 1 when decomposition is
        # expected and a GPU with triton exists (else 0). After backward it compares parameters
        # and gradients and expects the counter to reach 3 or 0 accordingly. Clears counters
        # before and after.
        torch._logging.set_logs(inductor=logging.DEBUG)
        mat1 = torch.randn(b, m, k, device=GPU_TYPE).requires_grad_(True)
        mat2 = torch.randn(b, k, n, device=GPU_TYPE).requires_grad_(True)

        counters.clear()

        module = MyModule2().to(GPU_TYPE)
        traced = torch.compile(module)
        input = [mat1, mat2]
        ref = module(*input)
        res = traced(*input)

        self.compare_pred(module, traced, input)

        expected_val = 1 if should_decompose and HAS_GPU_AND_TRITON else 0
        self.assertEqual(
            counters["inductor"]["decompose_bmm"],
            expected_val,
        )

        ref.sum().backward()
        res.sum().backward()
        self.compare_parameters(module, traced)
        self.compare_gradients(module, traced)

        expected_val = 3 if should_decompose and HAS_GPU_AND_TRITON else 0
        self.assertEqual(
            counters["inductor"]["decompose_bmm"],
            expected_val,
        )
        counters.clear()

    @parametrize(
        "b,m,k,n,should_decompose",
        [(1, 2, 2, 2, True), (2, 2, 2, 2, False)],
    )
    def test_decompose_bmm_cpu(self, b, m, n, k, should_decompose):
        # [FUNCTION DOC] CPU variant of the batched matrix multiply decomposition test: compiles
        # on CPU, checks outputs, and expects the decompose bmm counter to be 1 if decomposition
        # is expected, else 0. No backward pass is run.
        torch._logging.set_logs(inductor=logging.DEBUG)
        mat1 = torch.randn(b, m, k)
        mat2 = torch.randn(b, k, n)

        counters.clear()

        module = MyModule2()
        traced = torch.compile(module)
        input = [mat1, mat2]
        self.compare_pred(module, traced, input)

        expected_val = 1 if should_decompose else 0
        self.assertEqual(
            counters["inductor"]["decompose_bmm"],
            expected_val,
        )
        counters.clear()

    @parametrize(
        "m,k,n, should_decompose",
        [(20480, 5, 2, True), (20480, 32, 2, False), (2048, 2, 2, False)],
    )
    @parametrize("has_bias", [True, False])
    def test_decompose_linear(self, m, n, k, has_bias, should_decompose):
        # [FUNCTION DOC] Parametrized over sizes, bias flag and expected decomposition. Compiles
        # a linear module on the GPU and checks outputs. The forward counter checked is the
        # addmm one when the layer has a bias and the mm one otherwise, equal to 1 if
        # decomposition is expected and GPU with triton exist, else 0. After backward it
        # compares parameters and gradients and expects the mm counter to grow by the same
        # expected value. Clears counters at the end.
        torch._logging.set_logs(inductor=logging.DEBUG)
        input = torch.randn(m, k, device=GPU_TYPE).requires_grad_(True)

        counters.clear()

        module = MyModule(k, n, has_bias).to(GPU_TYPE)
        traced = torch.compile(module)
        input = [input]
        ref = module(*input)
        res = traced(*input)

        self.compare_pred(module, traced, input)

        expected_val = 1 if should_decompose and HAS_GPU_AND_TRITON else 0
        if has_bias:
            self.assertEqual(
                counters["inductor"]["decompose_addmm"],
                expected_val,
            )
        else:
            self.assertEqual(
                counters["inductor"]["decompose_mm"],
                expected_val,
            )
        decompose_mm_fwd = counters["inductor"]["decompose_mm"]

        ref.sum().backward()
        res.sum().backward()

        self.compare_parameters(module, traced)
        self.compare_gradients(module, traced)

        self.assertEqual(
            counters["inductor"]["decompose_mm"] - decompose_mm_fwd,
            expected_val,
        )
        counters.clear()

    # We have to increase tolerance for navi3 because all fp16, bf16
    # GEMMs operations have an accuracy issue caused by hardware limitation
    @patch_test_members(
        {
            "atol": 2e-3 if is_navi3_arch() else 1e-3,
            "rtol": 2e-3 if is_navi3_arch() else 1e-3,
        }
    )
    @parametrize(
        "m,k,n, should_decompose",
        [(20480, 5, 2, True), (20480, 32, 2, False), (2048, 2, 2, False)],
    )
    @parametrize("has_bias", [True, False])
    def test_decompose_linear_mixed_precision(
        self, m, n, k, has_bias, should_decompose
    ):
        # [FUNCTION DOC] Same checks as the linear decomposition test but run under bfloat16
        # autocast on the GPU. The class tolerances are patched to a looser value on Navi3
        # hardware via a decorator.
        with torch.amp.autocast(device_type=GPU_TYPE, dtype=torch.bfloat16):
            torch._logging.set_logs(inductor=logging.DEBUG)
            input = torch.randn(m, k, device=GPU_TYPE).requires_grad_(True)

            counters.clear()

            module = MyModule(k, n, has_bias).to(GPU_TYPE)
            traced = torch.compile(module)
            input = [input]
            ref = module(*input)
            res = traced(*input)

            self.compare_pred(module, traced, input)

            expected_val = 1 if should_decompose and HAS_GPU_AND_TRITON else 0
            if has_bias:
                self.assertEqual(
                    counters["inductor"]["decompose_addmm"],
                    expected_val,
                )
            else:
                self.assertEqual(
                    counters["inductor"]["decompose_mm"],
                    expected_val,
                )
            decompose_mm_fwd = counters["inductor"]["decompose_mm"]

            ref.sum().backward()
            res.sum().backward()

            self.compare_parameters(module, traced)
            self.compare_gradients(module, traced)

            self.assertEqual(
                counters["inductor"]["decompose_mm"] - decompose_mm_fwd,
                expected_val,
            )
            counters.clear()

    @parametrize(
        "m,k,n, should_decompose",
        [(20480, 5, 2, True), (20480, 32, 2, False), (2048, 2, 2, False)],
    )
    @parametrize("has_bias", [True, False])
    def test_decompose_mm(self, m, n, k, has_bias, should_decompose):
        # [FUNCTION DOC] Compiles a plain matrix multiply module on the GPU. Expects the
        # decompose mm counter to be 1 (or 0) after the forward pass and the increase after
        # backward to also be 1 (or 0), depending on the expected-decomposition flag and GPU
        # with triton availability; also compares outputs, parameters and gradients. The bias
        # flag parameter is not used.
        torch._logging.set_logs(inductor=logging.DEBUG)
        mat1 = torch.randn(m, k, device=GPU_TYPE).requires_grad_(True)
        mat2 = torch.randn(k, n, device=GPU_TYPE).requires_grad_(True)

        counters.clear()

        module = MyModule3().to(GPU_TYPE)
        traced = torch.compile(module)
        input = [mat1, mat2]
        ref = module(*input)
        res = traced(*input)

        self.compare_pred(module, traced, input)

        expected_val = 1 if should_decompose and HAS_GPU_AND_TRITON else 0
        self.assertEqual(
            counters["inductor"]["decompose_mm"],
            expected_val,
        )
        decompose_mm_fwd = counters["inductor"]["decompose_mm"]

        ref.sum().backward()
        res.sum().backward()
        self.compare_parameters(module, traced)
        self.compare_gradients(module, traced)

        expected_val = 1 if should_decompose and HAS_GPU_AND_TRITON else 0
        self.assertEqual(
            counters["inductor"]["decompose_mm"] - decompose_mm_fwd,
            expected_val,
        )
        counters.clear()

    @parametrize(
        "m,k,n, should_decompose",
        [(1, 64, 16, True), (2, 64, 16, False), (1, 64, 32, True)],
    )
    def test_decompose_mm_cpu(self, m, n, k, should_decompose):
        # [FUNCTION DOC] CPU variant of the plain matrix multiply decomposition test: compiles
        # on CPU, checks outputs and expects the decompose mm counter to be 1 if decomposition
        # is expected, else 0. No backward pass is run.
        torch._logging.set_logs(inductor=logging.DEBUG)
        mat1 = torch.randn(m, k)
        mat2 = torch.randn(k, n)
        counters.clear()

        module = MyModule3()
        traced = torch.compile(module)
        input = [mat1, mat2]
        self.compare_pred(module, traced, input)

        expected_val = 1 if should_decompose else 0
        self.assertEqual(
            counters["inductor"]["decompose_mm"],
            expected_val,
        )
        counters.clear()

    # We have to increase tolerance for navi3 because all fp16, bf16
    # GEMMs operations have an accuracy issue caused by hardware limitation
    @patch_test_members(
        {
            "atol": 8e-3 if is_navi3_arch() else 1e-3,
            "rtol": 8e-3 if is_navi3_arch() else 1e-3,
        }
    )
    @parametrize(
        "m,k,n, should_decompose",
        [(20480, 5, 2, True), (20480, 32, 2, False), (2048, 2, 2, False)],
    )
    @parametrize("has_bias", [True, False])
    def test_decompose_mm_mixed_precision(self, m, n, k, has_bias, should_decompose):
        # [FUNCTION DOC] Same as the GPU matrix multiply decomposition test but run under
        # bfloat16 autocast; tolerances are patched looser on Navi3 hardware via a decorator.
        # The bias flag parameter is unused.
        with torch.amp.autocast(device_type=GPU_TYPE, dtype=torch.bfloat16):
            torch._logging.set_logs(inductor=logging.DEBUG)
            mat1 = torch.randn(m, k, device=GPU_TYPE).requires_grad_(True)
            mat2 = torch.randn(k, n, device=GPU_TYPE).requires_grad_(True)

            counters.clear()

            module = MyModule3().to(GPU_TYPE)
            traced = torch.compile(module)
            input = [mat1, mat2]
            ref = module(*input)
            res = traced(*input)

            self.compare_pred(module, traced, input)

            expected_val = 1 if should_decompose and HAS_GPU_AND_TRITON else 0
            self.assertEqual(
                counters["inductor"]["decompose_mm"],
                expected_val,
            )
            decompose_mm_fwd = counters["inductor"]["decompose_mm"]

            ref.sum().backward()
            res.sum().backward()
            self.compare_parameters(module, traced)
            self.compare_gradients(module, traced)

            expected_val = 1 if should_decompose and HAS_GPU_AND_TRITON else 0
            self.assertEqual(
                counters["inductor"]["decompose_mm"] - decompose_mm_fwd,
                expected_val,
            )
            counters.clear()

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/153736")
    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/153735")
    @unittest.skip
    @parametrize("m,k,n, should_decompose", [(20480, 5, 2, True)])
    @parametrize("has_bias", [True, False])
    def test_dynamic_shape(self, m, n, k, has_bias, should_decompose):
        # [FUNCTION DOC] Dynamic-shape variant for the linear module, compiled with dynamic
        # shapes; checks the addmm counter in the biased case and later the total mm counter (1
        # with bias, 2 without when a GPU with triton exists). The test is disabled: it is
        # skipped unconditionally by a decorator, so it never runs.
        torch._logging.set_logs(inductor=logging.DEBUG)
        input = torch.randn(m, k, device=GPU_TYPE).requires_grad_(True)

        counters.clear()

        module = MyModule(k, n, has_bias).to(GPU_TYPE)
        traced = torch.compile(module, dynamic=True)
        input = [input]
        ref = module(*input)
        res = traced(*input)

        self.compare_pred(module, traced, input)

        expected_val = 1 if should_decompose and HAS_GPU_AND_TRITON else 0
        if has_bias:
            self.assertEqual(
                counters["inductor"]["decompose_addmm"],
                expected_val,
            )

        ref.sum().backward()
        res.sum().backward()

        self.compare_parameters(module, traced)
        self.compare_gradients(module, traced)

        expected_val = 0
        if HAS_GPU_AND_TRITON:
            expected_val = 1 if has_bias else 2

        self.assertEqual(
            counters["inductor"]["decompose_mm"],
            expected_val,
        )
        counters.clear()

    def test_realize_input(self):
        # [FUNCTION DOC] Compiles a function that transposes and makes contiguous its first
        # input before a matrix multiply with the second, runs it with generated-code capture,
        # and uses a file check to assert the generated code has exactly two kernel run calls.
        m = 20480
        k = 5
        n = 2
        torch._logging.set_logs(inductor=logging.DEBUG)
        input1 = torch.randn(m, k, device=GPU_TYPE).T.contiguous()
        input2 = torch.randn(k, n, device=GPU_TYPE)

        @torch.compile()
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Compiled function: transposes the first input, makes it
            # contiguous, and multiplies it with the second input.
            return x.T.contiguous() @ y

        _, code = run_and_get_code(foo, input1, input2)

        # two kernels generated
        FileCheck().check_count(".run(", 2, exactly=True).run(code[0])

    def test_check_device(self):
        # [FUNCTION DOC] Exercises the device-check helper with tensors on the GPU, on the CPU,
        # and mixed. Asserts it is true only when both inputs share a device type equal to the
        # named device (GPU type or cpu), false when the device is omitted for CPU inputs (the
        # helper default is cuda), false for mixed inputs with either device name, and false for
        # a device name that is not an actual device type such as the generic gpu or mtia.
        m = 5
        k = 5
        n = 2
        torch._logging.set_logs(inductor=logging.DEBUG)

        input1 = torch.randn(m, k, device=GPU_TYPE)
        input2 = torch.randn(k, n, device=GPU_TYPE)
        self.assertTrue(check_device(input1, input2, device=GPU_TYPE))
        self.assertFalse(check_device(input1, input2, device="cpu"))

        input1 = torch.randn(m, k)
        input2 = torch.randn(k, n)
        self.assertTrue(check_device(input1, input2, device="cpu"))
        self.assertFalse(check_device(input1, input2))

        input1 = torch.randn(m, k, device=GPU_TYPE)
        input2 = torch.randn(k, n)
        self.assertFalse(check_device(input1, input2, device="gpu"))
        self.assertFalse(check_device(input1, input2, device="cpu"))

        self.assertFalse(check_device(input1, input2, device="mtia"))

    @torch._inductor.config.patch(
        post_grad_fusion_options={
            "decompose_mm_pass": {"skip_dynamic_shape_dim_check": True},
        }
    )
    def test_dynamic_shape_decompose_addmm(self):
        # [FUNCTION DOC] With the decompose pass configured to skip the dynamic-shape dimension
        # check, compiles the addmm-based module on the GPU with dynamic shapes and a large row
        # count, compares the output with eager, and expects the decompose addmm counter to be
        # exactly 1.
        m, k, n = 19494144, 8, 8
        input = torch.randn(m, k, device=GPU_TYPE).requires_grad_(False)
        weight = torch.randn(k, n, device=GPU_TYPE).requires_grad_(False)
        bias = torch.randn(n, device=GPU_TYPE).requires_grad_(False)

        counters.clear()

        module = _TestDecomposeAddMM().to(GPU_TYPE)
        traced = torch.compile(module, dynamic=True)
        input = [bias, input, weight]

        self.compare_pred(module, traced, input)

        self.assertEqual(
            counters["inductor"]["decompose_addmm"],
            1,
        )
        counters.clear()


if __name__ == "__main__":
    run_tests()
