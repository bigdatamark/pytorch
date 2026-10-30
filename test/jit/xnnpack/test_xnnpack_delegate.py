# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for lowering TorchScript modules to the XNNPACK backend via the private
# backend-lowering binding. Import-time side effect: loads the XNNPACK backend shared library
# through the torch ops loader using a Buck-style target path, so the file only imports in that
# build setup. The file is not run by default; running it directly raises an error saying it must
# be enabled in the test discovery list first.

import unittest

import torch
import torch._C


torch.ops.load_library("//caffe2:xnnpack_backend")


class TestXNNPackBackend(unittest.TestCase):
    # [CLASS DOC] unittest case for the XNNPACK backend. Tests script a small module, lower it with
    # a compile spec naming forward input and output example tensors, and compare lowered output
    # against the scripted module or check the error text for invalid specs and ops.
    def test_xnnpack_constant_data(self):
        # [FUNCTION DOC] Lowers a module that adds a constant tensor attribute to its input, then
        # runs twenty random inputs through both the scripted and lowered modules and asserts the
        # outputs are close with tolerance 1e-3 (absolute and relative).
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a constant ones tensor of shape 4 by 4 by 4.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the constant ones tensor as an attribute.
                super().__init__()
                self._constant = torch.ones(4, 4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus the stored constant tensor.
                return x + self._constant

        scripted_module = torch.jit.script(Module())

        lowered_module = torch._C._jit_to_backend(
            "xnnpack",
            scripted_module,
            {
                "forward": {
                    "inputs": [torch.randn(4, 4, 4)],
                    "outputs": [torch.randn(4, 4, 4)],
                }
            },
        )

        for _ in range(20):
            sample_input = torch.randn(4, 4, 4)
            actual_output = scripted_module(sample_input)
            expected_output = lowered_module(sample_input)
            self.assertTrue(
                torch.allclose(actual_output, expected_output, atol=1e-03, rtol=1e-03)
            )

    def test_xnnpack_lowering(self):
        # [FUNCTION DOC] Checks lowering validation. Lowering with a compile spec lacking the
        # forward key must raise a runtime error naming the missing key; a spec with two example
        # inputs for a single-input forward must raise a runtime error about input count
        # mismatch; a valid spec must lower successfully and the result must be callable on a
        # one-element tensor.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward returns its input added to itself.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input doubled by adding it to itself.
                return x + x

        scripted_module = torch.jit.script(Module())

        faulty_compile_spec = {
            "backward": {
                "inputs": [torch.zeros(1)],
                "outputs": [torch.zeros(1)],
            }
        }
        error_msg = 'method_compile_spec does not contain the "forward" key.'

        with self.assertRaisesRegex(
            RuntimeError,
            error_msg,
        ):
            _ = torch._C._jit_to_backend(
                "xnnpack",
                scripted_module,
                faulty_compile_spec,
            )

        mismatch_compile_spec = {
            "forward": {
                "inputs": [torch.zeros(1), torch.zeros(1)],
                "outputs": [torch.zeros(1)],
            }
        }
        error_msg = (
            "method_compile_spec inputs do not match expected number of forward inputs"
        )

        with self.assertRaisesRegex(
            RuntimeError,
            error_msg,
        ):
            _ = torch._C._jit_to_backend(
                "xnnpack", scripted_module, mismatch_compile_spec
            )

        lowered = torch._C._jit_to_backend(
            "xnnpack",
            scripted_module,
            {
                "forward": {
                    "inputs": [torch.zeros(1)],
                    "outputs": [torch.zeros(1)],
                }
            },
        )
        lowered(torch.zeros(1))

    def test_xnnpack_backend_add(self):
        # [FUNCTION DOC] Lowers a three-addition module over two 1x512x512x3 inputs and asserts
        # that calling forward on the lowered module gives results close (tolerance 1e-3) to the
        # scripted module's result.
        class AddModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that adds two inputs and then adds the first input
            # twice more.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the sum of both inputs plus two extra copies of
                # the first input.
                z = x + y
                z = z + x
                z = z + x
                return z

        add_module = AddModule()
        sample_inputs = (torch.rand(1, 512, 512, 3), torch.rand(1, 512, 512, 3))
        sample_output = torch.zeros(1, 512, 512, 3)

        add_module = torch.jit.script(add_module)
        expected_output = add_module(sample_inputs[0], sample_inputs[1])

        lowered_add_module = torch._C._jit_to_backend(
            "xnnpack",
            add_module,
            {
                "forward": {
                    "inputs": [sample_inputs[0].clone(), sample_inputs[1].clone()],
                    "outputs": [sample_output],
                }
            },
        )

        actual_output = lowered_add_module.forward(sample_inputs[0], sample_inputs[1])
        self.assertTrue(
            torch.allclose(actual_output, expected_output, atol=1e-03, rtol=1e-03)
        )

    def test_xnnpack_broadcasting(self):
        # [FUNCTION DOC] Lowers an addition module whose inputs have different shapes that
        # broadcast (5x1x4x1 and 3x1x1) to a 5x3x4x1 output, and asserts the lowered forward output
        # is close to the scripted result with tolerance 1e-3.
        class AddModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that adds its two inputs with broadcasting.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the sum of the two inputs.
                return x + y

        add_module = AddModule()
        sample_inputs = (torch.rand(5, 1, 4, 1), torch.rand(3, 1, 1))
        sample_output = torch.zeros(5, 3, 4, 1)

        add_module = torch.jit.script(add_module)
        expected_output = add_module(sample_inputs[0], sample_inputs[1])

        lowered_add_module = torch._C._jit_to_backend(
            "xnnpack",
            add_module,
            {
                "forward": {
                    "inputs": [sample_inputs[0], sample_inputs[1]],
                    "outputs": [sample_output],
                }
            },
        )

        actual_output = lowered_add_module.forward(sample_inputs[0], sample_inputs[1])
        self.assertTrue(
            torch.allclose(actual_output, expected_output, atol=1e-03, rtol=1e-03)
        )

    def test_xnnpack_unsupported(self):
        # [FUNCTION DOC] Lowers a module that indexes its second input with an integer and slices,
        # and asserts lowering raises a runtime error whose message lists the unsupported select
        # and slice ops.
        class AddSpliceModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward uses an integer index and slicing, ops that
            # the backend does not support.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the first input plus the second input indexed at
                # position 1 on dimension 2.
                z = x + y[:, :, 1, :]
                return z

        sample_inputs = (torch.rand(1, 512, 512, 3), torch.rand(1, 512, 512, 3))
        sample_output = torch.zeros(1, 512, 512, 3)

        error_msg = (
            "the module contains the following unsupported ops:\n"
            "aten::select\n"
            "aten::slice\n"
        )

        add_module = torch.jit.script(AddSpliceModule())
        with self.assertRaisesRegex(
            RuntimeError,
            error_msg,
        ):
            _ = torch._C._jit_to_backend(
                "xnnpack",
                add_module,
                {
                    "forward": {
                        "inputs": [sample_inputs[0], sample_inputs[1]],
                        "outputs": [sample_output],
                    }
                },
            )


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
