# Owner(s): ["oncall: quantization"]
# [MODULE DOC] Tests for the TorchScript pass fusing a quantized add with a following relu.
# Scripts small modules calling quantized add, add_out, add_scalar and add_scalar_out followed
# by relu, runs the inline and fuse-add-relu passes, uses FileCheck to confirm the relu is gone
# and the fused op appears, and checks outputs equal the unfused result. Running the file
# directly raises an error via raise_on_run_directly.

# torch
import torch
from torch.testing import FileCheck
from torch.testing._internal.common_quantization import QuantizationTestCase
from torch.testing._internal.common_utils import raise_on_run_directly


class TestFusionPasses(QuantizationTestCase):
    # [CLASS DOC] QuantizationTestCase subclass with a single test for add plus relu fusion in
    # scripted graphs. Holds no state.
    def test_quantized_add_relu_fusion(self):
        # [FUNCTION DOC] Builds quantized uint8 inputs (scale 2, zero point 127, values from
        # -128 to 129) and, for four module variants (tensor add, out-variant add, scalar add,
        # scalar out-variant add), scripts the module, records the reference output, applies the
        # inline pass and the quantized add-relu fusion pass, asserts via FileCheck that the
        # aten relu and the unfused op disappear in favor of the fused op, and asserts the fused
        # output equals the reference.
        class MAdd(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying quantized add of two tensors (scale 1.0, zero
            # point 0) then relu.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns relu of the quantized add of x and y.
                a = torch.ops.quantized.add(x, y, 1.0, 0)
                relu_out = torch.relu(a)
                return relu_out

        A = torch.arange(-128, 130, dtype=torch.float)
        B = torch.arange(-128, 130, dtype=torch.float)
        scale = 2.0
        zero_point = 127
        qA = torch.quantize_per_tensor(
            A, scale=scale, zero_point=zero_point, dtype=torch.quint8
        )
        qB = torch.quantize_per_tensor(
            B, scale=scale, zero_point=zero_point, dtype=torch.quint8
        )

        # Check quantized add + relu fusion
        m = MAdd()
        scripted_m = torch.jit.script(m)
        ref_output = scripted_m(qA, qB)

        # Must inline the graph.
        # In this test case since we are directly calling ops
        # it does not matter, however if we are calling nn
        # modules we have to inline graph.
        torch._C._jit_pass_inline(scripted_m.graph)
        torch._C._jit_pass_fuse_quantized_add_relu(scripted_m.graph)
        FileCheck().check_not("aten::relu").check("quantized::add_relu").run(
            scripted_m.graph
        )
        output = scripted_m(qA, qB)
        self.assertEqual(ref_output, output)

        class MAddOut(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying the quantized out-variant add writing into z,
            # then relu.
            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Returns relu of the quantized add_out result of x, y
                # and z.
                a = torch.ops.quantized.add_out(x, y, z)
                relu_out = torch.relu(a)
                return relu_out

        qC = torch._empty_affine_quantized(
            qA.shape, scale=scale, zero_point=zero_point, dtype=torch.quint8
        )
        # Check quantized add + relu fusion
        m = MAddOut()
        scripted_m = torch.jit.script(m)
        ref_output = scripted_m(qA, qB, qC)
        # Must inline the graph.
        # In this test case since we are directly calling ops
        # it does not matter, however if we are calling nn
        # modules we have to inline graph.
        torch._C._jit_pass_inline(scripted_m.graph)
        torch._C._jit_pass_fuse_quantized_add_relu(scripted_m.graph)
        FileCheck().check_not("aten::relu").check_not("quantized::add_out").check(
            "quantized::add_relu_out"
        ).run(scripted_m.graph)
        output = scripted_m(qA, qB, qC)
        self.assertEqual(ref_output, output)

        class MAddScalar(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying quantized add of a tensor and a float scalar,
            # then relu.
            def forward(self, x, y: float):
                # [FUNCTION DOC] HELPER: Returns relu of the quantized scalar add of x and the
                # float y.
                a = torch.ops.quantized.add_scalar(x, y)
                relu_out = torch.relu(a)
                return relu_out

        # Check quantized add + relu fusion
        m = MAddScalar()
        scripted_m = torch.jit.script(m)
        ref_output = scripted_m(qA, 3.0)
        torch._C._jit_pass_inline(scripted_m.graph)
        torch._C._jit_pass_fuse_quantized_add_relu(scripted_m.graph)
        FileCheck().check_not("aten::relu").check_not("quantized::add_scalar(").check(
            "quantized::add_scalar_relu"
        ).run(scripted_m.graph)
        output = scripted_m(qA, 3.0)
        self.assertEqual(ref_output, output)

        class MAddScalarOut(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying the quantized scalar add with an output
            # tensor, then relu.
            def forward(self, x, y: float, z):
                # [FUNCTION DOC] HELPER: Returns relu of the quantized scalar add_out result of
                # x, the float y and output z.
                a = torch.ops.quantized.add_scalar_out(x, y, z)
                relu_out = torch.relu(a)
                return relu_out

        qC = torch._empty_affine_quantized(
            qA.shape, scale=scale, zero_point=zero_point, dtype=torch.quint8
        )
        m = MAddScalarOut()
        scripted_m = torch.jit.script(m)
        ref_output = scripted_m(qA, 3.0, qC)
        torch._C._jit_pass_inline(scripted_m.graph)
        torch._C._jit_pass_fuse_quantized_add_relu(scripted_m.graph)
        FileCheck().check_not("aten::relu").check_not(
            "quantized::add_scalar_out"
        ).check("quantized::add_scalar_relu_out").run(scripted_m.graph)
        output = scripted_m(qA, 3.0, qC)
        self.assertEqual(ref_output, output)


if __name__ == "__main__":
    raise_on_run_directly("test/test_quantization.py")
