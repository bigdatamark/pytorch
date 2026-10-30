# Owner(s): ["module: functorch"]
# [MODULE DOC] Tests for the out_dtype higher-order op: tracing with make_fx, export, dynamo,
# error handling for invalid operands, and inductor decomposition to int_mm on CUDA. Import-time
# side effect: instantiates the device-generic test class for CUDA only.
import unittest

import torch
import torch._dynamo
import torch._inductor
import torch._inductor.decomposition
from torch._higher_order_ops.out_dtype import out_dtype
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    run_tests, TestCase, IS_WINDOWS, IS_FBCODE, IS_REMOTE_GPU
)
from torch.testing._internal.common_quantization import skipIfNoDynamoSupport
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import SM80OrLater, _get_torch_cuda_version
from torch.testing._internal.common_device_type import instantiate_device_type_tests


@unittest.skipIf(not torch._dynamo.is_dynamo_supported(), "dynamo isn't support")
class TestOutDtypeOp(TestCase):
    # [CLASS DOC] Device-independent tests of out_dtype; skipped when dynamo is unsupported.
    # Covers make_fx/export graphs, numerics versus manual casting, and
    # ValueError/RuntimeError cases.
    def test_out_dtype_make_fx(self):
        # [FUNCTION DOC] Traces a module calling out_dtype (int8 mm with int32 output)
        # via make_fx, also after functionalization; checks numerical equality, that the
        # generated code contains the higher-order op and aten mm, and inspects node
        # metadata dtypes.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding an int8 weight; forward applies
            # out_dtype to mm with int32 output.
            def __init__(self, weight):
                # [FUNCTION DOC] HELPER: Stores the given weight tensor as an
                # attribute after base init.
                super().__init__()
                self.weight = weight

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the int32-output matmul of the
                # input with the stored weight.
                return out_dtype(
                    torch.ops.aten.mm.default, torch.int32, x, self.weight
                )

        weight = torch.randint(-128, 127, (5, 5), dtype=torch.int8)
        m = M(weight)
        x = torch.randint(-128, 127, (5, 5), dtype=torch.int8)

        gm = make_fx(m)(x)
        self.assertTrue(torch.allclose(m(x), gm(x)))

        gm = make_fx(torch.func.functionalize(M(weight)))(x)
        self.assertTrue(torch.allclose(m(x), gm(x)))

        FileCheck().check("torch.ops.higher_order.out_dtype").check("aten.mm.default").run(gm.code)
        self.assertTrue(torch.allclose(m(x), gm(x)))
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target is out_dtype:
                # Result of this node should be int32
                self.assertTrue(node.meta["val"].dtype, torch.int32)
                # Argument of this node should be int8
                self.assertTrue(node.args[2].meta["val"].dtype, torch.int8)

    def test_out_dtype_op_functional(self):
        # [FUNCTION DOC] Exports the out_dtype module with strict export, checks the
        # exported graph code has the higher-order op and mm, that the exported module
        # matches eager, and node dtype metadata.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding an int8 weight; forward applies
            # out_dtype to mm with int32 output.
            def __init__(self, weight):
                # [FUNCTION DOC] HELPER: Stores the given weight tensor as an
                # attribute after base init.
                super().__init__()
                self.weight = weight

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the int32-output matmul of the
                # input with the stored weight.
                return out_dtype(
                    torch.ops.aten.mm.default, torch.int32, x, self.weight
                )

        weight = torch.randint(-128, 127, (5, 5), dtype=torch.int8)
        m = M(weight)
        x = torch.randint(-128, 127, (5, 5), dtype=torch.int8)
        ep = torch.export.export(m, (x,), strict=True)
        FileCheck().check("torch.ops.higher_order.out_dtype").check(
            "aten.mm.default"
        ).run(ep.graph_module.code)
        self.assertTrue(torch.allclose(m(x), ep.module()(x)))
        for node in ep.graph.nodes:
            if node.op == "call_function" and node.target is out_dtype:
                # Result of this node should be int32
                self.assertTrue(node.meta["val"].dtype, torch.int32)
                # Argument of this node should be int8
                self.assertTrue(node.args[2].meta["val"].dtype, torch.int8)

    def test_out_dtype_mm_numerical(self):
        # [FUNCTION DOC] Checks the traced out_dtype mm result equals a plain mm done on
        # inputs manually cast to int32.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding an int8 weight; forward applies
            # out_dtype to mm with int32 output.
            def __init__(self, weight):
                # [FUNCTION DOC] HELPER: Stores the given weight tensor as an
                # attribute after base init.
                super().__init__()
                self.weight = weight

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the int32-output matmul of the
                # input with the stored weight.
                return out_dtype(
                    torch.ops.aten.mm.default, torch.int32, x, self.weight
                )

        weight = torch.randint(-128, 127, (5, 5), dtype=torch.int8)
        m = M(weight)
        x = torch.randint(-128, 127, (5, 5), dtype=torch.int8)

        gm = make_fx(m)(x)

        x_casted = x.to(torch.int32)
        weight_casted = weight.to(torch.int32)
        numerical_res = torch.ops.aten.mm.default(x_casted, weight_casted)
        self.assertTrue(torch.allclose(numerical_res, gm(x)))

    def test_out_dtype_dynamo(self):
        # [FUNCTION DOC] Compiles a function using out_dtype on a scalar-multiply op
        # with the eager backend (full graph) and checks it matches the uncompiled
        # result.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Applies out_dtype to the scalar multiply
            # overload with int32 output.
            return out_dtype(
                torch.ops.aten.mul.Scalar, torch.int32, x, y
            )

        inp = (torch.randint(-128, 127, (5, 5), dtype=torch.int8), 3.0)

        compiled = torch.compile(f, backend="eager", fullgraph=True)
        self.assertTrue(torch.allclose(f(*inp), compiled(*inp)))

    def test_out_dtype_mul_scalar_numerical(self):
        # [FUNCTION DOC] Traces out_dtype on scalar multiply with make_fx and compares
        # to the op run on an int32-cast input.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Applies out_dtype to the scalar multiply
            # overload with int32 output.
            return out_dtype(
                torch.ops.aten.mul.Scalar, torch.int32, x, y
            )

        inp = (torch.randint(-128, 127, (5, 5), dtype=torch.int8), 3.0)

        gm = make_fx(f)(*inp)
        numerical_res = torch.ops.aten.mul.Scalar(inp[0].to(dtype=torch.int32), 3)
        self.assertTrue(torch.allclose(numerical_res, gm(*inp)))

    def test_out_dtype_non_functional(self):
        # [FUNCTION DOC] Exporting a module that uses out_dtype with an in-place add
        # overload must raise a ValueError saying the first argument must be a
        # functional operator.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward applies out_dtype to an in-
            # place add overload.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns out_dtype applied to the in-
                # place add with int32 output.
                return out_dtype(
                    torch.ops.aten.add_.Tensor, torch.int32, x, y
                )

        with self.assertRaisesRegex(ValueError, "out_dtype's first argument needs to be a functional operator"):
            _ = torch.export.export(
                M(),
                (
                    torch.randint(-128, 127, (5, 5), dtype=torch.int8),
                    torch.randint(-128, 127, (5, 5), dtype=torch.int8),
                ),
                strict=True,
            )

    def test_out_dtype_non_op_overload(self):
        # [FUNCTION DOC] Calling out_dtype with a plain python callable instead of an
        # operator overload must raise a ValueError stating the first argument must be
        # an OpOverload.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns out_dtype applied to the plain add
            # callable with int32 output.
            return out_dtype(
                torch.add, torch.int32, x, y
            )

        with self.assertRaisesRegex(ValueError, "out_dtype's first argument must be an OpOverload"):
            f(torch.randint(-128, 127, (5, 5), dtype=torch.int8), torch.randint(-128, 127, (5, 5), dtype=torch.int8))

    def test_out_dtype_no_autograd(self):
        # [FUNCTION DOC] Verifies that out_dtype on float inputs requiring grad runs
        # (with and without no_grad), but backward through the result raises a
        # RuntimeError about missing grad_fn.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns out_dtype applied to mm with int32
            # output.
            return out_dtype(
                torch.ops.aten.mm.default, torch.int32, x, y
            )

        inp = (torch.randn(5, 5, requires_grad=True), torch.randn(5, 5, requires_grad=True))
        # error is delayed
        f(*inp)

        with torch.no_grad():
            f(*inp)

        with self.assertRaisesRegex(RuntimeError, "does not require grad and does not have a grad_fn"):
            out = f(*inp)
            loss = out - torch.ones(out.shape)
            loss.backward()

    def test_out_dtype_wrong_output(self) -> None:
        # [FUNCTION DOC] out_dtype on ops that return multiple tensors or a list of
        # tensors must raise a ValueError saying only single-tensor-returning ops are
        # supported.
        def multiple_out(x):
            # [FUNCTION DOC] HELPER: Applies out_dtype to topk, which returns a
            # tuple of two tensors.
            return out_dtype(
                torch.ops.aten.topk.default, torch.int32, x, 5
            )

        inp = (torch.randn(10),)

        with self.assertRaisesRegex(ValueError, "out_dtype's can only apply to ops that return a single tensor"):
            multiple_out(*inp)

        def singleton_list_out(x):
            # [FUNCTION DOC] HELPER: Applies out_dtype to split_copy, which returns
            # a list of tensors.
            return out_dtype(
                torch.ops.aten.split_copy.Tensor, torch.int32, x, 10
            )

        with self.assertRaisesRegex(ValueError, "out_dtype's can only apply to ops that return a single tensor"):
            singleton_list_out(*inp)

@unittest.skipIf(not torch._dynamo.is_dynamo_supported(), "dynamo isn't supported")
class TestOutDtypeOpDevice(TestCase):
    # [CLASS DOC] Device-generic out_dtype tests, instantiated for CUDA only; skipped
    # without dynamo.
    @unittest.skipIf(IS_WINDOWS, "_int_mm unavailable")
    @unittest.skipIf(not SM80OrLater, "_int_mm unavailable")
    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "cublas runtime error")
    @unittest.skipIf(_get_torch_cuda_version() >= (11, 7), "_int_mm unavailable")
    @skipIfNoDynamoSupport
    def test_out_dtype_inductor_decomp(self, device) -> None:
        # [FUNCTION DOC] Compiles an out_dtype int8 mm with max-autotune and checks it
        # matches the reference int_mm result in both eager and compiled form. Skipped
        # on Windows, pre-SM80 GPUs, remote GPU fbcode, and CUDA versions at least 11.7.
        def func(x, w):
            # [FUNCTION DOC] HELPER: Returns out_dtype applied to mm with int32
            # output.
            return out_dtype(torch.ops.aten.mm.default, torch.int32, x, w)

        w = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device=device)
        x = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device=device)

        ref = torch._int_mm(x, w)
        test_out = func(x, w)
        func_comp = torch.compile(func, fullgraph=True, mode="max-autotune")
        test_out_c = func_comp(x, w)
        self.assertTrue(torch.allclose(ref, test_out))
        self.assertTrue(torch.allclose(ref, test_out_c))

    def test_out_dtype_inductor_decomp_trace(self, device) -> None:
        # [FUNCTION DOC] Traces out_dtype with inductor's decomposition table and
        # symbolic tracing, asserting the generated code is a single int_mm call.
        def func(x, w):
            # [FUNCTION DOC] HELPER: Returns out_dtype applied to mm with int32
            # output.
            return out_dtype(torch.ops.aten.mm.default, torch.int32, x, w)

        w = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device=device)
        x = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device=device)

        # Check that make_fx with inductor decomps produces _int_mm
        decomp_table = torch._inductor.decomposition.select_decomp_table()
        gm = make_fx(func, decomp_table, tracing_mode="symbolic")(x, w)
        self.assertExpectedInline(gm.code.strip(), """\
def forward(self, x_1, w_1):
    _int_mm = torch.ops.aten._int_mm.default(x_1, w_1);  x_1 = w_1 = None
    return _int_mm""")

    def test_out_dtype_int_mm_default_trace(self, device) -> None:
        # [FUNCTION DOC] Traces out_dtype with symbolic tracing and no decompositions,
        # asserting the generated code keeps the higher-order out_dtype call.
        def func(x, w):
            # [FUNCTION DOC] HELPER: Returns out_dtype applied to mm with int32
            # output.
            return out_dtype(torch.ops.aten.mm.default, torch.int32, x, w)

        w = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device=device)
        x = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device=device)

        # By default, out_dtype is preserved in the trace
        gm = make_fx(func, tracing_mode="symbolic")(x, w)
        self.assertExpectedInline(gm.code.strip(), """\
def forward(self, x_1, w_1):
    out_dtype = torch.ops.higher_order.out_dtype(torch.ops.aten.mm.default, torch.int32, x_1, w_1);  x_1 = w_1 = None
    return out_dtype""")


instantiate_device_type_tests(TestOutDtypeOpDevice, globals(), only_for="cuda")

if __name__ == '__main__':
    run_tests()
