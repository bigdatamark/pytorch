# Owner(s): ["NNC"]

# [MODULE DOC] Tests for the Python bindings of the tensor expression (NNC) compiler in
# torch._C._te: building buffers and loop nests by hand, running them with the interpreter or
# LLVM codegen, compiling TorchScript IR graphs through TensorExprKernel (including shape
# annotation and custom lowerings), and the unary math functions on expression handles. Import
# time: records in LLVM_ENABLED whether the LLVM backend is built in; tests needing it are
# skipped otherwise.
import torch
import numpy as np
import torch._C._te as te

from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.jit_utils import JitTestCase
import unittest

LLVM_ENABLED = torch._C._llvm_enabled()


def construct_adder(n: int, dtype=torch.float32):
    # [FUNCTION DOC] HELPER: Builds an interpreter codegen object computing the elementwise sum
    # of two one-dimensional buffers named A and B of length n and element type dtype (default
    # float32) into an output buffer C. Lowers the compute to a loop nest, prepares it for
    # codegen, simplifies the root statement and returns the ir_eval codegen taking the buffers
    # A, B and C in that order.
    A = te.BufHandle("A", [n], dtype)
    B = te.BufHandle("B", [n], dtype)

    def compute(i):
        # [FUNCTION DOC] HELPER: Compute body for the output element at index i; returns the
        # sum of loads of A and B at that index.
        return A.load([i]) + B.load([i])

    C = te.Compute("C", [n], compute)

    loopnest = te.LoopNest([C])
    loopnest.prepare_for_codegen()
    stmt = te.simplify(loopnest.root_stmt())

    return te.construct_codegen("ir_eval", stmt, [A, B, C])


class TestTensorExprPyBind(JitTestCase):
    # [CLASS DOC] Tests of low-level tensor expression bindings: codegen calls, external calls,
    # dynamic shapes, dtype validation and TensorExprKernel compilation of parsed graphs,
    # compared against eager results.
    def test_simple_sum(self):
        # [FUNCTION DOC] Builds the adder for 32 elements, runs it on two random tensors with
        # the call method and checks the output tensor equals their sum.
        n = 32
        cg = construct_adder(n)

        tA = torch.randn(n)
        tB = torch.randn(n)
        tC = torch.empty(n)
        cg.call([tA, tB, tC])
        torch.testing.assert_close(tA + tB, tC)

    def test_call_raw(self):
        # [FUNCTION DOC] Builds a float64 adder for 16 elements and runs it with the raw call
        # form that takes data pointers instead of tensors; checks the output equals the sum.
        n = 16
        cg = construct_adder(n, dtype=torch.float64)

        tA = torch.randn(n, dtype=torch.float64)
        tB = torch.randn(n, dtype=torch.float64)
        tC = torch.empty(n, dtype=torch.float64)
        cg.call_raw([tA.data_ptr(), tB.data_ptr(), tC.data_ptr()])
        torch.testing.assert_close(tA + tB, tC)

    def test_external_calls(self):
        # [FUNCTION DOC] Builds an external call node to the ATen matmul entry point on a 1x4
        # and a 4x1 buffer producing 1x1, lowers it in a loop nest and runs it through the
        # interpreter with ones inputs; checks the result equals torch matmul.
        dtype = torch.float32

        A = te.BufHandle("A", [1, 4], dtype)
        B = te.BufHandle("B", [4, 1], dtype)
        C = te.BufHandle("C", [1, 1], dtype)

        s = te.ExternalCall(C, "nnc_aten_matmul", [A, B], [])

        loopnest = te.LoopNest(s, [C])
        loopnest.prepare_for_codegen()
        codegen = te.construct_codegen("ir_eval", s, [A, B, C])

        tA = torch.ones(1, 4)
        tB = torch.ones(4, 1)
        tC = torch.empty(1, 1)
        codegen.call([tA, tB, tC])
        torch.testing.assert_close(torch.matmul(tA, tB), tC)

    def test_dynamic_shape(self):
        # [FUNCTION DOC] Builds a subtraction kernel whose length is a symbolic int32 variable
        # passed as an extra codegen argument, then runs it for lengths 8 and 31 through the
        # nested helper and checks the elementwise difference each time.
        dN = te.VarHandle(torch.int32)
        A = te.BufHandle([dN], torch.float64)
        B = te.BufHandle([dN], torch.float64)

        def compute(i):
            # [FUNCTION DOC] HELPER: Compute body returning the difference of the loads of A
            # and B at index i.
            return A.load(i) - B.load(i)

        C = te.Compute("C", [dN], compute)

        loopnest = te.LoopNest([C])
        loopnest.prepare_for_codegen()

        cg = te.construct_codegen("ir_eval", loopnest.simplify(), [A, B, C, dN])

        def test_with_shape(n):
            # [FUNCTION DOC] HELPER: Runs the compiled kernel on random double tensors of
            # length n, passing n as the dynamic size argument, and checks the output equals
            # the difference.
            tA = torch.randn(n, dtype=torch.double)
            tB = torch.randn(n, dtype=torch.double)
            tC = torch.empty(n, dtype=torch.double)
            cg.call([tA, tB, tC, n])
            torch.testing.assert_close(tA - tB, tC)

        test_with_shape(8)
        test_with_shape(31)

    def test_dynamic_shape_2d(self):
        # [FUNCTION DOC] Same as the dynamic shape test but with two symbolic dimensions (rows
        # and columns) over 2D buffers; runs for sizes 2x4 and 5x3 and checks the elementwise
        # difference.
        dN = te.VarHandle(torch.int32)
        dM = te.VarHandle(torch.int32)
        A = te.BufHandle([dN, dM], torch.float64)
        B = te.BufHandle([dN, dM], torch.float64)

        def compute(i, j):
            # [FUNCTION DOC] HELPER: Compute body returning the difference of the loads of A
            # and B at row i and column j.
            return A.load([i, j]) - B.load([i, j])

        C = te.Compute("C", [dN, dM], compute)

        loopnest = te.LoopNest([C])
        loopnest.prepare_for_codegen()

        cg = te.construct_codegen("ir_eval", loopnest.simplify(), [A, B, C, dN, dM])

        def test_with_shape(n, m):
            # [FUNCTION DOC] HELPER: Runs the compiled kernel on random double matrices of
            # shape n by m, passing both sizes as dynamic arguments, and checks the difference.
            tA = torch.randn(n, m, dtype=torch.double)
            tB = torch.randn(n, m, dtype=torch.double)
            tC = torch.empty(n, m, dtype=torch.double)
            cg.call([tA, tB, tC, n, m])
            torch.testing.assert_close(tA - tB, tC)

        test_with_shape(2, 4)
        test_with_shape(5, 3)

    def test_dtype_error(self):
        # [FUNCTION DOC] Checks that a buffer can be created with a valid dtype, and that
        # passing an invalid dtype string makes the constructor raise TypeError.
        te.BufHandle("a", [1], torch.float32)  # ok
        self.assertRaises(TypeError, lambda: te.BufHandle("a", [1], "float55"))

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_tensor_inputs(self):
        # [FUNCTION DOC] Skipped without LLVM. Parses a textual IR graph adding three 4x4 float
        # tensors, compiles it with TensorExprKernel, and checks that both the compiled run and
        # the fallback path match eager addition.
        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Eager reference computing the sum of its three inputs.
            return a + b + c

        device, size = "cpu", (4, 4)
        x = torch.rand(size, device=device)
        y = torch.rand(size, device=device)
        z = torch.rand(size, device=device)

        graph_str = """
graph(%a.1 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cpu),
      %b.1 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cpu),
      %c.1 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cpu)):
  %6 : int = prim::Constant[value=1]()
  %7 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cpu) = aten::add(%a.1, %b.1, %6)
  %3 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cpu) = aten::add(%7, %c.1, %6)
  return (%3)
        """
        graph = torch._C.parse_ir(graph_str)

        kernel = te.TensorExprKernel(graph)
        res1 = kernel.run((x, y, z))
        res2 = kernel.fallback((x, y, z))
        correct = f(x, y, z)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_scalar_inputs(self):
        # [FUNCTION DOC] Skipped without LLVM. Same as the tensor input test but with three
        # zero-dimension float tensors, using a graph typed for scalars-as-tensors.
        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Eager reference computing the sum of its three inputs.
            return a + b + c

        x = torch.tensor(0.1, dtype=torch.float, device="cpu")
        y = torch.tensor(0.6, dtype=torch.float, device="cpu")
        z = torch.tensor(0.7, dtype=torch.float, device="cpu")

        graph_str = """
graph(%a.1 : Float(requires_grad=0, device=cpu),
      %b.1 : Float(requires_grad=0, device=cpu),
      %c.1 : Float(requires_grad=0, device=cpu)):
  %3 : int = prim::Constant[value=1]()
  %6 : Float(requires_grad=0, device=cpu) = aten::add(%a.1, %b.1, %3)
  %9 : Float(requires_grad=0, device=cpu) = aten::add(%6, %c.1, %3)
  return (%9)
        """
        graph = torch._C.parse_ir(graph_str)

        kernel = te.TensorExprKernel(graph)
        res1 = kernel.run((x, y, z))
        res2 = kernel.fallback((x, y, z))
        correct = f(x, y, z)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_shape_prop(self):
        # [FUNCTION DOC] Skipped without LLVM. Parses a graph without shape info and checks
        # that TensorExprKernel raises RuntimeError. Then annotates the input shapes from
        # example tensors, runs shape propagation on the graph, compiles successfully and
        # checks the elementwise product against eager.
        device, size = "cpu", (4, 4)
        x = torch.rand(size, device=device)
        y = torch.rand(size, device=device)

        graph_str = """
graph(%a : Tensor, %b : Tensor):
  %c : Tensor = aten::mul(%a, %b)
  return (%c)
        """
        graph = torch._C.parse_ir(graph_str)

        exception_thrown = False
        try:
            kernel = te.TensorExprKernel(graph)
        except RuntimeError:
            # Graph doesn't have shape info for inputs => compilation should
            # fail
            exception_thrown = True
        if not exception_thrown:
            raise AssertionError("expected RuntimeError for missing shape info")

        # Inject shape info and try compiling again
        example_inputs = [torch.rand(4, 4), torch.rand(4, 4)]
        torch._C._te.annotate_input_shapes(graph, example_inputs)
        torch._C._jit_pass_propagate_shapes_on_graph(graph)

        # Now compilation should pass
        kernel = te.TensorExprKernel(graph)

        res = kernel.run((x, y))
        correct = torch.mul(x, y)
        np.testing.assert_allclose(res.numpy(), correct.numpy(), atol=1e-5)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_shape_prop_module(self):
        # [FUNCTION DOC] Skipped without LLVM. Scripts a small module and takes its graph.
        # Checks that compiling without shape info fails, that annotating input shapes fails
        # while the unused self argument is present, then removes that argument, annotates,
        # propagates shapes, compiles and compares the result with the module forward.
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Scripted module computing x times x plus y.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the square of x plus y.
                return x * x + y

        graph = torch.jit.script(TestModule()).graph

        # Try compiling the graph as-is. It should fail because it doesn't have
        # shape info.
        exception_thrown = False
        try:
            kernel = te.TensorExprKernel(graph)
        except RuntimeError:
            exception_thrown = True
        if not exception_thrown:
            raise AssertionError("expected RuntimeError for missing shape info")

        # Try injecting shape info for graph inputs
        example_inputs = [torch.rand(4, 4), torch.rand(4, 4)]

        exception_thrown = False
        try:
            torch._C._te.annotate_input_shapes(graph, example_inputs)
        except RuntimeError:
            # Graph has a 'self' argument for which we can't set shapes
            exception_thrown = True
        if not exception_thrown:
            raise AssertionError("expected RuntimeError for self argument")

        # Remove 'self' argument and try annotating shapes one more time
        torch._C._te.remove_unused_self_argument(graph)

        # Inject shape info and try compiling again
        torch._C._te.annotate_input_shapes(graph, example_inputs)
        torch._C._jit_pass_propagate_shapes_on_graph(graph)

        # Now compilation should pass
        kernel = te.TensorExprKernel(graph)

        device, size = "cpu", (4, 4)
        x = torch.rand(size, device=device)
        y = torch.rand(size, device=device)

        res = kernel.run((x, y))
        correct = TestModule().forward(x, y)
        np.testing.assert_allclose(res.numpy(), correct.numpy(), atol=1e-5)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_t(self):
        # [FUNCTION DOC] Skipped without LLVM. Compiles a parsed graph with a transpose-op on a
        # 3x4 tensor and checks both the compiled and fallback results against the eager
        # result.
        def f(a):
            # [FUNCTION DOC] HELPER: Eager reference returning the transpose of its input.
            return a.t()

        device, size = "cpu", (3, 4)
        x = torch.rand(size, device=device)

        graph_str = """
graph(%a.1 : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu)):
  %3 : Float(4, 3, strides=[4, 1], requires_grad=0, device=cpu) = aten::t(%a.1)
  return (%3)
        """
        graph = torch._C.parse_ir(graph_str)

        kernel = te.TensorExprKernel(graph)
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))
        correct = f(x)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_transpose(self):
        # [FUNCTION DOC] Skipped without LLVM. Compiles a graph transposing the last two
        # dimensions of a 3x4 tensor and checks compiled and fallback results against eager.
        def f(a):
            # [FUNCTION DOC] HELPER: Eager reference swapping the last two dimensions of its
            # input.
            return a.transpose(-1, -2)

        device, size = "cpu", (3, 4)
        x = torch.rand(size, device=device)

        graph_str = """
graph(%a.1 : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu)):
  %2 : int = prim::Constant[value=-1]()
  %3 : int = prim::Constant[value=-2]()
  %4 : Float(4, 3, strides=[4, 1], requires_grad=0, device=cpu) = aten::transpose(%a.1, %2, %3)
  return (%4)
        """
        graph = torch._C.parse_ir(graph_str)

        kernel = te.TensorExprKernel(graph)
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))
        correct = f(x)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_permute(self):
        # [FUNCTION DOC] Skipped without LLVM. Compiles a graph permuting a 3x4x5 tensor with
        # the order 2, 1, 0 and checks compiled and fallback results against eager.
        def f(a):
            # [FUNCTION DOC] HELPER: Eager reference permuting the dimensions of its input in
            # reverse order.
            return a.permute([2, 1, 0])

        device, size = "cpu", (3, 4, 5)
        x = torch.rand(size, device=device)

        graph_str = """
graph(%a.1 : Float(3, 4, 5, strides=[20, 5, 1], requires_grad=0, device=cpu)):
  %1 : int = prim::Constant[value=2]()
  %2 : int = prim::Constant[value=1]()
  %3 : int = prim::Constant[value=0]()
  %4 : int[] = prim::ListConstruct(%1, %2, %3)
  %5 : Float(5, 4, 3, strides=[12, 3, 1], requires_grad=0, device=cpu) = aten::permute(%a.1, %4)
  return (%5)
        """
        graph = torch._C.parse_ir(graph_str)

        kernel = te.TensorExprKernel(graph)
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))
        correct = f(x)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_custom_lowering(self):
        # [FUNCTION DOC] Skipped without LLVM. Builds a 2x2 tensor with NaN on the diagonal and
        # a graph using the nan_to_num op. Registers a custom lowering for that op when
        # creating the kernel and checks compiled and fallback results equal the eager
        # nan_to_num result.
        def f(a):
            # [FUNCTION DOC] HELPER: Eager reference replacing NaN values in its input.
            return a.nan_to_num()

        device = "cpu"
        x = torch.ones((2, 2), device=device)
        x[0, 0] = x[1, 1] = torch.nan
        graph_str = """
graph(%x : Float(2, 2, strides=[2, 1], requires_grad=0, device=cpu)):
    %none : NoneType = prim::Constant()
    %y : Float(2, 2, strides=[2, 1], requires_grad=0, device=cpu) = aten::nan_to_num(%x, %none, %none, %none)
    return (%y)
        """
        graph = torch._C.parse_ir(graph_str)

        def my_custom_lowering(inputs, out_shape, out_stride, out_type, device):
            # [FUNCTION DOC] HELPER: Custom lowering handed to TensorExprKernel for the
            # nan_to_num op. Receives input handles, output shape, stride, type and device (only
            # inputs and shape are used) and returns a computed tensor that outputs zero where
            # the loaded value is NaN and the loaded value otherwise.
            def compute(idxs):
                # [FUNCTION DOC] HELPER: Body for the custom lowering: loads the input element
                # at the given indices and returns a conditional expression yielding zero when
                # it is NaN, else itself.
                load = inputs[0].as_buf().load(idxs)
                return te.ifThenElse(
                    te.ExprHandle.isnan(load), te.ExprHandle.float(0.0), load
                )

            return te.Compute2("custom_nan_to_num", out_shape, compute)

        kernel = te.TensorExprKernel(graph, {"aten::nan_to_num": my_custom_lowering})
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))
        correct = f(x)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_kernel_with_expand(self):
        # [FUNCTION DOC] Skipped without LLVM. Compiles a graph expanding a 1x3x1 tensor to
        # 2x3x4 and checks compiled and fallback results against eager expand.
        def f(a):
            # [FUNCTION DOC] HELPER: Eager reference expanding its input to 2x3x4.
            return a.expand((2, 3, 4))

        device = "cpu"
        x = torch.rand((1, 3, 1), device=device)
        graph_str = """
graph(%a : Float(1, 3, 1, strides=[3, 1, 1], requires_grad=0, device=cpu)):
  %1 : int = prim::Constant[value=2]()
  %2 : int = prim::Constant[value=3]()
  %3 : int = prim::Constant[value=4]()
  %4 : int[] = prim::ListConstruct(%1, %2, %3)
  %5 : bool = prim::Constant[value=0]()
  %6 : Float(2, 3, 4, strides=[12, 4, 0], requires_grad=0, device=cpu) = aten::expand(%a, %4, %5)
  return (%6)
        """
        graph = torch._C.parse_ir(graph_str)

        kernel = te.TensorExprKernel(graph)
        res1 = kernel.run((x,))
        res2 = kernel.fallback((x,))
        correct = f(x)
        np.testing.assert_allclose(res1.numpy(), correct.numpy(), atol=2e-3)
        np.testing.assert_allclose(res2.numpy(), correct.numpy(), atol=2e-3)

    @unittest.skipIf(not LLVM_ENABLED, "LLVM backend not enabled")
    def test_alloc_in_loop(self):
        # [FUNCTION DOC] Skipped without LLVM. Builds a loop nest storing an element of a into
        # a temporary buffer and then into b, wrapped in four nested loops of 100 iterations,
        # lowers it with LLVM codegen and runs it on one-element ones tensors via raw pointers.
        # Checks that temporary allocation inside loops compiles and runs; no value is
        # asserted.
        a, tmp, b = (
            te.BufHandle(name, [1], torch.float32) for name in ["a", "tmp", "b"]
        )
        body = te.Block([tmp.store([0], a.load([0])), b.store([0], tmp.load([0]))])
        for _ in range(4):
            i = te.VarHandle("i", torch.int32)
            body = te.For.make(i, 0, 100, body)
        nest = te.LoopNest(body, [b])
        nest.prepare_for_codegen()
        f = te.construct_codegen("llvm", nest.simplify(), [a, b])
        ta, tb = (torch.ones(1) for _ in range(2))
        f.call([ta.data_ptr(), tb.data_ptr()])


class TestExprHandlePyBind(JitTestCase):
    # [CLASS DOC] Tests for the free math functions applied to expression handles.
    def test_unary_ops(self):
        # [FUNCTION DOC] For each pair of a torch unary function and its tensor expression
        # counterpart (trigonometric, exponential, logarithmic, rounding, erf, gamma and
        # others), builds an interpreter kernel for ten random floats and checks the result
        # matches the torch reference within 1e-3 tolerance; raises AssertionError on mismatch.
        unary_operators = {
            torch.sin: torch._C._te.sin,
            torch.cos: torch._C._te.cos,
            torch.tan: torch._C._te.tan,
            torch.asin: torch._C._te.asin,
            torch.acos: torch._C._te.acos,
            torch.atan: torch._C._te.atan,
            torch.sinh: torch._C._te.sinh,
            torch.cosh: torch._C._te.cosh,
            torch.tanh: torch._C._te.tanh,
            torch.sigmoid: torch._C._te.sigmoid,
            torch.exp: torch._C._te.exp,
            torch.expm1: torch._C._te.expm1,
            torch.abs: torch._C._te.abs,
            torch.log: torch._C._te.log,
            torch.log2: torch._C._te.log2,
            torch.log10: torch._C._te.log10,
            torch.log1p: torch._C._te.log1p,
            torch.erf: torch._C._te.erf,
            torch.erfc: torch._C._te.erfc,
            torch.sqrt: torch._C._te.sqrt,
            torch.rsqrt: torch._C._te.rsqrt,
            torch.ceil: torch._C._te.ceil,
            torch.floor: torch._C._te.floor,
            torch.round: torch._C._te.round,
            torch.trunc: torch._C._te.trunc,
            torch.lgamma: torch._C._te.lgamma,
            torch.frac: torch._C._te.frac,
        }

        def construct_te_fn(op, n: int, dtype=torch.float32):
            # [FUNCTION DOC] HELPER: Builds an ir_eval codegen applying the given expression
            # function to each element of a one-dimensional buffer of length n and dtype dtype,
            # taking the input and output buffers.
            A = torch._C._te.BufHandle("A", [n], dtype)

            def compute(i):
                # [FUNCTION DOC] HELPER: Compute body applying the given operation to the load
                # at index i.
                return op(A.load([i]))

            C = te.Compute("C", [n], compute)

            loopnest = te.LoopNest([C])
            loopnest.prepare_for_codegen()
            stmt = te.simplify(loopnest.root_stmt())

            return te.construct_codegen("ir_eval", stmt, [A, C])

        n = 10
        a = torch.rand(n)
        for torch_op, te_op in unary_operators.items():
            ref = torch_op(a)

            te_fn = construct_te_fn(te_op, n, torch.float32)
            res = torch.empty(n)
            te_fn.call([a, res])
            if not torch.allclose(ref, res, atol=1e-3, rtol=1e-3):
                raise AssertionError("tensor expression result does not match reference")


if __name__ == "__main__":
    run_tests()
