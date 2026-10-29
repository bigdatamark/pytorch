# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for the pre_dispatch_eager dynamo backend. Each test compiles a small
# function using grad-mode or autocast context managers with that backend and compares forward
# outputs and input gradients against plain eager execution. No import-time side effects besides
# imports.
import torch
import torch._dynamo
import torch._dynamo.test_case


class PreDispatchTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case (dynamo TestCase) holding comparison tests between eager and
    # torch.compile with the pre_dispatch_eager backend for grad-mode and autocast regions.
    def test_no_grad_simple(self):
        # [FUNCTION DOC] Compiles a function that runs a sine, a cosine under no_grad, then
        # multiplies; checks the compiled output and the gradient of the input match eager.
        def f(a):
            # [FUNCTION DOC] HELPER: Function under test: sine of the input, cosine of it
            # computed inside a no_grad region, returns the product of the first value and the
            # sine of the second.
            b = a.sin()
            with torch.no_grad():
                c = b.cos()
            return b * c.sin()

        f_compiled = torch.compile(f, backend="pre_dispatch_eager")

        a_ref = torch.randn(4, requires_grad=True)
        a_test = a_ref.detach().clone().requires_grad_(True)

        out_ref = f(a_ref)
        out_test = f_compiled(a_test)
        self.assertEqual(out_ref, out_test)

        out_ref.sum().backward()
        out_test.sum().backward()
        self.assertEqual(a_ref.grad, a_test.grad)

    def test_enable_grad_and_no_grad(self):
        # [FUNCTION DOC] Compiles a function that nests an enable_grad region inside a no_grad
        # region and checks forward output and input gradient match eager execution.
        def f(a):
            # [FUNCTION DOC] HELPER: Function under test: scales the input, performs further
            # multiplications inside no_grad with a nested enable_grad block, and returns the sum
            # of all intermediates.
            b = a * 2
            with torch.no_grad():
                c = b * 3
                with torch.enable_grad():
                    d = c * 4
                e = d * 5
            return b + c + d + e

        f_compiled = torch.compile(f, backend="pre_dispatch_eager")

        a_ref = torch.randn(4, requires_grad=True)
        a_test = a_ref.detach().clone().requires_grad_(True)

        out_ref = f(a_ref)
        out_test = f_compiled(a_test)
        self.assertEqual(out_ref, out_test)

        out_ref.sum().backward()
        out_test.sum().backward()
        self.assertEqual(a_ref.grad, a_test.grad)

    def test_autocast_simple(self):
        # [FUNCTION DOC] Compiles a function using a CPU autocast region around a matmul and
        # checks the output and the input gradient match eager execution.
        def f(a):
            # [FUNCTION DOC] HELPER: Function under test: doubles the input, runs a matrix
            # multiply under CPU autocast, returns the sum of the doubled input and the product.
            b = a * 2
            with torch.amp.autocast(device_type="cpu"):
                c = torch.matmul(b, b)
            return b + c

        f_compiled = torch.compile(f, backend="pre_dispatch_eager")

        a_ref = torch.randn(4, device="cpu", requires_grad=True)
        a_test = a_ref.detach().clone().requires_grad_(True)

        out_ref = f(a_ref)
        out_test = f_compiled(a_test)
        self.assertEqual(out_ref, out_test)

        out_ref.sum().backward()
        out_test.sum().backward()
        self.assertEqual(a_ref.grad, a_test.grad)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
