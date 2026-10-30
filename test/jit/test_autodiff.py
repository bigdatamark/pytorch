# Owner(s): ["oncall: jit"]
# ruff: noqa: F841
# [MODULE DOC] TorchScript autodiff tests: gradient functions for lists containing undefined
# tensors, and consistency of requires_grad on scripted outputs with eager outputs when values are
# profiled twice, pass through ignored Python calls, or are computed under no-grad mode. Raises an
# error if run directly; run via the main jit test file.

from typing import List

import torch
from torch.testing._internal.common_utils import (
    raise_on_run_directly,
    skipIfTorchDynamo,
)
from torch.testing._internal.jit_utils import JitTestCase


@skipIfTorchDynamo()
class TestAutodiffJit(JitTestCase):
    def test_undefined_tensor_lists(self):
        def fn(tensor_list: List[torch.Tensor], add_tensor):
            cat = torch.cat(tensor_list, dim=1)
            r = torch.sin(cat + add_tensor)
            return r

        fn_s = torch.jit.script(fn)

        a = torch.rand((3, 6), requires_grad=True)
        b = torch.rand((3, 10), requires_grad=True)
        x = [a, b]
        y = torch.rand((3, 16), requires_grad=True)

        ret = fn_s(x, y)
        ret.sum().backward()
        ret = fn_s(x, y)
        ret.sum().backward()

        ret = fn_s(x, y)
        s = ret.sum()

        # backward_fn expects 2 inputs: (grad_output, current_grad_r)
        # current_grad_r is provided because we need to add this contribution
        # to grad_r when we return it.
        backward_fn = s.grad_fn.next_functions[0][0]

        # check behavior with defined tensor
        grad_out = torch.rand((3, 16))
        grad_inputs = backward_fn(grad_out, None)

        # expect 3 tensors: grad_y, grad_a, grad_b
        self.assertEqual(3, len(grad_inputs))
        for x in grad_inputs:
            self.assertTrue(isinstance(x, torch.Tensor))

        # now test with undefined grad_out
        grad_inputs = backward_fn(None, None)

        # expect all of them to be None
        self.assertEqual(3, len(grad_inputs))
        for x in grad_inputs:
            if x is not None:
                self.assertEqual(0, torch.max(torch.abs(x)).item())

    def test_requires_grad_outputs(self):
        # outputs should require_grad only if eager outputs would require_grad.
        def fn(a, b, c):
            return a.relu() + b.relu(), c.relu()

        a = torch.rand((10, 10), requires_grad=False)
        b = torch.rand((10, 10), requires_grad=False)
        c = torch.rand((10, 10), requires_grad=True)

        fn_s = torch.jit.script(fn)

        for _ in range(4):
            x, y = fn_s(a, b, c)
            self.assertFalse(x.requires_grad)
            self.assertTrue(y.requires_grad)

    def test_requires_grad_outputs_profiled_twice(self):
        # the value "r" is used twice, by gammaln and by entr, so it is profiled twice.
        # So during autodiff graph formation the profile nodes are unmerged because
        # they are aliasing. Then the DifferentiableGraph doesn't have a profile
        # node on the output. The requires_grad info should then be added onto the
        # [CLASS DOC] JIT test case for autodiff behavior, skipped under Dynamo.
        # output value (otherwise autodiff will make the output require_grad).
        # [FUNCTION DOC] Scripts a function that concatenates a list of tensors, adds a tensor and
        # applies sine; runs it with grad-requiring inputs and backpropagates a few times so
        # autodiff is applied. Takes the backward function of the last result and checks it
        # returns three gradients for a defined output gradient, and that with an undefined output
        # gradient every returned gradient is either absent or all zeros.
        # Note: this relies on gammaln and entr not having autodiff implementations.
        # [FUNCTION DOC] HELPER: Concatenates the tensors in the list along dimension one, adds
        # the second argument and returns the sine.
        def fn(a, b, c):
            r = a.relu().relu()
            return torch.special.gammaln(r), torch.special.entr(r), c.cos().relu()

        fn_s = torch.jit.script(fn)

        a = torch.rand((10, 10), requires_grad=False)
        b = torch.rand((10, 10), requires_grad=False)
        c = torch.rand((10, 10), requires_grad=True)

        for _ in range(4):
            x_s, y_s, z_s = fn_s(a, b, c)
            x, y, z = fn(a, b, c)

            self.assertEqual(x_s.requires_grad, x.requires_grad)
            self.assertEqual(y_s.requires_grad, y.requires_grad)
            self.assertEqual(z_s.requires_grad, z.requires_grad)

    def test_requires_grad_outputs_side_effects(self):
        # same as above, but also add a CallFunction in between.
        @torch.jit.ignore
        def python_fn(x):
            return x.relu()

        def fn(a, b, c):
            r = a.relu().relu()
            z = python_fn(r)
            return torch.relu(r), torch.nn.functional.gelu(r), c.cos().relu()

        fn_s = torch.jit.script(fn)

        a = torch.rand((10, 10), requires_grad=False)
        b = torch.rand((10, 10), requires_grad=False)
        c = torch.rand((10, 10), requires_grad=True)

        for _ in range(4):
            x_s, y_s, z_s = fn_s(a, b, c)
            x, y, z = fn(a, b, c)

            self.assertEqual(x_s.requires_grad, x.requires_grad)
            self.assertEqual(y_s.requires_grad, y.requires_grad)
            self.assertEqual(z_s.requires_grad, z.requires_grad)

    # [FUNCTION DOC] Runs a scripted function repeatedly on two non-grad inputs and one grad
    # input. Checks that the first output (from the non-grad inputs) does not require grad and the
    # second does, matching eager semantics.
    def test_autodiff_requires_grad_nograd(self):
        @torch.jit.ignore
        # [FUNCTION DOC] HELPER: Returns the sum of relu applied to the first two arguments, and
        # relu of the third.
        def python_fn(x):
            return x.relu()

        def fn(a, b, c):
            x = a.sin().relu()
            y = python_fn(b)
            with torch.no_grad():
                z = x + c
            return x, y, z

        fn_s = torch.jit.script(fn)

        a = torch.rand((10, 10), requires_grad=True)
        b = torch.rand((10, 10), requires_grad=True)
        # [FUNCTION DOC] Checks requires_grad of scripted outputs against eager outputs, over four
        # runs, for a function where an intermediate value is used by two operations that lack
        # autodiff formulas, so it is profiled twice. The comment in the test explains that the
        # differentiable graph then lacks a profile node on the output and requires_grad must be
        # added to the value.
        c = torch.rand((10, 10), requires_grad=True)

        for _ in range(4):
            x_s, y_s, z_s = fn_s(a, b, c)
            x, y, z = fn(a, b, c)

            self.assertEqual(x_s.requires_grad, x.requires_grad)
            # [FUNCTION DOC] HELPER: Applies relu twice, returns the log-gamma and the entropy of
            # that result, and a cosine followed by relu of the third argument.
            self.assertEqual(y_s.requires_grad, y.requires_grad)
            self.assertEqual(z_s.requires_grad, z.requires_grad)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
# [FUNCTION DOC] HELPER: Returns sine-relu of the first argument, the ignored function applied to
# the second, and the first result plus the third computed under no-grad mode.
# [FUNCTION DOC] HELPER: Python function, excluded from scripting, returning relu of x.
# [FUNCTION DOC] Compares requires_grad of scripted and eager outputs over four runs, for a
# function with an ignored Python call and a computation done under no-grad mode, with all three
# inputs requiring grad.
# [FUNCTION DOC] HELPER: Calls the ignored Python function on an intermediate value and discards
# the result; returns relu of that intermediate, a gelu of it and a cosine plus relu of the third
# argument.
# [FUNCTION DOC] HELPER: Python function, excluded from scripting, returning relu of x.
# [FUNCTION DOC] Same comparison as the profiled-twice test but with a call to an ignored Python
# function in the middle of the graph. Compares requires_grad of each scripted output with eager
# over four runs.
