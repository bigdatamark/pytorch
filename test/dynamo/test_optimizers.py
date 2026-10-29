# Owner(s): ["module: dynamo"]
"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_adam in OptimizerTests)
"""

# [MODULE DOC] End-to-end tests of torch.compile with optimizers: compiling a training step
# that updates a tensor requiring grad, compiling a closure that builds an Adagrad optimizer,
# and compiling the step method of a small custom optimizer. Defines MyOptimizer and one
# test class. No import-time side effects beyond imports.

import functools

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
from torch.nn import Parameter


class MyOptimizer(torch.optim.Optimizer):
    # [CLASS DOC] HELPER: Minimal optimizer with empty defaults. Its step adds one in place
    # to the first parameter of each group, or subtracts one if any parameter in the group
    # is complex. Holds no extra state.
    def __init__(self, params):
        # [FUNCTION DOC] HELPER: Initializes the base Optimizer with params and no defaults.
        super().__init__(params, {})

    def _init_group(self, params, group):
        # [FUNCTION DOC] HELPER: Appends every parameter of the group to the given list
        # (mutated) and returns True if any parameter is complex.
        any_complex = False
        for p in group["params"]:
            params.append(p)
            any_complex |= p.is_complex()
        return any_complex

    def step(self):
        # [FUNCTION DOC] HELPER: For each param group, collects parameters and modifies the
        # first one in place: subtract 1 if any is complex, otherwise add 1. Takes no closure.
        for group in self.param_groups:
            params = []
            any_complex = self._init_group(params, group)
            if any_complex:
                params[0] -= 1
            else:
                params[0] += 1


class End2EndTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Compile-with-optimizer regression tests. Stateless.
    # https://github.com/pytorch/torchdynamo/issues/1604
    def test_optimizing_over_tensor_with_requires_grad(self):
        # [FUNCTION DOC] Compiles a full training iteration (zero_grad, forward, loss,
        # backward, optimizer step) with a counting backend, running it twice with Adam over
        # an input tensor that requires grad, and checks the frame count is 2.
        class Net(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward does a batched matmul of x and y and
            # flattens from dimension 1.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns flatten of bmm(x, y) starting at dim 1.
                z = torch.bmm(x, y)
                z = torch.flatten(z, 1)
                return z

        def training_iter_fn(batch, model, optimizer):
            # [FUNCTION DOC] HELPER: One training step: zero grads, run model on the batch
            # mapping, cross-entropy loss against fixed targets, backward, optimizer step;
            # returns the loss.
            optimizer.zero_grad()
            out = model(**batch)
            target = torch.tensor([0, 7])
            loss = torch.nn.CrossEntropyLoss()(out, target)
            loss.backward()
            optimizer.step()
            return loss

        net = Net()
        input1 = torch.randn(2, 1, 4)
        input2 = torch.randn(2, 4, 8, requires_grad=True)
        optimizer = torch.optim.Adam([input2], lr=0.1)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_training_iter_fn = torch.compile(training_iter_fn, backend=cnts)
        batch = {"x": input1, "y": input2}
        for _ in range(2):
            opt_training_iter_fn(batch, net, optimizer)
        self.assertEqual(cnts.frame_count, 2)

    def test_state_dict(self):
        # [FUNCTION DOC] Compiles (eager backend) a function that constructs an Adagrad
        # optimizer plus a closure for it, then runs optimizer.step with that closure
        # outside compilation. Passes if nothing raises.
        @torch.compile(backend="eager")
        def _test_state_dict(weight, bias, input):
            # [FUNCTION DOC] HELPER: Compiled; creates an Adagrad optimizer over weight
            # and bias and a partial-bound loss closure, returning both.
            def fn_base(optimizer, weight, bias):
                # [FUNCTION DOC] HELPER: Zeroes grads, computes a squared linear-output sum
                # loss from the enclosing input, runs backward and returns the loss.
                optimizer.zero_grad()
                i = input
                loss = (weight.mv(i) + bias).pow(2).sum()
                loss.backward()
                return loss

            optimizer = torch.optim.Adagrad([weight, bias])
            fn = functools.partial(fn_base, optimizer, weight, bias)
            return optimizer, fn

        optimizer, fn = _test_state_dict(
            Parameter(torch.randn(10, 5)),
            Parameter(torch.randn(10)),
            torch.randn(5, requires_grad=True),
        )
        optimizer.step(fn)

    def test_init_group(self):
        # [FUNCTION DOC] For float32 and complex64, runs MyOptimizer.step eagerly on one
        # copy of a random parameter and compiled (eager backend, fullgraph) on another,
        # then checks the results are equal.
        for dtype in [torch.float32, torch.cfloat]:
            tensor = torch.randn(5, 5, dtype=dtype)
            params = Parameter(tensor.detach().clone(), requires_grad=False)
            opt_params = Parameter(tensor.detach().clone(), requires_grad=False)

            optim = MyOptimizer([params])
            optim.step()

            opt_optim = MyOptimizer([opt_params])
            opt_step = torch.compile(backend="eager", fullgraph=True)(opt_optim.step)
            opt_step()

            self.assertEqual(params, opt_params)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
