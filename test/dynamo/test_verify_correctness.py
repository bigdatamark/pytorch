# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests of the verify-correctness option: when enabled, outputs of a compiled
#   graph are checked against eager outputs and an error is raised on mismatch. Defines small
#   modules (a sequential network and a conv-batchnorm-relu block), a toy function, and a
#   transform that wrongly swaps multiplication nodes for additions to simulate a buggy backend.
#   No import-time side effects.
import operator

import torch
import torch._dynamo
import torch._dynamo.config as config
import torch._dynamo.test_case
from torch._dynamo.testing import same
from torch.fx._lazy_graph_module import _force_skip_lazy_graph_module


class Seq(torch.nn.Module):
    # [CLASS DOC] HELPER: Module wrapping a sequential stack of linear, relu, linear and sigmoid
    #   layers, all of width 10.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Build the sequential layers attribute.
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Run the sequential layers on the input and return the result.
        return self.layers(x)


class Conv_Bn_Relu(torch.nn.Module):
    # [CLASS DOC] HELPER: Module with a bias-free convolution, batch norm (eps 0.001) and relu.
    #   Extra keyword arguments go to the convolution. Not used by the tests in this file.
    def __init__(self, in_channels, out_channels, **kwargs):
        # [FUNCTION DOC] HELPER: Build the convolution (no bias, extra keyword args forwarded),
        #   batch norm and relu submodules.
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Apply convolution, batch norm and relu in order and return the
        #   result.
        return self.relu(self.bn(self.conv(x)))


def toy_example(a, b):
    # [FUNCTION DOC] HELPER: Toy function used as a compile target: normalize a by one plus its
    #   absolute value, negate b if its sum is negative, and return the product of the two.
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b


def transform(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # [FUNCTION DOC] HELPER: Deliberately incorrect graph transformation. Walks the nodes of the
    #   given graph module and replaces every multiplication function call with addition, lints
    #   the graph, recompiles the module, and returns it. Mutates the module in place.
    for node in gm.graph.nodes:
        # Checks if we're calling a function (i.e:
        # operator.add)
        if node.op == "call_function":
            # The target attribute is the function
            # that call_function calls.
            if node.target == operator.mul:
                node.target = operator.add

    gm.graph.lint()  # Does some checks to make sure the
    # Graph is well-formed.

    gm.recompile()
    return gm


@config.patch("verify_correctness", True)
class TestVerifyCorrectness(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case run with the verify-correctness config enabled for the class,
    #   checking when wrong compiled outputs are detected.
    def test_example_inputs(self):
        # [FUNCTION DOC] With a custom backend asserting on example inputs (including a tuple
        #   argument and a Python number), checks the backend sees inputs that yield an output
        #   with the same shape and device as eager and as the compiled call.
        def fn(a, bc, d):
            # [FUNCTION DOC] HELPER: Function under test: a divided by d minus the ratio of the
            #   two elements of a pair.
            b, c = bc
            return a / d - b / c

        def compiler_fn(graph, example_inputs):
            # [FUNCTION DOC] HELPER: Backend: runs the captured graph on the example inputs,
            #   saves the first output in an enclosing variable, and returns the graph forward
            #   function.
            nonlocal r1
            r1 = graph(*example_inputs)[0]
            return graph.forward

        a = torch.empty(2).fill_(1)
        b = torch.empty(2).fill_(2)
        c = torch.empty(2).fill_(3)
        d = 4
        r1 = None
        r2 = fn(a, (b, c), d)
        opt_fn = torch._dynamo.optimize_assert(compiler_fn)(fn)
        r3 = opt_fn(a, (b, c), d)

        self.assertIsNotNone(r1)

        self.assertEqual(r1.shape, r2.shape)
        self.assertEqual(r1.shape, r3.shape)
        self.assertEqual(r1.device, r2.device)
        self.assertEqual(r1.device, r3.device)

    @_force_skip_lazy_graph_module()
    def test_torchscript(self):
        # [FUNCTION DOC] Compiling a sequential module with the torchscript backend (lazy graph
        #   module disabled by decorator) gives output equal to eager.
        s = Seq()
        i = torch.randn(10)
        r1 = s(i)
        opt_s = torch.compile(s, backend="ts")
        r2 = opt_s(i)
        self.assertTrue(same(r1, r2))

    def test_incorrect_verify_true(self):
        # [FUNCTION DOC] With verification enabled, a backend returning a graph that is not
        #   equivalent to the original (multiplications changed to additions) must cause a
        #   runtime error when the compiled function runs; the test fails if no error occurs.
        """
        If a bad optimization return a graph that
        is not functionally equal to the original graph;
        When config.verify_correctness=True, it will
        check the correctness of outputs and raise an error
        """
        i1 = torch.randn(10)
        i2 = torch.randn(10)

        def incorrect_compile_fn(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Backend returning the forward function of the deliberately
            #   wrong transformed graph.
            return transform(gm).forward

        toy_example(i1, i2)
        try:
            opt_toy_example = torch.compile(toy_example, backend=incorrect_compile_fn)
            opt_toy_example(i1, i2)
        except RuntimeError:
            pass
        else:
            self.fail("expected failure")

    @config.patch("verify_correctness", False)
    def test_incorrect_verify_false(self):
        # [FUNCTION DOC] With verification disabled for this test, the same incorrect backend
        #   runs without error and its output differs from eager.
        """
        The bad optimization return a graph that
        is not functionally equal to the original graph;
        When config.verify_correctness=False, wrong outputs
        will return
        """
        i1 = torch.randn(10)
        i2 = torch.randn(10)

        def incorrect_compile_fn(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Backend returning the forward function of the deliberately
            #   wrong transformed graph.
            return transform(gm).forward

        r1 = toy_example(i1, i2)
        opt_toy_example = torch.compile(toy_example, backend=incorrect_compile_fn)
        r2 = opt_toy_example(i1, i2)
        self.assertTrue(not same(r1, r2))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
