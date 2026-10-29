# Owner(s): ["oncall: fx"]
# [MODULE DOC] Tests for the FX common-pass infrastructure, currently the common
# subexpression elimination pass. Defines small traceable functions (including ones that
# mutate tensors or inputs, call factory functions, or return lists), a list of passes, case
# lists and the available devices (cpu, plus cuda when available). Parametrized tests trace
# each function with make_fx, run the pass and compare the transformed module's output with
# the original function. Running the file directly raises an error pointing to the main FX
# test file.

import itertools

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.graph_module import GraphModule
from torch.fx.passes.dialect.common.cse_pass import CSEPass
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    raise_on_run_directly,
    TestCase,
)


def FactoryFunctionCall(x, device):
    # [FUNCTION DOC] HELPER: Test function. Creates a tensor of the same shape as the input
    # filled with the constant 3 on the given device, adds it to the input and returns the
    # sum.
    y = torch.full(x.shape, 3, device=device)
    z = torch.add(y, x)
    return z


def TorchTensorCall(x):
    # [FUNCTION DOC] HELPER: Test function. Creates a scalar tensor holding 3 with the torch
    # tensor constructor and returns the input plus it.
    y = torch.tensor(3)
    return x + y


def TakeList(x):
    # [FUNCTION DOC] HELPER: Test function. Concatenates a list containing the input twice
    # and returns the result, so the traced op takes a list argument.
    z = torch.cat([x, x])
    return z


def ReturnList(x):
    # [FUNCTION DOC] HELPER: Test function. Ignores its input; builds a ten element range
    # reshaped to five rows by two columns and splits it along the first dimension into
    # pieces of sizes 1 and 4. Returns the tuple of pieces, so the traced op returns a list.
    a = torch.arange(10).reshape(5, 2)
    z = torch.split(a, [1, 4])
    return z


def Mutation(x):
    # [FUNCTION DOC] HELPER: Test function. Adds 2 to the input, mutates that result in
    # place by adding 1, and returns the input plus the mutated value. The input is not
    # mutated.
    y = x + 2
    y.add_(1)
    return x + y


def MutationInput(x):
    # [FUNCTION DOC] HELPER: Test function. Mutates the input in place by adding 1, then
    # returns the input plus the input plus 2. Side effect: modifies its argument.
    x.add_(1)
    y = x + 2
    return x + y


def MutationFactory(x, device):
    # [FUNCTION DOC] HELPER: Test function. Builds a constant 3 tensor shaped like the input
    # on the given device, mutates it in place by adding 1 and returns the input plus it.
    y = torch.full(x.shape, 3, device=device)
    y.add_(1)
    return x + y


def MutationTorchTensorCall(x):
    # [FUNCTION DOC] HELPER: Test function. Builds a scalar tensor 3, mutates it in place by
    # adding 1 and returns the input plus it.
    y = torch.tensor(3)
    y.add_(1)
    return x + y


def MutationMetadata(x):
    # [FUNCTION DOC] HELPER: Test function. Resizes the input in place to two elements (a
    # metadata mutation) and returns it. Side effect: changes the argument's shape.
    x.resize_(2)
    return x


Passes = [CSEPass]
Test_Cases = [
    TakeList,
    ReturnList,
    Mutation,
    MutationInput,
    MutationMetadata,
    MutationTorchTensorCall,
]
Factory_Test_Cases = [FactoryFunctionCall, MutationFactory]
Devices = ["cpu"]
if torch.cuda.is_available():
    Devices.append("cuda")


def name_fn(common_pass, f, device):
    # [FUNCTION DOC] HELPER: Builds the test-case name for the parametrized tests from the
    # pass class name (instantiating the pass to read it), the function name and the device,
    # joined by underscores. Returns a string.
    """Names parameterized test cases."""
    return f"{type(common_pass()).__name__}_{f.__name__}_{device}"


@instantiate_parametrized_tests
class TestCommonPass(TestCase):
    # [CLASS DOC] Parametrized tests that run common FX passes on make_fx traced graphs and
    # check that the transformed module still computes what the original function computes.
    # Methods are expanded per pass, function and device by the parametrization decorator.
    # No state.
    @parametrize(
        "common_pass,f,device", itertools.product(Passes, Test_Cases, Devices), name_fn
    )
    def test_correctness(self, common_pass, f, device):
        # [FUNCTION DOC] For each pass, test function and device: traces the function on a
        # random tensor of ten elements, applies the pass, requires the result's
        # graph_module to be a GraphModule (raises AssertionError otherwise), then runs the
        # transformed module on a clone of the input and compares with running the original
        # function directly via assertEqual. The clone keeps in-place mutating functions
        # from affecting the reference run.
        inp = torch.randn(10, device=device)

        traced_m = make_fx(f)(inp)
        P = common_pass()

        res = P(traced_m)
        modified_m = res.graph_module
        if not isinstance(modified_m, GraphModule):
            raise AssertionError(
                f"Expected modified_m to be GraphModule, got {type(modified_m)}"
            )

        inp_copy = inp.clone()
        expected = f(inp)
        result = modified_m(inp_copy)

        self.assertEqual(result, expected)

    @parametrize(
        "common_pass,f,device",
        itertools.product(Passes, Factory_Test_Cases, Devices),
        name_fn,
    )
    def test_correctness_factory(self, common_pass, f, device):
        # [FUNCTION DOC] Same as the correctness test but for the factory-function cases,
        # which take a device argument as well as a tensor: traces with the device passed as
        # an extra example input, applies the pass, requires a GraphModule result
        # (AssertionError otherwise) and compares the transformed module's output on a
        # cloned input with the original function using assertEqual.
        inp = torch.randn(10, device=device)
        traced_m = make_fx(f)(inp, device)
        P = common_pass()

        res = P(traced_m)
        modified_m = res.graph_module
        if not isinstance(modified_m, GraphModule):
            raise AssertionError(
                f"Expected modified_m to be GraphModule, got {type(modified_m)}"
            )

        inp_copy = inp.clone()
        expected = f(inp, device)
        result = modified_m(inp_copy, device)

        self.assertEqual(result, expected)


if __name__ == "__main__":
    raise_on_run_directly("test/test_fx.py")
