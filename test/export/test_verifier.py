# [MODULE DOC] Tests for the export Verifier and ExportedProgram validation. Covers accepting
# valid exported programs (plain, higher order cond), rejecting call_module nodes,
# non-functional operators (also inside nested branch submodules), and validation failures
# from tampered signatures, state dict contents and output node counts. All tests require
# dynamo support; the cond ones are skipped on Windows.
# Owner(s): ["oncall: export"]
import unittest

import torch
from functorch.experimental import control_flow
from torch import Tensor
from torch._dynamo.eval_frame import is_dynamo_supported
from torch._export.verifier import SpecViolationError, Verifier
from torch.export import export
from torch.export.exported_program import InputKind, InputSpec, TensorArgument
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase


@unittest.skipIf(not is_dynamo_supported(), "dynamo isn't supported")
class TestVerifier(TestCase):
    # [CLASS DOC] Test case for graph level Verifier checks and program level validate checks.
    # Each test defines small local modules, exports them in strict mode, optionally corrupts
    # the program or graph, then expects either success or a SpecViolationError.
    def test_verifier_basic(self) -> None:
        # [FUNCTION DOC] Exports a module that adds two tensors and checks that Verifier
        # accepts it without raising.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module whose forward returns the sum of its two
            # inputs.
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns the elementwise sum of the two tensors.
                return x + y

        f = Foo()

        ep = export(f, (torch.randn(100), torch.randn(100)), strict=True)

        verifier = Verifier()
        verifier.check(ep)

    def test_verifier_call_module(self) -> None:
        # [FUNCTION DOC] Traces a module with a linear submodule using plain symbolic tracing,
        # which leaves a call_module node, and checks that the verifier graph module check
        # raises SpecViolationError.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module wrapping one linear layer of width ten.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the base constructor and stores a ten by ten
                # linear layer as the attribute linear.
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Applies the linear layer to the input and returns the
                # result.
                return self.linear(x)

        gm = torch.fx.symbolic_trace(M())

        verifier = Verifier()
        with self.assertRaises(SpecViolationError):
            verifier._check_graph_module(gm)

    def test_verifier_no_functional(self) -> None:
        # [FUNCTION DOC] Exports an addition module, decomposes it, then rewrites each add
        # node target to the in-place add overload, which is not functional. Checks that the
        # verifier raises SpecViolationError.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module whose forward returns the sum of its two
            # inputs.
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns the elementwise sum of the two tensors.
                return x + y

        f = Foo()

        ep = export(
            f, (torch.randn(100), torch.randn(100)), strict=True
        ).run_decompositions({})
        for node in ep.graph.nodes:
            if node.target == torch.ops.aten.add.Tensor:
                node.target = torch.ops.aten.add_.Tensor

        verifier = Verifier()
        with self.assertRaises(SpecViolationError):
            verifier.check(ep)

    @unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
    def test_verifier_higher_order(self) -> None:
        # [FUNCTION DOC] Exports a module using the control flow cond operator with branch
        # functions for add and subtract, and checks the verifier accepts the program with its
        # higher order operator and submodules.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module that conditionally adds or subtracts its inputs
            # depending on whether the first input sum exceeds two.
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Defines the two branch functions and returns the cond
                # result over the inputs.
                def true_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    # [FUNCTION DOC] HELPER: Cond true branch; returns the sum of the two
                    # inputs.
                    return x + y

                def false_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    # [FUNCTION DOC] HELPER: Cond false branch; returns the first input minus
                    # the second.
                    return x - y

                return control_flow.cond(x.sum() > 2, true_fn, false_fn, [x, y])

        f = Foo()

        ep = export(f, (torch.randn(3, 3), torch.randn(3, 3)), strict=True)

        verifier = Verifier()
        verifier.check(ep)

    @unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
    def test_verifier_nested_invalid_module(self) -> None:
        # [FUNCTION DOC] Exports and decomposes a cond module, then rewrites the add node in
        # the true branch submodule to the in-place add overload. Checks that the verifier
        # detects the violation in the nested graph and raises SpecViolationError.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module that conditionally adds or subtracts its inputs
            # depending on whether the first input sum exceeds two.
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Defines the two branch functions and returns the cond
                # result over the inputs.
                def true_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    # [FUNCTION DOC] HELPER: Cond true branch; returns the sum of the two
                    # inputs.
                    return x + y

                def false_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    # [FUNCTION DOC] HELPER: Cond false branch; returns the first input minus
                    # the second.
                    return x - y

                return control_flow.cond(x.sum() > 2, true_fn, false_fn, [x, y])

        f = Foo()

        ep = export(
            f, (torch.randn(3, 3), torch.randn(3, 3)), strict=True
        ).run_decompositions({})
        for node in ep.graph_module.true_graph_0.graph.nodes:
            if node.target == torch.ops.aten.add.Tensor:
                node.target = torch.ops.aten.add_.Tensor

        verifier = Verifier()
        with self.assertRaises(SpecViolationError):
            verifier.check(ep)

    def test_ep_verifier_basic(self) -> None:
        # [FUNCTION DOC] Exports a module with a linear layer and checks that program
        # validation succeeds without raising.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module wrapping one linear layer of width ten.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the base constructor and stores a ten by ten
                # linear layer as the attribute linear.
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Applies the linear layer to the input and returns the
                # result.
                return self.linear(x)

        ep = export(M(), (torch.randn(10, 10),), strict=True)
        ep.validate()

    def test_ep_verifier_invalid_param(self) -> None:
        # [FUNCTION DOC] Exports a module with a registered parameter, replaces the first
        # input spec by a parameter spec pointing to a target name absent from the state dict,
        # and expects validation to raise SpecViolationError mentioning it is not in the state
        # dict. Then adds a plain tensor under that name to the state dict and expects
        # SpecViolationError saying it is not a Parameter instance.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module that owns a registered parameter named a of
            # length one hundred and adds it to the sum of its inputs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the base constructor and registers the
                # parameter a, a random vector of length one hundred.
                super().__init__()
                self.register_parameter(
                    name="a", param=torch.nn.Parameter(torch.randn(100))
                )

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns the sum of both inputs and the parameter.
                return x + y + self.a

        ep = export(M(), (torch.randn(100), torch.randn(100)), strict=True)

        # Parameter doesn't exist in the state dict
        ep.graph_signature.input_specs[0] = InputSpec(
            kind=InputKind.PARAMETER, arg=TensorArgument(name="p_a"), target="bad_param"
        )
        with self.assertRaisesRegex(SpecViolationError, "not in the state dict"):
            ep.validate()

        # Add non-torch.nn.Parameter parameter to the state dict
        ep.state_dict["bad_param"] = torch.randn(100)
        with self.assertRaisesRegex(
            SpecViolationError, "not an instance of torch.nn.Parameter"
        ):
            ep.validate()

    def test_ep_verifier_invalid_buffer(self) -> None:
        # [FUNCTION DOC] Exports a module holding a plain tensor attribute, replaces the first
        # input spec by a persistent buffer spec whose target is missing from the state dict,
        # and expects validation to raise SpecViolationError saying it is not in the state
        # dict.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module storing a scalar tensor attribute named a and
            # adding it to the sum of its inputs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the base constructor and stores the scalar
                # tensor three as attribute a.
                super().__init__()
                self.a = torch.tensor(3.0)

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns the sum of both inputs and the scalar tensor
                # attribute.
                return x + y + self.a

        ep = export(M(), (torch.randn(100), torch.randn(100)), strict=True)

        # Buffer doesn't exist in the state dict
        ep.graph_signature.input_specs[0] = InputSpec(
            kind=InputKind.BUFFER,
            arg=TensorArgument(name="c_a"),
            target="bad_buffer",
            persistent=True,
        )
        with self.assertRaisesRegex(SpecViolationError, "not in the state dict"):
            ep.validate()

    def test_ep_verifier_buffer_mutate(self) -> None:
        # [FUNCTION DOC] Exports a module with one parameter and two buffers whose forward
        # mutates the second buffer in place, and checks that validation of the program with a
        # mutated buffer succeeds.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module with a parameter and two buffers; forward mixes
            # the inputs with them and increments the second buffer in place.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the base constructor and creates one parameter
                # and two buffers holding scalar values.
                super().__init__()

                self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))

                self.my_buffer1 = torch.nn.Buffer(torch.tensor(3.0))
                self.my_buffer2 = torch.nn.Buffer(torch.tensor(4.0))

            def forward(self, x1, x2):
                # [FUNCTION DOC] HELPER: Returns the first input plus the parameter, times the
                # first buffer, plus the second input times the second buffer, after adding
                # one to the second buffer in place. The addition happens after output
                # computation, so the output uses the old value.
                # Use the parameter, buffers, and both inputs in the forward method
                output = (
                    x1 + self.my_parameter
                ) * self.my_buffer1 + x2 * self.my_buffer2

                # Mutate one of the buffers (e.g., increment it by 1)
                self.my_buffer2.add_(1.0)
                return output

        ep = export(M(), (torch.tensor(5.0), torch.tensor(6.0)), strict=True)
        ep.validate()

    def test_ep_verifier_invalid_output(self) -> None:
        # [FUNCTION DOC] Exports a buffer mutating module, then rewrites the output node
        # arguments to contain only one original output plus the first graph node, changing
        # the number of outputs relative to the signature. Expects validation to raise
        # SpecViolationError mentioning the number of output nodes.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module with a parameter and two buffers; forward mixes
            # the inputs with them and increments the second buffer in place.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the base constructor and creates one parameter
                # and two buffers holding scalar values.
                super().__init__()

                self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))

                self.my_buffer1 = torch.nn.Buffer(torch.tensor(3.0))
                self.my_buffer2 = torch.nn.Buffer(torch.tensor(4.0))

            def forward(self, x1, x2):
                # [FUNCTION DOC] HELPER: Returns the first input plus the parameter, times the
                # first buffer, plus the second input times the second buffer, after adding
                # one to the second buffer in place. The addition happens after output
                # computation, so the output uses the old value.
                # Use the parameter, buffers, and both inputs in the forward method
                output = (
                    x1 + self.my_parameter
                ) * self.my_buffer1 + x2 * self.my_buffer2

                # Mutate one of the buffers (e.g., increment it by 1)
                self.my_buffer2.add_(1.0)
                return output

        ep = export(M(), (torch.tensor(5.0), torch.tensor(6.0)), strict=True)

        output_node = list(ep.graph.nodes)[-1]
        output_node.args = (
            (
                output_node.args[0][0],
                next(iter(ep.graph.nodes)),
            ),
        )

        with self.assertRaisesRegex(SpecViolationError, "Number of output nodes"):
            ep.validate()


if __name__ == "__main__":
    run_tests()
