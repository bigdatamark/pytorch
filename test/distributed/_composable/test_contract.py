# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Tests for the composable-API decorator `contract` from
# torch.distributed._composable. Defines a small toy model and one TestCase checking that
# contracted APIs can add hooks, are forbidden from changing parameter FQNs, keep per-module
# state, record themselves in a per-module registry, and can take a list of modules. XX

from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributed._composable import _get_registry, contract
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class ToyModel(nn.Module):
    # [CLASS DOC] HELPER: Toy model with two two-layer Linear stacks (seq1, seq2), a learnable
    # parameter p, and a plain tensor attribute b (not a registered buffer) that forward mutates
    # in place with the input sums. forward takes two inputs x and y.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds seq1/seq2 (two 10x10 Linear layers each), the 10x10
        # parameter p, and the one-element zero tensor b.
        super().__init__()
        self.seq1 = nn.Sequential(*[nn.Linear(10, 10) for _ in range(2)])
        self.seq2 = nn.Sequential(*[nn.Linear(10, 10) for _ in range(2)])
        self.p = nn.Parameter(torch.randn(10, 10), requires_grad=True)
        self.b = torch.zeros(1)  # buffer

    def forward(self, x, y):
        # [FUNCTION DOC] HELPER: Under no_grad adds sum(x) + sum(y) into b in place, then
        # returns p plus seq1 applied to x plus seq2 applied to y.
        with torch.no_grad():
            self.b += x.sum() + y.sum()

        return self.p + self.seq1(x) + self.seq2(y)


class TestContract(TestCase):
    # [CLASS DOC] Tests for the `contract` decorator: hook registration, FQN-preservation check,
    # per-module state, the module registry, and APIs that take multiple modules.
    @skipIfTorchDynamo("Dynamo does not support the state key")
    def test_add_hooks(self):
        # [FUNCTION DOC] Applies a contracted no-op API that registers forward-pre, forward,
        # backward-pre and backward hooks (all pass-throughs) to seq1 and seq2 of a model, runs
        # forward/backward on it and on an undecorated deepcopy, and asserts all parameters
        # compare equal.
        def forward_pre_hook(
            module: nn.Module, inp: tuple[torch.Tensor]
        ) -> tuple[torch.Tensor]:
            # [FUNCTION DOC] HELPER: Pass-through forward pre-hook; returns the inputs
            # unchanged.
            return inp

        def forward_hook(
            module: nn.Module, inp: tuple[torch.Tensor], out: torch.Tensor
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Pass-through forward hook; returns the output unchanged.
            return out

        def backward_pre_hook(
            module: nn.Module, grad_output: torch.Tensor
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Pass-through backward pre-hook; returns grad_output
            # unchanged.
            return grad_output

        def backward_hook(
            module: nn.Module,
            grad_input: tuple[torch.Tensor],
            grad_output: torch.Tensor,
        ) -> tuple[torch.Tensor]:
            # [FUNCTION DOC] HELPER: Pass-through backward hook; returns grad_input unchanged.
            return grad_input

        @contract()
        def noop_api(module: nn.Module) -> nn.Module:
            # [FUNCTION DOC] HELPER: Contracted API body that registers the four pass-through
            # hooks on the module and returns it.
            module.register_forward_pre_hook(forward_pre_hook)
            module.register_forward_hook(forward_hook)
            module.register_full_backward_pre_hook(backward_pre_hook)
            module.register_full_backward_hook(backward_hook)
            return module

        model = ToyModel()
        model_with_hooks = deepcopy(model)
        noop_api(model.seq1)
        noop_api(model.seq2)

        x, y = torch.randn(10, 10), torch.randn(10, 10)
        model(x, y).sum().backward()
        model_with_hooks(x, y).sum().backward()

        for p1, p2 in zip(model.parameters(), model_with_hooks.parameters()):
            self.assertEqual(p1, p2)

    @skipIfTorchDynamo("Dynamo does not support the state key")
    def test_modify_fqn(self):
        # [FUNCTION DOC] Checks that a contracted API whose body wraps the module in a new
        # parent (which changes parameter FQNs) raises RuntimeError mentioning that composable
        # APIs cannot modify FQNs.
        class ModelWrapper(nn.Module):
            # [CLASS DOC] HELPER: Minimal wrapper module holding the given module as attribute
            # `module` and forwarding to it; used to force a change in parameter names.
            def __init__(self, module):
                # [FUNCTION DOC] HELPER: Stores the wrapped module.
                super().__init__()
                self.module = module

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the wrapped module applied to x.
                return self.module(x)

        @contract()
        def wrap_module(module: nn.Module) -> nn.Module:
            # [FUNCTION DOC] HELPER: Contracted API body that returns a ModelWrapper around the
            # module.
            return ModelWrapper(module)

        model = ToyModel()

        regex = "Checking parameters: Composable distributed API implementations cannot modify FQNs."
        with self.assertRaisesRegex(RuntimeError, regex):
            wrap_module(model.seq1)

    @skipIfTorchDynamo("Dynamo does not support the state key")
    def test_state(self):
        # [FUNCTION DOC] Checks that the contract API's state accessor returns per-module
        # mutable state: the API sets a dummy value of 7, a forward pre-hook verifies it and
        # sets 8, and the test asserts 7 before and 8 after running the model forward.
        def check_and_update_state_hook(
            module: nn.Module, inp: tuple[torch.Tensor]
        ) -> tuple[torch.Tensor]:
            # [FUNCTION DOC] HELPER: Forward pre-hook asserting the API state's dummy_state is
            # 7, then sets it to 8; returns the inputs unchanged.
            self.assertEqual(api.state(module).dummy_state, 7)
            api.state(module).dummy_state = 8
            return inp

        # FIXME: circular reference looks a bit weird. Shall we make .state a
        # top-level API instead attached to contract API?
        @contract()
        def api(module: nn.Module) -> nn.Module:
            # [FUNCTION DOC] HELPER: Contracted API body that sets dummy_state to 7 in the
            # module's state and registers the checking pre-hook; returns the module.
            api.state(module).dummy_state = 7
            module.register_forward_pre_hook(check_and_update_state_hook)
            return module

        model = ToyModel()
        api(model.seq1)

        self.assertEqual(api.state(model.seq1).dummy_state, 7)
        model(torch.zeros(10, 10), torch.zeros(10, 10))
        self.assertEqual(api.state(model.seq1).dummy_state, 8)

    @skipIfTorchDynamo("Dynamo does not support the state key")
    def test_registry(self):
        # [FUNCTION DOC] Checks the per-module registry of applied APIs: applying api1 then api2
        # to the root gives registry sizes 1 then 2, children have no registry (None), and
        # re-applying api1 raises AssertionError saying it has already been applied.
        # KNOWN ISSUE: the check on the registry keys passes a list as the first argument to
        # assertTrue, so it is always true and never compares the keys with the expected names.
        @contract()
        def api1(module: nn.Module) -> nn.Module:
            # [FUNCTION DOC] HELPER: Contracted no-op API; returns the module.
            return module

        @contract()
        def api2(module: nn.Module) -> nn.Module:
            # [FUNCTION DOC] HELPER: Contracted no-op API; returns the module.
            return module

        model = ToyModel()
        model = api1(model)
        self.assertEqual(1, len(_get_registry(model)))
        self.assertTrue("api1" in _get_registry(model))
        model = api2(model)
        self.assertEqual(2, len(_get_registry(model)))
        self.assertTrue([_get_registry(model).keys()], ["api1", "api2"])
        self.assertEqual(None, _get_registry(model.seq1))
        self.assertEqual(None, _get_registry(model.seq2))

        with self.assertRaisesRegex(AssertionError, "api1 has already been applied"):
            model = api1(model)

    @skipIfTorchDynamo("Dynamo does not support the state key")
    def test_multi_module_api(self):
        # [FUNCTION DOC] Applies an API taking a list of modules to three disjoint groups of a
        # Sequential and asserts modules share state and registry iff they were in the same
        # call; then asserts applying the API again to an already-covered module raises
        # AssertionError.
        @contract()
        def multi_module_api(modules: list[nn.Module]) -> nn.Module:
            # [FUNCTION DOC] HELPER: Contracted multi-module API body; returns the list
            # unchanged.
            return modules

        model = nn.Sequential(*[nn.Linear(3, 3) for _ in range(5)])
        multi_module_api([model[0], model[1]])
        multi_module_api([model[2], model[3]])
        multi_module_api([model[4]])
        # Check that modules have the same state and registry iff they shared
        # the same API call
        states = [multi_module_api.state(module) for module in model]
        self.assertEqual(states[0], states[1])
        self.assertEqual(states[2], states[3])
        self.assertNotEqual(states[0], states[2])
        self.assertNotEqual(states[0], states[4])
        self.assertNotEqual(states[2], states[4])
        registries = [_get_registry(module) for module in model]
        self.assertEqual(registries[0], registries[1])
        self.assertEqual(registries[2], registries[3])
        self.assertNotEqual(registries[0], registries[2])
        self.assertNotEqual(registries[0], registries[4])
        self.assertNotEqual(registries[2], registries[4])
        # Check that applying an API to a module multiple times errors
        model = nn.Sequential(*[nn.Linear(3, 3) for _ in range(5)])
        multi_module_api([model[0], model[1]])
        with self.assertRaisesRegex(
            AssertionError,
            "Each distinct composable distributed API can only be applied to "
            r"a module once. multi_module_api has already been applied to the "
            "following module:",
        ):
            multi_module_api([model[0], model[2]])


if __name__ == "__main__":
    run_tests()
