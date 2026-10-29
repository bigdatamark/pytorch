#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Tests for torch.distributed.nn.jit.instantiator, which generates remote module
# templates. Defines a TorchScript interface (MyModuleInterface) and a trivial module used as
# inputs. Import-time side effect: if torch.distributed is unavailable the process prints a
# message to stderr and exits with status 0, skipping everything.

import sys

import torch
import torch.distributed as dist
from torch import nn, Tensor


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.distributed.nn.jit import instantiator
from torch.testing._internal.common_utils import run_tests, TestCase


@torch.jit.interface
class MyModuleInterface:
    # [CLASS DOC] HELPER: TorchScript interface with a single forward method taking a tensor, an
    # int and a string with default value default, returning a tuple of tensor, int and str. Used
    # to check argument and return type extraction.
    def forward(
        self, tensor: Tensor, number: int, word: str = "default"
    ) -> tuple[Tensor, int, str]:
        # [FUNCTION DOC] HELPER: interface method signature only; the body does nothing.
        pass


class MyModule(nn.Module):
    # [CLASS DOC] HELPER: empty nn.Module subclass used as a placeholder module type.
    pass


def create_module():
    # [FUNCTION DOC] HELPER: returns a new MyModule instance (not used by the tests below).
    return MyModule()


class TestInstantiator(TestCase):
    # [CLASS DOC] Tests for the instantiator helpers: interface signature parsing and the
    # generated scriptable and non scriptable remote module templates.
    def test_get_arg_return_types_from_interface(self):
        # [FUNCTION DOC] Asserts get_arg_return_types_from_interface returns the argument names
        # string (tensor, number, word), the typed arguments string, and the return type rendered
        # with the typing style name Tuple for the interface.
        (
            args_str,
            arg_types_str,
            return_type_str,
        ) = instantiator.get_arg_return_types_from_interface(MyModuleInterface)
        self.assertEqual(args_str, "tensor, number, word")
        self.assertEqual(arg_types_str, "tensor: Tensor, number: int, word: str")
        self.assertEqual(return_type_str, "Tuple[Tensor, int, str]")

    def test_instantiate_scripted_remote_module_template(self):
        # [FUNCTION DOC] Generates the scriptable template for the interface and asserts the
        # resulting module has remote forward and generated methods attributes.
        generated_module = instantiator.instantiate_scriptable_remote_module_template(
            MyModuleInterface
        )
        self.assertTrue(hasattr(generated_module, "_remote_forward"))
        self.assertTrue(hasattr(generated_module, "_generated_methods"))

    def test_instantiate_non_scripted_remote_module_template(self):
        # [FUNCTION DOC] Generates the non scriptable template (no interface) and asserts the
        # same two attributes exist on the generated module.
        generated_module = (
            instantiator.instantiate_non_scriptable_remote_module_template()
        )
        self.assertTrue(hasattr(generated_module, "_remote_forward"))
        self.assertTrue(hasattr(generated_module, "_generated_methods"))


if __name__ == "__main__":
    run_tests()
