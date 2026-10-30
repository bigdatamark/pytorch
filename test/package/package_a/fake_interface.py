# [MODULE DOC] Fixture module for TorchScript interface tests with torch.package. Defines a
# TorchScript module interface plus two alternative modules that implement it and a module that
# holds an interface-typed submodule. The interface decorator is applied at import time.
import torch
from torch import Tensor


@torch.jit.interface
class ModuleInterface(torch.nn.Module):
    # [CLASS DOC] TorchScript module interface (marked with the jit interface decorator)
    # declaring a single method named one that takes two tensors and returns a tensor.
    def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
        # [FUNCTION DOC] HELPER: Interface method declaration; the body is a no-op placeholder
        # and the interface only fixes the signature, so this does not compute anything.
        pass


class OrigModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Implementation of the interface. forward adds the input to the result
    # of one applied to the input twice, plus one.
    """A module that implements ModuleInterface."""

    def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
        # [FUNCTION DOC] HELPER: Returns the sum of both tensor arguments plus one.
        return inp1 + inp2 + 1

    def two(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] HELPER: Returns the input tensor plus two. Not part of the interface.
        return input + 2

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] HELPER: Returns the input plus the result of the one method called with
        # the input for both arguments, plus one.
        return input + self.one(input, input) + 1


class NewModule(torch.nn.Module):
    # [CLASS DOC] HELPER: A second implementation of the interface whose one method multiplies
    # instead of adds, used to swap implementations behind the interface.
    """A *different* module that implements ModuleInterface."""

    def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
        # [FUNCTION DOC] HELPER: Returns the elementwise product of both arguments plus one.
        return inp1 * inp2 + 1

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] HELPER: Calls the one method with the input and the input plus one, and
        # returns the result.
        return self.one(input, input + 1)


class UsesInterface(torch.nn.Module):
    # [CLASS DOC] HELPER: Module with an attribute annotated with the interface type; the
    # constructor assigns an OrigModule instance to it. Exercises packaging of modules reached
    # through an interface.
    proxy_mod: ModuleInterface

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes the base module and sets the interface-typed
        # submodule attribute to a new OrigModule.
        super().__init__()
        self.proxy_mod = OrigModule()

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] HELPER: Delegates to the interface submodule, calling its one method
        # with the input as both arguments, and returns the result.
        return self.proxy_mod.one(input, input)
