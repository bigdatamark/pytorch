# [MODULE DOC] Dynamo polyfills for Tensor._make_subclass and Tensor.__new__, registered
# via substitute_in_graph. Exports make_subclass and tensor_pynew. Import-time side
# effect: registers the substitutions.
from typing import Any

import torch

from ..decorators import substitute_in_graph


@substitute_in_graph(  # type: ignore[arg-type]
    torch.Tensor._make_subclass
)
def make_subclass(
    cls: type[Any], data: torch.Tensor, requires_grad: bool = False, **kwargs: Any
) -> Any:
    # [FUNCTION DOC] Polyfill of Tensor._make_subclass, approximating the C implementation.
    # Args: cls (target tensor class), data (tensor), requires_grad flag, extra kwargs.
    # Raises AssertionError if any extra kwargs are given. Detaches data, sets
    # requires_grad only if it differs (mutates the detached alias), then returns a plain
    # Tensor wrapping it when cls is Tensor, else data viewed as cls via as_subclass.
    # Runs with torch function subclass dispatch disabled.
    with torch._C.DisableTorchFunctionSubclass():
        # This is a rough approximation of `THPVariable_make_subclass`. It should
        # suffice for most of Dynamo tracing purposes.
        # https://github.com/pytorch/pytorch/blob/ccfde4dadfa3c342076a1ee387017f84dd4ad2f7/torch/csrc/autograd/python_variable.cpp#L597-L650
        if len(kwargs) != 0:
            raise AssertionError(
                "_make_subclass only supports requires_grad as keyword arg"
            )
        data = data.detach()

        # Avoid unnecessary `requires_grad` mutation, which isn't supported in Dynamo.
        if data.requires_grad != requires_grad:
            data.requires_grad = requires_grad

        # Dynamo can't yet handle upcasting to base tensor type via `as_subclass`.
        if cls is torch.Tensor:
            return torch.Tensor(data)

        # Calling `as_subclass` because
        # 1. Dynamo knows how to handle it
        # 2. the C impls match at this point -- both `THPVariable_make_subclass` and
        #    `THPVariable_as_subclass` calls `THPVariable_NewWithVar`.
        return data.as_subclass(cls)


@substitute_in_graph(  # type: ignore[arg-type]
    torch.Tensor.__new__
)
def tensor_pynew(
    cls: type[Any], data: torch.Tensor, requires_grad: bool = False, **kwargs: Any
) -> Any:
    # [FUNCTION DOC] Polyfill of Tensor.__new__ when called with data: forwards cls, data,
    # requires_grad and kwargs to make_subclass and returns the resulting tensor.
    return make_subclass(cls, data, requires_grad, **kwargs)


__all__ = [
    "make_subclass",
    "tensor_pynew",
]
