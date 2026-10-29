# [MODULE DOC] Test-only custom op _inductor_test.realize, an identity-like clone op used to force
# Inductor to realize a tensor. Import-time side effects: defines the op in the _inductor_test library
# and registers clone implementations for CPU, CUDA, MPS and Meta.
from typing import Any

import torch.library
from torch import Tensor
from torch.autograd import Function


_test_lib_def = torch.library.Library("_inductor_test", "DEF")
_test_lib_def.define("realize(Tensor self) -> Tensor", tags=torch.Tag.pt2_compliant_tag)

_test_lib_impl = torch.library.Library("_inductor_test", "IMPL")
for dispatch_key in ("CPU", "CUDA", "MPS", "Meta"):
    _test_lib_impl.impl("realize", lambda x: x.clone(), dispatch_key)


class Realize(Function):
    # [CLASS DOC] Autograd Function wrapping the realize op; backward passes the gradient through.
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx: object, x: Tensor) -> Tensor:
        # [FUNCTION DOC] Forward: calls the realize op on x and returns the result.
        return torch.ops._inductor_test.realize(x)

    @staticmethod
    # types need to stay consistent with _SingleLevelFunction
    def backward(ctx: Any, *grad_output: Any) -> Any:
        # [FUNCTION DOC] Backward: returns the first incoming gradient unchanged.
        return grad_output[0]


def realize(x: Tensor) -> Tensor:
    # [FUNCTION DOC] Public entry point: applies Realize to x, returning a cloned tensor that is realized
    # in the compiled graph and differentiable.
    return Realize.apply(x)
