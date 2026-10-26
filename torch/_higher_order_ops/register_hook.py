# [MODULE DOC]
# Defines the register_hook higher order operator, which attaches a backward
# hook (a traced subgraph plus its free variables) to a tensor so that the hook
# is captured when tracing the joint forward/backward graph.
# Import-time side effect: creates the module-level singleton register_hook_op.
from collections.abc import Callable
from typing import Any

import torch
from torch._ops import HigherOrderOperator


class RegisterHookOp(HigherOrderOperator):
    # [CLASS DOC] HigherOrderOperator named "register_hook" (cacheable). Stateless;
    # calling it registers a gradient hook on a tensor and returns the same tensor.
    """HOP that registers a backward hook on a tensor.

    The hook body is a traced subgraph that takes (grad, *freevars)
    and returns the modified gradient. During make_fx tracing, the
    hook registers on the FakeTensor and fires on the accumulated
    gradient during backward tracing, capturing the hook's effect
    in the joint graph.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes the operator under the name "register_hook"
        # with caching enabled.
        super().__init__("register_hook", cacheable=True)

    def __call__(
        self,
        tensor: torch.Tensor,
        hook_body: Callable[..., Any],
        *freevars: Any,
    ) -> torch.Tensor:
        # [FUNCTION DOC] Registers a gradient hook on the given tensor that calls
        # hook_body with the incoming gradient followed by the free variables.
        # Args: tensor to hook, hook_body callable, extra freevars. Returns the
        # same tensor object. Side effect: mutates the tensor's hook list. Does
        # not dispatch through the HOP machinery (no super call).
        def hook(grad: torch.Tensor) -> torch.Tensor | None:
            # [FUNCTION DOC] HELPER: Closure invoked on backward; forwards the
            # gradient and captured freevars to hook_body and returns its result
            # (a replacement gradient, or None to keep the gradient unchanged).
            return hook_body(grad, *freevars)

        tensor.register_hook(hook)
        return tensor


register_hook_op = RegisterHookOp()
