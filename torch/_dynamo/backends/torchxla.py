# [MODULE DOC] XLA backends for Dynamo. Defines the experimental backend openxla_eval, a boxed
# variant, a shared helper that lazily compiles via torch_xla's dynamo bridge on first call, and
# the openxla backend (AOT autograd with the boxed forward compiler). Import-time side effects:
# registers the experimental backend openxla_eval and the backend openxla in the registry.
import logging
from collections.abc import Callable
from typing import Any

import torch
from functorch.compile import make_boxed_func
from torch import fx

from ..backends.common import aot_autograd
from .registry import CompiledFn, register_backend, register_experimental_backend


log = logging.getLogger(__name__)


@register_experimental_backend
def openxla_eval(
    model: fx.GraphModule, fake_tensor_inputs: list[torch.Tensor]
) -> CompiledFn:
    # [FUNCTION DOC] Experimental backend: returns an unboxed XLA-compiled callable for the
    # graph module by delegating to xla_backend_helper.
    return xla_backend_helper(model, fake_tensor_inputs, boxed=False)


def openxla_eval_boxed(
    model: fx.GraphModule, fake_tensor_inputs: list[torch.Tensor]
) -> Callable[..., Any]:
    # [FUNCTION DOC] HELPER: Same as openxla_eval but returns a boxed-calling-convention
    # callable; used as the forward compiler for the openxla AOT autograd backend.
    return xla_backend_helper(model, fake_tensor_inputs, boxed=True)


def xla_backend_helper(
    model: fx.GraphModule, fake_tensor_inputs: list[torch.Tensor], boxed: bool = False
) -> Callable[..., Any]:
    # [FUNCTION DOC] HELPER: Imports the torch_xla dynamo bridge (raises ImportError with install
    # instructions if missing) and returns a forward function. On its first call the function
    # extracts the compiled graph from the model using the actual args, drops the model
    # reference, and from then on calls the compiled graph. The result is wrapped as a boxed
    # function when boxed is true. The fake tensor inputs are unused.
    try:
        import torch_xla.core.dynamo_bridge as bridge
    except ImportError as e:
        raise ImportError(
            "Please follow the instruction in https://github.com/pytorch/xla#pytorchxla to install torch_xla"
        ) from e

    compiled_graph = None

    def fwd(*args: torch.Tensor) -> Any:
        # [FUNCTION DOC] HELPER: Lazily compiles the graph on first call via the bridge, caching
        # it in a closure variable (and releasing the model), then runs it on args and returns
        # the result.
        nonlocal model
        nonlocal compiled_graph
        if compiled_graph is None:
            compiled_graph = bridge.extract_compiled_graph(model, args)
            del model
        return compiled_graph(*args)

    return make_boxed_func(fwd) if boxed else fwd


openxla = aot_autograd(
    fw_compiler=openxla_eval_boxed,
)
register_backend(name="openxla", compiler_fn=openxla)
