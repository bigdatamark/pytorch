# [MODULE DOC]
# Prototype executor for traced prim graphs: execute runs a GraphModule and
# make_traced wraps a function so it is traced under TorchRefsMode before execution.
from collections.abc import Callable
from typing import Any, TypeVar
from typing_extensions import ParamSpec, TypeVarTuple, Unpack

from torch._prims.context import TorchRefsMode
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx, wrapper_and_args_for_make_fx


T = TypeVar("T")
P = ParamSpec("P")
Ts = TypeVarTuple("Ts")


def execute(
    gm: GraphModule,
    *args: Unpack[Ts],
    executor: str = "aten",
    executor_parameters: dict | None = None,
) -> Any:
    # [FUNCTION DOC] Runs graph module gm on args. Only the "aten" executor is supported
    # (calls the module forward directly); executor_parameters is unused.
    # Raises ValueError for any other executor name.
    """
    Prototype ATen executor.

    Just executes the context's graph.
    """

    if executor == "aten":
        return gm.forward(*args)

    msg = f"Received unexpected value for 'executor': {executor}. Allowed values are: aten."
    raise ValueError(msg)


def make_traced(fn: Callable[P, T]) -> Callable[P, T]:
    # [FUNCTION DOC] Returns a wrapper of fn that, when called, traces fn with make_fx under
    # TorchRefsMode and runs the resulting graph through execute. The wrapper accepts an
    # optional executor keyword (default "aten") which is removed from the kwargs.
    """
    Returns a function that, when called, will
    trace its torch operations to prims and then
    execute those prims on the requested trace executor
    (possibly lowering them to that trace executor first).

    Only supports the torch operations defined in _torch_to_reference_map
    in context.py and operations with positional args. All args must
    be tensors.
    In the near future all these restrictions will be lifted.

    Example usage:

    def foo(a, b):
      return torch.add(a, b)

    traced_foo = make_traced(foo)

    a = torch.randn((1, 2, 3, 4, 5), device='cuda')
    b = torch.randn((1, 2, 3, 4, 5), device='cuda')
    result = traced_foo(a, b, executor='aten')
    """

    def _traced(*args: P.args, **kwargs: P.kwargs) -> T:
        # [FUNCTION DOC] HELPER: Wrapper body. Pops the executor name, builds the make_fx
        # wrapper and flat args, traces under TorchRefsMode, then executes the graph.
        executor = str(kwargs.pop("executor", "aten"))

        # TODO: caching
        wrapped, all_args = wrapper_and_args_for_make_fx(fn, args, kwargs)

        with TorchRefsMode():
            gm = make_fx(wrapped)(all_args)
        return execute(gm, all_args, executor=executor)

    return _traced  # type: ignore[return-value]
