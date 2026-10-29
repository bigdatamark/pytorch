# mypy: allow-untyped-defs
# [MODULE DOC]
# Python entry points for TorchScript Await objects (lazy/asynchronous values).
# Wraps the C++ awaitable primitives and registers the wait/nowait functions as
# builtins for the prim::awaitable_wait and prim::awaitable_nowait operators.
# Import-time side effects: sets the public module of _Await to torch.jit and
# registers two builtins in the global builtin table.
import torch
from torch._jit_internal import _Await
from torch.jit._builtins import _register_builtin
from torch.utils import set_module


set_module(_Await, "torch.jit")


def _awaitable(func, *args, **kwargs):
    # [FUNCTION DOC] HELPER: Creates a lazy Await object bound to func and the
    # given positional and keyword arguments by delegating to the C++ binding.
    # The function is not run until the result is requested. Returns the Await.
    r"""Create Await object that will call specified function with specified args, when it is requested for the result."""
    return torch._C._awaitable(func, *args, **kwargs)


def _awaitable_wait(aw):
    # [FUNCTION DOC] HELPER: Requests the result of an Await object via the C++
    # binding; if the Await is not yet complete its function runs immediately.
    # Returns the result value.
    r"""Request await the result of execution, if Await is not completed yet, the func will be called immediately."""
    return torch._C._awaitable_wait(aw)


def _awaitable_nowait(o):
    # [FUNCTION DOC] HELPER: Creates an already-completed Await object that
    # holds the given value o, via the C++ binding. Returns the Await.
    r"""Create completed Await with specified result."""
    return torch._C._awaitable_nowait(o)


_register_builtin(_awaitable_wait, "prim::awaitable_wait")
_register_builtin(_awaitable_nowait, "prim::awaitable_nowait")
