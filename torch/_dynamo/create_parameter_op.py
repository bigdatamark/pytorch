# [MODULE DOC] Support for tracing creation of torch.nn.Parameter under Dynamo/AOTAutograd.
# Provides an autograd Function that turns a pre-made storage-less placeholder
# Parameter into the real parameter inside the graph, a factory for such
# placeholders, and a thread-local flag (with a context manager and a query
# function) controlling whether Parameter creation is converted to this traceable form.
# Import-time side effects: creates a module-level thread-local object and a doc string constant.
import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch


# See [Note: Metadata mutation in proxy tracing] for why sacrificial parameter mutates
# metadata during proxy tracing and we should remove the sacrificial parameter logic.
doc = """
This is used when dynamo traces torch.nn.Parameter, which normally would not trace properly
with AOTAutograd.  We instead create a placeholder torch.nn.Parameter before the graph, which
becomes a graph arg and has no storage backing it.  At the point in the graph where the parameter
actually should be created we mutate this sacrificial placeholder into it.  This allows gradients
to flow into the parameter as if it were an input to the graph (which is the only thing we are
allowed to compute gradients on).
""".strip()


class TracableCreateParameter(torch.autograd.Function):
    # [CLASS DOC] Autograd Function that mutates a sacrificial placeholder Parameter
    # into a parameter backed by the given tensor, so gradients flow to the placeholder
    # (a graph input). Stateless; used only through its apply method.
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx: Any, tensor: Any, placeholder: Any) -> torch.nn.Parameter:
        # [FUNCTION DOC] Detaches the tensor if it requires grad, then sets the
        # placeholder's storage/metadata to that tensor in place and returns the
        # placeholder (now a Parameter). Mutates the placeholder. ctx is unused.
        if tensor.requires_grad:
            tensor = tensor.detach()
        return placeholder.set_(tensor)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[None, torch.Tensor]:
        # [FUNCTION DOC] Returns no gradient for the source tensor and passes the
        # first incoming output gradient through to the placeholder input.
        grad =grad_outputs[0]
        return None, grad  # grad flows to placeholder


def tracable_create_parameter(
    tensor: torch.Tensor, placeholder: torch.nn.Parameter
) -> torch.nn.Parameter:
    # [FUNCTION DOC] Applies the traceable create-parameter Function to tensor and
    # placeholder, with grad mode set to the placeholder's requires_grad flag for
    # the duration of the call. Returns the resulting Parameter (the placeholder).
    with torch.set_grad_enabled(placeholder.requires_grad):
        out = TracableCreateParameter.apply(tensor, placeholder)
    return out


def new_parameter_placeholder(
    size: tuple[int, ...], dtype: torch.dtype, device: torch.device, requires_grad: bool
) -> torch.nn.Parameter:
    # [FUNCTION DOC] Builds an empty Parameter of the given size, dtype, device and
    # requires_grad, then resizes its untyped storage to zero bytes so it has no
    # backing memory. Returns the placeholder.
    """Create a placeholder to be passed to the above functions"""
    result = torch.nn.Parameter(
        torch.empty(size, dtype=dtype, device=device), requires_grad=requires_grad
    )
    # TODO(jansel): alloc followed by free is inefficient, need a way to allocate an unbacked tensor.
    # Allocating a zero tensor would cause assert failures in autograd.
    result.untyped_storage().resize_(0)
    return result


_TLS = threading.local()


@contextmanager
def do_not_convert_to_tracable_parameter() -> Generator[bool, None, None]:
    # [FUNCTION DOC] Context manager that sets the thread-local convert flag to
    # false for its body (yielding false) and restores the previous value on exit,
    # even on error. Side effect: mutates thread-local state.
    old_flag = getattr(_TLS, "convert_tracable_parameter", True)
    _TLS.convert_tracable_parameter = False
    try:
        yield False
    finally:
        _TLS.convert_tracable_parameter = old_flag


def can_convert_to_tracable_parameter() -> bool:
    # [FUNCTION DOC] Returns the thread-local flag saying whether Parameter creation
    # may be converted to the traceable form; defaults to true when never set.
    return getattr(_TLS, "convert_tracable_parameter", True)
