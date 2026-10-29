# mypy: allow-untyped-defs
# [MODULE DOC] Deprecated CUDA-specific autocast wrappers. Defines autocast (subclass of torch.amp autocast fixed to
# the cuda device), a private _cast shim, and custom_fwd/custom_bwd wrappers delegating to torch.amp with
# device_type cuda. All are marked deprecated and emit FutureWarning when used.
import functools
import sys
from typing import Any
from typing_extensions import deprecated

import torch


__all__ = ["autocast", "custom_fwd", "custom_bwd"]


@deprecated(
    "`torch.cuda.amp.autocast(args...)` is deprecated. "
    "Please use `torch.amp.autocast('cuda', args...)` instead.",
    category=FutureWarning,
)
class autocast(torch.amp.autocast_mode.autocast):
    # [CLASS DOC] Deprecated alias of the generic autocast context manager/decorator with device type fixed to "cuda".
    # Adds TorchScript-aware behavior: when scripting, it only stores enabled, device and fast_dtype and
    # enter/exit/call become no-ops. On Python < 3.13 defines __new__ and __init_subclass__ so signature
    # inspection works despite the deprecated decorator.
    r"""See :class:`torch.autocast`.

    ``torch.cuda.amp.autocast(args...)`` is deprecated. Please use ``torch.amp.autocast("cuda", args...)`` instead.
    """

    # TODO: remove this conditional once we stop supporting Python < 3.13
    # Prior to Python 3.13, inspect.signature could not retrieve the correct
    # signature information for classes decorated with @deprecated (unless
    # the __new__ static method was explicitly defined);
    #
    # However, this issue has been fixed in Python 3.13 and later versions.
    if sys.version_info < (3, 13):

        def __new__(
            # [FUNCTION DOC] HELPER: Only defined for Python < 3.13; creates the instance ignoring arguments (they exist so
            # inspect.signature reports the right parameters).
            cls,
            enabled: bool = True,
            dtype: torch.dtype = torch.float16,
            cache_enabled: bool = True,
        ):
            return super().__new__(cls)

        def __init_subclass__(cls):
            # [FUNCTION DOC] HELPER: Only defined for Python < 3.13; intentionally does nothing (overrides the deprecated
            # decorator's subclass hook).
            pass

    def __init__(
        # [FUNCTION DOC] Configures autocast for device "cuda" with flags enabled, dtype (default float16) and cache_enabled.
        # Under TorchScript scripting only records enabled, device and fast_dtype; otherwise calls the parent
        # initializer.
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.float16,
        cache_enabled: bool = True,
    ):
        if torch._jit_internal.is_scripting():
            self._enabled = enabled
            self.device = "cuda"
            self.fast_dtype = dtype
            return
        super().__init__(
            "cuda", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled
        )

    def __enter__(self):
        # [FUNCTION DOC] Enters the autocast region and returns the context object; under scripting returns self without
        # changing any autocast state.
        if torch._jit_internal.is_scripting():
            return self
        return super().__enter__()

    # TODO: discuss a unified TorchScript-friendly API for autocast
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        # [FUNCTION DOC] Leaves the autocast region (restores prior autocast state) via the parent; no-op under scripting.
        if torch._jit_internal.is_scripting():
            return
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, func):
        # [FUNCTION DOC] Decorator form: returns func wrapped so it runs under autocast; returns func unchanged under scripting.
        if torch._jit_internal.is_scripting():
            return func
        return super().__call__(func)


# Preserved only for BC reasons
@deprecated(
    "`torch.cuda.amp.autocast_mode._cast(value, dtype)` is deprecated. "
    "Please use `torch.amp.autocast_mode._cast(value, 'cuda', dtype)` instead.",
    category=FutureWarning,
)
def _cast(value, dtype):
    # [FUNCTION DOC] HELPER: Deprecated backward-compat shim; casts value (tensor or nested container) to dtype using the
    # generic torch.amp cast helper with device type cuda.
    return torch.amp.autocast_mode._cast(value, "cuda", dtype)


@deprecated(
    "`torch.cuda.amp.custom_fwd(args...)` is deprecated. "
    "Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.",
    category=FutureWarning,
)
def custom_fwd(fwd=None, *, cast_inputs=None):
    # [FUNCTION DOC] Deprecated wrapper over torch.amp.custom_fwd with device_type cuda. Takes the forward function fwd
    # (optional, for decorator-with-arguments usage) and cast_inputs dtype; returns the decorated function or
    # a decorator.
    """
    ``torch.cuda.amp.custom_fwd(args...)`` is deprecated. Please use
    ``torch.amp.custom_fwd(args..., device_type='cuda')`` instead.
    """
    return functools.partial(torch.amp.custom_fwd, device_type="cuda")(
        fwd=fwd, cast_inputs=cast_inputs
    )


@deprecated(
    "`torch.cuda.amp.custom_bwd(args...)` is deprecated. "
    "Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.",
    category=FutureWarning,
)
def custom_bwd(bwd):
    # [FUNCTION DOC] Deprecated wrapper over torch.amp.custom_bwd with device_type cuda; returns the decorated backward function.
    """
    ``torch.cuda.amp.custom_bwd(args...)`` is deprecated. Please use
    ``torch.amp.custom_bwd(args..., device_type='cuda')`` instead.
    """
    return functools.partial(torch.amp.custom_bwd, device_type="cuda")(bwd)
