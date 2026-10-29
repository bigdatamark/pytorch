# mypy: allow-untyped-defs
# [MODULE DOC] Deprecated CPU-specific autocast wrapper. Defines the autocast class, a
# subclass of torch.amp.autocast_mode.autocast with the device fixed to cpu, decorated to
# emit a FutureWarning on use. Exports only autocast. No import-time side effects beyond
# applying the deprecation decorator.
import sys
from typing import Any
from typing_extensions import deprecated

import torch


__all__ = ["autocast"]


@deprecated(
    "`torch.cpu.amp.autocast(args...)` is deprecated. "
    "Please use `torch.amp.autocast('cpu', args...)` instead.",
    category=FutureWarning,
)
class autocast(torch.amp.autocast_mode.autocast):
    # [CLASS DOC] Deprecated CPU autocast context manager and function decorator. Takes
    # enabled, dtype (default bfloat16) and cache_enabled and delegates to the generic autocast
    # with device cpu. Has TorchScript-friendly branches: when scripting, it only records
    # enabled/device/fast_dtype and enter/exit/call become no-ops or pass-throughs.
    # On Python older than 3.13 it also defines __new__ and __init_subclass__ so that
    # signature introspection of the deprecated class works.
    r"""
    See :class:`torch.autocast`.
    ``torch.cpu.amp.autocast(args...)`` is deprecated. Please use ``torch.amp.autocast("cpu", args...)`` instead.
    """

    # TODO: remove this conditional once we stop supporting Python < 3.13
    # Prior to Python 3.13, inspect.signature could not retrieve the correct
    # signature information for classes decorated with @deprecated (unless
    # the __new__ static method was explicitly defined);
    #
    # However, this issue has been fixed in Python 3.13 and later versions.
    if sys.version_info < (3, 13):

        def __new__(
            cls,
            enabled: bool = True,
            dtype: torch.dtype = torch.bfloat16,
            cache_enabled: bool = True,
        ):
            # [FUNCTION DOC] HELPER: Constructs the instance without forwarding arguments to the base
            # class; exists only so inspect can see the correct signature (Python < 3.13). Returns the new object.
            return super().__new__(cls)

        def __init_subclass__(cls):
            # [FUNCTION DOC] HELPER: Intentionally does nothing; overrides the base hook so subclass creation is not affected by the deprecation wrapper.
            pass

    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        cache_enabled: bool = True,
    ):
        # [FUNCTION DOC] Initializes CPU autocast with enabled, dtype and cache_enabled. When running
        # under TorchScript it only stores the enabled flag, device cpu and fast dtype and returns;
        # otherwise calls the base initializer with device cpu.
        if torch._jit_internal.is_scripting():
            self._enabled = enabled
            self.device = "cpu"
            self.fast_dtype = dtype
            return
        super().__init__(
            "cpu", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled
        )

    def __enter__(self):
        # [FUNCTION DOC] Enters the autocast region; returns self under TorchScript, else delegates to the base class and returns its result.
        if torch._jit_internal.is_scripting():
            return self
        return super().__enter__()

    # TODO: discuss a unified TorchScript-friendly API for autocast
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        # [FUNCTION DOC] Exits the autocast region; a no-op under TorchScript, otherwise delegates exception info to the base class exit and returns its result.
        if torch._jit_internal.is_scripting():
            return
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, func):
        # [FUNCTION DOC] Decorator form: returns func unchanged under TorchScript, otherwise returns the base class wrapped version of func that runs inside autocast.
        if torch._jit_internal.is_scripting():
            return func
        return super().__call__(func)
