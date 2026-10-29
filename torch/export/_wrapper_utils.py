# [MODULE DOC] HELPER: Defines a tiny nn.Module wrapper that turns an arbitrary callable into a module,
# so plain functions can be passed to APIs (such as export) that require a Module. No import-time side effects.
import torch


class _WrapperModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Module that stores a callable in attribute f and whose forward simply delegates to it.
    # Holds no parameters or buffers of its own.
    def __init__(self, f):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] HELPER: Initializes the Module base class and stores the callable f as attribute f.
        super().__init__()
        self.f = f

    def forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] HELPER: Calls the wrapped callable with all given positional and keyword arguments and returns its result unchanged.
        return self.f(*args, **kwargs)
