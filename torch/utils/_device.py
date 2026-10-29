# mypy: allow-untyped-defs
# [MODULE DOC] Implements the default-device mechanism behind using a torch.device object as a
# context manager or decorator. Provides a TorchFunctionMode (DeviceContext) that injects a
# device keyword into factory functions when none is given, plus the decorator helpers.
# Holds the module global CURRENT_DEVICE. Called from C++ (torch/csrc/Device.cpp). No import-time
# side effects beyond defining names.
import functools

import torch
from torch._C import _len_torch_function_stack
from torch.overrides import _pop_mode, _push_mode, TorchFunctionMode
from torch.utils._contextlib import context_decorator


CURRENT_DEVICE: torch.device | None = None


@functools.lru_cache(1)
def _device_constructors():
    # [FUNCTION DOC] HELPER: Returns (cached after first call) the set of torch factory functions
    # that accept a device keyword and should receive the default device. Excludes the
    # "_like" functions and normal. Takes no arguments, has no side effects.
    return {
        # standard ones
        torch.empty,
        torch.empty_permuted,
        torch.empty_strided,
        torch.empty_quantized,
        torch.ones,
        torch.arange,
        torch.bartlett_window,
        torch.blackman_window,
        torch.eye,
        torch.fft.fftfreq,
        torch.fft.rfftfreq,
        torch.full,
        torch.hamming_window,
        torch.hann_window,
        torch.kaiser_window,
        torch.linspace,
        torch.logspace,
        torch.nested.nested_tensor,
        # This function doesn't actually take a device argument
        # torch.normal,
        torch.rand,
        torch.randn,
        torch.randint,
        torch.randperm,
        torch.range,
        torch.sparse_coo_tensor,
        torch.sparse_compressed_tensor,
        torch.sparse_csr_tensor,
        torch.sparse_csc_tensor,
        torch.sparse_bsr_tensor,
        torch.sparse_bsc_tensor,
        torch.tril_indices,
        torch.triu_indices,
        torch.zeros,
        torch.asarray,
        # weird ones
        torch.tensor,
        torch.as_tensor,
        torch.scalar_tensor,
        # *_like may contain device kwarg, but the user implicitly
        # expects a specific device even when kwarg unused.
        # torch.zeros_like,
        # torch.randint_like,
        # torch.randn_like,
        # torch.ones_like,
        # torch.full_like,
        # torch.empty_like,
    }


# NB: This is directly called from C++ in torch/csrc/Device.cpp
class DeviceContext(TorchFunctionMode):
    # [CLASS DOC] Torch function mode that makes a given device the default for factory
    # functions. State: device (a torch.device), prev_mode (an enclosing DeviceContext that was
    # displaced on entry, or None), old_device (saved previous CURRENT_DEVICE, set on enter).
    # Invariant: while active, the single DeviceContext sits at the bottom of the torch function
    # mode stack. Key methods: __enter__, __exit__, __torch_function__.
    def __init__(self, device) -> None:
        # [FUNCTION DOC] Converts the device argument (string, int or device) to a torch.device
        # and stores it; initializes prev_mode to None. Raises if the device is invalid.
        self.device = torch.device(device)
        self.prev_mode: DeviceContext | None = None

    def __enter__(self):
        # [FUNCTION DOC] Activates the mode: saves and replaces the global CURRENT_DEVICE, then
        # pops every mode off the torch function stack, pushes this mode at the bottom and
        # re-pushes the others above it. If an older DeviceContext is found it is recorded as
        # prev_mode and dropped from the stack. Returns None.
        global CURRENT_DEVICE
        self.old_device = CURRENT_DEVICE
        CURRENT_DEVICE = self.device
        # We need to put the device at the bottom of the stack
        # If we set default device within a function mode context
        # exiting that context mode will pop the device function mode off
        # of the stack incorrectly
        cur_stack = [_pop_mode() for _ in range(_len_torch_function_stack())]

        _push_mode(self)

        for mode in reversed(cur_stack):
            if isinstance(mode, DeviceContext):
                self.prev_mode = mode
            else:
                _push_mode(mode)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # [FUNCTION DOC] Deactivates the mode: restores CURRENT_DEVICE, pops all modes above the
        # bottom one (raising AssertionError if any is a DeviceContext), pops the bottom mode
        # (raising AssertionError if it is not a DeviceContext), re-pushes prev_mode if present,
        # then re-pushes the other modes in their original order. Does not suppress exceptions.
        global CURRENT_DEVICE
        CURRENT_DEVICE = self.old_device
        cur_stack = []
        # Invariant: there should only be one DeviceContext on the stack at any time
        # (At the bottom), pop all modes until we hit the bottom, assert it's a DeviceContext
        # or else someone else has popped it!
        for _ in range(_len_torch_function_stack() - 1):
            mode = _pop_mode()
            if isinstance(mode, DeviceContext):
                raise AssertionError(
                    "Found nested DeviceContext on the mode stack where none expected"
                )
            cur_stack.append(mode)

        if _len_torch_function_stack() > 0:
            mode = _pop_mode()
            if not isinstance(mode, DeviceContext):
                raise AssertionError(
                    "Expected a DeviceContext at the bottom of the mode stack"
                )
        if self.prev_mode is not None:
            _push_mode(self.prev_mode)

        for mode in reversed(cur_stack):
            _push_mode(mode)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] Intercepts torch function calls. If the function is a known device
        # constructor and no device keyword (or None) was passed, sets the keyword to this
        # mode's device. Calls the function with the possibly updated arguments and returns its
        # result.
        kwargs = kwargs or {}
        if func in _device_constructors() and kwargs.get("device") is None:
            kwargs["device"] = self.device
        return func(*args, **kwargs)


# NB: This is directly called from C++ in torch/csrc/Device.cpp
def device_decorator(device, func):
    # [FUNCTION DOC] Wraps func so that each call runs inside the given device used as a
    # context manager. Returns the wrapped callable.
    return context_decorator(lambda: device, func)


def set_device(device):
    # [FUNCTION DOC] Takes a device specifier and returns a decorator that wraps a function so
    # it runs with that default device. The device is converted to a torch.device eagerly.
    """
    Set the default device inside of the wrapped function by decorating it with this function.

    If you would like to use this as a context manager, use device as a
    context manager directly, e.g., ``with torch.device(device)``.
    """
    return lambda func: device_decorator(torch.device(device), func)
