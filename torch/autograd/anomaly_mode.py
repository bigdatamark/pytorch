# mypy: allow-untyped-defs
# [MODULE DOC] Autograd anomaly detection context managers: detect_anomaly and set_detect_anomaly.
# Both toggle the global anomaly-mode flags (enabled, check_nan) through torch functions and restore
# the previous values on exit. Exports both names via __all__. No import-time side effects.
r"""Autograd anomaly mode."""

import warnings

import torch


__all__ = ["detect_anomaly", "set_detect_anomaly"]


class detect_anomaly:
    # [CLASS DOC] Context manager enabling autograd anomaly detection (forward traceback recording and
    # optional NaN checks in backward). State: prev and prev_check_nan (flags captured at construction)
    # and check_nan. Slows execution; for debugging only.
    r"""Context-manager that enables anomaly detection for the autograd engine.

    This does two things:

    - Running the forward pass with detection enabled will allow the backward
      pass to print the traceback of the forward operation that created the failing
      backward function.
    - If ``check_nan`` is ``True``, any backward computation that generates "nan"
      value will raise an error. Default ``True``.

    .. warning::
        This mode should be enabled only for debugging as the different tests
        will slow down your program execution.

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ANOMALY)
        >>> import torch
        >>> from torch import autograd
        >>> class MyFunc(autograd.Function):
        ...     @staticmethod
        ...     def forward(ctx, inp):
        ...         return inp.clone()
        ...
        ...     @staticmethod
        ...     def backward(ctx, gO):
        ...         # Error during the backward pass
        ...         raise RuntimeError("Some error in backward")
        ...         return gO.clone()
        >>> def run_fn(a):
        ...     out = MyFunc.apply(a)
        ...     return out.sum()
        >>> inp = torch.rand(10, 10, requires_grad=True)
        >>> out = run_fn(inp)
        >>> out.backward()
            Traceback (most recent call last):
              File "<stdin>", line 1, in <module>
              File "/your/pytorch/install/torch/_tensor.py", line 93, in backward
                torch.autograd.backward(self, gradient, retain_graph, create_graph)
              File "/your/pytorch/install/torch/autograd/__init__.py", line 90, in backward
                allow_unreachable=True)  # allow_unreachable flag
              File "/your/pytorch/install/torch/autograd/function.py", line 76, in apply
                return self._forward_cls.backward(self, *args)
              File "<stdin>", line 8, in backward
            RuntimeError: Some error in backward
        >>> with autograd.detect_anomaly():
        ...     inp = torch.rand(10, 10, requires_grad=True)
        ...     out = run_fn(inp)
        ...     out.backward()
            Traceback of forward call that caused the error:
              File "tmp.py", line 53, in <module>
                out = run_fn(inp)
              File "tmp.py", line 44, in run_fn
                out = MyFunc.apply(a)
            Traceback (most recent call last):
              File "<stdin>", line 4, in <module>
              File "/your/pytorch/install/torch/_tensor.py", line 93, in backward
                torch.autograd.backward(self, gradient, retain_graph, create_graph)
              File "/your/pytorch/install/torch/autograd/__init__.py", line 90, in backward
                allow_unreachable=True)  # allow_unreachable flag
              File "/your/pytorch/install/torch/autograd/function.py", line 76, in apply
                return self._forward_cls.backward(self, *args)
              File "<stdin>", line 8, in backward
            RuntimeError: Some error in backward

    """

    def __init__(self, check_nan=True) -> None:
        # [FUNCTION DOC] Capture the current anomaly-enabled and check-nan flags for later restoration,
        # store check_nan, and emit a warning that anomaly detection is on and slows runtime.
        # Does not enable anything itself; enabling happens in __enter__.
        self.prev = torch.is_anomaly_enabled()
        self.check_nan = check_nan
        self.prev_check_nan = torch.is_anomaly_check_nan_enabled()
        warnings.warn(
            "Anomaly Detection has been enabled. "
            "This mode will increase the runtime "
            "and should only be enabled for debugging.",
            stacklevel=2,
        )

    def __enter__(self) -> None:
        # [FUNCTION DOC] Turn anomaly detection on globally, with NaN checking per check_nan.
        torch.set_anomaly_enabled(True, self.check_nan)

    def __exit__(self, *args: object) -> None:
        # [FUNCTION DOC] Restore the anomaly flags captured at construction; ignores exception info.
        torch.set_anomaly_enabled(self.prev, self.prev_check_nan)


class set_detect_anomaly:
    # [CLASS DOC] Context manager and function that sets anomaly detection on or off.
    # State: prev and prev_check_nan hold the flags before construction, restored on exit.
    r"""Context-manager that sets the anomaly detection for the autograd engine on or off.

    ``set_detect_anomaly`` will enable or disable the autograd anomaly detection
    based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

    See ``detect_anomaly`` above for details of the anomaly detection behaviour.

    Args:
        mode (bool): Flag whether to enable anomaly detection (``True``),
                     or disable (``False``).
        check_nan (bool): Flag whether to raise an error when the backward
                          generate "nan"

    """

    def __init__(self, mode: bool, check_nan: bool = True) -> None:
        # [FUNCTION DOC] Record the current anomaly flags, then immediately apply the given mode and
        # check_nan globally (so it also works as a plain function call, with effect before entering).
        self.prev = torch.is_anomaly_enabled()
        self.prev_check_nan = torch.is_anomaly_check_nan_enabled()
        torch.set_anomaly_enabled(mode, check_nan)

    def __enter__(self) -> None:
        # [FUNCTION DOC] No-op; the new mode was already applied in the constructor.
        pass

    def __exit__(self, *args: object) -> None:
        # [FUNCTION DOC] Restore the anomaly flags recorded at construction; ignores exception info.
        torch.set_anomaly_enabled(self.prev, self.prev_check_nan)
