# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines a torch function mode used during export tracing that rejects
# operations which change the global autograd (grad enabled) state.
# No import-time side effects beyond imports.
import torch
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode
from torch.overrides import TorchFunctionMode


class AutogradStateOpsFailSafeguard(TorchFunctionMode):
    # [CLASS DOC]
    # Stateless TorchFunctionMode. While active, intercepts every torch function call
    # and raises if the call is the low-level set-grad-enabled op that would change
    # the grad state during non-pre-dispatch proxy tracing. Everything else passes through.
    """
    Detect grad state ops during exporting the graph and fail the process by
    raising an error, to avoid unexpected behavior. Those grad mode ops could be:
    `torch.no_grad`
    `torch.enable_grad`
    `torch.set_grad_enabled`

    Export with predispatch mode is exempted.
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        # [FUNCTION DOC]
        # Mode hook. Args: the called function, its types, positional args, keyword args
        # (None treated as empty). If func is the internal set-grad-enabled op, checks that
        # exactly one argument is given (raises AssertionError otherwise); if a proxy
        # dispatch mode is active and not in pre-dispatch, and the requested state differs
        # from the current grad state, raises RuntimeError advising a no_grad context.
        # Returns the result of calling func with the original arguments.
        kwargs = kwargs or {}
        unsupported_grad_mode_ops = [
            torch._C._set_grad_enabled,
        ]
        # It's only enabled while tracing, by confirming the torch dispatch mode is
        # any active PROXY. This is to allow the autograd ops out of tracing.
        current_state = torch._C.is_grad_enabled()
        if func in unsupported_grad_mode_ops:
            if len(args) != 1:
                raise AssertionError(
                    f"Expected exactly 1 argument for grad mode op, but got {len(args)}"
                )
            changed_state = args[0]
            mode = torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY)
            # Intend to check if it's not the pre_dispatch mode. It's allowed to use
            # autograd ops in pre_dispatch mode, e.g. `torch.no_grad`
            if (
                mode
                and isinstance(mode, ProxyTorchDispatchMode)
                and not mode.pre_dispatch
                and changed_state != current_state
            ):
                raise RuntimeError(
                    f"Encountered autograd state manager op {func} trying to change global autograd state "
                    "while exporting. This is unsafe because we don't capture this op in torch.export "
                    "today, hence we can't reflect the user intention soundly. You can fix this by "
                    "adding a torch.no_grad() context around the export call."
                )
        return func(*args, **kwargs)
