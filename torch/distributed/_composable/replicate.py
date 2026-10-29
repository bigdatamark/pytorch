# mypy: allow-untyped-defs
# [MODULE DOC] Composable replicate API: wraps a module's parameters in a DistributedDataParallel
# instance created lazily at first forward, via forward hooks and a dynamically created
# subclass mixing in DDP. Contains the state class, DDP mixin, replicate and a fully-shard check.
import weakref
from collections.abc import Iterable
from typing import Any, NoReturn

import torch
import torch.nn as nn
from torch.distributed._composable_state import _State
from torch.nn.parallel import DistributedDataParallel

from .contract import _get_registry, contract


_ROOT_MODULE_PREFIX = ""


class _ReplicateState(_State):
    # [CLASS DOC] HELPER: Per-module state of replicate. Holds the module, collected parameter
    # list and names, recorded (deferred) DDP init args, queued comm-hook args, the no-sync
    # flag, an initialized flag, and later the DistributedDataParallel instance (_ddp).
    _ddp_weakref: weakref.ref

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Sets defaults: placeholder module, not initialized, empty
        # parameter list/names, no deferred init args, sync enabled, no comm hooks.
        super().__init__()
        self.module: nn.Module = nn.ParameterList()
        self.has_initialized: bool = False
        self._param_list: nn.ParameterList = nn.ParameterList()
        # TODO(@fegin): this variable is originally created for testing, we
        # should remove this if possible.
        self._orig_module = self.module
        self._param_names: list[str] = []
        self._no_sync: bool = False
        self._init_args: tuple[Any, ...] | None = None
        self._init_kwargs: dict[str, Any] = {}
        self._comm_hook_args: list[Any] = []

    def _collect_params(
        self,
        module: nn.Module,
        ignored_modules: set[nn.Module],
        ignored_params: set[nn.Parameter],
        prefix: str = _ROOT_MODULE_PREFIX,
    ) -> None:
        # [FUNCTION DOC] HELPER: Recursively appends, in module order, each non-ignored
        # parameter of module and its children to the parameter list, recording dotted names
        # relative to the root. Skips modules managed by fully_shard and ignored modules
        # (with all their descendants). Mutates the state; returns nothing.
        # skip if managed by fully_sharded API
        if _is_fully_sharded(module):
            return

        # if a module is ignored, all descendants of the module are ignored.
        if module in ignored_modules:
            return

        recurse_prefix = (
            f"{prefix}." if prefix != _ROOT_MODULE_PREFIX else _ROOT_MODULE_PREFIX
        )

        for n, p in module.named_parameters(recurse=False):
            if p not in ignored_params:
                self._param_list.append(p)
                self._param_names.append(f"{recurse_prefix}{n}")

        for name, child_module in module.named_children():
            self._collect_params(
                child_module,
                ignored_modules,
                ignored_params,
                prefix=f"{recurse_prefix}{name}",
            )

    def lazy_init(self) -> None:
        # [FUNCTION DOC] HELPER: Performs the deferred initialization (under a dynamo-disabled
        # inner function): calls init with the recorded args, registers queued comm hooks, then
        # clears the recorded args. Raises AssertionError if no args were recorded.
        @torch._disable_dynamo(recursive=True)
        def _lazy_init():
            # [FUNCTION DOC] HELPER: Body of lazy_init, run with dynamo disabled.
            if self._init_args is None:
                raise AssertionError
            self.init(*self._init_args, **self._init_kwargs)
            self.register_comm_hook()
            self._init_args = ()
            self._init_kwargs = {}

        _lazy_init()

    def init(
        self,
        module: nn.Module,
        ignored_modules: set[nn.Module],
        **kwargs,
    ) -> None:
        # [FUNCTION DOC] HELPER: One-time setup (no-op if already initialized). Stores module,
        # gathers ignored parameters (ignored modules plus fully_shard submodules), localizes
        # DTensor parameters, collects parameters, converts a device_id keyword into DDP's
        # device_ids (None for CPU or missing, else a one-element list), and builds a
        # DistributedDataParallel over the parameter list with the remaining kwargs. Keeps a
        # weak reference to the DDP instance in the module's state (used in tests).
        if self.has_initialized:
            return

        self.has_initialized = True
        self.module = module
        ignored_params = {p for m in ignored_modules for p in m.parameters()}
        for submodule in module.modules():
            if _is_fully_sharded(submodule):
                ignored_params.update(submodule.parameters())
        from torch.distributed.tensor.parallel.ddp import _localize_dtensor

        _localize_dtensor(module, ignored_params=ignored_params)
        self._collect_params(module, ignored_modules, ignored_params)

        if "device_id" in kwargs:
            # replicate() supports a small usability enhancement where
            # user can pass in device_id as a Union[int, torch.device] even for
            # CPU devices so users don't have to change code for CPU/GPU runs.
            # We derive the right device_ids to feed into DDP to support this.
            if kwargs["device_id"] is not None:
                device_id = kwargs["device_id"]
                # Convert to device_ids that DDP expects.
                if isinstance(device_id, torch.device) and device_id.type == "cpu":
                    # CPU modules receive device_ids None
                    kwargs["device_ids"] = None
                else:
                    # GPU modules expect device_ids=[cuda_device]
                    kwargs["device_ids"] = [device_id]
            else:
                kwargs["device_ids"] = None
            kwargs.pop("device_id")

        self._ddp = DistributedDataParallel(self._param_list, **kwargs)
        # Weakref to the DDP instance is currently only used for testing.
        replicate.state(self.module)._ddp_weakref = weakref.ref(self._ddp)

    def register_comm_hook(self) -> None:
        # [FUNCTION DOC] HELPER: Applies every queued (args, kwargs) comm hook registration to
        # the DDP instance, then empties the queue.
        for comm_args, comm_kwargs in self._comm_hook_args:
            self._ddp.register_comm_hook(*comm_args, **comm_kwargs)
        self._comm_hook_args.clear()

    def record_init_args(self, *args, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Saves positional and keyword args for the deferred init.
        self._init_args = args
        self._init_kwargs = kwargs

    def forward_pre_hook(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        # [FUNCTION DOC] HELPER: Forward pre-hook. Runs lazy init if recorded args remain, sets
        # DDP's gradient sync flag from the no-sync setting, marks this DDP as the active one
        # (class-level global), and returns DDP's pre-forward result (new args and kwargs).
        if self._init_args or self._init_kwargs:
            self.lazy_init()
        self._ddp.require_backward_grad_sync = not self._no_sync
        DistributedDataParallel._active_ddp_module = self._ddp
        return self._ddp._pre_forward(*args, **kwargs)

    def forward_post_hook(
        self,
        module: nn.Module,
        input: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Forward post-hook. Clears the globally active DDP module and
        # returns the output as processed by DDP's post-forward.
        DistributedDataParallel._active_ddp_module = None
        return self._ddp._post_forward(output)


def unimplemented_deepcopy(*args: Any, **kwargs: Any) -> NoReturn:
    # [FUNCTION DOC] Installed as the deepcopy method of replicated modules; always raises
    # AssertionError telling users to use state dicts for serialization.
    raise AssertionError(
        "DDP does not support deepcopy. Please use state dict for serialization."
    )


# Follow the same pattern as FSDP/fully_shard
class DDP:
    # [CLASS DOC] Mixin placed first in the bases of the dynamic class created by replicate.
    # Adds set_requires_gradient_sync and register_comm_hook to the module and makes
    # construction yield the original class.
    def __new__(cls, *args, **kwargs):
        # [FUNCTION DOC] Constructs an instance of the original (fourth-level) base class
        # found in the method resolution order, bypassing the mixin classes.
        """
        Override ``__new__`` to remove the DDP class and directly construct
        the original class for cases like indexing into a container module.
        """
        # Use index 2 since 0 is the dynamically constructed `DDP<...>` class
        # and index 1 is the `DDP` class itself
        orig_cls = cls.__mro__[2]
        return orig_cls.__new__(orig_cls, *args, **kwargs)

    def set_requires_gradient_sync(self, requires_gradient_sync: bool) -> None:
        # [FUNCTION DOC] Enables or disables gradient all-reduce on backward by storing the
        # inverse in the module's replicate state (used for gradient accumulation).
        """
        Sets if the module should sync gradients. This can be used to implement
        gradient accumulation without communication.

        Args:
            requires_gradient_sync (bool): Whether to reduce gradients for the
                module's parameters.
        """
        replicate.state(self)._no_sync = not requires_gradient_sync  # type: ignore[arg-type]

    def register_comm_hook(self, *args, **kwargs) -> None:
        # [FUNCTION DOC] Queues a DDP communication hook registration (args, kwargs) in the
        # state; it is applied when DDP is lazily created.
        replicate.state(self)._comm_hook_args.append((args, kwargs))  # type: ignore[arg-type]


@contract(state_cls=_ReplicateState)
def replicate(
    module: nn.Module,
    ignored_modules: Iterable[torch.nn.Module] | None = None,
    **kwargs,
) -> nn.Module:
    # [FUNCTION DOC] Applies DDP-style replication to module in place and returns it. Logs API
    # usage; validates device_id (int or torch.device, else RuntimeError) and that the module is
    # not fully_shard managed (RuntimeError). Registers the state's forward pre-hook and post-hook
    # (plus DTensor reconstruct/localize hooks when device_mesh is a slice of a root mesh),
    # records the deferred init args (module, ignored modules, kwargs), and swaps the module class
    # with a new subclass of the DDP mixin and the original class whose deepcopy raises.
    r"""Replicates a module

    Args:
        module (torch.nn.Module): module to replicate

    Example::
        >>> # xdoctest: +REQUIRES(module:torch._C._distributed_c10d)
        >>> module = nn.Linear(3, 3)
        >>> replicate(module)
    """
    torch._C._log_api_usage_once("torch.distributed.replicate")

    # TODO(fegin): using kwargs is not a good idea if we would like to make
    # replicate a formal API to replace DDP.
    if "device_id" in kwargs:
        if not isinstance(kwargs["device_id"], (int, torch.device)):
            raise RuntimeError(
                "Expected device_id to be int or torch.device, "
                f"but got {type(kwargs['device_id'])}"
            )

    if _is_fully_sharded(module):
        raise RuntimeError(
            "Cannot apply `replicate()` on a Module already managed by `fully_shard`"
        )

    if ignored_modules is None:
        ignored_modules = {}
    else:
        ignored_modules = set(ignored_modules)

    state = replicate.state(module)
    module.register_forward_pre_hook(state.forward_pre_hook, with_kwargs=True)
    device_mesh = kwargs.get("device_mesh")
    if device_mesh is not None:
        root_mesh = device_mesh._get_root_mesh()
        # if a root mesh is not the same as device_mesh,
        # meaning the device_mesh is sliced out from the root mesh.
        if root_mesh != device_mesh:
            # TODO: This is a temporary work around to enable DDP + TP.
            # We should do the logic in DDP so that the 2D implementation is
            # sound and the state_dict works out of the box.
            #
            # This won't conflict with what is done in DDP class as the module
            # replicate is going to pass is NOT the original module.
            from torch.distributed.tensor.parallel.ddp import (
                _localize_dtensor,
                _reconstruct_dtensor,
            )

            module.register_forward_pre_hook(_reconstruct_dtensor)
            module.register_forward_hook(_localize_dtensor)

    module.register_forward_hook(state.forward_post_hook)  # type: ignore[arg-type]

    state.record_init_args(module, ignored_modules, **kwargs)

    # Place DDP leftmost for highest priority in the method resolution order
    cls = module.__class__
    dct = {"__deepcopy__": unimplemented_deepcopy}
    new_cls = type(f"DDP{cls.__name__}", (DDP, cls), dct)
    module.__class__ = new_cls
    return module


def _is_fully_sharded(module: nn.Module) -> bool:
    # [FUNCTION DOC] HELPER: True if the module's composable registry contains fully_shard.
    r"""Check if module is marked with fully_shard."""
    registry = _get_registry(module)
    if registry is None:
        return False
    return "fully_shard" in registry
