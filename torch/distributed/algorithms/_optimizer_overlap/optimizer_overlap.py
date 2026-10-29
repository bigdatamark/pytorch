# mypy: allow-untyped-defs
# [MODULE DOC] Registry and wrappers that fuse a regular optimizer with DDP gradient communication
# (optimizer overlap). Holds a module-level mapping from optimizer classes to overlapped wrappers.
import inspect
from abc import ABC, abstractmethod

from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from torch.distributed.algorithms.ddp_comm_hooks.optimizer_overlap_hooks import (
    _hook_then_optimizer,
    _OptimizerHookState,
)
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.optim import as_functional_optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer


# Contains the mappings between the regular and overlapped optimizer types.
_registered_overlapped_optims: dict[type, type] = {}


def register_overlapped(optim_cls):
    # [FUNCTION DOC] Class decorator factory: registers the decorated overlapped-optimizer class for
    # optim_cls in the global registry and returns the class unchanged.
    # KNOWN ISSUE: the duplicate check tests the target class as a key although keys are optimizer
    # classes, and the error message indexes the registry by optim_cls (may raise KeyError).
    def decorator(target_overlapped_optim_cls):
        # [FUNCTION DOC] HELPER: Perform the registration described above and return the target class.
        if target_overlapped_optim_cls in _registered_overlapped_optims:
            raise ValueError(
                f"{target_overlapped_optim_cls} already registered with optim_cls "
                f"{_registered_overlapped_optims[optim_cls]} {optim_cls}, trying to"
                f"re-register it for {optim_cls} is not supported."
            )
        _registered_overlapped_optims[optim_cls] = target_overlapped_optim_cls
        return target_overlapped_optim_cls

    return decorator


class OverlappedOptimizer(ABC):
# [CLASS DOC] Abstract base for wrappers that tell DDP/FSDP how to overlap a particular optimizer
# with communication. Stores optim_cls; subclasses implement register_ddp and register_fsdp.
    def __init__(self, optim_cls: type) -> None:
        # [FUNCTION DOC] Store the optimizer class on the instance.
        """
        Initialize the OverlappedOptimizer.

        Overlappedoptimizer is a base class that child classes can implement to
        specify how different optimizers will register themselves with DDP.
        """
        self.optim_cls = optim_cls

    @abstractmethod
    def register_ddp(self, ddp: DistributedDataParallel) -> None:
        # [FUNCTION DOC] Abstract: register the overlapped optimizer with a DDP instance. Base raises NotImplementedError.
        """Registers the overlapped optimizer with DDP."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support overlapped DDP."
        )

    @abstractmethod
    def register_fsdp(self, fsdp: FullyShardedDataParallel) -> None:
        # [FUNCTION DOC] Abstract: register the overlapped optimizer with an FSDP instance. Base raises NotImplementedError.
        """Registers the overlapped optimizer with FSDP."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support overlapped FSDP."
        )


@register_overlapped(Optimizer)
class _OverlappedStandardOptimizer(OverlappedOptimizer):
# [CLASS DOC] HELPER: Overlapped wrapper for any standard Optimizer; converts it to a functional
# optimizer and holds an _OptimizerHookState. Registered for the Optimizer base class (fallback).
    """Overlaps a regular ``Optimizer``."""

    def __init__(self, optim_cls: type, params, *optim_args, **optim_kwargs) -> None:
        # [FUNCTION DOC] Build a functional optimizer via as_functional_optim from the extra args and
        # create the optimizer hook state over params.
        super().__init__(optim_cls)
        f_optim = as_functional_optim(self.optim_cls, *optim_args, **optim_kwargs)
        self._opt_hook_state = _OptimizerHookState(f_optim, params)

    def register_ddp(self, ddp_inst: DistributedDataParallel):
        # [FUNCTION DOC] Register a DDP comm hook that runs allreduce then the fused optimizer step
        # (no wrapped hook state). Mutates ddp_inst.
        # NOTE: using a custom communication hook and fused optimizer is not
        # yet supported.
        ddp_inst.register_comm_hook(  # type: ignore[operator]
            None,  # wrapped hook state
            _hook_then_optimizer(allreduce_hook, self._opt_hook_state),
        )

    # TODO: register_fsdp once FSDP supports communication hook.
    def register_fsdp(self, fsdp: FullyShardedDataParallel) -> None:
        # [FUNCTION DOC] Not supported yet: always raises NotImplementedError.
        """Register the overlapped optimizer with FSDP."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support overlapped FSDP."
        )


def _as_overlapped_optim(optim_cls: type, params, *args, **kwargs):
    # [FUNCTION DOC] HELPER: Walk the MRO of optim_cls and instantiate the first registered overlapped
    # wrapper; if none match, fall back to the standard overlapped wrapper. Returns the instance.
    """Return a new ``OverlappedOptimizer`` instance that supports ``optim_cls``."""
    for clz in inspect.getmro(optim_cls):
        try:
            return _registered_overlapped_optims[clz](
                optim_cls, params, *args, **kwargs
            )
        except KeyError:
            pass

    # Fallback to standard overlapped optimizer, which will raise errors if user
    # is attempting to use an unsupported optimizer.
    return _OverlappedStandardOptimizer(optim_cls, params, *args, **kwargs)
