# mypy: allow-untyped-defs
# [MODULE DOC] HELPER: Glue for composing Tensor Parallel with DistributedDataParallel.
# Converts DTensor parameters of a TP-parallelized module to plain local parameters (tagged with
# their DTensor spec) and registers hooks that convert them back around forward. Exports nothing
# publicly (empty __all__). No import-time side effects.
from typing import Any

import torch.nn as nn
from torch.distributed.tensor.parallel._data_parallel_utils import (
    _flatten_tensor,
    _unflatten_tensor,
)


__all__ = []  # type: ignore[var-annotated]


def _get_submodule_n_params(module: nn.Module, path: str):
    # [FUNCTION DOC] HELPER: Splits a dotted parameter path. If it contains a dot, returns the
    # submodule at the parent path and the final path component; otherwise returns the module
    # and the path unchanged. Raises if the parent submodule does not exist.
    """
    Get submodule and the direct path of parameter from the module
    """
    if "." in path:
        path_list = path.split(".")
        parent_module_path = ".".join(path_list[:-1])
        module = module.get_submodule(parent_module_path)
        path = path_list[-1]
    return module, path


def _update_module_param(param_list: list[tuple[nn.Module, str, nn.Parameter]]):
    # [FUNCTION DOC] HELPER: For each (parent module, attribute name, new value) triple, deletes
    # the existing attribute and sets the new value in its place (mutates the modules). Raises
    # an assertion error with no message if an attribute is missing.
    """
    Update parameters within the module
    """
    for item in param_list:
        parent_module, module_path, t = item
        if not hasattr(parent_module, module_path):
            raise AssertionError
        delattr(parent_module, module_path)
        setattr(parent_module, module_path, t)


def _reconstruct_dtensor(module: nn.Module, _input: Any):
    # [FUNCTION DOC] HELPER: Forward pre-hook. For every parameter carrying a saved sharding
    # info attribute, rebuilds a DTensor from it and replaces the parameter on its owning
    # submodule. The hook input argument is ignored. Mutates the module.
    """
    Reconstruct DTensor parameters from local tensors
    """
    param_list = []
    # TODO: To add perf optimizations to this iterations
    for name, t in module.named_parameters():
        if hasattr(t, "_st_info"):
            dtensor = _unflatten_tensor(t, t._st_info)
            param_list.append((*_get_submodule_n_params(module, name), dtensor))
    _update_module_param(param_list)  # type: ignore[arg-type]


def _localize_dtensor(
    module: nn.Module, *_: Any, ignored_params: set[nn.Parameter] | None = None
):
    # [FUNCTION DOC] HELPER: Forward hook / direct call. For each parameter (skipping those in
    # ignored_params, default empty) that is a DTensor, replaces it by a new nn.Parameter of its
    # local tensor tagged with the DTensor spec in a sharding info attribute. Plain parameters
    # are untouched. Extra positional hook arguments are ignored. Mutates the module.
    """
    Convert DTensor parameters to local tensors
    """
    if ignored_params is None:
        ignored_params = set()
    param_list = []
    for name, param in module.named_parameters():
        if param in ignored_params:
            continue
        t, sharding_info = _flatten_tensor(param)
        if sharding_info is not None:
            t = nn.Parameter(t)
            t._st_info = sharding_info  # type: ignore[attr-defined]
            param_list.append((*_get_submodule_n_params(module, name), t))
    _update_module_param(param_list)  # type: ignore[arg-type]


def _pre_dp_module_transform(module: nn.Module):
    # [FUNCTION DOC] HELPER: Prepares a TP-parallelized module for DDP wrapping. Immediately
    # localizes DTensor parameters, then registers a forward pre-hook that reconstructs DTensors
    # and a forward hook that localizes them again. Mutates the module; returns nothing.
    # KNOWN ISSUE: the docstring example uses a public name without underscore, an old
    # PairwiseParallel style, and assigns a return value, but the code returns None.
    """
    Enable the composability between Tensor Parallelism (TP) and Data
    Parallelism(DP) in PyTorch when using DDP. We need to convert Parameters which
    are DTensors to local tensors before wrapping with data parallelism API.
    We then register two hooks, one for converting local tensors back to DTensor
    preforward and one to convert DTensors back to tensors after Forward. By
    integrating this way, we avoid any special handling of DTensor parameters by DDP
    and get DTensor's gradients propagated back to DP, e.g. gradient buckets of DDP.

    For now, this API only works with ``DistributedDataParallel``. It will later support
    other DP methods such as FSDP.

    Args:
        module (:class:`nn.Module`):
            Module which has been applied TP on.

    Example::
        >>> # xdoctest: +SKIP("distributed")
        >>> from torch.distributed.tensor.parallel import parallelize_module, PairwiseParallel
        >>> from torch.nn.parallel import DistributedDataParallel as DDP
        >>> from torch.distributed.tensor.parallel.ddp import pre_dp_module_transform
        >>>
        >>> # Define the module.
        >>> m = module(...)
        >>> parallelize_module(m, PairwiseParallel())
        >>> m = pre_dp_module_transform(m)
        >>> m = DDP(m)
        >>>
    """

    _localize_dtensor(module, None, None)
    # TODO: To add test cases and ensure that it works for nested modules
    module.register_forward_pre_hook(_reconstruct_dtensor)
    module.register_forward_hook(_localize_dtensor)
