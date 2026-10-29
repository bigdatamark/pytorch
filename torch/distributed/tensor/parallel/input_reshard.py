# Copyright (c) Meta Platforms, Inc. and affiliates
# [MODULE DOC] Input resharding for Tensor Parallel activation checkpointing: provides
# input_reshard, which installs saved-tensor hooks on a module so inputs saved for backward
# are stored sharded across the TP mesh and restored (replicated) on unpack. Private pack and
# unpack hook functions live here too. No import-time side effects.
from functools import partial
from typing import Any

import torch
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _is_shard_like


__all__ = [
    "input_reshard",
]


def input_reshard(
    module: torch.nn.Module,
    tp_device_mesh: DeviceMesh,
    input_reshard_dim: int | None = None,
) -> torch.nn.Module:
    # [FUNCTION DOC] Registers a forward pre-hook and a forward hook on module. The pre-hook
    # enters a saved-tensors-hooks context whose pack step shards saved tensors along
    # input_reshard_dim over tp_device_mesh and whose unpack step restores them. The forward
    # hook exits that context. If input_reshard_dim is None nothing is registered. Returns the
    # same module (mutated by hook registration).
    # KNOWN ISSUE: the context is kept in one shared closure variable, so re-entrant or
    # concurrent forwards of the same module would overwrite/exit the wrong context.
    """
    Register hooks to an nn.Module for input resharding, enabling sharding and restoration during backward computation.

    Register hooks to an nn.Module with input resharding so that we can shard
    per the given `tp_device_mesh` and `input_reshard_dim` and restore the
    input back when recomputing the activations in the backward. The reason
    why we can do this is that for Tensor Parallel(TP), the input are same
    across all TP ranks.

    Args:
        module (:class:`nn.Module`):
            Module to be registered with input resharding.
        tp_device_mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology
            of devices for Tensor Parallel.
        input_reshard_dim (Optional[int]):
            The dimension of where we perform the sharding
            of input. If set None, there is no sharding of input.
            Default: None

    Return:
        A :class:`nn.Module` object registered with TP input resharding.
    """
    if input_reshard_dim is None:
        return module

    cx: torch.autograd.graph.saved_tensors_hooks | None = None

    def input_reshard_forward_pre_hook(_: torch.nn.Module, _i: tuple[Any, ...]) -> None:
        # [FUNCTION DOC] HELPER: Forward pre-hook; builds saved tensor hooks from the pack and
        # unpack functions, enters the context manually and stores it in the enclosing
        # variable cx. Returns nothing.
        saved_tensor_hooks = torch.autograd.graph.saved_tensors_hooks(
            partial(_pack_hook_tp, tp_device_mesh, input_reshard_dim),
            partial(_unpack_hook_tp, tp_device_mesh, input_reshard_dim),
        )
        saved_tensor_hooks.__enter__()
        nonlocal cx
        cx = saved_tensor_hooks  # type: ignore[name-defined]

    def input_reshard_backward_hook(
        _: torch.nn.Module, _i: tuple[Any, ...], _o: Any
    ) -> Any:
        # [FUNCTION DOC] HELPER: Forward hook (despite its name, runs after forward, not
        # backward); exits the saved-tensors-hooks context stored in cx. Returns None, so the
        # module output is unchanged. Raises if the pre-hook never ran (cx is None).
        nonlocal cx
        cx.__exit__()  # type: ignore[name-defined, union-attr]

    module.register_forward_pre_hook(input_reshard_forward_pre_hook)
    module.register_forward_hook(input_reshard_backward_hook)
    return module


def _pack_hook_tp(mesh: DeviceMesh, input_reshard_dim: int, x: torch.Tensor) -> Any:
    # [FUNCTION DOC] HELPER: Saved-tensor pack hook. A DTensor fully replicated on its
    # placements is redistributed to be sharded along input_reshard_dim on mesh. A plain tensor
    # with at least as many elements as the mesh size is wrapped as a DTensor (default
    # placement), redistributed to a shard on that dim, and its local shard returned. Anything
    # else is returned unchanged.
    """Hook function called after FWD to shard input."""
    if isinstance(x, DTensor) and all(p.is_replicate() for p in x._spec.placements):
        return x.redistribute(device_mesh=mesh, placements=[Shard(input_reshard_dim)])
    elif (
        not isinstance(x, DTensor)
        and isinstance(x, torch.Tensor)
        and x.numel() >= mesh.size()
    ):
        return (
            DTensor.from_local(x, device_mesh=mesh)
            .redistribute(device_mesh=mesh, placements=[Shard(input_reshard_dim)])
            .to_local()
        )
    else:
        return x


def _unpack_hook_tp(mesh: DeviceMesh, input_reshard_dim: int, x: Any) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Saved-tensor unpack hook, inverse of the pack hook. A DTensor with
    # a single shard-like placement is redistributed to replicated. A plain tensor with at least
    # as many elements as the mesh size is treated as a local shard along input_reshard_dim,
    # redistributed to replicated and returned as a local tensor. Otherwise returned unchanged.
    # KNOWN ISSUE: the element-count test is applied to the local shard, so a tensor packed
    # unchanged (small) but later passing the check could be wrongly treated as sharded.
    """Hook function called before activation recomputing in BWD to restore input."""
    if (
        isinstance(x, DTensor)
        and len(x._spec.placements) == 1
        and _is_shard_like(x._spec.placements[0])
    ):
        return x.redistribute(device_mesh=mesh, placements=[Replicate()])
    elif (
        not isinstance(x, DTensor)
        and isinstance(x, torch.Tensor)
        and x.numel() >= mesh.size()
    ):
        return (
            DTensor.from_local(
                x, device_mesh=mesh, placements=[Shard(input_reshard_dim)]
            )
            .redistribute(device_mesh=mesh, placements=[Replicate()])
            .to_local()
        )
    else:
        return x
