# [MODULE DOC] Public API of the lazy tensor core (LTC) package: thin Python wrappers over torch._C._lazy.
# Provides mark_step, wait_device_ops, sync_multi, get_tensor_id, to_cpu and save, and re-exports
# add_step_closure and run_step_closures from .closure. Importing loads the lazy C extension.
import pickle
from collections.abc import Sequence
from types import ModuleType
from typing import cast, TypeVar

import torch
import torch._C._lazy
from torch.serialization import DEFAULT_PROTOCOL
from torch.types import FileLike
from torch.utils._pytree import tree_flatten, tree_unflatten

from .closure import add_step_closure, run_step_closures


_T = TypeVar("_T")


def mark_step(device: str = "", wait: bool = False) -> None:
    # [FUNCTION DOC] Ends the current lazy trace: compiles/caches live lazy tensors' IR and starts execution on the
    # given device (empty string means default). If wait is true, blocks until CPU-side execution finishes.
    # Afterwards runs all registered step closures. Returns None.
    """Triggers a mark step, which amounts to
    - collecting a group of 'live' lazy tensors to index into the compilation cache
      (lowering/compiling their IR graphs if not cached)
    - kicking off execution of the compiled function
    - (optionally, wait=True) waiting for cpu-side execution to complete (does not sync the accelerator)
    """
    # TODO(whc) expand this to include backend hooks and align with XLA backend needs
    torch._C._lazy._mark_step(device, [], wait=wait)

    run_step_closures()


def wait_device_ops(devices: list[str] | None = None) -> None:
    # [FUNCTION DOC] Blocks until async operations on the given devices complete; None or empty means all local devices.
    """Waits for all the async operations on the given devices to complete.
    Args:
      devices (string..., optional): The devices whose async ops need to be waited
        for. If empty, all the local devices will be waited for.
    """
    if devices is None:
        devices = []
    torch._C._lazy._wait_device_ops(devices=devices)


def sync_multi(tensors: Sequence[torch.Tensor], devices: list[str]) -> None:
    # [FUNCTION DOC] Forces lowering/compilation of the IR of the given lazy tensors on the listed devices
    # so the compiled graph is cached. Returns None.
    """
    Sync the list of lazy tensors so there IR get lowered for the activate backend
    and the compiled computation graph get cached.
    """
    torch._C._lazy._sync_multi(list(tensors), devices)


def get_tensor_id(tensor: torch.Tensor) -> int:
    # [FUNCTION DOC] Returns the integer id the lazy tensor core assigned to the given lazy tensor.
    """Return a unique id of the lazy tensor maintained by LTC"""
    return torch._C._lazy._get_tensor_id(tensor)


def to_cpu(tensors: _T, devices: list[str] | None = None) -> _T:
    # [FUNCTION DOC] Syncs all tensors in an arbitrary nested structure (default device list is lazy), copies each to
    # CPU, and returns a structure of the same shape holding the CPU tensors.
    devices = devices or ["lazy"]

    flattened, spec = tree_flatten(tensors)
    sync_multi(flattened, devices)
    return cast(_T, tree_unflatten([t.to("cpu") for t in flattened], spec))


def save(
    tensors: object,
    f: FileLike,
    pickle_module: ModuleType = pickle,
    pickle_protocol: int = DEFAULT_PROTOCOL,
    _use_new_zipfile_serialization: bool = True,
    _disable_byteorder_record: bool = False,
) -> None:
    # [FUNCTION DOC] Saves a (nested) structure of lazy tensors to f: converts to CPU via to_cpu, then delegates to
    # torch.save with the given pickle module, protocol and private serialization flags. Returns None.
    torch.save(
        to_cpu(tensors),
        f,
        pickle_module=pickle_module,
        pickle_protocol=pickle_protocol,
        _use_new_zipfile_serialization=_use_new_zipfile_serialization,
        _disable_byteorder_record=_disable_byteorder_record,
    )
