# [MODULE DOC] Debug helper that detects CUDA graph replay on tensors whose storage died after
# capture. While a dispatch mode is active, tensors read during stream capture that were not
# produced inside the capture are tracked via weak references to their storage, together with
# creation and deletion tracebacks. Contains the tracker, the tracked-tensor record and the
# dispatch mode. No import-time side effects.
from __future__ import annotations

import textwrap
import traceback
import weakref
from typing import Any, TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_iter


if TYPE_CHECKING:
    from torch.cuda import _POOL_HANDLE


class _TrackedTensorInfo:
    # [CLASS DOC] HELPER: Record for one externally-provided tensor seen during graph capture.
    # Holds the data pointer, device index, creation tracebacks (Python and C++), a deletion
    # traceback filled in when the storage is released, and a weak reference to the storage.
    # Aliveness means the storage still exists and still has the same data pointer.
    __slots__ = (
        "weakref",
        "creation_traceback_py",
        "creation_traceback_cpp",
        "deletion_traceback_py",
        "data_ptr",
        "device_index",
        "_storage_data_ptr",
    )

    def __init__(self, tensor: Tensor) -> None:
        # [FUNCTION DOC] HELPER: Snapshot the tensor: capture the Python and C++ stack at creation,
        # remember data pointer and device index, and register a weak reference to the storage
        # whose release callback records the deletion stack (errors in it are swallowed).
        from torch.utils import get_cpp_backtrace

        self.creation_traceback_py = "".join(traceback.format_stack()[:-3])
        self.creation_traceback_cpp = get_cpp_backtrace()
        self.deletion_traceback_py: str | None = None
        self.data_ptr = tensor.data_ptr()
        self.device_index = tensor.device.index

        def on_release(_: weakref.ref[torch.UntypedStorage]) -> None:
            # [FUNCTION DOC] HELPER: Weakref callback run when the storage is freed; stores the
            # current Python stack as the deletion traceback. Never raises (runs under GC).
            try:
                self.deletion_traceback_py = "".join(traceback.format_stack()[:-1])
            except Exception:
                pass  # don't raise under GC

        storage = tensor.untyped_storage()
        self._storage_data_ptr = storage.data_ptr()
        self.weakref: weakref.ref[torch.UntypedStorage] = weakref.ref(
            storage, on_release
        )

    def is_alive(self) -> bool:
        # [FUNCTION DOC] HELPER: Returns True if the tracked storage still exists and its data
        # pointer is unchanged from when it was recorded; False otherwise.
        storage = self.weakref()
        if storage is None:
            return False
        return storage.data_ptr() == self._storage_data_ptr


class _CUDAGraphInputLivenessTracker:
    # [CLASS DOC] HELPER: Tracks which tensors a CUDA graph capture reads from outside the capture
    # and later verifies they are still alive at replay time. State: mapping from data pointer to
    # tracked info for external inputs, set of data pointers produced inside the capture, cache of
    # memory snapshots per capture pool, and the active dispatch mode. Invariant: a pointer is in
    # at most one of external inputs and internal outputs. Key methods: start, stop, check_alive.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initialize empty tracking state; no dispatch mode is active yet.
        self._external_inputs: dict[int, _TrackedTensorInfo] = {}
        self._internal_outputs: set[int] = set()
        self._memory_snapshots: dict[_POOL_HANDLE, list[dict[str, Any]]] = {}
        self._dispatch_mode: _TensorTrackingMode | None = None

    def start(self) -> None:
        # [FUNCTION DOC] HELPER: Create the tracking dispatch mode and enter it, so subsequent
        # dispatched ops are observed. Side effect: pushes a mode on the dispatch stack.
        self._dispatch_mode = _TensorTrackingMode(self)
        self._dispatch_mode.__enter__()

    def stop(self) -> None:
        # [FUNCTION DOC] HELPER: Exit and clear the dispatch mode if one is active; otherwise
        # does nothing.
        if self._dispatch_mode is not None:
            mode, self._dispatch_mode = self._dispatch_mode, None
            mode.__exit__(None, None, None)

    def track_external_input(self, tensor: Tensor) -> None:
        # [FUNCTION DOC] HELPER: Record the tensor as an external input unless its data pointer
        # is already known as an internal output or already tracked. Mutates tracking state.
        data_ptr = tensor.data_ptr()
        if (
            data_ptr not in self._internal_outputs
            and data_ptr not in self._external_inputs
        ):
            self._external_inputs[data_ptr] = _TrackedTensorInfo(tensor)

    def mark_internal_output(self, tensor: Tensor) -> None:
        # [FUNCTION DOC] HELPER: Record the tensor's data pointer as produced inside the capture,
        # unless it is already tracked as an external input.
        data_ptr = tensor.data_ptr()
        if data_ptr not in self._external_inputs:
            self._internal_outputs.add(data_ptr)

    def check_alive(self, capture_pools: list[_POOL_HANDLE]) -> None:
        # [FUNCTION DOC] HELPER: Find external inputs that are no longer alive and whose memory
        # is not an active block in any of the given capture pools. If any exist, raises
        # RuntimeError with a report of up to five dead tensors (creation, deletion tracebacks)
        # plus the replay traceback. Returns nothing when all inputs are fine. May query and
        # cache memory snapshots.
        dead = [
            i
            for i in self._external_inputs.values()
            if not i.is_alive()
            and not any(
                self._is_tensor_from_capture_pool(i, pool) for pool in capture_pools
            )
        ]
        if not dead:
            return

        def fmt(label: str, tb: str | None) -> str:
            # [FUNCTION DOC] HELPER: Format a labelled, indented traceback section; empty string
            # when the traceback is missing or empty.
            return f"  {label}:\n{textwrap.indent(tb.strip(), '    ')}\n" if tb else ""

        parts = [f"CUDA graph replay detected {len(dead)} dead tensor(s).\n"]
        for i, info in enumerate(dead[:5], 1):
            parts.append(f"Dead tensor #{i} (data_ptr={info.data_ptr:#x}):\n")
            parts.append(fmt("Creation Traceback (Python)", info.creation_traceback_py))
            parts.append(fmt("Creation Traceback (C++)", info.creation_traceback_cpp))
            parts.append(fmt("Deletion Traceback (Python)", info.deletion_traceback_py))
        if len(dead) > 5:
            parts.append(f"  ... and {len(dead) - 5} more\n")
        parts.append(
            fmt("Replay Traceback (Python)", "".join(traceback.format_stack()[:-2]))
        )
        raise RuntimeError("".join(parts))

    def _get_memory_snapshot(self, capture_pool: _POOL_HANDLE) -> list[dict[str, Any]]:
        # [FUNCTION DOC] HELPER: Return the allocator memory snapshot (without traces) for the
        # pool, computing it once and caching it per pool handle.
        if capture_pool not in self._memory_snapshots:
            self._memory_snapshots[capture_pool] = torch.cuda.memory.memory_snapshot(
                capture_pool, include_traces=False
            )
        return self._memory_snapshots[capture_pool]

    def _is_tensor_from_capture_pool(
        self, tensor: _TrackedTensorInfo, capture_pool: _POOL_HANDLE
    ) -> bool:
        # [FUNCTION DOC] HELPER: True if the tracked tensor's data pointer lies inside an active
        # block of the pool's snapshot on the same device. False if the device is unknown or no
        # block matches.
        device = tensor.device_index
        if device is None:
            return False
        tensor_ptr = tensor.data_ptr
        for meminfo in self._get_memory_snapshot(capture_pool):
            if meminfo["device"] != device:
                continue
            for block in meminfo["blocks"]:
                addr = block["address"]
                if (
                    addr <= tensor_ptr < addr + block["size"]
                    and "active" in block["state"]
                ):
                    return True
        return False


class _TensorTrackingMode(TorchDispatchMode):
    # [CLASS DOC] HELPER: Dispatch mode that, only while the current stream is capturing a CUDA
    # graph and an op would run on CUDA, runs the op and reports its tensor inputs and outputs
    # to the owning tracker. Supports higher-order operators. State: reference to the tracker.
    supports_higher_order_operators = True

    def __init__(self, tracker: _CUDAGraphInputLivenessTracker) -> None:
        # [FUNCTION DOC] HELPER: Store the tracker that receives input and output reports.
        self._tracker = tracker

    # Mirrors the CUDA dispatch key selection rule: if at least one tensor
    # input is a CUDA tensor, the CUDA dispatch key is selected and the op
    # runs (and is captured) on GPU.
    # See aten/src/ATen/core/dispatch/DispatchKeyExtractor.h for details.
    def _selects_cuda_dispatch_key(self, values: object) -> bool:
        # [FUNCTION DOC] HELPER: True if any tensor in the nested structure is a CUDA tensor.
        for v in tree_iter(values):
            if isinstance(v, Tensor) and v.is_cuda:
                return True
        return False

    def __torch_dispatch__(
        self,
        func: object,
        types: object,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        # [FUNCTION DOC] HELPER: Run the op and return its result unchanged. If the current
        # stream is capturing and a CUDA tensor is among the inputs, additionally track the
        # inputs as external and mark the outputs as internal after the call.
        kwargs = kwargs or {}
        inputs = [args, kwargs]
        if torch.cuda.is_current_stream_capturing() and self._selects_cuda_dispatch_key(
            inputs
        ):
            out = func(*args, **kwargs)  # type: ignore[operator]
            self._track_inputs(inputs)
            self._mark_outputs(out)
        else:
            out = func(*args, **kwargs)  # type: ignore[operator]
        return out

    def _track_inputs(self, values: object) -> None:
        # [FUNCTION DOC] HELPER: Report each CUDA or pinned tensor with a nonzero data pointer
        # in the nested structure to the tracker as an external input.
        for v in tree_iter(values):
            if not isinstance(v, Tensor) or v.data_ptr() == 0:
                continue
            if v.is_cuda or v.is_pinned():
                self._tracker.track_external_input(v)

    def _mark_outputs(self, values: object) -> None:
        # [FUNCTION DOC] HELPER: Report each CUDA tensor in the nested structure to the tracker
        # as an internal output.
        for v in tree_iter(values):
            if isinstance(v, Tensor) and v.is_cuda:
                self._tracker.mark_internal_output(v)
