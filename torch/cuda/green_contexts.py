# [MODULE DOC] CUDA green context support: the GreenContext class, which partitions SMs and/or
# configures workqueues via the cuda.bindings driver API and hands out streams bound to the context.
# Also holds driver-version and platform checks and workqueue scope parsing. Import-time side
# effect: none beyond importing cuda bindings helpers.
from __future__ import annotations

import functools
import sys
import warnings
from typing import Any
from typing_extensions import deprecated

import torch
from torch.cuda._utils import (
    _check_cuda_bindings,
    _cuda_bindings_driver as _drv,
    _cuda_bindings_runtime as _rt,
    _HAS_CUDA_BINDINGS,
)


__all__ = [
    "GreenContext",
]

_STREAMS_PER_GREEN_CONTEXT_POOL = 32

_WORKQUEUE_SCOPE_VALUES = {
    "device_ctx": 0,
    "balanced": 1,
}

_CONTEXT_STACK_DEPRECATION = (
    "`GreenContext.set_context` and `GreenContext.pop_context` are deprecated. "
    "Please create a stream with `GreenContext.Stream()` and use "
    "`torch.cuda.stream(stream)` instead."
)


# note: this can safely be cached in a process/thread because
# the driver version cannot change during the lifetime of a process
@functools.cache
def _get_driver_version() -> int:
    # [FUNCTION DOC] HELPER: Query and cache the CUDA driver version as an integer. On a query error,
    # emits a warning and returns -1 (the cached result is then -1 for the process).
    try:
        # pyrefly: ignore [missing-attribute]
        return _check_cuda_bindings(_drv.cuDriverGetVersion())
    except RuntimeError as e:
        warnings.warn(f"Error while querying CUDA driver version: {e}")
        return -1


def _ensure_driver_version(version: int, message: str) -> None:
    # [FUNCTION DOC] HELPER: Raise RuntimeError with the message if the driver version is unknown or
    # below the required version.
    drv_version = _get_driver_version()
    if drv_version < 0 or drv_version < version:
        raise RuntimeError(message)


def _ensure_supported() -> None:
    # [FUNCTION DOC] HELPER: Raise RuntimeError unless running on Nvidia CUDA (not ROCm), not on
    # Windows, with cuda.bindings available and a driver at least 12.8.
    if torch.version.cuda is None or torch.version.hip is not None:
        raise RuntimeError("Green Context is only supported on Nvidia CUDA")
    if sys.platform == "win32":
        raise RuntimeError("Green Context is not supported on Windows")
    if not _HAS_CUDA_BINDINGS:
        raise RuntimeError("GreenContext requires the cuda.bindings package")
    _ensure_driver_version(12080, "Green Context requires user mode driver 12.8+")


def _ensure_workqueue_supported() -> None:
    # [FUNCTION DOC] HELPER: Raise RuntimeError unless the driver is at least 13.1 (needed for
    # workqueue configuration).
    _ensure_driver_version(
        13010, "CUDA user mode driver too old to use workqueue configuration!"
    )


def _parse_workqueue_scope(workqueue_scope: str | None) -> int | None:
    # [FUNCTION DOC] HELPER: Map the scope name to its numeric value (device context 0, balanced 1).
    # Returns None for None; raises ValueError for any other name.
    if workqueue_scope is None:
        return None
    if workqueue_scope not in _WORKQUEUE_SCOPE_VALUES:
        raise ValueError(
            "workqueue_scope must be 'device_ctx' or 'balanced', "
            f"got '{workqueue_scope}'"
        )
    return _WORKQUEUE_SCOPE_VALUES[workqueue_scope]


class GreenContext:
    # [CLASS DOC] Wrapper around a CUDA green context (a partition of device SMs and/or workqueue
    # configuration). State: device id, green context and regular context driver handles, an optional
    # parent stream used by the deprecated set/pop context pair, and a fixed-size pool of lazily
    # created streams used round-robin (index of the most recently handed-out stream tracked).
    # Key methods: Stream (get a stream bound to the context), create, max_workqueue_concurrency.
    # Destruction releases the pooled streams and the green context.
    r"""Wrapper around a CUDA green context.

    .. warning::
       This API is in beta and may change in future releases.

    CUDA work should be placed on streams created from the green context:

    .. code-block:: python

        ctx = GreenContext(...)
        stream = ctx.Stream()
        with torch.cuda.stream(stream):
            # torch operations here are using resources from `ctx`
            pass

    Green-context streams are custom CUDA streams. Synchronization with other
    streams is the user's responsibility and should be handled with CUDA events,
    as with any other custom stream.
    """

    def __init__(
        self,
        *,
        num_sms: int | None = None,
        workqueue_scope: str | None = None,
        workqueue_concurrency_limit: int | None = None,
        device_id: int | None = None,
    ) -> None:
        # [FUNCTION DOC] Create a green context. Validates platform and driver support, requires at
        # least one of num_sms or workqueue_scope (and a scope when a concurrency limit is given),
        # ensures a primary context exists (warns and creates one if not), builds SM split and/or
        # workqueue resources, creates the green context and its regular context handle, and destroys
        # the green context if conversion fails. Raises RuntimeError or ValueError on bad arguments or
        # driver failures. Uses the current device when device_id is None.
        r"""Create a CUDA green context.

        At least one of ``num_sms`` or ``workqueue_scope`` must be specified.
        Both can be combined to partition SMs and configure workqueues in the
        same green context.

        Arguments:
            num_sms (int, optional): The number of SMs to use in the green
                context. When ``None``, SMs are not partitioned.
            workqueue_scope (str, optional): Workqueue sharing scope. One of
                ``"device_ctx"`` (shared across all contexts, default driver
                behavior) or ``"balanced"`` (non-overlapping workqueues with
                other balanced green contexts). When ``None``, no workqueue
                configuration is applied.
            workqueue_concurrency_limit (int, optional): Maximum number of
                concurrent stream-ordered workloads for the workqueue. Requires
                ``workqueue_scope`` to be set.
            device_id (int, optional): The device index of green context.
                When ``None``, the current device is used.
        """
        self._device_id = None
        self._green_ctx = None
        self._context = None
        _ensure_supported()

        scope_value = _parse_workqueue_scope(workqueue_scope)
        if scope_value is not None:
            _ensure_workqueue_supported()

        if num_sms is None and scope_value is None:
            raise RuntimeError(
                "At least one of num_sms or workqueue_scope must be specified"
            )
        if workqueue_concurrency_limit is not None and scope_value is None:
            raise RuntimeError(
                "workqueue_concurrency_limit requires workqueue_scope to be set"
            )

        if device_id is None:
            device_id = torch.cuda.current_device()

        # pyrefly: ignore [missing-attribute]
        current_ctx = _check_cuda_bindings(_drv.cuCtxGetCurrent())

        if int(current_ctx) == 0:
            warnings.warn(
                "Attempted to create a green context but there was no primary "
                "context! Creating a primary context...",
                stacklevel=2,
            )
            # pyrefly: ignore [missing-attribute]
            _check_cuda_bindings(_rt.cudaFree(0))

        # pyrefly: ignore [missing-attribute]
        device = _check_cuda_bindings(_drv.cuDeviceGet(device_id))
        resources = []

        if num_sms is not None:
            sm_resource = _check_cuda_bindings(
                _drv.cuDeviceGetDevResource(  # pyrefly: ignore [missing-attribute]
                    device,
                    _drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,  # pyrefly: ignore [missing-attribute]
                )
            )
            if num_sms <= 0 or num_sms > sm_resource.sm.smCount:
                raise RuntimeError(
                    "Invalid number of SMs requested for green context: "
                    f"{num_sms} (device has {sm_resource.sm.smCount} SMs)"
                )
            split_result, nb_groups, _remaining = _check_cuda_bindings(
                _drv.cuDevSmResourceSplitByCount(  # pyrefly: ignore [missing-attribute]
                    1, sm_resource, 0, num_sms
                )
            )
            if nb_groups != 1:
                raise RuntimeError("Failed to create single SM resource group")
            resources.append(split_result[0])

        if scope_value is not None:
            wq_resource = _check_cuda_bindings(
                _drv.cuDeviceGetDevResource(  # pyrefly: ignore [missing-attribute]
                    device,
                    _drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG,  # pyrefly: ignore [missing-attribute]
                )
            )
            wq_resource.wqConfig.sharingScope = scope_value
            if workqueue_concurrency_limit is not None:
                wq_resource.wqConfig.wqConcurrencyLimit = workqueue_concurrency_limit
            resources.append(wq_resource)

        desc = _check_cuda_bindings(
            _drv.cuDevResourceGenerateDesc(  # pyrefly: ignore [missing-attribute]
                resources, len(resources)
            )
        )
        green_ctx = _check_cuda_bindings(
            _drv.cuGreenCtxCreate(  # pyrefly: ignore [missing-attribute]
                desc,
                device,
                _drv.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM,  # pyrefly: ignore [missing-attribute]
            )
        )
        try:
            # pyrefly: ignore [missing-attribute]
            context = _check_cuda_bindings(_drv.cuCtxFromGreenCtx(green_ctx))
            if int(context) == 0:
                raise RuntimeError("Green ctx conversion to regular ctx failed!")
        except Exception:
            # pyrefly: ignore [missing-attribute]
            _check_cuda_bindings(_drv.cuGreenCtxDestroy(green_ctx))
            raise

        self._init_from_cuda_objects(device_id, green_ctx, context)

    def __del__(self) -> None:
        # [FUNCTION DOC] Finalizer: if a green context is held, destroy every created pooled stream
        # (newest first) and then the green context, converting driver errors into warnings so nothing
        # is raised during garbage collection.
        green_ctx = getattr(self, "_green_ctx", None)
        if green_ctx is None:
            return

        # attempt to destroy streams related to this green context
        # we ignore errors to avoid leaking exceptions during __del__
        end = min(_STREAMS_PER_GREEN_CONTEXT_POOL, self._curr_stream_idx + 1)
        for idx in reversed(range(end)):
            try:
                # pyrefly: ignore [missing-attribute]
                _check_cuda_bindings(_drv.cuStreamDestroy(self._green_ctx_streams[idx]))
            except RuntimeError as e:
                warnings.warn(
                    f"Error while destroying green context stream at idx {idx} "
                    f"for green context {green_ctx}: {e}"
                )
        self._green_ctx = None
        try:
            # pyrefly: ignore [missing-attribute]
            _check_cuda_bindings(_drv.cuGreenCtxDestroy(green_ctx))
        except RuntimeError as e:
            warnings.warn(f"Error while destroying green context {green_ctx}: {e}")

    def _init_from_cuda_objects(self, device_id: int, green_ctx, context) -> None:
        # [FUNCTION DOC] HELPER: Store the device id and driver handles and reset the parent stream,
        # the empty stream pool and the stream index.
        self._device_id = device_id
        self._green_ctx = green_ctx
        self._context = context
        self._parent_stream: torch.cuda.Stream | None = None
        self._green_ctx_streams: list[Any | None] = [
            None
        ] * _STREAMS_PER_GREEN_CONTEXT_POOL
        self._curr_stream_idx = -1

    @staticmethod
    def create(
        *,
        num_sms: int | None = None,
        workqueue_scope: str | None = None,
        workqueue_concurrency_limit: int | None = None,
        device_id: int | None = None,
    ) -> GreenContext:
        # [FUNCTION DOC] Compatibility factory: forwards all arguments to the constructor and returns
        # the new green context.
        r"""Create a CUDA green context.

        Kept for compatibility, see `GreenContext` constructor.
        """
        return GreenContext(
            num_sms=num_sms,
            workqueue_scope=workqueue_scope,
            workqueue_concurrency_limit=workqueue_concurrency_limit,
            device_id=device_id,
        )

    @staticmethod
    def max_workqueue_concurrency(device_id: int | None = None) -> int:
        # [FUNCTION DOC] Query the device's default workqueue concurrency limit (current device when
        # None). Requires platform and driver 13.1+ support; raises RuntimeError otherwise. Returns int.
        r"""Return the maximum workqueue concurrency limit for the device.

        This queries the device for the default number of concurrent
        stream-ordered workloads supported by workqueue configuration
        resources.

        Arguments:
            device_id (int, optional): The device index to query. When
                ``None``, the current device is used.
        """
        _ensure_supported()
        _ensure_workqueue_supported()
        if device_id is None:
            device_id = torch.cuda.current_device()

        # pyrefly: ignore [missing-attribute]
        device = _check_cuda_bindings(_drv.cuDeviceGet(device_id))
        wq_resource = _check_cuda_bindings(
            _drv.cuDeviceGetDevResource(  # pyrefly: ignore [missing-attribute]
                device,
                _drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG,  # pyrefly: ignore [missing-attribute]
            )
        )
        return wq_resource.wqConfig.wqConcurrencyLimit

    def _ensure_alive(self) -> None:
        # [FUNCTION DOC] HELPER: Raise RuntimeError if the green context or its context handle is gone.
        if self._green_ctx is None or self._context is None:
            raise RuntimeError("GreenContext has been destroyed")

    @deprecated(_CONTEXT_STACK_DEPRECATION, category=FutureWarning)
    def set_context(self) -> None:
        # [FUNCTION DOC] Deprecated. Remember the current stream as parent, record an event on it, make
        # the green context current (set if none, else push on the stack), make the device default
        # stream current after waiting on the event. Raises RuntimeError if destroyed or called twice
        # without pop_context.
        r"""Make the green context the current context.

        Deprecated. Create streams with :meth:`Stream` and use
        :func:`torch.cuda.stream` instead.
        """
        self._ensure_alive()
        if self._parent_stream is not None:
            raise RuntimeError("set_context called twice before pop_context")
        current_stream = torch.cuda.current_stream()
        self._parent_stream = current_stream

        event = torch.cuda.Event()
        event.record(current_stream)

        # pyrefly: ignore [missing-attribute]
        current_ctx = _check_cuda_bindings(_drv.cuCtxGetCurrent())
        if int(current_ctx) == 0:
            # pyrefly: ignore [missing-attribute]
            _check_cuda_bindings(_drv.cuCtxSetCurrent(self._context))
        else:
            # pyrefly: ignore [missing-attribute]
            _check_cuda_bindings(_drv.cuCtxPushCurrent(self._context))

        green_ctx_stream = torch.cuda.default_stream(self._device_id)
        event.wait(green_ctx_stream)
        torch.cuda.set_stream(green_ctx_stream)

    @deprecated(_CONTEXT_STACK_DEPRECATION, category=FutureWarning)
    def pop_context(self) -> None:
        # [FUNCTION DOC] Deprecated. Pop the green context from the context stack, make the parent stream
        # wait for work on the current stream via an event, and restore the parent stream. Raises
        # RuntimeError if there was no matching set_context or the popped context is not this one. The
        # parent stream is always cleared.
        r"""Assuming the green context is the current context, pop it from the
        context stack and restore the previous context.

        Deprecated. Create streams with :meth:`Stream` and use
        :func:`torch.cuda.stream` instead.
        """
        try:
            self._ensure_alive()
            if self._parent_stream is None:
                raise RuntimeError("pop_context called without matching set_context")

            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())

            # pyrefly: ignore [missing-attribute]
            popped = _check_cuda_bindings(_drv.cuCtxPopCurrent())
            # pyrefly: ignore [bad-argument-type]
            if int(popped) != int(self._context):
                raise RuntimeError("expected popped context to be the current ctx")

            event.wait(self._parent_stream)
            torch.cuda.set_stream(self._parent_stream)
        finally:
            self._parent_stream = None

    def Stream(self) -> torch.cuda.Stream:
        # [FUNCTION DOC] Hand out a stream bound to this green context. The first pool-size calls create
        # new non-blocking driver streams; later calls reuse pooled streams round-robin. Returns an
        # external stream wrapper on this device. Raises RuntimeError if the context was destroyed.
        r"""Return a CUDA stream associated with this green context.

        Use the returned stream with :func:`torch.cuda.stream` to run work on
        the green context. Synchronization with other streams is not automatic;
        use CUDA events as with any other custom stream.
        """
        self._ensure_alive()
        curr_idx = self._curr_stream_idx + 1
        idx = curr_idx % _STREAMS_PER_GREEN_CONTEXT_POOL
        if curr_idx < _STREAMS_PER_GREEN_CONTEXT_POOL:
            green_ctx_stream = _check_cuda_bindings(
                _drv.cuGreenCtxStreamCreate(  # pyrefly: ignore [missing-attribute]
                    self._green_ctx,
                    _drv.CUstream_flags.CU_STREAM_NON_BLOCKING,  # pyrefly: ignore [missing-attribute]
                    0,
                )
            )
            self._green_ctx_streams[idx] = green_ctx_stream
        else:
            green_ctx_stream = self._green_ctx_streams[idx]
        self._curr_stream_idx = curr_idx
        # pyrefly: ignore [bad-argument-type]
        return torch.cuda.ExternalStream(int(green_ctx_stream), self._device_id)
