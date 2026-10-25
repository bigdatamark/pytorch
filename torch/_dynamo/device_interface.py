# [MODULE DOC] Device abstraction layer used by Dynamo and Inductor. Defines the
# DeviceInterface base class plus per-backend implementations (CUDA, MTIA, XPU,
# CPU, MPS, TPU), a DeviceGuard context manager, and a global registry mapping
# device-type strings to interface classes. Also holds module-level caches of
# device properties / current device used by worker processes. Import-time side
# effects: looks up raw-stream getters from the C extension when the CUDA, MTIA
# and XPU backends are compiled (else the getter is None), and reads attributes
# of torch.cuda / torch.mtia / torch.xpu to build the interface classes.
"""
Device abstraction layer for TorchDynamo and Inductor backends.

This module provides a unified interface for different hardware backends (CUDA, XPU,
CPU, MPS, MTIA) through a common device interface. Key components include:

- DeviceInterface: Base class defining the common API for all device types
- Device-specific implementations: CudaInterface, XpuInterface, CpuInterface, MpsInterface, MtiaInterface
- Device registration system for managing available backends
- Worker APIs for multi-processing scenarios
- Stream and event management across different devices
- Device property caching for worker processes

The abstraction layer enables device-agnostic code in TorchDynamo while allowing
specialized implementations for each hardware backend's unique features.
"""

import inspect
import time
from collections import namedtuple
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch.utils._pallas import has_torch_tpu


get_cuda_stream: Callable[[int], int] | None
if torch.cuda._is_compiled():
    from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
else:
    get_cuda_stream = None

# Recording the device properties in the main process but used in worker process.
caching_worker_device_properties: dict[str, Any] = {}
caching_worker_current_devices: dict[str, int] = {}


class DeviceInterface:
    # [CLASS DOC] Abstract base describing the device runtime API (device
    # context, Event, Stream, Worker, device/stream queries, dtype and Triton
    # capability checks) that backends implement. It is used as a class, never
    # instantiated; subclasses override static/class methods. Most methods here
    # raise NotImplementedError; the defaults implemented are get_device_properties
    # (delegates to Worker), is_dtype_supported, is_triton_capable (False) and
    # raise_if_triton_unavailable.
    """
    This is a simple device runtime interface for Inductor. It enables custom
    backends to be integrated with Inductor in a device-agnostic semantic.
    """

    class device:
        # [CLASS DOC] Placeholder for the backend device context-manager class;
        # subclasses replace it. Construction always raises NotImplementedError.
        def __new__(cls, device: torch.types.Device) -> Any:
            # [FUNCTION DOC] Always raises NotImplementedError; backends must
            # supply their own device class.
            raise NotImplementedError

    class Event:
        # [CLASS DOC] Placeholder for the backend Event class; backends must
        # replace it with a subclass of torch.Event so Dynamo can capture it.
        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            # [FUNCTION DOC] Always raises NotImplementedError explaining that
            # Event must inherit from torch.Event to be captured by Dynamo.
            raise NotImplementedError(
                "Event should be inherited from torch.Event, otherwise, it couldn't be captured by dynamo."
            )

    class Stream:
        # [CLASS DOC] Placeholder for the backend Stream class; backends must
        # replace it with a subclass of torch.Stream so Dynamo can capture it.
        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            # [FUNCTION DOC] Always raises NotImplementedError explaining that
            # Stream must inherit from torch.Stream to be captured by Dynamo.
            raise NotImplementedError(
                "Stream should be inherited from torch.Stream, otherwise, it couldn't be captured by dynamo."
            )

    class Worker:
        # [CLASS DOC] Abstract worker-safe API for querying the current device
        # and device properties inside forked multiprocessing workers, relying on
        # values recorded in the main process. Subclasses implement the three
        # static methods below.
        """
        Worker API to query device properties that will work in multi processing
        workers that cannot use the GPU APIs (due to processing fork() and
        initialization time issues). Properties are recorded in the main process
        before we fork the workers.
        """

        @staticmethod
        def set_device(device: int) -> None:
            # [FUNCTION DOC] Abstract: record the device index as current for
            # workers. Raises NotImplementedError.
            raise NotImplementedError

        @staticmethod
        def current_device() -> int:
            # [FUNCTION DOC] Abstract: return the worker's current device index.
            # Raises NotImplementedError.
            raise NotImplementedError

        @staticmethod
        def get_device_properties(device: torch.types.Device = None) -> Any:
            # [FUNCTION DOC] Abstract: return cached properties of the device
            # (default current). Raises NotImplementedError.
            raise NotImplementedError

    @staticmethod
    def current_device() -> int:
        # [FUNCTION DOC] Abstract: return the current device index. Raises
        # NotImplementedError.
        raise NotImplementedError

    @staticmethod
    def set_device(device: torch.types.Device) -> None:
        # [FUNCTION DOC] Abstract: make the given device current. Raises
        # NotImplementedError.
        raise NotImplementedError

    @staticmethod
    def maybe_exchange_device(device: int) -> int:
        # [FUNCTION DOC] Abstract: switch to the device index if it is valid,
        # returning the previous index. Raises NotImplementedError.
        raise NotImplementedError

    @staticmethod
    def exchange_device(device: int) -> int:
        # [FUNCTION DOC] Abstract: switch to the device index and return the
        # previous index. Raises NotImplementedError.
        raise NotImplementedError

    @staticmethod
    def device_count() -> int:
        # [FUNCTION DOC] Abstract: return the number of devices. Raises
        # NotImplementedError.
        raise NotImplementedError

    @staticmethod
    def is_available() -> bool:
        # [FUNCTION DOC] Abstract: report whether the backend is usable. Raises
        # NotImplementedError.
        raise NotImplementedError

    @staticmethod
    def stream(stream: torch.Stream) -> Any:
        # [FUNCTION DOC] Abstract: return a context manager selecting the given
        # stream. Raises NotImplementedError.
        raise NotImplementedError

    @staticmethod
    def current_stream() -> torch.Stream:
        # [FUNCTION DOC] Abstract: return the current stream. Raises
        # NotImplementedError.
        raise NotImplementedError

    @staticmethod
    def set_stream(stream: torch.Stream) -> None:
        # [FUNCTION DOC] Abstract: make the given stream current. Raises
        # NotImplementedError.
        raise NotImplementedError

    @staticmethod
    def _set_stream_by_id(stream_id: int, device_index: int, device_type: int) -> None:
        # [FUNCTION DOC] HELPER: Abstract: set the current stream from its id,
        # device index and device type. Raises NotImplementedError.
        raise NotImplementedError

    @staticmethod
    def get_raw_stream(device_idx: int) -> int:
        # [FUNCTION DOC] Abstract: return the raw native stream handle (integer)
        # of the current stream on the device. Raises NotImplementedError.
        raise NotImplementedError

    @staticmethod
    def synchronize(device: torch.types.Device = None) -> None:
        # [FUNCTION DOC] Abstract: block until all work on the device finishes.
        # Raises NotImplementedError.
        raise NotImplementedError

    @classmethod
    def get_device_properties(cls, device: torch.types.Device = None) -> Any:
        # [FUNCTION DOC] Return device properties by delegating to the class's
        # Worker implementation (so cached values are used in workers).
        return cls.Worker.get_device_properties(device)

    @staticmethod
    def get_compute_capability(device: torch.types.Device = None) -> Any:
        # [FUNCTION DOC] Abstract: return the device compute capability.
        # Raises NotImplementedError.
        raise NotImplementedError

    @staticmethod
    def is_bf16_supported(including_emulation: bool = False) -> bool:
        # [FUNCTION DOC] Abstract: report bfloat16 support, optionally counting
        # emulation. Raises NotImplementedError.
        raise NotImplementedError

    @classmethod
    def is_dtype_supported(
        cls, dtype: torch.dtype, including_emulation: bool = False
    ) -> bool:
        # [FUNCTION DOC] Return True for every dtype except bfloat16; for
        # bfloat16 defer to the class's is_bf16_supported.
        return dtype != torch.bfloat16 or cls.is_bf16_supported(including_emulation)

    @staticmethod
    def memory_allocated(device: torch.types.Device = None) -> int:
        # [FUNCTION DOC] Abstract: return bytes currently allocated on the
        # device. Raises NotImplementedError.
        raise NotImplementedError

    @staticmethod
    def is_triton_capable(device: torch.types.Device = None) -> bool:
        # [FUNCTION DOC] Default capability check: returns False (no Triton
        # support) regardless of whether a Triton backend is installed.
        """
        Returns True if the device has Triton support, False otherwise, even if
        the appropriate Triton backend is not available.
        """
        return False

    @classmethod
    def raise_if_triton_unavailable(cls, device: torch.types.Device = None) -> None:
        # [FUNCTION DOC] Default check: raises TritonUnavailableError when the
        # device is not Triton-capable; otherwise returns None. Subclasses add
        # backend-specific checks. Caller must ensure triton is installed.
        """
        Raises a `TritonUnavailableError` with human-readable instructions if
        the Triton backend for the given device (or the default device if
        `device` is `None`) is not built. Implementations may raise a
        different, more specific error when the device itself is not
        Triton-capable (e.g. CUDA raises `GPUTooOldForTriton`), so callers
        that only want an availability answer should check
        `is_triton_capable()` first.

        The caller should ensure the presence of the 'triton' package before
        calling this method.
        """
        from torch._dynamo.exc import TritonUnavailableError

        if not cls.is_triton_capable():
            raise TritonUnavailableError(
                "This device is not capable of supporting Triton"
            )


class DeviceGuard:
    # [CLASS DOC] Context manager that switches the current device through a
    # DeviceInterface class. State: device_interface, idx (target index or None
    # for no-op), prev_idx (index to restore, -1 until entered). On exit it
    # restores the previous device. Never suppresses exceptions.
    """
    This class provides a context manager for device switching. This is a stripped
    down version of torch.{device_name}.device.

    The context manager changes the current device to the given device index
    on entering the context and restores the original device on exiting.
    The device is switched using the provided device interface.
    """

    def __init__(
        self, device_interface: type[DeviceInterface], index: int | None
    ) -> None:
        # [FUNCTION DOC] Store the interface class and target device index
        # (None means do nothing); initialize the saved previous index to -1.
        self.device_interface = device_interface
        self.idx = index
        self.prev_idx = -1

    def __enter__(self) -> None:
        # [FUNCTION DOC] If an index was given, switch to it via the interface's
        # exchange_device and remember the previous index. Returns None.
        if self.idx is not None:
            self.prev_idx = self.device_interface.exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any) -> Literal[False]:
        # [FUNCTION DOC] If an index was given, switch back to the saved
        # previous index via maybe_exchange_device. Always returns False so
        # exceptions propagate.
        # KNOWN ISSUE: the return value of maybe_exchange_device (the device
        # just left) overwrites self.idx, so re-entering the guard would target
        # a different device than originally requested.
        if self.idx is not None:
            self.idx = self.device_interface.maybe_exchange_device(self.prev_idx)
        return False


class CudaInterface(DeviceInterface):
    # [CLASS DOC] DeviceInterface for CUDA/ROCm. Most members are static aliases
    # to torch.cuda functions (device, Event, Stream, current_device, set_device,
    # device_count, stream, synchronize, memory_allocated, exchange helpers,
    # raw-stream getter). Adds a cached-properties Worker, compute-capability,
    # and Triton capability/availability checks.
    device = torch.cuda.device  # type: ignore[assignment]

    # register Event and Stream class into the backend interface
    # make sure Event and Stream are implemented and inherited from the torch.Event and torch.Stream
    Event = torch.cuda.Event  # type: ignore[assignment]
    Stream = torch.cuda.Stream  # type: ignore[assignment]

    # pyrefly: ignore [bad-override]
    class Worker:
        # [CLASS DOC] Worker-safe CUDA queries. Uses the module-level caches
        # (caching_worker_current_devices and caching_worker_device_properties,
        # keyed by "cuda") so forked workers need not touch CUDA.
        @staticmethod
        def set_device(device: int) -> None:
            # [FUNCTION DOC] Record the device index in the cache of worker
            # current devices (global mutation); does not touch the real device.
            caching_worker_current_devices["cuda"] = device

        @staticmethod
        def current_device() -> int:
            # [FUNCTION DOC] Return the cached current device index if set,
            # otherwise query torch.cuda.
            if "cuda" in caching_worker_current_devices:
                return caching_worker_current_devices["cuda"]
            return torch.cuda.current_device()

        @staticmethod
        def get_device_properties(device: torch.types.Device = None) -> Any:
            # [FUNCTION DOC] Resolve device (string, torch.device, int or None
            # meaning the worker's current device) to an index, lazily cache
            # properties of all CUDA devices on first call (global mutation),
            # and return the entry for that index. Raises AssertionError if a
            # string device is not of type cuda.
            # KNOWN ISSUE: the type check only runs for string inputs, and a
            # torch.device without an index yields index None, which fails when
            # used to index the cached list.
            if device is not None:
                if isinstance(device, str):
                    device = torch.device(device)
                    if device.type != "cuda":
                        raise AssertionError(
                            f"Expected device type 'cuda', got '{device.type}'"
                        )
                if isinstance(device, torch.device):
                    device = device.index
            if device is None:
                device = CudaInterface.Worker.current_device()

            if "cuda" not in caching_worker_device_properties:
                device_prop = [
                    torch.cuda.get_device_properties(i)
                    for i in range(torch.cuda.device_count())
                ]
                caching_worker_device_properties["cuda"] = device_prop

            return caching_worker_device_properties["cuda"][device]

    current_device = staticmethod(torch.cuda.current_device)
    set_device = staticmethod(torch.cuda.set_device)
    device_count = staticmethod(torch.cuda.device_count)
    stream = staticmethod(torch.cuda.stream)  # type: ignore[assignment]
    current_stream = staticmethod(torch.cuda.current_stream)
    set_stream = staticmethod(torch.cuda.set_stream)  # type: ignore[assignment]
    _set_stream_by_id = staticmethod(torch.cuda._set_stream_by_id)  # type: ignore[assignment]
    synchronize = staticmethod(torch.cuda.synchronize)
    get_device_properties = staticmethod(torch.cuda.get_device_properties)  # type: ignore[assignment]
    get_raw_stream = staticmethod(get_cuda_stream)  # type: ignore[assignment, arg-type]
    exchange_device = staticmethod(torch.cuda._exchange_device)  # type: ignore[arg-type, has-type]
    maybe_exchange_device = staticmethod(torch.cuda._maybe_exchange_device)  # type: ignore[arg-type, has-type]
    memory_allocated = staticmethod(torch.cuda.memory_allocated)
    is_bf16_supported = staticmethod(torch.cuda.is_bf16_supported)  # type: ignore[arg-type]

    # Can be mock patched by @patch decorator.
    @staticmethod
    def is_available() -> bool:
        # [FUNCTION DOC] Return whether CUDA is available (wrapper so tests can
        # patch it).
        return torch.cuda.is_available()

    @staticmethod
    def get_compute_capability(device: torch.types.Device = None) -> int | str:
        # [FUNCTION DOC] On CUDA return major*10+minor as an integer; on ROCm
        # return the GCN architecture name (text before the first colon).
        if torch.version.hip is None:
            major, min = torch.cuda.get_device_capability(device)
            return major * 10 + min
        else:
            return torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]

    @staticmethod
    def is_triton_capable(device: torch.types.Device = None) -> bool:
        # [FUNCTION DOC] True on ROCm, or on CUDA when the device's major
        # compute capability is at least 7. Uses cached Worker properties so it
        # is safe in spawned compile workers.
        # Use the Worker API (device properties cached in the main process
        # before fork) instead of torch.cuda.get_device_properties directly, so
        # the capability check stays safe when called from spawn-based compile
        # workers.
        return (
            torch.version.hip is not None
            or CudaInterface.Worker.get_device_properties(device).major >= 7
        )

    @staticmethod
    def raise_if_triton_unavailable(device: torch.types.Device = None) -> None:
        # [FUNCTION DOC] Raise GPUTooOldForTriton if the GPU is not Triton
        # capable; otherwise import triton backends and raise
        # TritonUnavailableError if the amd (ROCm) or nvidia backend is missing.
        # Requires the triton package to be installed (import error otherwise).
        from torch._dynamo.exc import TritonUnavailableError
        from torch._inductor.exc import GPUTooOldForTriton

        if not CudaInterface.is_triton_capable(device):
            device_props = torch.cuda.get_device_properties(device)
            raise GPUTooOldForTriton(device_props, inspect.currentframe())

        import triton.backends

        if torch.version.hip is not None:
            if "amd" not in triton.backends.backends:
                raise TritonUnavailableError("triton not built with the 'amd' backend")
        elif "nvidia" not in triton.backends.backends:
            raise TritonUnavailableError("triton not built with the 'nvidia' backend")


get_mtia_stream: Callable[[int], int] | None
if torch.mtia._is_compiled():
    from torch._C import _mtia_getCurrentRawStream as get_mtia_stream
else:
    get_mtia_stream = None


class MtiaInterface(DeviceInterface):
    # [CLASS DOC] DeviceInterface for MTIA devices; mirrors CudaInterface by
    # aliasing torch.mtia functions and providing a cached Worker, compute
    # capability, and an always-capable Triton check requiring the mtia backend.
    device = torch.mtia.device  # type: ignore[assignment]
    Event = torch.mtia.Event  # type: ignore[assignment]
    Stream = torch.mtia.Stream  # type: ignore[assignment]

    # pyrefly: ignore [bad-override]
    class Worker:
        # [CLASS DOC] Worker-safe MTIA queries using the module-level caches
        # keyed by "mtia".
        @staticmethod
        def set_device(device: int) -> None:
            # [FUNCTION DOC] Record the device index in the worker current-device
            # cache (global mutation).
            caching_worker_current_devices["mtia"] = device

        @staticmethod
        def current_device() -> int:
            # [FUNCTION DOC] Return the cached current device index if set,
            # otherwise query torch.mtia.
            if "mtia" in caching_worker_current_devices:
                return caching_worker_current_devices["mtia"]
            return torch.mtia.current_device()

        @staticmethod
        def get_device_properties(device: torch.types.Device = None) -> Any:
            # [FUNCTION DOC] Resolve device to an index (None means the worker's
            # current device), lazily cache properties of all MTIA devices
            # (global mutation) and return the entry. Raises AssertionError if a
            # string device is not of type mtia.
            # KNOWN ISSUE: type check only applies to string inputs, and a
            # torch.device with no index gives index None, which breaks lookup.
            if device is not None:
                if isinstance(device, str):
                    device = torch.device(device)
                    if device.type != "mtia":
                        raise AssertionError(
                            f"Expected device type 'mtia', got '{device.type}'"
                        )
                if isinstance(device, torch.device):
                    device = device.index
            if device is None:
                device = MtiaInterface.Worker.current_device()

            if "mtia" not in caching_worker_device_properties:
                device_prop = [
                    torch.mtia.get_device_properties(i)
                    for i in range(torch.mtia.device_count())
                ]
                caching_worker_device_properties["mtia"] = device_prop

            return caching_worker_device_properties["mtia"][device]

    current_device = staticmethod(torch.mtia.current_device)
    set_device = staticmethod(torch.mtia.set_device)  # type: ignore[assignment]
    device_count = staticmethod(torch.mtia.device_count)
    stream = staticmethod(torch.mtia.stream)  # type: ignore[assignment]
    current_stream = staticmethod(torch.mtia.current_stream)
    set_stream = staticmethod(torch.mtia.set_stream)  # type: ignore[assignment]
    _set_stream_by_id = staticmethod(torch.mtia._set_stream_by_id)  # type: ignore[assignment]
    synchronize = staticmethod(torch.mtia.synchronize)
    get_device_properties = staticmethod(torch.mtia.get_device_properties)  # type: ignore[assignment]
    get_raw_stream = staticmethod(get_mtia_stream)  # type: ignore[assignment, arg-type]
    exchange_device = staticmethod(torch.mtia._exchange_device)  # type: ignore[arg-type, has-type]
    maybe_exchange_device = staticmethod(torch.mtia._maybe_exchange_device)  # type: ignore[arg-type, has-type]
    memory_allocated = staticmethod(torch.mtia.memory_allocated)  # type: ignore[assignment]
    is_bf16_supported = staticmethod(torch.mtia.is_bf16_supported)  # type: ignore[arg-type]

    # Can be mock patched by @patch decorator.
    @staticmethod
    def is_available() -> bool:
        # [FUNCTION DOC] Return whether MTIA is available (patchable wrapper).
        ret = torch.mtia.is_available()
        return ret

    @staticmethod
    def get_compute_capability(device: torch.types.Device = None) -> Any:
        # [FUNCTION DOC] Return the MTIA device capability from torch.mtia.
        cc = torch.mtia.get_device_capability(device)
        return cc

    @staticmethod
    def is_triton_capable(device: torch.types.Device = None) -> bool:
        # [FUNCTION DOC] Always True: MTIA devices are considered Triton capable.
        return True

    @staticmethod
    def raise_if_triton_unavailable(device: torch.types.Device = None) -> None:
        # [FUNCTION DOC] Import triton backends and raise TritonUnavailableError
        # if the mtia backend is not registered. Requires triton installed.
        import triton.backends

        from torch._dynamo.exc import TritonUnavailableError

        if "mtia" not in triton.backends.backends:
            raise TritonUnavailableError("triton not built with the 'mtia' backend")


get_xpu_stream: Callable[[int], int] | None
if torch.xpu._is_compiled():
    from torch._C import _xpu_getCurrentRawStream as get_xpu_stream
else:
    get_xpu_stream = None


class XpuInterface(DeviceInterface):
    # [CLASS DOC] DeviceInterface for Intel XPU devices; aliases torch.xpu
    # functions, provides a cached Worker, compute capability, bf16 check and a
    # Triton check requiring the intel backend.
    device = torch.xpu.device  # type: ignore[assignment]
    Event = torch.xpu.Event  # type: ignore[assignment]
    Stream = torch.xpu.Stream  # type: ignore[assignment]

    # pyrefly: ignore [bad-override]
    class Worker:
        # [CLASS DOC] Worker-safe XPU queries using the module-level caches
        # keyed by "xpu".
        @staticmethod
        def set_device(device: int) -> None:
            # [FUNCTION DOC] Record the device index in the worker current-device
            # cache (global mutation).
            caching_worker_current_devices["xpu"] = device

        @staticmethod
        def current_device() -> int:
            # [FUNCTION DOC] Return the cached current device index if set,
            # otherwise query torch.xpu.
            if "xpu" in caching_worker_current_devices:
                return caching_worker_current_devices["xpu"]
            return torch.xpu.current_device()

        @staticmethod
        def get_device_properties(device: torch.types.Device = None) -> Any:
            # [FUNCTION DOC] Resolve device to an index (None means the worker's
            # current device), lazily cache properties of all XPU devices
            # (global mutation) and return the entry. Raises AssertionError if a
            # string device is not of type xpu.
            # KNOWN ISSUE: type check only applies to string inputs, and a
            # torch.device with no index gives index None, which breaks lookup.
            if device is not None:
                if isinstance(device, str):
                    device = torch.device(device)
                    if device.type != "xpu":
                        raise AssertionError(
                            f"Expected device type 'xpu', got '{device.type}'"
                        )
                if isinstance(device, torch.device):
                    device = device.index
            if device is None:
                device = XpuInterface.Worker.current_device()

            if "xpu" not in caching_worker_device_properties:
                device_prop = [
                    torch.xpu.get_device_properties(i)
                    for i in range(torch.xpu.device_count())
                ]
                caching_worker_device_properties["xpu"] = device_prop

            return caching_worker_device_properties["xpu"][device]

    current_device = staticmethod(torch.xpu.current_device)
    set_device = staticmethod(torch.xpu.set_device)
    device_count = staticmethod(torch.xpu.device_count)  # type: ignore[has-type]
    stream = staticmethod(torch.xpu.stream)  # type: ignore[assignment]
    current_stream = staticmethod(torch.xpu.current_stream)
    set_stream = staticmethod(torch.xpu.set_stream)  # type: ignore[assignment]
    _set_stream_by_id = staticmethod(torch.xpu._set_stream_by_id)  # type: ignore[assignment]
    synchronize = staticmethod(torch.xpu.synchronize)
    get_device_properties = staticmethod(torch.xpu.get_device_properties)  # type: ignore[assignment]
    get_raw_stream = staticmethod(get_xpu_stream)  # type: ignore[assignment, arg-type]
    exchange_device = staticmethod(torch.xpu._exchange_device)  # type: ignore[arg-type, has-type]
    maybe_exchange_device = staticmethod(torch.xpu._maybe_exchange_device)  # type: ignore[arg-type, has-type]
    memory_allocated = staticmethod(torch.xpu.memory_allocated)

    # Can be mock patched by @patch decorator.
    @staticmethod
    def is_available() -> bool:
        # [FUNCTION DOC] Return whether XPU is available (patchable wrapper).
        return torch.xpu.is_available()

    @staticmethod
    def get_compute_capability(device: torch.types.Device = None) -> Any:
        # [FUNCTION DOC] Return the XPU device capability from torch.xpu.
        cc = torch.xpu.get_device_capability(device)
        return cc

    @staticmethod
    def is_bf16_supported(including_emulation: bool = False) -> bool:
        # [FUNCTION DOC] Return torch.xpu bf16 support; the including_emulation
        # argument is ignored.
        return torch.xpu.is_bf16_supported()

    @staticmethod
    def is_triton_capable(device: torch.types.Device = None) -> bool:
        # [FUNCTION DOC] Always True: XPU devices are considered Triton capable.
        return True

    @staticmethod
    def raise_if_triton_unavailable(device: torch.types.Device = None) -> None:
        # [FUNCTION DOC] Import triton backends and raise TritonUnavailableError
        # if the intel backend is not registered. Requires triton installed.
        import triton.backends

        from torch._dynamo.exc import TritonUnavailableError

        if "intel" not in triton.backends.backends:
            raise TritonUnavailableError("triton not built with the 'intel' backend")


@dataclass
class CpuDeviceProperties:
    # [CLASS DOC] Simple record of CPU properties used by the CPU Worker:
    # multi_processor_count holds the number of logical CPUs.
    multi_processor_count: int


class CpuInterface(DeviceInterface):
    # [CLASS DOC] DeviceInterface for the CPU. Provides a timing-based Event, a
    # Worker reporting CPU count, trivial device/stream functions (device 0, no-op
    # synchronize, raw stream 0), bf16 always supported, and a Triton check that
    # requires the cpu backend.
    # pyrefly: ignore [bad-override]
    class Event(torch.Event):
        # [CLASS DOC] Host-clock event: stores a timestamp (attribute time, in
        # seconds from a monotonic performance counter) set by record, and
        # computes elapsed time between two events.
        def __init__(self, enable_timing: bool = True) -> None:
            # [FUNCTION DOC] Initialize the stored timestamp to zero.
            # enable_timing is accepted but ignored; the base class init is not
            # called.
            self.time = 0.0

        def elapsed_time(self, other: Any) -> float:
            # [FUNCTION DOC] Return the time between this event and the other
            # event, computed as (other minus self) timestamps scaled by 1000
            # (milliseconds). No validation that both were recorded.
            return (other.time - self.time) * 1000

        def record(self, stream: Any = None) -> None:
            # [FUNCTION DOC] Store the current performance-counter time in the
            # event; the stream argument is ignored.
            self.time = time.perf_counter()

    # pyrefly: ignore [bad-override]
    class Worker:
        # [CLASS DOC] Worker API for CPU; only get_device_properties is provided.
        @staticmethod
        def get_device_properties(
            device: torch.types.Device = None,
        ) -> CpuDeviceProperties:
            # [FUNCTION DOC] Return a CpuDeviceProperties whose count is the
            # machine CPU count from multiprocessing; the device argument is
            # ignored.
            import multiprocessing

            cpu_count = multiprocessing.cpu_count()
            return CpuDeviceProperties(cpu_count)

    @staticmethod
    def is_available() -> bool:
        # [FUNCTION DOC] Always True.
        return True

    @staticmethod
    def is_bf16_supported(including_emulation: bool = False) -> bool:
        # [FUNCTION DOC] Always True.
        return True

    @staticmethod
    def get_compute_capability(device: torch.types.Device = None) -> str:
        # [FUNCTION DOC] Return an empty string (CPU has no capability).
        return ""

    @staticmethod
    def get_raw_stream(device_idx: Any) -> int:
        # [FUNCTION DOC] Return 0; CPU has no raw stream.
        return 0

    @staticmethod
    def current_device() -> int:
        # [FUNCTION DOC] Always return device index 0.
        return 0

    @staticmethod
    def synchronize(device: torch.types.Device = None) -> None:
        # [FUNCTION DOC] No-op.
        pass

    @staticmethod
    def is_triton_capable(device: torch.types.Device = None) -> bool:
        # [FUNCTION DOC] Always True.
        return True

    @staticmethod
    def raise_if_triton_unavailable(device: torch.types.Device = None) -> None:
        # [FUNCTION DOC] Import triton backends and raise TritonUnavailableError
        # if the cpu backend is not registered. Requires triton installed.
        import triton.backends

        from torch._dynamo.exc import TritonUnavailableError

        if "cpu" not in triton.backends.backends:
            raise TritonUnavailableError("triton not built with the 'cpu' backend")


class MpsInterface(DeviceInterface):
    # [CLASS DOC] DeviceInterface for Apple MPS: single device 0, bf16 supported,
    # float64/complex128 unsupported, synchronize via torch.mps, and a Worker that
    # reports the GPU core count.
    @staticmethod
    def is_bf16_supported(including_emulation: bool = False) -> bool:
        # [FUNCTION DOC] Always True.
        return True

    @classmethod
    def is_dtype_supported(
        cls, dtype: torch.dtype, including_emulation: bool = False
    ) -> bool:
        # [FUNCTION DOC] Return False for float64 and complex128, True for any
        # other dtype.
        if dtype in [torch.float64, torch.complex128]:
            return False
        return True

    @staticmethod
    def is_available() -> bool:
        # [FUNCTION DOC] Return whether the MPS backend is available.
        return torch.backends.mps.is_available()

    @staticmethod
    def current_device() -> int:
        # [FUNCTION DOC] Always return device index 0.
        return 0

    @staticmethod
    def get_compute_capability(device: torch.types.Device = None) -> str:
        # [FUNCTION DOC] Return an empty string.
        return ""

    @staticmethod
    def synchronize(device: torch.types.Device = None) -> None:
        # [FUNCTION DOC] Block until queued MPS work completes
        # (torch.mps.synchronize); device is ignored.
        torch.mps.synchronize()

    # pyrefly: ignore [bad-override]
    class Worker:
        # [CLASS DOC] Worker API for MPS with get_device_properties and
        # current_device only.
        @staticmethod
        def get_device_properties(device: torch.types.Device = None) -> Any:
            # [FUNCTION DOC] Return a named tuple with multi_processor_count set
            # to the MPS GPU core count; device is ignored.
            return namedtuple("MPSProperties", ["multi_processor_count"])(
                torch.backends.mps.get_core_count()  # type: ignore[arg-type]
            )

        @staticmethod
        def current_device() -> int:
            # [FUNCTION DOC] Always return 0.
            return 0


class TpuInterface(DeviceInterface):
    # [CLASS DOC] DeviceInterface for TPU: single device 0, bf16 supported,
    # availability via has_torch_tpu, and several dtypes (float64, complex types,
    # half) unsupported. Worker reports one multiprocessor.
    @staticmethod
    def is_bf16_supported(including_emulation: bool = False) -> bool:
        # [FUNCTION DOC] Always True.
        return True

    @classmethod
    def is_dtype_supported(
        cls, dtype: torch.dtype, including_emulation: bool = False
    ) -> bool:
        # [FUNCTION DOC] Return False for float64, complex32, complex64,
        # complex128 and half; True for all other dtypes.
        return dtype not in (
            torch.float64,
            torch.complex32,
            torch.complex64,
            torch.complex128,
            torch.half,
        )

    @staticmethod
    def is_available() -> bool:
        # [FUNCTION DOC] Return whether the torch TPU package is present.
        return has_torch_tpu()

    @staticmethod
    def current_device() -> int:
        # [FUNCTION DOC] Always return device index 0.
        return 0

    @staticmethod
    def get_compute_capability(device: torch.types.Device = None) -> str:
        # [FUNCTION DOC] Return an empty string.
        return ""

    # pyrefly: ignore [bad-override]
    class Worker:
        # [CLASS DOC] Worker API for TPU with get_device_properties and
        # current_device only.
        @staticmethod
        def get_device_properties(device: torch.types.Device = None) -> Any:
            # [FUNCTION DOC] Return a named tuple with multi_processor_count
            # fixed at 1; device is ignored.
            return namedtuple("TPUProperties", ["multi_processor_count"])(
                1  # type: ignore[arg-type]
            )

        @staticmethod
        def current_device() -> int:
            # [FUNCTION DOC] Always return 0.
            return 0


device_interfaces: dict[str, type[DeviceInterface]] = {}
_device_initialized = False


def register_interface_for_device(
    device: str | torch.device, device_interface: type[DeviceInterface]
) -> None:
    # [FUNCTION DOC] Register an interface class under a device key in the global
    # registry. A torch.device is reduced to its type string (index dropped); a
    # string is used as given. Overwrites any existing entry.
    if isinstance(device, torch.device):
        device = device.type
    device_interfaces[device] = device_interface


def get_interface_for_device(device: str | torch.device) -> type[DeviceInterface]:
    # [FUNCTION DOC] Look up the interface class for a device string or
    # torch.device (reduced to its type). Lazily initializes the default
    # registrations on first use. Raises NotImplementedError if no interface is
    # registered for the key.
    if isinstance(device, torch.device):
        device = device.type
    if not _device_initialized:
        init_device_reg()
    if device in device_interfaces:
        return device_interfaces[device]
    raise NotImplementedError(f"No interface for device {device}")


def get_registered_device_interfaces() -> Iterable[tuple[str, type[DeviceInterface]]]:
    # [FUNCTION DOC] Return a view of (device key, interface class) pairs of the
    # registry, lazily initializing default registrations first.
    if not _device_initialized:
        init_device_reg()
    return device_interfaces.items()


def init_device_reg() -> None:
    # [FUNCTION DOC] Populate the global registry with the built-in interfaces:
    # cuda, xpu and mtia (both the bare type and one indexed key per detected
    # device), plus cpu, mps and tpu. Queries device counts, and sets the module
    # flag marking initialization complete. Safe to call repeatedly (re-registers).
    global _device_initialized
    register_interface_for_device("cuda", CudaInterface)
    for i in range(torch.cuda.device_count()):
        register_interface_for_device(f"cuda:{i}", CudaInterface)

    register_interface_for_device("xpu", XpuInterface)
    for i in range(torch.xpu.device_count()):
        register_interface_for_device(f"xpu:{i}", XpuInterface)

    register_interface_for_device("mtia", MtiaInterface)
    for i in range(torch.mtia.device_count()):
        register_interface_for_device(f"mtia:{i}", MtiaInterface)

    register_interface_for_device("cpu", CpuInterface)
    register_interface_for_device("mps", MpsInterface)
    register_interface_for_device("tpu", TpuInterface)

    _device_initialized = True
