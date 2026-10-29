# mypy: allow-untyped-defs
# [MODULE DOC]
# Thin ctypes layer over libcupti for the in-process CUPTI monitor. Defines the libcupti
# soname and minimum supported version, overhead-kind names, ctypes mirror structs for the
# subscriber and user-defined-record configuration, a no-op callback trampoline, helpers that
# resolve noisy runtime/driver API callback ids, the CuptiError exception, the _PyLibCupti
# wrapper class (one method per libcupti call, with result checking), and the cached
# pylibcupti accessor. Import-time behavior: imports cupti-python enums and raises
# ModuleNotFoundError with an install hint if the package is missing; builds the no-op
# callback object. libcupti itself is loaded only when pylibcupti is first called.
"""CUPTI type and constant definitions used by the in-process monitor.

This module reaches into CUPTI's ABI constants. It does not re-export
cupti-python's types; callers import those directly
(``from cupti.cupti import ActivityKind``). The enums this module itself needs at
runtime are imported eagerly below; ones used only in annotations are imported
under ``TYPE_CHECKING``. cupti-python is a hard requirement, so importing this
module without it raises ``ModuleNotFoundError`` (catchable by optional consumers).
"""

from __future__ import annotations

import ctypes
import logging
from collections.abc import Iterable  # noqa: TC003
from functools import lru_cache
from typing import Any, TYPE_CHECKING


logger = logging.getLogger(__name__)


# cupti-python enums used at runtime. cupti-python is a hard requirement of this
# module; its absence surfaces as a catchable ModuleNotFoundError for optional
# consumers (e.g. those probing the monitor import).
try:
    from cupti.cupti import (  # pyrefly: ignore[missing-import]
        Driver_api_trace_cbid,
        ExternalCorrelationKind,
        Runtime_api_trace_cbid,
    )
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "torch.profiler._cupti requires the cupti-python package. "
        "Install cupti-python to use the experimental CUPTI monitor."
    ) from exc

# Generated from the CUPTI ABI header (tools/gen_cupti_stubs.py): the
# CUpti_ActivityAttribute selectors, so their (ABI-renumbered) ints are never hardcoded.
from torch.profiler._cupti._cupti_stubs import ActivityAttr


if TYPE_CHECKING:
    # Used only in pylibcupti method signatures (Any to pyrefly; cupti has no stub).
    from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]


# libcupti soname for the supported CUDA major (the monitor's floor is 13.3). Loaded
# by name so dlopen returns the copy already mapped into the process: torch front-loads
# the nvidia-cuda-cupti wheel (torch._preload_cuda_deps) and cupti-python / kineto load
# it otherwise, so every consumer shares one CUPTI -- a second instance would collide
# with the stock profiler's subscriber (CUPTI_ERROR_MULTIPLE_SUBSCRIBERS).
LIBCUPTI_SONAME = "libcupti.so.13"

# CUPTI C-API result/flag constants (cupti_result.h / cupti_activity.h). These
# are stable ABI values, so they are spelled out rather than resolved.
CUPTI_SUCCESS = 0

# CUpti_ActivityAttribute selectors the monitor sets on its subscription
# (ActivityAttr.USER_DEFINED_RECORDS / .ENABLE_KERNEL_LATENCY_TIMESTAMPS /
# .TIMESTAMP_CALLBACK, ...). cupti-python does not surface this enum, so the values come
# from the generated _cupti_stubs module (tools/gen_cupti_stubs.py, straight from the ABI
# header) rather than hardcoded ints -- they are renumbered vs cupti-python's own enum.

# Minimum libcupti the monitor supports. The v2 user-defined-record API arrived in
# 13.2, but only 13.3 populates pBufferCompleteInfo->ppRecordLayouts (CUPTI's own
# per-kind record layout) that the monitor decodes against, so 13.3 is the floor.
LIBCUPTI_MIN_VERSION = 130300

# CUPTI overhead-kind codes (CUpti_ActivityOverheadKind in cupti_activity.h).
# cupti-python does not surface these as an enum, so the code -> name mapping is
# mirrored here.
OVERHEAD_KIND_NAMES: dict[int, str] = {
    0: "Unknown",
    1: "Driver Compiler",
    1 << 16: "Buffer Flush",
    2 << 16: "Instrumentation",
    3 << 16: "Resource",
    4 << 16: "Runtime Triggered Module Loading",
    5 << 16: "Lazy Function Loading",
    6 << 16: "Command Buffer Full",
    7 << 16: "Activity Buffer Request",
    8 << 16: "UVM Activity Init",
}


def _configure_ctypes(lib: ctypes.CDLL) -> None:
    # [FUNCTION DOC] HELPER:
    # Declares argument and return types on the loaded libcupti handle for every entry point
    # the wrapper uses: version, flush, hardware trace, result string, finalize, callback
    # enable, external-correlation push/pop (plain always, subscriber-aware variants if
    # present), and the subscriber-scoped activity API (subscribe, unsubscribe, dropped
    # records, set attribute, register callbacks, enable/disable, timestamp, per-cbid runtime
    # and driver enable). Newer entry points are bound only when the symbol exists, so an old
    # libcupti configures without error. Mutates the passed library object; returns nothing.
    lib.cuptiGetVersion.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
    lib.cuptiGetVersion.restype = ctypes.c_int
    lib.cuptiActivityFlushAll.argtypes = [ctypes.c_uint32]
    lib.cuptiActivityFlushAll.restype = ctypes.c_int
    lib.cuptiActivityEnableHWTrace.argtypes = [ctypes.c_uint8]
    lib.cuptiActivityEnableHWTrace.restype = ctypes.c_int
    lib.cuptiGetResultString.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.cuptiGetResultString.restype = ctypes.c_int
    lib.cuptiFinalize.argtypes = []
    lib.cuptiFinalize.restype = ctypes.c_int
    # Subscriber callbacks. Bound on the raw CDLL rather than taken from the cupti wheel,
    # whose enable_callback validates the subscriber against its own registry and rejects a
    # cuptiSubscribe_v2 handle with CUPTI_ERROR_INVALID_PARAMETER.
    lib.cuptiEnableCallback.argtypes = [
        ctypes.c_uint32,  # enable
        ctypes.c_void_p,  # CUpti_SubscriberHandle subscriber
        ctypes.c_int,  # CUpti_CallbackDomain domain
        ctypes.c_uint32,  # CUpti_CallbackId cbid
    ]
    lib.cuptiEnableCallback.restype = ctypes.c_int

    # User-defined-record (subscription) API -- present in libcupti >= 13.2; guarded
    # so configuring against an older libcupti still succeeds (the monitor's
    # LIBCUPTI_MIN_VERSION check then fails fast at start).
    if hasattr(lib, "cuptiSubscribe_v2"):
        lib.cuptiSubscribe_v2.argtypes = [
            ctypes.c_void_p,  # CUpti_SubscriberHandle* subscriber
            _CB_FUNC,  # CUpti_CallbackFunc callback
            ctypes.c_void_p,  # void* userdata
            ctypes.POINTER(_SubscriberParams),  # CUpti_SubscriberParams* pParams
        ]
        lib.cuptiSubscribe_v2.restype = ctypes.c_int
    if hasattr(lib, "cuptiUnsubscribe"):
        lib.cuptiUnsubscribe.argtypes = [ctypes.c_void_p]
        lib.cuptiUnsubscribe.restype = ctypes.c_int
    if hasattr(lib, "cuptiActivityGetNumDroppedRecords_v2"):
        lib.cuptiActivityGetNumDroppedRecords_v2.argtypes = [
            ctypes.c_void_p,  # CUpti_SubscriberHandle subscriber
            ctypes.c_void_p,  # CUcontext context
            ctypes.c_uint32,  # uint32_t streamId
            ctypes.POINTER(ctypes.c_size_t),  # size_t* dropped
        ]
        lib.cuptiActivityGetNumDroppedRecords_v2.restype = ctypes.c_int
    if hasattr(lib, "cuptiActivitySetAttribute_v2"):
        lib.cuptiActivitySetAttribute_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
        ]
        lib.cuptiActivitySetAttribute_v2.restype = ctypes.c_int
    if hasattr(lib, "cuptiActivityRegisterCallbacks_v2"):
        lib.cuptiActivityRegisterCallbacks_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.cuptiActivityRegisterCallbacks_v2.restype = ctypes.c_int
    if hasattr(lib, "cuptiActivityEnable_v2"):
        lib.cuptiActivityEnable_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.cuptiActivityEnable_v2.restype = ctypes.c_int
    if hasattr(lib, "cuptiActivityDisable_v2"):
        lib.cuptiActivityDisable_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.cuptiActivityDisable_v2.restype = ctypes.c_int
    if hasattr(lib, "cuptiGetTimestamp_v2"):
        lib.cuptiGetTimestamp_v2.argtypes = [
            ctypes.c_void_p,  # CUpti_SubscriberHandle subscriber
            ctypes.POINTER(ctypes.c_uint64),
        ]
        lib.cuptiGetTimestamp_v2.restype = ctypes.c_int
    # External correlation push/pop. The plain (v1) calls return
    # CUPTI_ERROR_NOT_COMPATIBLE while a user-defined-record subscriber is active
    # (same as cuptiGetTimestamp), so the subscriber-aware _v2 variants are required
    # on the v2 path; bind them when present (libcupti >= 13.3) and fall back to v1.
    lib.cuptiActivityPushExternalCorrelationId.argtypes = [
        ctypes.c_int,
        ctypes.c_uint64,
    ]
    lib.cuptiActivityPushExternalCorrelationId.restype = ctypes.c_int
    lib.cuptiActivityPopExternalCorrelationId.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.cuptiActivityPopExternalCorrelationId.restype = ctypes.c_int
    if hasattr(lib, "cuptiActivityPushExternalCorrelationId_v2"):
        lib.cuptiActivityPushExternalCorrelationId_v2.argtypes = [
            ctypes.c_void_p,  # CUpti_SubscriberHandle subscriber
            ctypes.c_int,
            ctypes.c_uint64,
        ]
        lib.cuptiActivityPushExternalCorrelationId_v2.restype = ctypes.c_int
    if hasattr(lib, "cuptiActivityPopExternalCorrelationId_v2"):
        lib.cuptiActivityPopExternalCorrelationId_v2.argtypes = [
            ctypes.c_void_p,  # CUpti_SubscriberHandle subscriber
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        lib.cuptiActivityPopExternalCorrelationId_v2.restype = ctypes.c_int
    # Collection-time noise filter: cuptiActivityEnableRuntimeApi_v2(sub, cbid, 0) stops
    # CUPTI generating RUNTIME records for that cbid (kineto disables cudaGetDevice/etc.
    # this way to shrink buffers). Only the subscriber-scoped _v2 form is bound: the
    # monitor is always on the UDR path, where the global form returns
    # CUPTI_ERROR_NOT_COMPATIBLE (like the timestamp / ext-correlation APIs).
    if hasattr(lib, "cuptiActivityEnableRuntimeApi_v2"):
        lib.cuptiActivityEnableRuntimeApi_v2.argtypes = [
            ctypes.c_void_p,  # CUpti_SubscriberHandle subscriber
            ctypes.c_uint32,  # CUpti_runtime_api_trace_cbid
            ctypes.c_uint8,  # enable
        ]
        lib.cuptiActivityEnableRuntimeApi_v2.restype = ctypes.c_int
    # Same noise filter for the DRIVER API (the v2 monitor enables DRIVER as a carrier,
    # so its noise cbids would otherwise fill the UDR buffers).
    if hasattr(lib, "cuptiActivityEnableDriverApi_v2"):
        lib.cuptiActivityEnableDriverApi_v2.argtypes = [
            ctypes.c_void_p,  # CUpti_SubscriberHandle subscriber
            ctypes.c_uint32,  # CUpti_driver_api_trace_cbid
            ctypes.c_uint8,  # enable
        ]
        lib.cuptiActivityEnableDriverApi_v2.restype = ctypes.c_int


# --- v2 user-defined-record ctypes structs --------------------------------
class _SubscriberParams(ctypes.Structure):
    # [CLASS DOC] HELPER:
    # ctypes mirror of the CUPTI subscriber parameter struct passed to the subscribe call:
    # struct size, subscriber name, previous subscriber name and size, a flag allowing
    # multiple subscribers, and padding. Field order and types must match the C header.
    """Mirror of CUpti_SubscriberParams (cupti_callbacks.h, CUPTI >= 13.2), the
    4th argument to cuptiSubscribe_v2. Member order/types must match CUPTI."""

    _fields_ = [
        ("structSize", ctypes.c_size_t),
        ("subscriberName", ctypes.c_char_p),
        ("oldSubscriberName", ctypes.c_char_p),
        ("oldSubscriberSize", ctypes.c_size_t),
        ("allowMultipleSubscribers", ctypes.c_uint8),
        ("padding", ctypes.c_uint8 * 7),
    ]


class _UDFieldSelection(ctypes.Structure):
    # [CLASS DOC] HELPER:
    # ctypes mirror of CUPTI's user-defined-record field selection: struct size, number of
    # fields, and a pointer to an array of integer field ids. The caller must keep the array
    # alive while the struct is in use.
    _fields_ = [
        ("structSize", ctypes.c_size_t),
        ("numFields", ctypes.c_size_t),
        ("pFieldIds", ctypes.POINTER(ctypes.c_int)),
    ]


class _UDActivityConfig(ctypes.Structure):
    # [CLASS DOC] HELPER:
    # ctypes mirror of CUPTI's per-kind user-defined activity configuration: struct size and
    # an embedded field selection. Passed to the subscriber-scoped activity enable call.
    _fields_ = [
        ("structSize", ctypes.c_size_t),
        ("fieldSelection", _UDFieldSelection),
    ]


# cuptiSubscribe requires a valid CUpti_CallbackFunc, but the monitor drives
# collection through the activity API, not callbacks -- a no-op suffices. Kept
# alive process-wide so the ctypes trampoline isn't garbage-collected.
_CB_FUNC = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_void_p
)


def _noop_callback(*_args: object) -> None:
    # [FUNCTION DOC] HELPER:
    # Callback body that ignores all arguments and does nothing; it satisfies CUPTI's
    # requirement of a valid callback function at subscribe time.
    pass


_NOOP_CB = _CB_FUNC(_noop_callback)


# Runtime-API cbids that are pure noise (no observer needs them); disabled at collection
# time when libcupti supports it so their RUNTIME records never reach the buffers. Matches
# the set kineto filters (see also the post-decode _RUNTIME_BLOCKLIST in monitor_trace).
_NOISY_RUNTIME_API_NAMES = ("cudaGetDevice", "cudaSetDevice", "cudaGetLastError")


def _noisy_runtime_cbids() -> list[int]:
    # [FUNCTION DOC] HELPER:
    # Looks up the runtime-API callback ids whose names (after stripping a trailing numeric
    # version suffix) are in the noisy-name list, using the cupti-python enum members.
    # Returns a list of integer ids; returns an empty list if the enum members cannot be
    # read (any exception is swallowed).
    """The cbid values for :data:`_NOISY_RUNTIME_API_NAMES`, resolved from the cupti enum
    (``_vNNNN`` version suffix stripped). Empty when the enum is unavailable."""
    try:
        members = Runtime_api_trace_cbid.__members__
    except Exception:
        return []
    out: list[int] = []
    for name, member in members.items():
        prefix, _, ver = name.rpartition("_v")
        base = prefix if prefix and ver.isdigit() else name
        if base in _NOISY_RUNTIME_API_NAMES:
            out.append(int(member.value))
    return out


# Noisy driver-API cbids. The v2 monitor enables DRIVER as a carrier (NCCL launches
# collective kernels via the driver API), so these would otherwise fill the UDR buffers.
# Driver cbid names are unversioned, so -- unlike the runtime names -- they resolve by
# direct name (no _vNNNN suffix). See also the post-decode allowlist (_DRIVER_REGISTERED
# in monitor_trace), which keeps any other unregistered driver api out of the trace.
_NOISY_DRIVER_API_NAMES = (
    "cuKernelGetAttribute",
    "cuDevicePrimaryCtxGetState",
    "cuCtxGetCurrent",
)


def _noisy_driver_cbids() -> list[int]:
    # [FUNCTION DOC] HELPER:
    # Looks up each noisy driver-API name directly as an attribute of the cupti-python
    # driver callback enum and returns the integer ids of those found (missing names are
    # skipped silently, so the result may be shorter than the name list or empty).
    """The cbid values for :data:`_NOISY_DRIVER_API_NAMES`, resolved by direct name
    (driver cbid names carry no version suffix). Empty when the enum is unavailable."""
    out: list[int] = []
    for name in _NOISY_DRIVER_API_NAMES:
        member = getattr(Driver_api_trace_cbid, name, None)
        if member is not None:
            out.append(int(member.value))
    return out


class CuptiError(RuntimeError):
    # [CLASS DOC]
    # Exception raised by the wrapper when a libcupti call returns a non-success status or
    # returns an invalid value (such as a null subscriber handle). The message names the call
    # and the CUPTI result string. Subclass of RuntimeError with no extra state.
    pass


class _PyLibCupti:
    # [CLASS DOC]
    # Python-friendly facade over a configured libcupti library handle. Holds that handle as
    # its only state. Each method wraps one CUPTI call, converting arguments to ctypes and
    # checking the status code; required calls raise CuptiError, best-effort calls return a
    # boolean or None instead. Covers version query, function-address lookup for the native
    # decoder, timestamps, subscription, user-defined-record arming, per-kind enable/disable,
    # callbacks, noise filtering, dropped-record counts, and external-correlation push/pop.
    # Normally obtained as the process-wide instance from pylibcupti.
    """Pythonic wrapper over libcupti's CUPTI Activity API: each method hides the
    ctypes marshalling and rc-checking behind a clean call, so callers (the
    monitor, enable_hes_early, the v2 path) never touch ctypes. Get the
    process-wide instance via :func:`pylibcupti`. Methods that must succeed raise
    CuptiError; genuinely-optional ones return a bool / None."""

    def __init__(self, lib: ctypes.CDLL) -> None:
        # [FUNCTION DOC] HELPER:
        # Stores the already-configured libcupti handle for later calls. No other effects.
        self._lib = lib

    def _result_string(self, rc: int) -> str:
        # [FUNCTION DOC] HELPER:
        # Converts a CUPTI status code to readable text via the library's result-string call.
        # Falls back to a generic text containing the numeric code if that lookup itself fails
        # or yields nothing. Never raises.
        result = ctypes.c_char_p()
        rc2 = self._lib.cuptiGetResultString(rc, ctypes.byref(result))
        if rc2 == CUPTI_SUCCESS and result.value is not None:
            return result.value.decode()
        return f"rc={rc}"

    def _check(self, rc: int, name: str) -> None:
        # [FUNCTION DOC] HELPER:
        # Raises CuptiError naming the call and the decoded status text when the status code
        # is not the success value; does nothing on success.
        if rc != CUPTI_SUCCESS:
            raise CuptiError(f"{name} failed with {self._result_string(rc)}")

    def get_version(self) -> int:
        # [FUNCTION DOC]
        # Returns the loaded libcupti version as an integer (for example 130300 for 13.3).
        # Raises CuptiError if the version call fails.
        version = ctypes.c_uint32()
        self._check(self._lib.cuptiGetVersion(ctypes.byref(version)), "cuptiGetVersion")
        return version.value

    def get_next_record_fn_address(self) -> int:
        # [FUNCTION DOC]
        # Returns the raw memory address of the library's v2 next-record iterator so the
        # native decode worker can call it directly, or 0 if the symbol is absent (older
        # libcupti). No side effects.
        """Raw address of ``cuptiActivityGetNextRecord_v2`` (the v2 record
        iterator), for the native decode worker to call directly -- so the native
        module needs no libcupti link, and every consumer shares the one libcupti
        loaded here. Returns 0 if the symbol is absent (libcupti < 13.2)."""
        if not hasattr(self._lib, "cuptiActivityGetNextRecord_v2"):
            return 0
        return (
            ctypes.cast(self._lib.cuptiActivityGetNextRecord_v2, ctypes.c_void_p).value
            or 0
        )

    def get_flush_fn_address(self) -> int:
        # [FUNCTION DOC]
        # Returns the raw memory address of the library's flush-all function for the native
        # decoder's periodic flush, or 0 if the symbol is missing. No side effects.
        """Raw address of ``cuptiActivityFlushAll``, for the native decode worker to
        drive the periodic plain flush directly -- so the native module needs no
        libcupti link either (mirrors ``get_next_record_fn_address``). Returns 0 if
        the symbol is absent."""
        if not hasattr(self._lib, "cuptiActivityFlushAll"):
            return 0
        return ctypes.cast(self._lib.cuptiActivityFlushAll, ctypes.c_void_p).value or 0

    def get_timestamp(self, sub_handle: int) -> int:
        # [FUNCTION DOC]
        # Reads CUPTI's current clock in nanoseconds for the given subscriber handle, using
        # the subscriber-aware call. Returns the integer timestamp. Raises CuptiError if the
        # call fails (including when the symbol is missing, which raises an attribute error).
        """CUPTI's normalized nanosecond clock for a subscriber -- the same timebase
        as activity record START/END timestamps, so a value captured here is directly
        comparable to decoded record timestamps. The subscriber-aware _v2 form is
        required while a user-defined-record subscriber is active: plain
        cuptiGetTimestamp returns CUPTI_ERROR_NOT_COMPATIBLE under UDR on libcupti
        13.3 (it works only with no active v2 subscriber)."""
        ts = ctypes.c_uint64()
        self._check(
            self._lib.cuptiGetTimestamp_v2(
                ctypes.c_void_p(sub_handle), ctypes.byref(ts)
            ),
            "cuptiGetTimestamp_v2",
        )
        return ts.value

    def arm_approx_timestamp_callback(
        self, sub_handle: int, callback_addr: int
    ) -> bool:
        # [FUNCTION DOC]
        # Sets the subscriber's timestamp-callback attribute to the given function address
        # (a zero address is passed as a null pointer) so CUPTI stamps records using that
        # clock. Best-effort: returns true if CUPTI accepted it, false otherwise; never raises
        # on a CUPTI error. Side effect: changes the subscriber's CUPTI attribute.
        """Arm this subscriber's timestamp callback via CUPTI_ACTIVITY_ATTR_TIMESTAMP_CALLBACK:
        CUPTI stamps the subscriber's activity records with ``callback_addr`` (a
        CUpti_TimestampCallbackFunc, uint64_t(*)(void)) instead of its default CPU timer -- used
        to put records directly on the profiler's approx clock (kineto's timebase). Unlike the
        global cuptiActivityRegisterTimestampCallback (CUPTI_ERROR_NOT_COMPATIBLE under
        user-defined records) this per-subscriber attribute coexists with UDR. Best-effort:
        returns False if CUPTI rejects it -- the attribute is beta and still usable only when
        multiple subscribers are NOT allowed."""
        val = ctypes.c_void_p(callback_addr or None)
        size = ctypes.c_size_t(ctypes.sizeof(ctypes.c_void_p))
        return (
            self._lib.cuptiActivitySetAttribute_v2(
                ctypes.c_void_p(sub_handle),
                ActivityAttr.TIMESTAMP_CALLBACK,
                ctypes.byref(size),
                ctypes.byref(val),
            )
            == CUPTI_SUCCESS
        )

    def disarm_approx_timestamp_callback(self, sub_handle: int) -> None:
        # [FUNCTION DOC]
        # Resets the subscriber's timestamp-callback attribute to null, restoring CUPTI's
        # default timer. The status code is ignored, so failures are silent. Called before
        # unsubscribing.
        """Restore CUPTI's default CPU timer for this subscriber (callback_addr=0), the
        inverse of arm_approx_timestamp_callback -- called before unsubscribe so a following
        consumer isn't left with our callback."""
        val = ctypes.c_void_p(None)
        size = ctypes.c_size_t(ctypes.sizeof(ctypes.c_void_p))
        self._lib.cuptiActivitySetAttribute_v2(
            ctypes.c_void_p(sub_handle),
            ActivityAttr.TIMESTAMP_CALLBACK,
            ctypes.byref(size),
            ctypes.byref(val),
        )

    def activity_flush_all(self) -> None:
        # [FUNCTION DOC]
        # Asks CUPTI to hand over all completed activity buffers (flags zero, never forced).
        # Side effect: triggers buffer-complete callbacks asynchronously. Raises CuptiError on
        # failure.
        """Hand over COMPLETED buffers only (``cuptiActivityFlushAll(0)``). The monitor
        never forces in-progress buffers (``CUPTI_ACTIVITY_FLAG_FLUSH_FORCED``): a
        forced flush consumes a still-running kernel's record (its real completion is
        then never re-delivered) and racing it against concurrent host activity is the
        flush race that corrupts the HES heap and freezes the decode worker."""
        self._check(self._lib.cuptiActivityFlushAll(0), "cuptiActivityFlushAll")

    def activity_get_num_dropped_records(
        self, sub_handle: int, ctx: int, stream_id: int
    ) -> int:
        # [FUNCTION DOC]
        # Returns the number of activity records CUPTI dropped for the given subscriber,
        # context and stream (zeros mean all). Uses the subscriber-scoped call; on any error
        # status it returns 0 rather than raising, so the count is best-effort.
        """Dropped-record count for the subscription.

        Must use the ``_v2`` (subscriber-scoped) entry point: the monitor enables and
        registers through the subscriber-scope API, and CUPTI refuses to mix scopes --
        the global ``cuptiActivityGetNumDroppedRecords`` returns
        CUPTI_ERROR_NOT_COMPATIBLE for every call once a subscriber is in use. That
        failure is otherwise invisible (the count is best-effort, so a bad status just
        yields 0) while CUPTI logs an internal error per call.
        """
        dropped = ctypes.c_size_t()
        rc = self._lib.cuptiActivityGetNumDroppedRecords_v2(
            ctypes.c_void_p(sub_handle),
            ctypes.c_void_p(ctx),
            ctypes.c_uint32(stream_id),
            ctypes.byref(dropped),
        )
        return dropped.value if rc == CUPTI_SUCCESS else 0

    def activity_enable_hw_trace(self, enabled: bool) -> None:
        # [FUNCTION DOC]
        # Turns CUPTI hardware tracing on or off process-wide. Raises CuptiError on failure.
        # Must be done before CUDA context creation to take effect as intended by callers.
        self._check(
            self._lib.cuptiActivityEnableHWTrace(1 if enabled else 0),
            "cuptiActivityEnableHWTrace",
        )

    def finalize(self) -> None:
        # [FUNCTION DOC]
        # Detaches and releases all of CUPTI for the whole process (a heavy global reset).
        # Raises CuptiError on failure. The monitor does not use it at stop.
        """cuptiFinalize -- detach and release ALL of CUPTI process-wide. This is a
        global, heavy reset for explicit *synchronous* teardown: e.g. releasing a
        stock Kineto session's CUPTI subscriber before a CUPTI-monitor session
        subscribes. The monitor itself never calls this at stop() -- it disarms
        user-defined records + unsubscribes instead, because cuptiFinalize is global
        (would clobber a concurrent consumer) and, run asynchronously (Kineto's
        TEARDOWN_CUPTI), can deadlock against another thread's CUPTI calls."""
        self._check(self._lib.cuptiFinalize(), "cuptiFinalize")

    # --- user-defined-records (subscription API) ---------------------------

    def subscribe(self, allow_multiple: bool = True, callback: Any = None) -> int:
        # [FUNCTION DOC]
        # Creates a CUPTI subscription named for the torch monitor. Args: allow_multiple
        # permits other CUPTI subscribers to coexist; callback is the single callback object
        # (the module no-op is used when none is given, looked up at call time). Returns the
        # opaque subscriber handle as an int. Raises CuptiError if CUPTI refuses (for example
        # another subscriber is attached) or hands back a null handle. The caller must keep
        # the callback object alive for the subscription's lifetime.
        """cuptiSubscribe_v2 -> opaque subscriber handle (the v2 activity API is
        subscription-scoped). ``allow_multiple`` requests coexistence with another CUPTI
        subscriber (e.g. Kineto); CUPTI returns
        CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED if it can't be honored.

        ``callback`` is the subscription's single ``CUpti_CallbackFunc`` -- a
        :data:`_CB_FUNC` instance the caller must keep alive for the life of the
        subscription. CUPTI allows exactly one per subscriber, so a consumer wanting
        several callbacks fans out inside it. Defaults to :data:`_NOOP_CB`, read from the
        module at call time so a consumer that swaps that global still gets its own
        callback on subscriptions it takes itself."""
        sub = ctypes.c_void_p()
        params = _SubscriberParams(
            structSize=ctypes.sizeof(_SubscriberParams),
            subscriberName=b"torch-cupti-monitor",
            oldSubscriberName=None,
            oldSubscriberSize=0,
            allowMultipleSubscribers=1 if allow_multiple else 0,
        )
        self._check(
            self._lib.cuptiSubscribe_v2(
                ctypes.byref(sub),
                _NOOP_CB if callback is None else callback,
                None,
                ctypes.byref(params),
            ),
            "cuptiSubscribe_v2",
        )
        if sub.value is None:
            raise CuptiError("cuptiSubscribe returned a null subscriber handle")
        return sub.value

    def unsubscribe(self, sub_handle: int) -> None:
        # [FUNCTION DOC]
        # Ends the CUPTI subscription identified by the handle. Raises CuptiError on failure.
        self._check(
            self._lib.cuptiUnsubscribe(ctypes.c_void_p(sub_handle)),
            "cuptiUnsubscribe",
        )

    def enable_callback(
        self, sub_handle: int, domain: int, cbid: int, enable: bool
    ) -> None:
        # [FUNCTION DOC]
        # Turns delivery of one subscriber callback, identified by domain and callback id, on
        # or off for the given subscription. Raises CuptiError on failure.
        """cuptiEnableCallback -- turn one ``(domain, cbid)`` subscriber callback on or
        off for this subscription. Callbacks fire synchronously on the application thread
        that made the CUDA call, unlike activity records."""
        self._check(
            self._lib.cuptiEnableCallback(
                1 if enable else 0, ctypes.c_void_p(sub_handle), domain, cbid
            ),
            "cuptiEnableCallback",
        )

    def arm_user_defined_records(
        self, sub_handle: int, request_addr: int, complete_addr: int
    ) -> None:
        # [FUNCTION DOC]
        # Switches the subscription into user-defined-record mode by setting the matching
        # attribute, then registers the native buffer-request and buffer-complete callback
        # addresses. Raises CuptiError if either step fails; the first step is not rolled
        # back when the second fails.
        """Turn on user-defined records for the subscription and register the v2
        buffer callbacks (the native pool's version=2 request/complete)."""
        enabled = ctypes.c_uint8(1)
        size = ctypes.c_size_t(1)
        self._check(
            self._lib.cuptiActivitySetAttribute_v2(
                ctypes.c_void_p(sub_handle),
                ActivityAttr.USER_DEFINED_RECORDS,
                ctypes.byref(size),
                ctypes.byref(enabled),
            ),
            "cuptiActivitySetAttribute_v2",
        )
        self._check(
            self._lib.cuptiActivityRegisterCallbacks_v2(
                ctypes.c_void_p(sub_handle),
                ctypes.c_void_p(request_addr),
                ctypes.c_void_p(complete_addr),
            ),
            "cuptiActivityRegisterCallbacks_v2",
        )

    def disarm_user_defined_records(self, sub_handle: int) -> None:
        # [FUNCTION DOC]
        # Turns user-defined-record mode off for the subscription. Best-effort: a nonzero
        # status is logged as a warning (with the decoded text) and not raised, so teardown
        # continues. Side effect: changes the CUPTI attribute and may log.
        """Turn user-defined records back off for the subscription (the inverse of
        arm_user_defined_records' set-attribute). UDR mode changes how CUPTI lays
        out activity records, so leaving it on can leave a following classic
        consumer (e.g. Kineto) unable to decode -- reset it before unsubscribing.

        Best-effort: some activity kinds (e.g. MEMCPY2) leave CUPTI rejecting the
        UDR-off toggle with CUPTI_ERROR_INVALID_OPERATION at teardown. The trace is
        already built by this point and we unsubscribe next regardless, so a failure
        here is logged, not raised -- crashing teardown would lose the export."""
        disabled = ctypes.c_uint8(0)
        size = ctypes.c_size_t(1)
        rc = self._lib.cuptiActivitySetAttribute_v2(
            ctypes.c_void_p(sub_handle),
            ActivityAttr.USER_DEFINED_RECORDS,
            ctypes.byref(size),
            ctypes.byref(disabled),
        )
        if rc != 0:
            logger.warning(
                "cuptiActivitySetAttribute_v2 (disarm UDR) failed with %s; "
                "continuing teardown",
                self._result_string(rc),
            )

    def enable_kernel_latency_timestamps(self, sub_handle: int, enable: bool) -> bool:
        # [FUNCTION DOC]
        # Sets the per-subscriber attribute that adds queued/submitted latency timestamps to
        # kernel records. Returns true if CUPTI accepted the setting, false if it rejected it;
        # never raises on a CUPTI error.
        """Toggle per-subscriber kernel latency-timestamp tracking (the queued/submitted
        fields on kernel records). Best-effort: returns False if CUPTI rejects the
        attribute (e.g. an ABI renumber on a newer version) so the session degrades to
        no queued/submitted rather than failing."""
        val = ctypes.c_uint8(1 if enable else 0)
        size = ctypes.c_size_t(1)
        return (
            self._lib.cuptiActivitySetAttribute_v2(
                ctypes.c_void_p(sub_handle),
                ActivityAttr.ENABLE_KERNEL_LATENCY_TIMESTAMPS,
                ctypes.byref(size),
                ctypes.byref(val),
            )
            == 0
        )

    def enable_cuda_event_device_timestamps(
        self, sub_handle: int, enable: bool
    ) -> bool:
        # [FUNCTION DOC]
        # Sets the per-subscriber attribute that adds device-side timestamps to CUDA_EVENT
        # records. Returns true if accepted, false if CUPTI rejects it; never raises on a
        # CUPTI error.
        """Toggle per-subscriber device-side timestamps on CUDA_EVENT records (the
        ``deviceTimestamp`` field, off by default). Best-effort: returns False if CUPTI
        rejects the attribute so the session degrades to no event timestamps."""
        val = ctypes.c_uint8(1 if enable else 0)
        size = ctypes.c_size_t(1)
        return (
            self._lib.cuptiActivitySetAttribute_v2(
                ctypes.c_void_p(sub_handle),
                ActivityAttr.ENABLE_CUDA_EVENT_DEVICE_TIMESTAMPS,
                ctypes.byref(size),
                ctypes.byref(val),
            )
            == 0
        )

    def activity_enable(
        self, sub_handle: int, kind: ActivityKind, field_ids: Iterable[int]
    ) -> None:
        # [FUNCTION DOC]
        # Enables one activity kind on the subscription with a user-defined field selection.
        # The selected ids are sorted with the mandatory kind field id (zero) placed first
        # and any other zeros dropped. Raises CuptiError if CUPTI rejects the enable. After a
        # successful enable of the runtime or driver kind, also disables that API's noisy
        # callback ids so they do not fill the buffers. Side effect: lazy import of the
        # cupti-python ActivityKind enum.
        """Enable a kind with a user-defined field selection. CUPTI requires the
        FIELD_KIND id (0) to be the first selected field."""
        ordered = (0, *sorted(f for f in field_ids if f != 0))
        arr = (ctypes.c_int * len(ordered))(*ordered)
        sel = _UDFieldSelection(
            structSize=ctypes.sizeof(_UDFieldSelection),
            numFields=len(ordered),
            pFieldIds=ctypes.cast(arr, ctypes.POINTER(ctypes.c_int)),
        )
        cfg = _UDActivityConfig(
            structSize=ctypes.sizeof(_UDActivityConfig), fieldSelection=sel
        )
        self._check(
            self._lib.cuptiActivityEnable_v2(
                ctypes.c_void_p(sub_handle), kind, ctypes.byref(cfg)
            ),
            "cuptiActivityEnable_v2",
        )
        # ``kind`` is a plain int here, so compare by value. The per-cbid runtime/driver
        # disable must follow the kind enable.
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

        k = int(kind)
        if k == int(ActivityKind.RUNTIME):
            self.disable_noisy_runtime_apis(sub_handle)
        elif k == int(ActivityKind.DRIVER):
            self.disable_noisy_driver_apis(sub_handle)

    def disable_noisy_runtime_apis(self, sub_handle: int) -> None:
        # [FUNCTION DOC]
        # For each noisy runtime-API callback id, tells CUPTI (subscriber-scoped call) to stop
        # generating records. Does nothing if no ids resolve or the entry point is missing.
        # Return codes are ignored. KNOWN ISSUE: unlike other wrapper methods the status of
        # these calls is not checked, so failures are silent.
        """Stop CUPTI emitting RUNTIME records for the noise-only cbids
        (cudaGetDevice/SetDevice/GetLastError) so they don't fill the UDR buffers, via the
        subscriber-scoped _v2 entry (the UDR path's form). A no-op if it is absent."""
        cbids = _noisy_runtime_cbids()
        fn_v2 = getattr(self._lib, "cuptiActivityEnableRuntimeApi_v2", None)
        if not cbids or fn_v2 is None:
            return
        for cbid in cbids:
            fn_v2(ctypes.c_void_p(sub_handle), ctypes.c_uint32(cbid), ctypes.c_uint8(0))

    def disable_noisy_driver_apis(self, sub_handle: int) -> None:
        # [FUNCTION DOC]
        # Same as the runtime variant but for the noisy driver-API callback ids: asks CUPTI
        # (subscriber-scoped) to stop generating their records; no-op if none resolve or the
        # entry point is missing; statuses are ignored.
        """Stop CUPTI emitting DRIVER records for the noise-only cbids
        (cuKernelGetAttribute/cuDevicePrimaryCtxGetState/cuCtxGetCurrent) so they don't
        fill the UDR buffers, via the subscriber-scoped _v2 entry (the UDR path's form). A
        no-op if it is absent."""
        cbids = _noisy_driver_cbids()
        fn_v2 = getattr(self._lib, "cuptiActivityEnableDriverApi_v2", None)
        if not cbids or fn_v2 is None:
            return
        for cbid in cbids:
            fn_v2(ctypes.c_void_p(sub_handle), ctypes.c_uint32(cbid), ctypes.c_uint8(0))

    def activity_disable(self, sub_handle: int, kind: ActivityKind) -> None:
        # [FUNCTION DOC]
        # Disables one activity kind on the subscription. Raises CuptiError on failure.
        self._check(
            self._lib.cuptiActivityDisable_v2(ctypes.c_void_p(sub_handle), kind),
            "cuptiActivityDisable_v2",
        )

    def activity_push_external_correlation_id(
        self,
        external_id: int,
        kind: ExternalCorrelationKind | None = None,
        sub_handle: int | None = None,
    ) -> bool:
        # [FUNCTION DOC]
        # Pushes an external-correlation id onto CUPTI's per-thread stack for the given kind
        # (default is the first custom kind). Uses the subscriber-aware variant when a handle
        # is supplied and that entry point exists, else the plain call. Returns true on
        # success, false on any CUPTI error (no exception). Side effect: changes CUPTI state.
        """Push an external-correlation id (default kind CUSTOM1) onto CUPTI's
        process-global stack. Best-effort: returns False on failure. Pass
        ``sub_handle`` on the v2 path -- the plain call returns NOT_COMPATIBLE while
        a user-defined-record subscriber is active, so the subscriber-aware _v2
        variant is used when a handle is given and it's available."""
        if kind is None:
            kind = ExternalCorrelationKind.CUSTOM1
        if sub_handle is not None and hasattr(
            self._lib, "cuptiActivityPushExternalCorrelationId_v2"
        ):
            rc = self._lib.cuptiActivityPushExternalCorrelationId_v2(
                sub_handle, int(kind), ctypes.c_uint64(external_id)
            )
        else:
            rc = self._lib.cuptiActivityPushExternalCorrelationId(
                int(kind), ctypes.c_uint64(external_id)
            )
        return rc == CUPTI_SUCCESS

    def activity_pop_external_correlation_id(
        self,
        kind: ExternalCorrelationKind | None = None,
        sub_handle: int | None = None,
    ) -> int | None:
        # [FUNCTION DOC]
        # Pops the most recent external-correlation id for the given kind (default first
        # custom kind), using the subscriber-aware variant when a handle is given and
        # available. Returns the popped id, or None on any CUPTI error (no exception).
        # Side effect: changes CUPTI's stack.
        """Pop the most recent external-correlation id (default kind CUSTOM1), or
        None on failure. Pass ``sub_handle`` on the v2 path (see the push docstring)."""
        if kind is None:
            kind = ExternalCorrelationKind.CUSTOM1
        last = ctypes.c_uint64()
        if sub_handle is not None and hasattr(
            self._lib, "cuptiActivityPopExternalCorrelationId_v2"
        ):
            rc = self._lib.cuptiActivityPopExternalCorrelationId_v2(
                sub_handle, int(kind), ctypes.byref(last)
            )
        else:
            rc = self._lib.cuptiActivityPopExternalCorrelationId(
                int(kind), ctypes.byref(last)
            )
        return last.value if rc == CUPTI_SUCCESS else None


@lru_cache(maxsize=1)
def pylibcupti() -> _PyLibCupti:
    # [FUNCTION DOC]
    # Returns the cached process-wide wrapper. On first call, loads libcupti by its soname
    # (raising OSError if it cannot be loaded), declares its ctypes prototypes, and wraps the
    # handle; later calls return the same object.
    """The process-wide CUPTI Activity API wrapper: libcupti loaded and ctypes
    prototypes bound once. All libcupti calls go through this object -- callers
    never touch the CDLL or ctypes directly."""
    lib = ctypes.CDLL(LIBCUPTI_SONAME)
    _configure_ctypes(lib)
    return _PyLibCupti(lib)
