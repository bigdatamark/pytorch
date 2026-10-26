# [MODULE DOC] Shared base infrastructure for Inductor code generation backends (Triton, C++, Halide, MPS, Pallas, ...).
# Defines: the backend registry (device -> scheduling / wrapper codegen / device op overrides), kernel argument
# bookkeeping (KernelArgs, TensorArg, SizeArg, WorkspaceArg, ...), common subexpression elimination (CSE,
# CSEVariable), the Kernel base class and the CSEProxy ops handler that routes ops calls to a kernel, default
# pointwise op decompositions/overrides (OpDecompositions, OpOverrides, pointwise_overrides_data), dtype
# propagation over loop bodies, and the KernelTemplate base class.
# Import-time side effects: creates module loggers and the empty global registries (device_codegens,
# device_op_overrides_dict, custom_backend_passes, custom_backend_codegen_configs); actual backends are
# registered lazily by init_backend_registration.
from __future__ import annotations

import atexit
import contextlib
import dataclasses
import enum
import functools
import itertools
import logging
import math
import operator
import os
import re
import tempfile
import threading
from abc import ABC, abstractmethod
from enum import auto, Enum
from itertools import chain
from typing import Any, cast, ClassVar, Generic, NamedTuple, TYPE_CHECKING
from typing_extensions import Self, TypeVar

import sympy

import torch
import torch.fx
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
from torch.utils import _pytree as pytree
from torch.utils._config_module import ConfigModule
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import Max
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.printers import PythonPrinter as _PythonPrinter
from torch.utils._sympy.symbol import free_symbol_is_type, symbol_is_type, SymT
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges

from .. import config, metrics
from ..dtype_propagation import DtypePropagationOpsHandler
from ..ops_handler import BasicMathOpsMixin, DefaultHandler
from ..shape_propagation import ShapePropagationOpsHandler
from ..utils import (
    boolean_ops,
    DeferredLineBase,
    generate_assert,
    get_current_backend,
    IndentedBuffer,
    ir_dataclass,
    ScopedDict,
    sympy_dot,
    sympy_index_symbol,
    sympy_subs,
    triton_type,
    unique,
)
from ..virtualized import (
    NullHandler,
    ops,
    OpsHandler,
    OpsValue,
    ReductionType,
    StoreMode,
    V,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, MutableMapping, Sequence

    from torch.fx import GraphModule

    from ..custom_graph_pass import CustomGraphModulePass
    from ..ir import Buffer, ChoiceCaller, FixedLayout, IRNode
    from ..loop_body import LoopBody
    from ..scheduler import BaseScheduling, Scheduler, SchedulerNode
    from ..shape_propagation import BlockShapeType
    from .wrapper import PythonWrapperCodegen

    _T = TypeVar("_T")
    SchedulingConstructor = Callable[[Scheduler | None], BaseScheduling]
    WrapperConstructor = type[PythonWrapperCodegen]
    SymbolLike = str | sympy.Symbol

    # OpVarT should really be Union[CSEVariable, str], however this
    # causes typing errors in subclasses (defined in other files).
    OpVarT = str

schedule_log = torch._logging.getArtifactLogger(__name__, "schedule")
log = logging.getLogger(__name__)


def data_type_logger(msg: str) -> None:
    # [FUNCTION DOC] HELPER: Logs a data type propagation message at debug level on the schedule artifact logger, only when debug is enabled.
    if schedule_log.isEnabledFor(logging.DEBUG):
        schedule_log.debug("Data type propagation: %s", msg)


@dataclasses.dataclass
class FileBackedGraphModule:
    # [CLASS DOC] Result of FX wrapper codegen. Wraps a GraphModule plus its compiled callable and mimics the module interface
    # (file name, source value, call). On creation it writes the graph code to a temporary python file so debugging
    # tools can find a source file; the file is removed at interpreter exit.
    """
    Output of FX wrapper codegen. Exposes the same methods as ModuleType, but these
    map back to a GraphModule instead of Python source.
    """

    gm: GraphModule
    compiled_fn: Callable[..., Any]

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Dataclass post-init hook. Creates a named temporary file (not auto deleted) with a .py suffix, registers an
        # exit handler to remove it, and writes the module source (the graph module code) into it. Side effects: file I/O
        # and atexit registration. Stores the open handle in the tempfile attribute.
        # Write the code to a file for compatibility with debugging utilities.
        # The file is deleted upon program termination.
        self.tempfile = tempfile.NamedTemporaryFile(  # noqa: SIM115
            mode="w+", suffix=".py", delete=False
        )
        atexit.register(os.remove, self.tempfile.name)
        with self.tempfile as f:
            f.write(self.value)

    @property
    def __file__(self) -> str:
        # [FUNCTION DOC] Property returning the path of the temporary file holding the generated source, so the object looks like a module with a file attribute.
        return self.tempfile.name

    def call(self, args: list[Any]) -> Any:
        # [FUNCTION DOC] Calls the compiled function with the given list of arguments unpacked positionally and returns its result.
        return self.compiled_fn(*args)

    @property
    def value(self) -> str:
        # [FUNCTION DOC] Property returning the python source code of the wrapped GraphModule as a string.
        return self.gm.code


class WorkspaceZeroMode(enum.Enum):
    # [CLASS DOC] Enum describing when a workspace buffer must be zeroed: uninitialized (no zeroing needed), zero on call (kernel
    # may leave the workspace dirty) and zero per graph (kernel must re-zero it itself). Provides combine and from_bool helpers.
    UNINITIALIZED = 0
    ZERO_ON_CALL = 1  # kernel may leave workspace dirty
    ZERO_PER_GRAPH = 2  # must be re-zeroed by kernel

    @staticmethod
    def combine(a: WorkspaceZeroMode, b: WorkspaceZeroMode) -> WorkspaceZeroMode:
        # [FUNCTION DOC] Merges two zero modes. Returns the first if both are equal or the second is uninitialized, returns the second
        # if the first is uninitialized, otherwise raises NotImplementedError (conflicting modes).
        if a == b or b == WorkspaceZeroMode.UNINITIALIZED:
            return a
        if a == WorkspaceZeroMode.UNINITIALIZED:
            return b
        raise NotImplementedError(f"WorkspaceZeroMode.combine({a!r}, {b!r})")

    @staticmethod
    def from_bool(zero_fill: bool) -> WorkspaceZeroMode:
        # [FUNCTION DOC] Maps a zero fill flag to a mode: zero on call if true, uninitialized if false.
        if zero_fill:
            return WorkspaceZeroMode.ZERO_ON_CALL
        return WorkspaceZeroMode.UNINITIALIZED


class CodegenSymbol(ABC):
    # [CLASS DOC] Abstract base for IR objects that may correspond to a variable in the wrapper code; requires a name and an example value.
    """
    An IR object possibly corresponding to a variable in the wrapper code.
    """

    @abstractmethod
    def get_name(self) -> str:
        # [FUNCTION DOC] Abstract: returns the name of the symbol in the generated code.
        pass

    @abstractmethod
    def get_example(self) -> torch.Tensor | sympy.Symbol:
        # [FUNCTION DOC] Abstract: returns an example tensor or sympy symbol representing this object.
        pass


@ir_dataclass(frozen=True)
class WorkspaceArg(CodegenSymbol):
    # [CLASS DOC] A temporary per-kernel scratch buffer (frozen IR dataclass), not registered as a real buffer so it is never
    # dead-code eliminated. Fields: count (element count as sympy expression), zero_mode, device, outer_name (name in
    # the wrapper), inner_name (kernel argument name, default ws_ptr) and dtype (default uint8). Implements enough of the
    # buffer interface (layout, size, stride, dtype, device, name) for allocation code to treat it like a buffer.
    """A temporary buffer used for a single kernel, then discarded.

    Not registered as a traditional buffer since there are no users,
    so it would be dead code eliminated.

    Args:
        nbytes: The size of the buffer in bytes.
        zero_fill: Whether the buffer should be initialized to zero.

    """

    count: sympy.Expr
    zero_mode: WorkspaceZeroMode
    device: torch.device
    outer_name: str
    inner_name: str = "ws_ptr"
    dtype: torch.dtype = torch.uint8

    @staticmethod
    def unique_name(prefix: str = "workspace_") -> str:
        # [FUNCTION DOC] Returns a fresh graph-unique workspace name by appending the next value of the graph's workspace id counter to the prefix. Mutates the counter.
        return f"{prefix}{next(V.graph.workspace_id)}"

    @staticmethod
    def can_join(a: WorkspaceArg, b: WorkspaceArg) -> bool:
        # [FUNCTION DOC] Returns true when two workspace args can be merged: same inner name, dtype and device.
        return (
            a.inner_name == b.inner_name and a.dtype == b.dtype and a.device == b.device
        )

    @staticmethod
    def join(a: WorkspaceArg, b: WorkspaceArg) -> WorkspaceArg:
        # [FUNCTION DOC] Returns a new WorkspaceArg whose count is the sum of both counts and whose zero mode is the combination of both modes; other fields come from the first arg.
        return WorkspaceArg(
            count=a.count + b.count,
            zero_mode=WorkspaceZeroMode.combine(a.zero_mode, b.zero_mode),
            dtype=a.dtype,
            device=a.device,
            inner_name=a.inner_name,
            outer_name=a.outer_name,
        )

    @staticmethod
    def maximum(a: WorkspaceArg, b: WorkspaceArg) -> WorkspaceArg:
        # [FUNCTION DOC] Returns a new WorkspaceArg whose count is the symbolic max of the two counts and zero mode the combination of both.
        # Raises AssertionError if dtype, device or inner name differ.
        if not (
            a.dtype == b.dtype and a.device == b.device and a.inner_name == b.inner_name
        ):
            raise AssertionError(
                "WorkspaceArg.maximum requires matching dtype, device, and inner_name"
            )
        return WorkspaceArg(
            count=Max(a.count, b.count),
            zero_mode=WorkspaceZeroMode.combine(a.zero_mode, b.zero_mode),
            dtype=a.dtype,
            device=a.device,
            inner_name=a.inner_name,
            outer_name=a.outer_name,
        )

    # These methods let WorkspaceArg pretend it is a buffer to reuse allocation code
    def get_device(self) -> torch.device:
        # [FUNCTION DOC] Returns the device of the workspace (buffer interface shim).
        return self.device

    get_device_or_error = get_device

    def get_dtype(self) -> torch.dtype:
        # [FUNCTION DOC] Returns the dtype of the workspace (buffer interface shim).
        return self.dtype

    def get_example(self) -> torch.Tensor | sympy.Symbol:
        # [FUNCTION DOC] Returns the example value of the fixed layout built from this workspace.
        return self.get_layout().get_example()

    def get_layout(self) -> FixedLayout:
        # [FUNCTION DOC] Builds and returns a FixedLayout with this device and dtype, a single dimension of size count and stride one.
        from ..ir import FixedLayout

        return FixedLayout(
            device=self.device,
            dtype=self.dtype,
            size=[self.count],
            stride=[1],
        )

    @property
    def layout(self) -> FixedLayout:
        # [FUNCTION DOC] Property alias returning the fixed layout of this workspace.
        return self.get_layout()

    get_output_spec = get_layout
    maybe_get_output_spec = get_layout
    maybe_get_layout = get_layout

    def get_offset(self) -> sympy.Expr:
        # [FUNCTION DOC] Returns the storage offset, always zero.
        return sympy.S.Zero

    def get_size(self) -> list[sympy.Expr]:
        # [FUNCTION DOC] Returns the size as a one element list containing count.
        return [self.count]

    def get_stride(self) -> list[sympy.Expr]:
        # [FUNCTION DOC] Returns the stride as a one element list containing one.
        return [sympy.S.One]

    def get_name(self) -> str:
        # [FUNCTION DOC] Returns the outer (wrapper level) name of the workspace.
        return self.outer_name

    def get_is_pinned(self) -> bool:
        # [FUNCTION DOC] Returns false; a workspace is never pinned memory.
        return False

    def get_inputs_that_alias_output(self) -> list[str]:
        # [FUNCTION DOC] Returns an empty list; a workspace never aliases kernel inputs and outputs.
        return []


class TritonScratchWorkspace:
    # [CLASS DOC] Describes a Triton scratch (global scratch) workspace: holds its size and a callable used to produce the dtype string for it.
    def __init__(self, size: int, generate_dtype_str: Callable[..., str]):
        # [FUNCTION DOC] Stores the workspace size and the callable that generates the dtype string.
        self.size = size
        self._generate_dtype_str = generate_dtype_str

    def generate_dtype_str(self) -> str:
        # [FUNCTION DOC] Calls the stored callable and returns the dtype string it produces.
        return self._generate_dtype_str()


@dataclasses.dataclass
class TensorArg:
    # [CLASS DOC] Dataclass describing a tensor kernel argument: kernel arg name, backing buffer name, dtype, an offset (used by C++ only, default zero) and an alias_of name (used by Halide only).
    name: str
    buffer: str
    dtype: torch.dtype
    offset: sympy.Expr = sympy.S.Zero  # c++ only
    alias_of: str | None = None  # halide only


@dataclasses.dataclass
class SizeArg:
    # [CLASS DOC] Dataclass describing a scalar size kernel argument: its name and the sympy expression it carries.
    name: str
    expr: sympy.Expr

    @property
    def alias_of(self) -> str | None:
        # [FUNCTION DOC] Property returning None; size args never alias another argument.
        return None


@dataclasses.dataclass
class ConstexprArg:
    # [CLASS DOC] Dataclass for a compile-time constant (constexpr) kernel argument; holds only its name.
    name: str


@dataclasses.dataclass
class TMADescriptorArg:
    # [CLASS DOC] Dataclass describing a TMA descriptor kernel argument: name, api type (experimental or stable), block shape and dtype (the last two only needed for the stable API).
    name: str
    api_type: str  # "experimental" or "stable"
    block_shape: list[sympy.Expr] | None  # only needed for "stable"
    dtype: torch.dtype | None  # only needed for "stable"


@dataclasses.dataclass
class DeviceCodegen:
    # [CLASS DOC] Dataclass bundling per-device codegen constructors registered for a device: scheduling constructor, Python wrapper
    # codegen class, and optional C++ wrapper and FX wrapper codegen classes.
    scheduling: SchedulingConstructor
    wrapper_codegen: WrapperConstructor
    cpp_wrapper_codegen: WrapperConstructor | None = None
    fx_wrapper_codegen: WrapperConstructor | None = None


KernelArgType = WorkspaceArg | TensorArg | SizeArg | TMADescriptorArg | ConstexprArg

# Device index to emit into generated code: either a literal compile-time index, or a
# code expression evaluated at run time (e.g. current_device_idx_expr() under
# compile-on-one-rank).
DeviceIdx = int | str

device_codegens: dict[str, DeviceCodegen] = {}


class DeviceOpOverrides:
    # [CLASS DOC] Base interface producing device specific code snippets (strings) for streams, device guards, kernel headers
    # and drivers used by the wrapper. Methods raise NotImplementedError unless noted; backends subclass and register
    # an instance per device with register_device_op_overrides.
    def import_get_raw_stream_as(self, name: str) -> str:
        # [FUNCTION DOC] Returns python code text for importing the raw stream getter under the given alias. Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def set_device(self, device_idx: DeviceIdx) -> str:
        # [FUNCTION DOC] Returns python code text for setting the current device to the given index. Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def synchronize(self) -> str:
        # [FUNCTION DOC] Returns python code text for synchronizing the device. Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def device_guard(self, device_idx: DeviceIdx) -> str:
        # [FUNCTION DOC] Returns python code text for a python device guard context for the given index. Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def current_device_idx_expr(self) -> str:
        # [FUNCTION DOC] Returns a runtime expression (string) that evaluates to the current device index, used under compile-on-one-rank
        # so the wrapper is rank agnostic. The base version raises RuntimeError with an actionable message naming the
        # subclass; only CUDA/ROCm provide an implementation.
        # Runtime expression evaluating to the current device index. Used under
        # compile-on-one-rank so the wrapper resolves its device at run time
        # (rank-agnostic) instead of baking the compile-time index. Only CUDA/ROCm
        # implements this today; raise something actionable rather than a bare
        # NotImplementedError from deep inside device-context codegen.
        raise RuntimeError(
            f"compile-on-one-rank (device-as-parameter) is not supported on "
            f"{type(self).__name__}: it has no current_device_idx_expr(), so the "
            f"generated wrapper would bake the compile-time device index and not be "
            f"rank-portable."
        )

    def current_stream(self) -> str:
        # [FUNCTION DOC] Returns python code text for obtaining the current stream. Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def stream_handle(self, stream_name: str) -> str:
        # [FUNCTION DOC] Returns the code text for the native handle of the named stream: the stream name followed by the native_handle attribute. Overridable.
        return f"{stream_name}.native_handle"

    def cpp_device_guard(self) -> str:
        # [FUNCTION DOC] Returns code text for a C++ device guard. Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def cpp_aoti_device_guard(self) -> str:
        # [FUNCTION DOC] Returns code text for an AOTI C++ device guard. Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def cpp_stream_guard(self) -> str:
        # [FUNCTION DOC] Returns code text for a C++ stream guard. Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def cpp_aoti_stream_guard(self) -> str:
        # [FUNCTION DOC] Returns code text for an AOTI C++ stream guard. Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def cpp_getStreamFromExternal(self) -> str:
        # [FUNCTION DOC] Returns code text for the C++ call that obtains a stream from an external handle. Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def kernel_header(self) -> str:
        # [FUNCTION DOC] Returns code text for the kernel source header. Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def kernel_driver(self) -> str:
        # [FUNCTION DOC] Returns code text for the kernel driver helper code. Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def cpp_stream_type(self) -> str:
        # [FUNCTION DOC] Returns code text for the C++ stream type name. Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def aoti_get_stream(self) -> str:
        # [FUNCTION DOC] Returns code text for the AOTI call that obtains the stream. Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def cpp_kernel_type(self) -> str:
        # [FUNCTION DOC] Returns code text for the C++ kernel (function handle) type name. Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def cpp_device_ptr(self) -> str:
        # [FUNCTION DOC] Returns code text for the C++ device pointer type name. Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def tma_descriptor_helpers(self) -> str:
        # [FUNCTION DOC] Returns code text for TMA descriptor helper code. Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def cpp_scratch(
        self, idx: int, workspace: TritonScratchWorkspace, prefix: str | None = None
    ) -> tuple[list[str], str] | None:
        # [FUNCTION DOC] Optionally returns a pair of (list of scratch definition lines, argument name) for a Triton scratch workspace
        # at the given index with an optional name prefix. Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        # optionally return (scratch definition, arg name)
        raise NotImplementedError


# Thread-safe lazy initialization for device op overrides
device_op_overrides_lock = threading.RLock()
device_op_overrides_dict: dict[str, DeviceOpOverrides] = {}
_device_op_overrides_initialized = False
custom_backend_passes: dict[str, CustomGraphModulePass | None] = {}
custom_backend_codegen_configs: dict[str, ConfigModule | None] = {}


# The code generated by Inductor consists of two main parts: kernel code and wrapper code.
# For any new backend looking to integrate with Inductor, customization of these two main
# parts are necessary to generate its specific code.
#
# Kernel code generation is determined by different Scheduling. Consequently, a new
# backend needs to provide a custom Scheduling for its unique kernel code generation. Currently,
# CppScheduling and TritonScheduling serve the C++/OpenMP and Triton backends, respectively.
#
# For the Wrapper, Inductor provides a PythonWrapperCodegen class to generate the Python wrapper code
# that bridges kernels. This allows out-of-tree backends to inherit from PythonWrapperCodegen,
# and override specific member functions to create backend-specific Python wrapper code.
#
# Other classes, such as CppKernel and TritonKernel, used for code generation, typically form part
# of the logic for either Scheduling or PythonWrapperCodegen. So the Scheduling and PythonWrapperCodegen interfaces
# provide flexibility to the backend. A backend can choose to implement these classes from scratch,
# or reuse them by extending and overriding as necessary. And Inductor provides the registration API,
# register_backend_for_device, to equip a new backend at runtime.
#
# Intel has developed a new backend on top of Triton to support Intel GPUs, leveraging these interfaces.
# This backend can be used as a reference:
# https://github.com/intel/intel-extension-for-pytorch/blob/5dcc9d57e5422cf295e1a1ee97896d6b6a554a85/intel_extension_for_pytorch/_inductor/__init__.py#L9
def register_backend_for_device(
    device: str,
    device_scheduling: SchedulingConstructor,
    device_wrapper_codegen: WrapperConstructor,
    device_cpp_wrapper_codegen: WrapperConstructor | None = None,
    device_fx_wrapper_codegen: WrapperConstructor | None = None,
    device_custom_pass: CustomGraphModulePass | None = None,
    device_custom_config: ConfigModule | None = None,
) -> None:
    # [FUNCTION DOC] Registers the codegen components for a device name: stores a DeviceCodegen record (scheduling, python wrapper,
    # optional C++ and FX wrappers) in the global device_codegens map, stores the optional custom graph pass and custom
    # config in their global maps. Raises AssertionError if a custom config is given that is not a ConfigModule or is the
    # default inductor config. Mutates global registries.
    device_codegens[device] = DeviceCodegen(
        device_scheduling,
        device_wrapper_codegen,
        device_cpp_wrapper_codegen,
        device_fx_wrapper_codegen,
    )
    custom_backend_passes[device] = device_custom_pass
    if device_custom_config:
        if not (
            isinstance(device_custom_config, ConfigModule)
            and device_custom_config is not config
        ):
            raise AssertionError(
                f"{device_custom_config=} cannot be the same as the default inductor config {config=}"
            )
    custom_backend_codegen_configs[device] = device_custom_config


class BackendFeature(Enum):
    # [CLASS DOC] Enum of optional features a backend may support (foreach, bucketize, inplace buffers, masked scatter, scan, sort, tuple reduction, store loop order preference, triton templates, reduce to single element).
    FOREACH = auto()
    BUCKETIZE = auto()
    INPLACE_BUFFERS = auto()
    MASKED_SCATTER_WITH_INDEX = auto()
    SCAN = auto()
    SORT = auto()
    TUPLE_REDUCTION = auto()
    PREFER_STORE_LOOP_ORDER = auto()
    TRITON_TEMPLATES = auto()
    REDUCE_TO_SINGLE_ELEMENT = auto()


def get_backend_features(
    device: torch.device | str | None,
) -> OrderedSet[BackendFeature]:
    # [FUNCTION DOC] Returns the set of BackendFeature values supported for a device (given as a torch device or device type string).
    # Returns an empty set for None. Ensures backends are registered, builds the scheduling object for the device type with
    # no scheduler and asks it for its features. Raises AssertionError for a bad argument type or when no scheduling is
    # registered for that device type.
    if device is None:
        return OrderedSet()
    init_backend_registration()
    if isinstance(device, torch.device):
        device_type = device.type
    else:
        if not isinstance(device, str):
            raise AssertionError(type(device))
        device_type = device
        device = torch.device(device_type)
    scheduling_ctor = get_scheduling_for_device(device_type)
    if not scheduling_ctor:
        raise AssertionError(f"no scheduling registered for device {device_type}")
    scheduling = scheduling_ctor(None)
    return scheduling.get_backend_features(device)


def has_backend_feature(
    device: torch.device | str | None, feature: BackendFeature
) -> bool:
    # [FUNCTION DOC] Returns true if the given feature is in the backend feature set of the device. Raises AssertionError if feature is not a BackendFeature. See also the graph level has_feature.
    """See also V.graph.has_feature"""
    if not isinstance(feature, BackendFeature):
        raise AssertionError(f"expected BackendFeature, got {type(feature)}")
    return feature in get_backend_features(device)


def get_scheduling_for_device(device: str) -> SchedulingConstructor | None:
    # [FUNCTION DOC] Returns the registered scheduling constructor for the device name, or None if the device is not registered.
    return device_codegens[device].scheduling if device in device_codegens else None


def get_wrapper_codegen_for_device(
    device: str, cpp_wrapper: bool = False, fx_wrapper: bool = False
) -> WrapperConstructor | None:
    # [FUNCTION DOC] Returns the wrapper codegen class registered for a device, or None if unregistered. Selects the FX wrapper if
    # fx_wrapper is set, else the C++ wrapper if cpp_wrapper is set (swapping in the array-ref CPU C++ wrapper when the
    # device is cpu, the AOTI stack allocation config is enabled and the registered class is the plain CPU C++ wrapper),
    # else the Python wrapper. Reads config at call time.
    if device in device_codegens:
        wrapper_codegen_obj: DeviceCodegen = device_codegens[device]
        if fx_wrapper:
            return wrapper_codegen_obj.fx_wrapper_codegen
        elif cpp_wrapper:
            cpp_wrapper_codegen = wrapper_codegen_obj.cpp_wrapper_codegen
            # allow_stack_allocation is a per-compile config, so the arrayref CPU
            # wrapper must follow the current config value rather than whatever it
            # happened to be at first registration (see init_backend_registration).
            if device == "cpu" and config.aot_inductor.allow_stack_allocation:
                from .cpp_wrapper_cpu import CppWrapperCpu
                from .cpp_wrapper_cpu_array_ref import CppWrapperCpuArrayRef

                if cpp_wrapper_codegen is CppWrapperCpu:
                    return CppWrapperCpuArrayRef
            return cpp_wrapper_codegen
        else:
            return wrapper_codegen_obj.wrapper_codegen
    return None


def get_custom_backend_pass_for_device(device: str) -> CustomGraphModulePass | None:
    # [FUNCTION DOC] Returns the custom graph pass registered for the device, or None.
    return custom_backend_passes.get(device)


def get_custom_backend_config_for_device(device: str) -> ConfigModule | None:
    # [FUNCTION DOC] Returns the custom codegen config registered for the device, or None.
    return custom_backend_codegen_configs.get(device)


@functools.cache
def init_backend_registration() -> None:
    # [FUNCTION DOC] Cached, runs once. Lazily imports backend modules and registers default scheduling and wrapper codegens for
    # cpu, cuda, tpu, xpu, mps and mtia if not already registered, using config to choose the backend implementation
    # (cpu_backend, cuda_backend, tpu_backend). Also registers a privateuse1 custom backend if its module provides
    # Scheduling and wrapper classes (a RuntimeError during lookup is ignored). Mutates global registries.
    """
    Register the backend for different devices, including the scheduling
    for kernel code generation and the host side wrapper code generation.
    """
    from .cpp import CppScheduling
    from .cpp_wrapper_cpu import CppWrapperCpu
    from .cpp_wrapper_gpu import CppWrapperGpu
    from .cpp_wrapper_mps import CppWrapperMps
    from .cuda_combined_scheduling import CUDACombinedScheduling
    from .halide import HalideScheduling
    from .mps import MetalScheduling
    from .pallas import PallasScheduling
    from .python_wrapper_mtia import PythonWrapperMtia
    from .triton import TritonScheduling
    from .wrapper import PythonWrapperCodegen
    from .wrapper_fxir import WrapperFxCodegen
    from .xpu.xpu_combined_scheduling import XPUCombinedScheduling

    if get_scheduling_for_device("cpu") is None:
        cpu_backends = {
            "cpp": CppScheduling,
            "halide": HalideScheduling,
            "triton": TritonScheduling,
            "pallas": PallasScheduling,
        }
        register_backend_for_device(
            "cpu",
            lambda scheduling: cpu_backends[config.cpu_backend](scheduling),
            PythonWrapperCodegen,
            # allow_stack_allocation selects CppWrapperCpuArrayRef, but that is a
            # per-compile config; the choice is made dynamically in
            # get_wrapper_codegen_for_device rather than frozen here at the
            # process's first registration.
            CppWrapperCpu,
            WrapperFxCodegen,
        )

    if get_scheduling_for_device("cuda") is None:
        # CUDACombinedScheduling combines Triton and CUDA C++ scheduling for CUDA devices via delegation
        cuda_backends = {
            "triton": CUDACombinedScheduling,
            "halide": HalideScheduling,
            "pallas": PallasScheduling,
        }
        register_backend_for_device(
            "cuda",
            lambda scheduling: cuda_backends[config.cuda_backend](scheduling),
            PythonWrapperCodegen,
            CppWrapperGpu,
            WrapperFxCodegen,
        )

    if get_scheduling_for_device("tpu") is None:
        register_backend_for_device(
            "tpu",
            PallasScheduling,
            PythonWrapperCodegen,
            # CppWrapperGpu,
            # WrapperFxCodegen,
        )

    if get_scheduling_for_device("xpu") is None:
        register_backend_for_device(
            "xpu",
            XPUCombinedScheduling,
            PythonWrapperCodegen,
            CppWrapperGpu,
            WrapperFxCodegen,
        )

    if get_scheduling_for_device("tpu") is None:
        tpu_backends = {
            "pallas": PallasScheduling,
        }
        register_backend_for_device(
            "tpu",
            lambda scheduling: tpu_backends[config.tpu_backend](scheduling),
            PythonWrapperCodegen,
        )

    if get_scheduling_for_device("mps") is None:
        register_backend_for_device(
            "mps",
            MetalScheduling,
            PythonWrapperCodegen,
            CppWrapperMps,
            WrapperFxCodegen,
        )

    if get_scheduling_for_device("mtia") is None:
        register_backend_for_device(
            "mtia",
            TritonScheduling,
            PythonWrapperMtia,
            CppWrapperGpu,
            WrapperFxCodegen,
        )

    private_backend = torch._C._get_privateuse1_backend_name()
    if (
        private_backend != "privateuseone"
        and get_scheduling_for_device(private_backend) is None
    ):
        from torch.utils.backend_registration import _get_custom_mod_func

        try:
            device_scheduling = _get_custom_mod_func("Scheduling")
            wrapper_codegen = _get_custom_mod_func("PythonWrapperCodegen")
            cpp_wrapper_codegen = _get_custom_mod_func("CppWrapperCodegen")
            fx_wrapper_codegen = _get_custom_mod_func("WrapperFxCodegen")
            if device_scheduling and wrapper_codegen and cpp_wrapper_codegen:
                register_backend_for_device(
                    private_backend,
                    device_scheduling,
                    wrapper_codegen,
                    cpp_wrapper_codegen,
                    fx_wrapper_codegen,
                )
        except RuntimeError:
            pass


def index_prevent_reordering(
    index: Sequence[sympy.Expr],
    index_vars: Sequence[sympy.Expr],
    sizes: Sequence[sympy.Expr],
) -> list[sympy.Expr]:
    # [FUNCTION DOC] Returns the list of index expressions with one extra appended: the dot product of the index variables with the contiguous strides of sizes. The extra term prevents the indexing from being reordered.
    from ..ir import FlexibleLayout

    # added contiguous index prevents reordering
    return [*index, sympy_dot(index_vars, FlexibleLayout.contiguous_strides(sizes))]


def register_device_op_overrides(
    device: str, device_op_overrides: DeviceOpOverrides
) -> None:
    # [FUNCTION DOC] Stores the DeviceOpOverrides object for a device name in the global map while holding the lock.
    with device_op_overrides_lock:
        device_op_overrides_dict[device] = device_op_overrides


def _initialize_device_op_overrides():
    # [FUNCTION DOC] HELPER: Thread safe, run-once initializer of the device op overrides map. Uses a flag (not the map contents) with double
    # checked locking; imports the MPS, CPU, CUDA, MTIA and XPU override modules (import triggers their registration)
    # and registers no-op CPU style overrides for tpu.
    # Use a flag rather than checking device_op_overrides_dict, since external/test
    # code may partially populate it before we are called.
    global _device_op_overrides_initialized
    if _device_op_overrides_initialized:
        return

    with device_op_overrides_lock:
        if _device_op_overrides_initialized:
            return

        from . import mps_device_op_overrides  # noqa: F401
        from .cpu_device_op_overrides import CpuDeviceOpOverrides
        from .cuda import device_op_overrides  # noqa: F401
        from .mtia import device_op_overrides as mtia_op_overrides  # noqa: F401
        from .xpu import device_op_overrides as xpu_op_overrides  # noqa: F401

        # TPU uses Pallas for codegen and only needs no-op overrides
        register_device_op_overrides("tpu", CpuDeviceOpOverrides())

        _device_op_overrides_initialized = True


def get_device_op_overrides(device: str) -> DeviceOpOverrides:
    # [FUNCTION DOC] Returns the DeviceOpOverrides registered for a device name after making sure defaults are initialized. Raises AssertionError if device is not a string, KeyError if unregistered.
    if not isinstance(device, str):
        raise AssertionError(type(device))
    _initialize_device_op_overrides()
    return device_op_overrides_dict[device]


DTYPE_TO_COMPUTATION_DTYPE: dict[torch.dtype, torch.dtype] = {
    torch.bfloat16: torch.float,
    torch.float16: torch.float,
    **{
        dtype: dtype
        for dtype in [
            torch.bool,
            torch.float32,
            torch.float64,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        ]
    },
}


def deduce_output_dtype_by_name(
    op_name: str,
    *args: Any,
    **kwargs: Any,
) -> torch.dtype | None:
    # [FUNCTION DOC] Deduces the output dtype of an inductor op from its name and arguments. Returns bool for boolean ops, the dtype
    # argument (keyword or positional) for type conversion, index/value expr, reduction, constant and bitcast ops, float
    # for rand and randn, int64 for index, seed and 64-bit random ops, the graph recorded dtype of the buffer for load,
    # store and store_reduction, and None when unknown.
    """
    Given op name and a list of input dtypes, deduce the output dtype
    """
    if op_name in boolean_ops():
        return torch.bool
    elif op_name in (
        "to_dtype",
        "index_expr",
        "value_expr",
    ):
        return kwargs["dtype"] if "dtype" in kwargs else args[-1]
    elif op_name in (
        "rand",
        "randn",
    ):
        return torch.float
    elif op_name in (
        "get_index",
        "randint64",
        "load_seed",
    ):
        return torch.int64
    elif op_name == "reduction":
        return kwargs["dtype"] if "dtype" in kwargs else args[1]
    elif op_name == "constant":
        return kwargs["dtype"] if "dtype" in kwargs else args[-1]
    elif op_name in (
        "load",
        "store",
        "store_reduction",
    ):
        buf_name = args[1]
        return V.graph.get_dtype(buf_name)  # type: ignore[arg-type]
    elif op_name == "to_dtype_bitcast":
        return kwargs["dtype"] if "dtype" in kwargs else args[-2]
    return None


def _is_runtime_triton_assert_target() -> bool:
    # [FUNCTION DOC] HELPER: Returns true if the current backend is triton, the current device is not cpu and the active kernel has a triton tensor rank attribute, i.e. runtime Triton asserts can be emitted.
    return (
        get_current_backend() == "triton"
        and V.graph.get_current_device_or_throw().type != "cpu"
        and hasattr(V.kernel, "triton_tensor_ndim")
    )


def check_dtype(
    buffer: IndentedBuffer, var: CSEVariableType, dtype: torch.dtype
) -> None:
    # [FUNCTION DOC] Emits a dtype check into the buffer for a CSE variable. For Triton on a GPU with the runtime dtype assert test
    # config, writes a static assert on the variable dtype. For the C++ backend with the static dtype assert config, writes
    # a C++ static assert comparing the variable type to the expected type (handling vector, mask and bool cases).
    # Raises AssertionError if the variable is not a C++ CSE variable in the C++ case. Writes to the buffer.
    backend = get_current_backend()
    if (
        config.test_configs.runtime_triton_dtype_assert
        and _is_runtime_triton_assert_target()
    ):
        buffer.writeline(f"tl.static_assert({var}.dtype == {triton_type(dtype)})")
    elif config.test_configs.static_cpp_dtype_assert and backend == "cpp":
        from .cpp_utils import CppCSEVariable, DTYPE_TO_CPP

        if not isinstance(var, CppCSEVariable):
            raise AssertionError(type(var))
        if dtype == torch.bool:
            if var.is_vec:
                is_same_dt = f"IsVecMaskType<decltype({var})>::value"
            else:
                # operator&(bool, bool) returns int and it can be used as boolean in C++
                is_same_dt = f"std::is_same_v<decltype({var}), bool> || std::is_same_v<decltype({var}), int>"
        else:
            c_var_type = f"decltype({var})"
            if var.is_vec:
                c_var_type = f"typename {c_var_type}::value_type"
            is_same_dt = f"std::is_same_v<{c_var_type}, {DTYPE_TO_CPP[dtype]}>"

        buffer.writeline(f"static_assert({is_same_dt});")


def check_shape(
    buffer: IndentedBuffer, var: CSEVariableType, shape: BlockShapeType
) -> None:
    # [FUNCTION DOC] With the runtime Triton shape assert config enabled for a GPU Triton kernel, writes a static assert of the variable shape into the buffer. Raises AssertionError if shape is None.
    if shape is None:
        raise AssertionError("expected shape to be not None")
    if (
        config.test_configs.runtime_triton_shape_assert
        and _is_runtime_triton_assert_target()
    ):
        shape_str = (
            ", ".join(str(d) for d in shape) if len(shape) != 1 else f"{shape[0]},"
        )
        buffer.writeline(f"tl.static_assert({var}.shape == ({shape_str}))")


def check_nan(buffer: IndentedBuffer, var: CSEVariableType) -> None:
    # [FUNCTION DOC] For the Triton backend, writes a device assert into the buffer that the variable is neither NaN nor infinite; does nothing for other backends.
    backend = get_current_backend()
    if backend == "triton":
        msg = "NaN or Inf found"
        buffer.writeline(
            f"tl.device_assert(({var} == {var}) & ({var} != float('inf')) & ({var} != float('-inf')), '{msg}')"
        )


class DataTypePropagation:
    # [CLASS DOC] Propagates dtypes through the FX graphs of a LoopBody (root graph plus subblocks). Stores each node's dtype in
    # an OptimizationContext in node meta. Holds the mapping of graph key to FX graph.
    def __init__(self, body: LoopBody) -> None:
        # [FUNCTION DOC] Stores the loop body and builds the mapping from graph keys to graphs: the root block graph under root plus every subblock graph.
        self.body = body
        self.graphs: dict[Callable[..., Any] | str, Any] = {
            "root": body.root_block.graph
        }
        for k, v in body.subblocks.items():
            self.graphs[k] = v.graph

    def deduce_node_dtype_by_inputs(self, node: torch.fx.Node) -> torch.dtype | None:
        # [FUNCTION DOC] Returns the promoted dtype of the node's non-placeholder input nodes, or None when there are no such inputs or
        # any of them has no propagated dtype yet.
        inputs = node.all_input_nodes
        input_nodes = [
            n for n in inputs if isinstance(n, torch.fx.Node) and n.op != "placeholder"
        ]
        if len(input_nodes) == 0:
            return None

        all_input_nodes_propagated = all(
            OptimizationContext.key in n.meta
            and n.meta[OptimizationContext.key].dtype is not None
            for n in input_nodes
        )
        if not all_input_nodes_propagated:
            return None

        return functools.reduce(
            torch.promote_types,
            [n.meta[OptimizationContext.key].dtype for n in input_nodes],
        )

    def deduce_node_dtype_by_subgraph(self, node: torch.fx.Node) -> torch.dtype:
        # [FUNCTION DOC] Looks up the node's target subgraph, propagates dtypes through it and returns its dtype. Raises AssertionError if no dtype results.
        sub_graph = self.graphs[node.target]
        dtype = self.propagate_graph(sub_graph)
        if not dtype:
            raise AssertionError("expected subgraph to propagate a dtype")
        return dtype

    def deduce_node_dtype(self, node: torch.fx.Node) -> torch.dtype | None:
        # [FUNCTION DOC] Returns the dtype of an FX node or None. Placeholders and multi-argument outputs give None; getitem uses the
        # dtype of its source node; masked subblock calls use the subgraph dtype; otherwise tries the name based deduction and
        # falls back to input promotion. Raises AssertionError for unexpected node target types.
        if node.op == "placeholder":
            return None

        if node.target == "output" and len(node.args) != 1:
            # we can infer output node if it only have 1 arg
            return None

        if node.target is operator.getitem:
            node_arg = node.args[0]
            if not isinstance(node_arg, torch.fx.Node):
                raise AssertionError(type(node_arg))
            return self.deduce_node_dtype(node_arg)

        if not isinstance(node.target, str):
            raise AssertionError(type(node.target))

        if node.target.startswith("masked_subblock"):
            return self.deduce_node_dtype_by_subgraph(node)

        if (
            output_dtype := deduce_output_dtype_by_name(
                node.target,
                *node.args,
                **node.kwargs,
            )
        ) is not None:
            return output_dtype

        return self.deduce_node_dtype_by_inputs(node)

    def propagate_graph(self, graph: torch.fx.Graph) -> torch.dtype | None:
        # [FUNCTION DOC] Walks the graph nodes in order, computing each node dtype and storing it in the node's OptimizationContext (created
        # if missing). Returns the dtype of the output node (used as the dtype of masked subblocks), possibly None.
        # Raises AssertionError for an empty graph. Mutates node meta.
        if not graph.nodes:
            raise AssertionError("expected graph to have nodes")
        graph_dtype: torch.dtype | None = None
        # For masked_subblock, we use output's dtype to represent
        # the dtype of this subgraph. For other cases, graph_dtype
        # might be None
        for node in graph.nodes:
            if OptimizationContext.key in node.meta:
                opt_ctx = node.meta[OptimizationContext.key]
            else:
                opt_ctx = OptimizationContext()

            opt_ctx.dtype = self.deduce_node_dtype(node)
            node.meta[OptimizationContext.key] = opt_ctx
            if node.target == "output":
                graph_dtype = opt_ctx.dtype
        return graph_dtype

    def propagate(self) -> torch.dtype | None:
        # [FUNCTION DOC] Runs dtype propagation over the root graph and returns its output dtype.
        return self.propagate_graph(self.graphs["root"])

    @classmethod
    def propagate_loopbody(cls, body: LoopBody) -> torch.dtype | None:
        # [FUNCTION DOC] Class method: builds a propagator for the loop body, runs it and returns the root output dtype.
        return cls(body).propagate()

    @classmethod
    def propagate_scheduler_node(cls, node: SchedulerNode) -> torch.dtype | None:
        # [FUNCTION DOC] Class method: propagates dtypes over a SchedulerNode's loop body and returns the result. Raises AssertionError if the node is not a SchedulerNode or its body is not a LoopBody.
        from ..loop_body import LoopBody
        from ..scheduler import SchedulerNode

        if not isinstance(node, SchedulerNode):
            raise AssertionError(type(node))
        if not isinstance(node._body, LoopBody):
            raise AssertionError(type(node._body))
        return DataTypePropagation.propagate_loopbody(node._body)


class PythonPrinter(_PythonPrinter):
    # [CLASS DOC] Sympy to python source printer that simplifies expressions with the graph sizevars before printing and parenthesizes Mod terms to keep correct precedence.
    def doprint(
        self, expr: sympy.Expr, *, simplify: bool = True, p: bool = True
    ) -> str:
        # [FUNCTION DOC] Prints a sympy expression as python source. If simplify is set and the expression is a sympy expression and the graph has sizevars, it is simplified first. Returns the string.
        # TODO: why are people passing strings to the printer here :think:
        if simplify and isinstance(expr, sympy.Expr) and hasattr(V.graph, "sizevars"):
            expr = V.graph.sizevars.simplify(expr)
        return super().doprint(expr)

    def parenthesize(self, item: sympy.Expr, level: int, strict: bool = False) -> str:
        # [FUNCTION DOC] Wraps Mod expressions in parentheses to enforce precedence (works around wrong output for negatives times Mod in one sympy version); other items use the base behavior.
        if isinstance(item, sympy.Mod):
            # use parenthesis to enforce precedence.
            # in sympy 1.13.3, -2*Mod(x,y) becomes -2*x%y, which is wrong.
            # pyrefly: ignore [missing-attribute]
            return f"({self._print(item)})"
        else:
            return super().parenthesize(item, level, strict)


class OpDecompositions:
    # [CLASS DOC] Collection of static methods that decompose some inductor ops into other ops (via the ops dispatcher): reciprocal, square, erfc, expm1, log10, log2, exp2, log1p, sigmoid, relu, fma, rounding to int, remainder, etc.
    """
    Decomposes inductor ops
    """

    @staticmethod
    def identity(value: OpVarT) -> OpVarT:
        # [FUNCTION DOC] HELPER: Returns the value unchanged; used to trigger common subexpression elimination.
        # used to trigger cse
        return value

    @staticmethod
    def reciprocal(x: OpVarT) -> OpVarT:
        # [FUNCTION DOC] Decomposes reciprocal as float32 constant one divided by x.
        # Use float32 constant so that div_rn can be applied when
        # eager_numerics.division_rounding is enabled
        return ops.truediv(ops.constant(1.0, torch.float32), x)

    @staticmethod
    def square(x: OpVarT) -> OpVarT:
        # [FUNCTION DOC] Decomposes square as x multiplied by x.
        return ops.mul(x, x)

    @staticmethod
    def erfc(x: OpVarT) -> OpVarT:
        # [FUNCTION DOC] Decomposes erfc as one minus erf of x.
        return ops.sub(ops.constant(1, torch.float32), ops.erf(x))

    @staticmethod
    def erfcx(x: OpVarT) -> OpVarT:
        # [FUNCTION DOC] Decomposes erfcx as exp of the square of x times erfc of x.
        return ops.mul(ops.exp(ops.square(x)), ops.erfc(x))

    @staticmethod
    def expm1(x: OpVarT) -> OpVarT:
        # [FUNCTION DOC] Decomposes expm1 as exp of x minus one.
        return ops.sub(ops.exp(x), ops.constant(1, torch.float32))

    @staticmethod
    def log10(x: OpVarT) -> OpVarT:
        # [FUNCTION DOC] Decomposes log10 as natural log of x times one over ln 10.
        return ops.mul(ops.log(x), ops.constant(1 / math.log(10), torch.float32))

    @staticmethod
    def log2(x: OpVarT) -> OpVarT:
        # [FUNCTION DOC] Decomposes log2 as natural log of x times one over ln 2.
        return ops.mul(ops.log(x), ops.constant(1 / math.log(2), torch.float32))

    @staticmethod
    def exp2(x: OpVarT) -> OpVarT:
        # [FUNCTION DOC] Decomposes exp2 as exp of x times ln 2.
        return ops.exp(ops.mul(x, ops.constant(math.log(2), torch.float32)))

    @staticmethod
    def log1p(x: OpVarT) -> OpVarT:
        # [FUNCTION DOC] Decomposes log1p as log of x plus one.
        return ops.log(ops.add(x, ops.constant(1, torch.int32)))

    @staticmethod
    def sigmoid(x: OpVarT) -> OpVarT:
        # [FUNCTION DOC] Decomposes sigmoid as one over one plus exp of negative x.
        one = ops.constant(1, torch.int32)
        return ops.truediv(one, ops.add(one, ops.exp(ops.neg(x))))

    @staticmethod
    def relu(x: OpVarT) -> OpVarT:
        # [FUNCTION DOC] Decomposes relu as the maximum of x and zero.
        return ops.maximum(x, ops.constant(0, torch.int32))

    @staticmethod
    def fma(x: OpVarT, y: OpVarT, z: OpVarT) -> OpVarT:
        # [FUNCTION DOC] Decomposes fused multiply add as x times y plus z; fallback for backends that do not override it (Halide).
        # for backends that don't override this (halide)
        return ops.add(ops.mul(x, y), z)

    @staticmethod
    def mul_rn(x: OpVarT, y: OpVarT) -> OpVarT:
        # [FUNCTION DOC] Decomposes multiply with round-to-nearest as a plain multiply; fallback when a backend has no special version.
        # for backends that don't override this, just use regular mul
        return ops.mul(x, y)

    @staticmethod
    def div_rn(x: OpVarT, y: OpVarT) -> OpVarT:
        # [FUNCTION DOC] Decomposes divide with round-to-nearest as a plain true division; fallback when a backend has no special version.
        # for backends that don't override this, just use regular div
        return ops.truediv(x, y)

    @staticmethod
    def floor_to_int(a: OpVarT, dtype: torch.dtype) -> OpVarT:
        # [FUNCTION DOC] Decomposes floor to int as floor then conversion to the given dtype.
        return ops.to_dtype(ops.floor(a), dtype)

    @staticmethod
    def ceil_to_int(a: OpVarT, dtype: torch.dtype) -> OpVarT:
        # [FUNCTION DOC] Decomposes ceil to int as ceil then conversion to the given dtype.
        return ops.to_dtype(ops.ceil(a), dtype)

    @staticmethod
    def trunc_to_int(a: OpVarT, dtype: torch.dtype) -> OpVarT:
        # [FUNCTION DOC] Decomposes trunc to int as trunc then conversion to the given dtype.
        return ops.to_dtype(ops.trunc(a), dtype)

    @staticmethod
    def remainder(a: OpVarT, b: OpVarT) -> OpVarT:
        # [FUNCTION DOC] Decomposes remainder with the sign of the divisor: computes mod, and where the result is nonzero and its sign bit
        # differs from the divisor's sign bit, adds the divisor back.
        r = ops.mod(a, b)
        cond = ops.and_(
            ops.ne(r, ops.constant(0, torch.int32)),
            ops.ne(ops.signbit(r), ops.signbit(b)),
        )
        return ops.where(cond, ops.add(r, b), r)

    @staticmethod
    def round_to_int(a: OpVarT, dtype: torch.dtype) -> OpVarT:
        # [FUNCTION DOC] Decomposes round to int as round then conversion to the given dtype.
        return ops.to_dtype(ops.round(a), dtype)


_RE_PAREN_NOT_NEEDED = re.compile(r"[a-z0-9_.]+|\([^)]*\)|", flags=re.IGNORECASE)


def _all_in_parens(string: str) -> bool:
    # [FUNCTION DOC] HELPER: Returns true if the string is entirely enclosed by one matching pair of outer parentheses (the opening paren at the start
    # pairs with the final character). Raises AssertionError if parentheses are unbalanced.
    # KNOWN ISSUE: indexes the first character, so an empty string would raise an IndexError.
    if string[0] != "(" or len(string) < 2:
        return False
    count = 1
    for i, char in enumerate(string[1:]):
        if char == "(":
            count += 1
        elif char == ")":
            count -= 1
        if count == 0 and i != len(string) - 2:
            return False
    if count != 0:
        raise AssertionError(f"expected balanced count == 0, got {count}")
    return True


# pyrefly: ignore [inconsistent-inheritance]
class OpOverrides(BasicMathOpsMixin, OpDecompositions, OpsHandler[Any]):
    # [CLASS DOC] Default string-producing ops handler for backends: inherits math mixin and decompositions and the ops handler
    # interface. Provides code text for bitwise and logical ops, and stubs that raise for ops that must be handled by
    # CSEProxy (load, store, reduction, scan, sort, bucketize, ...) or only by specific backends. Pointwise overrides are
    # attached to subclasses dynamically by _initialize_pointwise_overrides.
    @staticmethod
    def paren(string: OpVarT) -> OpVarT:
        # [FUNCTION DOC] Returns the string unchanged if it is a CSEVariable, a simple token or already fully parenthesized; otherwise returns it wrapped in parentheses.
        if (
            isinstance(string, CSEVariable)
            or _RE_PAREN_NOT_NEEDED.fullmatch(string)
            or _all_in_parens(string)
        ):
            # don't put extra parens for strings that are already wrapped in parens

            return string
        return f"({string})"

    @staticmethod
    def constant(value: bool | float | int, dtype: torch.dtype) -> OpVarT:
        # [FUNCTION DOC] Returns the textual representation of the constant value; the dtype is ignored.
        return repr(value)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def bitwise_not(x: OpVarT) -> OpVarT:
        # [FUNCTION DOC] Returns code text for bitwise not of the parenthesized operand.
        return f"~{OpOverrides.paren(x)}"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def logical_not(a: OpVarT) -> OpVarT:
        # [FUNCTION DOC] Returns code text testing that the parenthesized operand equals zero.
        return f"{OpOverrides.paren(a)} == 0"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def bitwise_and(x: OpVarT, y: OpVarT) -> OpVarT:
        # [FUNCTION DOC] Returns code text for bitwise and of the two parenthesized operands.
        return f"{OpOverrides.paren(x)} & {OpOverrides.paren(y)}"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def bitwise_or(x: OpVarT, y: OpVarT) -> OpVarT:
        # [FUNCTION DOC] Returns code text for bitwise or of the two parenthesized operands.
        return f"{OpOverrides.paren(x)} | {OpOverrides.paren(y)}"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def bitwise_xor(x: OpVarT, y: OpVarT) -> OpVarT:
        # [FUNCTION DOC] Returns code text for bitwise exclusive or of the two parenthesized operands.
        return f"{OpOverrides.paren(x)} ^ {OpOverrides.paren(y)}"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def bitwise_left_shift(x: OpVarT, y: OpVarT) -> OpVarT:
        # [FUNCTION DOC] Returns code text for left shift of x by y with parenthesized operands.
        return f"{OpOverrides.paren(x)} << {OpOverrides.paren(y)}"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def bitwise_right_shift(x: OpVarT, y: OpVarT) -> OpVarT:
        # [FUNCTION DOC] Returns code text for right shift of x by y with parenthesized operands.
        return f"{OpOverrides.paren(x)} >> {OpOverrides.paren(y)}"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def int_truediv(a: OpVarT, b: OpVarT) -> OpVarT:
        # [FUNCTION DOC] Implements integer true division by delegating to true division of the ops dispatcher.
        # KNOWN ISSUE: marked as wrong in the source; it is only exact for values up to 2 to the 53rd power.
        # TODO: this is wrong
        # TODO: an easy bandaid is to generate runtime asserts that it's
        # <= 2**53, which is when this equation is correct
        return ops.truediv(a, b)

    @staticmethod
    def load_seed(name: str, offset: OpVarT) -> OpVarT:
        # [FUNCTION DOC] Loads the seed from buffer name at the integer offset via the ops dispatcher and returns the loaded variable.
        return ops.load(name, sympy.Integer(offset))

    def indirect_indexing(
        self,
        var: OpVarT,
        size: sympy.Expr | int,
        check: bool = True,
        wrap_neg: bool = True,
    ) -> sympy.Symbol:
        # [FUNCTION DOC] Returns a sympy index symbol named after the variable; check and wrap_neg are ignored here (handled by CSEProxy).
        return sympy_index_symbol(str(var))

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> None:
        # [FUNCTION DOC] Raises NotImplementedError: bounds checks are handled by CSEProxy.
        raise NotImplementedError(
            f"{type(self).__name__}: check_bounds should be handled by CSEProxy"
        )

    def load(self, name: str, index: sympy.Expr) -> OpVarT:
        # [FUNCTION DOC] Raises NotImplementedError: loads are handled by CSEProxy.
        raise NotImplementedError(
            f"{type(self).__name__}: load should be handled by CSEProxy"
        )

    def store(
        self, name: str, index: sympy.Expr, value: OpVarT, mode: StoreMode = None
    ) -> None:
        # [FUNCTION DOC] Raises NotImplementedError: stores are handled by CSEProxy.
        raise NotImplementedError(
            f"{type(self).__name__}: store should be handled by CSEProxy"
        )

    def device_assert_async(self, cond: CSEVariable, msg: str) -> None:
        # [FUNCTION DOC] Raises NotImplementedError: asynchronous device asserts are handled by CSEProxy.
        raise NotImplementedError(
            f"{type(self).__name__}: device_assert_async should be handled by CSEProxy"
        )

    def store_reduction(self, name: str, index: sympy.Expr, value: OpVarT) -> None:
        # [FUNCTION DOC] Raises NotImplementedError: store reductions are handled by CSEProxy.
        raise NotImplementedError(
            f"{type(self).__name__}: store_reduction should be handled by CSEProxy"
        )

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: OpVarT | tuple[OpVarT, ...],
    ) -> OpVarT | tuple[OpVarT, ...]:
        # [FUNCTION DOC] Raises NotImplementedError: reductions are handled by CSEProxy.
        raise NotImplementedError(
            f"{type(self).__name__}: reduction should be handled by CSEProxy"
        )

    def scan(
        self,
        dtypes: tuple[torch.dtype, ...],
        combine_fn: Callable[
            [tuple[OpVarT, ...], tuple[OpVarT, ...]],
            tuple[OpVarT, ...],
        ],
        values: tuple[OpVarT, ...],
    ) -> tuple[OpVarT, ...]:
        # [FUNCTION DOC] Raises NotImplementedError: scans are handled by CSEProxy.
        raise NotImplementedError(
            f"{type(self).__name__}: scan should be handled by CSEProxy"
        )

    def sort(
        self,
        dtypes: tuple[torch.dtype, ...],
        values: tuple[OpVarT, ...],
        stable: bool,
        descending: bool,
    ) -> tuple[OpVarT, ...]:
        # [FUNCTION DOC] Raises NotImplementedError: sorts are handled by CSEProxy.
        raise NotImplementedError(
            f"{type(self).__name__}: sort should be handled by CSEProxy"
        )

    def bucketize(
        self,
        values: OpVarT,
        boundaries: tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: OpVarT,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: tuple[str, sympy.Expr] | None = None,
        sorter_indices: OpVarT | None = None,
    ) -> OpVarT:
        # [FUNCTION DOC] Raises NotImplementedError: bucketize is handled by CSEProxy.
        raise NotImplementedError(
            f"{type(self).__name__}: bucketize should be handled by CSEProxy"
        )

    def halide_clamp(self, value: OpVarT, size: sympy.Expr, check: bool) -> OpVarT:
        # [FUNCTION DOC] Raises NotImplementedError: clamp is implemented only for the Halide backend.
        raise NotImplementedError(
            f"{type(self).__name__}: halide_clamp only implemented for Halide backend"
        )

    def dot(self, x: OpVarT, y: OpVarT) -> OpVarT:
        # [FUNCTION DOC] Raises NotImplementedError: dot is implemented only for the Triton backend.
        raise NotImplementedError(
            f"{type(self).__name__}: dot only implemented for Triton backend"
        )

    def inline_asm_elementwise(
        self,
        *inputs: OpVarT,
        asm: str,
        constraints: str | None = None,
        dtype: torch.dtype = torch.float32,
        is_pure: bool = True,
        pack: int = 1,
        input_dtypes: tuple[torch.dtype, ...] | None = None,
    ) -> OpVarT:
        # [FUNCTION DOC] Raises NotImplementedError: inline assembly elementwise is implemented only for the Triton backend.
        raise NotImplementedError(
            f"{type(self).__name__}: inline_asm_elementwise only implemented for Triton backend"
        )

    def output(self, *args: OpVarT) -> None:
        # [FUNCTION DOC] Raises AssertionError: the output op must not appear at codegen time.
        raise AssertionError(
            f"{type(self).__name__}: ops.output should not appear at codegen time"
        )

    def placeholder(self, index: int) -> OpVarT:
        # [FUNCTION DOC] Raises AssertionError: the placeholder op must not appear at codegen time.
        raise AssertionError(
            f"{type(self).__name__}: ops.placeholder should not appear at codegen time"
        )

    @staticmethod
    def _unimplemented(name: str) -> Callable[..., OpVarT]:
        # [FUNCTION DOC] HELPER: Builds and returns a function standing in for an unsupported op name; calling it raises NotImplementedError naming the class and op. The function is named after the op and flagged as unimplemented.
        def unimplemented(self: OpOverrides, *args: Any, **kwargs: Any) -> OpVarT:
            # [FUNCTION DOC] HELPER: Raises NotImplementedError saying the handler class does not implement the op.
            raise NotImplementedError(
                f"{type(self).__name__} does not implement ops.{name}"
            )

        unimplemented.__name__ = name
        unimplemented.is_unimplemented = True  # type: ignore[attr-defined]
        return unimplemented

    @classmethod
    def _is_unimplemented(cls, name: str) -> bool:
        # [FUNCTION DOC] HELPER: Class method: returns true when the named op is missing, equals the default from the base ops handler, or carries the unimplemented marker.
        fn = getattr(cls, name, None)
        default_fn = getattr(OpsHandler, name, None)
        return not fn or fn == default_fn or getattr(fn, "is_unimplemented", False)

    @classmethod
    def _initialize_pointwise_overrides(cls, target: str) -> None:
        # [FUNCTION DOC] HELPER: Class method: installs pointwise op implementations for a backend target (triton, cpp, cppvec, halide or mps) as
        # static methods from pointwise_overrides_data. Ops with no implementation for the target get an unimplemented stub if
        # not already implemented. Raises AssertionError for an unknown target or when the class already defines an op that is
        # also being installed. Mutates the class; sets the implementation function's name.
        if target not in ("triton", "cpp", "cppvec", "halide", "mps"):
            raise AssertionError(target)

        for funcname, data in pointwise_overrides_data.items():
            impl = getattr(data, target)
            if impl is None:
                if cls._is_unimplemented(funcname):
                    setattr(cls, funcname, cls._unimplemented(funcname))
            else:
                if funcname in cls.__dict__:
                    raise AssertionError(
                        f"multiple definitions of {funcname} on {cls.__name__}"
                    )
                impl.__name__ = funcname
                setattr(cls, funcname, staticmethod(impl))


@dataclasses.dataclass
class OverridesData:
    # [CLASS DOC] Dataclass describing one pointwise op override: name, C++ implementation callable (required), optional Triton, C++
    # vectorized, Halide and MPS callables (None means not implemented), and the elementwise type promotion kind.
    name: str
    cpp: Callable[..., str]
    # None when not impl in libdevice/triton
    triton: Callable[..., str] | None = None
    # None when not impl in aten/.../vec
    cppvec: Callable[..., str] | None = None
    type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND = (
        ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    halide: Callable[..., str] | None = None
    mps: Callable[..., str] | None = None


def _triton_bessel(order: int, kind: str, x: str) -> str:
    # [FUNCTION DOC] HELPER: Returns Triton code text for a Bessel function of the given order and kind (j or y) from libdevice, yielding NaN in the input dtype for infinite inputs to match eager.
    # PyTorch eager returns NaN for +/-inf, while libdevice returns
    # the mathematical limit. x - x yields NaN in x's dtype
    # (avoids fp32 literal promotion under tl.where for float64 inputs)
    return (
        f"tl.where(tl_math.abs({x}) == float('inf'), "
        f"{x} - {x}, "
        f"libdevice.{kind}{order}({x}))"
    )


def _triton_cyl_bessel_i(order: int, x: str) -> str:
    # [FUNCTION DOC] HELPER: Returns Triton code text for the modified cylindrical Bessel i function of the given order, yielding NaN in the input dtype for infinite inputs to match eager.
    # PyTorch's Cephes-derived kernels return NaN for infinities; libdevice
    # returns signed infinities, so synthesize a same-dtype NaN with x - x.
    return (
        f"tl.where(tl.abs({x}) == float('inf'), "
        f"{x} - {x}, "
        f"libdevice.cyl_bessel_i{order}({x}))"
    )


# NB: if you add a new special function, don't forget to update
# torch._inductor.ops_handler too
pointwise_overrides_data: dict[str, OverridesData] = dict(
    airy_ai=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"airy_ai_forward({x})",
        name="special_airy_ai",
    ),
    bessel_j0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"bessel_j0_forward({x})",
        triton=lambda x: _triton_bessel(0, "j", x),
        name="special_bessel_j0",
    ),
    bessel_j1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"bessel_j1_forward({x})",
        triton=lambda x: _triton_bessel(1, "j", x),
        name="special_bessel_j1",
    ),
    bessel_y0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"bessel_y0_forward({x})",
        triton=lambda x: _triton_bessel(0, "y", x),
        name="special_bessel_y0",
    ),
    bessel_y1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"bessel_y1_forward({x})",
        triton=lambda x: _triton_bessel(1, "y", x),
        name="special_bessel_y1",
    ),
    digamma=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_digamma({x})",
        cppvec=lambda x: f"{x}.digamma()",
        name="digamma",
    ),
    # no cpp nor triton implementation for entr, it is defined as decomposition
    # erf, erfc
    erfcx=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_erfcx({x})",
        triton=lambda x: f"libdevice.erfcx({x})",
        name="special_erfcx",
    ),
    fma=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y, z: f"std::fma({x}, {y}, {z})",
        cppvec=lambda x, y, z: f"fmadd({x}, {y}, {z})",
        # NOTE: We use tl.fma instead of libdevice.fma because libdevice.fma
        # has Flush-To-Zero (FTZ) behavior for subnormal values, which causes
        # numerical differences compared to eager CUDA execution. tl.fma
        # preserves subnormals and matches eager's FMA precision.
        triton=lambda x, y, z: f"tl.fma({x}, {y}, {z})",
        name="fma",
    ),
    # mul_rn: Multiplication with round-to-nearest. This prevents Triton's
    # compiler from fusing the multiplication with subsequent operations,
    # which is needed to match eager's rounding behavior in operations like
    # addcmul where the product must be rounded before use.
    # Note: libdevice.mul_rn is not supported on ROCm, so we fall back to
    # regular multiplication there.
    mul_rn=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
        cpp=lambda x, y: f"({x}) * ({y})",  # C++ doesn't need special handling
        triton=lambda x, y: f"({x}) * ({y})"
        if torch.version.hip
        else f"libdevice.mul_rn({x}, {y})",
        name="mul_rn",
    ),
    # erfinv, exp2, expit, gammaln
    igamma=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"calc_igamma({x}, {y})",
        name="igamma",
    ),
    igammac=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"calc_igammac({x}, {y})",
        name="igammac",
    ),
    gammainc=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"calc_igamma({x}, {y})",
        name="special_gammainc",
    ),
    gammaincc=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"calc_igammac({x}, {y})",
        name="special_gammaincc",
    ),
    i0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_i0({x})",
        triton=lambda x: _triton_cyl_bessel_i(0, x),
        cppvec=lambda x: f"{x}.i0()",
        name="i0",
    ),
    i0e=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_i0e({x})",
        cppvec=lambda x: f"{x}.i0e()",
        name="special_i0e",
    ),
    i1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_i1({x})",
        triton=lambda x: _triton_cyl_bessel_i(1, x),
        name="special_i1",
    ),
    i1e=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_i1e({x})",
        name="special_i1e",
    ),
    log_ndtr=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_log_ndtr({x})",
        name="special_log_ndtr",
    ),
    # logit
    modified_bessel_i0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"modified_bessel_i0_forward({x})",
        triton=lambda x: _triton_cyl_bessel_i(0, x),
        name="special_modified_bessel_i0",
    ),
    modified_bessel_i1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"modified_bessel_i1_forward({x})",
        triton=lambda x: _triton_cyl_bessel_i(1, x),
        name="special_modified_bessel_i1",
    ),
    modified_bessel_k0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"modified_bessel_k0_forward({x})",
        name="special_modified_bessel_k0",
    ),
    modified_bessel_k1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"modified_bessel_k1_forward({x})",
        name="special_modified_bessel_k1",
    ),
    # multigamma
    ndtr=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_ndtr({x})",
        name="special_ndtr",
    ),
    ndtri=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_ndtri({x})",
        name="special_ndtri",
    ),
    polygamma=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x,
        y: f"{x} == 0 ? calc_digamma({y}) : ({x} == 1 ? trigamma({y}) : calc_polygamma({y}, {x}))",
        name="polygamma",
    ),
    # psi - alias to digamma
    # round
    scaled_modified_bessel_k0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"scaled_modified_bessel_k0_forward({x})",
        name="special_scaled_modified_bessel_k0",
    ),
    scaled_modified_bessel_k1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"scaled_modified_bessel_k1_forward({x})",
        name="special_scaled_modified_bessel_k1",
    ),
    # sinc
    spherical_bessel_j0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"spherical_bessel_j0_forward({x})",
        name="special_spherical_bessel_j0",
    ),
    zeta=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"zeta({x}, {y})",
        name="special_zeta",
    ),
    chebyshev_polynomial_t=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"chebyshev_polynomial_t_forward({x}, {y})",
        name="special_chebyshev_polynomial_t",
    ),
    chebyshev_polynomial_u=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"chebyshev_polynomial_u_forward({x}, {y})",
        name="special_chebyshev_polynomial_u",
    ),
    chebyshev_polynomial_v=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"chebyshev_polynomial_v_forward({x}, {y})",
        name="special_chebyshev_polynomial_v",
    ),
    chebyshev_polynomial_w=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"chebyshev_polynomial_w_forward({x}, {y})",
        name="special_chebyshev_polynomial_w",
    ),
    legendre_polynomial_p=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"legendre_polynomial_p_forward({x}, {y})",
        name="special_legendre_polynomial_p",
    ),
    shifted_chebyshev_polynomial_t=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"shifted_chebyshev_polynomial_t_forward({x}, {y})",
        name="special_shifted_chebyshev_polynomial_t",
    ),
    shifted_chebyshev_polynomial_u=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"shifted_chebyshev_polynomial_u_forward({x}, {y})",
        name="special_shifted_chebyshev_polynomial_u",
    ),
    shifted_chebyshev_polynomial_v=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"shifted_chebyshev_polynomial_v_forward({x}, {y})",
        name="special_shifted_chebyshev_polynomial_v",
    ),
    shifted_chebyshev_polynomial_w=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"shifted_chebyshev_polynomial_w_forward({x}, {y})",
        name="special_shifted_chebyshev_polynomial_w",
    ),
    hermite_polynomial_h=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"hermite_polynomial_h_forward({x}, {y})",
        name="special_hermite_polynomial_h",
    ),
    hermite_polynomial_he=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"hermite_polynomial_he_forward({x}, {y})",
        name="special_hermite_polynomial_he",
    ),
    laguerre_polynomial_l=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"laguerre_polynomial_l_forward({x}, {y})",
        name="special_laguerre_polynomial_l",
    ),
)


def is_buffer_removed(name: str) -> bool:
    # [FUNCTION DOC] Returns true if the buffer name is in any removed or in-place-to-remove set of the current graph or the current kernel.
    return any(
        name in x
        for x in (
            V.graph.removed_buffers,
            V.kernel.removed_buffers,
            V.graph.inplaced_to_remove,
            V.kernel.inplaced_to_remove,
        )
    )


class DeferredLine(DeferredLineBase):
    # [CLASS DOC] A deferred code line that is dropped (evaluates to None) if its associated buffer name has been removed. Holds the buffer name and the line text.
    """A line that can be 'unwritten' by adding name to V.graph.removed_buffers"""

    def __init__(self, name: str, line: str):
        # [FUNCTION DOC] Stores the buffer name and line text. Raises AssertionError if the line is itself a deferred line.
        super().__init__(line)
        self.name = name
        if isinstance(line, DeferredLineBase):
            raise AssertionError("line must not be a DeferredLineBase")

    def __call__(self) -> str | None:
        # [FUNCTION DOC] Returns the line text unless the buffer has been removed, in which case returns None.
        if not is_buffer_removed(self.name):
            return self.line
        return None

    def _new_line(self, line: str) -> DeferredLine:
        # [FUNCTION DOC] HELPER: Builds a new DeferredLine with the same buffer name and the given line text.
        return DeferredLine(self.name, line)


class BracesBuffer(IndentedBuffer):
    # [CLASS DOC] Indented buffer that emits braces: entering an indent context writes an opening brace and increases indentation; leaving writes a closing brace and decreases it (the reverse for negative offsets).
    def indent(self, offset: int = 1) -> contextlib.AbstractContextManager[None]:
        # [FUNCTION DOC] Returns a context manager that, for a positive offset, writes opening braces and indents on entry and dedents with closing braces on exit; negative offsets do the opposite. Mutates indentation and buffer contents.
        @contextlib.contextmanager
        def ctx() -> Iterator[None]:
            # [FUNCTION DOC] HELPER: Generator based context manager implementing the brace and indent logic described by the enclosing indent method.
            for _ in range(offset):
                self.writeline("{")
                self._indent += 1
            for _ in range(-offset):
                self._indent -= 1
                self.writeline("}")
            yield
            for _ in range(-offset):
                self.writeline("{")
                self._indent += 1
            for _ in range(offset):
                self._indent -= 1
                self.writeline("}")

        return ctx()


class InplacedBuffer(NamedTuple):
    # [CLASS DOC] Named tuple for an in-place kernel buffer: the inner kernel argument name and the list of outer buffer names that share it.
    inner_name: str
    other_names: list[str]


@dataclasses.dataclass
class ArgName:
    # [CLASS DOC] Dataclass for a kernel argument name with a flag for whether it should be declared as a Triton constexpr.
    name: str
    # is_constexpr=True is used to attach a " : tl.constexpr" into the argument list
    is_constexpr: bool = False

    def full_name(self) -> str:
        # [FUNCTION DOC] Returns the name, followed by a constexpr annotation if the flag is set.
        return f"{self.name}{' : tl.constexpr' if self.is_constexpr else ''}"


class RemovedArg:
    # [CLASS DOC] Sentinel type marking a kernel argument that has been removed; its string form is REMOVED.
    def __str__(self) -> str:
        # [FUNCTION DOC] Returns the text REMOVED.
        return "REMOVED"


REMOVED = RemovedArg()


class KernelArgs:
    # [CLASS DOC] Tracks the arguments of a generated kernel. State: input_buffers (outer name to inner arg name), output_buffers
    # (outer name to inner name or removed marker), inplace_buffers (outer name to InplacedBuffer or removed marker),
    # sizevars (sympy expr to arg name) and workspace_args. Provides name allocation, in-place merging, removal and
    # argument definition generation for C++ and Python/Triton.
    @staticmethod
    def _lookup(
        prefix: str,
        odict: dict[_T, str | RemovedArg] | dict[_T, str],
        name: _T,
    ) -> str:
        # [FUNCTION DOC] HELPER: Returns the name stored for the key in the dict; if missing or marked removed, allocates a new name from the prefix and the current dict length, stores and returns it. Mutates the dict.
        result: str | RemovedArg = odict.get(name, REMOVED)
        if isinstance(result, RemovedArg):
            odict[name] = new_result = f"{prefix}{len(odict)}"
            return new_result
        return result

    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes empty input, output, in-place and size variable maps and an empty workspace argument list.
        self.input_buffers: dict[str, str] = {}
        self.output_buffers: dict[str, str | RemovedArg] = {}
        self.inplace_buffers: dict[str, InplacedBuffer | RemovedArg] = {}
        self.sizevars: dict[sympy.Expr, str] = {}
        self.workspace_args: list[WorkspaceArg] = []

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns a string showing the input, output, in-place and size maps.
        return "KernelArgs({})".format(
            ", ".join(
                map(
                    repr,
                    [
                        self.input_buffers,
                        self.output_buffers,
                        self.inplace_buffers,
                        self.sizevars,
                    ],
                )
            )
        )

    @staticmethod
    def _buffer_is_marked_removed(name: Any) -> bool:
        # [FUNCTION DOC] HELPER: Returns true if the given value is the removed marker (needed by MTIA).
        # this function is needed by MTIA
        return isinstance(name, RemovedArg)

    def input(self, name: str) -> str:
        # [FUNCTION DOC] Returns the kernel argument name to use when reading a buffer. Resolves mutation aliases through the scheduler if
        # present; reuses the output or in-place name if the buffer is already one; otherwise allocates a seed or in_ptr name.
        # Raises AssertionError if the buffer was removed. May mutate input_buffers.
        if V.graph.scheduler:
            name = V.graph.scheduler.mutation_real_name.get(name, name)
        if name in V.graph.removed_buffers:
            raise AssertionError(name)
        if name in self.output_buffers:
            return cast(str, self.output_buffers[name])
        if name in self.inplace_buffers:
            return cast(InplacedBuffer, self.inplace_buffers[name]).inner_name
        if name.startswith("seed"):
            return self._lookup("seed", self.input_buffers, name)
        return self._lookup("in_ptr", self.input_buffers, name)

    def output(self, name: str) -> str:
        # [FUNCTION DOC] Returns the kernel argument name to use when writing a buffer. Resolves mutation aliases through the scheduler; reuses
        # the in-place name if present; otherwise allocates an out_ptr name. Raises AssertionError if the buffer was removed.
        if V.graph.scheduler:
            name = V.graph.scheduler.mutation_real_name.get(name, name)
        if name in V.graph.removed_buffers:
            raise AssertionError(name)
        if name in self.inplace_buffers:
            return cast(InplacedBuffer, self.inplace_buffers[name]).inner_name
        return self._lookup("out_ptr", self.output_buffers, name)

    def make_inplace(self, input_name: str, output_name: str) -> None:
        # [FUNCTION DOC] Makes output_name share memory with input_name as an in-place kernel argument. Propagates unaligned status from input
        # to output in the graph. If the input is already in-place, adds the output to its group; otherwise creates a new
        # in_out_ptr group with both names. Raises AssertionError if the output is already in-place or an existing group is removed.
        if input_name in V.graph.unaligned_buffers:
            V.graph.unaligned_buffers.add(output_name)
        if output_name in self.inplace_buffers:
            raise AssertionError(output_name)
        if input_name in self.inplace_buffers:
            buf = self.inplace_buffers[input_name]
            if isinstance(buf, RemovedArg):
                raise AssertionError("buf must not be a RemovedArg")
            buf.other_names.append(output_name)
            self.inplace_buffers[output_name] = buf
        else:
            alive_buffers = [
                val
                for val in self.inplace_buffers.values()
                if not isinstance(val, RemovedArg)
            ]
            removed_buffers = [
                val
                for val in self.inplace_buffers.values()
                if isinstance(val, RemovedArg)
            ]
            inplace_buffer_idx = len(unique(alive_buffers)) + len(removed_buffers)
            buf = InplacedBuffer(
                f"in_out_ptr{inplace_buffer_idx}",
                [input_name, output_name],
            )
            self.inplace_buffers[input_name] = buf
            self.inplace_buffers[output_name] = buf

    def workspace(
        self, nelem: sympy.Expr, zero_fill: bool, dtype: torch.dtype = torch.uint8
    ) -> tuple[str, str, int]:
        # [FUNCTION DOC] Allocates or extends a workspace buffer of nelem elements with the given zero fill and dtype. Creates a new
        # WorkspaceArg on the current device; if an existing workspace arg can be joined with it (same inner name, dtype,
        # device) the two are merged and the offset (the previous count) is returned. Returns (inner name, outer name,
        # offset). Raises AssertionError on name clashes. Mutates workspace_args.
        """
        Allocate or extend a workspace buffer of nelem elements.

        This function manages the allocation of a workspace buffer. It either creates
        a new WorkspaceArg or extends an existing one.

        Note:
        - Calling this function will in-place mutate the args by adding or updating
        a WorkspaceArg.
        - The codegen for generating the Python argdefs and call_defs will check
        this field and allocate the buffer accordingly.
        - A new argument "ws_ptr" will be present in the generated code.

        Args:
            nelem (sympy.Expr): The number of elements to allocate.
            zero_fill (bool): Whether to initialize the buffer to zero.
            dtype (torch.dtype): the dtype of the workspace tensor

        Returns:
            Tuple[str, str, int]: A tuple containing:
                - "ws_ptr": A string identifier for the workspace pointer.
                - "workspace_{i}": a graph level unique identifier for
                    the workspace tensor.
                - offset: An integer representing the item offset in the workspace.
        """
        arg = WorkspaceArg(
            count=nelem,
            zero_mode=WorkspaceZeroMode.from_bool(zero_fill),
            device=V.graph.get_current_device_or_throw(),
            outer_name=WorkspaceArg.unique_name(),
            dtype=dtype,
        )
        for i, existing_arg in enumerate(self.workspace_args):
            if WorkspaceArg.can_join(existing_arg, arg):
                offset = existing_arg.count
                self.workspace_args[i] = WorkspaceArg.join(existing_arg, arg)
                return existing_arg.inner_name, existing_arg.outer_name, offset
            if not (
                existing_arg.inner_name != arg.inner_name
                and existing_arg.outer_name != arg.outer_name
            ):
                raise AssertionError(existing_arg)
        self.workspace_args.append(arg)
        return arg.inner_name, arg.outer_name, 0

    def semaphores(self, min_size: sympy.Expr) -> str:
        # [FUNCTION DOC] Lazily adds a graph wide semaphore buffer workspace argument (int32-sized elements, zeroed per graph) of at least
        # min_size on the current device and returns its inner kernel argument name. The outer name includes the device type
        # and, unless compile-on-one-rank is enabled, the device index. Raises AssertionError if a different semaphore arg
        # is already present. Mutates workspace_args.
        # KNOWN ISSUE: the docstring says repeated calls return the same buffer, but the arg is appended on every call, so
        # duplicates can accumulate in workspace_args; the dtype used is uint32, not int32.
        """
        Lazily allocate a graph-wide semaphores buffer with at least min_size.  This is a single buffer shared by
        all kernels and zero initialized once at graph start.  Each kernel must leave the buffer zeroed on exit.

        Warning: multiple calls to this function will return the same buffer.

        Args:
            min_size: the number of int32 semaphores required

        Returns:
            name of the semaphores buffer
        """
        from torch.fx.experimental.proxy_tensor import _coor_enabled

        current_device = V.graph.get_current_device_or_throw()
        # This name is emitted into the wrapper, so under compile-on-one-rank it cannot
        # carry the rank's device index (the graph is single-device there, so the type
        # alone still distinguishes buffers).
        suffix = "" if _coor_enabled() else f"_{current_device.index}"
        arg = WorkspaceArg(
            count=min_size,
            zero_mode=WorkspaceZeroMode.ZERO_PER_GRAPH,
            dtype=torch.uint32,
            inner_name="sem_ptr",
            outer_name=f"semaphores_{current_device.type}{suffix}",
            device=current_device,
        )
        for existing_arg in self.workspace_args:
            if existing_arg.inner_name == arg.inner_name:
                if arg != existing_arg:
                    raise AssertionError((arg, existing_arg))
        self.workspace_args.append(arg)
        return arg.inner_name

    def seed_offset(self, name: str, value: int) -> str:
        # [FUNCTION DOC] Lifts an integer seed offset constant into a kernel size argument to improve cache hits. Returns the existing arg name if the value is already present; otherwise registers it under the given name (made unique by a count suffix on name collisions). Raises AssertionError for non-int values.
        if not isinstance(value, int):
            raise AssertionError((type(value), value))
        # here we are lifting a constant integer into an arg to the kernel to try to get additional cache hits
        value = sympy.Integer(value)
        if value in self.sizevars:
            return self.sizevars[value]
        if name in self.sizevars.values():
            name = (
                f"{name}{sum(1 for v in self.sizevars.values() if v.startswith(name))}"
            )
        self.sizevars[value] = name
        return name

    def size(self, name: sympy.Symbol) -> str:
        # [FUNCTION DOC] Returns the kernel argument name for a sympy symbol size variable, allocating a ks name when new; the seed symbol keeps the fixed name seed. Raises AssertionError if not a sympy Symbol.
        if not isinstance(name, sympy.Symbol):
            raise AssertionError((type(name), name))
        if name.name == "seed":
            self.sizevars[name] = "seed"  # don't manage the name of seeds
            return "seed"
        return self._lookup("ks", self.sizevars, name)

    def call_names(self) -> Iterator[str]:
        # [FUNCTION DOC] Returns an iterator over the outer names of input buffers, output buffers and size variables, in that order.
        return chain(
            self.input_buffers.keys(), self.output_buffers.keys(), self.sizevars.keys()
        )

    def arg_name(self, name: str) -> str | None:
        # [FUNCTION DOC] Returns the inner kernel argument name for an outer buffer name, looking in in-place, output and input maps in that order, or None if absent or removed.
        """
        Returns inner name of a given outer name.
        """
        inplaced = self.inplace_buffers.get(name, None)
        if inplaced is not None and not isinstance(inplaced, RemovedArg):
            return inplaced.inner_name
        output_name = self.output_buffers.get(name, None)
        if output_name is not None and not isinstance(output_name, RemovedArg):
            return output_name
        return self.input_buffers.get(name, None)

    def wrap_ptr_arg(self, buf: str, dtype: torch.dtype) -> str:
        # [FUNCTION DOC] Returns the buffer name unchanged; subclasses may wrap pointer arguments.
        return buf

    def wrap_size_arg(self, size: SymbolLike) -> str:
        # [FUNCTION DOC] Returns the size argument converted to a string; subclasses may wrap it.
        return str(size)

    def cpp_argdefs(
        self, dtype_to_cpp_type: dict[torch.dtype, str] | None = None
    ) -> tuple[list[str], list[str], list[str]]:
        # [FUNCTION DOC] Builds C++ kernel parameters. Returns (argument definitions, call arguments, argument types) for in-place buffers
        # (non-const pointers), inputs (const pointers), outputs (pointers) and size variables (float for unbacked floats,
        # otherwise the index type). Uses the given dtype to C++ type map or the default. Ensures size variables are computed
        # in the wrapper. Raises AssertionError if workspace args exist (unsupported on CPU).
        from .cpp_utils import INDEX_TYPE

        if dtype_to_cpp_type is None:
            from .cpp_utils import DTYPE_TO_CPP

            dtype_to_cpp_type = DTYPE_TO_CPP

        call_args = []
        arg_defs = []
        arg_types = []
        for inplaced in unique(self.inplace_buffers.values()):
            if isinstance(inplaced, RemovedArg):
                continue
            outer = inplaced.other_names[-1]
            inner = inplaced.inner_name
            dtype = V.graph.get_dtype(outer)
            cpp_dtype = dtype_to_cpp_type[dtype]
            arg_defs.append(f"{cpp_dtype}* {inner}")
            call_args.append(self.wrap_ptr_arg(outer, dtype))
            arg_types.append(f"{cpp_dtype}*")
        for outer, inner in self.input_buffers.items():
            if outer in self.inplace_buffers:
                continue
            dtype = V.graph.get_dtype(outer)
            cpp_dtype = dtype_to_cpp_type[dtype]
            arg_defs.append(f"const {cpp_dtype}* {inner}")
            call_args.append(self.wrap_ptr_arg(outer, dtype))
            arg_types.append(f"const {cpp_dtype}*")
        for outer, maybe_inner in self.output_buffers.items():
            if outer in self.inplace_buffers or isinstance(maybe_inner, RemovedArg):
                continue
            dtype = V.graph.get_dtype(outer)
            cpp_dtype = dtype_to_cpp_type[dtype]
            arg_defs.append(f"{cpp_dtype}* {maybe_inner}")
            call_args.append(self.wrap_ptr_arg(outer, dtype))
            arg_types.append(f"{cpp_dtype}*")
        for outer, inner in self.sizevars.items():
            if isinstance(outer, sympy.Symbol) and symbol_is_type(
                outer, (SymT.UNBACKED_FLOAT)
            ):
                arg_defs.append(f"const float {inner}")
                arg_types.append("const float")
            else:
                arg_defs.append(f"const {INDEX_TYPE} {inner}")
                arg_types.append(f"const {INDEX_TYPE}")
            call_args.append(self.wrap_size_arg(outer))
            if V.graph.wrapper_code:
                V.graph.wrapper_code.ensure_size_computed(outer)
        if self.workspace_args:
            raise AssertionError("Workspace not supported on CPU ")
        return arg_defs, call_args, arg_types

    def python_argdefs(
        self,
    ) -> tuple[list[ArgName], list[str], list[KernelArgType], list[Any]]:
        # [FUNCTION DOC] Builds Python/Triton kernel parameters. Returns (ArgName list, call argument names, precompile arg descriptors,
        # argument types) covering in-place buffers, inputs and outputs (skipping removed ones), size variables (also ensuring
        # they are computed in the wrapper) and workspace args.
        arg_defs: list[ArgName] = []
        call_args: list[str] = []
        arg_types: list[Any] = []
        precompile_args: list[KernelArgType] = []
        for inplaced in unique(self.inplace_buffers.values()):
            if isinstance(inplaced, RemovedArg):
                continue
            arg_defs.append(ArgName(inplaced.inner_name))
            call_args.append(inplaced.other_names[-1])
            arg_types.append(V.graph.get_dtype(inplaced.other_names[-1]))
            precompile_args.append(
                TensorArg(
                    name=inplaced.inner_name,
                    buffer=inplaced.other_names[-1],
                    dtype=V.graph.get_dtype(inplaced.other_names[-1]),
                )
            )
        for outer, inner in chain(
            self.input_buffers.items(),
            # pyrefly: ignore [bad-argument-type]
            self.output_buffers.items(),
        ):
            if outer in self.inplace_buffers or isinstance(inner, RemovedArg):
                continue
            arg_defs.append(ArgName(inner))
            call_args.append(outer)
            arg_types.append(V.graph.get_dtype(outer))
            precompile_args.append(
                TensorArg(
                    name=inner,
                    buffer=outer,
                    dtype=V.graph.get_dtype(outer),
                )
            )
        for outer, inner in self.sizevars.items():
            arg_defs.append(ArgName(inner))
            call_args.append(outer)
            arg_types.append(type(outer))
            precompile_args.append(SizeArg(inner, outer))
            if V.graph.wrapper_code:
                V.graph.wrapper_code.ensure_size_computed(outer)
        for arg in self.workspace_args:
            arg_defs.append(ArgName(arg.inner_name))
            call_args.append(arg.outer_name)
            precompile_args.append(arg)
            arg_types.append(arg.dtype)
        return arg_defs, call_args, precompile_args, arg_types

    def aliases(self) -> Iterator[tuple[str, str]]:
        # [FUNCTION DOC] Generator yielding pairs of (kernel argument name, in-place argument name) for each input or output that shares an in-place buffer, skipping buffers scheduled for in-place removal.
        for inplaced in unique(self.inplace_buffers.values()):
            if isinstance(inplaced, RemovedArg):
                continue
            for other in inplaced.other_names:
                if (
                    other in V.graph.inplaced_to_remove
                    or other in V.kernel.inplaced_to_remove
                ):
                    continue
                if other in self.input_buffers:
                    yield self.input_buffers[other], inplaced.inner_name
                if other in self.output_buffers:
                    yield cast(str, self.output_buffers[other]), inplaced.inner_name

    def is_removed(self, name: str) -> bool:
        # [FUNCTION DOC] Returns true if the buffer is marked removed in both the output buffers and in-place buffers maps.
        return isinstance(
            self.output_buffers.get(name, REMOVED), RemovedArg
        ) and isinstance(self.inplace_buffers.get(name, REMOVED), RemovedArg)

    # Includes inplace buffers, excludes removed buffers.  Essentially,
    # after you do a call into this kernel, which buffers actually contain
    # updated data?  Modeled off of python_argdefs.
    def live_output_buffers(self) -> OrderedSet[str]:
        # [FUNCTION DOC] Returns the ordered set of outer buffer names whose data is updated after calling the kernel: the last name of each live in-place group plus non-removed non-in-place outputs.
        live_outs: OrderedSet[str] = OrderedSet()
        for inplaced in unique(self.inplace_buffers.values()):
            if isinstance(inplaced, RemovedArg):
                continue
            live_outs.add(inplaced.other_names[-1])
        for outer, inner in self.output_buffers.items():
            if outer in self.inplace_buffers or isinstance(inner, RemovedArg):
                continue
            live_outs.add(outer)
        return live_outs


class CSEVariable:
    # [CLASS DOC] A named expression in generated code, with annotation hooks for backends (overridden through the kernel's variable
    # factory). State: name, bounds (value ranges), use_count (starts at one), dtype and shape. Equality and hashing are
    # by name only.
    """A CSEVariable is just a name for an expression but it is useful to be able to annotate them on a backend dependent basis.
    To do so, the backends can simply overload `Kernel.create_cse_var`
    The "CSEVariable.update_on_args" method gives you a hook for annotations
    See example of TritonCSEVariable in triton.py
    """

    def __init__(
        self,
        name: str,
        bounds: ValueRanges[Any],
        dtype: torch.dtype | None = None,
        shape: BlockShapeType = None,
    ):
        # [FUNCTION DOC] Stores name, bounds, dtype and shape and starts the use count at one. Raises AssertionError if bounds is not a ValueRanges.
        super().__init__()
        if not isinstance(bounds, ValueRanges):
            raise AssertionError(type(bounds))
        self.name = name
        self.bounds = bounds
        self.use_count = 1  # track how many times this expression is used
        self.dtype = dtype
        self.shape = shape

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns the variable name.
        return self.name

    def __hash__(self) -> int:
        # [FUNCTION DOC] Hashes the variable name.
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal to another object if it is a CSEVariable with the same name.
        return isinstance(other, CSEVariable) and other.name == self.name

    def update_on_args(self, name: str, args: Any, kwargs: Any) -> None:
        # [FUNCTION DOC] Hook called with the op name and its arguments after the variable is created; the default does nothing, backends override to annotate variables.
        pass

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns a representation with the class name and variable name.
        return f"{self.__class__.__name__}({self.name!r})"


AugmentedKeyT = TypeVar("AugmentedKeyT", default=str)
CSEVariableType = TypeVar("CSEVariableType", bound=CSEVariable, default=CSEVariable)

if TYPE_CHECKING:
    ReductionCacheKey = tuple[
        torch.dtype,
        ReductionType,
        CSEVariable | tuple[CSEVariable, ...],
    ]


class CSE(Generic[CSEVariableType, AugmentedKeyT]):
    # [CLASS DOC] Common subexpression elimination state for kernel code generation. Holds the code prefix and suffix text added
    # around generated assignments, the name prefix and counter for new variables, the expression cache, a store cache
    # (buffer name to value), a reduction cache, invalidated stores and a name to variable map. Caches can be shared across
    # clones and scoped copies.
    """Common subexpression elimination"""

    def __init__(
        self,
        prefix: str = "",
        suffix: str = "",
        name_prefix: str = "tmp",
        iter_buffers: itertools.count[int] | None = None,
        store_cache: MutableMapping[str, CSEVariableType] | None = None,
        reduction_cache: MutableMapping[ReductionCacheKey, CSEVariableType]
        | None = None,
        varname_map: dict[str, CSEVariableType] | None = None,
    ):
        # [FUNCTION DOC] Initializes prefix, suffix, name prefix, empty expression cache, and store and reduction caches, variable counter and name map (reusing the given objects when provided).
        self.prefix = prefix
        self.suffix = suffix
        self._cache: MutableMapping[AugmentedKeyT, CSEVariableType] = {}
        self.name_prefix = name_prefix
        self.store_cache: MutableMapping[str, CSEVariableType] = store_cache or {}
        self.reduction_cache: MutableMapping[ReductionCacheKey, CSEVariableType] = (
            reduction_cache or {}
        )
        self.iter_buffer_ids: itertools.count[int] = iter_buffers or itertools.count()
        self.invalidated_stores: OrderedSet[str] = OrderedSet()
        self.varname_map: dict[str, CSEVariableType] = varname_map or {}

    def invalidate(self, keep_vars: OrderedSet[CSEVariable]) -> None:
        # [FUNCTION DOC] Drops cached store entries and cached expressions whose variables are not in keep_vars; dropped store names are recorded as invalidated stores. Mutates caches.
        for name, tmp in [*self.store_cache.items()]:
            if tmp not in keep_vars:
                del self.store_cache[name]
                self.invalidated_stores.add(name)
        if keep_vars:
            self._cache = {k: v for k, v in self._cache.items() if v in keep_vars}
        else:
            self._cache = {}

    def clone(self) -> Self:
        # [FUNCTION DOC] Returns a new CSE of the same class sharing the store cache, reduction cache, variable counter and name map, but with a fresh expression cache.
        return type(self)(
            prefix=self.prefix,
            suffix=self.suffix,
            name_prefix=self.name_prefix,
            iter_buffers=self.iter_buffer_ids,
            store_cache=self.store_cache,
            varname_map=self.varname_map,
            reduction_cache=self.reduction_cache,
        )

    def scoped_copy(self) -> Self:
        # [FUNCTION DOC] Returns a clone whose expression, reduction and store caches are scoped dictionaries layered over the current ones, so additions are not visible in the original.
        """Return a copy of using ScopedDict so changes to *_cache aren't visible in self"""
        new_cse = self.clone()
        new_cse._cache = ScopedDict(self._cache)
        new_cse.reduction_cache = ScopedDict(self.reduction_cache)
        new_cse.store_cache = ScopedDict(self.store_cache)
        return new_cse

    def augment_key(self, cache_key: str) -> AugmentedKeyT:
        # [FUNCTION DOC] Returns the cache key used for lookups; the default is the key itself. Backends override to add backend specific information.
        "Override this method to augment cache key with backend specifics"
        return cast(AugmentedKeyT, cache_key)

    def put(self, cache_key: str, val: CSEVariableType) -> None:
        # [FUNCTION DOC] Stores the value in the expression cache under the augmented key.
        self._cache[self.augment_key(cache_key)] = val

    def contains(self, cache_key: str) -> bool:
        # [FUNCTION DOC] Returns true if the augmented key is present in the expression cache.
        return self.augment_key(cache_key) in self._cache

    def try_get(self, cache_key: str) -> CSEVariableType | None:
        # [FUNCTION DOC] Returns the cached variable for the key or None.
        return self._cache.get(self.augment_key(cache_key), None)

    def get(self, cache_key: str) -> CSEVariableType:
        # [FUNCTION DOC] Returns the cached variable for the key. Raises KeyError if absent.
        return self._cache[self.augment_key(cache_key)]

    def generate(
        self,
        buffer: IndentedBuffer,
        expr: str | CSEVariable | OpsValue | IndentedBuffer | DeferredLineBase,
        *,
        bounds: ValueRanges[Any] = ValueRanges.unknown(),
        write: bool = True,
        assignment: bool = True,
        dtype: torch.dtype | None = None,
        shape: BlockShapeType = None,
    ) -> CSEVariableType:
        # [FUNCTION DOC] Returns a CSE variable for an expression. A CSEVariable argument has its bounds tightened and use count raised and
        # is returned. Otherwise the cache key is derived from the string, indented buffer text, or deferred line text; on a
        # cache hit the cached variable gets tightened bounds and a higher use count. On a miss a new variable is created and
        # cached, and if write is set the assignment (or bare expression) is written to the buffer with prefix and suffix,
        # preceded by originating node info; runtime dtype assertions may be emitted. Raises AssertionError for invalid
        # combinations (neither write nor assignment, deferred line without assignment, non-string expression).
        if isinstance(expr, OpsValue):
            expr = expr.value

        if not (write or assignment):
            raise AssertionError("expected write or assignment to be set")
        if isinstance(expr, CSEVariable):
            # If the expressions were always created with all the information, we could
            # assert expr.bounds == bounds, but sometimes the expression is created
            # with the loose ValueRanges.unknown(), so we need to tighten the bounds
            expr.bounds = expr.bounds.tighten(bounds)
            expr.use_count += 1
            return cast(CSEVariableType, expr)
        elif isinstance(expr, IndentedBuffer):
            cache_key = expr.getvalue()
        elif isinstance(expr, DeferredLineBase):
            cache_key = expr.line
        else:
            if not isinstance(expr, str):
                raise AssertionError(f"expected str, got {type(expr)}")
            cache_key = expr
        var = self.try_get(cache_key)
        if shape is None and not assignment:
            # since there's no assignment to a variable, use any shape here
            # other than None to avoid the unknown shape failures
            shape = ()
        if not var:
            var = self.newvar(bounds, dtype, shape)
            self.put(cache_key, var)
            if write:
                if V.kernel.current_node:
                    V.kernel.current_node.codegen_originating_info(
                        buffer, only_once=True
                    )
                if isinstance(expr, IndentedBuffer):
                    if assignment:
                        buffer.writeline(f"{self.prefix}{var} =")
                    buffer.splice(expr)
                    buffer.writeline(self.suffix)
                elif isinstance(expr, DeferredLineBase):
                    if not assignment:
                        raise AssertionError("expected assignment to be set")
                    buffer.writeline(
                        expr._new_line(f"{self.prefix}{var} = {expr.line}{self.suffix}")
                    )
                else:
                    if assignment:
                        line = f"{self.prefix}{var} = {expr}{self.suffix}"
                    else:
                        line = f"{expr}{self.suffix}"
                    buffer.writeline(line)

                    # cpp backend cannot determine is_vec at this point
                    if (
                        assignment
                        and (
                            config.test_configs.runtime_triton_dtype_assert
                            or config.test_configs.static_cpp_dtype_assert
                        )
                        and dtype is not None
                        and get_current_backend() != "cpp"
                    ):
                        check_dtype(buffer, var, dtype)

        else:
            var.bounds = var.bounds.tighten(bounds)
            var.use_count += 1

        return var

    def newvar(
        self,
        bounds: ValueRanges[Any] = ValueRanges.unknown(),
        dtype: torch.dtype | None = None,
        shape: BlockShapeType = None,
    ) -> CSEVariableType:
        # [FUNCTION DOC] Creates a new variable named with the name prefix and the next counter value via the kernel's variable factory, records it in the name map and returns it.
        var_name = f"{self.name_prefix}{next(self.iter_buffer_ids)}"
        var = cast(
            "CSEVariableType", V.kernel.create_cse_var(var_name, bounds, dtype, shape)
        )
        self.varname_map[var_name] = var
        return var

    def namedvar(
        self,
        name: str,
        bounds: ValueRanges[Any] = ValueRanges.unknown(),
        dtype: torch.dtype | None = None,
        shape: BlockShapeType = None,
    ) -> CSEVariableType:
        # [FUNCTION DOC] Creates a variable with a caller chosen name via the kernel's variable factory and records it in the name map. Raises an error through the check if the name already exists.
        torch._check_value(
            name not in self.varname_map, lambda: f"duplicate name: {name}"
        )
        var = cast(
            "CSEVariableType", V.kernel.create_cse_var(name, bounds, dtype, shape)
        )
        self.varname_map[name] = var
        return var


class CodeGen:
    # [CLASS DOC] Base class that is a context manager owning an exit stack, used to scope handlers installed while generating code.
    def __init__(self) -> None:
        # [FUNCTION DOC] Creates the exit stack.
        super().__init__()
        self.exit_stack = contextlib.ExitStack()

    def __enter__(self) -> Self:
        # [FUNCTION DOC] Enters the exit stack and returns self.
        self.exit_stack.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # [FUNCTION DOC] Exits the exit stack, unwinding anything entered through it.
        self.exit_stack.__exit__(exc_type, exc_val, exc_tb)


class Kernel(CodeGen, Generic[CSEVariableType]):
    # [CLASS DOC] Base class for generated kernels and code buffers. State: kernel args, load/compute/store code buffers, a CSE
    # object, counters of loads, stores and reductions, sets of kept and stored buffer names, removed and in-place-to-remove
    # buffer names, the current scheduler node and its value bounds, in-place update hints, and kernel name. As a context
    # manager it installs the CSEProxy ops handler and itself as the active kernel; on exit it removes kernel-local buffers.
    # Class attributes: variable prefix, suffix and the overrides factory. Subclasses implement load, store, reductions, etc.
    """Base class for generated kernels and their code buffers."""

    newvar_prefix: str = ""
    suffix: str = ""
    overrides: Callable[[], OpsHandler[Any]] | None = None

    def __init__(
        self, args: KernelArgs | None = None, increase_kernel_count: bool = True
    ) -> None:
        # [FUNCTION DOC] Initializes the kernel: bumps the global generated kernel count unless disabled, creates args (if not given), the three code buffers, the CSE object, counters and bookkeeping sets and maps.
        super().__init__()
        if increase_kernel_count:
            # pyrefly: ignore [bad-assignment]
            metrics.generated_kernel_count += 1
        self.args = args or KernelArgs()
        self.loads = IndentedBuffer()
        self.compute = IndentedBuffer()
        self.stores = IndentedBuffer()

        self.atomic_add_found = False
        self.num_load = 0
        self.num_store = 0
        self.num_reduction = 0

        self.cse: CSE[CSEVariableType, Any] = CSE(self.newvar_prefix, self.suffix)
        self.must_keep_buffers: OrderedSet[str] = OrderedSet()
        self.store_buffer_names: OrderedSet[str] = OrderedSet()
        self._load_mask: str | None = None
        self._load_other: None | int | float = None
        # OrderedSet in set_current_node
        self.current_node: SchedulerNode | None = None
        self.node_to_bounds: dict[torch.fx.Node, ValueRanges[Any]] | None = None

        self.removed_buffers: OrderedSet[str] = OrderedSet()
        self.inplaced_to_remove: OrderedSet[str] = OrderedSet()

        # key: the buffer to write
        # value: the buffer to read and whose memory can be reused for
        #   the buffer specified by key
        self.inplace_update_buffers: dict[str, str] = {}
        # Set minimum number of elements processed per thread.
        self.min_elem_per_thread = 1
        self.kernel_name: str | None = None

    @contextlib.contextmanager
    def set_current_node(self, node: SchedulerNode) -> Iterator[None]:
        # [FUNCTION DOC] Context manager that makes the node the current node and loads its computed value bounds, restoring the previous current node on exit (the bounds are not restored). Reads the node loop body.
        prior = self.current_node
        self.current_node = node
        self.node_to_bounds = node._body.bounds().get_bounds()
        try:
            yield
        finally:
            self.current_node = prior

    @contextlib.contextmanager
    def swap_buffers(
        self,
        lb: IndentedBuffer,
        cb: IndentedBuffer | None = None,
        sb: IndentedBuffer | None = None,
    ) -> Iterator[None]:
        # [FUNCTION DOC] Context manager that temporarily replaces the load, compute and store buffers (compute defaults to the load buffer; if no
        # store buffer is given a scratch one is used) and switches to a scoped copy of CSE. Restores all on exit. Raises
        # AssertionError if code was stored into the scratch store buffer.
        if cb is None:
            cb = lb
        if disallow_stores := sb is None:
            sb = IndentedBuffer()
        loads = self.loads
        compute = self.compute
        stores = self.stores
        cse = self.cse
        self.loads = lb
        self.compute = cb
        self.stores = sb
        self.cse = cse.scoped_copy()
        try:
            yield
        finally:
            self.loads = loads
            self.compute = compute
            self.stores = stores
            self.cse = cse
            # pyrefly: ignore [unbound-name]
            if disallow_stores:
                if sb:
                    raise AssertionError("unexpected store inside swap_buffers")

    def emit_kernel_override(
        self,
        wrapper,
        src_code: str,
        kernel_name: str,
        node_schedule,
        kernel_path: str,
        get_kernel_metadata,
    ) -> bool:
        # [FUNCTION DOC] Hook letting external template handlers override how the kernel is emitted into the wrapper. The default does nothing and returns false meaning use the default path.
        """Override kernel emission. Return True if overridden, False to use default.

        External template handlers (e.g. Helion) can override this method
        to implement custom kernel emission to the wrapper.
        """
        return False

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        # [FUNCTION DOC] Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def indirect_load(self, name: str, index: sympy.Expr) -> CSEVariable:
        # [FUNCTION DOC] Loads a value whose index depends on a previously loaded value: temporarily redirects the load buffer to the compute buffer (loads may have dependencies), performs the load, and restores. Returns the loaded variable.
        """A load that depends on an index we have read"""
        prior = self.loads
        try:
            # put the load in the compute section as it might have deps
            self.loads = self.compute
            return self.load(name, index)
        finally:
            self.loads = prior

    def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable) -> None:
        # [FUNCTION DOC] Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        # [FUNCTION DOC] Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def device_assert_async(self, cond: CSEVariable, msg: str) -> None:
        # [FUNCTION DOC] Raises NotImplementedError: should be handled by CSEProxy.
        raise NotImplementedError(
            f"{type(self).__name__}: device_assert_async should be handled by CSEProxy"
        )

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: CSEVariable | tuple[CSEVariable, ...],
    ) -> CSEVariable | tuple[CSEVariable, ...]:
        # [FUNCTION DOC] Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def partial_accumulate(
        self,
        name: str,
        reduction_type: ReductionType,
        value: CSEVariable,
        extra_meta: dict[str, Any],
    ) -> None:
        # [FUNCTION DOC] Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def scan(
        self,
        dtypes: tuple[torch.dtype, ...],
        combine_fn: Callable[
            [tuple[CSEVariable, ...], tuple[CSEVariable, ...]], tuple[CSEVariable, ...]
        ],
        values: tuple[CSEVariable, ...],
    ) -> tuple[CSEVariable, ...]:
        # [FUNCTION DOC] Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def sort(
        self,
        dtypes: tuple[torch.dtype, ...],
        values: tuple[CSEVariable, ...],
        stable: bool,
        descending: bool,
    ) -> tuple[CSEVariable, ...]:
        # [FUNCTION DOC] Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def var_ranges(self) -> dict[sympy.Symbol, sympy.Expr]:
        # [FUNCTION DOC] Returns the mapping of iteration variables to ranges. Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def bucketize(
        self,
        values: CSEVariable,
        boundaries: tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: CSEVariable,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: tuple[str, sympy.Expr] | None = None,
        sorter_indices: CSEVariable | None = None,
    ) -> CSEVariable:
        # [FUNCTION DOC] Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it. See the bucketize note in CSEProxy.
        """
        See [Note: Inductor bucketize op]
        """
        raise NotImplementedError

    @property
    def assert_function(self) -> str:
        # [FUNCTION DOC] Property returning the name of the device assert function used for index checks. Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def indirect_assert(
        self,
        var: CSEVariable | str,
        lower: str | None,
        upper: str | None,
        mask: CSEVariable | str | None = None,
    ) -> str:
        # [FUNCTION DOC] Builds and returns the text of an assert statement that var lies in the half open range from lower (inclusive, optional)
        # to upper (exclusive, optional), optionally masked so masked-off lanes pass; message says index out of bounds.
        # Raises AssertionError for bad argument types or when both bounds are missing.
        if isinstance(var, CSEVariable):
            var = str(var)
        if not isinstance(var, str):
            raise AssertionError(type(var))
        if not (lower is None or isinstance(lower, str)):
            raise AssertionError(f"expected lower to be None or str, got {type(lower)}")
        if not (upper is None or isinstance(upper, str)):
            raise AssertionError(f"expected upper to be None or str, got {type(upper)}")
        if lower and upper:
            # The conditions need to be in parens because of Python's operator precedence.
            # It'd be less error-prone to use and/or/not, which is supported by triton
            cond = f"({lower} <= {var}) & ({var} < {upper})"
            cond_print = f"{lower} <= {var} < {upper}"
        elif lower:
            cond = f"{lower} <= {var}"
            cond_print = cond
        else:
            if not upper:
                raise AssertionError("expected upper to be set")
            cond = f"{var} < {upper}"
            cond_print = cond

        if mask:
            cond = f"({cond}) | ~({mask})"

        return f'{self.assert_function}({cond}, "index out of bounds: {cond_print}")'

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> None:
        # [FUNCTION DOC] Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def index_to_str(self, index: sympy.Expr) -> str:
        # [FUNCTION DOC] Converts an index expression to code text. Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        raise NotImplementedError

    def __enter__(self) -> Self:
        # [FUNCTION DOC] Enters the base context, then installs the CSEProxy wrapping this kernel's overrides as the ops handler and this kernel as the active kernel handler. Raises AssertionError if no overrides factory is set.
        super().__enter__()
        if not self.overrides:
            raise AssertionError("expected overrides to be set")
        self.exit_stack.enter_context(
            V.set_ops_handler(CSEProxy(self, self.overrides()))
        )
        self.exit_stack.enter_context(V.set_kernel_handler(self))
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # [FUNCTION DOC] Removes kernel-local buffers, then exits the exit stack.
        self.remove_kernel_local_buffers()
        super().__exit__(exc_type, exc_val, exc_tb)

    def remove_kernel_local_buffers(self) -> None:
        # [FUNCTION DOC] Removes buffers that are stored in this kernel and whose last use is also in this fusion group (not must-keep, not
        # an input, and removable through fusion per the scheduler), decrementing the store count. Handles in-place buffers
        # specially (removes the in-place entry only if every alias is removed). Does nothing without a scheduler (template kernels).
        # Mutates args and removed sets.
        """
        Any buffers that are both created and have a last use in the
        same kernel can be removed.

        Note that V.graph.scheduler can be None when codegening triton template
        kernels.
        """
        scheduler = V.graph.scheduler
        if not scheduler:
            return
        fused_node_names = OrderedSet(
            scheduler.name_to_buf[buf].defining_op_name()
            for buf in self.store_buffer_names
            if buf in scheduler.name_to_buf
        )
        names_to_remove: OrderedSet[str] = OrderedSet()
        for name in self.store_buffer_names:
            if (
                name not in self.must_keep_buffers
                and name not in self.args.input_buffers
                and scheduler.can_buffer_be_removed_through_fusion(
                    name, fused_node_names
                )
            ):
                self.num_store -= 1
                names_to_remove.add(name)

        for name in names_to_remove:
            if name in self.args.inplace_buffers:
                buf = self.args.inplace_buffers[name]
                if isinstance(buf, RemovedArg):
                    continue
                remove = all(n in names_to_remove for n in buf.other_names)
                if remove:
                    self.remove_inplace_buffer(name)
                self.inplaced_to_remove.add(name)
            else:
                self.remove_buffer(name)

    def remove_buffer(self, name: str) -> None:
        # [FUNCTION DOC] Marks the output buffer as removed (keeps the dict entry so argument name numbering stays unique) and records it in the removed set.
        # Assign a special value instead of deleting the entry
        # because we still rely on output_buffers's length to
        # generate unique arg name.
        log.debug("remove_buffer(%r)", name)
        self.args.output_buffers[name] = REMOVED
        self.removed_buffers.add(name)

    def remove_inplace_buffer(self, name: str) -> None:
        # [FUNCTION DOC] Marks the in-place buffer as removed and records it in the removed set.
        log.debug("removing_inplace_buffer(%r)", name)
        self.args.inplace_buffers[name] = REMOVED
        self.removed_buffers.add(name)

    def rename_indexing(
        self, index: list[sympy.Expr] | tuple[sympy.Expr, ...] | sympy.Expr
    ) -> sympy.Expr:
        # [FUNCTION DOC] Simplifies an index expression (or each of a list or tuple of them) and replaces size-like free symbols (unbacked ints,
        # sizes, precomputed sizes, unbacked floats) by kernel argument names, adding kernel args as needed. Returns the new
        # expression, or a list for sequence input.
        # adds the necessary kernel args for index expressions
        # and renames variables in index expressions to kernel arg names
        if isinstance(index, (list, tuple)):
            return [self.rename_indexing(x) for x in index]
        index = V.graph.sizevars.simplify(index)
        sorted_symbols = sorted(index.free_symbols, key=lambda s: s.name)
        replacements = {
            x: self.args.size(x)
            for x in sorted_symbols
            if symbol_is_type(
                x,
                (
                    SymT.UNBACKED_INT,
                    SymT.SIZE,
                    SymT.PRECOMPUTED_SIZE,
                    SymT.UNBACKED_FLOAT,
                ),
            )
        }
        return sympy_subs(index, replacements)

    def create_cse_var(self, *args: Any, **kwargs: Any) -> CSEVariable:
        # [FUNCTION DOC] Creates and returns a plain CSEVariable from the given arguments; backends override to return annotated variables.
        return CSEVariable(*args, **kwargs)

    def arg_name(self, node: IRNode) -> str | None:
        # [FUNCTION DOC] Returns the kernel argument name of an IR node's buffer, or None if node is None or unknown.
        """
        Returns arg name of a given input or output node.
        """
        if node is None:
            return None
        return self.args.arg_name(node.get_name())

    def record_op_trace(
        self,
        name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any = None,
    ) -> None:
        # [FUNCTION DOC] Hook called for each traced op with name, arguments and result; the default does nothing.
        pass


@dataclasses.dataclass
class OptimizationContext:
    # [CLASS DOC] Dataclass stored in FX node meta under the key opt_ctx: carries the node's propagated dtype and an op name.
    key: ClassVar[str] = "opt_ctx"

    dtype: torch.dtype | None = None
    ops_name: str = ""


@functools.cache
def jinja2_env() -> Any:
    # [FUNCTION DOC] Cached; returns a jinja2 environment with strict undefined handling, or None if jinja2 is not installed.
    try:
        import jinja2

        return jinja2.Environment(
            undefined=jinja2.StrictUndefined,
        )
    except ImportError:
        return None


class KernelTemplate:
    # [CLASS DOC] Base class for kernel templates (children: Triton and CUTLASS templates). Holds a name and an optional source hash; provides jinja template helpers and choice generation wrappers that catch NotImplementedError.
    """
    Base class for defining kernel templates.

    Children classes: TritonTemplate, CUTLASSTemplate
    """

    @staticmethod
    def indent_except_first(
        source: str, num_indents: int, indents_spacing: int = 4
    ) -> str:
        # [FUNCTION DOC] Returns the source with every line except the first prefixed by the given number of indentation levels of spaces.
        lines = source.splitlines(True)
        if len(lines) > 1:
            lines[1:] = [
                (" " * indents_spacing * num_indents) + line for line in lines[1:]
            ]
        return "".join(lines)

    @staticmethod
    def _template_from_string(source: str) -> Any:
        # [FUNCTION DOC] HELPER: Compiles jinja source text with a registered indent_except_first filter and returns the template, or None if jinja2
        # is unavailable. Template syntax errors are re-raised as a detailed error showing the failing line and column context.
        env = jinja2_env()
        if env is None:
            return None
        env.filters["indent_except_first"] = KernelTemplate.indent_except_first
        from jinja2 import TemplateSyntaxError

        try:
            return env.from_string(source)
        except TemplateSyntaxError as e:

            class DetailedTemplateSyntaxError(TemplateSyntaxError):
                # [CLASS DOC] Template syntax error subclass that keeps the original error and formats a message with the failing line and surrounding source lines.
                def __init__(self, original_error: TemplateSyntaxError) -> None:
                    # [FUNCTION DOC] Copies message, line number, name and filename from the original error and keeps the original.
                    super().__init__(
                        original_error.message,
                        original_error.lineno,
                        original_error.name,
                        original_error.filename,
                    )
                    self.original_error = original_error

                def __str__(self) -> str:
                    # [FUNCTION DOC] Returns a multi-line description: the error line, message and a context window of nearby source lines, marking the failing line and column when available.
                    error_info = f"Error in template at line {self.lineno}\n"
                    error_info += f"Error message: {self.message}\n"
                    if hasattr(self.original_error, "source"):
                        lines = self.original_error.source.split("\n")
                        error_info += "Context:\n"
                        start = max(0, self.lineno - 2)
                        end = min(len(lines), self.lineno + 2)
                        for i in range(start, end):
                            if i == self.lineno - 1:
                                error_info += f"{i + 1}: --> {lines[i]}\n"
                                if hasattr(self.original_error, "column"):
                                    error_info += (
                                        "     "
                                        + " " * (self.original_error.column - 1)
                                        + "^\n"
                                    )
                            else:
                                error_info += f"{i + 1}:     {lines[i]}\n"
                    return error_info

            raise DetailedTemplateSyntaxError(e) from e

    @staticmethod
    def _fake_get_dtype(
        fake_outs: list[Buffer] | Buffer,
    ) -> Callable[[str], torch.dtype]:
        # [FUNCTION DOC] HELPER: Returns a dtype lookup function that answers from the given fake output buffers and falls back to the graph's real lookup for other names.
        _get_dtype_real = V.graph.get_dtype
        if isinstance(fake_outs, (list, tuple)):
            lookup = {buf.get_name(): buf.get_dtype() for buf in fake_outs}
        else:
            lookup = {fake_outs.get_name(): fake_outs.get_dtype()}

        def get_dtype(name: str) -> torch.dtype:
            # [FUNCTION DOC] HELPER: Returns the dtype from the fake buffers map if present, else the real graph lookup.
            result = lookup.get(name)
            if result is not None:
                return result
            return _get_dtype_real(name)

        return get_dtype

    def __init__(self, name: str, hash: str | None = None) -> None:
        # [FUNCTION DOC] Stores the template name and optional source hash.
        self.name = name
        self._hash = hash

    @property
    def uid(self) -> str:
        # [FUNCTION DOC] Property returning a unique, reproducible id for the template (defaults to its name); override to add a prefix.
        """
        entry point to override for templates to ensure a uid e.g. through a prefix

        the purpose of this is that every KernelTemplate/ExternKernelChoice is unique
        in the system, but reproducible e.g. restarting pytorch should yield the same id
        """
        # TODO(coconutruben): add some central registration to assert on global uniqueness
        return self.name

    @property
    def src_hash(self) -> str | None:
        # [FUNCTION DOC] Property returning the optional source hash used to detect template changes; override if detection differs.
        """
        source hash for a Template.

        Templates can optionally provide a src hash to make it easier to cache/validate that
        a template has not changed from one version to another. Override this if that detection
        is different for your specific Template
        """
        return self._hash

    def choice_or_none(self, **kwargs: Any) -> ChoiceCaller | None:
        # [FUNCTION DOC] Tries to generate a ChoiceCaller with the given keyword arguments; returns it, or None if generation failed with NotImplementedError.
        """
        Maybe generates a new ChoiceCaller and returns it, or None if generation fails.

        kwargs: Additional kwargs to be passed to self.generate() to generate a new ChoiceCaller.
        """
        temp_choices: list[Any] = []
        result = self.maybe_append_choice(temp_choices, **kwargs)
        if result is None and len(temp_choices) == 1:
            return temp_choices[0]
        return None

    def maybe_append_choice(
        self, choices: list[Any], **kwargs: Any
    ) -> NotImplementedError | None:
        # [FUNCTION DOC] Tries to generate a choice and append it to the list; returns None on success or the NotImplementedError (logged at info level) on failure.
        """
        Maybe generates a new ChoiceCaller and appends it into existing choices.
        Returns None if success, otherwise returns the error.

        choices: A list of ChoiceCallers.
        kwargs: Additional kwargs to be passed to self.generate() to generate a new ChoiceCaller.
        """

        try:
            choices.append(self.generate(**kwargs))
            return None
        except NotImplementedError as e:
            log.info(
                "Cannot Append Choice: %s. KernelTemplate type is %s",
                e,
                type(self),
                stack_info=log.getEffectiveLevel() < logging.INFO,
            )
            return e

    def generate(self, **kwargs: Any) -> ChoiceCaller:
        # [FUNCTION DOC] Generates a ChoiceCaller from the arguments. Not implemented in the base class; raises NotImplementedError. Subclasses/backends override it.
        """
        Generates a ChoiceCaller instance from the given arguments.
        """

        raise NotImplementedError


class CSEProxy(DefaultHandler):
    # [CLASS DOC] Ops handler used while generating a kernel: forwards every op to the parent (backend) handler, performs common
    # subexpression elimination of the result via the kernel's CSE, attaches value bounds, dtype and shape, optionally emits
    # dtype/shape/NaN asserts, and records op traces. Loads, stores, reductions, scans, sorts and bucketize are routed to the
    # kernel with bookkeeping of counts and the store cache.
    """A ops handler that proxies calls to `kernel` and its
    handler and returns `CSEVariable`s with correct shape and dtype.
    """

    name = "CSEProxy"

    def __init__(self, kernel: Kernel[Any], parent_handler: OpsHandler[Any]):
        # [FUNCTION DOC] Stores the kernel and the parent handler and creates a value range analysis object.
        super().__init__()
        from ..bounds import ValueRangeAnalysis

        self.vr_analysis = ValueRangeAnalysis()
        self.kernel = kernel
        self.parent_handler = parent_handler

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # [FUNCTION DOC] HELPER: Default handler for any op: computes value bounds, calls the parent handler op, determines output dtype and shape
        # (from the kernel for masked ops on triton/cpp, otherwise from the dtype and shape propagation handlers for the
        # triton, cpp and mps backends), then maps CSE generation over the (possibly nested) result with checks. Raises
        # AssertionError if the dtype is missing on triton or cpp. Records an op trace and returns the CSE'd result.
        bounds = self._bound_variable(name, *args, **kwargs)

        value = getattr(self.parent_handler, name)(*args, **kwargs)
        dtype_handler = DtypePropagationOpsHandler()
        shape_handler = ShapePropagationOpsHandler()

        backend = get_current_backend()

        shape_op = getattr(shape_handler, name)
        output_dtype = None
        output_shape = None

        if name == "masked" and backend == "triton":
            output_dtype = value.dtype
            output_shape = value.shape
        elif name == "masked" and backend == "cpp":
            output_dtype = V.interpreter.current_node.meta.get(
                OptimizationContext.key, None
            ).dtype
            # TODO: fix me
            output_shape = None
        elif backend in ("triton", "cpp", "mps"):
            dtype_op = getattr(dtype_handler, name)
            output_dtype = dtype_op(*args, **kwargs)
            output_shape = shape_op(*args, **kwargs)

        if backend in ("triton", "cpp"):
            # maybe there are some exceptions on mps?
            if output_dtype is None:
                raise AssertionError("expected output_dtype to be not None")

        output_idx = 0

        def do_cse(v: str | CSEVariable) -> CSEVariable:
            # [FUNCTION DOC] HELPER: Converts one result value into a CSE variable: picks its dtype and shape from the outputs (by running index), fills
            # missing dtype/shape on existing variables, generates via the CSE with bounds, updates the variable with op arguments,
            # and emits dtype, shape and NaN asserts when the test configs enable them.
            # we tree_map over the output, so we need to fetch corresponding dtype
            nonlocal output_idx
            var_dtype: torch.dtype | None = (
                output_dtype[output_idx]
                if isinstance(output_dtype, (list, tuple))
                else output_dtype
            )
            var_shape: BlockShapeType = (
                output_shape[output_idx]  # type: ignore[assignment]
                if isinstance(output_shape, (list, tuple))
                and len(output_shape) > 0
                and isinstance(output_shape[0], (list, tuple))
                else output_shape
            )
            output_idx += 1

            # some cpp op implementations don't set the dtype
            if isinstance(v, CSEVariable):
                if backend == "cpp" and v.dtype is None:
                    v.dtype = var_dtype
                if v.shape is None:
                    v.shape = var_shape

            csevar = V.kernel.cse.generate(
                V.kernel.compute,
                v,
                bounds=bounds,
                # pyrefly: ignore[bad-argument-type]
                dtype=output_dtype,
                shape=output_shape,
            )

            csevar.update_on_args(name, args, kwargs)

            if (
                config.test_configs.runtime_triton_dtype_assert
                or config.test_configs.static_cpp_dtype_assert
            ) and var_dtype is not None:
                check_dtype(V.kernel.compute, csevar, var_dtype)

            if (
                config.test_configs.runtime_triton_shape_assert
                and _is_runtime_triton_assert_target()
            ):
                shape_to_check = csevar.shape if csevar.shape is not None else var_shape
                if shape_to_check is not None:
                    check_shape(V.kernel.compute, csevar, shape_to_check)

            if config.runtime_triton_nan_asserts:
                check_nan(V.kernel.compute, csevar)

            return csevar

        result = pytree.tree_map(do_cse, value)
        self.kernel.record_op_trace(name, args, kwargs, result)
        return result

    def _bound_variable(self, name: str, *args: Any, **kwargs: Any) -> ValueRanges[Any]:
        # [FUNCTION DOC] HELPER: Returns value ranges for the op result. Unknown for template kernels or when no interpreter is active; the FX node
        # bound if the current node is this op; otherwise, if configured, computed from the argument bounds with the value range
        # analysis (unknown for set_indirect, reduction and scan nodes). Raises AssertionError on unexpected state or keyword args.
        """
        If the variable comes from an FX node, we forward the bound we have already computed
        Else, if the variable when codegen'ing another op, we try to compute its bounds
        """
        from ..bounds import ValueRangeAnalysis
        from ..select_algorithm import TritonTemplateKernel
        from .cutlass.kernel import CUTLASSTemplateKernel

        if isinstance(V.kernel, TritonTemplateKernel):
            return ValueRanges.unknown()

        if isinstance(V.kernel, CUTLASSTemplateKernel):
            return ValueRanges.unknown()

        if isinstance(V.interpreter, NullHandler):
            return ValueRanges.unknown()

        fx_node = V.interpreter.current_node
        if fx_node.target == name and self.kernel.node_to_bounds is not None:
            if not isinstance(self.kernel.node_to_bounds, dict):
                raise AssertionError(type(self.kernel.node_to_bounds))
            return self.kernel.node_to_bounds.get(fx_node, ValueRanges.unknown())
        elif config.compute_all_bounds and hasattr(ValueRangeAnalysis, name):
            # These create lots of inner strings. We would need to compute the bounds at the ops
            # We will also likely not get much from computing VRs on these nodes
            if any(s in fx_node.target for s in ("set_indirect", "reduction", "scan")):
                return ValueRanges.unknown()

            # We assume that the inputs come from `ops.` and are not strings. If you want to generate
            # intermediary strings, wrap them in CSE variables with properly initialised bounds.

            # If there is no FX bound but we know how to compute one we do so
            if kwargs:
                raise AssertionError("expected no kwargs")

            def arg_to_bound(x: Any) -> Any:
                # [FUNCTION DOC] HELPER: Converts an argument to its bounds: CSE variable bounds, sympy expression bounds, or the value itself.
                if isinstance(x, CSEVariable):
                    return x.bounds
                elif isinstance(x, sympy.Expr):
                    return bound_sympy(x)
                else:
                    return x

            arg_bounds = list(map(arg_to_bound, args))
            return getattr(self.vr_analysis, name)(*arg_bounds)
        return ValueRanges.unknown()

    def indirect_indexing(
        self,
        var: CSEVariable,
        size: sympy.Expr | int,
        check: bool = True,
        wrap_neg: bool = True,
    ) -> sympy.Symbol:
        # [FUNCTION DOC] Converts a tensor variable to a sympy index symbol, wrapping negative indices (adding size, using a where when mixed) if
        # wrap_neg, and tightening the variable's bounds. Then asks the parent handler for the index symbol and, when assertions
        # are generated for check, requests lower/upper bound checks from the kernel unless value ranges prove them unnecessary.
        # Raises AssertionError if size is not an integer or sympy expression.
        if isinstance(size, int):
            size = sympy.Integer(size)
        if not isinstance(size, sympy.Expr):
            raise AssertionError((type(size), size))
        # Skip CSE since this doesn't return an expression

        if var.bounds.lower < 0:
            if wrap_neg:
                stm = ops.add(var, ops.index_expr(size, torch.long))
                # Mixed negative and non-negative
                if var.bounds.upper >= 0:
                    lt = ops.lt(var, 0)
                    stm = ops.where(lt, stm, var)
            else:
                stm = var

            # Propagate bounds as we know how to compute them properly.
            # When wrap_neg=False, negative indices stay negative and must keep
            # their original bounds so lower-bound checks are not elided.
            new_bounds = var.bounds if not wrap_neg else ValueRanges.unknown()
            if (
                wrap_neg
                and var.bounds != ValueRanges.unknown()
                and isinstance(size, sympy.Number)
            ):
                # Take the negative part of the bound and add size to it
                # Then take union of that and the positive part
                # This is a tighter bound than that of a generic ops.where, as we have info on the cond
                neg_bounds = var.bounds & ValueRanges(-int_oo, -1)
                new_bounds = ValueRanges(
                    neg_bounds.lower + size, neg_bounds.upper + size
                )
                # We don't have a good way of representing the empty range
                if var.bounds.upper >= 0:
                    pos = var.bounds & ValueRanges(0, int_oo)
                    new_bounds = new_bounds | pos

            var = self.kernel.cse.generate(
                self.kernel.compute,
                stm,
                bounds=new_bounds,
                dtype=var.dtype,
                shape=var.shape,
            )

        sympy_var = self.parent_handler.indirect_indexing(var, size, check)
        if generate_assert(check):
            assert_lower = not (var.bounds.lower >= 0)
            # value ranges cannot x < s when x and s are symbols
            assert_upper = not isinstance(size, sympy.Number) or not (
                var.bounds.upper < size
            )
            self.kernel.check_bounds(sympy_var, size, assert_lower, assert_upper)
        return sympy_var

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> None:
        # [FUNCTION DOC] Delegates bounds checking to the kernel.
        return self.kernel.check_bounds(expr, size, lower, upper)

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        # [FUNCTION DOC] Loads a buffer at an index. Marks the buffer must-keep if it was invalidated; indirect loads (index containing temporary
        # variables) go to the kernel's indirect load; if a value for the buffer is in the store cache it is returned; otherwise
        # loads through the kernel, counts the load when the variable is new and records an op trace. Returns the variable.
        if name in self.kernel.cse.invalidated_stores:
            # A load from an invalidated store requires us to
            # keep the actual buffer around
            V.kernel.must_keep_buffers.add(name)
        if free_symbol_is_type(index, SymT.TMP):
            return self.kernel.indirect_load(name, index)
        store_cache = self.kernel.cse.store_cache
        if name in store_cache:
            return store_cache[name]
        out = self.kernel.load(name, index)
        # count load that is not in the store_cache, and also not in the
        # cse cache.
        if out.use_count == 1:
            self.kernel.num_load += 1
        self.kernel.record_op_trace("load", (name, index), {}, out)
        return out

    def _update_store_cache(self, name: str, value: CSEVariable) -> None:
        # [FUNCTION DOC] HELPER: Records the value as the latest stored value of the buffer in the store cache, and for buffers known to the graph also for every buffer that mutates it.
        self.kernel.cse.store_cache[name] = value
        if self.kernel.current_node and name in V.graph.name_to_buffer:
            buf = self.kernel.current_node.get_output(name)
            for other_name in buf.get_mutations():
                self.kernel.cse.store_cache[other_name] = value

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        # [FUNCTION DOC] Stores a value: records the buffer in stored names, updates the store cache unless the mode is atomic add, and when the
        # buffer is not removed performs the kernel store and counts it. Records an op trace.
        self.kernel.store_buffer_names.add(name)
        # Update store cache when mode is None or "tma"
        if mode != "atomic_add":
            self._update_store_cache(name, value)
        if name not in V.graph.removed_buffers:
            self.kernel.store(name, index, value, mode=mode)
            self.kernel.num_store += 1
        self.kernel.record_op_trace("store", (name, index, value, mode), {})

    def device_assert_async(self, cond: CSEVariable, msg: str) -> None:
        # [FUNCTION DOC] Forwards an asynchronous device assert to the kernel and records an op trace.
        self.kernel.device_assert_async(cond, msg)
        self.kernel.record_op_trace("device_assert_async", (cond, msg), {})

    # pyrefly: ignore [bad-override]
    def partial_accumulate(self, *args: Any) -> None:
        # [FUNCTION DOC] Forwards partial accumulation to the kernel.
        self.kernel.partial_accumulate(*args)

    def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable) -> None:
        # [FUNCTION DOC] Records the stored name and cache value, and if the buffer is not removed counts a store and forwards to the kernel's store reduction (returning its result).
        self.kernel.store_buffer_names.add(name)
        self._update_store_cache(name, value)

        if name not in V.graph.removed_buffers:
            self.kernel.num_store += 1
            return self.kernel.store_reduction(name, index, value)

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: CSEVariable | tuple[CSEVariable, ...],
    ) -> CSEVariable | tuple[CSEVariable, ...]:
        # [FUNCTION DOC] Counts a reduction and forwards to the kernel's reduction, returning its result.
        self.kernel.num_reduction += 1
        return self.kernel.reduction(dtype, src_dtype, reduction_type, value)

    def scan(
        self,
        dtypes: tuple[torch.dtype, ...],
        combine_fn: Callable[
            [tuple[CSEVariable, ...], tuple[CSEVariable, ...]],
            tuple[CSEVariable, ...],
        ],
        values: tuple[CSEVariable, ...],
    ) -> tuple[CSEVariable, ...]:
        # [FUNCTION DOC] Forwards to the kernel's scan and returns its result.
        return self.kernel.scan(dtypes, combine_fn, values)

    def sort(
        self,
        dtypes: tuple[torch.dtype, ...],
        values: tuple[CSEVariable, ...],
        stable: bool,
        descending: bool,
    ) -> tuple[CSEVariable, ...]:
        # [FUNCTION DOC] Forwards to the kernel's sort and returns its result.
        return self.kernel.sort(dtypes, values, stable, descending)

    def bucketize(
        self,
        values: CSEVariable,
        boundaries: tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: CSEVariable,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: tuple[str, sympy.Expr] | None = None,
        sorter_indices: CSEVariable | None = None,
    ) -> CSEVariable:
        # [FUNCTION DOC] Forwards to the kernel's bucketize and returns the bucket index variable. The long docstring defines the inputs, boundaries and sorter semantics.
        """
        [Note: Inductor bucketize op]

        Inputs:
        -------
        values: the values to be bucketized.
        boundaries: a tuple containing
          (a) the name of the boundaries tensor (which must be sorted, unless
          the sorting tensor is present),
          (b) the length of the tensor in the last dimension (i.e. the length of
          one set of boundaries),
          (c) the number of elements in the underlying storage (i.e. the length
          of the flattened tensor, ignoring striding), and
          (d) the stride of the tensor in the last dimension.
        boundary_indices: indices into a flattened version of the boundaries
        tensor, of the same size and shape as "values".  Each index points to
        the first element in the set of boundaries to be used for the
        corresponding value.
        indexing_dtype: the dtype to use when indexing into the boundaries
        tensor.  This must be int64 or int32.  This additionally specifies the
        dtype of the return value.
        right: see "Details" below.
        sorter: an optional tuple containing
          (a) the name of an optional sorting tensor, used to access unsorted
          boundaries without reordering the boundaries tensor, and
          (b) the stride of the tensor in the last dimension.
        The values in the sorting tensor are used as indices into the *last*
        dimension of the boundaries tensor, with all other indices matching.
        The size of the sorting and boundaries tensors must be equivalent.
        sorter_indices: must be present if the sorting array is present; see
        "boundary_indices" for the equivalent definition for the boundaries
        tensor.

        Output:
        -------
        The buckets each value belongs in, within a given set of boundaries.  0
        indicates a position before the first boundary, and len(boundaries_set)
        represents a position after the last boundary.

        Details:
        --------
        Given a value and a set of boundaries, calculate the bucket that each
        value belongs to.  This works differently in 1-D and N-D cases.

        for values [[-1, 0, 1, 2], [3, 4, 5, 9]], boundaries [0, 4, 4, 8], right=True
        return =   [[ 0, 1, 1, 1], [1, 3, 3, 4]].

        for values [[-1, 0, 1, 2], [3, 4, 5, 9]], boundaries [[0, 4], [4, 8]], right=True
        return =   [[ 0, 1, 1, 1], [0, 1, 1, 2]]

        Note that in the N-D boundaries case, the shape of "values" and
        "boundaries" must match in every dimension _except_ the last.

        When right == False, bucket i refers to range (boundaries[i], boundaries[i+1]].
        When right == True,  bucket i refers to range [boundaries[i], boundaries[i+1]).

        Boundaries must be non-decreasing, or a sorter must be provided which
        would re-index offsets in a non-decreasing order (e.g. the second output
        of torch.sort(offsets)).  Otherwise, the result is undefined.
        """
        return self.kernel.bucketize(
            values,
            boundaries,
            boundary_indices,
            indexing_dtype,
            right,
            sorter,
            sorter_indices,
        )
