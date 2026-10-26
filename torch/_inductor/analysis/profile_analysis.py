# [MODULE DOC]
# Analysis of Chrome/Perfetto JSON profiler traces: estimates per-kernel FLOPs and
# memory traffic from op input shapes (using adapters registered per aten op and the
# flop counter registry), joins them to kernel events through External ids, computes
# achieved FLOPS and bandwidth percentages versus datasheet peak numbers
# (analysis.device_info), and renders text tables for one profile or a diff of two.
# Main contents: ProfileEvent, KernelStats, Device, JsonProfile, ParseException, the
# adapter registry (adapters_map / register_adapter) and a command line main with
# diff, analysis, augment_trace and combine modes. Import-time side effects: builds
# the adapter registry via decorators and defines the dtype name table.
import json
import logging
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch._inductor.analysis.device_info import DeviceInfo, lookup_device_info
from torch._inductor.utils import tabulate_2d, zip_dicts
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet
from torch.utils.flop_counter import flop_registry


log = logging.getLogger(__name__)


ATEN_PREFIX = "aten::"


@dataclass
class ProfileEvent:
    # [CLASS DOC] Plain record of one profiled event: category, key (name), self device
    # time in milliseconds, and the run-averaged call count. Not used elsewhere in this module.
    category: str
    key: str
    self_device_time_ms: float
    # the benchmark is run multiple times and we average the count across all the
    # runs. It should be an integer but define a float just in case.
    count: float


# adapters convert the json trace into a format that works with flops_counter
ArgsType = tuple[tuple[Any, ...], dict[Any, Any]]
AdapterType = Callable[[tuple[Any, ...], tuple[Any, ...]], ArgsType]
adapters_map: dict[str, AdapterType] = {}


def parse_list(lst: str) -> list[int]:
    # [FUNCTION DOC]
    # Parses a string such as a bracketed comma separated list of integers into a list
    # of ints by stripping brackets and splitting on commas. Raises ValueError on
    # non-integer items (including an empty list string).
    lst = lst.replace("[", "").replace("]", "")
    substrings = lst.split(",")

    return [int(substring.strip()) for substring in substrings]


def register_adapter(
    aten: str | list[str],
) -> Callable[
    [AdapterType],
    AdapterType,
]:
    # [FUNCTION DOC]
    # Decorator factory: registers the decorated adapter function in the module-level
    # adapters_map under one aten op name or each name in a list. The decorator
    # returns the function unchanged. Side effect: mutates adapters_map.
    def decorator(func: AdapterType) -> AdapterType:
        # [FUNCTION DOC] HELPER: Performs the registration described above and returns func.
        # KNOWN ISSUE: declares a global named _adapters_map that does not exist and is unused.
        # pyrefly: ignore [unknown-name]
        global _adapters_map

        if isinstance(aten, str):
            adapters_map[aten] = func
        else:
            for at in aten:
                adapters_map[at] = func
        return func

    return decorator


@register_adapter(["_slow_conv2d_forward"])
def _slow_conv2d_adapter(
    shapes: tuple[Any, ...], concrete: tuple[Any, ...]
) -> tuple[tuple[Any], dict[Any, Any]]:
    # [FUNCTION DOC] HELPER: Adapter for slow conv2d forward: appends a False "transposed" flag to
    # the shapes, copies the fifth concrete input (stride) into the fourth slot, and
    # delegates to conv_adapter. Raises ParseException if fewer than 5 concrete inputs.
    tmp = list(shapes)
    tmp.append(False)
    tmp2 = list(concrete)
    if len(tmp2) < 5:
        raise ParseException("slow conv2d has less than 5 concrete inputs")
    tmp2[3] = tmp2[4]
    return conv_adapter(tuple(tmp), tuple(tmp2))


@register_adapter(
    ["convolution", "_convolution", "cudnn_convolution", "convolution_overrideable"]
)
def conv_adapter(
    shapes: tuple[Any, ...], concrete: tuple[Any, ...]
) -> tuple[tuple[Any], dict[Any, Any]]:
    # [FUNCTION DOC] HELPER: Adapter for convolution ops. With 4 shapes the conv is treated as
    # non-transposed; with more than 6 the transposed flag is read from index 6 and
    # normalized to a bool; any other count raises ParseException. For non-transposed
    # convs, computes the output spatial dims from input, kernel and the stride parsed
    # from concrete input 3, and passes the full output shape as the out_val keyword.
    # Returns (positional shapes, keyword args) for the flop counter function.
    tmp = list(shapes)
    if len(tmp) == 4:
        transposed = False
    elif len(tmp) > 6:
        transposed = bool(tmp[6])
        tmp[6] = transposed
    else:
        raise ParseException(f"Convolution has the wrong number of inputs: {len(tmp)}")

    kwargs: dict[Any, Any] = {}
    if not transposed:
        # calculate output shape if not transposed.
        def conv_out_dims(x: int, kernel: int, stride: int) -> int:
            # [FUNCTION DOC] HELPER: Output length along one axis: (x - kernel) // stride + 1 (no padding or dilation).
            return (x - kernel) // stride + 1

        stride = parse_list(concrete[3])
        inp = shapes[0]
        w = shapes[1]
        out_x_y = [conv_out_dims(*args) for args in zip(inp[2:], w[2:], stride)]
        out = [inp[0], w[0]] + out_x_y  # we only need the xy values
        kwargs["out_val"] = out

    return tuple(tmp), kwargs


def default_adapter(
    shapes: tuple[Any], concrete: tuple[Any]
) -> tuple[tuple[Any], dict[Any, Any]]:
    # [FUNCTION DOC] HELPER: Fallback adapter: passes the shapes through unchanged with no keyword args.
    return shapes, {}


@register_adapter("addmm")
def addmm_adapter(
    shapes: tuple[Any], concrete: tuple[Any]
) -> tuple[tuple[Any], dict[Any, Any]]:
    # [FUNCTION DOC] HELPER: Adapter for addmm: keeps only the first three input shapes, no kwargs.
    tmp = list(shapes)[:3]
    return tuple(tmp), {}


@register_adapter("bmm")
def bmm_adapter(
    shapes: tuple[Any], concrete: tuple[Any]
) -> tuple[tuple[Any], dict[Any, Any]]:
    # [FUNCTION DOC] HELPER: Adapter for bmm: keeps only the first two input shapes, no kwargs.
    tmp = list(shapes)
    return tuple(tmp[:2]), {}


@register_adapter("baddbmm")
def baddbmm_adapter(
    shapes: tuple[Any], concrete: tuple[Any]
) -> tuple[tuple[Any], dict[Any, Any]]:
    # [FUNCTION DOC] HELPER: Adapter for baddbmm: keeps only the first three input shapes, no kwargs.
    tmp = list(shapes)[:3]
    return tuple(tmp), {}


@register_adapter("mm")
def mm_adapter(
    shapes: tuple[Any], concrete: tuple[Any]
) -> tuple[tuple[Any], dict[Any, Any]]:
    # [FUNCTION DOC] HELPER: Adapter for mm: passes the shapes through unchanged, no kwargs.
    return shapes, {}


def _parse_kernel_name(name: str) -> str | None:
    # [FUNCTION DOC] HELPER: Maps a trace event name to an aten op name: strips the aten prefix
    # when present, otherwise guesses from substrings (conv, addmm, bmm, baddbmm, _mm);
    # returns None when nothing matches.
    # KNOWN ISSUE: the baddbmm branch is unreachable because a name containing baddbmm
    # also contains bmm, which is tested first.
    """
    parse the name of the kernel from the event name.
    """
    if name.startswith(ATEN_PREFIX):
        return name[len(ATEN_PREFIX) :]
    elif "conv" in name:
        return "convolution"
    elif "addmm" in name:
        return "addmm"
    elif "bmm" in name:
        return "bmm"
    elif "baddbmm" in name:
        return "baddbmm"
    elif "_mm" in name:
        return "mm"
    else:
        return None


def _calculate_flops(event: dict[str, Any]) -> int:
    # [FUNCTION DOC] HELPER: Estimates FLOPs for a trace event. Returns a nonzero kernel_flop arg
    # if already present. Otherwise parses the op name, looks the aten op up in the flop
    # registry, requires Input Dims and Concrete Inputs args, converts them with the
    # registered adapter (or the default one) and calls the registry flop function.
    # Returns 0 for unknown ops, missing args, or a ParseException (logged as warning).
    """
    This function has to parse the kernel name, which is error prone. There doesn't seem to be another solution that
    will support all the different backends that can generate kernels, so make sure to update this function when new
    ops and backends are desired.
    """
    name = event["name"]
    if "kernel_flop" in event["args"] and event["args"]["kernel_flop"] != 0:
        return event["args"]["kernel_flop"]
    op_name = _parse_kernel_name(name)
    if op_name is None:
        return 0

    op_obj = getattr(torch.ops.aten, op_name, None)
    if op_obj is None or op_obj not in flop_registry:
        return 0

    flop_function = flop_registry[op_obj]

    if "Input Dims" not in event["args"] or "Concrete Inputs" not in event["args"]:
        return 0
    input_shapes = event["args"]["Input Dims"]
    concrete = event["args"]["Concrete Inputs"]
    if op_name in adapters_map:
        try:
            args, kwargs = adapters_map[op_name](input_shapes, concrete)
        except ParseException as e:
            msg = f"Failed to parse {op_name} with {e}"
            log.warning(msg)
            return 0
    else:
        try:
            args, kwargs = default_adapter(input_shapes, concrete)
        except ParseException as e:
            msg = f"Failed to parse {op_name} with {e}"
            log.warning(msg)
            return 0
    return flop_function(*args, **kwargs)


def _get_size_from_string(type_string: str) -> int:
    # [FUNCTION DOC] HELPER: Returns the element size in bytes of the torch dtype with the given
    # attribute name, or 1 when torch has no such attribute.
    if not hasattr(torch, type_string):
        return 1
    else:
        return getattr(torch, type_string).itemsize


def _default_estimate_gb(event: dict[str, Any]) -> float:
    # [FUNCTION DOC] HELPER: Default memory estimate in GB: sums, over all inputs, the element
    # count (product of the flattened dims) times element size from the type name,
    # divided by 1e9. Requires Input Dims and Input type args (KeyError otherwise).
    sizes_and_types = zip(event["args"]["Input Dims"], event["args"]["Input type"])
    bw = 0
    for size, typ in sizes_and_types:
        isize = _get_size_from_string(typ)
        bw += isize * math.prod(pytree.tree_flatten(size)[0])
    return bw / 1e9


def _estimate_gb(event: dict[str, Any]) -> float:
    # [FUNCTION DOC] HELPER: Best-effort GB of memory traffic for an event. Uses a nonzero
    # kernel_num_gb arg if present; returns 0 without Input type / Input Dims args;
    # uses op-specific formulas for addmm, mm, baddbmm, bmm and convolution variants
    # (matmul formulas count reads of both operands plus the output, conv assumes
    # weights are read once) and falls back to the default input-size estimate.
    # Raises AssertionError when matmul operand shapes are inconsistent.
    # KNOWN ISSUE: the matmul formula multiplies the byte total by 2, so it is not a plain read count.
    """
    Our best effort to estimate the gb, should be refactored soon with MemoryCounter.
    """
    name = event["name"]
    if "kernel_num_gb" in event["args"] and event["args"]["kernel_num_gb"] != 0:
        return event["args"]["kernel_num_gb"]
    if "Input type" not in event["args"] or "Input Dims" not in event["args"]:
        return 0
    op_name = _parse_kernel_name(name)
    if op_name is None:
        return _default_estimate_gb(event)

    op_obj = getattr(torch.ops.aten, op_name, None)
    if op_obj is None:
        return _default_estimate_gb(event)

    if "Input Dims" not in event["args"] or "Concrete Inputs" not in event["args"]:
        return _default_estimate_gb(event)
    input_shapes = event["args"]["Input Dims"]

    # NOTE these will be refactored into a similar object to FlopCounter soon
    def mm_formula(M: int, N: int, K: int, size: int) -> int:
        # [FUNCTION DOC] HELPER: Bytes for an M x K by K x N matmul: twice the sum of the three
        # matrix element counts times the element size.
        return 2 * (M * K + N * K + M * N) * size

    if op_name == "addmm":
        add_in_size = math.prod(pytree.tree_flatten(input_shapes[0])[0])
        add_type_size = _get_size_from_string(event["args"]["Input type"][0])
        M = input_shapes[1][0]
        N = input_shapes[1][1]
        if input_shapes[1][1] != input_shapes[2][0]:
            raise AssertionError(
                f"Shape mismatch for addmm: {input_shapes[1][1]} != {input_shapes[2][0]}"
            )
        K = input_shapes[2][1]
        mul_type_size = _get_size_from_string(event["args"]["Input type"][1])
        return (mm_formula(M, N, K, mul_type_size) + add_in_size * add_type_size) / 1e9
    elif op_name == "mm":
        M = input_shapes[0][0]
        N = input_shapes[0][1]
        if input_shapes[0][1] != input_shapes[1][0]:
            raise AssertionError(
                f"Shape mismatch for mm: {input_shapes[0][1]} != {input_shapes[1][0]}"
            )
        K = input_shapes[1][1]
        type_size = _get_size_from_string(event["args"]["Input type"][0])
        return mm_formula(M, N, K, type_size) / 1e9
    elif op_name == "baddbmm":
        add_in_size = math.prod(pytree.tree_flatten(input_shapes[0])[0])
        add_type_size = _get_size_from_string(event["args"]["Input type"][0])
        B = input_shapes[0][0]
        M = input_shapes[1][1]
        N = input_shapes[1][2]
        K = input_shapes[2][2]
        mul_type_size = _get_size_from_string(event["args"]["Input type"][1])
        return (
            B * mm_formula(M, N, K, mul_type_size) + add_in_size * add_type_size
        ) / 1e9
    elif op_name == "bmm":
        add_in_size = math.prod(pytree.tree_flatten(input_shapes[0])[0])
        add_type_size = _get_size_from_string(event["args"]["Input type"][0])
        B = input_shapes[0][0]
        M = input_shapes[0][1]
        N = input_shapes[0][2]
        K = input_shapes[1][2]
        mul_type_size = _get_size_from_string(event["args"]["Input type"][1])
        return (
            B * mm_formula(M, N, K, mul_type_size) + add_in_size * add_type_size
        ) / 1e9
    elif op_name in [
        "convolution",
        "_convolution",
        "cudnn_convolution",
        "_slow_conv2d_forward",
    ]:
        concrete = event["args"]["Concrete Inputs"]

        def conv_out_dim(x: int, kernel: int, stride: int) -> int:
            # [FUNCTION DOC] HELPER: Output length along one axis: (x - kernel) // stride + 1 (no padding or dilation).
            return (x - kernel) // stride + 1

        stride = parse_list(
            concrete[3] if op_name != "_slow_conv2d_forward" else concrete[4]
        )
        inp = input_shapes[0]
        w = input_shapes[1]
        out_x_y = [conv_out_dim(*args) for args in zip(inp[2:], w[2:], stride)]
        out = [inp[0], w[0]] + out_x_y
        # each output element reads in * w * w chunk
        input_reads = out[0] * out[1] * out[2] * out[3] * inp[1] * w[2] * w[3]
        # Assume weights are in cache, so only read once
        weight_reads = w[0] * w[1] * w[2] * w[3]
        return (input_reads + weight_reads) / 1e9

    return _default_estimate_gb(event)


def _create_extern_mapping(
    data: dict[str, Any],
) -> defaultdict[int, list[dict[str, Any]]]:
    # [FUNCTION DOC] HELPER: Builds a mapping from External id to the list of cpu_op trace events
    # carrying it. Raises ParseException if two cpu_op events share an External id.
    """
    compute a mapping from external ids to non kernels, which contain the information we need to estimate flops etc
    """
    extern_mapping: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in data["traceEvents"]:
        if (
            "args" not in event
            or "External id" not in event["args"]
            or event["cat"] != "cpu_op"
        ):
            continue
        if len(extern_mapping[event["args"]["External id"]]) > 0:
            raise ParseException("duplicate external id in event")
        extern_mapping[event["args"]["External id"]].append(event)
    return extern_mapping


def _augment_trace_helper(data: dict[str, Any]) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Annotates a trace in place. For each kernel event with an External id,
    # finds the matching cpu op, computes FLOPs (falling back to the kernel event) and
    # estimated GB, and stores kernel_flop and kernel_num_gb in both events' args.
    # Kernels without External id are skipped (logged); a kernel without args raises
    # ParseException. KNOWN ISSUE: a kernel whose External id has no cpu_op yields an IndexError.
    extern_mapping = _create_extern_mapping(data)

    for event in data["traceEvents"]:
        if "cat" not in event or event["cat"] != "kernel":
            continue
        if "args" not in event:
            raise ParseException(f"kernel has no args: {event}")
        if "External id" not in event["args"]:
            event_str = f"kernel has no External id: {event}"
            log.info(event_str)
            continue

        external_op = extern_mapping[event["args"]["External id"]][0]
        flops = _calculate_flops(external_op)
        if flops == 0:
            flops = _calculate_flops(event)
        external_op["args"]["kernel_flop"] = flops
        external_op["args"]["kernel_num_gb"] = _estimate_gb(external_op)
        event["args"]["kernel_flop"] = external_op["args"]["kernel_flop"]
        event["args"]["kernel_num_gb"] = external_op["args"]["kernel_num_gb"]
    return data


_dtype_map: dict[str, torch.dtype] = {
    "float": torch.float,
    "float32": torch.float,
    "double": torch.double,
    "float64": torch.double,
    "int": torch.int,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int,
    "long": torch.long,
    "long int": torch.long,
    "signed char": torch.int8,
    "unsigned char": torch.uint8,
    "bool": torch.bool,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "c10::BFloat16": torch.bfloat16,
    "c10::Half": torch.float16,
    "c10::complex<float>": torch.complex64,
    "c10::complex<double>": torch.complex128,
}


@dataclass(frozen=True)
class KernelStats:
    # [CLASS DOC] Immutable (hashable) per-kernel-launch statistics: achieved flops and
    # bandwidth rates (per second and GB/s), latency in microseconds, and achieved
    # flops/bandwidth as percent of device peak. Stored in ordered sets per kernel name.
    flops: int
    bw: float
    latency: float  # us
    achieved_flops: float
    achieved_bandwidth: float


KernelNameMap = defaultdict[str, OrderedSet[KernelStats]]


@dataclass(frozen=False)
class Device:
    # [CLASS DOC] Mutable record of one device in a trace: name, index (trace device id),
    # optional datasheet DeviceInfo (None if unknown), and stats mapping kernel name
    # to the set of KernelStats observed.
    name: str
    index: int
    info: DeviceInfo | None
    stats: KernelNameMap

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns a short string with the device name, index and datasheet info.
        return f"Device({self.name}, {self.index}): {self.info}"


DeviceMap = dict[int, Device]
Table = tuple[list[str], dict[str, list[str]]]


class JsonProfile:
    # [CLASS DOC] Wrapper around a loaded Chrome/Perfetto JSON trace. State: path, data
    # (parsed JSON), events (the traceEvents list), benchmark_name, dtype (default dtype
    # used when an event gives none) and _devices (device id -> Device). Methods augment
    # the trace with flop/memory estimates, compute per-kernel stats, build text
    # reports (single or side-by-side diff with another profile), dump and combine.
    _devices: DeviceMap

    def __init__(
        self,
        path: str,
        benchmark_name: str | None = None,
        dtype: torch.dtype | str | None = None,
    ):
        # [FUNCTION DOC]
        # Loads the JSON trace at path (I/O; errors from open/json propagate), keeps its
        # events, and resolves dtype: None stays None, a torch dtype is kept, a string
        # is looked up in the dtype name table (unknown names give None). Then builds
        # the device table from the trace's deviceProperties.
        """
        Convenience class for running common operations on chrome/perfetto json traces.
        """
        self.path = path
        with open(path) as f:
            self.data = json.load(f)
            self.events = self.data["traceEvents"]
        self.benchmark_name = benchmark_name
        if dtype is None:
            self.dtype = None
        elif isinstance(dtype, torch.dtype):
            # pyrefly: ignore [bad-assignment]
            self.dtype = dtype
        else:
            # pyrefly: ignore [bad-assignment]
            self.dtype = _dtype_map.get(dtype)
        self._create_devices()

    def convert_dtype(self, event: dict[str, Any]) -> torch.dtype | None:
        # [FUNCTION DOC]
        # Picks a single dtype for an event. Without shape/type/concrete-input args, guesses
        # bfloat16 or float16 from the event name, else None. Otherwise requires the three
        # arg lists to have equal length (AssertionError) and be non-empty (RuntimeError),
        # picks the non-concrete input with the largest size, and maps its type name
        # through the dtype table (RuntimeError for unknown names).
        # KNOWN ISSUE: "size" is the sum of the dims, not their product, and is parsed with
        # parse_list, which expects a string while Input Dims entries are lists in JSON.
        """
        Each op has a list of dtypes for each input arg. We need to convert these into a single dtype for flop estimation.
        Issues:
         - converting the strings to concrete torch.dtypes
         - What if we have float32, float, float16 all in the inputs? Our choice is to use the largest buffer dtype.
        """

        if (
            "Input Dims" not in event["args"]
            or "Input type" not in event["args"]
            or "Concrete Inputs" not in event["args"]
        ):
            if "bfloat16" in event["name"]:
                return torch.bfloat16
            elif "float16" in event["name"]:
                return torch.float16
            else:
                return None

        input_sizes = event["args"]["Input Dims"]
        input_types = event["args"]["Input type"]
        concrete_inputs = event["args"]["Concrete Inputs"]
        if len(input_sizes) != len(input_types):
            raise AssertionError(
                f"input_sizes and input_types length mismatch: "
                f"{len(input_sizes)} != {len(input_types)}"
            )
        if len(input_types) != len(concrete_inputs):
            raise AssertionError(
                f"input_types and concrete_inputs length mismatch: "
                f"{len(input_types)} != {len(concrete_inputs)}"
            )

        if len(input_sizes) == 0:
            raise RuntimeError("Empty input_sizes and input_types")

        biggest_size = 0
        biggest_index = 0
        for i in range(len(input_sizes)):
            if concrete_inputs[i] != "":
                # concrete inputs are usually small tensors, so we can just skip
                continue
            my_size = input_sizes[i]
            total_size = sum(parse_list(my_size))
            if total_size > biggest_size:
                biggest_size = total_size
                biggest_index = i
        ret_type = input_types[biggest_index]
        if ret_type in _dtype_map:
            return _dtype_map[ret_type]
        raise RuntimeError(f"Unknown type: {ret_type}. Please add to _dtype_map.")

    def _create_devices(self) -> None:
        # [FUNCTION DOC] HELPER: Fills the device map from deviceProperties, looking up datasheet
        # info by name (logs when unsupported; the Device is still created with no info).
        self._devices = {}
        for dev in self.data["deviceProperties"]:
            name = dev["name"]
            device_info = lookup_device_info(name)

            if device_info is None:
                log.info(
                    "Unsupported device in profile: %s, please consider contributing to _device_mapping.",
                    name,
                )
            self._devices[dev["id"]] = Device(
                name, dev["id"], device_info, defaultdict(OrderedSet)
            )

    def calculate_flops(self, event: dict[str, Any]) -> int:
        # [FUNCTION DOC] Public wrapper returning the estimated FLOPs for a trace event.
        return _calculate_flops(event)

    def estimate_gb(self, event: dict[str, Any]) -> float:
        # [FUNCTION DOC] Public wrapper returning the estimated GB of memory traffic for an event.
        return _estimate_gb(event)

    def augment_trace(self) -> None:
        # [FUNCTION DOC] Annotates kernel and cpu op events of the loaded trace with kernel_flop
        # and kernel_num_gb (mutates the data in place; may raise ParseException).
        self.data = _augment_trace_helper(self.data)

    def _compute_stats(self) -> None:
        # [FUNCTION DOC] HELPER: For each kernel event on a known device, derives flops/s and GB/s
        # from kernel_flop / kernel_num_gb and the duration (AssertionError if the duration is
        # 0 while those args exist), and percent-of-peak values when datasheet info exists
        # (RuntimeError if no dtype can be determined; KeyError if the dtype has no peak
        # entry). Adds a KernelStats to the device's per-name set. Mutates device stats.
        # KNOWN ISSUE: the check for a name key looks in the event args instead of the
        # event itself, so events whose args lack a name are silently skipped.
        """populates the name -> stats map"""
        for event in self.events:
            if "cat" not in event or "args" not in event or event["cat"] != "kernel":
                continue
            if "device" not in event["args"]:
                continue
            dev_tmp = event["args"]["device"]
            if dev_tmp not in self._devices:
                continue
            dev = self._devices[event["args"]["device"]]

            dur = event["dur"]  # us
            if "kernel_flop" in event["args"]:
                if dur == 0:
                    raise AssertionError(
                        "Event duration is 0 for kernel_flop calculation"
                    )
                # 1,000,000us/s * flop / us
                op_flops = event["args"]["kernel_flop"] / (dur / 1e6)
            else:
                op_flops = 0

            if "kernel_num_gb" in event["args"]:
                if dur == 0:
                    raise AssertionError(
                        "Event duration is 0 for kernel_num_gb calculation"
                    )
                # 1,000,000us/s * gb  = gb/s
                op_gbps = event["args"]["kernel_num_gb"] / (dur / 1e6)
            else:
                op_gbps = 0

            if dev.info is not None:
                dtype = self.convert_dtype(event) or self.dtype
                if dtype is None:
                    raise RuntimeError(
                        "dtype is not found on tensor and default dtype is not set"
                    )
                achieved_flops = 100 * op_flops / (1e12 * dev.info.tops[dtype])
                achieved_bandwidth = 100 * op_gbps / dev.info.dram_bw_gbs
            else:
                achieved_flops = 0
                achieved_bandwidth = 0

            if "name" not in event["args"]:
                continue
            dev.stats[event["name"]].add(
                KernelStats(
                    flops=op_flops,
                    bw=op_gbps,
                    latency=dur,
                    achieved_bandwidth=achieved_bandwidth,
                    achieved_flops=achieved_flops,
                )
            )

    def _create_single_table(self, dev: Device) -> Table:
        # [FUNCTION DOC] HELPER: Builds (headers, rows) for one device: per kernel name, the launch
        # count and the averages (over launches with nonzero values) of flops, bandwidth,
        # latency, achieved flops percent and achieved bandwidth percent, formatted as
        # strings with four decimals. Raises AssertionError on an empty stats set.
        """Create a table with the devices mapped to indices."""
        headers = [
            "Kernel Name",
            "Kernel Count",
            "FLOPS",
            "Kernel Reads (GB)",
            "Dur (us)",
            "Achieved FLOPS %",
            "Achieved Bandwidth %",
        ]
        rows: dict[str, list[str]] = {}

        def safe_div_format(x: float, y: float) -> str:
            # [FUNCTION DOC] HELPER: Formats x divided by y with four decimals; returns "0.0" when y is zero.
            if y == 0:
                return "0.0"
            return f"{x / y:.4f}"

        for kernel_name, stats_set in dev.stats.items():
            ker_count = 0
            flops = 0
            flops_count = 0
            achieved_flops = 0.0
            bw = 0.0
            bw_count = 0
            achieved_bandwidth = 0.0
            latency = 0.0
            for stats in stats_set:
                if stats.flops != 0:
                    flops += stats.flops
                    achieved_flops += stats.achieved_flops
                    flops_count += 1
                if stats.bw != 0:
                    bw += stats.bw
                    achieved_bandwidth += stats.achieved_bandwidth
                    bw_count += 1
                latency += stats.latency
                ker_count += 1
            if ker_count == 0:
                raise AssertionError("ker_count is 0 when computing kernel stats")
            rows[kernel_name] = [
                str(ker_count),
                safe_div_format(flops, flops_count),
                safe_div_format(bw, bw_count),
                safe_div_format(latency, ker_count),
                safe_div_format(achieved_flops, flops_count),
                safe_div_format(achieved_bandwidth, bw_count),
            ]

        return headers, rows

    def _create_tables(self, devs: DeviceMap) -> dict[int, Table]:
        # [FUNCTION DOC] HELPER: Builds one stats table per device id.
        return {idx: self._create_single_table(dev) for idx, dev in devs.items()}

    def _combine_tables(
        self, table1: Table, table1_name: str, table2: Table, table2_name: str
    ) -> Table:
        # [FUNCTION DOC] HELPER: Merges two tables side by side keyed by kernel name; headers get the
        # given table name prefixes, and a kernel missing in one table is filled with
        # "Empty" placeholders. Returns the combined (headers, rows).
        new_headers = (
            ["Kernel Name"]
            + [f"{table1_name} {head}" for head in table1[0][1:]]
            + [f"{table2_name} {head}" for head in table2[0][1:]]
        )
        t1_length = len(table1[0][1:])
        t2_length = len(table2[0][1:])
        new_rows = {}

        for key, row1, row2 in zip_dicts(
            table1[1],
            table2[1],
            d1_default=["Empty"] * t1_length,
            d2_default=["Empty"] * t2_length,
        ):
            if row1 is None:
                raise AssertionError("row1 must not be None in table combination")
            if row2 is None:
                raise AssertionError("row2 must not be None in table combination")
            new_rows[key] = row1 + row2
        return new_headers, new_rows

    def report(
        self, other: Optional["JsonProfile"] = None, name_limit: int = 40
    ) -> str:
        # [FUNCTION DOC]
        # Computes stats and returns a text report: one table per device, each titled by
        # the device. With another profile, tables of both are combined side by side using
        # their benchmark names (or "Table 1"/"Table 2"); both must have identical device
        # ids (AssertionError otherwise). Kernel names are truncated to name_limit chars.
        def create_ret(
            table_headers: list[str], table_rows: dict[str, list[str]]
        ) -> str:
            # [FUNCTION DOC] HELPER: Renders a headers/rows table as text with truncated kernel names.
            table_flattened = [
                [kernel_name[:name_limit], *kernel_vals]
                for kernel_name, kernel_vals in table_rows.items()
            ]
            return tabulate_2d(table_flattened, headers=table_headers)

        if other is not None:
            self._compute_stats()
            other._compute_stats()

            self_tables = self._create_tables(self._devices)
            other_tables = self._create_tables(other._devices)

            self_name = (
                self.benchmark_name if self.benchmark_name is not None else "Table 1"
            )
            other_name = (
                other.benchmark_name if other.benchmark_name is not None else "Table 2"
            )

            ret = []
            if self._devices.keys() != other._devices.keys():
                raise AssertionError(
                    f"Device key mismatch: {self._devices.keys()} != {other._devices.keys()}"
                )
            for device_idx, t1, t2 in zip_dicts(
                self_tables, other_tables, d1_default=None, d2_default=None
            ):
                if t1 is None:
                    raise AssertionError("t1 must not be None for device comparison")
                if t2 is None:
                    raise AssertionError("t2 must not be None for device comparison")
                table_headers, table_rows = self._combine_tables(
                    t1, self_name, t2, other_name
                )
                tab_string = create_ret(table_headers, table_rows)
                # pyrefly: ignore [bad-argument-type]
                ret.append(f"{self._devices[device_idx]}:\n{tab_string}")
            return "\n".join(ret)
        self._compute_stats()

        self_tables = self._create_tables(self._devices)

        ret = []
        for idx, table in self_tables.items():
            table_headers, table_rows = table
            tab_string = create_ret(table_headers, table_rows)
            # pyrefly: ignore [bad-argument-type]
            ret.append(f"{self._devices[idx]}:\n{tab_string}")
        return "\n".join(ret)

    def dump(self, out: str) -> None:
        # [FUNCTION DOC] Writes the (possibly augmented) trace data as JSON to the file at out (I/O).
        with open(out, "w") as f:
            json.dump(self.data, f)

    def combine_with(self, other: "JsonProfile") -> "JsonProfile":
        # [FUNCTION DOC]
        # Returns a new JsonProfile whose trace events are the concatenation of both
        # profiles, device properties are merged without duplicate ids, and other top-level
        # keys come from self. Round-trips through a temporary JSON file (deleted afterwards);
        # the benchmark name joins both names and dtype is self's or else other's.
        """
        Combine this profile with another profile by merging their trace events.
        Returns a new JsonProfile object with combined data.
        """
        # Create a new combined data structure
        combined_data = {
            "traceEvents": self.data["traceEvents"] + other.data["traceEvents"],
            "deviceProperties": self.data.get("deviceProperties", []),
        }

        # Merge device properties, avoiding duplicates
        other_device_props = other.data.get("deviceProperties", [])
        existing_device_ids = OrderedSet(
            [dev["id"] for dev in combined_data["deviceProperties"]]
        )

        for device_prop in other_device_props:
            if device_prop["id"] not in existing_device_ids:
                combined_data["deviceProperties"].append(device_prop)

        # Copy any other top-level properties from the first profile
        for key, value in self.data.items():
            if key not in combined_data:
                combined_data[key] = value

        import os

        # Create a temporary file to write the combined data
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            json.dump(combined_data, tmp_file)
            tmp_path = tmp_file.name

        try:
            # Create new JsonProfile from the combined data
            combined_profile = JsonProfile(
                tmp_path,
                benchmark_name=f"{self.benchmark_name or 'Profile1'}_+_{other.benchmark_name or 'Profile2'}",
                dtype=self.dtype or other.dtype,
            )
            return combined_profile
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)


class ParseException(RuntimeError):
    # [CLASS DOC] Error raised when trace event data cannot be parsed into op arguments
    # (bad input counts, duplicate external ids, kernels without args).
    pass


def main() -> None:
    # [FUNCTION DOC]
    # Command line entry point. Options: diff (two traces plus names and dtype; prints a
    # side-by-side report), analysis (one trace; prints report), augment_trace (writes an
    # annotated trace), combine (merges two or more traces into the last-named output file;
    # prints an error message and returns if fewer than two inputs), name_limit. Prints to
    # stdout and writes files; multiple modes can run in one invocation.
    """
    Main function for the profile analysis script.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diff",
        nargs=5,
        metavar=(
            "input_file1",
            "name1",
            "input_file2",
            "name2",
            "dtype",
        ),
        help="Two json traces to compare with, specified as <file1> <name1> <file2> <name2> <dtype>",
    )
    parser.add_argument(
        "--name_limit",
        type=int,
        help="the maximum name size in the final report",
    )
    parser.add_argument(
        "--augment_trace",
        "-a",
        nargs=3,
        metavar=("input_file", "output_file", "dtype"),
        help="Augment a trace with inductor meta information. Provide input and output file paths.",
    )
    parser.add_argument(
        "--analysis",
        nargs=2,
        metavar=("input_file", "dtype"),
        help="Run analysis on a single trace, specified as <file> <dtype>",
    )
    parser.add_argument(
        "--combine",
        nargs="+",
        metavar=("input_files", "output_file"),
        help="Combine multiple profiles into a single profile by merging trace events. Specify as <input_file1> \
<input_file2> [input_file3 ...] <output_file>. The last argument is the output file, all preceding arguments are \
input files to combine.",
    )
    args = parser.parse_args()

    if args.diff:
        p1 = JsonProfile(args.diff[0], args.diff[1], dtype=args.diff[4])
        p1.augment_trace()
        p2 = JsonProfile(args.diff[2], args.diff[3], dtype=args.diff[4])
        p2.augment_trace()
        if args.name_limit:
            print(p1.report(p2, name_limit=args.name_limit))
        else:
            print(p1.report(p2))
    if args.analysis:
        p1 = JsonProfile(
            args.analysis[0],
            dtype=args.analysis[1],
        )
        p1.augment_trace()
        if args.name_limit:
            print(p1.report(name_limit=args.name_limit))
        else:
            print(p1.report())
    if args.augment_trace:
        p = JsonProfile(args.augment_trace[0], dtype=args.augment_trace[2])
        p.augment_trace()
        p.dump(args.augment_trace[1])
    if args.combine:
        input_files = args.combine[:-1]  # All arguments except the last one
        output_file = args.combine[-1]  # Last argument is the output file

        if len(input_files) < 2:
            print("Error: At least 2 input files are required for combining")
            return

        # Load the first profile
        combined = JsonProfile(input_files[0], dtype=None)

        # Iteratively combine with all other profiles
        for input_file in input_files[1:]:
            profile = JsonProfile(input_file, dtype=None)
            combined = combined.combine_with(profile)

        combined.dump(output_file)
        print(f"Successfully combined {', '.join(input_files)} into {output_file}")


if __name__ == "__main__":
    main()
