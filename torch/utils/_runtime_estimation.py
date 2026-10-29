# [MODULE DOC] Heuristic runtime estimation for aten ops on a GPU: compute time from flop
# counts (flop_registry) and peak TFLOPS, and memory transfer time from tensor byte sizes
# and DRAM bandwidth. Optionally pins hardware numbers to a named datasheet device for
# deterministic estimates. Defines OrderedSets of view ops, tensor-creation ops and their
# union (_IGNORE_OPS) that need no benchmarking. Import-time side effects: none beyond
# building these constants.
import torch
from torch._inductor.analysis.device_info import datasheet_dram_bw_gbs, datasheet_tops
from torch._inductor.utils import get_device_tflops, get_gpu_dram_gbps
from torch.fx.experimental.symbolic_shapes import (
    optimization_hint,
    statically_known_true,
)
from torch.utils._ordered_set import OrderedSet

from .flop_counter import flop_registry


aten = torch.ops.aten

_FLOAT_TYPES = OrderedSet(
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ]
)

# No fall-back kernel needed/exists for view ops
_VIEW_OPS = OrderedSet(
    [
        aten.lift_fresh,
        aten.t,
        aten.transpose,
        aten.view,
        aten.detach,
        aten._unsafe_view,
        aten.split,
        aten.adjoint,
        aten.as_strided,
        aten.diagonal,
        aten.expand,
        aten.expand_as,
        aten.movedim,
        aten.permute,
        aten.select,
        aten.squeeze,
        aten.mT,
        aten.mH,
        aten.real,
        aten.imag,
        aten.view_as,
        aten.unflatten,
        aten.unfold,
        aten.unbind,
        aten.unsqueeze,
        aten.vsplit,
        aten.hsplit,
        aten.split_with_sizes,
        aten.swapaxes,
        aten.swapdims,
        aten.chunk,
    ]
)
# We can ignore benchmarking tensor create ops
_CREATE_OPS = OrderedSet(
    [
        aten.randint,
        aten.randn,
        aten.rand,
        aten.randn_like,
        aten.rand_like,
        aten.randint_like,
        aten.arange,
        aten.ones_like,
        aten.zeros_like,
    ]
)

_IGNORE_OPS = _VIEW_OPS | _CREATE_OPS


def flops_to_ns(
    flops: float | int, dtype: "torch.dtype", gpu_type: str | None = None
) -> float:
    # [FUNCTION DOC] Convert a flop count to estimated nanoseconds for the given dtype.
    # Uses datasheet TFLOPS for gpu_type if given (tf32 chosen from the matmul fp32 precision
    # setting), else the current device's TFLOPS. Assumes 75 percent of peak and halves flops
    # to MACs. Returns 0.0 if peak is zero. Raises ValueError if gpu_type has no datasheet entry.
    """Convert a FLOPs count to estimated nanoseconds on the GPU.

    Uses 75% of theoretical peak and converts FLOPs to MACs (divide by 2).

    If ``gpu_type`` names a device in the datasheet, its pinned datasheet
    TFLOPS are used instead of querying the current device, making the
    estimate deterministic and hardware-independent.
    """
    if gpu_type is not None:
        is_tf32 = torch.backends.cuda.matmul.fp32_precision == "tf32"
        tflops = datasheet_tops(dtype, is_tf32=is_tf32, device_name=gpu_type)
        if tflops is None:
            raise ValueError(
                f"gpu_type {gpu_type!r} has no datasheet entry for {dtype}"
            )
    else:
        tflops = get_device_tflops(dtype)
    peak_gpu_flops = tflops * 1e12
    if peak_gpu_flops == 0:
        return 0.0
    macs = flops / 2
    return (macs / (0.75 * peak_gpu_flops)) * 1e9


def get_compute_time(
    func_packet, args, kwargs, out, out_dtypes, node_meta=None, gpu_type=None
) -> float:  # type: ignore[no-untyped-def]
    # [FUNCTION DOC] Estimate compute time in ns of an aten op. If the op packet is in
    # flop_registry: requires exactly one entry in out_dtypes (raises AssertionError
    # otherwise), pops it (mutating the set), calls the registered flop formula with args,
    # kwargs, out_val and optional _node_meta, and converts flops via flops_to_ns.
    # Ops not in the registry return 0.0.
    """
    Estimates the compute time of an aten operator.

    Args:
        func_packet: The operator overload packet.
        args: The arguments to the operator.
        kwargs: The keyword arguments to the operator.
        out: The output of the operator.
        out_dtypes: The output data types.
        node_meta: Optional FX node meta dict. Passed through to the flop
            formula as ``_node_meta`` kwarg so formulas can read annotations
            like ``sparsity_hint``.
        gpu_type: Optional datasheet device name to pin the peak FLOPS to
            instead of querying the current device.

    Returns:
        float: The estimated compute time in nanoseconds.
    """
    if func_packet in flop_registry:
        if len(out_dtypes) != 1:
            raise AssertionError(
                f"Only support single out dtype got {out_dtypes} for {func_packet}"
            )
        dtype = out_dtypes.pop()
        flop_count_func = flop_registry[func_packet]
        extra_kwargs = {}
        if node_meta is not None:
            extra_kwargs["_node_meta"] = node_meta
        flop_count = flop_count_func(*args, **kwargs, out_val=out, **extra_kwargs)
        return flops_to_ns(flop_count, dtype, gpu_type=gpu_type)
    return 0.0


def get_num_bytes(t: torch.Tensor) -> int:
    # [FUNCTION DOC] Bytes actually touched by tensor t: product of sizes over dims whose
    # stride is not statically known to be zero (broadcast dims count as one element), with
    # symbolic sizes resolved via optimization_hint (fallback 0), times element size.
    """
    Calculates the memory consumption of a tensor.

    Args:
        t (torch.Tensor): The input tensor.

    Returns:
        int: The memory consumption of the tensor in bytes.
    """
    real_numel = 1
    for size, stride in zip(t.shape, t.stride()):
        # For dims with stride=0 (expanded/broadcast), only 1 element accessed
        if not statically_known_true(stride == 0):
            real_numel *= optimization_hint(size, fallback=0)

    return real_numel * t.element_size()


def get_transfer_time(flat_args_kwargs, flat_outs, gpu_type=None) -> float:  # type: ignore[no-untyped-def]
    # [FUNCTION DOC] Estimate memory transfer time in ns: total bytes of all tensors among
    # the flat inputs plus flat outputs divided by DRAM bandwidth (GB/s equals bytes per ns).
    # Bandwidth comes from the datasheet for gpu_type, else from the current device.
    # Raises ValueError if gpu_type is not in the datasheet. Non-tensor entries are ignored.
    """
    Estimates the memory transfer time of input and output tensors.

    Args:
        flat_args_kwargs (List[torch.Tensor]): The flat list of arguments and keyword arguments.
        flat_outs (List[torch.Tensor]): The flat list of outputs.
        gpu_type: Optional datasheet device name to pin the DRAM bandwidth to
            instead of querying the current device.

    Returns:
        float: The estimated memory transfer time in nanoseconds.
    """
    if gpu_type is not None:
        gpu_memory_bandwidth = datasheet_dram_bw_gbs(gpu_type)
        if gpu_memory_bandwidth is None:
            raise ValueError(f"gpu_type {gpu_type!r} not found in datasheet")
    else:
        gpu_memory_bandwidth = get_gpu_dram_gbps()
    read_bytes = sum(
        get_num_bytes(t) for t in flat_args_kwargs if isinstance(t, torch.Tensor)
    )
    write_bytes = sum(
        get_num_bytes(t) for t in flat_outs if isinstance(t, torch.Tensor)
    )
    counted_bytes = read_bytes + write_bytes
    # The GPU memory bandwidth is in GB/s so the transfer time is in nanoseconds
    transfer_time = counted_bytes / gpu_memory_bandwidth
    return transfer_time
