# mypy: allow-untyped-defs
# [MODULE DOC] Helpers for Triton kernel codegen in Inductor: map kernel argument objects to Triton
# signature type strings, build the signature metadata mapping, compute alignment / equal-to-1 /
# 2GB-pointer-range attribute descriptors, and pick tile hints. Reads the current graph through
# the virtualized V handle. No import-time side effects beyond imports.
from typing import Any

import sympy

import torch
from torch.utils._sympy.symbol import symbol_is_type, SymT

from .. import config
from ..runtime.hints import AttrsDescriptorWrapper, DeviceProperties
from ..utils import (
    _type_of,
    device_supports_fp64,
    expr_fits_within_32bit,
    triton_version_uses_attrs_dict,
)
from ..virtualized import V
from .common import (
    ArgName,
    ConstexprArg,
    KernelArgType,
    SizeArg,
    TensorArg,
    TMADescriptorArg,
    WorkspaceArg,
)


def should_unwrap_unspec_arg(name: str):
    # [FUNCTION DOC] Decide whether an unspecialized 0-d tensor argument named name should be passed
    # to the kernel as a scalar. Returns False if the name is not an unspec arg; True on any
    # non-CPU device; on CPU True only when the buffer is not in the graph's mutated buffers.
    if V.graph.is_unspec_arg(name):
        # Unwrap on all devices except CPU
        if V.graph.get_current_device_or_throw().type != "cpu":
            return True
        # Only unwrap on CPU if the input is not used as an output
        if name not in V.graph.mutated_buffers:
            return True
    return False


def use_uint8_triton_storage_for_cuda_float8_e4m3fn(
    dtype: torch.dtype,
    arg_name: str | None = None,
    *,
    device: torch.device | None = None,
) -> bool:
    # [FUNCTION DOC] Return True if a float8_e4m3fn pointer argument should be typed as raw uint8
    # storage in the Triton signature. Requires CUDA (not HIP), dtype float8_e4m3fn, an input
    # pointer name (when arg_name is given) and compute capability below 89. The device defaults
    # to the current graph device; if there is no graph device, returns False.
    # Triton rejects fp8e4nv pointer types before sm89, but eager CUDA can
    # still dequantize float8_e4m3fn values by treating storage as raw bytes.
    if dtype != torch.float8_e4m3fn or torch.version.hip is not None:
        return False
    if arg_name is not None and not arg_name.startswith("in_ptr"):
        return False

    if device is None:
        try:
            device = V.graph.get_current_device_or_throw()
        except AttributeError:
            return False

    if device.type != "cuda":
        return False

    return DeviceProperties.create(device).cc < 89


def signature_of(arg: KernelArgType, *, size_dtype: str | None) -> str:
    # [FUNCTION DOC] Return the Triton signature type string for one kernel argument.
    # Tensors give pointer types (scalars if unwrapped unspec args, uint8 pointer for fp8 on old
    # GPUs); size args give constexpr, fp32/fp64, i1, i32 or i64 depending on value and size_dtype
    # (when size_dtype is None, i32 is chosen if provably fitting and a guard is added to the graph
    # sizevars); workspace args use their dtype; TMA descriptors give nvTmaDesc or a tensordesc
    # string; constexpr args give constexpr.
    # Raises NotImplementedError for an unknown size_dtype or argument type, and AssertionError for
    # a TMA descriptor with missing block_shape/dtype or unexpected api_type.
    if isinstance(arg, TensorArg):
        typ = _type_of(arg.dtype)
        if should_unwrap_unspec_arg(arg.buffer):
            # had unwrapped 0d tensor as scalar
            new_typ = typ.lstrip("*")
            if new_typ in ["fp16", "bf16"]:
                return "fp32"
            else:
                return new_typ
        elif use_uint8_triton_storage_for_cuda_float8_e4m3fn(arg.dtype, arg.name):
            return "*u8"
        else:
            return typ
    if isinstance(arg, SizeArg):
        if arg.expr is None:
            if triton_version_uses_attrs_dict():
                # In newer versions of Triton, the signature includes "None" args
                # and their type is marked as "constexpr"
                return "constexpr"
            else:
                # In older versions of Triton...
                # From triton/runtime/jit.py
                # `None` is nullptr.  Implicitly convert to *i8.
                return "*i8"
        elif _arg_equals_1(arg) and triton_version_uses_attrs_dict():
            # In new versions of Triton, if we have an equal-to-1 arg that's marked as a constant,
            # it should be marked as "constexpr" in the signature.
            return "constexpr"
        elif isinstance(arg.expr, (float, sympy.Float)):
            # Python floats are natively fp64, so use fp64 to preserve precision
            if config._use_fp64_for_unbacked_floats and device_supports_fp64(
                V.graph.current_device
            ):
                return "fp64"
            return "fp32"
        elif isinstance(arg.expr, sympy.Symbol) and symbol_is_type(
            arg.expr, (SymT.UNBACKED_FLOAT)
        ):
            # Unbacked floats from .item() should preserve fp64 precision
            if config._use_fp64_for_unbacked_floats and device_supports_fp64(
                V.graph.current_device
            ):
                return "fp64"
            return "fp32"
        elif isinstance(arg.expr, bool):
            return "i1"

        # if this is an integer
        if size_dtype == "tl.int32":
            return "i32"
        elif size_dtype == "tl.int64":
            return "i64"
        elif size_dtype is None:
            # no hint: we'll see if we know that this is a 32-bit int, and guard if possible.
            int_max = torch.iinfo(torch.int32).max
            if expr_fits_within_32bit(arg.expr):
                V.graph.sizevars.check_leq(arg.expr, int_max)
                return "i32"
            else:
                return "i64"
        else:
            raise NotImplementedError(f"unhandled size_dtype {size_dtype}")
    if isinstance(arg, WorkspaceArg):
        return _type_of(arg.dtype)
    if isinstance(arg, TMADescriptorArg):
        if arg.api_type == "experimental":
            return "nvTmaDesc"
        else:
            # https://github.com/triton-lang/triton/blob/9695baed9b46cf957e08b157bb4133f4a4b331c5/python/triton/runtime/jit.py#L360-L363
            if arg.api_type != "stable":
                raise AssertionError(
                    f"expected api_type == 'stable', got {arg.api_type}"
                )
            if arg.block_shape is None:
                raise AssertionError("expected block_shape to not be None")
            if arg.dtype is None:
                raise AssertionError("expected dtype to not be None")
            inner = _type_of(arg.dtype)[1:]  # strip the `*`: *fp32 -> fp32
            return f"tensordesc<{inner}{list(arg.block_shape)}>"
    if isinstance(arg, ConstexprArg):
        return "constexpr"
    raise NotImplementedError(f"unhandled {type(arg)}: {arg}")


def non_constexpr_signature(signature):
    # [FUNCTION DOC] Return a new list of the signature entries that are not ConstexprArg, in order.
    new_signature = []
    for arg in signature:
        if not isinstance(arg, ConstexprArg):
            new_signature.append(arg)

    return new_signature


def select_tile_hint(size_hints, signature):
    # [FUNCTION DOC] Choose a TileHint for a pointwise kernel: None unless there are exactly two size
    # hints; SQUARE if the signature has four non-constexpr args, otherwise DEFAULT.
    """Pick TileHint.SQUARE vs TileHint.DEFAULT for 2D pointwise; None for 1D.

    SQUARE applies when the kernel has exactly 2 size hints and 4 non-constexpr
    signature args (input, output, and 2 numel args -> a square 2D tile).
    """
    from torch._inductor.runtime.hints import TileHint

    if len(size_hints) != 2:
        return None
    if len(non_constexpr_signature(signature)) == 4:
        return TileHint.SQUARE
    return TileHint.DEFAULT


def signature_to_meta(
    signature: list[KernelArgType],
    *,
    size_dtype: str | None,
    argdefs: list[ArgName],
    indices: list[int] | None = None,
    is_template: bool = False,
) -> dict[str, str]:
    # [FUNCTION DOC] Build the mapping from argument name (taken from argdefs at the given indices,
    # default all) to its Triton signature type string via signature_of. is_template disables the
    # forced 64-bit dtype for symbolic size args. Pure apart from reading graph state.
    if indices is None:
        indices = list(range(len(signature)))

    def _decide_tl_dtype(arg):
        # [FUNCTION DOC] HELPER: Return the size dtype for one arg: 64-bit for size args named with
        # the ks prefix when block pointers are off and not a template, else the given size_dtype.
        # Even if the ks0 symbol itself is within tl.int32 range, it's
        # risky to use tl.int32 dtype since we may have ks0*ks1 later
        # for kernels like torch.mean when dynamic shape is enabled.
        #
        # Check config.triton.use_block_ptr, since Triton block pointer
        # does not support 64bit indexing:
        # https://gist.github.com/shunting314/6a41c776171720ce4561f202dcde0ad6
        #
        # If the triton metadata is for a template, don't use tl.int64 index.
        # Templates like flex attention/decoding uses block pointers which
        # does not support 64 bit indexing.
        if (
            not config.triton.use_block_ptr
            and not is_template
            and isinstance(arg, SizeArg)
            and arg.name.startswith("ks")
        ):
            return "tl.int64"
        return size_dtype

    return {
        argdefs[i].name: signature_of(arg, size_dtype=_decide_tl_dtype(arg))
        for i, arg in zip(indices, signature)
    }


def _get_buffer_layout(buf_name: str) -> "torch._inductor.ir.Layout":
    # [FUNCTION DOC] HELPER: Look up the layout of buf_name from the scheduler if one exists, else from
    # the graph buffer; if no buffer is found it must be the current kernel's output node, whose layout
    # is returned. Raises AssertionError if that name does not match.
    """Get the layout for a buffer, handling both scheduler buffers and graph inputs."""
    if V.graph.scheduler:
        layout = V.graph.scheduler.get_buffer_layout(buf_name)
    else:
        buffer = V.graph.try_get_buffer(buf_name)
        # output arg
        if not buffer:
            if buf_name != V.kernel.output_node.name:
                raise AssertionError(
                    f"expected buf_name == output_node.name, got {buf_name}"
                )
            layout = V.kernel.output_node.layout
        else:
            layout = buffer.get_layout()
    return layout


def is_unaligned_buffer_name(buf_name: str) -> bool:
    # [FUNCTION DOC] Return True if the named buffer may be unaligned: True if recorded in the graph's
    # unaligned buffers; False for graph inputs and constants not so recorded; otherwise True only for
    # a non-owning layout whose alignment cannot be guarded.
    if buf_name in V.graph.unaligned_buffers:
        return True

    if buf_name in V.graph.graph_inputs:
        # See Note: [Input Alignment handling in Inductor]
        # For graph inputs that is not recorded in V.graph.unaligned_buffers,
        # we know for sure the tensor is aligned.
        return False

    if buf_name in V.graph.constants:
        # all constants are assumed to be aligned
        return False

    layout = _get_buffer_layout(buf_name)
    if isinstance(layout, torch._inductor.ir.NonOwningLayout):
        return not layout.maybe_guard_aligned()
    else:
        return False


def is_unaligned_buffer(arg: TensorArg):
    # [FUNCTION DOC] Return whether the buffer behind a tensor argument may be unaligned.
    return is_unaligned_buffer_name(arg.buffer)


def _arg_equals_1(arg: KernelArgType) -> bool:
    # [FUNCTION DOC] HELPER: True if arg is a size arg with an integer expression statically known
    # to equal 1.
    return (
        isinstance(arg, SizeArg)
        and isinstance(arg.expr, (int, sympy.Integer))
        and V.graph.sizevars.statically_known_equals(arg.expr, 1)  # type: ignore[arg-type]
    )


def equal_1_arg_indices(
    args: list[KernelArgType],
    *,
    indices: list[int] | None = None,
) -> tuple[int, ...]:
    # [FUNCTION DOC] Return the tuple of indices (from indices, default positions in args) whose
    # argument is a size arg statically equal to 1.
    if indices is None:
        indices = list(range(len(args)))

    equal_to_1 = tuple(i for i, arg in zip(indices, args) if _arg_equals_1(arg))

    return equal_to_1


def _is_tensor_within_2gb(arg: TensorArg) -> bool:
    # [FUNCTION DOC] HELPER: Return True if the tensor's storage size in bytes is statically known to
    # be at most 2**31 - 1. Uses the graph input layout when available, else the buffer layout. Any
    # exception during analysis is swallowed and yields False.
    """Check if a tensor argument's storage is provably within 2GB.

    Mirrors HIPBackend.is_within_2gb() but uses compile-time symbolic analysis
    instead of runtime tensor inspection. This enables canonicalize_pointers to
    decompose pointer arithmetic into (splat(base), offset) form for buffer ops.
    """
    MAX_BYTES = 2**31 - 1
    try:
        # Graph inputs aren't tracked by the scheduler; get their layout
        # from the graph_inputs dict to avoid KeyError in get_buffer_layout.
        if arg.buffer in V.graph.graph_inputs:
            inp = V.graph.graph_inputs[arg.buffer]
            if hasattr(inp, "get_layout"):
                layout = inp.get_layout()
            else:
                return False
        else:
            layout = _get_buffer_layout(arg.buffer)
        storage_bytes = layout.storage_size() * arg.dtype.itemsize
        return V.graph.sizevars.statically_known_true(storage_bytes <= MAX_BYTES)
    except Exception:
        return False


def config_of(
    args: list[KernelArgType],
    *,
    indices: list[int] | None = None,
    pointer_range_override: tuple[int, ...] | None = None,
    skip_cpp_wrapper_input_tensor_alignment: bool = False,
) -> Any:
    # [FUNCTION DOC] Build the Triton attribute descriptor (AttrsDescriptorWrapper) for kernel args:
    # indices divisible by 16 (only if config divisible_by_16), indices equal to 1, and on HIP the
    # indices of tensors provably within 2GB (or pointer_range_override if given). With
    # skip_cpp_wrapper_input_tensor_alignment, graph-input tensors not listed in inputs_to_check are
    # treated as aligned-checkable only under cpp wrapper non-AOT mode.
    if indices is None:
        indices = list(range(len(args)))

    def is_aligned(x: KernelArgType, alignment: int, include_tensor: bool) -> bool:
        # [FUNCTION DOC] HELPER: True if arg x is known to be a multiple of alignment: tensors by byte
        # offset and buffer alignment (only if include_tensor), size args by static divisibility
        # (False for None/float/bool/load_seed_offset), workspaces always, TMA/constexpr never.
        # Raises NotImplementedError for unknown arg types.
        """
        Roughly follow triton code here:
        https://github.com/triton-lang/triton/blob/5282ed890d453e10b9ee30076ef89115dd197761/python/triton/runtime/jit.py#L208-L222
        """
        if isinstance(x, TensorArg):
            if include_tensor:
                offset_aligned = V.graph.sizevars.statically_known_multiple_of(
                    x.offset * x.dtype.itemsize,
                    alignment,  # type: ignore[arg-type]
                )
                return offset_aligned and not is_unaligned_buffer(x)
            else:
                return False
        if isinstance(x, SizeArg):
            # TODO(voz): These are kinda redundant, if we can solve out statically_known_multiple_of with
            # _maybe_evaluate_static...
            if x.name.startswith("load_seed_offset"):
                return False
            if x.expr is None:
                return False
            if isinstance(x.expr, (float, bool)):
                return False
            return V.graph.sizevars.statically_known_multiple_of(x.expr, alignment)  # type: ignore[arg-type]
        if isinstance(x, WorkspaceArg):
            # We allocate the workspace ourselves, so it is always aligned
            return True
        if isinstance(x, (TMADescriptorArg, ConstexprArg)):
            return False
        raise NotImplementedError(f"unhandled {type(x)}: {x}")

    def include_tensor_alignment(arg: KernelArgType) -> bool:
        # [FUNCTION DOC] HELPER: Whether tensor alignment should be considered for arg; False only for
        # non-checked graph inputs under cpp wrapper (non-AOT) when skipping is requested.
        if (
            not skip_cpp_wrapper_input_tensor_alignment
            or not V.graph.cpp_wrapper
            or V.graph.aot_mode
            or not isinstance(arg, TensorArg)
            or arg.buffer not in V.graph.graph_inputs
        ):
            return True

        try:
            input_idx = V.graph.graph_input_names.index(arg.buffer)
        except ValueError:
            return True
        return input_idx not in (V.graph.inputs_to_check or ())

    if config.triton.divisible_by_16:
        divisible_by_16 = tuple(
            i
            for i, arg in zip(indices, args)
            if is_aligned(
                arg,
                alignment=16,
                include_tensor=include_tensor_alignment(arg),
            )
        )
    else:
        divisible_by_16 = ()

    equal_to_1 = equal_1_arg_indices(args, indices=indices)

    # On AMD/HIP, tag tensor args whose storage fits in 2GB so Triton
    # can use 32-bit pointer offsets and emit buffer load/store ops.
    if pointer_range_override is not None:
        pointer_range_32 = pointer_range_override
    elif torch.version.hip is not None:
        pointer_range_32 = tuple(
            i
            for i, arg in zip(indices, args)
            if isinstance(arg, TensorArg) and _is_tensor_within_2gb(arg)
        )
    else:
        pointer_range_32 = ()

    # pyrefly: ignore [bad-argument-count, bad-argument-type]
    return AttrsDescriptorWrapper(divisible_by_16, equal_to_1, pointer_range_32)
