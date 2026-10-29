# mypy: allow-untyped-defs
# [MODULE DOC] Support utilities for DebugMode: string formatting of tensors, shapes, meshes and
# placements, a weak-reference tensor id tracker, two tensor hash functions (norm and hash_tensor),
# stack trace capture and filtering, op naming, and runners for registered dispatch hooks.
# Module-level state: flags for inductor benchmarking and triton output recording, optional triton
# hash functions, and three lists of registered dispatch hooks (record, log, pre-log). Importing
# has no side effects beyond defining these globals.
"""
Utility functions for DebugMode: tensor formatting, hashing, stack traces, and hook runners.
"""

import inspect
import os
import traceback
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch._subclasses.fake_tensor import is_fake_tensor
from torch.fx.graph import _parse_stack_trace
from torch.utils._dtype_abbrs import dtype_abbrs
from torch.utils._pytree import tree_map
from torch.utils._traceback import CapturedTraceback
from torch.utils.weak import WeakIdRef


if TYPE_CHECKING:
    from torch.utils._debug_mode._calls import _DebugCall


REDISTRIBUTE_FUNC = "redistribute_input"

# Tracks if we're in inductor benchmarking, and temporarily disables logging
# (for ignoring autotuning kernel launches which don't affect the user-facing result)
_IN_INDUCTOR_BENCHMARK: bool = False
# For record_outputs, log_tensor_hashes hooks for triton kernels.
# Stores kernel outputs in call.record["output"]
_RECORD_TRITON_OUTPUTS: bool = False
# Annotates kernel output hashes, and stores them in call.post_hashes
_TRITON_OUTPUT_HASH_FN: Callable | None = None
# Annotates kernel input hashes, and stores them in call.pre_hashes
_TRITON_INPUT_HASH_FN: Callable | None = None

# registered dispatch call hooks
_DISPATCH_RECORD_HOOKS: list[Callable] = []
_DISPATCH_LOG_HOOKS: list[Callable] = []
_DISPATCH_PRE_LOG_HOOKS: list[Callable] = []


def _stringify_shape(shape) -> str:
    # [FUNCTION DOC] HELPER: Format a shape as a bracketed, comma-separated list of sizes.
    return f"[{', '.join([str(x) for x in shape])}]"


def _stringify_device_mesh(mesh) -> str:
    # [FUNCTION DOC] HELPER: Format a device mesh by its dimension sizes only, prefixed with DM.
    return f"DM({', '.join([str(s) for s in mesh.shape])})"


def _stringify_placement(placement) -> str:
    # [FUNCTION DOC] HELPER: Format a sequence of placements as a bracketed comma-separated list.
    return f"[{', '.join([str(p) for p in placement])}]"


def _stringify_attributes(tensor, attributes) -> str:
    # [FUNCTION DOC] HELPER: For each requested attribute name that the tensor has, collect its
    # value; return a braced name=value list, or an empty string if none are present.
    pairs = {}
    for attr in attributes:
        if hasattr(tensor, attr):
            pairs[attr] = getattr(tensor, attr)
    if len(pairs) == 0:
        return ""
    return f"{{{', '.join([f'{k}={v}' for k, v in pairs.items()])}}}"


def _stringify_dtensor_spec(spec) -> str:
    # [FUNCTION DOC] HELPER: Format a DTensorSpec using its placements and shard order via the
    # spec class's formatting helper (imported lazily to avoid a distributed import at load).
    from torch.distributed.tensor._dtensor_spec import DTensorSpec

    return DTensorSpec.format_shard_order_str(spec.placements, spec.shard_order)


class TensorIdTracker:
    # [CLASS DOC] Assigns small stable integer ids to tensors by identity, using weak references
    # so tensors are not kept alive. State: mapping from weak id reference to id, and a counter
    # for the next id. Entries are removed when the tensor is garbage collected. Ids are never
    # reused. Key method: _id.
    def __init__(self) -> None:
        # [FUNCTION DOC] Initialize an empty id mapping with the counter at zero.
        self.tensor_memo: dict[WeakIdRef, int] = {}
        self.next_tensor_id = 0

    def _id(self, tensor) -> int:
        # [FUNCTION DOC] HELPER: Return the id for the tensor, allocating the next id on first
        # sight. Registers a finalizer that drops the mapping when the tensor dies. Runs with the
        # Python dispatcher disabled.
        # KNOWN ISSUE: a new finalizer is registered on every call, not only the first.
        with torch._C._DisablePythonDispatcher():
            o = WeakIdRef(tensor)

            def del_memo() -> None:
                # [FUNCTION DOC] HELPER: Finalizer callback; removes the tensor's entry if present.
                self.tensor_memo.pop(o, None)

            weakref.finalize(tensor, del_memo)
            if o not in self.tensor_memo:
                self.tensor_memo[o] = self.next_tensor_id
                self.next_tensor_id += 1
            return self.tensor_memo[o]


def _tensor_debug_string(tensor, attributes, tensor_memo=None) -> str:
    # [FUNCTION DOC] HELPER: Build a short description of a tensor: a prefix (dt for DTensor, ft
    # for fake tensor, t otherwise) with optional id from the tracker, then dtype abbreviation,
    # shape and selected attributes; DTensors append their spec. Raises RuntimeError if the input
    # is not a tensor.
    """Convert tensor to debug string representation."""

    if isinstance(tensor, torch.Tensor):
        tensor_debug_str = f"{dtype_abbrs[tensor.dtype]}{_stringify_shape(tensor.shape)}{_stringify_attributes(tensor, attributes)}"
        id_str = f"${tensor_memo._id(tensor)}" if tensor_memo is not None else ""
        if isinstance(tensor, torch.distributed.tensor.DTensor):
            # omitted device mesh
            return f"dt{id_str}: {tensor_debug_str}| {_stringify_dtensor_spec(tensor._spec)}"
        elif is_fake_tensor(tensor):
            return f"ft{id_str}: {tensor_debug_str}"
        else:
            return f"t{id_str}: {tensor_debug_str}"
    else:
        raise RuntimeError(f"Unsupported tensor type: {type(tensor)}")


def _arg_to_str(arg, attributes, tensor_memo=None) -> str:
    # [FUNCTION DOC] HELPER: Convert a nested argument structure to a string, replacing tensors
    # with their debug strings and DTensorSpecs with their formatted spec, leaving other leaves
    # as they are.
    from torch.distributed.tensor._dtensor_spec import DTensorSpec

    def to_str(x):
        # [FUNCTION DOC] HELPER: Leaf mapper: tensor to debug string, DTensorSpec to spec string,
        # anything else unchanged.
        if isinstance(x, torch.Tensor):
            return _tensor_debug_string(x, attributes, tensor_memo)
        elif isinstance(x, DTensorSpec):
            return _stringify_dtensor_spec(x)
        return x

    arg = tree_map(to_str, arg)
    return str(arg)


def norm_hash_fn(t: torch.Tensor, use_scalar: bool = False) -> torch.Tensor | float:
    # [FUNCTION DOC] Hash a tensor by its L1 norm. Non-floating, non-complex inputs are converted
    # to float; the data is made contiguous and cast to float64 (complex128 for complex input).
    # Returns the norm as a tensor, or a Python number if use_scalar is set. Runs with the Python
    # dispatcher disabled.
    # KNOWN ISSUE: the docstring claims NaN/inf values are replaced with fixed numbers, but the
    # code does not do this, so NaN/inf propagate into the result.
    """
    from Observer. Computes a hash for a tensor by converting it to float (if needed), making it contiguous,
    replacing NaN/inf values with fixed numbers, and then computing the L1 norm in float64 or complex128.
    This is used to generate a deterministic summary value for tensor comparison.
    """
    with torch._C._DisablePythonDispatcher():
        if not (t.is_floating_point() or t.is_complex()):
            t = t.float()
        t = t.contiguous()

        if t.is_complex():
            t_float = t.to(dtype=torch.complex128)
        else:
            t_float = t.to(dtype=torch.float64)

        out = t_float.norm(p=1)
        if use_scalar:
            return out.item()
        return out


def _compute_rel_diff(hash1, hash2):
    # [FUNCTION DOC] HELPER: Relative difference of two hash values: absolute difference divided
    # by the larger magnitude, floored at a tiny epsilon to avoid division by zero.
    # Relative difference: |hash1 - hash2| / max(|hash1|, |hash2|, eps)
    numerator = abs(hash1 - hash2)
    denominator = max(abs(hash1), abs(hash2), 1e-10)
    return numerator / denominator


def hash_tensor_fn(t: torch.Tensor, use_scalar: bool = False) -> torch.Tensor | int:
    # [FUNCTION DOC] Hash a tensor with the built-in tensor hash. DTensors are first reduced to
    # their local shard; floats are cast to float64, complex values are viewed as float64 pairs,
    # everything else is cast to int64. Empty tensors hash to a zero uint64 scalar. Returns a
    # tensor, or a Python number if use_scalar is set.
    """
    wrapper over torch.hash_tensor
    """
    if isinstance(t, torch.distributed.tensor.DTensor):
        t = t.to_local()

    if t.is_floating_point():
        t_clean = t.to(dtype=torch.float64)
    elif t.is_complex():
        t_clean = t.to(dtype=torch.complex128).view(torch.float64)
    else:
        t_clean = t.to(dtype=torch.int64)

    if t.numel() > 0:
        out = torch.hash_tensor(t_clean)
    else:
        out = torch.zeros((), device=t_clean.device, dtype=torch.uint64)

    if use_scalar:
        return out.item()  # type: ignore[attribute]
    return out


def _get_stack_trace() -> str:
    # [FUNCTION DOC] HELPER: Capture the current stack, drop the innermost four frames (DebugMode
    # internals) and frames from files deemed uninteresting, and return the formatted text.
    from torch.fx.experimental.symbolic_shapes import uninteresting_files

    summary = CapturedTraceback.extract().summary()
    summary = summary[:-4]  # filter out DebugMode frames
    summary = [
        frame for frame in summary if frame.filename not in uninteresting_files()
    ]
    summary = traceback.StackSummary.from_list(summary)
    return "".join(summary.format())


def _get_user_stack_trace(stack_trace_str: str) -> str | None:
    # [FUNCTION DOC] HELPER: Parse a formatted stack trace and pick the frame selected by the
    # parser when files inside the torch package directory are filtered out. Returns a one-line
    # description (file, line, function, code) or None when no user frame is found.
    # Extract user code stack trace, filtering out torch internals.
    torch_dir = os.path.dirname(inspect.getfile(torch))
    filter_fn = lambda file, name, code: not file.startswith(torch_dir + os.path.sep)  # noqa: E731
    trace = _parse_stack_trace(stack_trace_str, filter_fn=filter_fn)
    if trace:
        return f"File: {trace.file}:{trace.lineno} in {trace.name}, code: {trace.code}"
    return None


def _maybe_get_autograd_trace() -> str | None:
    # [FUNCTION DOC] HELPER: If an autograd node is currently executing and its metadata has a
    # recorded forward traceback, return it joined into one string; otherwise return None.
    if torch._C._current_autograd_node() is not None:
        tb = torch._C._current_autograd_node().metadata.get("traceback_")  # type: ignore[attr-defined]
        if tb:
            return "".join(tb)
    return None


def _get_op_name(op) -> str:
    # [FUNCTION DOC] HELPER: Name an operator: qualified name for an op overload, module-dot-name
    # for objects having both, otherwise the string form of the object.
    if isinstance(op, torch._ops.OpOverload):
        op_name = op.__qualname__
    elif hasattr(op, "__module__") and hasattr(op, "__name__"):
        op_name = f"{op.__module__}.{op.__name__}"
    else:
        op_name = str(op)
    return op_name


def _run_hook(hook, *args):
    # [FUNCTION DOC] HELPER: Call the hook with the arguments and return its result, which must
    # be None or a mapping. Raises AssertionError for any other return type.
    out = hook(*args)
    if out is not None and not isinstance(out, dict):
        raise AssertionError(f"hook must return None or dict, got {type(out).__name__}")
    return out


def _run_dispatch_pre_log_hooks(call: "_DebugCall", func, types, args, kwargs) -> None:
    # [FUNCTION DOC] HELPER: Run every registered pre-log hook before the op executes and merge
    # each non-None result into the call's log mapping (created if absent). Mutates the call.
    if _DISPATCH_PRE_LOG_HOOKS:
        for hook in _DISPATCH_PRE_LOG_HOOKS:
            hook_out = _run_hook(hook, func, types, args, kwargs, call)
            if hook_out is not None:
                # Store pre-hook results in call.log
                if call.log is None:
                    call.log = {}
                call.log.update(hook_out)


def _run_dispatch_hooks(call: "_DebugCall", func, types, args, kwargs, result) -> None:
    # [FUNCTION DOC] HELPER: After the op runs, run record hooks and store their merged results
    # as the call's record if non-empty; then run log hooks, merging results into the call's log
    # mapping (created if absent, existing pre-hook entries preserved). Mutates the call.
    if _DISPATCH_RECORD_HOOKS:
        record = {}
        for hook in _DISPATCH_RECORD_HOOKS:
            hook_out = _run_hook(hook, func, types, args, kwargs, result)
            if hook_out is not None:
                record.update(hook_out)
        if record:
            call.record = record

    if _DISPATCH_LOG_HOOKS:
        # Preserve existing log from pre-hooks (e.g., input_hash)
        if call.log is None:
            call.log = {}
        for hook in _DISPATCH_LOG_HOOKS:
            hook_out = _run_hook(hook, func, types, args, kwargs, result)
            if hook_out is not None:
                call.log.update(hook_out)
