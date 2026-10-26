# [MODULE DOC] Source-code generator for the tensor-subclass wrapper used by AOTAutograd.
# Builds Python source text (via PySourceBuilder) for a function that flattens subclass
# inputs into plain tensors/opaques/symints, calls the compiled function, and rebuilds
# subclass outputs through __tensor_unflatten__. All indices, attribute names, types and
# symint positions come from SubclassCreationMeta/PlainTensorMeta and are baked into the
# generated text. Entry points: codegen_subclass_wrapper and codegen_backward_subclass_fns.
# No import-time side effects beyond imports.
"""
Codegen for AOTDispatchSubclassWrapper.

Generates a Python function that replaces the data-driven
runtime_unwrap_tensor_subclasses / wrap_tensor_subclasses loop with
a straight-line function where all metadata (indices, attr names,
subclass types, symint positions) is baked in at compile time.
"""

import functools
import keyword
from collections.abc import Callable, Iterable
from typing import cast, TYPE_CHECKING

from torch import SymInt

from .codegen import _compile_and_exec_source, PySourceBuilder
from .schemas import ActInputPaths, OpaqueMeta, PlainTensorMeta, SubclassCreationMeta
from .utils import import_async_collective_tensor_type


if TYPE_CHECKING:
    from torch.distributed._functional_collectives import AsyncCollectiveTensor


def _is_symint_placeholder(x: None | int | SymInt) -> bool:
    # [FUNCTION DOC] HELPER: Returns True if a size/stride entry needs a runtime value:
    # the entry is None (already replaced after make_runtime_safe) or a SymInt that is not
    # a nested int. Concrete ints and nested ints return False. Pure function.
    """Check whether a size/stride entry is symbolic and needs a runtime value.

    Works both before make_runtime_safe() (entries are SymInt) and after
    (symbolic entries replaced with None, nested ints with -1).
    """
    if x is None:
        return True
    if isinstance(x, SymInt) and not x.node.is_nested_int():
        return True
    return False


def _compute_placeholders(outer: Iterable[None | int | SymInt]) -> list[bool]:
    # [FUNCTION DOC] HELPER: Maps each size/stride entry to a boolean saying whether it is
    # a symint placeholder (see _is_symint_placeholder). Returns a list of the same length.
    return [_is_symint_placeholder(s) for s in outer]


def _safe_attr_access(var: str, attr: str) -> str:
    # [FUNCTION DOC] HELPER: Returns source text that reads attribute attr from variable
    # var: plain dotted access when attr is a valid non-keyword identifier, otherwise a
    # getattr call with the quoted name.
    if attr.isidentifier() and not keyword.iskeyword(attr):
        return f"{var}.{attr}"
    return f"getattr({var}, {attr!r})"


class _OutputWrapState:
    # [CLASS DOC] HELPER: Compile-time bookkeeping used while emitting output-wrapping code.
    # Tracks how many flat outputs must still be consumed at minimum (remaining_min_count,
    # symint outputs excluded) and at maximum (remaining_max_count, all optional symints
    # included), plus the global name of the runtime predicate that decides whether an
    # optional symint output is present. Both counts must reach zero when emission ends.
    def __init__(
        self,
        *,
        remaining_min_count: int,
        remaining_max_count: int,
        optional_symint_check: str,
    ) -> None:
        # [FUNCTION DOC] HELPER: Stores the initial min/max remaining output counts and the
        # name of the optional-symint check global (empty string when no optional symints).
        self.remaining_min_count = remaining_min_count
        self.remaining_max_count = remaining_max_count
        self.optional_symint_check = optional_symint_check

    def mark_required(self) -> None:
        # [FUNCTION DOC] HELPER: Records one definitely-present output: decrements both the
        # min and max remaining counts.
        self.remaining_min_count -= 1
        self.remaining_max_count -= 1

    def mark_optional(self) -> None:
        # [FUNCTION DOC] HELPER: Records one possibly-present (symint) output slot:
        # decrements only the max remaining count.
        self.remaining_max_count -= 1


def _maybe_wait_async_collective_tensor(
    x: object,
    AsyncCollectiveTensor: type["AsyncCollectiveTensor"],
) -> object:
    # [FUNCTION DOC] HELPER: Runtime helper bound into generated code. If x is an
    # AsyncCollectiveTensor, calls trigger_wait on it and returns the resulting tensor;
    # otherwise returns x unchanged. AsyncCollectiveTensor is passed in as a parameter.
    """Wait on ACT values and leave all other runtime inputs unchanged."""
    if isinstance(x, AsyncCollectiveTensor):
        return cast("AsyncCollectiveTensor", x).trigger_wait()
    return x


def _codegen_unwrap_subclass(
    state: PySourceBuilder,
    meta: SubclassCreationMeta,
    var: str,
    indent: int = 1,
    include_symints: bool = True,
    act_input_paths: set[tuple[str, ...]] | None = None,
    act_wait_fn: str | None = None,
) -> None:
    # [FUNCTION DOC] HELPER: Emits, into the PySourceBuilder, statements that flatten the
    # subclass held in variable var into the list named unwrapped_args, following meta.attrs
    # in order: plain tensors and opaques are appended (after an ACT wait call when the attr
    # path is in act_input_paths), nested subclasses are recursed into, then (if
    # include_symints) the symbolic outer sizes and strides are appended.
    # Raises AssertionError if an ACT path continues past a leaf or stops at a subclass, or
    # if an ACT path exists but act_wait_fn is None. Side effect: appends lines to state.
    """Emit code to recursively unwrap a single subclass input."""
    act_input_paths = act_input_paths or set()

    def emit_plain_tensor_symints(var: str, meta: PlainTensorMeta) -> None:
        # [FUNCTION DOC] HELPER: When include_symints is set, emits code that appends the
        # symbolic size and stride entries (as marked by the meta placeholders) of the plain
        # tensor in var to unwrapped_args. Emits nothing otherwise.
        if not include_symints:
            return

        if any(meta.size_symbol_placeholders):
            size_var = state.fresh_name("_size")
            state.emit(f"{size_var} = {var}.size()", indent=indent)
            for i, is_sym in enumerate(meta.size_symbol_placeholders):
                if is_sym:
                    state.emit(f"unwrapped_args.append({size_var}[{i}])", indent=indent)

        if any(meta.stride_symbol_placeholders):
            stride_var = state.fresh_name("_stride")
            state.emit(f"{stride_var} = {var}.stride()", indent=indent)
            for i, is_sym in enumerate(meta.stride_symbol_placeholders):
                if is_sym:
                    state.emit(
                        f"unwrapped_args.append({stride_var}[{i}])", indent=indent
                    )

    for attr, attr_meta in meta.attrs.items():
        attr_expr = _safe_attr_access(var, attr)
        attr_act_input_paths = {
            path[1:] for path in act_input_paths if path and path[0] == attr
        }
        match attr_meta:
            case PlainTensorMeta():
                if attr_act_input_paths:
                    if attr_act_input_paths != {()}:
                        raise AssertionError(
                            f"ACT path for {attr} continues past a leaf meta"
                        )
                    if act_wait_fn is None:
                        raise AssertionError("missing ACT wait function")
                    resolved_var = state.fresh_name("_resolved")
                    state.emit(
                        f"{resolved_var} = {act_wait_fn}({attr_expr})",
                        indent=indent,
                    )
                    attr_expr = resolved_var
                state.emit(f"unwrapped_args.append({attr_expr})", indent=indent)
                emit_plain_tensor_symints(attr_expr, attr_meta)
            case OpaqueMeta():
                if attr_act_input_paths:
                    if attr_act_input_paths != {()}:
                        raise AssertionError(
                            f"ACT path for {attr} continues past a leaf meta"
                        )
                    if act_wait_fn is None:
                        raise AssertionError("missing ACT wait function")
                    resolved_var = state.fresh_name("_resolved")
                    state.emit(
                        f"{resolved_var} = {act_wait_fn}({attr_expr})",
                        indent=indent,
                    )
                    state.emit(f"unwrapped_args.append({resolved_var})", indent=indent)
                else:
                    state.emit(
                        f"unwrapped_args.append({attr_expr})",
                        indent=indent,
                    )
            case SubclassCreationMeta():
                if () in attr_act_input_paths:
                    raise AssertionError(f"ACT path for {attr} stops at subclass meta")
                inner_var = state.fresh_name("_inner")
                state.emit(f"{inner_var} = {attr_expr}", indent=indent)
                _codegen_unwrap_subclass(
                    state,
                    attr_meta,
                    inner_var,
                    indent=indent,
                    include_symints=include_symints,
                    act_input_paths=attr_act_input_paths,
                    act_wait_fn=act_wait_fn,
                )

    if include_symints:
        size_placeholders = _compute_placeholders(meta.outer_size)
        stride_placeholders = _compute_placeholders(meta.outer_stride)
        if any(size_placeholders) or any(stride_placeholders):
            size_var = state.fresh_name("_size")
            state.emit(f"{size_var} = {var}.size()", indent=indent)
            for i, is_sym in enumerate(size_placeholders):
                if is_sym:
                    state.emit(f"unwrapped_args.append({size_var}[{i}])", indent=indent)

            stride_var = state.fresh_name("_stride")
            state.emit(f"{stride_var} = {var}.stride()", indent=indent)
            for i, is_sym in enumerate(stride_placeholders):
                if is_sym:
                    state.emit(
                        f"unwrapped_args.append({stride_var}[{i}])", indent=indent
                    )


def _concrete_value(val: None | int | SymInt) -> int:
    # [FUNCTION DOC] HELPER: Returns the int to bake into generated code for a non-symbolic
    # size/stride entry: the int itself, or -1 for a nested-int SymInt. Raises
    # AssertionError for anything else (None or a non-nested SymInt).
    """Get the concrete int value for a non-symbolic size/stride entry.

    Used for entries that are NOT symbolic placeholders, meaning they are
    concrete ints or nested ints (represented as -1 after make_runtime_safe).
    """
    if isinstance(val, int):
        return val
    # Before make_runtime_safe: nested ints are SymInts; use -1 as dummy.
    # After make_runtime_safe: they're already -1.
    if isinstance(val, SymInt) and val.node.is_nested_int():
        return -1
    raise AssertionError(f"Expected concrete int, got {type(val)}: {val}")


def _is_optional_symint_output(val: object) -> bool:
    # [FUNCTION DOC] HELPER: Runtime predicate used by generated code; True if val is an int
    # or SymInt (so it can be an optional symint output rather than a tensor).
    return isinstance(val, (int, SymInt))


def _codegen_wrap_subclass(
    state: PySourceBuilder,
    meta: SubclassCreationMeta,
    output_state: _OutputWrapState,
) -> str:
    # [FUNCTION DOC] HELPER: Emits code that rebuilds one subclass output from the flat list
    # unwrapped_outs using a running index _out_idx: reads each attr (plain, opaque, or
    # recursively wrapped subclass), builds the inner-tensor dict, determines outer size and
    # stride (from a designated attr, or from baked ints plus consumed symint outputs), and
    # calls the subclass's __tensor_unflatten__ with registered globals for type and meta.
    # Updates output_state counts. Returns the name of the generated result variable.
    """Emit code to reconstruct one subclass output. Returns the variable name."""
    inner_dict_var = state.fresh_name("_out_inner")
    entries: list[str] = []
    attr_exprs: dict[str, str] = {}

    for attr, attr_meta in meta.attrs.items():
        match attr_meta:
            case PlainTensorMeta():
                attr_expr = state.fresh_name("_out_attr")
                state.emit(f"{attr_expr} = unwrapped_outs[_out_idx]")
                state.emit("_out_idx += 1")
                output_state.mark_required()
                _emit_optional_plain_tensor_symint_skips(state, output_state, attr_meta)
            case OpaqueMeta():
                attr_expr = state.fresh_name("_out_attr")
                state.emit(f"{attr_expr} = unwrapped_outs[_out_idx]")
                state.emit("_out_idx += 1")
                output_state.mark_required()
            case SubclassCreationMeta():
                attr_expr = _codegen_wrap_subclass(state, attr_meta, output_state)
        attr_exprs[attr] = attr_expr
        entries.append(f"{attr!r}: {attr_expr}")

    state.emit(f"{inner_dict_var} = {{{', '.join(entries)}}}")

    # Reconstruct outer_size and outer_stride
    size_placeholders = _compute_placeholders(meta.outer_size)
    stride_placeholders = _compute_placeholders(meta.outer_stride)

    def _build_tuple(
        outer: Iterable[None | int | SymInt], placeholders: list[bool]
    ) -> str:
        # [FUNCTION DOC] HELPER: Builds the source text of a size/stride tuple: symbolic
        # positions consume the next flat output (emitting a read and counting it as
        # required), other positions are baked as concrete ints. Single-element tuples get
        # a trailing comma.
        parts: list[str] = []
        for val, is_sym in zip(outer, placeholders):
            if is_sym:
                sym_expr = state.fresh_name("_out_sym")
                state.emit(f"{sym_expr} = unwrapped_outs[_out_idx]")
                state.emit("_out_idx += 1")
                output_state.mark_required()
                parts.append(sym_expr)
            else:
                parts.append(repr(_concrete_value(val)))
        if len(parts) == 1:
            return f"({parts[0]},)"
        return f"({', '.join(parts)})"

    def _consume_optional_placeholders(placeholders: list[bool]) -> None:
        # [FUNCTION DOC] HELPER: For each symbolic placeholder, emits the optional-symint
        # skip logic so any symint outputs present are skipped (used when size/stride come
        # from an inner attr instead of the flat outputs).
        for is_sym in placeholders:
            if is_sym:
                _emit_optional_symint_skip(state, output_state)

    outer_size_from_attr = meta.outer_size_from_attr
    outer_stride_from_attr = meta.outer_stride_from_attr
    if outer_size_from_attr is not None:
        size_expr = f"{attr_exprs[outer_size_from_attr]}.size()"
        _consume_optional_placeholders(size_placeholders)
    else:
        size_expr = _build_tuple(meta.outer_size, size_placeholders)

    if outer_stride_from_attr is not None:
        stride_expr = f"{attr_exprs[outer_stride_from_attr]}.stride()"
        _consume_optional_placeholders(stride_placeholders)
    else:
        stride_expr = _build_tuple(meta.outer_stride, stride_placeholders)

    type_name = state.add_global(
        state.fresh_name("_subclass_type"),
        meta.original_subclass_type or type(meta.original_subclass),
    )
    meta_name = state.add_global(state.fresh_name("_meta"), meta.meta)

    result_var = state.fresh_name("_out")
    state.emit(
        f"{result_var} = {type_name}.__tensor_unflatten__("
        f"{inner_dict_var}, {meta_name}, {size_expr}, {stride_expr})"
    )
    return result_var


def _count_output_args(
    meta: PlainTensorMeta | SubclassCreationMeta,
    *,
    include_subclass_symints: bool,
) -> int:
    # [FUNCTION DOC] HELPER: Counts how many flat output slots a meta occupies. Plain tensor:
    # its arg_count if symints are included, else 1. Subclass: sum over attrs (opaque = 1,
    # others recursive) plus, when symints are included, all symbolic outer size/stride
    # entries; when not included, those entries are counted only if size/stride are not
    # taken from an inner attr.
    if isinstance(meta, PlainTensorMeta):
        return meta.arg_count if include_subclass_symints else 1

    total = 0
    for attr_meta in meta.attrs.values():
        if isinstance(attr_meta, OpaqueMeta):
            total += 1
        else:
            total += _count_output_args(
                attr_meta, include_subclass_symints=include_subclass_symints
            )

    if include_subclass_symints:
        total += sum(_compute_placeholders(meta.outer_size))
        total += sum(_compute_placeholders(meta.outer_stride))
    else:
        if meta.outer_size_from_attr is None:
            total += sum(_compute_placeholders(meta.outer_size))
        if meta.outer_stride_from_attr is None:
            total += sum(_compute_placeholders(meta.outer_stride))
    return total


def _emit_optional_symint_skip(
    state: PySourceBuilder,
    output_state: _OutputWrapState,
) -> None:
    # [FUNCTION DOC] HELPER: Emits code for one symint output slot that may or may not be
    # present at runtime. The generated code computes how many outputs remain, decides
    # whether the next flat output is consumed as a symint (index bump) using the
    # remaining-count windows and the optional-symint predicate, and asserts the layout is
    # consistent. Calls output_state.mark_optional.
    remaining_var = state.fresh_name("_remaining_outs")
    include_var = state.fresh_name("_include_symints")
    later_min_count = output_state.remaining_min_count
    later_max_count = output_state.remaining_max_count - 1
    max_fits = (
        f"{remaining_var} - 1 >= {later_min_count} and "
        f"{remaining_var} - 1 <= {later_max_count}"
    )
    min_fits = (
        f"{remaining_var} >= {later_min_count} and {remaining_var} <= {later_max_count}"
    )

    state.emit(f"{remaining_var} = _num_wrapped_outs - _out_idx")
    state.emit(
        f"{include_var} = ({max_fits}) and "
        f"{output_state.optional_symint_check}(unwrapped_outs[_out_idx])"
    )
    state.emit(
        f"assert {include_var} or ({min_fits}), "
        "'could not match output layout with optional SymInt outputs'"
    )
    state.emit(f"if {include_var}:")
    state.emit("_out_idx += 1", indent=2)
    output_state.mark_optional()


def _emit_optional_plain_tensor_symint_skips(
    state: PySourceBuilder,
    output_state: _OutputWrapState,
    meta: PlainTensorMeta,
) -> None:
    # [FUNCTION DOC] HELPER: For a plain tensor meta, emits one optional-symint skip for
    # every symbolic size placeholder, then for every symbolic stride placeholder.
    for is_sym in meta.size_symbol_placeholders:
        if is_sym:
            _emit_optional_symint_skip(state, output_state)
    for is_sym in meta.stride_symbol_placeholders:
        if is_sym:
            _emit_optional_symint_skip(state, output_state)


def _emit_output_wrapping(
    state: PySourceBuilder,
    out_metas: list[PlainTensorMeta | SubclassCreationMeta],
    num_fw_outs_saved_for_bw: int | None,
) -> tuple[list[str], int]:
    # [FUNCTION DOC] HELPER: Emits the whole output-wrapping section: computes min/max
    # flat output counts across out_metas, registers the optional-symint predicate when
    # they differ, emits the output index setup and a length assertion (excluding
    # num_fw_outs_saved_for_bw trailing activations), wraps each meta (plain or subclass),
    # and emits a final consumption assertion. Returns the result expression names and the
    # total arg_count tallied. Raises AssertionError if the compile-time counts do not
    # balance to zero.
    """Emit wrapping code for output metas.

    Returns (result_exprs, num_args_tallied) where result_exprs are Python
    expression strings referencing each wrapped output.
    """
    result_exprs: list[str] = []
    num_args_tallied = 0
    saved_for_bw = num_fw_outs_saved_for_bw or 0
    min_counts = [
        _count_output_args(meta, include_subclass_symints=False) for meta in out_metas
    ]
    max_counts = [
        _count_output_args(meta, include_subclass_symints=True) for meta in out_metas
    ]
    min_wrapped_outputs = sum(min_counts)
    max_wrapped_outputs = sum(max_counts)

    optional_symint_check = (
        state.add_global(
            state.fresh_name("_is_optional_symint_output"),
            _is_optional_symint_output,
        )
        if min_wrapped_outputs != max_wrapped_outputs
        else ""
    )
    state.emit("_out_idx = 0")
    if saved_for_bw:
        state.emit(f"_num_wrapped_outs = len(unwrapped_outs) - {saved_for_bw}")
    else:
        state.emit("_num_wrapped_outs = len(unwrapped_outs)")
    if min_wrapped_outputs != max_wrapped_outputs:
        state.emit(
            "assert "
            f"{min_wrapped_outputs} <= _num_wrapped_outs <= {max_wrapped_outputs}, "
            f"f'expected between {min_wrapped_outputs} and {max_wrapped_outputs} "
            "wrapped outputs, got {_num_wrapped_outs}'"
        )
    else:
        state.emit(
            f"assert _num_wrapped_outs == {max_wrapped_outputs}, "
            f"f'expected {max_wrapped_outputs} wrapped outputs, "
            "got {_num_wrapped_outs}'"
        )

    output_state = _OutputWrapState(
        remaining_min_count=min_wrapped_outputs,
        remaining_max_count=max_wrapped_outputs,
        optional_symint_check=optional_symint_check,
    )
    for meta in out_metas:
        if isinstance(meta, PlainTensorMeta):
            out_expr = state.fresh_name("_out_plain")
            state.emit(f"{out_expr} = unwrapped_outs[_out_idx]")
            state.emit("_out_idx += 1")
            output_state.mark_required()
            _emit_optional_plain_tensor_symint_skips(state, output_state, meta)
            result_exprs.append(out_expr)
            num_args_tallied += meta.arg_count
        else:
            result_var = _codegen_wrap_subclass(state, meta, output_state)
            result_exprs.append(result_var)
            num_args_tallied += meta.arg_count

    if output_state.remaining_min_count != 0 or output_state.remaining_max_count != 0:
        raise AssertionError(
            "output wrapping counts did not balance: "
            f"{output_state.remaining_min_count}, "
            f"{output_state.remaining_max_count}"
        )
    state.emit(
        "assert _out_idx == _num_wrapped_outs, "
        "f'wrapped {_out_idx} outputs, expected {_num_wrapped_outs}'"
    )
    return result_exprs, num_args_tallied


def _emit_input_unwrapping(
    state: PySourceBuilder,
    inp_metas: list[PlainTensorMeta | SubclassCreationMeta],
    frozen_inp_indices: frozenset[int] = frozenset(),
    include_symints: bool = True,
    act_input_paths_by_input: dict[int, set[tuple[str, ...]]] | None = None,
    act_wait_fn: str | None = None,
) -> None:
    # [FUNCTION DOC] HELPER: Emits code appending each input to unwrapped_args. Plain tensor
    # inputs are appended directly (through the ACT wait function if flagged); frozen
    # subclass inputs append None; other subclass inputs get a runtime exact-type
    # assertion then are unwrapped via _codegen_unwrap_subclass. Raises AssertionError on
    # inconsistent ACT paths or a missing wait function.
    """Emit unwrapping code for input metas into unwrapped_args.

    Caller must have already emitted ``unwrapped_args = []``.
    """
    act_input_paths_by_input = act_input_paths_by_input or {}
    for i, meta in enumerate(inp_metas):
        input_act_paths = act_input_paths_by_input.get(i, set())
        if isinstance(meta, PlainTensorMeta):
            if input_act_paths:
                if input_act_paths != {()}:
                    raise AssertionError(
                        f"ACT path for input {i} continues past a plain meta"
                    )
                if act_wait_fn is None:
                    raise AssertionError("missing ACT wait function")
                state.emit(f"unwrapped_args.append({act_wait_fn}(args[{i}]))")
            else:
                state.emit(f"unwrapped_args.append(args[{i}])")
        elif i in frozen_inp_indices:
            # Frozen by inductor freezing: constant already baked into graph.
            state.emit("unwrapped_args.append(None)")
        else:
            if () in input_act_paths:
                raise AssertionError(f"ACT path for input {i} stops at subclass meta")
            inp_var = state.fresh_name("_inp")
            type_name = state.add_global(
                state.fresh_name("_expected_type"),
                meta.original_subclass_type or type(meta.original_subclass),
            )
            state.emit(f"{inp_var} = args[{i}]")
            state.emit(
                f"assert type({inp_var}) is {type_name}, "
                f"f'expected {{{type_name}}}, got {{type({inp_var})}}'",
            )
            _codegen_unwrap_subclass(
                state,
                meta,
                inp_var,
                indent=1,
                include_symints=include_symints,
                act_input_paths=input_act_paths,
                act_wait_fn=act_wait_fn,
            )


def _codegen_subclass_wrapper_source(
    inp_metas: list[PlainTensorMeta | SubclassCreationMeta],
    out_metas: list[PlainTensorMeta | SubclassCreationMeta],
    num_fw_outs_saved_for_bw: int | None,
    frozen_inp_indices: frozenset[int] = frozenset(),
    act_input_paths: ActInputPaths | None = None,
) -> tuple[str, dict[str, object]]:
    # [FUNCTION DOC] HELPER: Builds the full source of the forward wrapper function
    # inner_fn(args): unwrap inputs, pass through trailing extra args, clear args, call
    # compiled_fn, wrap outputs, and return the wrapped tuple (plus saved activations when
    # num_fw_outs_saved_for_bw is given). Registers an ACT wait helper global when
    # act_input_paths is non-empty. Returns (source, globals) without compiled_fn.
    """Generate source and globals for a subclass wrapper.

    Returns (source, globals_dict).  The globals_dict will NOT contain
    ``compiled_fn`` — the caller is responsible for adding it before exec.
    """
    state = PySourceBuilder()

    state.emit("def inner_fn(args):", indent=0)

    act_input_paths_by_input: dict[int, set[tuple[str, ...]]] = {}
    act_wait_fn = None
    if act_input_paths:
        AsyncCollectiveTensor = import_async_collective_tensor_type()
        act_wait_fn = state.add_global(
            state.fresh_name("_maybe_wait_act"),
            functools.partial(
                _maybe_wait_async_collective_tensor,
                AsyncCollectiveTensor=AsyncCollectiveTensor,
            ),
        )
        for i, attr_path in act_input_paths:
            act_input_paths_by_input.setdefault(i, set()).add(attr_path)

    # --- Input unwrapping ---
    state.emit("unwrapped_args = []")
    _emit_input_unwrapping(
        state,
        inp_metas,
        frozen_inp_indices=frozen_inp_indices,
        act_input_paths_by_input=act_input_paths_by_input,
        act_wait_fn=act_wait_fn,
    )

    # Pass through any trailing args not covered by inp_metas
    # (e.g. rng seed/offset added by FunctionalizedRngRuntimeWrapper).
    num_inp_metas = len(inp_metas)
    state.emit(f"unwrapped_args.extend(args[{num_inp_metas}:])")
    state.emit("args.clear()")

    # --- Call compiled function ---
    state.emit("unwrapped_outs = compiled_fn(unwrapped_args)")

    # --- Output wrapping ---
    result_exprs, _ = _emit_output_wrapping(state, out_metas, num_fw_outs_saved_for_bw)
    result_tuple = f"({', '.join(result_exprs)},)" if result_exprs else "()"
    if num_fw_outs_saved_for_bw is not None:
        state.emit(
            f"_activation_start = len(unwrapped_outs) - {num_fw_outs_saved_for_bw}"
        )
        state.emit(f"return {result_tuple} + tuple(unwrapped_outs[_activation_start:])")
    else:
        state.emit(f"return {result_tuple}")

    source = state.getvalue()
    return source, state.globals


def _codegen_subclass_wrap_source(
    out_metas: list[PlainTensorMeta | SubclassCreationMeta],
) -> tuple[str, dict[str, object]]:
    # [FUNCTION DOC] HELPER: Builds source of wrap_fn(unwrapped_outs) that rebuilds
    # subclass/plain outputs from a flat list and returns them as a tuple. Returns
    # (source, globals).
    """Generate source for wrapping flat outputs into subclasses.

    Used for the backward epilogue. Shares output-wrapping logic with
    _codegen_subclass_wrapper_source via _emit_output_wrapping.
    """
    state = PySourceBuilder()
    state.emit("def wrap_fn(unwrapped_outs):", indent=0)
    result_exprs, _ = _emit_output_wrapping(
        state, out_metas, num_fw_outs_saved_for_bw=None
    )
    result_tuple = f"({', '.join(result_exprs)},)" if result_exprs else "()"
    state.emit(f"return {result_tuple}")
    source = state.getvalue()
    return source, state.globals


def codegen_backward_subclass_fns(
    grad_input_metas: list[PlainTensorMeta | SubclassCreationMeta] | None = None,
) -> tuple[Callable[..., object], Callable[..., object] | None]:
    # [FUNCTION DOC] Compiles the backward-pass helpers. Always builds unwrap_fn, which
    # returns its argument list as a new list. If grad_input_metas is given, also generates
    # and compiles wrap_fn that rebuilds grad inputs into subclasses; otherwise wrap_fn is
    # None. Returns (unwrap_fn, wrap_fn). Side effect: executes generated source.
    """Generate codegen'd unwrap and wrap functions for the backward pass.

    Returns (unwrap_fn, wrap_fn). unwrap_fn is used by the backward prologue
    to unwrap non-tangent subclass inputs (always an identity in AOT dispatch
    since the compiled forward operates on unwrapped inner tensors). wrap_fn
    is used by the backward epilogue to wrap flat grad inputs back into
    subclasses; it is None when grad_input_metas is None.
    """
    source = "def unwrap_fn(args):\n    return list(args)"
    globals_dict: dict[str, object] = {}
    unwrap_fn = _compile_and_exec_source(
        source, globals_dict, "unwrap_fn", "backward_subclass_unwrap"
    )

    wrap_fn = None
    if grad_input_metas is not None:
        wrap_source, wrap_globals = _codegen_subclass_wrap_source(grad_input_metas)
        wrap_fn = _compile_and_exec_source(
            wrap_source, wrap_globals, "wrap_fn", "backward_subclass_wrapper"
        )

    return unwrap_fn, wrap_fn


def codegen_subclass_wrapper(
    compiled_fn: Callable[..., object],
    inp_metas: list[PlainTensorMeta | SubclassCreationMeta],
    out_metas: list[PlainTensorMeta | SubclassCreationMeta],
    num_fw_outs_saved_for_bw: int | None,
    frozen_inp_indices: frozenset[int] = frozenset(),
    act_input_paths: ActInputPaths | None = None,
) -> Callable[..., object]:
    # [FUNCTION DOC] Public entry point: generates the forward wrapper source for the given
    # input/output metas, binds compiled_fn into its globals, compiles it, and returns the
    # resulting callable (taking a single args list, which it clears) that unwraps subclass
    # inputs, calls compiled_fn, and re-wraps outputs.
    """Generate a specialized wrapper function for subclass unwrap/wrap."""
    source, globals_dict = _codegen_subclass_wrapper_source(
        inp_metas,
        out_metas,
        num_fw_outs_saved_for_bw,
        frozen_inp_indices,
        act_input_paths=act_input_paths,
    )
    globals_dict["compiled_fn"] = compiled_fn
    return _compile_and_exec_source(
        source, globals_dict, "inner_fn", "subclass_wrapper", wrapped_fn=compiled_fn
    )
