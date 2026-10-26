# [MODULE DOC] Helpers for AOTAutograd's handling of traceable tensor subclasses: building
# rebuild metadata (create_subclass_meta / create_subclass_metadata), flattening subclasses
# into plain tensors/opaques/symints both at trace time (unwrap_tensor_subclasses) and at
# runtime (runtime_unwrap_tensor_subclasses), re-wrapping flat values (wrap_tensor_subclasses
# and its joint variant), remapping static-input indices, and recomputing mutated input
# indices after flattening. Import-time side effects: rebinds the module-level name zip to a
# strict zip that raises on length mismatch.
"""
This file contains utilities for tracing through __torch_dispatch__ based tensor subclasses and modes.
AOTAutograd's responsibility is to trace through all pytorch capabilities that live in the pytorch dispatcher,
and this includes tensor subclasses that implement __torch_dispatch__.
"""

import collections
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeGuard, TypeVar

import torch
import torch.utils._pytree as pytree
from torch import SymInt, Tensor
from torch._custom_class_base import CustomClassBase
from torch._library.fake_class_registry import maybe_unwrap_fake_script_object
from torch._library.opaque_object import is_opaque_symbolic_type
from torch._subclasses.fake_tensor import get_plain_tensors
from torch.fx.experimental.symbolic_shapes import guard_or_false, sym_eq
from torch.types import IntLikeType
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    TraceableWrapperSubclass,
)

from .descriptors import (
    AOTInput,
    AOTOutput,
    DummyAOTInput,
    SubclassGetAttrAOTInput,
    SubclassGetAttrAOTOutput,
    SubclassSizeAOTInput,
    SubclassSizeAOTOutput,
    SubclassStrideAOTInput,
    SubclassStrideAOTOutput,
)
from .schemas import (
    FakifiedFlatArgs,
    FxValue,
    MutationType,
    OpaqueMeta,
    PlainTensorMeta,
    SubclassCreationMeta,
    ViewAndMutationMeta,
)
from .utils import strict_zip


zip = strict_zip

T = TypeVar("T", bound=torch.Tensor)


def requires_subclass_dispatch(
    args: FakifiedFlatArgs, fw_metadata: ViewAndMutationMeta
) -> bool:
    # [FUNCTION DOC] Returns True if the runtime must unwrap/wrap subclasses: any flattened
    # tensor argument is a traceable wrapper subclass, or any forward-graph output meta is a
    # SubclassCreationMeta. Pure; args are flattened with pytree.
    args_flattened = pytree.arg_tree_leaves(*args)
    any_subclass_args = any(
        is_traceable_wrapper_subclass(x)
        for x in args_flattened
        if isinstance(x, Tensor)
    )
    from torch._functorch._aot_autograd.schemas import SubclassCreationMeta

    any_subclass_outputs = any(
        type(x) is SubclassCreationMeta for x in fw_metadata.subclass_fw_graph_out_meta
    )
    # This tells us whether or not we need to perform any unwrapping/wrapping of tensor subclasses at runtime.
    return bool(any_subclass_args or any_subclass_outputs)


from .schemas import MemoryFormatMeta


def maybe_suggest_memory_format(
    t: Tensor, with_memory_format: bool
) -> MemoryFormatMeta | None:
    # [FUNCTION DOC] Returns None when with_memory_format is False, else the
    # MemoryFormatMeta computed from tensor t.
    if not with_memory_format:
        return None

    return MemoryFormatMeta.from_tensor(t)


def _safe_metadata_eq(
    outer: Iterable[IntLikeType],
    inner: Iterable[IntLikeType],
) -> bool:
    # [FUNCTION DOC] HELPER: Compares two int/SymInt sequences for equality without adding
    # guards; returns True only if equality is provably true, else False.
    return guard_or_false(sym_eq(tuple(outer), tuple(inner)))


def _find_attr_matching_outer_metadata(
    a: Tensor,
    inner_keys: Sequence[str],
    *,
    metadata_fn: Callable[[Tensor], Iterable[IntLikeType]],
) -> str | None:
    # [FUNCTION DOC] HELPER: Among the inner attrs of subclass a that hold Tensors, finds
    # one whose size (or stride, per metadata_fn) can stand in for a's own. Returns the first
    # tensor attr name if every tensor attr's metadata provably equals a's, else None (also
    # None when there are no tensor attrs).
    tensor_attrs =[key for key in inner_keys if isinstance(getattr(a, key), Tensor)]
    if not tensor_attrs:
        return None

    outer_metadata = tuple(metadata_fn(a))
    # Only use an attr as the runtime source when every tensor attr agrees
    # with the wrapper metadata; otherwise there is no single attr to trust.
    if all(
        _safe_metadata_eq(outer_metadata, metadata_fn(getattr(a, key)))
        for key in tensor_attrs
    ):
        return tensor_attrs[0]
    return None


def _symint_placeholders(lst: Iterable[IntLikeType]) -> tuple[bool, ...]:
    # [FUNCTION DOC] HELPER: Returns a tuple of booleans, True for each entry that is a
    # SymInt but not a nested int.
    return tuple(isinstance(s, SymInt) and not s.node.is_nested_int() for s in lst)


def get_subclass_typing_container(
    tensor_subclass: torch.Tensor,
) -> dict[type[torch.Tensor], list[type[torch.Tensor]]]:
    # [FUNCTION DOC] Recursively walks a traceable subclass and returns a mapping (default
    # dict of lists) from each subclass type found to the list of instances of that type,
    # including nested inner subclasses. Plain tensors and opaques are skipped. Raises
    # AssertionError if an inner attr is neither a Tensor nor a CustomClassBase.
    # KNOWN ISSUE: the docstring says it maps "inner tensors to subclass types", but the
    # result maps subclass types to lists of subclass instances.
    """
    Given a subclass, returns a recursive dictionary mapping each
    inner tensors to its subclass types.
    """

    def _get_types_for_subclass(tensor_subclass: torch.Tensor) -> None:
        # [FUNCTION DOC] HELPER: Records the subclass instance under its type in the
        # tracker, then recurses into Tensor inner attrs; opaque attrs are ignored. Does
        # nothing for non-subclass tensors.
        if not is_traceable_wrapper_subclass(tensor_subclass):
            return
        tracker[type(tensor_subclass)].append(tensor_subclass)
        inner_keys, _ = tensor_subclass.__tensor_flatten__()
        for key in inner_keys:
            match getattr(tensor_subclass, key):
                case torch.Tensor() as inner_value:
                    _get_types_for_subclass(inner_value)
                case CustomClassBase():
                    pass
                case unexpected:
                    raise AssertionError(
                        f"expected Tensor or CustomClassBase, got {type(unexpected)}"
                    )

    tracker: dict[Any, list[Any]] = collections.defaultdict(list)
    _get_types_for_subclass(tensor_subclass)
    return tracker


def create_subclass_metadata(
    a: Any,
    start_idx: int,
    count_symints: bool,
    with_memory_format: bool = False,
) -> tuple[Any, int]:
    # [FUNCTION DOC] Builds rebuild metadata for tensor a whose flattened values start at
    # start_idx. A plain tensor yields a PlainTensorMeta (with symint size/stride
    # placeholders if count_symints). A subclass yields a SubclassCreationMeta recording
    # nested attr metas (OpaqueMeta for opaque attrs), flatten metadata, outer size/stride,
    # the original subclass, optional memory format, and which inner attr can supply outer
    # size/stride; its arg_count includes appended size/stride symints when count_symints.
    # Returns (meta, next flat index). Raises RuntimeError for a non-symbolic opaque attr
    # and AssertionError for attrs that are neither Tensor nor opaque.
    if not is_traceable_wrapper_subclass(a):
        size_symbol_placeholders = (
            _symint_placeholders(a.size()) if count_symints else ()
        )
        stride_symbol_placeholders = (
            _symint_placeholders(a.stride()) if count_symints else ()
        )
        meta = PlainTensorMeta(
            start_idx + 1,
            memory_format=maybe_suggest_memory_format(a, with_memory_format),
            size_symbol_placeholders=size_symbol_placeholders,
            stride_symbol_placeholders=stride_symbol_placeholders,
        )
        return (
            meta,
            start_idx + meta.arg_count,
        )

    inner_keys, metadata = a.__tensor_flatten__()
    new_start_idx = start_idx
    attrs: dict[str, SubclassCreationMeta | PlainTensorMeta | OpaqueMeta] = {}

    for key in inner_keys:
        inner_value = getattr(a, key)
        match inner_value:
            case CustomClassBase():
                # During tracing, opaques are wrapped in FakeScriptObject;
                # unwrap to check the real type.
                real_type = type(maybe_unwrap_fake_script_object(inner_value))
                if not is_opaque_symbolic_type(real_type):
                    raise RuntimeError(
                        f"{real_type.__name__!r} found in tensor attrs of "
                        f"{type(a).__name__}.__tensor_flatten__(). "
                        "Only tensors and reference-type opaques are allowed "
                        "in tensor attrs."
                    )
                attrs[key] = OpaqueMeta()
                new_start_idx += 1
            case Tensor():
                new_subclass_meta, new_start_idx = create_subclass_metadata(
                    inner_value,
                    new_start_idx,
                    count_symints=count_symints,
                    with_memory_format=with_memory_format,
                )
                attrs[key] = new_subclass_meta
            case _:
                raise AssertionError(
                    f"expected Tensor or CustomClassBase, got {type(inner_value)}"
                )

    # It *must* be because is_traceable_wrapper_subclass() - but mypy is not smart.
    if not isinstance(a, Tensor):
        raise AssertionError(f"expected Tensor, got {type(a)}")

    new_start_idx = (
        new_start_idx
        + count_symints * len(enumerate_filter_symints(a.size()))
        + count_symints * len(enumerate_filter_symints(a.stride()))
    )

    return (
        SubclassCreationMeta(
            flat_tensor_start_idx=start_idx,
            arg_count=new_start_idx - start_idx,
            included_subclass_symints=count_symints,
            attrs=attrs,
            meta=metadata,
            outer_size=a.size(),  # type: ignore[attr-defined, arg-type]
            outer_stride=a.stride(),  # type: ignore[arg-type]
            original_subclass=a,
            memory_format=maybe_suggest_memory_format(a, with_memory_format),
            outer_size_from_attr=_find_attr_matching_outer_metadata(
                a, inner_keys, metadata_fn=lambda t: t.size()
            ),
            outer_stride_from_attr=_find_attr_matching_outer_metadata(
                a, inner_keys, metadata_fn=lambda t: t.stride()
            ),
        ),
        new_start_idx,
    )


# Given a flat list of arguments, some of which may be tensor subclasses,
# computes metadata about "how to reconstruct the current list of subclasses,
# if we were given their flattened dense tensors instead"
def create_subclass_meta(
    curr_args: list[Any] | tuple[Any, ...],
    *,
    count_symints: bool = True,
    with_memory_format: bool = False,
) -> list[PlainTensorMeta | SubclassCreationMeta]:
    # [FUNCTION DOC] For a flat list of args, returns one meta per arg describing how to
    # rebuild it from flattened tensors: a SubclassCreationMeta for traceable subclasses, a
    # PlainTensorMeta otherwise, tracking a running flat index by each meta's arg_count.
    # Keyword-only count_symints and with_memory_format are forwarded for subclasses.
    # Raises AssertionError if a traceable wrapper subclass is not a Tensor.
    # KNOWN ISSUE: plain args get a PlainTensorMeta with no symint placeholders even when
    # count_symints is true, unlike create_subclass_metadata's plain-tensor branch.
    idx = 0
    infos: list[PlainTensorMeta | SubclassCreationMeta] = []
    for a in curr_args:
        if is_traceable_wrapper_subclass(a):
            if not isinstance(a, Tensor):
                raise AssertionError(
                    f"expected Tensor for traceable wrapper subclass, got {type(a)}"
                )
            start_idx = idx
            subclass_meta, _ = create_subclass_metadata(
                a,
                start_idx,
                count_symints=count_symints,
                with_memory_format=with_memory_format,
            )
            infos.append(subclass_meta)
            cnt = subclass_meta.arg_count
        else:
            infos.append(
                PlainTensorMeta(
                    idx,
                    memory_format=maybe_suggest_memory_format(a, with_memory_format),
                )
            )
            cnt = 1
        idx += cnt
    return infos


def enumerate_filter_symints(lst: Iterable[IntLikeType]) -> list[tuple[int, SymInt]]:
    # [FUNCTION DOC] Returns (position, value) pairs for the entries of lst that are SymInts
    # but not nested ints.
    # Capture all SymInts from the iterable.
    def symint_check(s: IntLikeType) -> TypeGuard[SymInt]:
        # [FUNCTION DOC] HELPER: True if s is a non-nested SymInt.
        return isinstance(s, SymInt) and not s.node.is_nested_int()

    return [(i, s) for i, s in enumerate(lst) if symint_check(s)]


def compute_symint_placeholders(lst: Iterable[None | int | SymInt]) -> list[bool]:
    # [FUNCTION DOC] Returns a list of booleans, True for each entry that is None (the
    # marker make_runtime_safe leaves for non-nested symbolic ints).
    # Non-nested symints are replaced with None in `make_runtime_safe()`
    return [s is None for s in lst]


# Intended to make it easier to define function that is
# either (AOTInput -> AOTInput) or (AOTOutput -> AOTOutput)
# but not the other combos
AOTDescriptor = TypeVar("AOTDescriptor", AOTInput, AOTOutput)


# This function takes in a pytree of arguments and unwraps any tensor
# subclasses.
#
# NOTE: The reason for "append_symints":
#
# * At compile time: we append extra symint args when unwrapping primals
# (but not tangents, because they should always share symints with primals).
# We also append extra symints when unwrapping the subclass outputs of the
# traced function, so we can return them as extra outputs
#
# * At runtime: we similarly append subclass sizes when we unwrap subclass
# primals (but not tangents) on entry to the forward. See the runtime version of
# this function below.
def unwrap_tensor_subclasses(
    wrapped_args: list[FxValue],
    wrapped_args_descs: Sequence[AOTDescriptor],
    *,
    append_symints: bool,
) -> tuple[list[FxValue], list[AOTDescriptor]]:
    # [FUNCTION DOC] Trace-time flattening of a list of values (with parallel AOT
    # descriptors) into plain tensors/opaques. Each traceable subclass is expanded
    # depth-first over its __tensor_flatten__ attrs, each leaf getting a SubclassGetAttr
    # descriptor chain. If append_symints, the non-nested SymInt sizes and strides are
    # appended (with size/stride descriptors) after each subclass, and also after plain
    # tensor leaves that came from a subclass attr. Opaque objects are wrapped as fake
    # script objects when a fake mode is active. Returns (flat values, flat descriptors).
    def _maybe_fakeify_opaque(v: Any) -> Any:
        # [FUNCTION DOC] HELPER: If v is a custom-class opaque and a fake mode is active
        # and its type is a registered custom class, returns its fake wrapper; otherwise
        # returns v unchanged.
        # Registered opaque types need to be wrapped as FakeScriptObject for
        # compile-time FX tracing (proxy slot tracking, hashability, etc.).
        if isinstance(v, CustomClassBase):
            from torch._guards import detect_fake_mode
            from torch._library.fake_class_registry import maybe_to_fake_obj
            from torch._library.opaque_object import is_custom_class

            fake_mode = detect_fake_mode()
            if fake_mode is not None and is_custom_class(type(v)):
                return maybe_to_fake_obj(fake_mode, v)
        return v

    def flatten_subclass(
        t: FxValue,
        desc: AOTDescriptor,
        *,
        out: tuple[list[FxValue], list[AOTDescriptor]],
    ) -> None:
        # [FUNCTION DOC] HELPER: Recursive worker; appends the flattened values and their
        # descriptors of t to the out lists (mutates out). Non-subclass values are appended
        # as is; subclasses are expanded per attr, then symint sizes/strides are appended
        # when append_symints is set.
        # unwrap a subclass into plain tensors and their size/stride if "append_symint"
        # is True
        if not is_traceable_wrapper_subclass(t):
            out[0].append(_maybe_fakeify_opaque(t))
            out[1].append(desc)
            if (
                append_symints
                and isinstance(t, Tensor)
                and isinstance(
                    desc, (SubclassGetAttrAOTInput, SubclassGetAttrAOTOutput)
                )
            ):
                sizes = enumerate_filter_symints(t.size())
                strides = enumerate_filter_symints(t.stride())
                out[0].extend(s for _, s in sizes)
                out[0].extend(s for _, s in strides)
                if isinstance(desc, AOTInput):
                    out[1].extend(SubclassSizeAOTInput(desc, i) for i, _ in sizes)
                    out[1].extend(SubclassStrideAOTInput(desc, i) for i, _ in strides)
                else:
                    out[1].extend(SubclassSizeAOTOutput(desc, i) for i, _ in sizes)
                    out[1].extend(SubclassStrideAOTOutput(desc, i) for i, _ in strides)
            return

        attrs, _ = t.__tensor_flatten__()

        SubclassGetAttr: Callable[[AOTInput | AOTOutput, str], AOTDescriptor]
        SubclassSize: Callable[[AOTInput | AOTOutput, int], AOTDescriptor]
        SubclassStride: Callable[[AOTInput | AOTOutput, int], AOTDescriptor]
        if isinstance(desc, AOTInput):
            SubclassGetAttr = SubclassGetAttrAOTInput  # type: ignore[bad-assignment]
            SubclassSize = SubclassSizeAOTInput  # type: ignore[bad-assignment]
            SubclassStride = SubclassStrideAOTInput  # type: ignore[bad-assignment]
        else:
            SubclassGetAttr = SubclassGetAttrAOTOutput  # type: ignore[bad-assignment]
            SubclassSize = SubclassSizeAOTOutput  # type: ignore[bad-assignment]
            SubclassStride = SubclassStrideAOTOutput  # type: ignore[bad-assignment]

        for attr in attrs:
            inner_value = getattr(t, attr)
            n_desc: Any = SubclassGetAttr(desc, attr)
            flatten_subclass(inner_value, n_desc, out=out)

        if append_symints:
            sizes = enumerate_filter_symints(t.size())
            strides = enumerate_filter_symints(t.stride())
            out[0].extend(s for _, s in sizes)
            out[0].extend(s for _, s in strides)
            out[1].extend(SubclassSize(desc, i) for i, _ in sizes)
            out[1].extend(SubclassStride(desc, i) for i, _ in strides)

    xs_inner: list[FxValue] = []
    descs_inner: list[AOTDescriptor] = []

    for x, desc in zip(wrapped_args, wrapped_args_descs):
        flatten_subclass(x, desc, out=(xs_inner, descs_inner))

    return xs_inner, descs_inner


# subclass_metas is needed at runtime to compute which indices are symints in
# the outer_size/outer_stride
def runtime_unwrap_tensor_subclasses(
    wrapped_args: list[Tensor | int],
    *,
    append_symints: bool,
    subclass_metas: list[PlainTensorMeta | SubclassCreationMeta] | None = None,
) -> list[int | Tensor | SymInt | CustomClassBase]:
    # [FUNCTION DOC] Runtime flattening of wrapped_args: non-subclass args pass through;
    # subclasses are expanded into their inner tensors/opaques (via get_plain_tensors when
    # subclass_metas is None, otherwise guided by the per-arg SubclassCreationMeta, and
    # appending runtime SymInt sizes/strides at metadata-designated positions when
    # append_symints). Returns the flat list. Raises AssertionError if append_symints is set
    # without subclass_metas, a meta has the wrong type, or size/stride lengths mismatch.
    def flatten_subclass(
        x: Tensor | TraceableWrapperSubclass,
        subclass_meta: PlainTensorMeta | SubclassCreationMeta | OpaqueMeta | None,
        *,
        out: list[CustomClassBase | SymInt | Tensor | int],
    ) -> list[CustomClassBase | SymInt | Tensor | int]:
        # [FUNCTION DOC] HELPER: Appends x's flattened values to out (mutating and
        # returning it). Plain values are appended (plus symint sizes/strides for plain
        # tensors with PlainTensorMeta when append_symints); subclasses recurse over their
        # attrs and then append the symbolic outer size/stride entries.
        def append_plain_tensor_symints(tensor: Tensor, meta: PlainTensorMeta) -> None:
            # [FUNCTION DOC] HELPER: Appends to out the size and stride entries of tensor
            # at positions marked symbolic in meta. Raises AssertionError on a length
            # mismatch between the tensor and the placeholder lists.
            size = tensor.size()
            if len(size) != len(meta.size_symbol_placeholders):
                raise AssertionError(
                    f"size length mismatch: {len(size)} != {len(meta.size_symbol_placeholders)}"
                )
            out.extend(
                [
                    r
                    for (r, is_symint) in zip(size, meta.size_symbol_placeholders)
                    if is_symint
                ]
            )

            stride = tensor.stride()
            if len(stride) != len(meta.stride_symbol_placeholders):
                raise AssertionError(
                    f"stride length mismatch: {len(stride)} != {len(meta.stride_symbol_placeholders)}"
                )
            out.extend(
                [
                    r
                    for (r, is_symint) in zip(stride, meta.stride_symbol_placeholders)
                    if is_symint
                ]
            )

        if not is_traceable_wrapper_subclass(x):
            out.append(x)
            if append_symints and isinstance(subclass_meta, PlainTensorMeta):
                append_plain_tensor_symints(x, subclass_meta)
            return out

        if not isinstance(x, Tensor):
            raise AssertionError(f"expected Tensor, got {type(x)}")
        if not isinstance(subclass_meta, SubclassCreationMeta):
            raise AssertionError("subclass_meta should be a SubclassCreationMeta")

        attrs, _ = x.__tensor_flatten__()

        for attr in attrs:
            inner_value = getattr(x, attr)
            match inner_value:
                case CustomClassBase():
                    out.append(inner_value)
                case Tensor():
                    inner_meta = subclass_meta.attrs.get(attr)
                    flatten_subclass(inner_value, inner_meta, out=out)
                case _:
                    raise AssertionError(
                        f"expected Tensor or CustomClassBase, got {type(inner_value)}"
                    )

        if append_symints:
            # outer_size
            size = x.size()
            symint_placeholders = compute_symint_placeholders(subclass_meta.outer_size)
            if len(size) != len(symint_placeholders):
                raise AssertionError(
                    f"size length mismatch: {len(size)} != {len(symint_placeholders)}"
                )
            out.extend(
                [r for (r, is_symint) in zip(size, symint_placeholders) if is_symint]
            )

            # outer_stride
            stride = x.stride()
            symint_placeholders = compute_symint_placeholders(
                subclass_meta.outer_stride
            )
            if len(stride) != len(symint_placeholders):
                raise AssertionError(
                    f"stride length mismatch: {len(stride)} != {len(symint_placeholders)}"
                )
            out.extend(
                [r for (r, is_symint) in zip(stride, symint_placeholders) if is_symint]
            )
        return out

    xs_inner: list[int | Tensor | SymInt | CustomClassBase] = []

    if append_symints:
        if subclass_metas is None:
            raise AssertionError(
                "subclass_metas must not be None when append_symints is True"
            )

    for idx, x in enumerate(wrapped_args):
        if not is_traceable_wrapper_subclass(x):
            xs_inner.append(x)
            continue

        if subclass_metas is None:
            get_plain_tensors(x, out=xs_inner)
        else:
            subclass_meta = subclass_metas[idx]
            if not isinstance(subclass_meta, SubclassCreationMeta):
                raise AssertionError(
                    f"expected SubclassCreationMeta, got {type(subclass_meta)}"
                )
            flatten_subclass(x, subclass_meta, out=xs_inner)

    return xs_inner


def unwrap_tensor_subclasses_with_indices_to_original(
    wrapped_args: list[Any],
) -> tuple[list[Any], list[int]]:
    # [FUNCTION DOC] Flattens each arg (without symints) and returns the concatenated flat
    # list plus a parallel list giving, for each flat value, the index of the original arg
    # it came from.
    ret_unwrapped = []
    ret_indices_to_original = []
    for i, a in enumerate(wrapped_args):
        a_unwrapped, _ = unwrap_tensor_subclasses(
            [a], [DummyAOTInput(9999)], append_symints=False
        )
        ret_unwrapped.extend(a_unwrapped)
        n = len(a_unwrapped)
        ret_indices_to_original.extend([i] * n)

    return ret_unwrapped, ret_indices_to_original


def remap_unwrapped_subclass_arg_indices(
    wrapped_args: list[Any], static_input_indices: list[int]
) -> list[int]:
    # [FUNCTION DOC] Maps indices of static inputs in the original arg list to indices in
    # the flattened (subclass-unwrapped, symint-appended) arg list. Each original arg expands
    # to a number of slots; only slots that hold tensors/opaques (not forwarded SymInts)
    # of a static arg are reported. Returns the new indices in increasing order.
    def static_arg_slots(arg: Any, *, include_plain_tensor_symints: bool) -> list[bool]:
        # [FUNCTION DOC] HELPER: Returns one boolean per flattened slot of arg: True for
        # tensor/opaque slots, False for forwarded SymInt slots. Raises AssertionError for
        # inner attrs that are neither Tensor nor opaque.
        # Forwarded SymInts occupy flattened arg positions, but static input indices
        # describe tensor/opaque slots whose value should not be copied.
        if not is_traceable_wrapper_subclass(arg):
            if include_plain_tensor_symints and isinstance(arg, Tensor):
                num_symints = len(enumerate_filter_symints(arg.size())) + len(
                    enumerate_filter_symints(arg.stride())
                )
                return [True] + [False] * num_symints
            return [True]

        slots = []
        attrs, _ = arg.__tensor_flatten__()
        for attr in attrs:
            inner_value = getattr(arg, attr)
            match inner_value:
                case CustomClassBase():
                    slots.append(True)
                case Tensor():
                    slots.extend(
                        static_arg_slots(inner_value, include_plain_tensor_symints=True)
                    )
                case _:
                    raise AssertionError(
                        f"expected Tensor or CustomClassBase, got {type(inner_value)}"
                    )

        slots.extend(
            [False]
            * (
                len(enumerate_filter_symints(arg.size()))
                + len(enumerate_filter_symints(arg.stride()))
            )
        )
        return slots

    static_input_indices_set = set(static_input_indices)
    new_ind = 0
    remapped_static_indices = []
    for i, arg in enumerate(wrapped_args):
        slots = (
            static_arg_slots(arg, include_plain_tensor_symints=False)
            if is_traceable_wrapper_subclass(arg)
            else [True]
        )

        for is_static_arg_slot in slots:
            if i in static_input_indices_set and is_static_arg_slot:
                remapped_static_indices.append(new_ind)

            new_ind += 1

    return remapped_static_indices


# Turns a flattened list of tensor arguments into (maybe) subclass tensors.
# This function is used both at trace time and runtime, so we have an is_runtime flag telling us which context we're in.
def wrap_tensor_subclasses(
    unwrapped_args: Sequence[Any],
    *,
    subclass_metas: list[PlainTensorMeta | SubclassCreationMeta],
    num_fw_outs_saved_for_bw: int | None = None,
    included_subclass_symints: bool = False,
    is_runtime: bool = False,
    make_subclass_override: Callable[..., Any] | None = None,
) -> tuple[Any, ...]:
    # [FUNCTION DOC] Rebuilds wrapped values from the flat list unwrapped_args using
    # subclass_metas: plain metas pick their flat value by index; subclass metas are rebuilt
    # through make_subclass_override if given, else the meta's creation_fn (with is_runtime).
    # Checks each subclass meta's included_subclass_symints equals the argument. If
    # num_fw_outs_saved_for_bw is set, the trailing activations are appended unchanged to the
    # result; otherwise all flat values must be consumed. Returns a tuple. Raises
    # AssertionError on meta type mismatch or count mismatch.
    # pyrefly: ignore [implicit-any]
    wrapped_args = []
    num_args_tallied = 0
    for subclass_meta in subclass_metas:
        if isinstance(subclass_meta, PlainTensorMeta):
            wrapped_args.append(unwrapped_args[subclass_meta.unwrapped_idx])
            num_args_tallied += subclass_meta.arg_count
        else:
            if not isinstance(subclass_meta, SubclassCreationMeta):
                raise AssertionError(
                    f"expected SubclassCreationMeta, got {type(subclass_meta)}"
                )
            if subclass_meta.included_subclass_symints != included_subclass_symints:
                raise AssertionError(
                    f"included_subclass_symints mismatch: {subclass_meta.included_subclass_symints} != {included_subclass_symints}"
                )

            if make_subclass_override:
                wrapped_args.append(
                    make_subclass_override(subclass_meta, is_runtime, unwrapped_args)
                )
            else:
                wrapped_args.append(
                    subclass_meta.creation_fn(
                        unwrapped_args,
                        is_runtime=is_runtime,
                    )
                )
            num_args_tallied += subclass_meta.arg_count

    # Note: [Partitioner handling for Subclasses, Part 2]
    # At the beginning of AOTAutograd, we collect metadata on the inputs and outputs of the user fw,
    # to figure out which inputs/outputs are subclasses, and how to reconstruct the subclasses after flattening them.
    #
    # When this function is called at runtime in the forward,
    # we have been passed a list of (flattened) dense-tensor fw-outs, and need to reconstruct any subclass fw outs.
    #
    # One reasonable question that you should ask: when should the dense_tensor -> subclass_tensor wrapping happen?
    # Answer: we do it **inside of our compiled autograd.Function**.
    # This seems like morally the right place: autograd happens above subclass desugaring,
    # so autograd should see actual tensor subclasses at runtime, and not flattened dense tensors.
    #
    # This causes a tricky interaction though: when we run the min-cut partitioner to divvy up the joint graph
    # into a forward and backward graph, we end up with some activations that show up as extra outputs
    # in the compiled forward graph, that are **not** user outputs.
    # These activations are not visible to the user, and so there's no need for us to wrap them back into subclasses.
    #
    # On top of that, when we first computed subclass metadata (in `run_functionalized_fw_and_collect_metadata`),
    # we computed subclass metadata on every forward output, but this did **not** include activations
    # created by the partitioner.
    # as a result, `unwrapped_args` here will correspond to (*unwrapped_user_fw_outs, *activations),
    # but `subclass_metas` will only correspond to subclass metadata on `user_fw_outs`.
    # We then need to make sure that we return (*wrapped_user_fw_outs, *activations).
    if num_fw_outs_saved_for_bw is not None:
        if len(unwrapped_args) != num_args_tallied + num_fw_outs_saved_for_bw:
            raise AssertionError(
                f"Expected the number actual unwrapped-subclass outputs {len(unwrapped_args)} to equal "
                f"the number of args calculated from subclasses ({num_args_tallied}) plus the number of "
                f"additional activations saved for the backward pass ({num_fw_outs_saved_for_bw})"
            )
        activations = unwrapped_args[num_args_tallied:]
        if isinstance(wrapped_args, tuple) and isinstance(activations, tuple):
            return wrapped_args + activations
        return tuple(list(wrapped_args) + list(activations))
    else:
        if len(unwrapped_args) != num_args_tallied:
            raise AssertionError(
                f"Expected {len(unwrapped_args)} == {num_args_tallied}"
            )
        return tuple(wrapped_args)


# Given a bunch of "dense" tensor arguments, this function (potentially) wraps them into tensor subclasses.
# This function carefully handles the inference vs. joint cases:
# - when is_joint_structure is True, args is (primals, tangents)
# - when is_joint_structure is False, args is [*primals]
def wrap_tensor_subclasses_maybe_joint(
    unwrapped_args: Sequence[Any],
    *,
    is_joint_structure: bool,
    meta: ViewAndMutationMeta,
) -> tuple[Any, ...]:
    # [FUNCTION DOC] Wraps flat dense args back into subclasses for either structure. Joint
    # (is_joint_structure): unwrapped_args must be a 2-tuple of (primals, tangents) lists;
    # primals use subclass_inp_meta with symints, tangents use subclass_tangent_meta without;
    # returns (wrapped_primals, wrapped_tangents). Inference: wraps the single list with
    # subclass_inp_meta and returns the wrapped tuple. Raises AssertionError on bad structure.
    # Since this function is reused for both inference and joint graphs,
    if is_joint_structure:
        if not (isinstance(unwrapped_args, tuple) and len(unwrapped_args) == 2):
            unwrapped_len = (
                len(unwrapped_args)
                if isinstance(unwrapped_args, (tuple, list))
                else "N/A"
            )
            raise AssertionError(
                f"expected tuple of length 2 for joint structure, "
                f"got {type(unwrapped_args)} with length {unwrapped_len}"
            )
        if not (
            isinstance(unwrapped_args[0], (tuple, list))
            and isinstance(unwrapped_args[1], (tuple, list))
        ):
            raise AssertionError(
                f"expected primals and tangents to be tuple or list, got {type(unwrapped_args[0])} and {type(unwrapped_args[1])}"
            )
        primals, tangents = unwrapped_args[0], unwrapped_args[1]
        wrapped_primals = wrap_tensor_subclasses(
            primals,
            subclass_metas=meta.subclass_inp_meta,
            included_subclass_symints=True,
        )
        wrapped_tangents = wrap_tensor_subclasses(
            tangents,
            subclass_metas=meta.subclass_tangent_meta,
            included_subclass_symints=False,
        )
        return (wrapped_primals, wrapped_tangents)
    else:
        wrapped_args = wrap_tensor_subclasses(
            unwrapped_args,
            subclass_metas=meta.subclass_inp_meta,
            included_subclass_symints=True,
        )
        return wrapped_args


def compute_inner_mutated_inp_indices_from_subclass_meta(
    fw_metadata: ViewAndMutationMeta,
    inner_metadata: ViewAndMutationMeta,
) -> list[int]:
    # [FUNCTION DOC] Computes the indices, in the flattened inner-graph input list, of inputs
    # mutated outside the graph. Each outer input's info is replicated across the flat
    # slots of its subclass (arg_count slots; one for plain tensors), then slots whose
    # mutation_type is MUTATED_OUT_GRAPH are returned. With no subclass_inp_meta, returns
    # inner_metadata.mutated_inp_runtime_indices as is. Raises AssertionError when lengths or
    # per-input infos of the two metadata objects are inconsistent.
    # Note: [Recomputing subclass mutation handling]
    #
    # Generally, if a subclass requires grad, its components will not require grad.
    # But for the purposes of tracking returned tensors, we should treat those component
    # tensors as if they require grad.
    #
    # For example, if the subclass tensor requires grad and will be mutated in a way that
    # requires us to handle the mutation outside of the graph, we need to return it
    # from the forward graph. The inner_meta data won't consider the component tensors
    # as if they need to be returned, because they don't require grad; but really, we
    # should handle those tensors the same way we handle the subclass tensor itself; i.e.
    # if we'd include the subclass tensor as part of the outputs, then we should also
    # include the component tensors.
    #
    # To do this, we patch num_mutated_inp_runtime_indices below by expanding the inputs
    # from the outer subclass tensors and propagating

    updated_input_info = []
    inner_idx = 0
    if not fw_metadata.subclass_inp_meta:
        # Sometimes we don't have subclass info, e.g. synthetic_base codepaths
        return inner_metadata.mutated_inp_runtime_indices
    if len(fw_metadata.subclass_inp_meta) != len(fw_metadata.input_info):
        raise AssertionError(
            f"subclass_inp_meta length ({len(fw_metadata.subclass_inp_meta)}) != input_info length ({len(fw_metadata.input_info)})"
        )
    for outer_idx, inp_meta in enumerate(fw_metadata.subclass_inp_meta):
        if isinstance(inp_meta, PlainTensorMeta):
            if outer_idx >= len(fw_metadata.input_info):
                raise AssertionError(
                    f"outer_idx ({outer_idx}) >= len(fw_metadata.input_info) ({len(fw_metadata.input_info)})"
                )
            if inner_metadata is not None:
                if inner_idx >= len(inner_metadata.input_info):
                    raise AssertionError(
                        f"inner_idx ({inner_idx}) >= len(inner_metadata.input_info) ({len(inner_metadata.input_info)})"
                    )
                if (
                    inner_metadata.input_info[inner_idx]
                    != fw_metadata.input_info[outer_idx]
                ):
                    raise AssertionError(
                        f"input_info mismatch at inner_idx={inner_idx}, outer_idx={outer_idx}: "
                        f"{inner_metadata.input_info[inner_idx]} != {fw_metadata.input_info[outer_idx]}"
                    )
            updated_input_info.append(fw_metadata.input_info[outer_idx])
            inner_idx += 1
        else:
            if inp_meta.original_subclass is None:
                raise AssertionError(
                    "inp_meta.original_subclass must not be None for SubclassCreationMeta"
                )
            for _ in range(inp_meta.arg_count):
                updated_input_info.append(fw_metadata.input_info[outer_idx])
                inner_idx += 1
    if inner_metadata is not None:
        if len(inner_metadata.input_info) != len(updated_input_info):
            raise AssertionError(
                f"inner_metadata.input_info length ({len(inner_metadata.input_info)}) "
                f"!= updated_input_info length ({len(updated_input_info)})"
            )

    return [
        i
        for i, inp in enumerate(updated_input_info)
        if inp.mutation_type == MutationType.MUTATED_OUT_GRAPH
    ]
