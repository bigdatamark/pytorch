# [MODULE DOC] Core of Inductor lowering: translates ATen/prim FX graph nodes into Inductor IR.
# Holds the global registries (lowerings, user_lowerings, fallbacks, needs_realized_inputs,
# foreach_ops, layout-constraint table) and the register_lowering decorator machinery that wraps
# each lowering with argument broadcasting, type promotion and IR validation. Then defines
# lowerings for pointwise, reduction, view, indexing, scatter, pooling, collective and other ops.
# Import-time side effects: populates the registries, registers lowerings and fallbacks, and
# imports submodules for additional lowerings at the end.
from __future__ import annotations

import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import operator
import os
import sys
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, cast, Literal, TYPE_CHECKING, TypeGuard, TypeVar
from typing_extensions import ParamSpec
from unittest.mock import patch

import sympy

import torch
import torch.ao.quantization.fx._decomposed
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.utils import counters
from torch._functorch._aot_autograd.descriptors import (
    SavedForBackwardsAOTOutput,
    SavedForBackwardsNoVcCheckAOTOutput,
)
from torch._higher_order_ops.associative_scan import associative_scan_op
from torch._higher_order_ops.triton_kernel_wrap import triton_kernel_wrapper_mutation
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import is_custom_class_obj
from torch._library.utils import get_layout_constraint_tag
from torch._prims_common import (
    canonicalize_dim,
    canonicalize_dims,
    check,
    dtype_to_type,
    elementwise_dtypes,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    get_computation_dtype,
    is_boolean_dtype,
    is_float_dtype,
    is_integer_dtype,
    Number,
)
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.fx.experimental.symbolic_shapes import (
    free_unbacked_symbols,
    has_free_unbacked_symbols,
    resolve_unbacked_bindings,
    SymTypes,
)
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import (
    CeilDiv,
    FloorDiv,
    Identity,
    Max,
    Min,
    Mod,
    ModularIndexing,
)
from torch.utils._triton import has_triton_reduction_ordering

from .._dynamo.utils import import_submodule
from . import config, inductor_prims, ir, test_operators  # NOQA: F401
from .decomposition import decompositions, get_decompositions
from .ir import (
    BaseView,
    DtypeView,
    ExpandView,
    IndexingConstant,
    IRNode,
    is_triton,
    MutableBox,
    OnlineSoftmaxReduction,
    ops_wrapper,
    PermuteView,
    Pointwise,
    Reduction,
    SqueezeView,
    TensorBox,
    validate_ir,
    View,
)
from .utils import (
    ceildiv,
    convert_symint_to_expr,
    decode_device,
    is_dynamic,
    is_gpu,
    is_nvidia_sm100_or_later,
    is_pointwise_use,
    is_triton_fp8_dtype_supported,
    is_view,
    needs_fallback_due_to_atomic_add_limitations,
    pad_listlike,
    register_op_dtype_propagation_rules,
    register_op_requires_libdevice_fp64,
    sympy_product,
    use_scatter_fallback,
)
from .virtualized import ops, V


if TYPE_CHECKING:
    from .ir import BaseConstant
    from .ops_handler import ReductionType


_T = TypeVar("_T")
_P = ParamSpec("_P")

# TODO(jansel): we should implement decomps or lowerings for these
# https://github.com/pytorch/torchdynamo/issues/327
FALLBACK_ALLOW_LIST = OrderedSet(
    [
        "torchvision::roi_align",
        "aten::index_add",
    ]
)

log = logging.getLogger(__name__)
lowerings: dict[Callable[..., Any] | str, Callable[..., Any]] = {}
# User-registered lowerings that take priority over built-in lowerings.
user_lowerings: dict[torch._ops.OpOverload, Callable[..., Any]] = {}
# Use maybe_layout_constraints to access this dict, we lazily register tag-based layout constraints
_maybe_layout_constraints: dict[torch._ops.OpOverload, Callable[..., Any] | None] = {}
fallbacks = OrderedSet[torch._ops.OpOverload]()
aten = torch.ops.aten
tr_c10d = torch.ops.tr_c10d
prims = torch.ops.prims
needs_realized_inputs = OrderedSet[torch._ops.OpOverload]()
foreach_ops = OrderedSet[torch._ops.OpOverload](
    [torch._higher_order_ops._foreach_map]  # type: ignore[list-item]
)
# TODO(rec): torch._higher_order_ops._foreach_map is not an OpOverload
# so why is it in foreach_ops?
inplace_foreach_ops = OrderedSet[torch._ops.OpOverload]()
inplaceable_foreach_ops: dict[torch._ops.OpOverload, torch._ops.OpOverload] = {}
quantized_decomposed = torch.ops.quantized_decomposed
_SAVED_FOR_BACKWARDS_OUTPUT_DESC_TYPES = (
    SavedForBackwardsAOTOutput,
    SavedForBackwardsNoVcCheckAOTOutput,
)


def cur_node_has_non_foreach_users() -> bool:
    # [FUNCTION DOC] HELPER: Return True if the current graph node has any user whose users include a
    # non-foreach call_function node; False otherwise. Reads V.graph.current_node.
    for node in V.graph.current_node.users:
        for user in node.users:
            if not (user.op == "call_function" and (user.target in foreach_ops)):
                return True

    return False


# group by device, whether any of the inputs are dynamic
# note arg_pairs may or may not be a pair
# foreach_map for example just passes output buffers here
def group_foreach_args(
    arg_pairs: Iterable[Any],
) -> defaultdict[tuple[Any, bool], list[tuple[int, Any]]]:
    # [FUNCTION DOC] HELPER: Group foreach arguments by (device, use_foreach flag). arg_pairs is an
    # iterable whose items are either iterables of args or single values. Returns a mapping from
    # (device, bool) to a list of (original index, args) tuples. use_foreach is true when inputs are
    # static or config.combo_kernel_foreach_dynamic_shapes is set. Raises an error if an item has no
    # TensorBox to take a device from.
    out = defaultdict(list)
    unpack_args = False
    for i, args in enumerate(arg_pairs):
        if not isinstance(args, Iterable):
            unpack_args = True
            args = (args,)
        use_foreach = (
            not is_dynamic(*args) or config.combo_kernel_foreach_dynamic_shapes
        )
        device = None
        for t in args:
            if isinstance(t, TensorBox):
                device = t.data.get_device()
                break
        if device is None:
            raise AssertionError("foreach op should have at least one tensor arg")
        if unpack_args:
            (args,) = args
        out[(device, use_foreach)].append((i, args))
    return out


def maybe_layout_constraints(fn: Callable[..., Any]) -> Callable[..., Any] | None:
    # [FUNCTION DOC] Return the layout-constraint function for an op, or None. Only OpOverloads have
    # constraints: a layout tag on the op takes priority, otherwise the explicit registry is consulted.
    """Get layout constraints. Returns None if there are no layout constraints."""
    if not isinstance(fn, torch._ops.OpOverload):
        # Only OpOverloads have layout constraints.
        return None

    if maybe_layout_tag := get_layout_constraint_tag(fn, with_default=False):
        return tag_to_layout_constraint(maybe_layout_tag)

    if fn in _maybe_layout_constraints:
        return _maybe_layout_constraints[fn]
    return None


def tag_to_layout_constraint(
    tag: torch._C.Tag,
) -> Callable[..., tuple[Any, Any]] | None:
    # [FUNCTION DOC] HELPER: Map a layout-constraint tag to its constraint function (exact strides,
    # contiguous strides, fixed stride order) or None for flexible layout. Raises an error for unknown tags.
    if tag == torch._C.Tag.needs_exact_strides:
        return constrain_to_fake_tensors
    if tag == torch._C.Tag.needs_contiguous_strides:  # type: ignore[attr-defined]
        return require_contiguous_strides
    if tag == torch._C.Tag.needs_fixed_stride_order:
        return constrain_to_fx_strides
    if tag == torch._C.Tag.flexible_layout:
        return None
    raise AssertionError(f"Unknown layout constraint tag: {tag}")


def assert_nyi(cond: bool, msg: str) -> None:
    # [FUNCTION DOC] HELPER: Raise NotImplementedError saying inductor does not support msg when cond is false.
    if not cond:
        raise NotImplementedError(f"inductor does not support {msg}")


def add_needs_realized_inputs(fn):
    # [FUNCTION DOC] HELPER: Mark an op (or each op in a list/set/tuple, recursively) as needing realized
    # inputs by adding it to needs_realized_inputs; for an overload packet all its overloads are added.
    # Mutates global state. Returns a list only for the collection case, else None.
    if isinstance(fn, (list, set, tuple, OrderedSet)):  # noqa: set_linter
        return [add_needs_realized_inputs(x) for x in fn]
    needs_realized_inputs.add(fn)
    if isinstance(fn, torch._ops.OpOverloadPacket):
        needs_realized_inputs.update(fn.op_overloads())


def add_layout_constraint(
    fn: torch._ops.OpOverloadPacket | torch._ops.OpOverload,
    constraint: Callable[..., tuple[Any, Any]],
) -> None:
    # [FUNCTION DOC] HELPER: Register a layout constraint function for an op; for an overload packet
    # it is registered for every overload. Mutates the layout-constraint registry.
    if isinstance(fn, torch._ops.OpOverloadPacket):
        for overload in fn.op_overloads():
            _maybe_layout_constraints[overload] = constraint
    else:
        _maybe_layout_constraints[fn] = constraint


add_needs_realized_inputs(
    [
        aten.as_strided,
        aten.as_strided_copy,
        aten.avg_pool2d,
        aten.avg_pool2d_backward,
        aten.bmm,
        aten.convolution,
        aten.convolution_backward,
        aten.max_pool2d_with_indices,
        aten.max_pool3d_with_indices,
        aten.max_pool2d_with_indices_backward,
        aten.mm,
        aten.upsample_nearest2d,
        aten._upsample_nearest_exact2d,
        aten._int_mm,
    ]
)

# TODO(jansel): ezyang says we won't need this in the future, try removing it
# based on https://github.com/pytorch/pytorch/blob/9e3eb329df8f701/c10/core/ScalarType.h#L28
DTYPE_ID_LOOKUP = {
    0: torch.uint8,
    1: torch.int8,
    2: torch.int16,
    3: torch.int32,
    4: torch.int64,
    5: torch.float16,
    6: torch.float32,
    7: torch.float64,
    8: torch.complex32,
    9: torch.complex64,
    10: torch.complex32,
    11: torch.bool,
    15: torch.bfloat16,
    # TODO(jansel): add quantized types?
    #  _(c10::qint8, QInt8) /* 12 */
    # _(c10::quint8, QUInt8) /* 13 */
    # _(c10::qint32, QInt32) /* 14 */
    # _(c10::quint4x2, QUInt4x2) /* 16 */
    # _(c10::quint2x4, QUInt2x4) /* 17 */
}


def decode_dtype(dtype: int | torch.dtype) -> torch.dtype:
    # [FUNCTION DOC] HELPER: Convert an integer dtype id to a torch dtype using DTYPE_ID_LOOKUP; non-int
    # inputs are returned unchanged. Raises an error if the id is not in the table.
    if not isinstance(dtype, int):
        return dtype
    if dtype not in DTYPE_ID_LOOKUP:
        raise AssertionError(f"id {dtype} missing from DTYPE_ID_LOOKUP")

    dtype = DTYPE_ID_LOOKUP[dtype]
    return dtype


def is_integer_type(x: Any) -> TypeGuard[TensorBox | IRNode | sympy.Expr | int]:
    # [FUNCTION DOC] HELPER: True if x is an integer-valued thing: a tensor/IR node of integer or bool
    # dtype, a sympy expression known to be integer, or a Python int.
    if isinstance(x, (TensorBox, IRNode)):
        return is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
    elif isinstance(x, sympy.Expr):
        return x.is_integer is True  # type: ignore[attr-defined]
    else:
        return isinstance(x, int)


def is_boolean_type(x: Any) -> TypeGuard[TensorBox | IRNode | bool]:
    # [FUNCTION DOC] HELPER: True if x is a boolean: a tensor/IR node with bool dtype or a Python bool.
    if isinstance(x, (TensorBox, IRNode)):
        return is_boolean_dtype(x.get_dtype())
    else:
        return isinstance(x, bool)


def get_promoted_dtype(
    *args: Any,
    type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND,
    return_compute_dtype: bool = False,
) -> torch.dtype:
    # [FUNCTION DOC] HELPER: Compute the dtype that results from elementwise type promotion of args
    # (IR tensors, numbers or sympy values) under type_promotion_kind. Returns the compute dtype if
    # return_compute_dtype is set, else the result dtype.
    def construct_input(inp: Any) -> Any:
        # [FUNCTION DOC] HELPER: Turn an arg into something usable for type promotion: numbers and sympy
        # values pass through, IR tensors become a placeholder tensor of the same rank and dtype.
        if isinstance(inp, (Number, sympy.Basic)):
            return inp
        else:
            dim = len(inp.get_size())
            # construct a tmp tensor to feed into torch.result_type
            return torch.zeros([1] * dim, dtype=inp.get_dtype())

    inps = [construct_input(arg) for arg in args]
    compute_dtype, result_dtype = elementwise_dtypes(
        *inps, type_promotion_kind=type_promotion_kind
    )
    return compute_dtype if return_compute_dtype else result_dtype


def get_overloads(aten_fn):
    # [FUNCTION DOC] HELPER: Normalize aten_fn to a list and append every overload of any overload
    # packet in it that does not already have a lowering. Returns the list.
    if not isinstance(aten_fn, (list, tuple)):
        aten_fn = [aten_fn]
    else:
        aten_fn = list(aten_fn)

    for fn in list(aten_fn):
        if isinstance(fn, torch._ops.OpOverloadPacket):
            for other_fn in fn.op_overloads():
                if other_fn not in lowerings:
                    aten_fn.append(other_fn)

    return aten_fn


def in_namespace(
    op: Any | torch._ops.OpOverloadPacket | torch._ops.OpOverload, namespace: str
) -> bool:
    # [FUNCTION DOC] HELPER: True if op is an overload packet or overload whose qualified name contains namespace.
    if isinstance(op, torch._ops.OpOverloadPacket):
        return namespace in op._qualified_op_name
    elif isinstance(op, torch._ops.OpOverload):
        return namespace in op.name()
    return False


def maybe_copy_cpu_scalar(x: TensorBox, device: torch.device) -> TensorBox:
    # [FUNCTION DOC] Copy a single-element CPU scalar tensor (a reinterpret view with concrete size 0 or
    # one element) to the target device when its device differs; otherwise return x unchanged.
    """
    Copy cpu scalar if doesn't not match with given `device`
    """
    if not isinstance(x.data, ir.ReinterpretView) or has_free_unbacked_symbols(
        x.get_size()
    ):
        return x
    size = V.graph.sizevars.guarding_hints_or_throw(x.get_size())
    cur_device = x.get_device()
    if (
        cur_device is not None
        and cur_device.type == "cpu"
        and cur_device != device
        and (len(size) == 0 or (len(size) == 1 and size[0] == 1))
    ):
        return TensorBox(ir.StorageBox(ir.DeviceCopy.create(x, cur_device, False)))
    return x


def transform_args(
    args: list[Any],
    kwargs: dict[str, Any],
    broadcast: bool,
    type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND | None,
    convert_input_to_bool: bool,
) -> tuple[list[Any], dict[str, Any]]:
    # [FUNCTION DOC] Prepare args and kwargs for a lowering. Only TensorBox values are transformed.
    # When type promotion or bool conversion is requested, computes the promoted (or bool) dtype,
    # moves CPU scalars to the common device, and converts tensors/constants to that dtype. When
    # broadcast is set, broadcasts all tensors to a common shape and expands constants to it.
    # Returns the new (args, kwargs); returns inputs unchanged if there are no tensors.
    """
    Transforms arguments for broadcasting and type promotion
    """

    args_indices = [i for i, x in enumerate(args) if isinstance(x, TensorBox)]
    kwargs_indices = [k for k, v in kwargs.items() if isinstance(v, TensorBox)]
    # check that there's something to transform
    if not args_indices and not kwargs_indices:
        return args, kwargs

    if type_promotion_kind or convert_input_to_bool:
        if convert_input_to_bool:
            dtype = torch.bool
        else:
            # FIXME this is a crude approximation for promoting args
            promoting_args = [
                a
                for a in args
                if isinstance(a, (Number, sympy.Basic)) or hasattr(a, "dtype")
            ]
            # only consider tensor kwargs for promotion, for now
            promoting_args.extend(a for a in kwargs.values() if hasattr(a, "dtype"))
            dtype = get_promoted_dtype(
                *promoting_args,
                type_promotion_kind=type_promotion_kind,  # type: ignore[arg-type]
            )

        device = (
            args[args_indices[0]] if args_indices else kwargs[kwargs_indices[0]]
        ).get_device()

        for i in args_indices:
            args[i] = maybe_copy_cpu_scalar(args[i], device)

        for k in kwargs_indices:
            kwargs[k] = maybe_copy_cpu_scalar(kwargs[k], device)

        # sometimes args are an immutable list so we can't mutate them
        def promote(arg: Any) -> Any:
            # [FUNCTION DOC] HELPER: Cast a TensorBox to the promoted dtype, rebuild a Constant with that dtype and device, pass other values through.
            if isinstance(arg, TensorBox):
                return to_dtype(arg, dtype)
            elif isinstance(arg, ir.Constant):
                return ir.Constant(value=arg.value, dtype=dtype, device=device)
            else:
                return arg

        args = [promote(a) for a in args]
        kwargs = {k: promote(v) for k, v in kwargs.items()}

    if broadcast:
        broadcasted = broadcast_tensors(
            *list(
                itertools.chain(
                    (args[i] for i in args_indices),
                    (kwargs[k] for k in kwargs_indices),
                )
            )
        )
        size = list(broadcasted[0].get_size())

        for i, x in zip(args_indices, broadcasted[: len(args_indices)]):
            args[i] = x
        for k, x in zip(kwargs_indices, broadcasted[len(args_indices) :]):
            kwargs[k] = x

        for i in range(len(args)):
            if isinstance(args[i], ir.Constant):
                args[i] = ExpandView.create(args[i], size)
        for k in kwargs:
            if isinstance(kwargs[k], ir.Constant):
                kwargs[k] = ExpandView.create(kwargs[k], size)

    return args, kwargs


def _register_foreach_lowering(
    aten_fn: torch._ops.OpOverload, decomp_fn: Callable[..., Any]
) -> Callable[..., Any]:
    # [FUNCTION DOC] HELPER: Register decomp_fn as the lowering for a foreach aten op (all overloads):
    # wraps it to validate the produced IR, records the overloads in foreach_ops and lowerings.
    # Returns the wrapped function.
    """
    Add a foreach lowering to lowerings dict.

    Arguments:
        aten_fn: torch.ops.aten.* fn we are lowering
        decomp_fn: alternate implementation on our IR
        broadcast: True to apply broadcasting to tensor inputs
        type_promotion_kind: kind of type promotion applied to tensor inputs, `None` means no type promotion
        convert_input_to_bool: some logical ops require inputs are converted to bool
    """

    @functools.wraps(decomp_fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] HELPER: Call the foreach lowering and validate the IR it returns.
        out = decomp_fn(*args, **kwargs)
        validate_ir(out)
        return out

    aten_fns = get_overloads(aten_fn)
    foreach_ops.update(aten_fns)
    lowerings.update(dict.fromkeys(aten_fns, wrapped))
    return wrapped


def _register_lowering(
    aten_fn,
    decomp_fn: Callable[..., Any],
    broadcast: bool,
    type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND | None,
    convert_input_to_bool: bool,
    lowering_dict: dict[Callable[..., Any] | str, Callable[..., Any]],
):
    # [FUNCTION DOC] HELPER: Register decomp_fn as lowering for aten_fn (all overloads) in lowering_dict.
    # The wrapper unpacks a single list/tuple argument, rejects out= kwargs unless all ops are
    # fallbacks or c10d functional ops, applies transform_args (broadcast, type promotion, bool
    # conversion), calls decomp_fn, validates the IR, and returns it. Returns the wrapper.
    """
    Add a lowering to lowerings dict

    Arguments:
        aten_fn: torch.ops.aten.* fn we are lowering
        decomp_fn: alternate implementation on our IR
        broadcast: True to apply broadcasting to tensor inputs
        type_promotion_kind: kind of type promotion applied to tensor inputs, `None` means no type promotion
        convert_input_to_bool: some logical ops require inputs are converted to bool
    """

    @functools.wraps(decomp_fn)
    def wrapped(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper that normalizes arguments via transform_args, calls the lowering, and validates the output IR.
        # Raises an error if out= is passed for non-fallback ops.
        args: list[Any] = list(args)
        kwargs: dict[str, Any] = dict(kwargs)
        unpacked = False
        # TODO maybe we need to use pytrees here
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            unpacked = True
            args = list(args[0])

        if not all(
            (fn in fallbacks or in_namespace(fn, "_c10d_functional")) for fn in aten_fn
        ):
            # explicitly assert for "out=" ops for better error messages
            if any(x == "out" for x in kwargs):
                raise AssertionError("out= ops aren't yet supported")

        args, kwargs = transform_args(
            args, kwargs, broadcast, type_promotion_kind, convert_input_to_bool
        )

        if unpacked:
            args = [args]

        out = decomp_fn(*args, **kwargs)
        validate_ir(out)

        return out

    aten_fn = get_overloads(aten_fn)

    lowering_dict.update(dict.fromkeys(aten_fn, wrapped))
    return wrapped


def register_lowering(
    aten_fn,
    broadcast=False,
    type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND
    | None = ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    convert_input_to_bool=False,
    lowering_dict=lowerings,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    # [FUNCTION DOC] Decorator factory: returns a partial of _register_lowering configured with the given
    # op(s), broadcast, type promotion kind (default DEFAULT), bool conversion and target dict.
    """
    Shim to support decorator syntax.
    """
    return functools.partial(
        _register_lowering,
        aten_fn,
        broadcast=broadcast,
        type_promotion_kind=type_promotion_kind,
        convert_input_to_bool=convert_input_to_bool,
        lowering_dict=lowering_dict,
    )


def broadcast_symbolic_shapes(a, b):
    # [FUNCTION DOC] Broadcast two symbolic shapes. Aligns from the right, treating size-one dims as
    # broadcastable, otherwise requires equal sizes via sizevars (may add guards) and prefers the
    # simpler expression. Returns the broadcast shape as a tuple.
    """
    Broadcasting logic based on symbolic shapes.

    We give the shapes 0 and 1 concrete values, while all other shapes
    are symbolic sympy formulas.
    """
    b = tuple(b)
    if not a or a == b:
        return b

    output = []
    for x, y in itertools.zip_longest(reversed(a), reversed(b), fillvalue=sympy.S.One):
        if V.graph.sizevars.is_size_one_or_false(y):
            output.append(x)
        elif V.graph.sizevars.is_size_one_or_false(x):
            output.append(y)
        else:
            V.graph.sizevars.check_equals(x, y)
            if len(sympy.expand(y).free_symbols) < len(sympy.expand(x).free_symbols):
                output.append(y)  # prefer shorter formula
            else:
                output.append(x)
    return tuple(reversed(output))


def promote_constants(
    inputs: Sequence[_T],
    override_return_dtype: torch.dtype | None = None,
    type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND | None = None,
    round_scalar_constants: bool = False,
    round_scalars_to_tensor_dtype: bool = False,
) -> Sequence[_T | BaseView | BaseConstant]:
    # [FUNCTION DOC] Convert raw Python scalars and sympy expressions in inputs into IR constants.
    # If no input is a scalar/sympy value, inputs are returned unchanged. If all are scalars, each becomes a
    # Constant/IndexingConstant of the overridden or promoted dtype. Otherwise scalars become constants
    # of the first tensor's dtype expanded to its size, optionally rounded to fp16/bf16 to mimic eager.
    # Raises an error if both override_return_dtype and type_promotion_kind are given.
    """Convert raw Python scalars and sympy expressions in inputs to IR constants.

    When a tensor input is present, scalars become Constants of the tensor's
    dtype, broadcast to its size. For bf16/fp16 tensors, the scalar value is
    additionally rounded to the tensor dtype to match eager kernels that cast
    scalar operands to the common dtype: always for comparison ops
    (override_return_dtype == torch.bool) and for callers passing
    round_scalar_constants (e.g. remainder); on CPU and MPS only for ops
    passing round_scalars_to_tensor_dtype (e.g. add/sub, whose CUDA eager
    kernels keep scalars at opmath precision).
    """
    if not (override_return_dtype is None or type_promotion_kind is None):
        raise AssertionError(
            "only one of override_return_dtype or type_promotion_kind may be given"
        )

    if override_return_dtype is None and type_promotion_kind is None:
        type_promotion_kind = ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT

    if not any(isinstance(x, (sympy.Basic, int, float)) for x in inputs):
        return inputs
    if all(isinstance(x, (int, float, sympy.Basic)) for x in inputs):
        dtype = override_return_dtype or get_promoted_dtype(
            *inputs,
            # pyrefly: ignore [bad-argument-type]
            type_promotion_kind=type_promotion_kind,
        )

        def const_func(x):
            # [FUNCTION DOC] HELPER: Wrap a scalar as a Constant (or an IndexingConstant for sympy expressions) with the promoted dtype on the default device.
            if isinstance(x, sympy.Basic):
                return ir.IndexingConstant(
                    index=x, dtype=dtype, device=decode_device(None)
                )
            else:
                return ir.Constant(value=x, dtype=dtype, device=decode_device(None))

        return [const_func(x) for x in inputs]
    ex = next(x for x in inputs if isinstance(x, (TensorBox, ExpandView, ir.Constant)))
    tensor_dtype = ex.get_dtype()

    # Round scalars to the tensor's dtype where eager does; see docstring.
    if tensor_dtype in (
        torch.bfloat16,
        torch.float16,
    ) and (
        override_return_dtype == torch.bool
        or round_scalar_constants
        or (
            round_scalars_to_tensor_dtype
            and ex.get_device_or_error().type in ("cpu", "mps")
        )
    ):
        _round_scalar = lambda v: torch.tensor(v, dtype=tensor_dtype).item()  # noqa: E731
    else:
        _round_scalar = lambda v: v  # noqa: E731

    out = []
    for x in inputs:
        if isinstance(x, (int, float)):
            out.append(
                ExpandView.create(
                    ir.Constant(
                        value=_round_scalar(x),
                        dtype=tensor_dtype,
                        device=ex.get_device_or_error(),
                    ),
                    list(ex.get_size()),
                )
            )
        elif isinstance(x, sympy.Basic):
            out.append(
                ExpandView.create(
                    IndexingConstant(
                        index=x, dtype=tensor_dtype, device=ex.get_device_or_error()
                    ),
                    list(ex.get_size()),
                )
            )
        else:
            out.append(x)

    return out


def _add_with_alpha_fma(a, b, alpha):
    # [FUNCTION DOC] HELPER: Build a Pointwise node computing a + alpha * b with a fused multiply-add
    # (fma(b, alpha, a)) in the promoted dtype, to match eager CUDA rounding. alpha may be a number or sympy expression.
    """Compute a + alpha * b using FMA for CUDA floating-point precision."""
    dtype = get_promoted_dtype(
        a,
        b,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    )
    a_loader = a.make_loader()
    b_loader = b.make_loader()

    def inner_fn(idx):
        # [FUNCTION DOC] HELPER: Load a and b at idx and return fma of b, the alpha constant and a.
        a_val = a_loader(idx)
        b_val = b_loader(idx)
        if isinstance(alpha, sympy.Basic):
            alpha_expr = ops.index_expr(alpha, dtype)
        else:
            alpha_expr = ops.constant(alpha, dtype)
        return ops.fma(b_val, alpha_expr, a_val)

    return Pointwise.create(
        device=a.get_device(),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=a.get_size(),
    )


def _has_saved_lowp_output_use(node: torch.fx.Node | None) -> bool:
    # [FUNCTION DOC] HELPER: True if the fx node has a non-output user and is also an output that
    # the AOT descriptors mark as saved-for-backward; False for None or otherwise.
    if node is None:
        return False

    has_non_output_user = False
    for user in node.users:
        if user.op != "output":
            has_non_output_user = True
            break

    if not has_non_output_user:
        return False

    for user in node.users:
        if user.op != "output":
            continue

        descs = user.meta.get("desc")
        if descs is None:
            continue

        output_values = pytree.tree_leaves(user.args[0])
        for output_value, desc in zip(output_values, descs):
            if output_value is node and isinstance(
                desc, _SAVED_FOR_BACKWARDS_OUTPUT_DESC_TYPES
            ):
                return True

    return False


def make_pointwise(
    fn: Callable[..., Any],
    override_return_dtype: torch.dtype | None = None,
    override_device: torch.device | None = None,
    override_fn_when_input_bool: Callable[..., Any] | None = None,
    allow_alpha: bool = False,
    use_fma_for_alpha: bool = False,
    triton_fallback: Callable[..., _T] | None = None,
    round_scalars_to_tensor_dtype: bool = False,
) -> Callable[..., TensorBox | _T]:
    # [FUNCTION DOC] Wrap an elementwise ops function fn into a lowering that builds a Pointwise IR node
    # (define-by-run). Options: fixed output dtype/device, alternative fn for bool outputs, alpha support
    # (optionally via FMA on CUDA floats), a triton fallback used for triton-backed inputs, and rounding of
    # scalars to tensor dtype. Returns the callable taking IR inputs.
    """Wraps a pointwise fn and returns a function representing the pointwise in
    the define-by-run IR."""

    def inner(*inputs: Any, alpha=None) -> TensorBox | _T:  # noqa: docstring_linter
        # [FUNCTION DOC] HELPER: Lowering body: dispatch to triton_fallback if applicable, promote scalar constants,
        # apply alpha by multiplying the last input (or FMA path), then create a Pointwise node over the first
        # input's size. Optionally emulates fp16/bf16 rounding casts for precision-emulation configs. Picks the
        # GPU device among inputs if any. Raises an error on alpha misuse or ndim mismatch.
        if triton_fallback is not None and any(
            isinstance(inp, IRNode) and is_triton(inp) for inp in inputs
        ):
            if allow_alpha:  # not implemented
                raise AssertionError("expected: not allow_alpha")
            return triton_fallback(*inputs)

        # pyrefly: ignore [bad-assignment]
        inputs = promote_constants(
            inputs,
            override_return_dtype,
            round_scalars_to_tensor_dtype=round_scalars_to_tensor_dtype,
        )
        if allow_alpha:
            if alpha is not None and alpha != 1:
                # Use FMA for add-with-alpha on Triton GPU floating-point.
                # Eager CUDA/ROCm computes a + alpha * b as fma(b, alpha, a).
                if use_fma_for_alpha and isinstance(inputs[0], IRNode):
                    inp_device = inputs[0].get_device()
                    if (
                        inputs[0].get_dtype().is_floating_point
                        and inp_device is not None
                        and inp_device.type == "cuda"
                    ):
                        return _add_with_alpha_fma(inputs[0], inputs[1], alpha)

                # pyrefly: ignore [bad-assignment]
                inputs = list(inputs)
                # pyrefly: ignore [unsupported-operation]
                inputs[-1] = mul(inputs[-1], alpha)
        else:
            if alpha is not None:
                raise AssertionError("expected: alpha is None")
        loaders = [x.make_loader() for x in inputs]
        ranges = inputs[0].get_size()
        dtype = override_return_dtype or inputs[0].get_dtype()

        for other in inputs[1:]:
            if not (
                isinstance(other, ir.BaseConstant)
                or len(ranges) == len(other.get_size())
            ):
                raise AssertionError(f"ndim mismatch {fn} {ranges} {other.get_size()}")

        # in tracing, we will annotate pointwise nodes that correspond to the output of
        # a pointwise node that would have been run in eager. intermediary pointwise nodes
        # during decompositions are not annotated.
        low_pr_fp = (torch.bfloat16, torch.float16)
        current_node = (
            getattr(V.graph, "current_node", None) if V.graph is not None else None
        )
        emulate_precision_casts = (
            current_node is not None
            and current_node.meta is not None
            and current_node.meta.get("low_precision_pointwise_barrier", False)
        )
        truncate_saved_output = (
            config.emulate_precision_casts_on_saved_tensors
            and not emulate_precision_casts
            and dtype in low_pr_fp
            and _has_saved_lowp_output_use(current_node)
        )
        emulate_output_cast = (
            emulate_precision_casts or truncate_saved_output
        ) and dtype in low_pr_fp

        def inner_fn(index):
            # [FUNCTION DOC] HELPER: Pointwise inner function: load each input at index, apply fn (or the bool override), and optionally
            # round through low precision dtypes. Raises an error if the index rank differs from ranges.
            if len(index) != len(ranges):
                raise AssertionError(f"wrong ndim {index} {ranges}")
            if dtype == torch.bool and override_fn_when_input_bool is not None:
                return override_fn_when_input_bool(*[load(index) for load in loaders])
            else:
                inputs_loaded = []
                for inp_index, load in enumerate(loaders):
                    out = load(index)
                    inp_dtype = inputs[inp_index].get_dtype()
                    if emulate_precision_casts and inp_dtype in low_pr_fp:
                        downcast = ops.to_dtype(out, inp_dtype, use_compute_types=False)
                        out = ops.to_dtype(downcast, inp_dtype)
                    inputs_loaded.append(out)

                out = fn(*inputs_loaded)
                if emulate_output_cast:
                    # fp16/bf16 kernels are computed in fp32. Casting down to fp16/bf16 here,
                    # then upcasting again, to emulate casts that eager would do.
                    downcast = ops.to_dtype(out, dtype, use_compute_types=False)
                    return ops.to_dtype(downcast, dtype)
                return out

        if not override_device:
            device = None
            for i in inputs:
                # pyrefly: ignore [missing-attribute]
                if is_gpu(i.get_device().type):
                    device = i.get_device()
                    break
            if not device:
                device = inputs[0].get_device()

        # pyrefly: ignore [unbound-name]
        device = override_device or device

        return Pointwise.create(
            device=device,  # type: ignore[arg-type]
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=ranges,
        )

    return inner


foreach_node_t = TypeVar("foreach_node_t", bound=IRNode)


def make_foreach_pointwise(
    pw_fn: Callable[..., foreach_node_t],
    allow_alpha: bool = False,
    scalar_kwarg: str = "alpha",
) -> Callable[..., list[foreach_node_t]]:
    # [FUNCTION DOC] HELPER: Lift a pointwise lowering pw_fn to foreach ops over lists of tensors. Returns a callable
    # that accepts list or scalar inputs, broadcasts scalars to list length, groups entries by device, applies pw_fn
    # per element and returns a list of outputs. scalar_kwarg names the keyword receiving alpha/value.
    def inner(*inputs: list[list[TensorBox]], alpha=1, value=1) -> list[foreach_node_t]:
        # [FUNCTION DOC] HELPER: Foreach wrapper body. Extracts the scalar (alpha, or value which may be a trailing positional),
        # decides whether outputs must be realized (no users, inplace foreach, or non-foreach users), then runs the grouped loop.
        # Raises an error if no input is a list.
        # For ops like addcmul/addcdiv, the scalar `value` arrives as a
        # positional arg (not keyword) due to the ATen schema. Extract it
        # from the end of inputs if present.
        # pyrefly: ignore [bad-assignment]
        inputs = list(inputs)
        if (
            scalar_kwarg == "value"
            and inputs
            and not isinstance(inputs[-1], (list, tuple))
        ):
            # pyrefly: ignore [missing-attribute]
            scalar_val = inputs.pop()
        elif scalar_kwarg == "value":
            scalar_val = value
        else:
            scalar_val = alpha

        realize_outputs = (
            len(V.graph.current_node.users) == 0
            or V.graph.current_node.target in inplace_foreach_ops
            or cur_node_has_non_foreach_users()
        )

        a_list_input = None
        for input in inputs:
            if isinstance(input, (list, tuple)):
                a_list_input = input
                break
        if a_list_input is None:
            raise AssertionError("at least one input must be a list to a foreach op")

        # broadcast scalar inputs to match length of list inputs
        broadcast_inputs = []
        for input in inputs:
            if not isinstance(input, (list, tuple)):
                broadcast_inputs.append([input] * len(a_list_input))
            else:
                # pyrefly: ignore [bad-argument-type]
                broadcast_inputs.append(input)

        groups = group_foreach_args(zip(*broadcast_inputs))

        def apply_fn(args) -> foreach_node_t:
            # [FUNCTION DOC] HELPER: Apply pw_fn to one element's args, passing the scalar keyword when alpha is allowed.
            if allow_alpha:
                return pw_fn(*args, **{scalar_kwarg: scalar_val})
            else:
                return pw_fn(*args)

        return foreach_group_loop(groups, len(a_list_input), apply_fn, realize_outputs)

    return inner


def foreach_group_loop(
    groups: Mapping[tuple[Any, bool], list[tuple[int, Any]]],
    num_outputs: int,
    apply_fn: Callable[..., foreach_node_t],
    realize_outputs: bool,
) -> list[foreach_node_t]:
    # [FUNCTION DOC] HELPER: Apply apply_fn to every grouped element, storing results by original index. When the
    # backend supports foreach, grouping uses foreach and realize_outputs is set, outputs are realized and
    # registered as a horizontally fusable operation list. Returns the list of outputs; raises an error if any is missing.
    """
    Common loop over grouped foreach arguments.

    Args:
        groups: Result of group_foreach_args - dict mapping (device, use_foreach) to groups
        num_outputs: Number of outputs to produce
        apply_fn: Function to apply to each set of args, returns the output
        realize_outputs: Whether to realize outputs for foreach fusion
    """
    outputs: list[foreach_node_t | None] = [None] * num_outputs
    for (device, use_foreach), group in groups.items():
        operation_list: list[str] = []
        for output_ind, args in group:
            output = apply_fn(args)
            outputs[output_ind] = output

            if (
                V.graph.has_feature(device, BackendFeature.FOREACH)
                and use_foreach
                and realize_outputs
            ):
                output.realize()
                operation_list.append(output.get_operation_name())

        if operation_list:
            V.graph.register_operation_list(operation_list)

    if not all(x is not None for x in outputs):
        raise AssertionError("expected: all(x is not None for x in outputs)")

    return cast(list[foreach_node_t], outputs)


def to_dtype(
    x: TensorBox, dtype: torch.dtype, copy: bool = False, use_compute_types: bool = True
):
    # [FUNCTION DOC] Convert x to dtype via a pointwise node. If dtypes match returns x (or a clone when copy is set).
    # use_compute_types controls whether intermediate compute-type promotion is used; for fp16/bf16 without it the result is upcast back to compute type.
    src_dtype = x.get_dtype()
    if src_dtype == dtype:
        return clone(x) if copy else x

    def _to_dtype(x):
        # [FUNCTION DOC] HELPER: Pointwise body doing the dtype conversion, with a compute-type upcast after a raw low-precision cast.
        result = ops.to_dtype(
            x,
            dtype,
            src_dtype=src_dtype,
            use_compute_types=use_compute_types,
        )
        low_pr_fp = (torch.bfloat16, torch.float16)
        if not use_compute_types and dtype in low_pr_fp:
            # Upcast back to compute type so fused consumers see a compute-type
            # value. Without this, a raw low-precision value gets a redundant
            # downcast from the consumer's input emulation.
            result = ops.to_dtype(result, dtype)
        return result

    return make_pointwise(_to_dtype, override_return_dtype=dtype)(x)


_FLOAT8_E8M0FNU_TO_FLOAT_DTYPES = (
    torch.float32,
    torch.float64,
    torch.float16,
    torch.bfloat16,
)


def _float8_e8m0fnu_to_dtype(x: TensorBox, dtype: torch.dtype) -> TensorBox:
    # [FUNCTION DOC] HELPER: Convert a float8_e8m0fnu tensor to a float dtype by bitcasting to uint8 and building float32 bits from the exponent.
    x_u8 = to_dtype_bitcast(x, torch.uint8)

    def _to_float(value):
        # [FUNCTION DOC] HELPER: Map E8M0 byte to float32 bits (shift into exponent; 0 maps to 2^-127, 255 to NaN) and cast to the target dtype.
        # E8M0's exponent bits map directly to FP32; only encodings 0 and 255
        # need special handling for 2^-127 and NaN, respectively.
        value_i32 = ops.to_dtype(value, torch.int32)
        f32_bits = ops.bitwise_left_shift(value_i32, ops.constant(23, torch.int32))
        f32_bits = ops.where(
            ops.eq(value, ops.constant(0, torch.uint8)),
            ops.constant(0x00400000, torch.int32),
            f32_bits,
        )
        f32_bits = ops.where(
            ops.eq(value, ops.constant(255, torch.uint8)),
            ops.constant(0x7F800001, torch.int32),
            f32_bits,
        )
        dequant = ops.to_dtype_bitcast(f32_bits, torch.float32, src_dtype=torch.int32)
        if dtype != torch.float32:
            return ops.to_dtype(dequant, dtype)
        return dequant

    return make_pointwise(_to_float, override_return_dtype=dtype)(x_u8)


@register_lowering(torch._higher_order_ops._foreach_map, type_promotion_kind=None)
def _foreach_map(subgraph, *args, **kwargs):
    # [FUNCTION DOC] Lowering for the foreach_map higher-order op. Runs the user subgraph through PointwiseSubgraphLowering to
    # inline it into the main graph, groups the resulting outputs by device, realizes them and registers them as a
    # horizontally fusable operation list when foreach is supported. Returns the list of outputs.
    """
    This lowers an invocation of foreach_map
    The way this works is that an arbitrary N-arg func is provided by the user, looped over by the
    polyfill with the same semantics as a foreach op (a loop applying an n-ary function to n args)
    and then traced into a subgraph by dynamo.
    This code allows us to inline the subgraph into the main graph lowering using the PontwiseSubgraphLowering.
    The graph outputs represent the vertically fused sequence of ops, and then register_operation_list
    below registers the buffers as horizontally fuseable in the scheduler.
    """
    from .subgraph_lowering import PointwiseSubgraphLowering

    inputs = args

    gm = subgraph.graph_module
    pw_subgraph = PointwiseSubgraphLowering(gm, root_graph_lowering=V.graph)
    with V.set_graph_handler(pw_subgraph):  # type: ignore[arg-type]
        pw_subgraph.run(*inputs)

    sub_outputs = pw_subgraph.graph_outputs
    # group outputs by device and register as foreach
    if not (sub_outputs):  # mypy lol
        raise AssertionError("expected: sub_outputs")
    groups = group_foreach_args(sub_outputs)

    outputs = [None] * len(sub_outputs)
    for (device, use_foreach), group in groups.items():
        operation_list: list[str] = []
        for (
            output_ind,
            output,
        ) in group:
            outputs[output_ind] = output

            if V.graph.has_feature(device, BackendFeature.FOREACH) and use_foreach:
                output.realize()
                operation_list.append(output.get_operation_name())

        if operation_list:
            V.graph.register_operation_list(operation_list)

    if not (all(x is not None for x in outputs)):
        raise AssertionError("expected: all(x is not None for x in outputs)")
    return outputs


@register_lowering(prims.convert_element_type, type_promotion_kind=None)
def _convert_element_type(x: TensorBox, dtype: torch.dtype):
    # [FUNCTION DOC] Lowering for prims.convert_element_type. Complex conversions use an in-place copy into an empty tensor
    # (or a fallback kernel for empty tensors); float8_e8m0fnu to float uses a special path; otherwise delegates to to_dtype
    # with a copy, honoring the precision-emulation config for fp16/bf16.
    if dtype.is_complex or x.get_dtype().is_complex:
        if x.get_size():
            # Decompose since aa aten fallback is more friendly for c++ codegen.
            # This decomposition doesn't work for empty tensor, which needs more investigation.
            dst = empty_like(x, dtype=dtype)
            ir.InplaceCopyFallback.create(dst, x)
            return dst
        else:
            return fallback_handler(
                prims.convert_element_type.default, add_to_fallback_set=False
            )(x, dtype)
    src_dtype = x.get_dtype()
    if src_dtype == torch.float8_e8m0fnu and dtype in _FLOAT8_E8M0FNU_TO_FLOAT_DTYPES:
        return _float8_e8m0fnu_to_dtype(x, dtype)

    low_pr_fp = (torch.bfloat16, torch.float16)
    # In precision-emulation mode, explicit lowp casts must materialize the
    # storage dtype. Later pointwise barriers will widen from that rounded value.
    use_compute_types = not (
        config.emulate_precision_casts
        and (src_dtype in low_pr_fp or dtype in low_pr_fp)
    )
    return to_dtype(x, dtype, copy=True, use_compute_types=use_compute_types)


def to_dtype_bitcast(x: TensorBox, dtype: torch.dtype, *, copy=False):
    # [FUNCTION DOC] Reinterpret x's bits as dtype. Same dtype returns x (or a clone). Equal bit widths give a DtypeView;
    # different widths fall back to the eager view.dtype kernel on a contiguous input.
    x_dtype = x.get_dtype()
    if x_dtype == dtype:
        return clone(x) if copy else x

    def _get_primitive_bitwidth(dtype):
        # [FUNCTION DOC] HELPER: Return the bit width of a dtype (bool counts as 8).
        if dtype.is_floating_point:
            return torch.finfo(dtype).bits
        elif dtype == torch.bool:
            # torch.iinfo doesn't support bool; bools are stored as uint8 (8 bits)
            return 8
        else:
            return torch.iinfo(dtype).bits

    src_bits = _get_primitive_bitwidth(x_dtype)
    dst_bits = _get_primitive_bitwidth(dtype)
    if src_bits != dst_bits:
        # fallback to aten eager implementation for differing bitwidths
        x_cont = ir.ExternKernel.require_contiguous_strides(x)
        return fallback_handler(aten.view.dtype)(x_cont, dtype)
    else:
        return TensorBox(DtypeView.create(x, dtype))


@register_lowering(aten.view.dtype, type_promotion_kind=None)
def _view_dtype(x: TensorBox, dtype: torch.dtype):
    # [FUNCTION DOC] Lowering for aten.view.dtype: complex dtypes use a ComplexView; others delegate to to_dtype_bitcast.
    if dtype.is_complex or x.get_dtype().is_complex:
        return TensorBox.create(
            ir.ComplexView.create(torch.ops.aten.view.dtype, x, dtype)
        )
    return to_dtype_bitcast(x, dtype)


def to_device(x: TensorBox, device: torch.device, *, copy=False, non_blocking=False):
    # [FUNCTION DOC] Move x to device. Returns x (or a clone if copy) when already there; otherwise creates a DeviceCopy node.
    device = decode_device(device)
    if x.get_device() == device:
        return clone(x) if copy else x
    return TensorBox.create(ir.DeviceCopy.create(x, device, non_blocking))


@register_lowering(prims.device_put, type_promotion_kind=None)
def _device_put(x: TensorBox, device: torch.device, non_blocking=False):
    # [FUNCTION DOC] Lowering for prims.device_put; always copies via to_device.
    return to_device(x, device, copy=True, non_blocking=non_blocking)


def register_pointwise(
    aten_fn,
    name=None,
    broadcast=True,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    convert_input_to_bool=False,
    override_return_dtype=None,
    override_fn_when_input_bool=None,
    allow_alpha=False,
    use_fma_for_alpha=False,
    triton_fallback=None,
    round_scalars_to_tensor_dtype=False,
):
    # [FUNCTION DOC] Register a pointwise lowering for aten_fn that maps to the backend op named name (default the function name).
    # Registers dtype propagation rules, builds the lowering with make_pointwise, registers it for the aten op and, if a prim of the
    # same name exists, for that too. Returns the lowering function.
    """A pointwise function that maps ops.{name} to inputs"""
    name = name or aten_fn.__name__
    fn = ops_wrapper(name)

    register_op_dtype_propagation_rules(
        name, type_promotion_kind, override_return_dtype
    )

    if override_fn_when_input_bool is not None:
        override_fn_when_input_bool = ops_wrapper(override_fn_when_input_bool)

    fn = make_pointwise(
        fn,
        override_return_dtype=override_return_dtype,
        override_fn_when_input_bool=override_fn_when_input_bool,
        allow_alpha=allow_alpha,
        use_fma_for_alpha=use_fma_for_alpha,
        triton_fallback=triton_fallback,
        round_scalars_to_tensor_dtype=round_scalars_to_tensor_dtype,
    )
    fn = register_lowering(
        aten_fn,
        broadcast=broadcast,
        type_promotion_kind=type_promotion_kind,
        convert_input_to_bool=convert_input_to_bool,
    )(fn)

    if hasattr(prims, name):
        register_lowering(
            getattr(prims, name),
            type_promotion_kind=None,
            convert_input_to_bool=convert_input_to_bool,
        )(fn)
    return fn


register_op_dtype_propagation_rules(
    "ldexp",
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    override_return_dtype=None,
)


@register_lowering(aten.ldexp, broadcast=True, type_promotion_kind=None)
def ldexp_lowering(x: TensorBox, n: TensorBox):
    # [FUNCTION DOC] Lowering for aten.ldexp. Uses native ldexp when x is floating and n is an integer; otherwise computes
    # x * 2**n in float32 (integer x) or x's dtype using pow.
    ldexp_fn = ops_wrapper("ldexp")

    x_dtype = x.get_dtype()
    n_dtype = n.get_dtype()

    x_is_float = x_dtype.is_floating_point
    n_is_int = not n_dtype.is_floating_point and n_dtype != torch.bool

    if x_is_float and n_is_int:
        # Use native ldexp
        def compute_ldexp(x, n):
            # [FUNCTION DOC] HELPER: Pointwise body calling the native ldexp op.
            return ldexp_fn(x, n)

        return make_pointwise(compute_ldexp)(x, n)
    else:
        # Fall back to decomposition: x * pow(2, n)
        out_dtype = torch.float32 if is_integer_type(x) else x_dtype

        def compute_fallback(x, n):
            # [FUNCTION DOC] HELPER: Pointwise body computing x times 2 raised to n, with n converted to the output dtype.
            n_out_type = ops.to_dtype(n, out_dtype)
            two = ops.constant(2.0, out_dtype)
            pow_result = ops.pow(two, n_out_type)
            return ops.mul(x, pow_result)

        return make_pointwise(
            compute_fallback,
            override_return_dtype=out_dtype,
        )(x, n)


def register_frexp():
    # [FUNCTION DOC] HELPER: Register the lowering for aten.frexp (and the prim of the same name if present). Builds two
    # pointwise functions, one for the mantissa and one for the int32 exponent, and a combined lowering returning both.
    # Returns the combined lowering. Called once at import.
    """A pointwise function that maps ops.frexp to inputs"""
    name = "frexp"
    frexp = ops_wrapper("frexp")

    def frexp0(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Pointwise body returning the mantissa (first result) of the frexp op.
        return frexp(*args, **kwargs)[0]  # type: ignore[index]

    def frexp1(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Pointwise body returning the exponent (second result) of the frexp op.
        return frexp(*args, **kwargs)[1]  # type: ignore[index]

    pw_fns = [
        make_pointwise(frexp0),
        make_pointwise(frexp1, override_return_dtype=torch.int32),
    ]

    def fn(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Return the (mantissa, exponent) pair of IR tensors for the given inputs.
        return pw_fns[0](*args, **kwargs), pw_fns[1](*args, **kwargs)

    fn = register_lowering(
        aten.frexp,
    )(fn)

    if hasattr(prims, name):
        register_lowering(
            getattr(prims, name),
            type_promotion_kind=None,
        )(fn)
    return fn


register_frexp()


def register_foreach_pointwise(
    aten_fn,
    pointwise_lowering_fn,
    allow_alpha=False,
    scalar_kwarg="alpha",
):
    # [FUNCTION DOC] HELPER: Register a foreach lowering for aten_fn built from a per-tensor pointwise lowering,
    # with optional alpha/value scalar kwarg handling. Returns the registered lowering.
    fn = make_foreach_pointwise(
        pointwise_lowering_fn, allow_alpha=allow_alpha, scalar_kwarg=scalar_kwarg
    )
    fn = _register_foreach_lowering(aten_fn, fn)
    return fn


@register_lowering(aten.where, broadcast=False, type_promotion_kind=None)
def where(cond, a, b):
    # [FUNCTION DOC] Lowering for aten.where. Converts python scalar branches to constants like the other branch, promotes the
    # two value branches to a common dtype, broadcasts tensors, expands constants, and builds a pointwise select node.
    def fn(*args):
        # [FUNCTION DOC] HELPER: Pointwise body calling the where op.
        return ops.where(*args)

    if isinstance(a, (float, int)):
        a = constant_like(a)(b)
    if isinstance(b, (float, int)):
        b = constant_like(b)(a)

    args = [cond, a, b]
    dtype = get_promoted_dtype(
        args[1], args[2], type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    indices = [i for i, x in enumerate(args) if isinstance(x, TensorBox)]
    for i, x in zip(indices, broadcast_tensors(*[args[i] for i in indices])):
        args[i] = x
    for i in range(len(args)):
        if isinstance(args[i], ir.Constant):
            args[i] = ExpandView.create(args[i], list(args[indices[0]].get_size()))
    return make_pointwise(fn, override_return_dtype=dtype)(
        args[0], to_dtype(args[1], dtype), to_dtype(args[2], dtype)
    )


@register_lowering(aten.broadcast_tensors, broadcast=False, type_promotion_kind=None)
def broadcast_tensors(*inputs):
    # [FUNCTION DOC] Lowering for aten.broadcast_tensors. Computes the common symbolic broadcast shape and expands each input
    # whose shape differs. A single list/tuple argument is unpacked. Returns the inputs unchanged for a single tensor, else a list.
    if len(inputs) == 1:
        if isinstance(inputs[0], (list, tuple)):
            return broadcast_tensors(*inputs[0])
        return inputs
    target: list[sympy.Expr] = functools.reduce(
        broadcast_symbolic_shapes, (x.get_size() for x in inputs), ()
    )
    outputs = []
    for x in inputs:
        if (sizes := tuple(x.get_size())) == target:
            pass

        elif len(sizes) != len(target) or any(
            V.graph.sizevars.is_size_one_or_false(a)
            != V.graph.sizevars.is_size_one_or_false(b)
            for a, b in zip(sizes, target)
        ):
            x = expand(x, target)
        outputs.append(x)
    return outputs


@register_lowering([aten.alias, aten.detach, aten.detach_, aten.lift, prims.view_of])
def nop(x):
    # [FUNCTION DOC] No-op lowering (alias, detach, lift, view_of): returns x; autograd handles semantics.
    return x  # AOT autograd handles this for us


if hasattr(aten, "lift_fresh"):
    register_lowering(aten.lift_fresh)(nop)


@register_lowering(aten.squeeze, type_promotion_kind=None)
def squeeze(x, dim=None):
    # [FUNCTION DOC] Lowering for aten.squeeze. With no dim, squeezes all size-one dims through a SqueezeView. With dims (int or
    # sequence), guards them to ints, canonicalizes, and removes those that are known to be size one via a view. Returns x if nothing changes.
    if not (isinstance(x, TensorBox)):
        raise AssertionError("expected: isinstance(x, TensorBox)")
    if dim is None:
        return TensorBox(SqueezeView.create(x.data))

    dim = (
        V.graph.sizevars.guard_int(dim)
        if isinstance(dim, (int, sympy.Expr))
        else tuple(V.graph.sizevars.guard_int(d) for d in dim)
    )
    dim = canonicalize_dims(len(x.get_size()), dim)  # type: ignore[call-overload]
    dims = OrderedSet((dim,) if not isinstance(dim, tuple) else dim)

    new_shape = []
    for d, s in enumerate(x.get_size()):
        if not (d in dims and V.graph.sizevars.guard_or_false(sympy.Eq(s, 1))):
            new_shape.append(s)

    # squeeze does nothing if the size isn't 1
    return view(x, new_shape) if new_shape != x.get_size() else x


@register_lowering(aten.squeeze_copy, type_promotion_kind=None)
def squeeze_copy(x, dim=None):
    # [FUNCTION DOC] Lowering for aten.squeeze_copy: squeeze then clone.
    return clone(squeeze(x, dim))


@register_lowering([aten.squeeze_])
def squeeze_(x, dim=None):
    # [FUNCTION DOC] Lowering for in-place squeeze_: squeezes and rebinds x's underlying data to the result, mutating the TensorBox. Returns x.
    val = squeeze(x, dim)
    if not (isinstance(x, TensorBox)):
        raise AssertionError("expected: isinstance(x, TensorBox)")
    if not (isinstance(val, TensorBox)):
        raise AssertionError("expected: isinstance(val, TensorBox)")
    x.data = val.data
    return x


@register_lowering(aten.isinf)
def isinf(x):
    # [FUNCTION DOC] Lowering for aten.isinf: all-False bool tensor for integer inputs, else a pointwise isinf producing bool.
    if is_integer_type(x):
        return full_like(x, False, dtype=torch.bool)
    fn = ops_wrapper("isinf")
    return make_pointwise(fn, override_return_dtype=torch.bool)(x)


@register_lowering(aten.isnan)
def isnan(x):
    # [FUNCTION DOC] Lowering for aten.isnan: all-False bool tensor for integer inputs, else a pointwise isnan producing bool.
    if is_integer_type(x):
        return full_like(x, False, dtype=torch.bool)
    fn = ops_wrapper("isnan")
    return make_pointwise(fn, override_return_dtype=torch.bool)(x)


@register_lowering(aten.ceil)
def ceil(x):
    # [FUNCTION DOC] Lowering for aten.ceil: clone for integer inputs, otherwise pointwise ceil.
    if is_integer_type(x):
        return clone(x)
    fn = ops_wrapper("ceil")
    return make_pointwise(fn)(x)


@register_lowering(aten.floor)
def floor(x):
    # [FUNCTION DOC] Lowering for aten.floor: clone for integer inputs, otherwise pointwise floor.
    if is_integer_type(x):
        return clone(x)
    fn = ops_wrapper("floor")
    return make_pointwise(fn)(x)


@register_lowering(aten.round.default)
def round(x):
    # [FUNCTION DOC] Lowering for aten.round: clone for integer inputs, otherwise pointwise round.
    if is_integer_type(x):
        return clone(x)
    else:
        fn = ops_wrapper("round")
        return make_pointwise(fn)(x)


@register_lowering(aten.trunc)
def trunc(x):
    # [FUNCTION DOC] Lowering for aten.trunc: clone for integer inputs, otherwise pointwise trunc.
    if is_integer_type(x):
        return clone(x)
    fn = ops_wrapper("trunc")
    return make_pointwise(fn)(x)


@register_lowering(aten.expand, type_promotion_kind=None)
def expand(x, sizes, *, implicit=False):
    # [FUNCTION DOC] Lowering for aten.expand. The implicit flag is ignored. Constants become an ExpandView directly; tensors
    # must be TensorBox with list/tuple sizes (else an error is raised). Returns x if sizes are equal. For static sizes it marks
    # reuse on the input (graph-level reuse in deterministic modes) so the expanded input may be realized, then returns an ExpandView.
    # `implicit` is autograd-internal metadata (see aten::expand schema); it
    # does not affect the produced tensor, so the lowering ignores it. Without
    # this kwarg the lowering rejects graphs produced by dynamo autograd where
    # aten.expand.default is emitted with implicit=False.
    del implicit
    (x,) = promote_constants([x])
    if isinstance(x, ir.BaseConstant):
        return ExpandView.create(x, tuple(sizes))
    if not (isinstance(x, TensorBox)):
        raise AssertionError("expected: isinstance(x, TensorBox)")
    if not (isinstance(sizes, (list, tuple))):
        raise AssertionError("expected: isinstance(sizes, (list, tuple))")
    if tuple(x.get_size()) == tuple(sizes):
        return x

    if not free_unbacked_symbols(x.get_size()):
        x_size_product = V.graph.sizevars.guarding_hint_or_throw(
            sympy_product(x.get_size())
        )
        # TODO: It would be better to realize the input if any of its sizes
        # are unbacked, because typically the size will be non-zero.  However,
        # this cannot be done directly as below as we'll choke on the size_hint
        # here
        if x_size_product > 0 and not free_unbacked_symbols(sizes):
            # Broadcast loop reuse is not graph fanout; keep the graph-fanout
            # read-count heuristic from materializing cheap expanded producers.
            # In deterministic modes, preserve the old materialization boundary
            # since fusing through expanded inputs can change reduction numerics.
            x.mark_reuse(
                V.graph.sizevars.guarding_hint_or_throw(sympy_product(sizes))
                // x_size_product,
                graph_reuse=config.deterministic
                or torch.are_deterministic_algorithms_enabled(),
            )
    return TensorBox(ExpandView.create(x.data, tuple(sizes)))


@register_lowering(prims.broadcast_in_dim, type_promotion_kind=None)
def broadcast_in_dim(a, shape, broadcast_dimensions):
    # [FUNCTION DOC] Lowering for prims.broadcast_in_dim: unsqueezes a at every dim not listed in broadcast_dimensions and expands to shape.
    s = list(shape)
    for broadcast_dimension in broadcast_dimensions:
        s[broadcast_dimension] = -1

    v = a
    for idx, x in enumerate(s):
        if x != -1:
            v = unsqueeze(v, idx)

    return expand(v, shape)


@register_lowering(aten.expand_as, type_promotion_kind=None)
def expand_as(x, y):
    # [FUNCTION DOC] Lowering for aten.expand_as: expand x to the size of y.
    return expand(x, y.get_size())


@register_lowering(aten.repeat)
def repeat(x, repeats):
    # [FUNCTION DOC] Lowering for aten.repeat. Left-pads dims with one if repeats is longer. A zero repeat yields an empty tensor;
    # repeats only over size-one dims reduce to a cloned expand; otherwise builds a Pointwise that indexes the input with modular
    # indexing per repeated dim and marks the input reuse. Raises an error on rank mismatch.
    old_size = list(x.get_size())
    if len(repeats) > len(old_size):
        old_size = [sympy.S.One] * (len(repeats) - len(old_size)) + old_size
        x = view(x, list(old_size))
    if len(repeats) != len(x.get_size()):
        raise AssertionError("expected: len(repeats) == len(x.get_size())")

    new_size = list(x.get_size())

    zero_tensor = False
    for i in range(len(repeats)):
        if repeats[i] == 0:
            zero_tensor = True
        new_size[i] = new_size[i] * repeats[i]

    if zero_tensor:
        return empty(new_size, dtype=x.get_dtype(), device=x.get_device())
    if all((a == 1 or b == 1) for a, b in zip(repeats, old_size)):
        return clone(expand(x, new_size))

    x_loader: Callable[[Any], Any]

    def inner_fn(index):
        # [FUNCTION DOC] HELPER: Map an output index to the input index (modulo old size, or zero for size-one dims) and load it.
        if len(index) != len(repeats):
            raise AssertionError("expected: len(index) == len(repeats)")
        index = list(index)
        for i in range(len(repeats)):
            if repeats[i] != 1:
                if old_size[i] == 1:
                    index[i] = sympy.S.Zero
                else:
                    index[i] = ModularIndexing(index[i], 1, old_size[i])
        return x_loader(index)

    # TODO Laith is there better check
    if not free_unbacked_symbols(old_size) and not free_unbacked_symbols(new_size):
        old_size_product = V.graph.sizevars.guarding_hint_or_throw(
            sympy_product(old_size)
        )
        if old_size_product > 0:
            # maybe realize the input but skip for unbacked symints since it'll
            # choke on the size hint.
            x.mark_reuse(
                V.graph.sizevars.guarding_hint_or_throw(sympy_product(new_size))
                // old_size_product
            )

    x_loader = x.make_loader()
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(new_size),
    )


@register_lowering(aten._unsafe_view, type_promotion_kind=None)
@register_lowering(aten.view, type_promotion_kind=None)
@register_lowering(aten.reshape, type_promotion_kind=None)
def view(x: ir.IRNode, sizes: Sequence[sympy.Expr]) -> TensorBox:
    # [FUNCTION DOC] Lowering for aten.view/reshape/_unsafe_view: wraps the node (or raw IR node) in a View with the given sizes and returns a TensorBox.
    # post-mm_args operands are raw ReinterpretView/StorageBox nodes, and taking
    # .data on those would drop the view itself
    data = x.data if isinstance(x, TensorBox) else x
    return TensorBox(View.create(data, sizes))


@register_lowering(aten.permute, type_promotion_kind=None)
def permute(x, dims):
    # [FUNCTION DOC] Lowering for aten.permute: returns a PermuteView of x with dims. Raises an error if x is not a TensorBox or dims not a list/tuple.
    if not (isinstance(x, TensorBox)):
        raise AssertionError("expected: isinstance(x, TensorBox)")
    if not (isinstance(dims, (list, tuple))):
        raise AssertionError("expected: isinstance(dims, (list, tuple))")
    return TensorBox(PermuteView.create(x.data, tuple(dims)))


# Note: logic in this function need to be always synchronized with
# slice_forward in fake implementation.
def _register_unbacked_slice_size_bindings(dim, start, end, step, size):
    # [FUNCTION DOC] HELPER: For a slice on dim, look through the current FX node's unbacked symbol bindings to find
    # the symbol for the output size and for the storage offset. If the size symbol exists, registers a DynamicSliceSize buffer and
    # operation for it. Returns (sym_size, sym_storage), each possibly None. Must stay in sync with the fake slice implementation.
    """Register DynamicSliceSize for any unbacked size bindings on the current FX node.

    Unbacked symbols may have been allocated at trace time of the slice
    that inductor must define so the assertion
    new_unbacked_defs >= renamed_unbacked_bindings passes in run_node.

    Returns (sym_size, sym_storage) parsed from the bindings, or (None, None).
    """
    from torch.fx.experimental.symbolic_shapes import (
        CallMethodKey,
        resolve_unbacked_bindings,
    )

    # current_node may be None when slice_ is called from template rendering
    # (e.g. cpp_template_kernel.slice_nd) rather than FX graph lowering.
    current_node = V.graph.current_node
    node_unbacked_bindings = resolve_unbacked_bindings(
        V.graph.sizevars.shape_env,
        current_node.meta.get("unbacked_bindings", {})
        if current_node is not None
        else {},
    )
    sym_size, sym_storage = None, None
    if node_unbacked_bindings:
        if len(node_unbacked_bindings) > 2:
            raise AssertionError(node_unbacked_bindings)
        for sym, keypath in node_unbacked_bindings.items():
            if keypath == (CallMethodKey("size"), pytree.SequenceKey(dim)):
                sym_size = sym
            elif keypath == (CallMethodKey("storage_offset"),):
                sym_storage = sym
        if sym_size is not None:
            b_size = ir.DynamicSliceSize(sym_size, start, end, step, size)
            b_size.name = V.graph.register_buffer(b_size)
            V.graph.register_operation(b_size)
        # NOTE: storage_offset registration is not handled here — it is only
        # needed in the ambiguous (unbacked start/end) path, where it is
        # handled via DynamicSelectStorageOffset after this helper returns.
    return sym_size, sym_storage


def _compute_slice_index(index, size, default=None):
    # [FUNCTION DOC] HELPER: Resolve a slice start/end index against size with python semantics (negative wrap, clamp to [0, size]).
    # Returns default if index is None, an expression when the position can be proven, or a Min/Max expression for partial knowledge; None if undeterminable.
    if index is None:
        return default

    guard = V.graph.sizevars.guard_or_false
    index = sympy.expand(index)
    size = sympy.expand(size)
    if guard(sympy.And(sympy.Ge(index, 0), sympy.Le(index, size))):
        return index
    elif guard(sympy.And(sympy.Lt(index, 0), sympy.Ge(index, -size))):
        return index + size
    elif guard(sympy.Gt(index, size)):
        return size
    elif guard(sympy.Lt(index, -size)):
        return 0
    elif guard(sympy.Ge(index, 0)):
        # If index >= 0, the resolved index is at most min(index, size).
        return Min(index, size)
    elif guard(sympy.Lt(index, 0)):
        # If index < 0, wrap and clamp: the resolved index is at least 0.
        return Max(index + size, 0)
    return None


def _clamp_slice_end_to_start(end, start):
    # [FUNCTION DOC] HELPER: Return end clamped to be at least start, using static knowledge when available or a Max expression.
    if V.graph.sizevars.statically_known_geq(end, start):
        return end
    if V.graph.sizevars.statically_known_leq(end, start):
        return start
    return Max(end, start)


@register_lowering(aten.slice, type_promotion_kind=None)
def slice_(x, dim=0, start=0, end=sys.maxsize, step=1, clamp=True):
    # [FUNCTION DOC] Lowering for aten.slice. Returns x for a statically known no-op slice. Resolves start/end indices; when both are
    # known creates a SliceView (registering unbacked size bindings). When start or end is unbacked, creates extern buffers for the
    # size/storage offset symbols and builds an as_strided view instead. Raises an error on inconsistent unbacked bindings.
    """
    Lowers a slice call, creating ExternKernels for the output size & storage offset symbols,
    if the indices are unbacked and appropriate semantics aren't known.
    If they are known (indices are static/backed/unbacked with info), a SliceView is created.
    """

    if not (isinstance(x, TensorBox)):
        raise AssertionError("expected: isinstance(x, TensorBox)")
    dim = _validate_dim(x, dim, 0)
    size = x.get_size()[dim]
    step = sympy.expand(step)
    if not (isinstance(step, sympy.Expr) or step > 0):
        raise AssertionError(step)

    # maybe apply slice optimization
    try:
        if (
            start == 0
            and V.graph.sizevars.statically_known_leq(size, end)
            and step == 1
        ):
            _, sym_storage = _register_unbacked_slice_size_bindings(
                dim, start, end, step, size
            )
            if sym_storage is not None:
                raise AssertionError(
                    "Unexpected storage_offset unbacked binding for no-op slice"
                )
            return x
    except TypeError:
        pass

    start_index, end_index = None, None
    # ambiguous_slice=False means we know what semantics this slice call follows,
    # and don't need to generate an extern kernel to represent the output size.
    # This is assumed True for clamp=False
    # (meant to follow standard indexing semantics: 0 <= index < size)
    ambiguous_slice = clamp
    if ambiguous_slice:
        start_index = _compute_slice_index(start, size, 0)
        # Special case: if end is maxsize (unbounded), use size directly
        # This matches the logic in fake_impls.py
        if end is not None and V.graph.sizevars.statically_known_equals(
            end, sys.maxsize
        ):
            end_index = size
        else:
            end_index = _compute_slice_index(end, size, size)
        if start_index is not None and end_index is not None:
            start, end = start_index, end_index
            ambiguous_slice = False

    if not ambiguous_slice:
        _, sym_storage = _register_unbacked_slice_size_bindings(
            dim, start, end, step, size
        )
        if sym_storage is not None:
            raise AssertionError(
                "Unexpected storage_offset unbacked binding when both "
                "start and end indices are resolved"
            )

        return TensorBox(
            ir.SliceView.create(x.data, dim, start, end, step, clamp=clamp)
        )  # go to SliceView/ReinterpretView

    # unbacked territory: unbacked start / end
    if not (clamp):
        raise AssertionError("expected: clamp")
    if not (start_index is None or end_index is None):
        raise AssertionError("expected: start_index is None or end_index is None")
    sym_size, sym_storage = _register_unbacked_slice_size_bindings(
        dim, start, end, step, x.get_size()[dim]
    )
    if sym_size is None:
        raise AssertionError("expected: sym_size is not None")

    if x.maybe_get_layout() is None:
        # realize tensor before accessing layout
        x.realize()
    stride = x.maybe_get_stride()

    if start_index is not None:
        # we shouldn't have allocated storage offset symbol if start index was determinable
        if sym_storage is not None:
            raise AssertionError("expected: sym_storage is None")
        if stride is None:
            return TensorBox(
                ir.SliceView.create_with_size(x.data, dim, start_index, sym_size, step)
            )
        new_storage_offset = x.get_layout().offset + start_index * stride[dim]
    else:
        if stride is None:
            raise AssertionError("expected: stride is not None")
        b_storage = ir.DynamicSelectStorageOffset(
            sym_storage,
            start,
            x.get_layout().offset,
            stride[dim],
            x.get_size()[dim],
            clamp=True,
        )
        b_storage.name = V.graph.register_buffer(b_storage)
        V.graph.register_operation(b_storage)
        new_storage_offset = sym_storage

    new_sizes = list(x.get_size())
    new_strides = list(stride)
    new_sizes[dim] = sym_size
    new_strides[dim] *= step
    return as_strided(
        x,
        new_sizes,
        new_strides,
        new_storage_offset,
        storage_offset_relative_to_input_storage=False,
    )


@register_lowering(aten.as_strided, type_promotion_kind=None)
def as_strided(
    x,
    size,
    stride,
    storage_offset=None,
    *,
    storage_offset_relative_to_input_storage=True,
):
    # [FUNCTION DOC] Lowering for aten.as_strided. Realizes the input (unwrapping any view while keeping its dtype/device), then returns a
    # ReinterpretView over the storage with a FixedLayout of the requested size, stride and offset. Explicit offsets are made relative to a graph
    # input's own storage offset. Raises NotImplementedError if the input cannot be realized to storage and layout.
    explicit_storage_offset = (
        storage_offset is not None and storage_offset_relative_to_input_storage
    )
    new_device = None
    new_dtype = None
    if isinstance(x, TensorBox) and isinstance(x.data, ir.BaseView):
        # Note: Merging views
        # When we use as_strided, we can rewrite the size/stride/offset
        # of the incoming buffer x. If x is a view, we would overwrite
        # its metadata. Except for dtype, which we need to propagate.

        # Technically device is not needed because it is not possible
        # to have a cross-device view today.
        new_device = x.get_device()
        new_dtype = x.dtype
        if storage_offset is None and x.maybe_get_layout() is not None:
            storage_offset = x.get_layout().offset
        x = x.data.unwrap_view()
    x.realize()
    if not ir.is_storage_and_layout(x):
        raise NotImplementedError(f"unrealized as_strided({x}, ...)")
    storage, old_layout = ir.as_storage_and_layout(x)
    storage_offset = (
        convert_symint_to_expr(storage_offset) if storage_offset is not None else 0
    )
    storage_data = storage.data if isinstance(storage, ir.StorageBox) else storage
    if explicit_storage_offset and isinstance(storage_data, ir.InputBuffer):
        # Runtime graph input pointers already include the input tensor's
        # storage_offset(), but explicit as_strided offsets are storage-relative.
        storage_offset = sympy.expand(
            storage_offset
            - V.graph.graph_input_storage_offsets.get(storage_data.get_name(), 0)
        )
    new_layout = ir.FixedLayout(
        new_device if new_device else old_layout.device,
        new_dtype if new_dtype else old_layout.dtype,
        [sympy.expand(s) for s in size],
        [sympy.expand(s) for s in stride],
        sympy.expand(storage_offset),
    )
    return TensorBox(ir.ReinterpretView(data=storage, layout=new_layout))


@register_lowering(aten.as_strided_, type_promotion_kind=None)
def as_strided_(x, size, stride, storage_offset=None):
    # [FUNCTION DOC] Lowering for in-place as_strided_: rebinds x's data to the strided view and returns x.
    if not (isinstance(x, TensorBox)):
        raise AssertionError("expected: isinstance(x, TensorBox)")
    x.data = as_strided(x, size, stride, storage_offset).data
    return x


@register_lowering(aten.as_strided_copy, type_promotion_kind=None)
def as_strided_copy(x, size, stride, storage_offset=None):
    # [FUNCTION DOC] Lowering for aten.as_strided_copy: as_strided then clone.
    result = as_strided(x, size, stride, storage_offset)
    return clone(result)


def pointwise_cat(inputs, dim=0):
    # [FUNCTION DOC] Concatenate inputs along dim as a single Pointwise node. For every output index, each input is loaded under a mask
    # on the dim coordinate (shifted by the input's start offset) and the results are chained with select ops. Returns a Pointwise with
    # size equal to the first input's with the dim extent set to the sum; device and dtype come from the first input.
    # (inclusive, exclusive)
    inputs_ranges: list[tuple[sympy.Expr, sympy.Expr]] = []
    prev_end = 0
    for inp in inputs:
        inputs_ranges.append((prev_end, prev_end + inp.get_size()[dim]))  # type: ignore[arg-type]
        prev_end = inputs_ranges[-1][-1]  # type: ignore[assignment]

    inputs_loaders = [inp.make_loader() for inp in inputs]

    def inner_fn(idx):
        # [FUNCTION DOC] HELPER: For one output index, build a masked load per input and combine them with nested where ops chosen by range of the dim coordinate.
        idx_dim = ops.index_expr(idx[dim], torch.int64)

        masks = []
        masked_loads = []
        for i in range(len(inputs)):
            start = (
                ops.constant(0, torch.int64)
                if i == 0
                else ops.index_expr(inputs_ranges[i][0], torch.int64)
            )
            end = ops.index_expr(inputs_ranges[i][1], torch.int64)

            start_cond = ops.ge(idx_dim, start)
            end_cond = ops.lt(idx_dim, end)
            if i == 0:
                mask = end_cond
            elif i == len(inputs) - 1:
                mask = start_cond
            else:
                mask = ops.and_(start_cond, end_cond)

            masks.append(mask)
            idx_load = list(idx)

            # if we're concatting [4], [2]
            # when we index the second tensor for 5 we want to index 5 - 4
            # Use Identity to prevent expansion of index * stride to keep expression
            # in same int bitwidth as shape
            idx_load[dim] = Identity(idx_load[dim] - inputs_ranges[i][0])

            masked_loads.append(
                ops.masked(
                    mask,
                    lambda: inputs_loaders[i](idx_load),
                    0.0,  # this value should be unused
                ),
            )

        next_val = masked_loads[-1]
        for i in range((len(inputs)) - 2, -1, -1):
            next_val = ops.where(
                masks[i],
                masked_loads[i],
                next_val,
            )
        return next_val

    new_size = list(inputs[0].get_size())
    new_size[dim] = inputs_ranges[-1][-1]

    return Pointwise.create(
        device=inputs[0].get_device(),
        dtype=inputs[0].get_dtype(),
        inner_fn=inner_fn,
        ranges=new_size,
    )


@register_lowering(quantized_decomposed.quantize_per_channel, type_promotion_kind=None)
def quantized_decomposed_quantize_per_channel(
    input: TensorBox,
    scales: TensorBox,
    zero_points: TensorBox,
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> TensorBox:
    # [FUNCTION DOC] Lowering for quantized_decomposed.quantize_per_channel. Requires 1-D scales and zero_points, float32 input (bf16 is upcast)
    # and a valid axis, else raises an error. Output is a Pointwise computing round(input / scale) + zero_point, clamped to [quant_min, quant_max], cast to dtype.
    if len(scales.get_size()) != 1:
        raise AssertionError("expect scales 1 dim")
    if len(zero_points.get_size()) != 1:
        raise AssertionError("expect zero_points 1 dim")

    if input.get_dtype() == torch.bfloat16:
        input = to_dtype(input, torch.float32)
    if input.get_dtype() != torch.float32:
        raise AssertionError(
            f"Expecting input to have dtype torch.float32, but got dtype: {input.get_dtype()}"
        )
    if axis >= len(input.get_size()):
        raise AssertionError(f"Expecting axis to be < {len(input.get_size())}")

    input_loader = input.make_loader()
    scales_loader = scales.make_loader()
    zero_points_loader = zero_points.make_loader()

    def inner_fn(idx):
        # [FUNCTION DOC] HELPER: Quantize one element using the per-channel scale and zero point selected by the axis coordinate.
        channel_idx = (idx[axis],)

        input = input_loader(idx)
        scale = scales_loader(channel_idx)
        zero_point = zero_points_loader(channel_idx)
        qmin, qmax = _create_constants(quant_min, quant_max, dtype=torch.float32)

        if scales.dtype != torch.float32:
            scale = ops.to_dtype(scale, torch.float32)
        if zero_points.dtype != torch.int32:
            zero_point = ops.to_dtype(zero_point, torch.int32)
        inv_scale = ops.reciprocal(scale)
        val = ops.round(input * inv_scale) + zero_point
        clamped = ops.maximum(qmin, ops.minimum(qmax, val))
        return ops.to_dtype(clamped, dtype)

    return Pointwise.create(
        device=input.get_device(),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=input.get_size(),
    )


def _assert_async(cond, msg):
    # [FUNCTION DOC] HELPER: Lower an async device assertion. Realizes cond, casts it to bool and creates a realized Pointwise whose body emits
    # a device assert with msg (forcing realization). Returns the realized assertion op.
    cond.realize()
    cond = to_dtype(cond, torch.bool)

    def inner_fn(index):
        # [FUNCTION DOC] HELPER: Emit the device_assert_async op on the condition value at index.
        with ir.ComputedBuffer.force_realize():
            return ops.device_assert_async(cond.make_loader()(index), msg)

    assertion_op = Pointwise.create(
        device=cond.get_device(),
        dtype=cond.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(cond.get_size()),
    )
    assertion_op.realize()
    return assertion_op


@register_lowering(aten._assert_async.msg)
def lower_assert_async(cond, msg):
    # [FUNCTION DOC] Lowering for aten._assert_async.msg: delegates to _assert_async.
    return _assert_async(cond, msg)


@register_lowering(aten._functional_assert_async.msg)
def lower_assert_functional_async(cond, msg):
    # [FUNCTION DOC] Lowering for aten._functional_assert_async.msg: delegates to _assert_async.
    return _assert_async(cond, msg)


@register_lowering(
    quantized_decomposed.dequantize_per_channel, type_promotion_kind=None
)
def quantized_decomposed_dequantize_per_channel(
    input: TensorBox,
    scales: TensorBox,
    zero_points: TensorBox,
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    *,
    out_dtype: torch.dtype | None = None,
) -> TensorBox:
    # [FUNCTION DOC] Lowering for quantized_decomposed.dequantize_per_channel. Validates 1-D scales/zero_points, input dtype equal to dtype and axis range
    # (raises an error otherwise). Output is a Pointwise computing (input - zero_point) * scale in float32, cast to out_dtype (default float32).
    if len(scales.get_size()) != 1:
        raise AssertionError("expect scales 1 dim")
    if len(zero_points.get_size()) != 1:
        raise AssertionError("expect zero_points 1 dim")
    if input.get_dtype() != dtype:
        raise AssertionError(
            f"Expecting input to have dtype {dtype}, but got dtype: {input.get_dtype()}"
        )
    if axis >= len(input.get_size()):
        raise AssertionError(f"Expecting axis to be < {len(input.get_size())}")

    if out_dtype is None:
        out_dtype = torch.float32

    input_loader = input.make_loader()
    scales_loader = scales.make_loader()
    zero_points_loader = zero_points.make_loader()

    def inner_fn(idx):
        # [FUNCTION DOC] HELPER: Dequantize one element with the per-channel scale and zero point chosen by the axis coordinate.
        channel_idx = (idx[axis],)

        input = input_loader(idx)
        scale = scales_loader(channel_idx)
        zero_point = zero_points_loader(channel_idx)

        if scales.dtype != torch.float32:
            scale = ops.to_dtype(scale, torch.float32)
        if zero_points.dtype != torch.float32:
            zero_point = ops.to_dtype(zero_point, torch.float32)
        val = ops.sub(ops.to_dtype(input, torch.float32), zero_point) * scale
        val = ops.to_dtype(val, out_dtype)
        return val

    return Pointwise.create(
        device=input.get_device(),
        dtype=out_dtype,
        inner_fn=inner_fn,
        ranges=input.get_size(),
    )


@register_lowering(
    quantized_decomposed.quantize_per_tensor.default, type_promotion_kind=None
)
def quantized_decomposed_quantize_per_tensor_default(
    input: TensorBox,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> TensorBox:
    # [FUNCTION DOC] Lowering for quantized_decomposed.quantize_per_tensor.default (python float scale and int zero point). Requires float32 input
    # (bf16 upcast). Output is a Pointwise of round(input / scale) + zero_point clamped to the quant range, cast to dtype.
    if input.get_dtype() == torch.bfloat16:
        input = to_dtype(input, torch.float32)
    if input.get_dtype() != torch.float32:
        raise AssertionError(
            f"Expecting input to have dtype torch.float32, but got dtype: {input.get_dtype()}"
        )

    input_loader = input.make_loader()

    def inner_fn(idx, scale, zero_point):
        # [FUNCTION DOC] HELPER: Quantize one element using the scalar scale and zero point bound by partial application.
        input = input_loader(idx)
        inv_scale, zero_point = _create_constants(
            1.0 / scale, zero_point, dtype=torch.float32
        )
        val = ops.round(input * inv_scale) + zero_point
        qmin, qmax = _create_constants(quant_min, quant_max, dtype=torch.float32)
        clamped = ops.minimum(ops.maximum(val, qmin), qmax)
        return ops.to_dtype(clamped, dtype)

    return Pointwise.create(
        device=input.get_device(),
        dtype=dtype,
        inner_fn=functools.partial(
            inner_fn, scale=float(scale), zero_point=int(zero_point)
        ),
        ranges=input.get_size(),
    )


@register_lowering(
    quantized_decomposed.dequantize_per_tensor.default, type_promotion_kind=None
)
def quantized_decomposed_dequantize_per_tensor_default(
    input: TensorBox,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    *,
    out_dtype: torch.dtype | None = None,
) -> TensorBox:
    # [FUNCTION DOC] Lowering for quantized_decomposed.dequantize_per_tensor.default. Requires input dtype to equal dtype (raises an error otherwise).
    # Output Pointwise is (input - zero_point) * scale computed in float32 and cast to out_dtype (default float32).
    if input.get_dtype() != dtype:
        raise AssertionError(
            f"Expecting input to have dtype {dtype}, but got dtype: {input.get_dtype()}"
        )

    if out_dtype is None:
        out_dtype = torch.float32

    input_loader = input.make_loader()

    def inner_fn(idx, scale, zero_point):
        # [FUNCTION DOC] HELPER: Dequantize one element with the scalar scale and zero point.
        input = input_loader(idx)
        scale, zero_point = _create_constants(scale, zero_point, dtype=torch.float32)
        val = ops.sub(ops.to_dtype(input, torch.float32), zero_point) * scale
        val = ops.to_dtype(val, out_dtype)
        return val

    return Pointwise.create(
        device=input.get_device(),
        dtype=out_dtype,
        inner_fn=functools.partial(
            inner_fn, scale=float(scale), zero_point=int(zero_point)
        ),
        ranges=input.get_size(),
    )


@register_lowering(
    quantized_decomposed.quantize_per_tensor.tensor, type_promotion_kind=None
)
def quantized_decomposed_quantize_per_tensor_tensor(
    input: TensorBox,
    scale: TensorBox,
    zero_point: TensorBox,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> TensorBox:
    # [FUNCTION DOC] Lowering for quantized_decomposed.quantize_per_tensor.tensor with scalar-tensor scale and zero point (0-d or single-element, else raises an error).
    # Requires float32 input (bf16 upcast). On CPU uses fused multiply-add and round-to-int; elsewhere round then clamp to the quant range. Output cast to dtype.
    if input.get_dtype() == torch.bfloat16:
        input = to_dtype(input, torch.float32)
    if input.get_dtype() != torch.float32:
        raise AssertionError(
            f"Expecting input to have dtype torch.float32, but got dtype: {input.get_dtype()}"
        )
    if not (
        len(scale.get_size()) == 0
        or (len(scale.get_size()) == 1 and scale.get_size()[0] == 1)
    ):
        raise AssertionError("expect scale as scalar tensor")
    if not (
        len(zero_point.get_size()) == 0
        or (len(zero_point.get_size()) == 1 and zero_point.get_size()[0] == 1)
    ):
        raise AssertionError("expect zero_point as scalar tensor")

    input_loader = input.make_loader()
    scale_loader = scale.make_loader()
    zero_point_loader = zero_point.make_loader()

    device = input.get_device()

    def inner_fn(idx):
        # [FUNCTION DOC] HELPER: Load the scalar scale/zero point (as float32), then quantize one element using the CPU or generic path.
        input = input_loader(idx)
        _scale = scale_loader((0,) if len(scale.get_size()) == 1 else ())
        _zero_point = zero_point_loader((0,) if len(scale.get_size()) == 1 else ())
        if scale.dtype != torch.float32:
            _scale = ops.to_dtype(_scale, torch.float32)
        if zero_point.dtype != torch.float32:
            _zero_point = ops.to_dtype(_zero_point, torch.float32)
        if device and device.type == "cpu":
            val = ops.fma(input, ops.reciprocal(_scale), _zero_point)
            return ops.round_to_int(val, dtype)
        val = ops.round(input * ops.reciprocal(_scale)) + _zero_point
        qmin, qmax = _create_constants(quant_min, quant_max, dtype=torch.float32)
        clamped = ops.minimum(ops.maximum(val, qmin), qmax)
        return ops.to_dtype(clamped, dtype)

    return Pointwise.create(
        device=input.get_device(),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=input.get_size(),
    )


@register_lowering(
    quantized_decomposed.dequantize_per_tensor.tensor, type_promotion_kind=None
)
def quantized_decomposed_dequantize_per_tensor_tensor(
    input: TensorBox,
    scale: TensorBox,
    zero_point: TensorBox,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    *,
    out_dtype: torch.dtype | None = None,
) -> TensorBox:
    # [FUNCTION DOC] Lowering for quantized_decomposed.dequantize_per_tensor.tensor with scalar-tensor scale and zero point (validated, else raises an error) and
    # input dtype equal to dtype. Output Pointwise is (input - zero_point) * scale in float32 cast to out_dtype (default float32).
    if not (
        len(scale.get_size()) == 0
        or (len(scale.get_size()) == 1 and scale.get_size()[0] == 1)
    ):
        raise AssertionError("expect scale as scalar tensor")
    if not (
        len(zero_point.get_size()) == 0
        or (len(zero_point.get_size()) == 1 and zero_point.get_size()[0] == 1)
    ):
        raise AssertionError("expect zero_point as scalar tensor")
    if input.get_dtype() != dtype:
        raise AssertionError(
            f"Expecting input to have dtype {dtype}, but got dtype: {input.get_dtype()}"
        )

    if out_dtype is None:
        out_dtype = torch.float32

    input_loader = input.make_loader()
    scale_loader = scale.make_loader()
    zero_point_loader = zero_point.make_loader()

    def inner_fn(idx):
        # [FUNCTION DOC] HELPER: Load scalar scale/zero point as float32 and dequantize one element.
        input = input_loader(idx)
        _scale = scale_loader((0,) if len(scale.get_size()) == 1 else ())
        _zero_point = zero_point_loader((0,) if len(scale.get_size()) == 1 else ())
        if scale.dtype != torch.float32:
            _scale = ops.to_dtype(_scale, torch.float32)
        if zero_point.dtype != torch.float32:
            _zero_point = ops.to_dtype(_zero_point, torch.float32)
        val = ops.sub(ops.to_dtype(input, torch.float32), _zero_point) * _scale
        val = ops.to_dtype(val, out_dtype)
        return val

    return Pointwise.create(
        device=input.get_device(),
        dtype=out_dtype,
        inner_fn=inner_fn,
        ranges=input.get_size(),
    )


def _cat_inputs_recombine_reduction(inputs: list[TensorBox], dim: int) -> str | None:
    # [FUNCTION DOC] HELPER: Detect the pattern where all cat inputs derive from one reduction buffer whose output numel matches the cat output and whose
    # FX consumers all feed into the cat. Returns that buffer's name (so it is excluded from the reduction-fusion check) or None. Relies on FX origins of IR nodes.
    """If all cat inputs share a common upstream reduction buffer whose
    only consumers feed into this cat, return its name so it can be
    excluded from the can_fuse_reduction check.

    Checks common reads for an IR reduction whose numel matches the cat
    output, then verifies via FX origins that all of the reduction's
    consumers feed into the cat inputs."""
    if len(inputs) < 2:
        return None

    common_reads = inputs[0].get_read_names()
    for inp in inputs[1:]:
        common_reads = common_reads & inp.get_read_names()
    if not common_reads:
        return None

    # Find a common read that is an IR reduction buffer whose input
    # numel matches the cat output numel.
    cat_out_numel = convert_symint_to_expr(V.graph.current_node.meta["val"].numel())
    reduction_name = None
    reduction_buf = None
    for name in common_reads:
        buf = V.graph.try_get_buffer(name)
        if (
            buf is not None
            and isinstance(buf, ir.ComputedBuffer)
            and isinstance(buf.data, ir.Reduction)
        ):
            reduction_numel = sympy_product(buf.data.get_size()) * sympy_product(
                buf.data.get_reduction_size()
            )
            if V.graph.sizevars.statically_known_equals(cat_out_numel, reduction_numel):
                reduction_name = name
                reduction_buf = buf
                break

    if reduction_name is None:
        return None

    # Verify the reduction doesn't have consumers outside this cat's
    # computation. Each IR node tracks which FX nodes produced it
    # (origins). Collect the FX origins of all cat inputs, then check
    # that every FX user of the reduction's origins feeds into one of
    # the cat inputs.
    #
    # We also tried checking IR-level users via V.graph.name_to_users,
    # but at lowering time the cat inputs are unrealized TensorBox
    # wrappers (not named buffers), so name_to_users entries can't be
    # correlated back to the cat's input chain.
    #
    # TODO: origins is a set of FX nodes attached to IR nodes during
    # lowering — using it for correctness is fragile. A proper
    # buffer→FX node mapping would be better.
    origins = getattr(reduction_buf, "origins", None)
    if not origins:
        return None

    cat_input_origins: OrderedSet[torch.fx.Node] = OrderedSet()
    for inp in inputs:
        inp_origins = getattr(inp, "origins", None)
        if inp_origins:
            cat_input_origins.update(inp_origins)

    # Check that the reduction FX node's users all feed into the cat.
    # origins may include non-reduction nodes (e.g. pow that feeds into
    # mean), so filter to only reduction ops via torch.Tag.reduction.
    for origin in origins:
        if (
            origin.op == "call_function"
            and isinstance(origin.target, torch._ops.OpOverload)
            and torch.Tag.reduction in origin.target.tags
            and not all(u in cat_input_origins for u in origin.users)
        ):
            return None

    return reduction_name


@register_lowering(aten.cat)
def cat(inputs, dim=0):
    # [FUNCTION DOC] Lowering for aten.cat. CPU uint8/int8 inputs go to an eager fallback (channels-last for 4-D). A single input is cloned; 1-D empty
    # inputs are dropped like ATen. Inputs are promoted to a common dtype. Chooses between pointwise_cat (fusing producers into the concat) and
    # ConcatKernel (realize inputs into one buffer): ConcatKernel is used on CPU, with too many inputs, with reduction producers, or when inputs have
    # other pointwise consumers; pointwise_cat is used when forced by config or when it fuses with pointwise users or all inputs need copies anyway.
    # Returns a TensorBox.
    """Lower aten.cat, choosing between pointwise_cat and ConcatKernel."""
    cpu_device = inputs[0].get_device().type == "cpu"
    if cpu_device and all(
        input.get_dtype() in [torch.int8, torch.uint8] for input in inputs
    ):
        # TODO <leslie> Remove this fallback when we support vectorization
        # code gen with uint8 data type directly.
        for input in inputs:
            input.realize()
        if all(len(input.get_size()) == 4 for input in inputs):
            inputs, _ = require_channels_last(aten.cat, *inputs)
        return fallback_handler(aten.cat.default)(inputs, dim)

    if len(inputs) == 1:
        return clone(inputs[0])

    # ATen's cat skips 1-D empty tensors (cat_should_skip_tensor) during
    # dimension validation and concatenation. Replicate that here.
    def _cat_should_skip(t):
        # [FUNCTION DOC] HELPER: True if t is a 1-D tensor statically known to have zero elements, which ATen's cat ignores.
        size = t.get_size()
        return len(size) == 1 and V.graph.sizevars.statically_known_equals(size[0], 0)

    skip_mask = [_cat_should_skip(t) for t in inputs]
    if any(skip_mask):
        valid_inputs = [t for t, skip in zip(inputs, skip_mask) if not skip]
        if not valid_inputs:
            return clone(inputs[0])
        if len(valid_inputs) == 1:
            return clone(valid_inputs[0])
        inputs = valid_inputs

    dim = _validate_dim(inputs[0], dim, 0)
    dtype = get_promoted_dtype(
        *inputs, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    inputs = [to_dtype(inp, dtype) for inp in inputs]

    def unwrap_tensor(x: TensorBox | ir.StorageBox) -> ir.IRNode:
        # [FUNCTION DOC] HELPER: Peel TensorBox/StorageBox/view wrappers and return the underlying IR node.
        if isinstance(x, TensorBox):
            if isinstance(x.data, ir.BaseView):
                return x.data.unwrap_view()
            else:
                return x.data

        if isinstance(x, ir.StorageBox):
            return x.data

        return x

    def is_reduction(t):
        # [FUNCTION DOC] HELPER: True if t is a computed buffer wrapping a Reduction.
        return isinstance(t, ir.ComputedBuffer) and isinstance(t.data, ir.Reduction)

    def can_fuse_reduction(t, exclude: OrderedSet[str] = OrderedSet()):
        # [FUNCTION DOC] HELPER: Recursively test whether t is, or is a pointwise chain reading from, a reduction buffer (ignoring buffers named in exclude).
        if isinstance(t, (TensorBox, ir.StorageBox)):
            return can_fuse_reduction(unwrap_tensor(t), exclude)
        return (
            is_reduction(t)
            or isinstance(t, ir.Pointwise)
            and any(
                read not in exclude
                and can_fuse_reduction(V.graph.get_buffer(read), exclude)
                for read in t.get_read_names()
            )
        )

    # Pointwise cat evaluates every input's computation for each
    # output element (masked), so fusing reductions in is wasteful.
    # Exception: when inputs just recombine a reduction's output
    # (e.g. qknorm → RoPE → cat), we do not duplicate computation
    recombined = _cat_inputs_recombine_reduction(inputs, dim)
    exclude: OrderedSet[str] = OrderedSet([recombined]) if recombined else OrderedSet()
    fusable_reduction = any(can_fuse_reduction(t, exclude) for t in inputs)

    def should_lower_cat_input(x) -> bool:
        # [FUNCTION DOC] HELPER: Decide whether a cat input is worth fusing into a pointwise cat: true if it is storage that cannot be realized into the
        # concat buffer without a copy, or an unrealized Pointwise; false otherwise.
        # Unrealized inputs will not be storage and layouts, and we don't want to realize
        # them in case we want to fuse
        if ir.is_storage_and_layout(x):
            storage, _ = ir.as_storage_and_layout(x, freeze=False)
            return not ir.ConcatKernel.can_realize_into_without_copy(storage)

        if isinstance(x, (TensorBox, ir.StorageBox)):
            return should_lower_cat_input(unwrap_tensor(x))

        if isinstance(x, ir.Pointwise):
            return True

        return False

    if config.force_pointwise_cat:
        return pointwise_cat(inputs, dim)

    # TODO: We observed negative performance impact of pointwise_cat optimization on CPU so disabled it.
    #             We will revisit this later after enabling vectorization on index_expr.
    if cpu_device:
        return TensorBox(ir.ConcatKernel.create(inputs, dim))

    def op_count(x):
        # [FUNCTION DOC] HELPER: Estimate the op count of x by summing its pointwise inner-function ops and those of the pointwise buffers it reads (0 for direct memory reads).
        if isinstance(x, (TensorBox, ir.StorageBox)):
            return op_count(unwrap_tensor(x))

        # this will correspond to a direct memory read
        if not isinstance(x, ir.Pointwise):
            return 0

        count = x.inner_fn_opcount().num_ops
        for read in x.get_read_names():
            count += op_count(V.graph.get_buffer(read))

        return count

    # as inputs increase, possibility for register spilling also increases.
    # Past a certain threshold we only fuse if the input kernels are simple.
    MAX_SIMPLE_OP_COUNT = 2

    def additional_pointwise_ops(op: torch._ops.OpOverload):
        # [FUNCTION DOC] HELPER: True for ops (cat, constant_pad_nd) treated as pointwise-like users when deciding whether to fuse.
        return op in (aten.cat.default, aten.constant_pad_nd.default)

    if len(inputs) <= config.max_complex_pointwise_cat_inputs or (
        (len(inputs) <= config.max_pointwise_cat_inputs)
        and all(op_count(t) <= MAX_SIMPLE_OP_COUNT for t in inputs)
    ):
        pointwise_uses = all(
            is_pointwise_use(use, additional_pointwise_ops)
            for use in V.current_node.users
        )
        # fuse in case we will be used in a pointwise node, and there are any inputs we
        # we can prevent materialization of.
        fuse_pointwise_use = (
            any(should_lower_cat_input(inp) for inp in inputs) and pointwise_uses
        )

        # horizontal fuse in case all inputs will require a copy kernel anyway.
        # only horizontally fuse pointwise kernels

        # Skip pointwise_cat when any cat input has a fusible (pointwise)
        # multi-consumer — ConcatKernel + NonOwningLayout avoids redundant
        # reads. Also skip when input is an unrealized Pointwise with
        # multiple consumers to avoid recomputation (e.g. pad-as-cat).
        def any_input_has_multi_consumers() -> bool:
            # [FUNCTION DOC] HELPER: True if any FX input of the current cat node has other pointwise consumers, or is an unrealized Pointwise with
            # several consumers, in which case recomputation would occur and ConcatKernel is preferred. False if no current node or inputs have a single user.
            current_node = V.current_node
            if current_node is None:
                return False
            fx_args = current_node.args[0]
            if isinstance(fx_args, (list, tuple)):
                input_nodes = fx_args
            elif isinstance(fx_args, torch.fx.Node):
                input_nodes = [fx_args]
            else:
                return False

            if any(skip_mask):
                input_nodes = [n for n, skip in zip(input_nodes, skip_mask) if not skip]

            def is_unrealized_pointwise(x):
                # [FUNCTION DOC] HELPER: True if the unwrapped IR node is a Pointwise (not realized).
                if isinstance(x, (TensorBox, ir.StorageBox)):
                    return is_unrealized_pointwise(unwrap_tensor(x))
                return isinstance(x, ir.Pointwise)

            for arg, ir_input in zip(input_nodes, inputs):
                if not hasattr(arg, "users") or len(arg.users) <= 1:
                    continue
                # input will be computed multiple times because other consumers
                # (eg. pointwise) will also inline it. So we should realize-in-place via ConcatKernel
                if any(is_pointwise_use(u) for u in arg.users if u is not current_node):
                    return True
                # If input is an unrealized Pointwise with multiple consumers, pointwise_cat
                # will inline input without realizing it to memory, causing separate
                # realization cost for input. So we should realize-in-place via ConcatKernel
                if is_unrealized_pointwise(ir_input):
                    return True
            return False

        has_multi_consumers = any_input_has_multi_consumers()

        horizontal_fuse_cat = (
            all(should_lower_cat_input(inp) for inp in inputs) and not fusable_reduction
        )

        if not has_multi_consumers and (fuse_pointwise_use or horizontal_fuse_cat):
            return pointwise_cat(inputs, dim)

    return TensorBox(ir.ConcatKernel.create(inputs, dim))


@register_lowering(aten.diagonal, type_promotion_kind=None)
def diagonal(input, offset: int = 0, dim1: int = 0, dim2: int = 1):
    # [FUNCTION DOC] Lowering for aten.diagonal. Canonicalizes dims (must differ), computes the diagonal length (guarding on the sign of offset),
    # and returns a GenericView with the two dims removed and a trailing diagonal dim, mapping indices to the original with the offset added.
    original_shape = input.get_size()
    num_dims = len(original_shape)
    dim1 = canonicalize_dim(idx=dim1, rank=num_dims)
    dim2 = canonicalize_dim(idx=dim2, rank=num_dims)

    check(
        dim1 != dim2, lambda: f"diagonal dimensions cannot be identical {dim1}, {dim2}"
    )

    offset_negative = V.graph.sizevars.evaluate_expr(sympy.Lt(offset, 0))
    if offset_negative:
        diag_size = V.graph.sizevars.evaluate_max(
            V.graph.sizevars.evaluate_min(
                original_shape[dim1] + offset,
                original_shape[dim2],
            ),
            0,  # type: ignore[arg-type]
        )
    else:
        diag_size = V.graph.sizevars.evaluate_max(
            V.graph.sizevars.evaluate_min(
                original_shape[dim1],
                original_shape[dim2] - offset,
            ),
            0,  # type: ignore[arg-type]
        )

    base_idx = (0, 0)
    if offset_negative:
        base_idx = (-offset, 0)
    else:
        base_idx = (0, offset)

    sizes = [s for i, s in enumerate(original_shape) if i not in (dim1, dim2)]
    sizes.append(diag_size)

    def reindexer(idx):
        # [FUNCTION DOC] HELPER: Map a diagonal-view index to the original index: the last coordinate feeds both diagonal dims (plus offsets), others pass through in order.
        diag_idx = idx[-1]
        original_idx = [0] * len(original_shape)
        cur_dim = 0
        for d in range(num_dims):
            if d == dim1:
                original_idx[d] = diag_idx + base_idx[0]
            elif d == dim2:
                original_idx[d] = diag_idx + base_idx[1]
            else:
                original_idx[d] = idx[cur_dim]
                cur_dim += 1

        if cur_dim != len(original_shape) - 2:
            raise AssertionError("expected: cur_dim == len(original_shape) - 2")
        return original_idx

    return TensorBox(ir.GenericView.create(input, sizes, reindexer))


@register_lowering(aten.diagonal_copy, type_promotion_kind=None)
def diagonal_copy(input, offset: int = 0, dim1: int = 0, dim2: int = 1):
    # [FUNCTION DOC] Lowering for aten.diagonal_copy: diagonal then clone.
    return clone(diagonal(input, offset, dim1, dim2))


@register_lowering(aten.diagonal_scatter, type_promotion_kind=None)
def diagonal_scatter(input, src, offset: int = 0, dim1: int = 0, dim2: int = 1):
    # [FUNCTION DOC] Lowering for aten.diagonal_scatter: clones input, mutates the diagonal view of the clone to hold src, returns the clone.
    output = clone(input)
    target = diagonal(output, offset, dim1, dim2)
    mutate_to(target, src)
    return output


@register_lowering(aten.select, type_promotion_kind=None)
def select(x, dim, idx):
    # [FUNCTION DOC] Lowering for aten.select. Normalizes the index when its sign is known: unbacked indices become an as_strided view,
    # others a slice followed by squeeze. When the sign is unknown (unbacked), uses a DynamicSelectStorageOffset buffer bound to the
    # node's unbacked symbol and as_strided. Raises an error if the unbacked bindings are missing or not exactly one.
    idx = sympy.expand(idx)
    size = sympy.expand(x.get_size()[dim])
    actual_index = None

    if V.graph.sizevars.guard_or_false(sympy.Lt(idx, 0)):
        actual_index = idx + size
    elif V.graph.sizevars.guard_or_false(sympy.Ge(idx, 0)):
        actual_index = idx

    if actual_index is not None:
        if has_free_unbacked_symbols(idx):
            # Inductor could generate incorrect views for tensors with unbacked symbols here;
            # Squeeze operations are translated to views, resulting in incorrect strides.
            # Additionally, we want to avoid accidental unbacked unsqueeze semantics. To resolve this,
            # we use as_strided instead.
            # Removing this branch will cause test_unbacked_select_index_with_check to fail.

            # before accessing size, stride, and offset we need to realize.
            x.realize()
            new_size = x.get_size()
            new_stride = x.get_stride()
            new_storage_offset = x.get_layout().offset + new_stride[dim] * actual_index

            del new_size[dim]
            del new_stride[dim]
            return as_strided(
                x,
                new_size,
                new_stride,
                new_storage_offset,
                storage_offset_relative_to_input_storage=False,
            )
        else:
            # no need to clamp, this function handles negative indexing itself
            slice_result = slice_(x, dim, actual_index, actual_index + 1, clamp=False)
            return squeeze(slice_result, dim)

    # Unbacked Semantics:
    # When the index idx is unbacked (e.g., u0), we compute the index dynamically
    # during the lowering of the select operation using DynamicSelectStorageOffset.

    unbacked_bindings = resolve_unbacked_bindings(
        V.graph.sizevars.shape_env, V.graph.current_node.meta["unbacked_bindings"]
    )
    if unbacked_bindings is None:
        raise AssertionError("expected: unbacked_bindings is not None")
    if len(unbacked_bindings) != 1:
        raise AssertionError(unbacked_bindings)
    unbacked_offset_sym, _ = next(iter(unbacked_bindings.items()))

    # before accessing size, stride, and offset we need to realize.
    x.realize()
    new_size = x.get_size()
    new_stride = x.get_stride()
    new_storage_offset = unbacked_offset_sym
    buffer = ir.DynamicSelectStorageOffset(
        unbacked_offset_sym,
        idx,
        x.get_layout().offset,
        new_stride[dim],
        x.get_size()[dim],
        clamp=False,
    )
    buffer.name = V.graph.register_buffer(buffer)
    V.graph.register_operation(buffer)

    del new_size[dim]
    del new_stride[dim]
    return as_strided(
        x,
        new_size,
        new_stride,
        new_storage_offset,
        storage_offset_relative_to_input_storage=False,
    )


@register_lowering(aten.split, type_promotion_kind=None)
def split(x, sizes, dim=0):
    # [FUNCTION DOC] Lowering for aten.split. An int/SymInt size is turned into chunk sizes (last may be smaller); negative size or zero size with
    # non-empty dim raises RuntimeError; an empty dim returns one empty slice. Returns a list of slices along dim.
    dim = _validate_dim(x, dim, 0)
    sizes_ = sizes

    # If sizes is an integer (or a SymInt), we turn it into a list of sizes
    # by computing what the actual size of each chunk should be.
    if not isinstance(sizes, (list, tuple)):
        sizevars = V.graph.sizevars
        if sizevars.statically_known_lt(sizes, 0):
            raise RuntimeError(
                f"split expects split_size be non-negative, but got split_size={sizes}"
            )

        x_size = x.get_size()[dim]
        if sizevars.statically_known_equals(x_size, 0):
            return [slice_(x, dim, 0, 0, clamp=False)]

        if sizevars.statically_known_equals(sizes, 0):
            raise RuntimeError(
                "split_size can only be 0 if dimension size is 0, "
                f"but got dimension size of {x_size}"
            )

        chunks = sizevars.guard_int(FloorDiv(x_size + sizes - 1, sizes))
        sizes_ = [sizes] * chunks
        # The last chunk might have a smaller size than the rest.
        sizes_[-1] = x_size - (chunks - 1) * sizes

    # From this point, we assume that the sum of the sizes of all chunks
    # equals the size of the base tensor.
    result = []
    start = 0
    for size in sizes_:
        end = start + size
        # No need for clamping here, since we compute the exact
        # start and end values.
        result.append(slice_(x, dim, start, end, clamp=False))
        start = end
    return result


@register_lowering(aten.split_with_sizes, type_promotion_kind=None)
def split_with_sizes(x, sizes, dim=0):
    # [FUNCTION DOC] Lowering for aten.split_with_sizes: delegates to split.
    return split(x, sizes, dim)


@register_lowering(aten.unbind, type_promotion_kind=None)
def unbind(x, dim=0):
    # [FUNCTION DOC] Lowering for aten.unbind: guards the dim size to an int and returns a list of select results for each index.
    dim = _validate_dim(x, dim, 0)
    x_size = V.graph.sizevars.guard_int(x.get_size()[dim])
    result = [select(x, dim, i) for i in range(x_size)]
    return result


@register_lowering(aten.unfold, type_promotion_kind=None)
def unfold(x, dimension, size, step):
    # [FUNCTION DOC] Lowering for aten.unfold. Scalars use a slice of an unsqueezed tensor. Otherwise checks size and step against the dim, marks
    # reuse on the input, and returns a GenericView whose last dim is the window and whose unfolded dim is the number of windows.
    sizes = x.get_size()
    ndim = len(sizes)
    dim = canonicalize_dim(ndim, dimension)

    if ndim == 0:
        return slice_(unsqueeze(x, 0), end=size, clamp=False)

    dim_size = sizes[dim]
    sizevars = V.graph.sizevars
    sizevars.check_leq(size, dim_size)
    sizevars.check_lt(0, step)  # type: ignore[arg-type]

    new_dim_size = FloorDiv(dim_size - size, step) + 1
    if sizevars.guarding_hint_or_throw(dim_size) > 0:
        x.mark_reuse(
            sizevars.guarding_hint_or_throw(CeilDiv(new_dim_size * size, dim_size))
        )

    out_size = [*sizes[:dim], new_dim_size, *sizes[dim + 1 :], size]

    def reindexer(idx):
        # [FUNCTION DOC] HELPER: Map unfold view index to the input index: window position plus window index times step.
        dim_idx = idx[-1] + idx[dim] * step
        return (*idx[:dim], dim_idx, *idx[dim + 1 : -1])

    return TensorBox(ir.GenericView.create(x, out_size, reindexer))


@register_lowering(aten.unsqueeze, type_promotion_kind=None)
def unsqueeze(x, dim):
    # [FUNCTION DOC] Lowering for aten.unsqueeze: inserts a size-one dim at dim (validated with offset 1) via view.
    dim = _validate_dim(x, dim, 1)
    new_shape = list(x.get_size())
    new_shape.insert(dim, sympy.S.One)
    return view(x, new_shape)


@register_lowering(aten.unsqueeze_, type_promotion_kind=None)
def unsqueeze_(x, dim):
    # [FUNCTION DOC] Lowering for in-place unsqueeze_: rebinds x's data to the unsqueezed view and returns x.
    val = unsqueeze(x, dim)
    if not (isinstance(x, TensorBox)):
        raise AssertionError("expected: isinstance(x, TensorBox)")
    if not (isinstance(val, TensorBox)):
        raise AssertionError("expected: isinstance(val, TensorBox)")
    x.data = val.data
    return x


def _validate_dim(x, dim, offset=0):
    # [FUNCTION DOC] HELPER: Evaluate dim to a concrete int, wrap negatives using ndim plus offset, and return it. Raises an error if out of range.
    dim = V.graph.sizevars.shape_env.evaluate_expr(sympy.sympify(dim))
    ndim = len(x.get_size())
    if dim < 0:
        dim += ndim + offset
    if not (0 <= dim < ndim + offset):
        raise AssertionError("expected: 0 <= dim < ndim + offset")
    return dim


@register_lowering(aten.glu)
def glu(x, dim=-1):
    # [FUNCTION DOC] Lowering for aten.glu: splits x in half along dim (size guarded) and returns first half times sigmoid of the second half.
    dim = _validate_dim(x, dim, 0)
    # TODO: don't guard on static shape here
    new_len = V.graph.sizevars.guard_int(x.get_size()[dim]) // 2
    # no need to clamp, index is int based on input size
    a = slice_(x, dim, 0, new_len, clamp=False)
    b = slice_(x, dim, new_len, new_len * 2, clamp=False)
    return mul(a, sigmoid(b))


def fallback_handler(kernel, add_to_fallback_set=True):
    # [FUNCTION DOC] Create a lowering that calls a kernel through an extern FallbackKernel. Optionally records kernel in the fallbacks set (side effect). The
    # returned handler wraps IR outputs for lowering and is flagged as a fallback handler.
    if add_to_fallback_set:
        fallbacks.add(kernel)

    def handler(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Create the FallbackKernel for the given args and wrap resulting IR nodes as tensor boxes.
        def wrap_tensors(x):
            # [FUNCTION DOC] HELPER: Wrap IR nodes for lowering; pass others through.
            return x.wrap_for_lowering() if isinstance(x, ir.IRNode) else x

        return pytree.tree_map(
            wrap_tensors, ir.FallbackKernel.create(kernel, *args, **kwargs)
        )

    # This lets us detect that a lowering is a fallback handler.
    handler._is_fallback_handler = True  # type: ignore[attr-defined]

    return handler


@functools.cache
def _warn_complex_not_supported():
    # [FUNCTION DOC] HELPER: Emit (once, cached) a warning that complex operators are not code-generated.
    warnings.warn(
        "Torchinductor does not support code generation for complex operators. Performance may be worse than eager."
    )


# There are some types (CPU) which we accept as input but not as
# output.
def unsupported_input_tensor(t: torch.Tensor, node=None):
    # [FUNCTION DOC] Return True if inductor cannot read or write tensor t: complex (also emits a one-time warning), meta, sparse, unsupported fp8,
    # and float8_e8m0fnu except for views/bitcasts/memory movement/supported conversions determined from the FX node. Without a node, unsupported fp8 types return True.
    "Do not support reading or writing to this tensor"
    if t.is_complex():
        # Complex views are supported with IR ComplexView
        _warn_complex_not_supported()
        return True

    if t.is_meta:
        return True

    if t.is_sparse:
        return True

    if not is_triton_fp8_dtype_supported(t.dtype, t.device):
        from .codegen.triton_utils import (
            use_uint8_triton_storage_for_cuda_float8_e4m3fn,
        )

        if not use_uint8_triton_storage_for_cuda_float8_e4m3fn(
            t.dtype, device=t.device
        ):
            return True

        # uint8 storage reinterprets fp8 bytes: allow bitcast, views, memory
        # movement, and dequant (convert out of fp8)
        if not node:
            return True
        return not (
            isinstance(node.target, torch._ops.OpOverload)
            and node.target
            in (
                aten.view.dtype,
                aten.cat.default,
                aten.clone.default,
                aten._scaled_mm.default,
                aten._scaled_mm_v2.default,
                prims.convert_element_type.default,
            )
            or (isinstance(node.target, torch._ops.OpOverload) and is_view(node.target))
        )

    if t.dtype == torch.float8_e8m0fnu:
        if not node:
            return True

        # Allow bitcasts, views, memory movement, and supported conversions,
        # but not arithmetic.
        # TODO: delete once triton adds native support
        if not isinstance(node.target, torch._ops.OpOverload):
            return True
        if node.target in (
            aten.view.dtype,
            aten.cat.default,
            aten.clone.default,
            aten._scaled_mm.default,
            aten._scaled_mm_v2.default,
        ) or is_view(node.target):
            return False
        if node.target == torch.ops.prims.convert_element_type.default:
            return not (
                len(node.args) >= 2 and node.args[1] in _FLOAT8_E8M0FNU_TO_FLOAT_DTYPES
            )
        return True

    return False


def unsupported_output_tensor(t: torch.Tensor, node=None):
    # [FUNCTION DOC] Return True if inductor cannot write tensor t (reading may still work): unsupported inputs, unsupported fp8, or CPU tensors when cpp codegen is disabled.
    # Complex outputs of view.dtype/convert_element_type nodes are allowed.
    "Do not support writing tensor but can read from it"
    supported_complex_views = (
        aten.view.dtype,
        torch.ops.prims.convert_element_type.default,
    )
    if node is not None and node.target in supported_complex_views and t.is_complex():
        return False
    if unsupported_input_tensor(t, node):
        return True
    if not is_triton_fp8_dtype_supported(t.dtype, t.device):
        return True
    return t.is_cpu and config.disable_cpp_codegen


def fallback_node_due_to_unsupported_type(node: torch.fx.Node, allow_cpu_inputs=True):
    # [FUNCTION DOC] Decide whether an FX node must use a fallback kernel because an input or output tensor has an unsupported type. Placeholders,
    # view_as_complex and lift_fresh_copy are never forced to fall back. Inputs are checked for read support and the node's output for write support.
    # Custom fallback lowering
    if node.target is aten.view_as_complex.default:
        return False

    if node.op == "placeholder":
        return False

    # We should be able to remove this special case once `disable_cpp_codegen` is killed.
    if node.target is aten.lift_fresh_copy.default:
        return False

    def check_skip_condition(inp_out_node, is_output):
        # [FUNCTION DOC] HELPER: Check a node's fake-tensor metadata leaves with the input or output support predicate; True if any is unsupported.
        if not isinstance(inp_out_node, torch.fx.Node):
            return False

        if "val" not in inp_out_node.meta:
            return False

        for meta in pytree.tree_leaves(inp_out_node.meta["val"]):
            if not torch._subclasses.fake_tensor.is_fake_tensor(meta):
                continue

            if is_output:
                if unsupported_output_tensor(meta, node):
                    return True
            else:
                if unsupported_input_tensor(meta, node):
                    return True

        return False

    # only skip codegen if there is a cpu output, not input
    for arg in pytree.arg_tree_leaves(*node.args, **node.kwargs):
        if check_skip_condition(arg, is_output=False):
            return True

    return check_skip_condition(node, is_output=True)


def make_fallback(
    op,
    layout_constraint=None,
    warn=True,
    override_decomp=False,
    get_decomp_fn=None,
):
    # [FUNCTION DOC] Register an extern fallback lowering for op (an overload, overload packet or higher-order op). Raises an error if a
    # decomposition also exists (unless override_decomp), and in CI if a decomposition exists for the op (disabling dynamo suppress_errors first and logging a warning);
    # raises RuntimeError for unsupported op types. For each overload it marks inputs as needing realization, records the optional layout constraint,
    # and registers a fallback handler lowering. Mutates global registries.
    check_decomps = get_decomp_fn() if get_decomp_fn is not None else decompositions
    if op in check_decomps and not override_decomp:
        raise AssertionError(f"both a fallback and a decomp for same op: {op}")
    if (
        warn
        and bool(os.getenv("CI"))
        and get_decompositions([op])
        # if fallback_random, we allow not decomposing random
        and not (
            config.fallback_random
            and op in torch._decomp.decompositions_for_rng.extra_random_decomps
        )
        and not override_decomp
    ):
        # Note: 'warn' is holdover from when this was a warning, but for ops that previously
        # set warn=False we do not want a CI error.
        # Ignore the 'suppress errors' configs in CI, as this particular warning happens on startup anyway and is not
        # likely to be triggered preferentially on one CI config over another.
        if torch._dynamo.config.suppress_errors:
            torch._dynamo.config.suppress_errors = False
            log.warning(
                "A make_fallback error occurred in suppress_errors config,"
                " and suppress_errors is being disabled to surface it."
            )
        raise AssertionError(
            f"make_fallback({op}): a decomposition exists, we should switch to it."
            " To fix this error, either add a decomposition to core_aten_decompositions (preferred)"
            " or inductor_decompositions, and delete the corresponding `make_fallback` line."
            " Get help from the inductor team if unsure, don't pick arbitrarily to unblock yourself.",
        )

    def register_fallback(op_overload):
        # [FUNCTION DOC] HELPER: Register one op overload as a fallback: realized inputs, optional layout constraint, and a fallback_handler lowering without type promotion.
        add_needs_realized_inputs(op_overload)
        if layout_constraint is not None:
            add_layout_constraint(op_overload, layout_constraint)
        return register_lowering(op_overload, type_promotion_kind=None)(
            fallback_handler(op_overload)
        )

    if isinstance(op, torch._ops.OpOverloadPacket):
        for op_overload in op.op_overloads():
            register_fallback(op_overload)
    elif isinstance(op, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
        register_fallback(op)
    else:
        raise RuntimeError(f"Unsupported fallback {op} with type {type(op)}")


def philox_rand_offset(shape):
    # [FUNCTION DOC] Compute the Philox RNG offset increment for a random op of the given shape: a scalar int64 tensor equal to the number of elements.
    """
    TorchInductor offset calculation differs from PyTorch eager offset
    calculation for random ops (tl.rand vs torch.rand). In future, we should
    strive for same impl for tl.rand and torch.rand.
    """
    numel = 1
    for s in shape:
        numel = numel * s
    return tensor(numel, dtype=torch.int64)


@register_lowering(torch.ops.rngprims.philox_rand, type_promotion_kind=None)
def philox_rand(size, seed, offset, stride, device, dtype):
    # [FUNCTION DOC] Lowering for rngprims.philox_rand. Builds a Pointwise generating uniform random values from the seed and offset scalar tensors (cast to int32)
    # plus the contiguous position of each element, cast to dtype. stride is unused. Returns a pair (random values node, offset increment tensor).
    # stride arg is optional and will be used in future for distributed random
    # ops. Currently, its unused.
    random_pos = ir.FixedLayout(
        device,
        dtype,
        size,
        ir.FlexibleLayout.contiguous_strides(size),
    ).make_indexer()
    seed_loader = seed.make_loader()
    offset_loader = offset.make_loader()

    def inner_fn(index):
        # [FUNCTION DOC] HELPER: Generate one random value from the seed and offset-adjusted linear position.
        # Both seed and offset in the philox_rand op are tensors.
        # torch seed and offsets are of type int64, but tl.rand accepts int32
        seed_index_expr = ops.to_dtype(seed_loader([]), torch.int32)
        offset_index_expr = ops.to_dtype(offset_loader([]), torch.int32)
        # Get the offset'd position
        rand_index_expr = ops.add(
            ops.index_expr(random_pos(index), torch.int32), offset_index_expr
        )
        result = ops.rand(
            seed_index_expr,
            rand_index_expr,
        )
        return ops.to_dtype(result, dtype)

    random_values_node = Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=list(size),
    )

    offset_node = philox_rand_offset(size)
    return random_values_node, offset_node


@register_lowering(aten.native_dropout, type_promotion_kind=None)
def native_dropout(x, p, train):
    # [FUNCTION DOC] Lowering for aten.native_dropout. Only valid with fallback_random, in which case it creates a fallback kernel; otherwise raises an error since random ops are replaced earlier.
    if config.fallback_random:
        return pytree.tree_map(
            TensorBox.create,
            ir.FallbackKernel.create(aten.native_dropout.default, x, p, train),
        )
    else:
        raise AssertionError("should be handled in replace_random.py")


@register_lowering(aten.bernoulli_, type_promotion_kind=None)
def bernoulli_(x, *args):
    # [FUNCTION DOC] Lowering for in-place bernoulli_. Allowed only with fallback_random or on CPU (else raises an error). Realizes x and creates an in-place Bernoulli fallback kernel
    # (float or tensor probability overload). Returns x.
    if not (config.fallback_random or x.get_device() == torch.device("cpu")):
        raise AssertionError(
            "this should be handled in decomps unless config.fallback_random or the device is CPU"
        )
    x.realize()
    op_overload = (
        aten.bernoulli_.float
        if len(args) == 0 or isinstance(args[0], float)
        else aten.bernoulli_.Tensor
    )
    ir.InplaceBernoulliFallback(op_overload, x, *args)
    return x


@register_lowering(aten.bernoulli.p, type_promotion_kind=None)
def bernoulli_p(x, *args):
    # [FUNCTION DOC] Lowering for aten.bernoulli.p: same restrictions as bernoulli_; clones x and applies bernoulli_ to the clone.
    if not (config.fallback_random or x.get_device() == torch.device("cpu")):
        raise AssertionError(
            "this should be handled in decomps unless config.fallback_random or the device is CPU"
        )
    return bernoulli_(clone(x), *args)


# This shouldn't be called in general
@register_lowering(aten._foobar)
def _foobar(_):
    # [FUNCTION DOC] HELPER: Test placeholder lowering that always raises an error; not expected to be called.
    raise AssertionError


@functools.lru_cache(1)
def _warn_triton_random(salt):
    # [FUNCTION DOC] HELPER: Log (once per salt, cached) that triton random is used and may differ from eager.
    log.info("using triton random, expect difference from eager")


def warn_triton_random():
    # [FUNCTION DOC] HELPER: Emit the triton-random log message at most once per graph, keyed by graph creation time.
    # only warn once per graph
    _warn_triton_random(V.graph.creation_time)


fallback_rand_default = fallback_handler(aten.rand.default)
fallback_rand_generator = fallback_handler(aten.rand.generator)
fallback_randn_default = fallback_handler(aten.randn.default)
fallback_randn_generator = fallback_handler(aten.randn.generator)
make_fallback(aten.randint)
make_fallback(aten.rand_like, override_decomp=True)
make_fallback(aten.randn_like, override_decomp=True)
make_fallback(aten.randint_like, override_decomp=True)
make_fallback(aten.normal, override_decomp=True)
make_fallback(aten.rrelu_with_noise_functional)

# TODO: mlazos reevaluate if we want to codegen something different
make_fallback(torch.ops.streams.record_event.default)
make_fallback(torch.ops.streams.wait_event.default)
make_fallback(torch.ops.streams.synchronize_event.default)
make_fallback(torch.ops.streams.synchronize_device.default)
make_fallback(torch.ops.streams.synchronize_stream.default)
make_fallback(torch.ops.streams.wait_stream.default)


@register_lowering(aten.rand)
def rand(*args, **kwargs):
    # [FUNCTION DOC] Lowering for aten.rand. Uses the eager fallback for an explicit generator, or when fallback_random is set or pin_memory is requested;
    # otherwise raises an error since random ops should already have been replaced.
    if kwargs.get("generator") is not None:
        return fallback_rand_generator(*args, **kwargs)
    elif config.fallback_random or kwargs.get("pin_memory"):
        kwargs.pop("generator", None)
        return fallback_rand_default(*args, **kwargs)
    raise AssertionError("should have been handled in replace_random.py")


@register_lowering(aten.randn)
def randn(*args, **kwargs):
    # [FUNCTION DOC] Lowering for aten.randn. Same behavior as rand: fallback for generator, fallback_random or pin_memory; otherwise raises an error.
    if kwargs.get("generator") is not None:
        return fallback_randn_generator(*args, **kwargs)
    elif config.fallback_random or kwargs.get("pin_memory"):
        kwargs.pop("generator", None)
        return fallback_randn_default(*args, **kwargs)
    raise AssertionError("should have been handled in replace_random.py")


@register_lowering(inductor_prims.force_stride_order, type_promotion_kind=None)
def inductor_force_stride_order(input_tensor, stride):
    # [FUNCTION DOC] Lowering for inductor_prims.force_stride_order: requires the input to have the stride order implied by the given stride.
    stride_order = ir.get_stride_order(stride, V.graph.sizevars.shape_env)
    return ir.ExternKernel.require_stride_order(input_tensor, stride_order)


@register_lowering(inductor_prims.seed, type_promotion_kind=None)
def inductor_seed(device: torch.device):
    # [FUNCTION DOC] Lowering for inductor_prims.seed; always raises an error since it must be eliminated by the seed creation pass.
    raise AssertionError("should be handled in fuse_seed_creation_pass()")


@register_lowering(inductor_prims.seeds, type_promotion_kind=None)
def inductor_seeds(count, device):
    # [FUNCTION DOC] Lowering for inductor_prims.seeds: warns about triton random and returns a RandomSeeds node with count seeds on the device.
    warn_triton_random()
    return TensorBox.create(ir.RandomSeeds(count, decode_device(device)))


@register_lowering(inductor_prims.lookup_seed, type_promotion_kind=None)
def inductor_lookup_seed(seeds, index):
    # [FUNCTION DOC] Lowering for inductor_prims.lookup_seed: returns a scalar Pointwise that loads the seed at index from the seeds buffer.
    def inner_fn(_):
        # [FUNCTION DOC] HELPER: Load the seed value at the closed-over index, ignoring the iteration index.
        return ops.load_seed(seeds.get_name(), index)

    return Pointwise.create(
        device=seeds.get_device(),
        dtype=seeds.get_dtype(),
        inner_fn=inner_fn,
        ranges=[],
    )


def get_threads_per_round(device: torch.device):
    # [FUNCTION DOC] Return the number of threads in one round on a device: SM count times max threads per SM for CUDA (current device if no index), else a fixed CPU grain size (32768).
    if not isinstance(device, torch.device):
        device = torch.device(device)

    if device.type == "cuda":
        idx = device.index
        if idx is None:
            idx = torch.cuda.current_device()

        prop = torch.cuda.get_device_properties(idx)
        threads_per_round = (
            prop.multi_processor_count * prop.max_threads_per_multi_processor
        )
    else:
        _CPU_GRAIN_SIZE = 32768
        threads_per_round = _CPU_GRAIN_SIZE

    return threads_per_round


@register_lowering(inductor_prims.random, type_promotion_kind=None)
def inductor_random(
    size: list[int],
    seed: TensorBox,
    mode: str,
    *,
    offset: int = 0,
    align_dtype: torch.dtype = torch.float32,
):
    # [FUNCTION DOC] Lowering for inductor_prims.random (rand or randn). Requires fallback_random off and mode rand or randn (else raises an error). Builds a float32 Pointwise
    # whose values come from the seed and contiguous element position (plus offset); with align_random_eager on CUDA it emulates eager Philox usage with
    # threads-per-round and vector width (8 for half types, else 4). The result is realized and returned.
    if config.fallback_random:
        raise AssertionError("expected: not config.fallback_random")
    if mode not in ("rand", "randn"):
        raise AssertionError('expected: mode in ("rand", "randn")')
    size = [*size]
    dtype = torch.float32
    device = seed.get_device_or_error()
    random_pos = ir.FixedLayout(
        device, dtype, size, ir.FlexibleLayout.contiguous_strides(size), offset=offset
    ).make_indexer()
    seed_loader = seed.make_loader()

    if config.align_random_eager and device.type == "cuda":
        threads_per_round = get_threads_per_round(device)

        def _vec_from_dtype(dt: torch.dtype) -> int:
            # [FUNCTION DOC] HELPER: Vector width for eager-aligned random: 8 for fp16/bf16, otherwise 4.
            if dt in (torch.float16, torch.bfloat16):
                return 8
            return 4

        vec = _vec_from_dtype(align_dtype)

        def inner_fn(index):
            # [FUNCTION DOC] HELPER: Eager-aligned random value from the seed and base offset stored in the seed buffer.
            rng_seed = seed_loader([0])
            base_offset = seed_loader([1])
            return ops.rand_eager(
                rng_seed,
                base_offset,
                threads_per_round,
                ops.index_expr(random_pos(index), torch.int32),
                vec=int(vec),
            )
    else:

        def inner_fn(index):
            # [FUNCTION DOC] HELPER: Random value (rand or randn per mode) from the scalar seed and element position.
            return getattr(ops, mode)(
                seed_loader([]),
                ops.index_expr(random_pos(index), torch.int32),
            )

    result = Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=[*size],
    )
    result.realize()
    return result


make_fallback(inductor_prims.rand_eager_offset)
make_fallback(inductor_prims.rand_eager_offsets)


@register_lowering(inductor_prims.randint, type_promotion_kind=None)
def inductor_randint(
    low: int, high: int, size: list[int], seed: TensorBox, *, offset: int = 0
):
    # [FUNCTION DOC] Lowering for inductor_prims.randint: requires fallback_random off; returns an int64 Pointwise of random integers in [low, high) from the seed and element position plus offset.
    if config.fallback_random:
        raise AssertionError("expected: not config.fallback_random")
    size = [*size]
    dtype = torch.int64
    device = seed.get_device_or_error()
    random_pos = ir.FixedLayout(
        device, dtype, size, ir.FlexibleLayout.contiguous_strides(size), offset=offset
    ).make_indexer()
    seed_loader = seed.make_loader()

    def inner_fn(index):
        # [FUNCTION DOC] HELPER: Generate one random int64 in the range using the seed and position.
        return ops.randint64(
            seed_loader([]),
            ops.index_expr(random_pos(index), torch.int32),
            ops.value_expr(low, torch.int64),
            ops.value_expr(high, torch.int64),
        )

    return Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=[*size],
    )


def _boundaries_helper(tb: TensorBox) -> tuple[str, sympy.Expr, sympy.Expr, sympy.Expr]:
    # [FUNCTION DOC] HELPER: Return (buffer name, last-dim size, layout storage size, last stride) describing a boundaries tensor for the bucketize op.
    size = tb.get_size()
    stride = tb.get_stride()
    return (
        tb.get_name(),
        size[-1],
        tb.get_layout().storage_size(),
        stride[-1],
    )


def _realize_bucketize_lookup_tensor(tb: TensorBox) -> TensorBox:
    # [FUNCTION DOC] HELPER: Realize a bucketize lookup tensor into a named strided buffer so codegen can reference it.
    # ops.bucketize needs a named strided tensor.  realize_input() also replays
    # affine views over newly-realized inputs as ReinterpretViews when possible.
    return TensorBox(ir.ExternKernel.realize_input(tb))


def _sorter_helper(tb: TensorBox) -> tuple[str, sympy.Expr]:
    # [FUNCTION DOC] HELPER: Return (buffer name, last-dim stride) for a sorter tensor.
    return tb.get_name(), tb.get_stride()[-1]


def _bucketize_indices(
    tb: TensorBox, index: Sequence[sympy.Expr], index_dtype: torch.dtype
) -> Any:
    # [FUNCTION DOC] HELPER: Compute the flattened offset of the sequence row selected by all but the last index coordinate (including layout offset). Returns a plain int when it is a constant with empty index, else an index expression of index_dtype.
    strides = tb.get_stride()
    flattened_index = tb.get_layout().offset + sum(
        (s * i for s, i in zip(strides[:-1], index[:-1])), sympy.S.Zero
    )
    if not index and isinstance(flattened_index, sympy.Integer):
        return int(flattened_index)
    return ops.index_expr(flattened_index, index_dtype)


@register_lowering(aten.searchsorted.Tensor, type_promotion_kind=None)
def searchsorted(
    sorted_sequence: TensorBox,
    self: TensorBox,
    *,
    out_int32: bool = False,
    right: bool = False,
    side: str | None = None,
    sorter: TensorBox | None = None,
) -> TensorBox:
    # [FUNCTION DOC] Lowering for aten.searchsorted.Tensor. Falls back to the eager kernel if any of sorted_sequence, values or sorter lacks bucketize backend support.
    # Otherwise side == right overrides right, tensors are realized, and a Pointwise using the bucketize op (optionally with a sorter) is built over the values' shape with int32 or int64 output. The result is realized to avoid recomputation.
    validate_bucketize = lambda tb: V.graph.has_feature(  # noqa: E731
        tb, BackendFeature.BUCKETIZE
    )
    if (
        not validate_bucketize(sorted_sequence)
        or not validate_bucketize(self)
        or (sorter is not None and not validate_bucketize(sorter))
    ):
        return fallback_handler(aten.searchsorted.Tensor, add_to_fallback_set=False)(
            sorted_sequence,
            self,
            out_int32=out_int32,
            right=right,
            side=side,
            sorter=sorter,
        )

    # If side is present, override the value of right if needed.  This assumes that
    # validation of the two options being non-contradictory is already done by the
    # searchsorted meta-function.
    if side is not None and side == "right":
        right = True

    index_dtype = torch.int32 if out_int32 else torch.int64
    values_loader = self.make_loader()

    # The entire sorted_sequence tensor needs to be used by ops.bucketize, so we
    # need a named strided tensor for codegen below.
    sorted_sequence = _realize_bucketize_lookup_tensor(sorted_sequence)

    if sorter is not None:
        sorter = _realize_bucketize_lookup_tensor(sorter)

    if len(sorted_sequence.get_size()) == 1:

        def inner_fn(idx):
            # [FUNCTION DOC] HELPER: 1-D sequence case: bucketize a value against the whole sequence.
            val = values_loader(idx)
            return ops.bucketize(
                val,
                _boundaries_helper(sorted_sequence),
                _bucketize_indices(sorted_sequence, (), index_dtype),
                index_dtype,
                right,
                sorter=None if sorter is None else _sorter_helper(sorter),
                sorter_indices=(
                    None
                    if sorter is None
                    else _bucketize_indices(sorter, (), index_dtype)
                ),
            )

    else:

        def inner_fn(idx):
            # [FUNCTION DOC] HELPER: N-D sequence case: bucketize a value against the sequence row selected by the leading index coordinates.
            val = values_loader(idx)

            # Get index to the beginning of the sorted sequence within a flattened
            # version of the array.
            def get_flattened_index(tb: TensorBox):
                # [FUNCTION DOC] HELPER: Flattened row offset of a tensor for the current index.
                return _bucketize_indices(tb, idx, index_dtype)

            return ops.bucketize(
                val,
                _boundaries_helper(sorted_sequence),
                get_flattened_index(sorted_sequence),
                index_dtype,
                right,
                sorter=None if sorter is None else _sorter_helper(sorter),
                sorter_indices=None if sorter is None else get_flattened_index(sorter),
            )

    device = self.get_device()
    result = Pointwise.create(
        device=device,
        dtype=index_dtype,
        inner_fn=inner_fn,
        ranges=self.shape,
    )
    # see [NOTE: inductor bucketize realize]
    result.realize()

    return result


@register_lowering(
    aten.bucketize.Tensor, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
)
def bucketize(
    input: TensorBox,
    boundaries: TensorBox,
    *,
    out_int32: bool = False,
    right: bool = False,
):
    # [FUNCTION DOC] Lowering for aten.bucketize.Tensor. Boundaries must be 1-D (else raises an error). Falls back to eager if unsupported by the backend; otherwise
    # builds a Pointwise bucketize op against the realized boundaries with int32 or int64 output and realizes the result to avoid recomputing the binary search after broadcasts.
    if len(boundaries.get_size()) != 1:
        raise AssertionError("expected: len(boundaries.get_size()) == 1")

    if not (
        V.graph.has_feature(input, BackendFeature.BUCKETIZE)
        and V.graph.has_feature(boundaries, BackendFeature.BUCKETIZE)
    ):
        return fallback_handler(aten.bucketize.Tensor, add_to_fallback_set=False)(
            input, boundaries, out_int32=out_int32, right=right
        )

    # The entire boundaries tensor needs to be used by ops.bucketize, so we
    # need a named strided tensor for codegen below.
    boundaries = _realize_bucketize_lookup_tensor(boundaries)
    device = input.get_device()
    input_loader = input.make_loader()

    index_dtype = torch.int32 if out_int32 else torch.int64

    def inner_fn(index):
        # [FUNCTION DOC] HELPER: Bucketize one loaded value against the boundaries.
        val = input_loader(index)
        indices = ops.bucketize(
            val,
            _boundaries_helper(boundaries),
            _bucketize_indices(boundaries, (), index_dtype),
            index_dtype,
            right,
        )

        return indices

    result = Pointwise.create(
        device=device,
        dtype=index_dtype,
        inner_fn=inner_fn,
        ranges=input.get_size(),
    )

    # [NOTE: inductor bucketize realize]
    # bucketize_binary_search is relatively expensive, so we don't want to re-compute
    # it unnecessarily. If we run bucketize() and then broadcast the result, we don't
    # want this to be fused into a large number of duplicate bucketize() computations
    # for each of the elements in the result.
    #
    # If no broadcasting occurs, fusions can still occur in scheduler.py
    result.realize()

    return result


def _is_tensor_irnode(x):
    # [FUNCTION DOC] HELPER: True for IR nodes representing tensors (excludes non-tensor objects and opaque multi-outputs).
    return isinstance(x, ir.IRNode) and not isinstance(
        x, (ir.NonTensorObj, ir.OpaqueMultiOutput)
    )


def require_dense(_, *args, **kwargs):
    # [FUNCTION DOC] Layout constraint: require stride-1 (dense) layout on every tensor IR node among args and kwargs. Returns (args, kwargs). First parameter (node) is unused.
    args, kwargs = pytree.tree_map_only(
        _is_tensor_irnode, ir.ExternKernel.require_stride1, (args, kwargs)
    )
    return args, kwargs


def require_contiguous(_, *args, **kwargs):
    # [FUNCTION DOC] Layout constraint: require contiguous layout on every tensor IR node in args and kwargs; returns (args, kwargs).
    args, kwargs = pytree.tree_map_only(
        _is_tensor_irnode,
        ir.ExternKernel.require_contiguous,
        (args, kwargs),
    )
    return args, kwargs


def require_contiguous_strides(_, *args, **kwargs):
    # [FUNCTION DOC] Layout constraint: require contiguous strides on every tensor IR node in args and kwargs; returns (args, kwargs).
    # TODO: combine this with require_contiguous after
    # https://github.com/pytorch/pytorch/pull/148235 lands.
    args, kwargs = pytree.tree_map_only(
        _is_tensor_irnode,
        ir.ExternKernel.require_contiguous_strides,
        (args, kwargs),
    )
    return args, kwargs


def require_channels_last(_, *args, **kwargs):
    # [FUNCTION DOC] Layout constraint: require channels-last layout on every tensor IR node in args and kwargs; returns (args, kwargs).
    args, kwargs = pytree.tree_map_only(
        _is_tensor_irnode,
        ir.ExternKernel.require_channels_last,
        (args, kwargs),
    )
    return args, kwargs


def constrain_to_fake_tensor(arg, fake_arg):
    # [FUNCTION DOC] HELPER: Recursively require exact strides equal to those of the fake tensor arg for IR nodes, descending into dicts, tuples and lists; script objects and None pass through.
    if fake_arg is None:
        return arg
    if isinstance(fake_arg, FakeScriptObject) or is_custom_class_obj(fake_arg):
        return arg
    if isinstance(arg, ir.IRNode):
        return ir.ExternKernel.require_exact_strides(arg, fake_arg.stride())
    if isinstance(arg, dict):
        return {key: constrain_to_fake_tensor(arg[key], fake_arg[key]) for key in arg}
    elif isinstance(arg, (tuple, list)):
        return type(arg)(
            constrain_to_fake_tensor(a, f_a) for (a, f_a) in zip(arg, fake_arg)
        )
    return arg


def constrain_to_fake_tensors(args, kwargs, fake_args, fake_kwargs):
    # [FUNCTION DOC] Layout constraint: apply exact-stride constraints from the fake args/kwargs to each arg and kwarg; returns (args, kwargs).
    args = tuple(
        constrain_to_fake_tensor(arg, fake_arg)
        for arg, fake_arg in zip(args, fake_args)
    )
    kwargs = {k: constrain_to_fake_tensor(v, fake_kwargs[k]) for k, v in kwargs.items()}
    return args, kwargs


def constrain_to_fx_strides(fx_node, *args, **kwargs):
    # [FUNCTION DOC] Layout constraint: for tensor IR args, require the stride order of the corresponding FX node's fake value, recursing through dicts and sequences. Returns (args, kwargs).
    def get_fake_val(fx_arg):
        # [FUNCTION DOC] HELPER: Return the fake value metadata of an FX node, or the argument itself if not a node.
        if isinstance(fx_arg, torch.fx.Node):
            return fx_arg.meta.get("val")
        return fx_arg

    def apply_constraint(arg, fx_arg):
        # [FUNCTION DOC] HELPER: Apply the stride-order constraint to one argument against its FX counterpart, recursing into containers; mismatched types pass through unchanged.
        fake_val = get_fake_val(fx_arg)
        if _is_tensor_irnode(arg):
            if not isinstance(fake_val, torch.Tensor):
                return arg
            stride_order = ir.get_stride_order(
                fake_val.stride(), V.graph.sizevars.shape_env
            )
            return ir.ExternKernel.require_stride_order(arg, stride_order)
        if isinstance(arg, dict):
            if not isinstance(fake_val, dict):
                return arg
            return {key: apply_constraint(arg[key], fake_val[key]) for key in arg}
        if isinstance(arg, (tuple, list)):
            if not isinstance(fake_val, (tuple, list)):
                return arg
            return type(arg)(
                apply_constraint(a, fx_a) for a, fx_a in zip(arg, fake_val)
            )
        return arg

    args = tuple(
        apply_constraint(arg, fx_arg) for arg, fx_arg in zip(args, fx_node.args)
    )
    kwargs = {k: apply_constraint(v, fx_node.kwargs[k]) for k, v in kwargs.items()}
    return args, kwargs


def constrain_to_fx_strides_if_fallback_random(fx_node, *args, **kwargs):
    # [FUNCTION DOC] Layout constraint that applies constrain_to_fx_strides only when config.fallback_random is on; otherwise returns args and kwargs unchanged.
    if not config.fallback_random:
        return args, kwargs
    return constrain_to_fx_strides(fx_node, *args, **kwargs)


# native_dropout uses empty_like(input) internally, so bernoulli_ consumes
# RNG values in the input's stride order. Constrain input strides to match
# the FX graph (i.e. eager) so the dropout mask is identical.
add_layout_constraint(
    aten.native_dropout.default, constrain_to_fx_strides_if_fallback_random
)


def sdpa_constraint(fx_node, *args, **kwargs):
    # [FUNCTION DOC] Layout constraint for scaled-dot-product-attention fallbacks. For every tensor IR argument, picks the stride order of the fake value (forced contiguous
    # when the last dim is not the fastest, and (3,1,2,0) for efficient-attention-backward out/grad args), optionally caches results in the graph per (argument id, name, order),
    # and applies a constraint: non-CUDA/XPU simply requires the stride order; CUDA/XPU additionally keeps 8-aligned tensors, pads strides of the efficient-attention bias to
    # alignment (allowing zero-stride expanded dims), and otherwise requires the stride order. Returns (args, kwargs).
    """Apply stride constraints to SDPA inputs, ensuring dense last dimension."""

    def apply_constraint(idx, arg, fx_arg):
        # [FUNCTION DOC] HELPER: Constrain one SDPA argument (idx is its position, -1 for kwargs) against its FX node: computes the target stride order, consults/updates the sdpa constraint cache when enabled, and delegates to the inner constraint logic.
        if not _is_tensor_irnode(arg):
            return arg

        meta_val = fx_arg.meta["val"]
        meta_stride_expr = [
            s.node.expr if isinstance(s, torch.SymInt) else s for s in meta_val.stride()
        ]
        shape_env = V.graph.sizevars.shape_env
        stride_order = ir.get_stride_order(meta_val.stride(), shape_env)

        if stride_order and stride_order[-1] != 0:
            # contiguous stride order
            stride_order = list(reversed(range(len(arg.get_size()))))

        if (
            fx_node.target
            == aten._scaled_dot_product_efficient_attention_backward.default
            and idx in (0, 5)
        ):
            if len(stride_order) != 4:
                raise AssertionError("expected: len(stride_order) == 4")
            # The 0 and 5th arguments for aten._scaled_dot_product_efficient_attention_backward.default
            # are for out and gradient_out. They have to be in
            # (3, 1, 2, 0) stride order. Otherwise the kernel will crash.
            # Check https://github.com/pytorch/pytorch/issues/138772
            stride_order = (3, 1, 2, 0)

        # Cache keyed by (id(arg), arg_name, stride_order) to avoid
        # duplicate copy_input when the same tensor feeds multiple SDPA
        # positions (e.g., key=value).  Including arg_name handles
        # mutation: mark_buffer_mutated() renames the buffer in place,
        # so a mutated tensor has the same id but a different name,
        # causing a cache miss.
        cache_key = None
        if config.cache_sdpa_constraint:
            arg_name = arg.maybe_get_name()
            cache_key = (
                id(arg),
                arg_name,
                tuple(stride_order) if stride_order else None,
            )
            if cache_key in V.graph.sdpa_constraint_cache:
                return V.graph.sdpa_constraint_cache[cache_key]

        result = _apply_constraint_inner(
            idx, arg, meta_val, meta_stride_expr, stride_order
        )
        if cache_key is not None:
            V.graph.sdpa_constraint_cache[cache_key] = result
        return result

    def _apply_constraint_inner(idx, arg, meta_val, meta_stride_expr, stride_order):
        # [FUNCTION DOC] HELPER: Apply the SDPA stride constraint to a tensor. Non-GPU: require stride order. GPU: return arg for non 3-D/4-D tensors; realize and match insignificant strides when the
        # tensor is aligned to 8; pad strides for the efficient attention bias (dense last dim); match strides for views whose base is aligned; else require the stride order.
        # KNOWN ISSUE: the second check on is_aligned_tensor after the first identical early return is unreachable, and the same duplicate check appears twice more, so those branches are dead code.
        if not (meta_val.is_cuda or meta_val.is_xpu):
            return ir.ExternKernel.require_stride_order(arg, stride_order)

        # This is the minimum alignment required by SDPA kernels for attention_bias.
        # This value can be found in pytorch/aten/src/ATen/native/transformers/attention.cpp preprocess_mask
        ALIGNMENT = 8

        # effn_attn_fwd does requires dense last dim, not just alignment
        effn_attn_fwd_bias = (
            fx_node.target
            == torch.ops.aten._scaled_dot_product_efficient_attention.default
            and idx == 3
        )

        if not (isinstance(arg, TensorBox)):
            raise AssertionError("expected: isinstance(arg, TensorBox)")
        if len(arg.get_size()) not in (3, 4):
            return arg

        is_aligned_tensor = ir.is_aligned_realized_tensor(arg, ALIGNMENT)
        if is_aligned_tensor:
            return ir.try_match_insignificant_strides(
                ir.ExternKernel.realize_input(arg), meta_stride_expr
            )

        if (
            isinstance(arg, IRNode)
            and arg.maybe_get_stride() is not None
            and is_aligned_tensor
        ):
            return ir.try_match_insignificant_strides(
                ir.ExternKernel.realize_input(arg), meta_stride_expr
            )

        if effn_attn_fwd_bias:
            out_size = list(arg.get_size())

            expanded_dims = []
            # We require a dense last dimension, but the other strides
            # can be expanded, which results in a smaller tensor
            maybe_stride = arg.maybe_get_stride()
            for i in range(len(arg.get_size()) - 1):
                if V.graph.sizevars.statically_known_equals(meta_stride_expr[i], 0) or (
                    maybe_stride is not None
                    and V.graph.sizevars.statically_known_equals(maybe_stride[i], 0)
                ):
                    expanded_dims.append(i)

            # Now, pad strides to alignment
            out_strides = [-1] * len(out_size)
            out_strides[-1] = 1
            stride = 1
            for i in range(len(out_size) - 2, -1, -1):
                if out_strides[i + 1] != 0:
                    stride = stride * out_size[i + 1]

                # the expanded dims still need to be aligned, if they are,
                # we can make them expanded by setting the stride equal to 0
                if i in expanded_dims:
                    if V.graph.sizevars.statically_known_equals(
                        Mod(out_strides[i + 1], ALIGNMENT), 0
                    ):
                        out_strides[i] = 0
                        continue

                if not V.graph.sizevars.statically_known_equals(
                    Mod(stride, ALIGNMENT), 0
                ):
                    stride = ceildiv(stride, ALIGNMENT) * ALIGNMENT

                out_strides[i] = stride

            return ir.ExternKernel.require_exact_strides(arg, out_strides)

        if is_aligned_tensor:
            return ir.try_match_insignificant_strides(
                ir.ExternKernel.realize_input(arg), meta_stride_expr
            )

        if (
            isinstance(arg, IRNode)
            and arg.maybe_get_stride() is not None
            and is_aligned_tensor
        ):
            return ir.try_match_insignificant_strides(
                ir.ExternKernel.realize_input(arg), meta_stride_expr
            )

        def is_aligned(x):
            # [FUNCTION DOC] HELPER: True if the tensor is non-scalar and its last dim size is statically a multiple of the alignment (8).
            size = x.get_size()
            if len(size) == 0:
                return False
            return V.graph.sizevars.guard_or_false(
                sympy.Eq(Mod(size[-1], ALIGNMENT), 0)
            )

        if isinstance(arg.data, ir.BaseView):
            if not is_aligned(arg):
                if is_aligned(arg.unwrap_view()):
                    return ir.try_match_insignificant_strides(
                        ir.ExternKernel.realize_input(arg), meta_stride_expr
                    )

        return ir.ExternKernel.require_stride_order(arg, stride_order)

    args = tuple(
        apply_constraint(idx, arg, fx_arg)
        for idx, (arg, fx_arg) in enumerate(zip(args, fx_node.args))
    )
    kwargs = {k: apply_constraint(-1, v, fx_node.kwargs[k]) for k, v in kwargs.items()}
    return args, kwargs


# WIP
make_fallback(aten._adaptive_avg_pool3d)  # @isuruf
make_fallback(aten.adaptive_max_pool3d, override_decomp=True)
make_fallback(aten._scaled_dot_product_attention_math_for_mps)  # @malfet


# 1) Easy
make_fallback(aten.uniform, override_decomp=True)
make_fallback(aten.exponential.default, warn=False)  # (fails accuracy on test_torch.py)
make_fallback(aten._pdist_forward, require_contiguous)  # Has decomp. Needs benchmarks
make_fallback(aten.soft_margin_loss_backward, warn=False)  # py_impl?
make_fallback(aten._fused_rms_norm, warn=False)  # (MPS-only and faster than decomp)
if torch.xpu._is_compiled():
    make_fallback(
        aten.embedding_dense_backward, warn=False
    )  # (XPU-only and faster than decomp)

if torch.mtia._is_compiled():
    make_fallback(
        aten.native_layer_norm, warn=False
    )  # (MTIA-only and faster than decomp)

# 1.5) Easy or Impossible
make_fallback(aten._cdist_forward)  # p=2 should be feasible
make_fallback(aten._cdist_backward)

# 2) Medium
make_fallback(aten._trilinear)


# 3) Difficult
# Scans
# See the discussion at
# https://dev-discuss.pytorch.org/t/pytorch-sparse-gnn-compiler-rfc/1644/19
make_fallback(aten.segment_reduce.default)
make_fallback(aten._segment_reduce_backward.default)

# Histogram (need to implement Histogram IR)
make_fallback(aten.histc)
make_fallback(aten.histogram.bin_ct)
make_fallback(aten._histogramdd_bin_edges.default)
make_fallback(aten._histogramdd_from_bin_cts.default)

# Need templated kernel
make_fallback(aten.addbmm)
make_fallback(aten._addmm_activation, warn=False)

make_fallback(aten._grouped_mm, require_dense)

# Need templated kernel. Probably impossible to write efficiently
make_fallback(aten._cudnn_rnn, require_dense)
make_fallback(aten._cudnn_rnn_backward, require_contiguous)
make_fallback(aten.miopen_rnn, require_dense)
make_fallback(aten.miopen_rnn_backward, require_contiguous)

# Haven't checked but sound difficult / impossible
make_fallback(aten._embedding_bag, require_contiguous)
make_fallback(aten._embedding_bag_forward_only, require_contiguous)
make_fallback(aten._embedding_bag_backward)
make_fallback(aten._embedding_bag_per_sample_weights_backward)
make_fallback(aten._embedding_bag_per_sample_weights_backward)
make_fallback(aten._fused_moving_avg_obs_fq_helper)
make_fallback(aten._fused_moving_avg_obs_fq_helper_functional)


# 4) Backwards (try py_impl'ing them) when fwd is written as a decomp
make_fallback(aten.max_pool3d_with_indices_backward)
make_fallback(aten._adaptive_avg_pool2d_backward, require_dense)
make_fallback(aten._adaptive_avg_pool3d_backward)
make_fallback(aten.adaptive_max_pool2d_backward)
make_fallback(aten.adaptive_max_pool3d_backward)
make_fallback(aten.fractional_max_pool2d_backward)
make_fallback(aten.fractional_max_pool3d_backward)
make_fallback(aten.replication_pad1d_backward)
make_fallback(aten.replication_pad2d_backward)
make_fallback(aten.upsample_linear1d_backward)
make_fallback(aten.upsample_bicubic2d_backward, require_contiguous)
make_fallback(aten.upsample_trilinear3d_backward)
make_fallback(aten.grid_sampler_2d_backward)
make_fallback(aten._pdist_backward, require_contiguous)


# 5) Impossible (missing triton/CPU features)

# Sorting / Sorting-like
make_fallback(aten.nanmedian)
make_fallback(aten.multinomial.default, warn=False)
make_fallback(aten.randperm)
# see: https://github.com/pytorch/pytorch/pull/121354
make_fallback(aten.resize_)
make_fallback(aten.resize_as_)

# Linalg
make_fallback(aten._linalg_det)
make_fallback(aten.linalg_householder_product)
make_fallback(aten.linalg_inv_ex)
make_fallback(aten.linalg_ldl_factor_ex)
make_fallback(aten.linalg_ldl_solve)
make_fallback(aten.linalg_lu)
make_fallback(aten.linalg_lu_factor_ex)
make_fallback(aten.linalg_lu_solve)
make_fallback(aten.linalg_matrix_exp)
make_fallback(aten.linalg_matrix_sqrth)
make_fallback(aten.linalg_polar)
make_fallback(aten.linalg_qr)
make_fallback(aten._linalg_slogdet)
make_fallback(aten._linalg_solve_ex)
make_fallback(aten.linalg_solve_triangular)
make_fallback(aten._linalg_svd)
make_fallback(aten.lu_unpack)
make_fallback(aten.ormqr)
make_fallback(aten._linalg_check_errors)
make_fallback(aten.linalg_pinv.atol_rtol_tensor)
make_fallback(aten._linalg_eigh)
make_fallback(aten.triangular_solve)
make_fallback(aten.linalg_cholesky_ex)
make_fallback(aten.cholesky_inverse)
make_fallback(aten.cholesky_solve)
make_fallback(aten.geqrf)
make_fallback(aten._fft_r2c)  # needs complex as well

# Data dependent (are these necessary?)
make_fallback(aten.nonzero.default)
# Not data-dependent, but still using fallback
make_fallback(aten.nonzero_static.default)
# Data-dependent output size; route to ATen eager kernel (CPU/CUDA/XPU all have
# native implementations)
make_fallback(aten.bincount.default, warn=False)
make_fallback(aten._unique2.default, warn=False)
make_fallback(aten.unique_dim.default, warn=False)
make_fallback(aten.unique_consecutive.default, warn=False)
make_fallback(aten.unique_dim_consecutive.default, warn=False)

# Misc
make_fallback(aten.gcd.default, warn=False)
make_fallback(aten._thnn_fused_lstm_cell, require_dense)
make_fallback(torch._prims.rng_prims.run_and_save_rng_state)
make_fallback(torch._prims.rng_prims.run_with_rng_state)
make_fallback(torch._prims.rng_prims.graphsafe_run_with_rng_state)
make_fallback(torch._prims.rng_prims.run_dtensor_rng_op)

# AMP / GradScaler ops: both the in-place (_amp_update_scale_) and functional
# (_amp_update_scale) variants need explicit fallbacks.  Inductor uses the
# functional form when the input tensor is a computed (non-leaf) value.
make_fallback(aten._amp_update_scale_.default, warn=False)
make_fallback(aten._amp_update_scale.default, warn=False)


# Implemented / Half implemented
# Scans. Implemented for CUDA, missing CPU
make_fallback(aten.masked_scatter)
make_fallback(aten.masked_scatter_backward)

# Complex number support
make_fallback(aten.view_as_complex, require_contiguous)
make_fallback(aten.angle)  # needs complex

# Needs Sparse
make_fallback(aten._sparse_coo_tensor_with_dims_and_tensors)
make_fallback(aten.to_sparse)
make_fallback(aten._to_sparse)


@register_lowering(aten._to_dense.default, type_promotion_kind=None)
def _to_dense(x, dtype=None, masked_grad=None):
    # [FUNCTION DOC] Lowering for aten._to_dense. Realizes the input and routes the op through a FallbackKernel with a MultiOutput describing a single contiguous dense output of dtype
    # (default the input dtype). Raises an error if the realized input has no device. Returns a TensorBox of the output.
    x = ir.ExternKernel.realize_input(x)
    size = x.get_size()
    device = x.get_device()
    if device is None:
        raise AssertionError("realized _to_dense input must have a device")
    output_dtype = dtype if dtype is not None else x.get_dtype()

    def unflatten_args(tensor_args, non_tensor_args):
        # [FUNCTION DOC] HELPER: Rebuild the call arguments for the fallback kernel from tensor and non-tensor args.
        return [tensor_args[0], *non_tensor_args], {}

    # cpp_wrapper has no direct AOTI C shim for _to_dense, so route through the
    # proxy-executor fallback while still describing the single dense output.
    # MultiOutput lets this single-output op force a contiguous strided output
    # layout instead of inheriting anything from the opaque MKLDNN input.
    packed = ir.FallbackKernel(
        ir.MultiOutputLayout(device=device),
        aten._to_dense.default,
        [x],
        [dtype, masked_grad],
        unflatten_args,
    )
    out = ir.MultiOutput(
        ir.FixedLayout(
            device,
            output_dtype,
            size,
            ir.FlexibleLayout.contiguous_strides(size),
        ),
        packed,
        [],
    )
    packed.outputs = [out]
    return TensorBox.create(out)


# 6) Pattern-matched
make_fallback(
    aten._scaled_dot_product_efficient_attention.default,
    sdpa_constraint,
    warn=False,
)
make_fallback(
    aten._scaled_dot_product_efficient_attention_backward.default,
    sdpa_constraint,
    warn=False,
)
make_fallback(
    aten._scaled_dot_product_flash_attention.default,
    sdpa_constraint,
    warn=False,
)
make_fallback(
    aten._scaled_dot_product_flash_attention.quantized,
    warn=False,
)
make_fallback(
    aten._scaled_dot_product_flash_attention_backward.default,
    sdpa_constraint,
    warn=False,
)
make_fallback(
    aten._scaled_dot_product_cudnn_attention.default,
    sdpa_constraint,
    warn=False,
)
make_fallback(
    aten._scaled_dot_product_cudnn_attention_backward.default,
    sdpa_constraint,
    warn=False,
)
make_fallback(
    aten._scaled_dot_product_flash_attention_for_cpu.default,
    sdpa_constraint,
    warn=False,
)
make_fallback(
    aten._scaled_dot_product_flash_attention_for_cpu_backward.default,
    sdpa_constraint,
    warn=False,
)
make_fallback(
    aten._scaled_dot_product_fused_attention_overrideable.default,
    sdpa_constraint,
    warn=False,
)
make_fallback(
    aten._scaled_dot_product_fused_attention_overrideable_backward.default,
    sdpa_constraint,
    warn=False,
)
make_fallback(aten._flash_attention_forward.default, sdpa_constraint)
make_fallback(aten._flash_attention_forward.quantized)
make_fallback(aten._flash_attention_backward.default, sdpa_constraint)
make_fallback(aten._efficient_attention_forward.default, sdpa_constraint)
make_fallback(aten._efficient_attention_backward.default, sdpa_constraint)

# index_reduce requires fallback when use_scatter_fallback(...) returns True
make_fallback(aten.index_reduce)
make_fallback(aten.repeat_interleave.Tensor, override_decomp=True)

make_fallback(aten._weight_norm_interface_backward.default, require_contiguous)


# Register with type_promotion_kind None.
# For example, fp16.copy_(fp32) should **not** promote the first input's dtype.
@register_lowering(aten.copy, type_promotion_kind=None)
def copy(self, src, non_blocking=False):
    # [FUNCTION DOC] Lowering for aten.copy (no type promotion). Non-IR sources become tensors of self's dtype/device. src is moved to self's device and dtype, expanded to self's
    # size when shapes differ, and cloned. If self is pinned, the result is realized as pinned. Returns the new value.
    if not isinstance(src, ir.IRNode):
        src = tensor(src, dtype=self.get_dtype(), device=self.get_device())
    x = src
    if self.get_device() != src.get_device():
        x = to_device(x, self.get_device())
    if self.get_dtype() != src.get_dtype():
        x = to_dtype(x, self.get_dtype())

    if self.get_size() != src.get_size():
        out = expand(x, self.get_size())
        result = clone(out)
    else:
        result = clone(x)

    self_layout = self.maybe_get_layout()
    if self_layout is not None and self_layout.is_pinned:
        _realize_as_pinned(result)
    return result


@register_lowering(aten.clone)
def clone(x, *, memory_format=None):
    # [FUNCTION DOC] Lowering for aten.clone: a Pointwise copy of x using its loader; memory_format is ignored so layout can be decided later.
    # Don't materialize the layout here based on memory_format,
    # as we want to give the scheduler opportunity to perform layout optimization.
    # Let the downstream op handle the input stride as needed.
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=x.make_loader(),
        ranges=list(x.get_size()),
    )


def _realize_as_pinned(result):
    # [FUNCTION DOC] HELPER: Realize result and mark its layout as pinned memory.
    result.realize()
    result.data.data.get_layout().is_pinned = True


def clone_preserve_reinterpret_view(x):
    # [FUNCTION DOC] Clone x while preserving any chain of ReinterpretView layouts: the views are unwrapped, the base cloned, and the view layouts re-applied in reverse order. Returns a TensorBox.
    reinterpret_view_layouts = []
    if isinstance(x, TensorBox) and isinstance(x.data, ir.ReinterpretView):
        x = x.data  # unwrap TensorBox
        # pyrefly: ignore [bad-assignment]
        while isinstance(x, ir.ReinterpretView):
            reinterpret_view_layouts.append(x.get_layout())
            x = x.data
        x = TensorBox(x)

    x = clone(x)

    if reinterpret_view_layouts:
        x = x.data  # unwrap TensorBox
        for layout in reinterpret_view_layouts[::-1]:
            x = ir.ReinterpretView(data=x, layout=layout)
        x = TensorBox(x)

    return x


def lift_fresh_copy(x):
    # [FUNCTION DOC] Lowering for aten.lift_fresh_copy: clone x, and mark the clone pinned if the input layout is pinned. Registered when the aten op exists.
    result = clone(x)
    input_layout = x.maybe_get_layout()
    if input_layout is not None and input_layout.is_pinned:
        # torch.tensor(..., pin_memory=True) constants reach Inductor as a
        # pinned constant followed by lift_fresh_copy.
        _realize_as_pinned(result)
    return result


if hasattr(aten, "lift_fresh_copy"):
    register_lowering(aten.lift_fresh_copy)(lift_fresh_copy)


@register_lowering(prims.iota)
def iota(
    length,
    *,
    start,
    step,
    dtype,
    device,
    requires_grad,
):
    # [FUNCTION DOC] Lowering for prims.iota: a 1-D Pointwise of length elements with value start plus step times the index in dtype on the decoded device.
    def fn(index):
        # [FUNCTION DOC] HELPER: Compute the iota value for index.
        return ops.index_expr(step * index[0] + start, dtype=dtype)

    return Pointwise.create(
        device=decode_device(device),
        dtype=dtype,
        inner_fn=fn,
        ranges=[length],
    )


@register_lowering(aten.arange.start_step, type_promotion_kind=None)
def arange_start_step(
    start,
    end,
    step=1,
    *,
    dtype=None,
    device=None,
    layout=None,
    pin_memory=None,
    requires_grad=False,
):
    # [FUNCTION DOC] Lowering for aten.arange.start_step: requires dtype (raises an error if None); length is ceil((end - start) / step); delegates to iota, default device cpu.
    if dtype is None:
        raise AssertionError("expected: dtype is not None")
    length = ceildiv(end - start, step)
    return iota(
        length,
        start=start,
        step=step,
        dtype=dtype,
        device=device if device is not None else "cpu",
        requires_grad=requires_grad,
    )


@register_lowering(aten.select_scatter, type_promotion_kind=None)
def select_scatter(x, src, dim: int, index: int):
    # [FUNCTION DOC] Lowering for aten.select_scatter: casts src to x's dtype; wraps negative index; uses an eager fallback when the index sign is unknown (unbacked);
    # otherwise checks bounds and returns a Pointwise selecting the expanded src where the coordinate on dim equals index, else x.
    src = to_dtype(src, x.get_dtype())
    x_loader = x.make_loader()
    dim = _validate_dim(x, dim, 0)
    if V.graph.sizevars.guard_or_false(sympy.Lt(index, 0)):
        index = index + x.get_size()[dim]
    elif V.graph.sizevars.guard_or_false(sympy.Ge(index, 0)):
        pass
    else:
        # unbacked index
        return fallback_handler(aten.select_scatter.default)(x, src, dim, index)

    V.graph.sizevars.check_leq(0, index)  # type: ignore[arg-type]
    V.graph.sizevars.check_lt(index, x.get_size()[dim])  # type: ignore[arg-type]
    src = expand(unsqueeze(src, dim), x.get_size())
    src_loader = src.make_loader()

    def inner_fn(idx):
        # [FUNCTION DOC] HELPER: Choose src or x at idx based on equality of the dim coordinate with the select index.
        return ops.where(
            ops.eq(
                ops.index_expr(idx[dim], torch.int32),
                ops.index_expr(index, torch.int32),
            ),
            src_loader(idx),
            x_loader(idx),
        )

    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(x.get_size()),
    )


@register_lowering(aten.slice_scatter, type_promotion_kind=None)
def slice_scatter(x, src, dim=0, start=None, end=None, step=1):
    # [FUNCTION DOC] Lowering for aten.slice_scatter: returns x with the strided slice along dim replaced by src. src is cast to x's dtype. With unbacked bounds, indices are resolved (or
    # the eager fallback is used if undeterminable); otherwise start/end are normalized. Checks src shape matches the slice (raises an error on rank mismatch), expands src, and builds a
    # Pointwise that selects src inside the slice (masked by start, end and step) and x outside; a full-range slice just reads src.
    """Lower slice_scatter with exact source shape checks."""
    src = to_dtype(src, x.get_dtype())
    x_loader = x.make_loader()
    dim = _validate_dim(x, dim, 0)
    dim_size = x.get_size()[dim]

    if any(free_unbacked_symbols(x) for x in (start, end, dim_size)):
        start_index = _compute_slice_index(start, dim_size, 0)
        if end is not None and V.graph.sizevars.statically_known_equals(
            end, sys.maxsize
        ):
            end_index = dim_size
        else:
            end_index = _compute_slice_index(end, dim_size, dim_size)

        if start_index is None or end_index is None:
            return fallback_handler(aten.slice_scatter.default)(
                x, src, dim, start, end, step
            )

        start = start_index
        end = _clamp_slice_end_to_start(end_index, start)
    else:
        # pyrefly: ignore [bad-argument-type]
        start, end = ir.SliceView.normalize_start_end(x, dim, start, end)

    src_size = list(x.get_size())
    src_size[dim] = FloorDiv(end - start + (step - 1), step)
    if len(src.get_size()) != len(src_size):
        raise AssertionError("expected src and slice to have the same rank")
    for actual, expected in zip(src.get_size(), src_size):
        V.graph.sizevars.check_equals(actual, expected)
    src = expand(src, src_size)
    src_loader = src.make_loader()

    def inner_fn(idx):
        # [FUNCTION DOC] HELPER: Pick src (at the slice-relative index) where the dim coordinate lies within [start, end) on the step grid, else x. Full-range step-1 slices return src directly.
        if start == 0 and end == dim_size and step == 1:
            # selecting every element is the same as just src.clone()
            return src_loader(idx)

        idx_dim = ops.index_expr(idx[dim], torch.int64)
        src_idx = list(idx)
        src_idx[dim] = FloorDiv(idx[dim] - start, step)

        mask = []
        if start != 0:
            mask.append(
                ops.ge(
                    idx_dim,
                    ops.index_expr(sympy.expand(start), torch.int64),
                )
            )
        if end != dim_size:
            mask.append(
                ops.lt(
                    idx_dim,
                    ops.index_expr(sympy.expand(end), torch.int64),
                )
            )
        if step != 1:
            mask.append(
                ops.eq(
                    ops.index_expr(
                        ModularIndexing(idx[dim] - start, 1, step), torch.int64
                    ),
                    ops.constant(0, torch.int64),
                )
            )
        if not (mask):
            raise AssertionError("expected: mask")
        mask = functools.reduce(ops.and_, mask)
        src_val = ops.masked(
            mask,
            lambda: src_loader(src_idx),
            0 if is_integer_type(x) else 0.0,
        )
        return ops.where(
            mask,
            src_val,
            x_loader(idx),
        )

    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(x.get_size()),
    )


def _unwrap(x):
    # [FUNCTION DOC] HELPER: Return the first leaf element of nested non-empty lists/tuples, else x itself.
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return _unwrap(x[0])
    return x


@register_lowering([torch.tensor, aten.scalar_tensor, prims.scalar_tensor])
def tensor(data, *, dtype=None, device=None, layout=None, pin_memory=False):
    # [FUNCTION DOC] Lowering for torch.tensor / scalar_tensor. Only strided layout (jagged treated as strided) is supported (else NotImplementedError). dtype defaults to int64 for integer data or the default float dtype.
    # Sympy and python scalars become 0-d constants (fp16/bf16 values are rounded), short lists (up to 8) become a binary-search select over the index, and larger data is added as a graph tensor constant.
    # Pinned memory results are realized as pinned. Returns an IR tensor.
    # Match eager/meta scalar_tensor behavior; NJT handles scalar broadcasting.
    if layout == torch.jagged:
        layout = torch.strided
    assert_nyi(layout in (None, torch.strided), f"layout={layout}")
    if isinstance(_unwrap(data), int):
        dtype = dtype or torch.int64
    else:
        dtype = dtype or torch.get_default_dtype()

    ranges: list[sympy.Expr] = []

    _truncate_fp = dtype in (torch.bfloat16, torch.float16)

    if isinstance(data, sympy.Basic):

        def inner_fn(index):
            # [FUNCTION DOC] HELPER: Produce the sympy expression value cast to dtype, rounding through low precision for fp16/bf16.
            result = ops.index_expr(data, dtype)
            if _truncate_fp:
                result = ops.to_dtype(result, dtype, use_compute_types=False)
                result = ops.to_dtype(result, dtype)
            return result

    elif isinstance(data, (float, int)):
        if _truncate_fp and isinstance(data, float):
            data = torch.tensor(data, dtype=dtype).item()

        def inner_fn(index):
            # [FUNCTION DOC] HELPER: Produce the constant scalar value.
            return ops.constant(data, dtype)

    elif len(data) == 0 or isinstance(data[0], (float, int)) and len(data) <= 8:
        # inline small tensors
        ranges.append(sympy.Integer(len(data)))

        def inner_fn(index):
            # [FUNCTION DOC] HELPER: Select element data[index[0]] via a nested where over index comparisons; empty data yields zero.
            def binary_search(start, end):
                # [FUNCTION DOC] HELPER: Recursive bisection producing a constant for single entries and a where on the midpoint otherwise.
                if start >= end:
                    raise AssertionError("expected: start < end")
                if end - start == 1:
                    return ops.constant(data[start], dtype)
                mid = (end - start) // 2 + start
                return ops.where(
                    ops.lt(
                        ops.index_expr(index[0], torch.int64),
                        ops.constant(mid, torch.int64),
                    ),
                    binary_search(start, mid),
                    binary_search(mid, end),
                )

            if len(data) == 0:
                return ops.constant(0, dtype)
            return binary_search(0, len(data))

    else:
        return V.graph.add_tensor_constant(
            torch.tensor(
                data, dtype=dtype, device=device, pin_memory=pin_memory or False
            )
        )

    result = Pointwise.create(
        device=decode_device(device),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=ranges,
    )
    if pin_memory:
        _realize_as_pinned(result)
    return result


@register_lowering(torch.as_tensor)
def as_tensor(data, dtype=None, device=None):
    # [FUNCTION DOC] Lowering for torch.as_tensor: for an IR tensor applies optional dtype and device conversions and returns it; otherwise builds a tensor from data.
    if isinstance(data, TensorBox):
        if dtype is not None:
            data = to_dtype(data, dtype)
        if device is not None:
            data = to_device(data, device)
        return data
    return tensor(data, dtype=dtype, device=device)


@register_lowering(torch.LongTensor)
def long_tensor(data):
    # [FUNCTION DOC] Lowering for torch.LongTensor: builds an int64 tensor from data.
    return tensor(data, dtype=torch.int64)


@register_lowering(aten._local_scalar_dense)
def _local_scalar_dense(data):
    # [FUNCTION DOC] Lowering for aten._local_scalar_dense (item). Looks up the single unbacked symbol binding of the current node, registers a DynamicScalar buffer that generates the item call, and returns the
    # node's value as a sympy expression (not a tensor). Raises an error if the unbacked bindings are missing or not exactly one. Side effect: registers a buffer and operation.
    # This is interesting!  Most lowerings return tensors, so you can just
    # return the buffer you allocated and it will get used (or not used, if
    # it's dead.)  But _local_scalar_dense (aka item) returns an int,
    # not a Tensor, so you would have a type mismatch if you return a buffer;
    # we are obligated to return a sympy expression instead.  However,
    # we need to actually codegen the .item() call somehow.  We do this
    # by registering a faux buffer for the DynamicScalar IR node, which is
    # solely responsible for generating this .item().  The buffer is
    # not used for anything (notice we discard it); at codegen time,
    # the "buffer" just gets assigned None.
    unbacked_bindings = resolve_unbacked_bindings(
        V.graph.sizevars.shape_env, V.graph.current_node.meta["unbacked_bindings"]
    )
    if unbacked_bindings is None:
        raise AssertionError("expected: unbacked_bindings is not None")
    if len(unbacked_bindings) != 1:
        raise AssertionError(unbacked_bindings)
    # NB: Have to be very careful here.  V.graph.current_node.meta["val"]
    # seemingly also contains a symbol which you want to do binding for,
    # but it actually isn't.  In particular, if we have later performed
    # a deferred runtime assert saying that u0 == s0, you will actually
    # see s0 from expr!  This is bad because we need to actually generate
    # the assert that says u0 == s0, so we need to know where to get u0
    # from (this call).  In particular, we must use unbacked_bindings, which
    # is guaranteed to have the original, unreplaced symbol in question.
    #
    # NB2: Another thing we have to be very careful about are symbol bindings
    # that require nontrivial refinement, e.g., when you have a binding site
    # x: Sym(u0 * 4) = y.item().  Here, the code generation must do a division
    # in order to appropriately bind u0.  This is communicated via the keypath
    # in unbacked_bindings, and we need to hold onto it in order to generate
    # code appropriately for this case.
    binding_sym, keypath = next(iter(unbacked_bindings.items()))
    buffer = ir.DynamicScalar(binding_sym, keypath, data)
    buffer.name = V.graph.register_buffer(buffer)
    V.graph.register_operation(buffer)
    # NB: the replaced expr is OK to use directly downstream, we want
    # simplifications in this case!
    val = V.graph.current_node.meta["val"]
    if isinstance(val, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return val.node.expr
    else:
        return sympy.sympify(val)


@register_lowering(aten._assert_scalar)
def _assert_scalar(data, msg):
    # [FUNCTION DOC] Lowering for aten._assert_scalar: no-op returning None; runtime asserts are generated elsewhere at codegen time.
    # NB: These will be handled at codegen time
    # Not sure if we are guaranteed to be able to serve out truth from the
    # deferred_runtime_asserts, TODO: try this assert out
    # See [NOTE] Codegen runtime asserts in Inductor
    # assert bool(data.scalar), data
    return None


@register_lowering(aten._assert_tensor_metadata)
def _assert_tensor_metadata(
    a, size=None, stride=None, dtype=None, *, device=None, layout=None
):
    # [FUNCTION DOC] Lowering for aten._assert_tensor_metadata: no-op returning None.
    return None


def _full(fill_value, device, dtype, size):
    # [FUNCTION DOC] HELPER: Build a Pointwise of the given size and dtype filled with fill_value, which may be a python scalar, sympy expression, a constant IR value, or a 0-d IR tensor
    # (its single value is broadcast). Raises an error if a tensor fill value is not 0-d.
    value = fill_value
    if not isinstance(fill_value, (int, float)) and hasattr(value, "value"):
        value = value.value

    if isinstance(value, (int, float)):

        def inner_fn(index):
            # [FUNCTION DOC] HELPER: Return the constant value.
            return ops.constant(value, dtype)

    elif isinstance(value, sympy.Basic):

        def inner_fn(index):
            # [FUNCTION DOC] HELPER: Return the sympy expression value as an index expression.
            return ops.index_expr(value, dtype)

    else:
        if len(value.get_size()) != 0:
            raise AssertionError("expected: len(value.get_size()) == 0")
        value_loader = value.make_loader()

        def inner_fn(index):
            # [FUNCTION DOC] HELPER: Return the 0-d tensor value loaded without indices.
            return value_loader([])

    return Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=list(size),
    )


def full_like(x, fill_value, **kwargs):
    # [FUNCTION DOC] HELPER: Implement full_like: create a tensor like x filled with fill_value, forwarding keyword arguments.
    return create_tensor_like(tensor_constructor(fill_value))(x, **kwargs)


def tensor_constructor(fill_value):
    # [FUNCTION DOC] HELPER: Build a creator for tensors filled with fill_value (zeros, ones, full), taking sizes and tensor options.
    # torch.zeros, torch.ones, etc
    def inner(
        *size,
        dtype=None,
        device=None,
        layout=None,
        pin_memory=False,
        memory_format=None,
    ):
        # [FUNCTION DOC] HELPER: Create a filled tensor. Only strided layout and no memory_format are supported (NotImplementedError otherwise). Sizes may be given as varargs or one sequence;
        # SymInt sizes raise an error (sizes must already be sympy). Result is pinned if requested.
        assert_nyi(layout in (None, torch.strided), f"layout={layout}")
        assert_nyi(not memory_format, "memory_format")
        device = decode_device(device)
        dtype = dtype or torch.get_default_dtype()
        if len(size) == 1 and isinstance(size[0], (list, tuple, torch.Size)):
            size = tuple(size[0])
        # See https://github.com/pytorch/pytorch/issues/118102
        # All sizes at lowering time should be sympy.Symbol, not SymInt!
        for s in size:
            if isinstance(s, torch.SymInt):
                raise AssertionError("expected: not isinstance(s, torch.SymInt)")
        size = [sympy.expand(s) for s in size]
        full_pointwise = _full(fill_value, decode_device(device), dtype, size)

        if pin_memory:
            _realize_as_pinned(full_pointwise)

        return full_pointwise

    return inner


@register_lowering([torch.empty, aten.empty])
def empty(
    *size,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    # [FUNCTION DOC] Lowering for torch.empty/aten.empty: normalizes the size argument and delegates to empty_strided with no strides.
    device = decode_device(device)
    if len(size) == 1 and isinstance(size[0], (list, tuple, torch.Size)):
        size = tuple(size[0])
    return empty_strided(
        size, None, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


def create_tensor_like(creation_fn):
    # [FUNCTION DOC] HELPER: Turn a creation function into its X_like variant (for example zeros_like built from zeros) that copies size, dtype and device from a reference tensor.
    """
    Shim to convert X_like(...) into X(...).  For example zeros_like() into zeros().
    """

    def _constant_like(
        x, *, dtype=None, device=None, layout=None, pin_memory=False, memory_format=None
    ):
        # [FUNCTION DOC] HELPER: Create a tensor shaped like x via creation_fn; dtype defaults to x's, device to x's. pin_memory and non-strided layouts are not supported (NotImplementedError).
        assert_nyi(not pin_memory, "pin_memory")
        assert_nyi(layout in (None, torch.strided), f"layout={layout}")
        if dtype is None:
            dtype = x.get_dtype()
        else:
            dtype = decode_dtype(dtype)
        device = device or x.get_device()
        size = list(x.get_size())
        return creation_fn(
            size, dtype=dtype, device=device, layout=layout, pin_memory=pin_memory
        )

    return _constant_like


def constant_like(fill_value):
    # [FUNCTION DOC] HELPER: Build a like-creator filling with fill_value.
    return create_tensor_like(tensor_constructor(fill_value))


empty_like = register_lowering(aten.empty_like)(create_tensor_like(empty))
ones_like = create_tensor_like(tensor_constructor(1))
zeros_like = create_tensor_like(tensor_constructor(0))


def new_constant(fill_value):
    # [FUNCTION DOC] HELPER: Build the new_zeros/new_ones/new_full style lowering for fill_value.
    def _new_constant(
        x, size, *, dtype=None, layout=None, device=None, pin_memory=None
    ):
        # [FUNCTION DOC] HELPER: Create a filled tensor of the given size list/tuple with dtype/device defaulting to x's. pin_memory and non-strided layouts are not supported.
        if not (isinstance(size, (list, tuple))):
            raise AssertionError("expected: isinstance(size, (list, tuple))")
        assert_nyi(not pin_memory, "pin_memory")
        assert_nyi(layout in (None, torch.strided), f"layout={layout}")
        # pyrefly: ignore [bad-argument-type]
        dtype = decode_dtype(dtype) or x.get_dtype()
        device = device or x.get_device()
        size = [sympy.Integer(s) for s in size]
        return _full(fill_value, decode_device(device), dtype, size)

    return _new_constant


@register_lowering(aten.new_empty)
def new_empty(x, size, *, dtype=None, layout=None, device=None, pin_memory=None):
    # [FUNCTION DOC] Lowering for aten.new_empty: empty_strided of the given size with dtype and device defaulting to x's.
    if dtype is None:
        dtype = x.get_dtype()
    if device is None:
        device = x.get_device()
    return empty_strided(
        size,
        None,
        dtype=dtype,
        layout=layout,
        device=decode_device(device),
        pin_memory=pin_memory,
    )


@register_lowering(aten.empty_strided)
def empty_strided(
    size, stride, *, dtype=None, layout=None, device=None, pin_memory=None
):
    # [FUNCTION DOC] Lowering for aten.empty_strided. Size must be list/tuple and stride list/tuple/None (else raises an error); only strided layout. Creates a realized zero-range computed buffer
    # (so it becomes a no-op scheduler node) and then gives it a fixed layout with the requested (or contiguous) strides and pinned flag. Returns the pointwise node.
    if not (isinstance(size, (list, tuple))):
        raise AssertionError("expected: isinstance(size, (list, tuple))")
    if not (isinstance(stride, (list, tuple, type(None)))):
        raise AssertionError("expected: isinstance(stride, (list, tuple, type(None)))")
    assert_nyi(layout in (None, torch.strided), f"layout={layout}")
    # pyrefly: ignore [bad-argument-type]
    dtype = decode_dtype(dtype) or torch.get_default_dtype()
    device = device or torch.tensor(0.0).device
    device = decode_device(device)
    pointwise = _full(fill_value=0, device=device, dtype=dtype, size=size)
    pointwise.realize()
    buffer = pointwise.data.data
    # explicitly set ranges to zeros in order to make a NopKernelSchedulerNode
    buffer.data = dataclasses.replace(buffer.data, ranges=[0] * len(size))
    if not (isinstance(buffer, ir.ComputedBuffer)):
        raise AssertionError("expected: isinstance(buffer, ir.ComputedBuffer)")
    size = [sympy.expand(s) for s in size]
    stride = (
        [sympy.expand(s) for s in stride]
        if stride
        else ir.FlexibleLayout.contiguous_strides(size)
    )
    buffer.layout = ir.FixedLayout(
        device=device,
        dtype=dtype,
        size=size,
        stride=stride,
        is_pinned=pin_memory or False,
    )
    return pointwise


@register_lowering(aten.new_empty_strided)
def new_empty_strided(
    x, size, stride, *, dtype=None, layout=None, device=None, pin_memory=None
):
    # [FUNCTION DOC] Lowering for aten.new_empty_strided: empty_strided with dtype and device defaulting to x's.
    if dtype is None:
        dtype = x.get_dtype()
    if device is None:
        device = x.get_device()
    return empty_strided(
        size,
        stride,
        dtype=dtype,
        layout=layout,
        device=decode_device(device),
        pin_memory=pin_memory,
    )


@register_lowering(prims.copy_strided.default)
def copy_strided(x, stride):
    # [FUNCTION DOC] Lowering for prims.copy_strided: derives the stride order from the strides (via size hints) and requires x to have that order.
    stride = V.graph.sizevars.guarding_hints_or_throw(stride)
    stride_order = sorted(range(len(stride)), key=stride.__getitem__)
    return ir.ExternKernel.require_stride_order(x, stride_order)


@register_lowering([torch.full, aten.full])
def full(size, fill_value, **kwargs):
    # [FUNCTION DOC] Lowering for torch/aten full: raises an error if dtype is missing (decomposition should supply it), else creates a tensor filled with fill_value.
    if kwargs.get("dtype") is None:
        raise AssertionError("dtype should be handled by decomposition")
    return tensor_constructor(fill_value)(size, **kwargs)


@register_lowering(aten._efficientzerotensor, type_promotion_kind=None)
def _efficientzerotensor(
    size, *, dtype=None, layout=None, device=None, pin_memory=False
):
    # [FUNCTION DOC] Lowering for aten._efficientzerotensor: creates a scalar zero constant (outside any dispatch modes) and returns it expanded to size. Only strided layout is supported.
    assert_nyi(layout in (None, torch.strided), f"layout={layout}")
    dtype = torch.get_default_dtype() if dtype is None else decode_dtype(dtype)
    with torch.utils._python_dispatch._disable_current_modes():
        scalar = torch.zeros((), dtype=dtype, device=decode_device(device))
    return expand(V.graph.add_tensor_constant(scalar), size)


@register_lowering(aten.gather, type_promotion_kind=None)
def gather(x, dim, index, sparse_grad=False):
    # [FUNCTION DOC] Lowering for aten.gather. Empty index yields a new empty tensor of the index shape. 0-d inputs are expanded to one element. Builds a Pointwise over the index shape
    # that replaces the coordinate on dim with the (bounds-checked, non-wrapping) index value and loads from x. sparse_grad is ignored. Raises an error if x is not a TensorBox.
    # sparse_grad doesn't affect forward computation,
    # and backward tracing is taken care of by AOT Autograd
    if not (isinstance(x, TensorBox)):
        raise AssertionError("expected: isinstance(x, TensorBox)")
    if index.get_numel() == 0:
        # Empty index case. Return an empty array with the same shape
        return new_empty(x, index.get_size())

    size = x.get_size()
    offset = len(size) == 0
    dim = _validate_dim(x, dim, offset)

    if offset:
        x = expand(x, [1])
        size = [1]

    x_loader = x.make_loader()
    index_loader = index.make_loader()

    def fn(idx):
        # [FUNCTION DOC] HELPER: Load x with the dim coordinate replaced by the gathered index.
        idx = list(idx)
        gather_idx = ops.indirect_indexing(index_loader(idx), size[dim], wrap_neg=False)
        if len(idx) == 0:
            idx = [gather_idx]
        else:
            idx[dim] = gather_idx
        return x_loader(idx)

    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=fn,
        ranges=index.get_size(),
    )


@register_lowering(aten.embedding, type_promotion_kind=None)
def embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    # [FUNCTION DOC] Lowering for aten.embedding. Sparse gradients use the eager fallback. Otherwise weight and indices must be TensorBoxes with integer indices (raises an error otherwise); builds a
    # Pointwise over indices shape plus embedding dims that looks up weight rows by index. padding_idx and scale_grad_by_freq do not affect the forward pass.
    # KNOWN ISSUE: the later check that raises an error if sparse is set is unreachable because sparse returns earlier.
    if sparse:
        return fallback_handler(aten.embedding.default)(
            weight, indices, padding_idx, scale_grad_by_freq, sparse
        )

    if sparse:
        raise AssertionError("expected: not sparse")
    if not (isinstance(weight, TensorBox)):
        raise AssertionError("expected: isinstance(weight, TensorBox)")
    if not (isinstance(indices, TensorBox)):
        raise AssertionError("expected: isinstance(indices, TensorBox)")
    if "int" not in str(indices.get_dtype()):
        raise AssertionError('expected: "int" in str(indices.get_dtype())')

    weight_loader = weight.make_loader()
    indices_loader = indices.make_loader()
    indices_ndim = len(indices.get_size())
    weight_size = weight.get_size()
    new_size = [*indices.get_size(), *weight_size[1:]]

    def fn(idx):
        # [FUNCTION DOC] HELPER: Look up the weight row selected by the index at the leading coordinates, keeping the embedding-dim coordinates.
        if len(idx) != len(new_size):
            raise AssertionError(f"{idx} != {new_size}")
        var_index = indices_loader(idx[:indices_ndim])
        weight_idx = [
            ops.indirect_indexing(var_index, weight_size[0], wrap_neg=False)
        ] + [*idx[indices_ndim:]]
        return weight_loader(weight_idx)

    return Pointwise.create(
        device=weight.get_device(),
        dtype=weight.get_dtype(),
        inner_fn=fn,
        ranges=new_size,
    )


def check_and_broadcast_indices(indices, device):
    # [FUNCTION DOC] HELPER: Validate and broadcast index tensors for advanced indexing. Raises an error if any index is not an integer or bool/uint8 type, NotImplementedError for bool/uint8 indices
    # or indices on another device; requires at least one non-None index. Returns (indices with non-None entries broadcast together, positions of the non-None entries).
    if not (
        all(
            i.get_dtype() in (torch.int64, torch.int32, torch.bool, torch.uint8)
            for i in indices
            if i is not None
        )
    ):
        raise AssertionError(
            f"indices must be int64, byte or bool. Got {[i.get_dtype() for i in indices if i is not None]}"
        )
    if any(
        i.get_dtype() in (torch.bool, torch.uint8) for i in indices if i is not None
    ):
        raise NotImplementedError("Fallback for bool indices")

    valid_idxs = [i for i, x in enumerate(indices) if isinstance(x, TensorBox)]
    if len(valid_idxs) <= 0:
        raise AssertionError("requires at least 1 non-None index")
    new_indices = [None] * len(indices)
    for i, x in zip(valid_idxs, broadcast_tensors(*[indices[i] for i in valid_idxs])):
        # Eager allows indices to be CPU tensor when running on CUDA
        # FIXME: Calling to_device(x, device) should work but
        # test_advancedindex_mixed_cpu_devices still fails
        if x.get_device() != device:
            raise NotImplementedError("Fallback when indices is on a different device")
        new_indices[i] = x
    return new_indices, valid_idxs


def index_output_size_and_inner_fn(
    x_size,
    indices,
    tensor_indices,
    tensor_size,
    indices_loaders,
    indexed_size,
    x_loader,
    check,
    wrap_neg=True,
):
    # [FUNCTION DOC] HELPER: Compute the output shape and index mapping function for advanced (tensor) indexing. Output dims are the non-indexed dims with the broadcast index shape inserted at
    # the first indexed position (or at the front if the tensor indices are non-consecutive, as in eager). Returns (output_size, fn) where fn maps an output index to the input index list (bounds-checked
    # indirect indexing for indexed dims, wrapping negatives if wrap_neg) and, if x_loader is given, loads from x.
    # Note that behavior of indexing differs when there are non consecutive
    # tensors. In this case, the tensor index is pulled to the beginning.
    #
    # Suppose a = torch.arange(3 * 4 * 5 * 6 * 7).view(3, 4, 5, 6, 7)
    #         x = torch.tensor[1,2]
    # Then, a[:,x,:,x,:] will have shape 2,3,5,7 as due to x,:,x then 2 will
    # be pulled to the front.
    non_consecutive_tensors = False
    for previous, current in itertools.pairwise(tensor_indices):
        if current - previous != 1:
            non_consecutive_tensors = True

    output_size = [x_size[i] for i, val in enumerate(indices) if val is None]
    output_size = [*output_size, *x_size[len(output_size) + len(tensor_indices) :]]

    first_tensor_index = tensor_indices[0]
    if non_consecutive_tensors:
        output_size = tensor_size + output_size
    else:
        output_size = (
            output_size[:first_tensor_index]
            + tensor_size
            + output_size[first_tensor_index:]
        )

    def fn(idx):
        # [FUNCTION DOC] HELPER: Map an output index to an input index: unindexed dims copy the matching output coordinates, indexed dims use their index tensor loaders on the broadcast coordinates.
        # Raises an error on inconsistent lengths or missing loaders.
        if len(idx) != len(output_size):
            raise AssertionError("expected: len(idx) == len(output_size)")
        if len(indices_loaders) != len(indexed_size):
            raise AssertionError("expected: len(indices_loaders) == len(indexed_size)")

        rank = len(tensor_size)
        new_index = []
        first_tensor_index = tensor_indices[0]
        start_offset = 0 if non_consecutive_tensors else first_tensor_index
        next_idx = 0
        for i in range(tensor_indices[-1] + 1):
            if i == start_offset:
                next_idx += rank
            if indices[i] is None:
                if next_idx >= len(idx):
                    raise AssertionError("expected: next_idx < len(idx)")
                new_index.append(idx[next_idx])
                next_idx += 1
            else:
                loader = indices_loaders[i]
                if loader is None:
                    raise AssertionError("expected: loader is not None")
                size = indexed_size[i]
                new_index.append(
                    ops.indirect_indexing(
                        loader(idx[start_offset : start_offset + rank]),
                        size,
                        check=check,
                        wrap_neg=wrap_neg,
                    )
                )
        new_index = [
            *new_index,
            *idx[next_idx:],
        ]
        return new_index if x_loader is None else x_loader(new_index)

    return output_size, fn


def index_impl(x, indices, check):
    # [FUNCTION DOC] HELPER: Build a Pointwise gather-by-indices node for x using index_impl_helper (check controls bounds checking).
    output_size, inner_fn, _ = index_impl_helper(x, indices, check)

    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=output_size,
    )


def index_impl_helper(x, indices, check, wrap_neg=True):
    # [FUNCTION DOC] HELPER: Prepare advanced indexing of x. Validates indices, broadcasts them, and returns (output_size, inner_fn loading x, index_inner_fn mapping output index to input index).
    # Raises an error if indices is not a list/tuple or has no tensor, and IndexError when indexing a zero-size dim with non-empty indices under check.
    if not (isinstance(indices, (list, tuple))):
        raise AssertionError("expected: isinstance(indices, (list, tuple))")
    x_loader = x.make_loader()
    indices, tensor_indices = check_and_broadcast_indices(indices, x.get_device())
    if len(tensor_indices) <= 0:
        raise AssertionError("Must have at least one valid idx")

    indices_loaders = [i.make_loader() if i is not None else None for i in indices]
    # no guards on output size, all the guards are set in broadcast_tensors

    # We can use the first one since they are all required to be the same size
    tensor_size = list(indices[tensor_indices[0]].get_size())

    x_size = x.get_size()

    indexed_size = [x_size[i] for i in range(len(indices)) if indices[i] is not None]
    if check and 0 in indexed_size and 0 not in tensor_size:
        raise IndexError("index is out of bounds for dimension with size 0")

    indexed_size = [x_size[i] for i in range(len(indices))]
    output_size, index_inner_fn = index_output_size_and_inner_fn(
        x_size,
        indices,
        tensor_indices,
        tensor_size,
        indices_loaders,
        indexed_size,
        None,
        check=check,
        wrap_neg=wrap_neg,
    )

    def inner_fn(idx):
        # [FUNCTION DOC] HELPER: Load x at the input index computed from the output index.
        return x_loader(index_inner_fn(idx))

    return output_size, inner_fn, index_inner_fn


@register_lowering(aten.index, type_promotion_kind=None)
def index(x, indices):
    # [FUNCTION DOC] Lowering for aten.index with bounds checks. Falls back to the eager kernel (after realizing x) if the index cannot be lowered, such as boolean indices.
    try:
        return index_impl(x, indices, check=True)
    except NotImplementedError:
        # Fallback to ATen for boolean indexing
        x.realize()
        return fallback_handler(aten.index.Tensor, add_to_fallback_set=False)(
            x, indices
        )


@register_lowering(aten._unsafe_index, type_promotion_kind=None)
def _unsafe_index(x, indices):
    # [FUNCTION DOC] Lowering for aten._unsafe_index: indexing without bounds checks.
    return index_impl(x, indices, check=False)


# All the indexing decompositions are written in terms of index, index_put, and index_put_
# We cannot have this lowering as a decomposition as it introduces
# mutation in the graph, which is bad for Aot Autograd. Aot Autograd runs dead
# code elimination and common subexpression elimination optimizations, which
# assume graphs to be side-effect free. More details at
# https://github.com/pytorch/torchdynamo/issues/1235
# and
# https://github.com/pytorch/torchdynamo/issues/1863
@register_lowering(aten.index_put, type_promotion_kind=None)
def index_put(x, indices, values, accumulate=False):
    # [FUNCTION DOC] Lowering for aten.index_put (out of place): index_put_ on a clone, with checks, without realizing values. Kept as a lowering because it introduces mutation.
    return index_put_impl_(
        clone(x), indices, values, accumulate, check=True, may_realize=False
    )


@register_lowering(aten._unsafe_index_put)
def _unsafe_index_put(x, indices, values, accumulate=False):
    # [FUNCTION DOC] Lowering for aten._unsafe_index_put: clone then index_put_ without bounds checks.
    return index_put_impl_(
        clone(x), indices, values, accumulate, check=False, may_realize=False
    )


def index_put_as_masked_fill(self, indices, value, accumulate):
    # [FUNCTION DOC] HELPER: Implement index_put with a single boolean mask and single value as a masked fill: moves value to self's device, adds self if accumulating, and mutates self to where(mask, value, self).
    if value.get_device() != self.get_device():
        value = to_device(value, self.get_device())
    if accumulate:
        value = add(self, value)
    return mutate_to(self, where(indices[0], value, self))


def index_put_fallback(self, indices, values, accumulate):
    # [FUNCTION DOC] HELPER: Lower index_put_ to the eager IndexPutFallback kernel and return self. If the node uses boolean indexing that is CUDA-graph-unsafe (and graph partitioning is off),
    # records a reason on the graph disabling cudagraphs, including the stack trace if available.
    from .utils import _fx_node_is_input_dependent_cudagraph_unsafe

    op_overload = getattr(aten.index_put_, V.graph.current_node.target._overloadname)  # type: ignore[union-attr]

    # Check if any index is a boolean tensor - if so, mark as cudagraph-unsafe
    # because boolean indices trigger .nonzero() during CUDA graph capture
    # When graph_partition is enabled, skip - partitioning handles this
    fx_node = V.graph.current_node
    if (
        not config.graph_partition
        and fx_node is not None
        and _fx_node_is_input_dependent_cudagraph_unsafe(fx_node)
    ):
        msg = "index_put_ fallback with boolean indexing is not compatible with CUDA graphs"
        if stack_trace := fx_node.meta.get("stack_trace", None):
            msg = f"{msg} Found from : \n {stack_trace}"
        V.graph.disable_cudagraphs_reason = msg

    ir.IndexPutFallback(op_overload, self, indices, values, accumulate)
    return self


@register_lowering(aten.index_put_, type_promotion_kind=None)
def index_put_(self, indices, values, accumulate=False):
    # [FUNCTION DOC] Lowering for aten.index_put_ with bounds checks; may realize values to avoid overlap hazards.
    return index_put_impl_(
        self, indices, values, accumulate, check=True, may_realize=True
    )


@register_lowering(inductor_prims._unsafe_index_put_, type_promotion_kind=None)
def _unsafe_index_put_(self, indices, values, accumulate=False):
    # [FUNCTION DOC] Lowering for the unsafe in-place index_put_ (no bounds checks).
    return index_put_impl_(
        self, indices, values, accumulate, check=False, may_realize=True
    )


def index_put_impl_(self, indices, values, accumulate, check, may_realize=False):
    # [FUNCTION DOC] HELPER: Core in-place index_put. Optionally realizes values if they overlap self (unless indices come from randperm). Uses a masked fill for single boolean index with a single value,
    # the eager fallback for deterministic mode, boolean indices, atomic-add-limited GPU dtypes, or unsupported indices. Otherwise broadcasts indices, expands values to the indexed shape,
    # and registers a Scatter (atomic add if accumulating) as a mutation of self. 0-d inputs are viewed as one element and restored. Returns self.
    if may_realize:

        def indice_slice_from_randperm(indice):
            # [FUNCTION DOC] HELPER: True if the index tensor is a view of an extern randperm kernel result, which guarantees unique indices.
            # Refer to: https://github.com/pytorch/pytorch/pull/139366#discussion_r1825424660
            # For this specific pattern, indices is unique as coming from torch.randperm.
            # However, as the content of the indices is unknown, we have to check this specific pattern.
            if isinstance(indice, TensorBox) and isinstance(indice.data, ir.BaseView):
                indice = indice.data.unwrap_view()
                return (
                    isinstance(indice, ir.StorageBox)
                    and isinstance(indice.data, ir.ExternKernel)
                    and getattr(indice.data, "fx_node", None)
                    and indice.data.fx_node.target is torch.ops.aten.randperm.default
                )
            return False

        if ir.try_get_name(self) in values.get_read_names() and not all(
            indice_slice_from_randperm(indice) for indice in indices
        ):
            # Fix issue: https://github.com/pytorch/pytorch/issues/138908
            # When self and values have memory overlapping, indices may
            # contain duplicate values, potentially causing incorrect results since
            # the load of `values` might contain modified value from the store of `self`.
            # To address this, store values in a temporary buffer in such cases.
            values.realize()

    # Dispatch to masked fill for single boolean index with single value
    if (
        values.get_numel() == 1
        and len(indices) == 1
        and indices[0].get_dtype() in (torch.bool, torch.uint8)
    ):
        mask = indices[0]
        for _ in range(len(mask.get_size()), len(self.get_size())):
            mask = unsqueeze(mask, -1)
        return index_put_as_masked_fill(self, [mask], values, accumulate)

    # Fallback in torch deterministic mode
    if torch.are_deterministic_algorithms_enabled():
        return index_put_fallback(self, indices, values, accumulate)

    # Fallback if there is a boolean index
    for index in indices:
        if index is not None and index.get_dtype() in (torch.bool, torch.uint8):
            return index_put_fallback(self, indices, values, accumulate)

    x_size = self.get_size()
    x_ndim = len(x_size)

    device = self.get_device()
    if (
        accumulate
        and device is not None
        and is_gpu(device.type)
        and needs_fallback_due_to_atomic_add_limitations(self.get_dtype())
    ):
        # self is an scalar Tensor
        if x_ndim == 0:
            self = view(self, [1])
        self = index_put_fallback(self, indices, values, accumulate)
        if x_ndim == 0:
            self = view(self, [])
        return self

    values = to_dtype(values, self.get_dtype())

    try:
        indices, tensor_indices = check_and_broadcast_indices(
            indices, self.get_device()
        )
    except NotImplementedError:
        return index_put_fallback(self, indices, values, accumulate)

    indices_loaders = [i.make_loader() if i is not None else None for i in indices]

    if not (isinstance(self, TensorBox)):
        raise AssertionError("expected: isinstance(self, TensorBox)")
    self.realize()

    # self is an scalar Tensor
    if x_ndim == 0:
        self = view(self, [1])

    # We can use the first one since they are all required to be the same size
    tensor_size = list(indices[tensor_indices[0]].get_size())
    indexed_size = [x_size[i] for i in range(len(indices))]

    expected_vals_size, inner_fn = index_output_size_and_inner_fn(
        x_size,
        indices,
        tensor_indices,
        tensor_size,
        indices_loaders,
        indexed_size,
        None,
        check=check,
    )
    values = expand(values, expected_vals_size)
    # all guards are set above during broadcast_tensors and expand

    device = self.get_device()
    if device is None:
        raise AssertionError("expected: device is not None")
    scatter = ir.Scatter(
        device=device,
        dtype=self.get_dtype(),
        inner_fn=values.make_loader(),
        ranges=expected_vals_size,  # iter_ranges,
        output_indexer=inner_fn,
        scatter_mode="atomic_add" if accumulate else None,
    )
    buffer = ir.ComputedBuffer(
        name=None,
        layout=ir.MutationLayoutSHOULDREMOVE(self),
        data=scatter,
    )
    buffer.name = V.graph.register_buffer(buffer)
    V.graph.register_operation(buffer)

    if x_ndim == 0:
        self = view(self, [])
    return self


fallback__unsafe_masked_index = fallback_handler(
    aten._unsafe_masked_index.default, add_to_fallback_set=False
)

fallback__unsafe_masked_index_put_accumulate = fallback_handler(
    aten._unsafe_masked_index_put_accumulate.default, add_to_fallback_set=False
)


@register_lowering(aten._unsafe_masked_index, type_promotion_kind=None)
def _unsafe_masked_index(self, mask, indices, fill):
    # [FUNCTION DOC] Lowering for aten._unsafe_masked_index: gathers self at indices (no bounds check, no negative wrapping) where the mask is true and uses fill elsewhere, as a Pointwise.
    ranges, _, _unsafe_index_fn = index_impl_helper(
        self, indices, check=False, wrap_neg=False
    )
    mask_loader = mask.make_loader()
    self_loader = self.make_loader()

    def inner_fn(idx):
        # [FUNCTION DOC] HELPER: Convert the mask to bool and load the indexed value under it, else fill.
        if mask.dtype != torch.bool:
            mask_val = ops.to_dtype(mask_loader(idx), torch.bool)
        else:
            mask_val = mask_loader(idx)
        return ops.masked(mask_val, lambda: self_loader(_unsafe_index_fn(idx)), fill)

    return Pointwise.create(
        device=self.get_device(),
        dtype=self.get_dtype(),
        inner_fn=inner_fn,
        ranges=ranges,
    )


@register_lowering(aten._unsafe_masked_index_put_accumulate, type_promotion_kind=None)
def _unsafe_masked_index_put_accumulate(x, mask, indices, values):
    # [FUNCTION DOC] Lowering for aten._unsafe_masked_index_put_accumulate: zeroes masked-out values, clamps indices into range, and accumulates with an unsafe index_put on a clone of x.
    masked_value = where(mask, values, 0)
    shape = x.get_size()
    clamped_indices = [
        clamp(indices[i], -shape[i], shape[i] - 1) if indices[i] else None
        for i in range(len(indices))
    ]
    # TODO: use a masked store for this. currently only triton
    # supports masked stores and cpp backend does not.
    return _unsafe_index_put(x, clamped_indices, masked_value, accumulate=True)


@make_pointwise
def clamp(a, min, max):
    # [FUNCTION DOC] Pointwise clamp lowering: elementwise max(min, min(max, a)).
    return ops.maximum(min, ops.minimum(max, a))


@register_lowering(aten.as_strided_scatter, type_promotion_kind=None)
def as_strided_scatter(self, src, size, stride, storage_offset=None):
    # [FUNCTION DOC] Lowering for aten.as_strided_scatter: clones self, copies src into a strided view of the clone, and returns the clone.
    output = clone(self)
    output_view = as_strided(output, size, stride, storage_offset)
    copy_(output_view, src)
    return output


@register_lowering(aten.scatter, type_promotion_kind=None)
def scatter(x, dim: int, index, src, **kwargs):
    # [FUNCTION DOC] Lowering for aten.scatter: scatter_ applied to a clone of x.
    return scatter_(clone(x), dim, index, src, **kwargs)


def scatter_fallback(
    op_overload: torch._ops.OpOverload,
    self,
    dim: int,
    index,
    src,
    *,
    reduce: str | None = None,
    include_self: bool = True,
):
    # [FUNCTION DOC] HELPER: If the backend policy requires it for this op, reduce mode and dtypes, emit a ScatterFallback kernel and return self; otherwise return None.
    src_is_tensor = isinstance(src, TensorBox)
    if use_scatter_fallback(
        op_overload,
        reduce,
        self.get_dtype(),
        cast(torch.dtype, src.get_dtype() if src_is_tensor else type(src)),
        # pyrefly: ignore [missing-attribute]
        src.get_device().type if src_is_tensor else "not impl",
        src_is_tensor,
    ):
        ir.ScatterFallback(
            op_overload,
            self,
            dim,
            index,
            src,
            reduce=reduce,
            include_self=include_self,
        )
        return self

    return None


@register_lowering(aten.scatter_, type_promotion_kind=None)
def scatter_(self, dim: int, index, src, *, reduce: str | None = None):
    # [FUNCTION DOC] Lowering for in-place scatter_. reduce must be None, add or multiply (else raises an error). With no reduce tries the eager fallback; otherwise maps add to sum and multiply to prod
    # and delegates to scatter_reduce_.
    if reduce not in (None, "add", "multiply"):
        raise AssertionError('expected: reduce in (None, "add", "multiply")')
    if reduce is None:
        op_overload = getattr(aten.scatter_, V.graph.current_node.target._overloadname)  # type: ignore[union-attr]
        fallback_result = scatter_fallback(
            op_overload, self, dim, index, src, reduce=reduce
        )
        if fallback_result is not None:
            return fallback_result

    if reduce == "add":
        reduce = "sum"
    elif reduce == "multiply":
        reduce = "prod"
    return scatter_reduce_(self, dim, index, src, reduce)


@register_lowering(aten.scatter_add, type_promotion_kind=None)
def scatter_add(x, dim: int, index, src):
    # [FUNCTION DOC] Lowering for aten.scatter_add: scatter_add_ on a clone.
    return scatter_add_(clone(x), dim, index, src)


@register_lowering(aten.scatter_add_, type_promotion_kind=None)
def scatter_add_(x, dim: int, index, src):
    # [FUNCTION DOC] Lowering for in-place scatter_add_: scatter_reduce_ with sum.
    return scatter_reduce_(x, dim, index, src, "sum")


@register_lowering(aten.scatter_reduce, type_promotion_kind=None)
def scatter_reduce(x, dim: int, index, src, reduction_type, **kwargs):
    # [FUNCTION DOC] Lowering for aten.scatter_reduce: scatter_reduce_ on a clone.
    return scatter_reduce_(clone(x), dim, index, src, reduction_type, **kwargs)


@register_lowering(aten.scatter_reduce_, type_promotion_kind=None)
def scatter_reduce_(self, dim: int, index, src, reduce, *, include_self: bool = True):
    # [FUNCTION DOC] Lowering for in-place scatter_reduce_. Validates reduce (raises an error for unsupported modes). Numbers become filled tensors; uses the eager fallback when required; otherwise
    # treats 0-d inputs as one element, returns early on empty index, requires an integer index, realizes self, and registers a Scatter mutation (atomic add for sum; other reductions are not supported
    # natively and trip an assertion). If include_self is false, first zeroes the target elements. Returns self.
    if reduce not in (None, "sum", "prod", "mean", "amax", "amin"):
        raise AssertionError(
            'expected: reduce in (None, "sum", "prod", "mean", "amax", "amin")'
        )
    if not (
        len(aten.scatter_reduce_.overloads()) == 1
        and "two" in aten.scatter_reduce_.overloads()
    ):
        raise AssertionError(
            "aten.scatter_reduce_.two is not the unique overload of aten.scatter_reduce_"
        )

    if isinstance(src, Number):
        src = full_like(self, src)

    fallback_result = scatter_fallback(
        aten.scatter_reduce_.two,
        self,
        dim,
        index,
        src,
        reduce=reduce,
        include_self=include_self,
    )

    if fallback_result:
        return fallback_result

    if not (isinstance(self, TensorBox)):
        raise AssertionError("expected: isinstance(self, TensorBox)")

    ndim = len(self.get_size())
    if ndim == 0:
        self = view(self, [1])

    if isinstance(src, TensorBox) and len(src.get_size()) == 0:
        src = view(src, [1])

    if isinstance(index, TensorBox) and len(index.get_size()) == 0:
        index = view(index, [1])

    if index.get_numel() == 0:
        return self

    if "int" not in str(index.get_dtype()):
        raise AssertionError('expected: "int" in str(index.get_dtype())')

    dim = _validate_dim(self, dim)

    self.realize()
    index_loader = index.make_loader()
    src_loader = src.make_loader() if isinstance(src, TensorBox) else None

    def output_indexer(idx):
        # [FUNCTION DOC] HELPER: Map an index position to the output position by replacing the dim coordinate with the (non-wrapping) loaded index.
        # self is captured from the end of the function, so it may have 0 dim
        shape = self.get_size()
        ndim = len(shape)
        indirect_idx = list(idx)
        indirect_idx[dim] = ops.indirect_indexing(
            index_loader(idx), 1 if ndim == 0 else shape[dim], wrap_neg=False
        )
        return indirect_idx

    def fn(idx):
        # [FUNCTION DOC] HELPER: Value to scatter: loaded from src, or the scalar src constant.
        if src_loader:
            return src_loader(idx)
        else:
            # src is a scalar
            # pyrefly: ignore [bad-argument-type]
            return ops.constant(src, self.get_dtype())

    def backend_reduce_str(reduce):
        # [FUNCTION DOC] HELPER: Map reduce kind to the scatter mode: sum becomes atomic_add, None stays None, anything else raises an error.
        if reduce == "sum":
            return "atomic_add"
        else:
            # TODO: Need to support more reduction type
            if reduce is not None:
                raise AssertionError("expected: reduce is None")
            return None

    device = self.get_device()
    if device is None:
        raise AssertionError("expected: device is not None")

    if not include_self:
        # zero out the corresponding elements first
        zero_out = ir.Scatter(
            device=device,
            dtype=self.get_dtype(),
            inner_fn=lambda index: ops.constant(0, self.get_dtype()),
            ranges=index.get_size(),
            output_indexer=output_indexer,
            scatter_mode=None,
        )
        buffer = ir.ComputedBuffer(
            name=None,
            layout=ir.MutationLayoutSHOULDREMOVE(self),
            data=zero_out,
        )
        buffer.name = V.graph.register_buffer(buffer)
        V.graph.register_operation(buffer)

    # self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0
    # self[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1
    # self[i][j][index[i][j][k]] += src[i][j][k]  # if dim == 2
    scatter = ir.Scatter(
        device=device,
        dtype=self.get_dtype(),
        inner_fn=fn,
        ranges=index.get_size(),
        output_indexer=output_indexer,
        scatter_mode=backend_reduce_str(reduce),
    )
    buffer = ir.ComputedBuffer(
        name=None,
        layout=ir.MutationLayoutSHOULDREMOVE(self),
        data=scatter,
    )
    buffer.name = V.graph.register_buffer(buffer)
    V.graph.register_operation(buffer)

    if ndim == 0:
        self = view(self, [])
    return self


def upsample_nearestnd(
    x,
    output_size,
    scales_x: tuple[float | None, ...],
    n: int = 2,
    exact: bool = False,
):
    # [FUNCTION DOC] HELPER: Nearest-neighbor upsampling over the last n dims. Hints x for realization, computes inverse scales from input/output sizes (or provided scales), and builds a Pointwise
    # that maps each output coordinate to floor(scale * coordinate), with a 0.5 offset in exact mode, as a bounds-unchecked indirect index. Returns a TensorBox. Raises an error if scales_x has the wrong length.
    x.realize_hint()  # elements are reused
    x_loader = x.make_loader()
    i_sizes = x.get_size()[-n:]
    batch = x.get_size()[:-n]
    i_sizes = [V.graph.sizevars.guard_int(i) for i in i_sizes]

    if len(scales_x) != n:
        raise AssertionError("expected: len(scales_x) == n")
    o_sizes = output_size

    inv_scales = [i / o for i, o in zip(i_sizes, o_sizes)]
    for i, scale in enumerate(scales_x):
        if scale is not None:
            inv_scales[i] = 1.0 / scale

    def scale_fn(x, scale, size):
        # [FUNCTION DOC] HELPER: Map an output coordinate to an input index with the nearest or nearest-exact formula in float32.
        # Nearest Exact: input_index = round(scale * (output_index + 0.5) - 0.5)
        #                            = floor(scale * (output_index + 0.5))
        # Nearest: input_index = floor(scale * output_index)
        x = ops.index_expr(x, torch.float32)
        if exact:
            x = ops.add(x, ops.constant(0.5, torch.float32))
        x = ops.mul(x, ops.constant(scale, torch.float32))
        x = ops.to_dtype(x, torch.int32)
        return ops.indirect_indexing(x, size, check=False)

    def fn(idx):
        # [FUNCTION DOC] HELPER: Load x at batch coordinates and scaled spatial coordinates.
        x = idx[-n:]
        b = idx[:-n]
        return x_loader(
            [*b, *[scale_fn(i, s, size) for i, s, size in zip(x, inv_scales, i_sizes)]]
        )

    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=fn,
        ranges=[*batch, *o_sizes],
    )


@register_lowering(aten.upsample_nearest1d.default)
def upsample_nearest1d(x, output_size, scales: float | None = None):
    # [FUNCTION DOC] Lowering for aten.upsample_nearest1d: nearest upsampling over one spatial dim.
    return upsample_nearestnd(x, output_size, (scales,), n=1)


@register_lowering(aten._upsample_nearest_exact1d.default)
def _upsample_nearest_exact1d(x, output_size, scales: float | None = None):
    # [FUNCTION DOC] Lowering for aten._upsample_nearest_exact1d: exact-mode nearest upsampling over one spatial dim.
    return upsample_nearestnd(x, output_size, (scales,), n=1, exact=True)


@register_lowering(aten.upsample_nearest2d.default)
def upsample_nearest2d(
    x, output_size, scales_h: float | None = None, scales_w: float | None = None
):
    # [FUNCTION DOC] Lowering for aten.upsample_nearest2d: nearest upsampling over two spatial dims with optional scales.
    return upsample_nearestnd(x, output_size, (scales_h, scales_w), n=2)


@register_lowering(aten._upsample_nearest_exact2d.default)
def _upsample_nearest_exact2d(
    x, output_size, scales_h: float | None = None, scales_w: float | None = None
):
    # [FUNCTION DOC] Lowering for aten._upsample_nearest_exact2d: exact-mode nearest upsampling over two spatial dims.
    return upsample_nearestnd(x, output_size, (scales_h, scales_w), n=2, exact=True)


@register_lowering(aten.upsample_nearest3d.default)
def upsample_nearest3d(
    x,
    output_size,
    scales_d: float | None = None,
    scales_h: float | None = None,
    scales_w: float | None = None,
):
    # [FUNCTION DOC] Lowering for aten.upsample_nearest3d: nearest upsampling over three spatial dims with optional scales.
    return upsample_nearestnd(x, output_size, (scales_d, scales_h, scales_w), n=3)


@register_lowering(aten._upsample_nearest_exact3d.default)
def _upsample_nearest_exact3d(
    x,
    output_size,
    scales_d: float | None = None,
    scales_h: float | None = None,
    scales_w: float | None = None,
):
    # [FUNCTION DOC] Lowering for aten._upsample_nearest_exact3d: exact-mode nearest upsampling over three spatial dims.
    return upsample_nearestnd(
        x, output_size, (scales_d, scales_h, scales_w), n=3, exact=True
    )


def _create_constants(*args, dtype):
    # [FUNCTION DOC] HELPER: Return a tuple of ops constants of the given dtype, one per argument.
    return tuple(ops.constant(a, dtype) for a in args)


@register_lowering(prims.rev.default)
def rev(x, dims):
    # [FUNCTION DOC] Lowering for prims.rev: reverses x along the given (already canonical) dims via a Pointwise that mirrors those coordinates.
    # note - dims pre-canonicalized
    x_loader = x.make_loader()
    sizes = x.get_size()

    def loader(idx):
        # [FUNCTION DOC] HELPER: Mirror the index on each reversed dim and load from x; raises an error on rank mismatch.
        idx = list(idx)
        if len(idx) != len(sizes):
            raise AssertionError("expected: len(idx) == len(sizes)")
        for dim in dims:
            idx[dim] = (sizes[dim] - 1) - idx[dim]

        return x_loader(idx)

    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=loader,
        ranges=sizes,
    )


def inplace_constant_pad_nd(
    x: TensorBox, padding: Sequence[int], fill_value: float
) -> TensorBox | None:
    # [FUNCTION DOC] HELPER: Try to implement a 2-D right-pad of the last dim in place: if the padded width fits within the existing row stride, resize the buffer (via an as_strided view of the larger padded size),
    # fill the padding columns, record the padded size on the graph, and bump a counter. Returns the resized view, or None whenever it does not apply (padding fusable with the consumer, non-2-D, left/top/bottom padding,
    # unrealizable input, non-unit inner stride, or insufficient row stride). Realizes and freezes the layout of x as a side effect.
    """
    This optimization changes the semantics of padding from 'clone'
    style to 'view' style.

    Thanks to functionalization, this change can still maintain numerical
    correctness.
    """

    def _padding_can_be_fused():
        # [FUNCTION DOC] HELPER: Conservative check whether the pad could fuse into its consumer; False only when the single user is a matmul-like op (mm/addmm), where in-place padding saves bandwidth.
        """
        Conservatively check if padding can be fused with downstream op.
        1. if the downstream op is a sum, then there is little benefit to
           do inplace padding
        2. if the downstream op is a matmul, doing inplace padding can
           save membw.
        """
        current_node = V.graph.current_node
        if current_node is None:
            return True  # be conservative
        users = tuple(current_node.users)
        if len(users) == 1 and users[0].target in (
            aten.mm.default,
            aten.addmm.default,
        ):
            return False

        return True  # be conservative

    if _padding_can_be_fused():
        return None

    # Only handle 2D case for now
    if len(padding) != 4 or len(x.get_size()) != 2:
        return None

    # No harm to realize since we already know that
    # the op can not be fused into the single user.
    # It need to be realized later anyways.
    x.realize()

    # If x is a view (e.g. a SliceView), realizing it just realizing the
    # underlying storage. x itself is still a view.
    if (
        not isinstance(x, ir.TensorBox)
        or not isinstance(x.data, ir.StorageBox)
        or not (
            isinstance(x.data.data, ir.ComputedBuffer)
            or (
                config.can_inplace_pad_graph_input
                and isinstance(x.data.data, ir.InputBuffer)
            )
        )
        or not x.data.data.name
    ):
        return None
    x.freeze_layout()

    _, layout = ir.as_storage_and_layout(x)
    strides = layout.stride
    if strides[1] != 1:
        return None

    if padding[0] != 0 or padding[2] != 0 or padding[3] != 0:
        return None

    npad = padding[1]
    if npad == 0:
        return None

    stride0 = strides[0]
    rowsize = layout.size[1]

    if stride0 < rowsize + npad:
        return None

    bufname = x.data.data.name
    padded_size = [layout.size[0], layout.size[1] + npad]
    V.graph.buffer_to_padded_size[bufname] = padded_size
    resized_x = as_strided(
        x,
        padded_size,
        layout.stride,
        layout.offset,
        storage_offset_relative_to_input_storage=False,
    )

    sliced_x = slice_(resized_x, dim=1, start=rowsize, end=rowsize + npad, clamp=False)
    fill_(sliced_x, fill_value)

    counters["inductor"]["inplace_padding"] += 1
    return resized_x


def _pad_as_cat(
    x: TensorBox, padding: Sequence[int], fill_value: float
) -> TensorBox | None:
    # [FUNCTION DOC] HELPER: Rewrite a single-dimension non-negative right-pad as concatenation of x with a constant-filled tensor along that dim. Returns None for symbolic padding, left padding, trimming, multi-dim padding, no padding, or CPU.
    # Increments a counter on success. Returns the cat result.
    """Decompose right-pad into cat([x, fill], dim) and delegate to cat lowering.

    The cat lowering already has heuristics for choosing between pointwise_cat
    (fusion) and ConcatKernel (memory planning / zero-copy).  By routing through
    cat() we reuse those heuristics rather than duplicating them here.
    """
    # Bail out for symbolic padding, dynamic shapes
    if not all(isinstance(p, int) for p in padding):
        return None

    sizes = x.get_size()
    ndim = len(sizes)
    pad_pairs = list(zip(padding[::2], padding[1::2]))

    # Only support single-dimension right-pad
    pad_dim = None
    pad_amount = None
    for i, (left, right) in enumerate(pad_pairs):
        if left != 0:
            return None
        if right > 0:
            if pad_dim is not None:
                return None  # multi-dim pad
            pad_dim = ndim - 1 - i  # padding format is reversed dim order
            pad_amount = right
        elif right < 0:
            return None  # trim, not pad

    if pad_dim is None:
        return None

    # CPU cat always uses ConcatKernel (no pointwise_cat), which adds
    # extra kernel launches for the fill.  Skip pad-as-cat on CPU.
    device = x.get_device()
    if device is not None and device.type == "cpu":
        return None

    # Build the fill tensor for the padding region
    pad_shape = list(sizes)
    pad_shape[pad_dim] = pad_amount
    dtype = x.get_dtype()
    fill_value_typed = dtype_to_type(dtype)(fill_value)
    pad_tensor = tensor_constructor(fill_value_typed)(
        pad_shape, dtype=dtype, device=device
    )

    counters["inductor"]["pad_rewritten_as_cat"] += 1
    return cat([x, pad_tensor], pad_dim)


@register_lowering(aten.constant_pad_nd, type_promotion_kind=None)
def constant_pad_nd(x, padding, fill_value=0):
    # [FUNCTION DOC] Lowering for aten.constant_pad_nd. Padding length must be even (raises an error otherwise). All-zero padding is a clone. Tries in-place padding (when enabled), then pad-as-cat, and otherwise builds a Pointwise over the padded
    # shape that loads from x at the shifted index under a range mask and uses the fill value outside. Padding is given in reverse dim order as (low, high) pairs.
    if len(padding) % 2 != 0:
        raise AssertionError("expected: (len(padding) % 2) == 0")
    if all(p == 0 for p in padding):
        return clone(x)

    if config.inplace_padding:
        out = inplace_constant_pad_nd(x, padding, fill_value)
        if out:
            return out
            # fall through if can not inplace the padding

    out = _pad_as_cat(x, padding, fill_value)
    if out is not None:
        return out

    sizes = x.get_size()

    bounds = list(reversed(list(zip(padding[::2], padding[1::2]))))
    n = len(sizes) - len(bounds)

    # if padding is a complicated expression, hoist it
    bounds_precomp: list[tuple[sympy.Symbol, Any]] = []
    for l, h in bounds:
        bounds_precomp.append((V.graph.sizevars.lookup_precomputed_size(l), h))  # type: ignore[arg-type]

    output_size = list(sizes[:n])
    mask_sizes = []
    for (low, high), size in zip(bounds, sizes[n:]):
        mask_sizes.append(size)
        output_size.append(sympy.expand(size + low + high))
    if len(output_size) != len(sizes):
        raise AssertionError("expected: len(output_size) == len(sizes)")
    fill_value = dtype_to_type(x.get_dtype())(fill_value)

    def mask(index):
        # [FUNCTION DOC] HELPER: Combine low/high range masks over the padded dims and load x under that mask, else fill value.
        mask = []
        for idx, (low, high), length in zip(index[n:], bounds, mask_sizes):
            if low != 0:
                mask.append(range_mask_low(idx, 0))
            if high != 0:
                mask.append(range_mask_high(idx, length))
        mask = functools.reduce(ops.and_, mask)
        return ops.masked(mask, lambda: x_loader(index), fill_value)

    def offset_fn(index):
        # [FUNCTION DOC] HELPER: Shift the padded index by the low padding (precomputed) and evaluate the masked load.
        new_index = list(index[:n])
        for idx, (low, _high) in zip(index[n:], bounds_precomp):
            new_index.append(idx - low)
        if len(new_index) != len(index):
            raise AssertionError("expected: len(new_index) == len(index)")
        return mask(new_index)

    x_loader = x.make_loader()
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=offset_fn,
        ranges=output_size,
    )


def range_mask_low(i: sympy.Expr, low: sympy.Expr | int):
    # [FUNCTION DOC] HELPER: Mask that is true when index i is greater than or equal to low.
    return ops.ge(
        ops.index_expr(i, torch.int64),
        ops.index_expr(sympy.Integer(low), torch.int64),
    )


def range_mask_high(i: sympy.Expr, high: sympy.Expr):
    # [FUNCTION DOC] HELPER: Mask that is true when index i is less than high.
    return ops.lt(
        ops.index_expr(i, torch.int64),
        ops.index_expr(high, torch.int64),
    )


def range_mask(i: sympy.Expr, high: sympy.Expr, low: sympy.Expr):
    # [FUNCTION DOC] HELPER: Mask that is true when i lies within [low, high).
    return ops.and_(
        range_mask_low(i, low),
        range_mask_high(i, high),
    )


def constant_boundary_condition(
    x, fill_value, padding=None, pad_fill_value=1.0, dim=None
):
    # [FUNCTION DOC] HELPER: Build a loader over x that returns fill_value outside the spatial bounds of the last dim dims (extended by padding if given), and inside the padding region returns the value from a nested boundary loader with pad_fill_value;
    # inside the bounds loads x.
    h = x.get_size()[-dim:]
    x_loader = x.make_loader()
    # pyrefly: ignore [unsupported-operation]
    padding_h = padding or [0] * dim

    def load(index):
        # [FUNCTION DOC] HELPER: Apply the bounds mask to the index and load the proper value.
        prefix = index[:-dim]
        ih = index[-dim:]

        mask = functools.reduce(
            ops.and_,
            # pyrefly: ignore [bad-argument-type, no-matching-overload]
            [range_mask(ih[i], h[i] + padding_h[i], -padding_h[i]) for i in range(dim)],
        )
        return (
            ops.masked(
                mask,
                lambda: constant_boundary_condition(x, pad_fill_value, dim=dim)(
                    [*prefix, *ih]
                ),
                fill_value,
            )
            if padding
            else ops.masked(mask, lambda: x_loader([*prefix, *ih]), fill_value)
        )

    return load


def pooling_size(x, i, kernel_size, stride, padding, ceil_mode, *, dilation=None):
    # [FUNCTION DOC] HELPER: Compute pooling output length for spatial dim i from input size x and kernel, stride, padding and dilation. In ceil mode the alternate size is used if the last window starts within input/padding
    # and differs from floor mode. Returns (output size, effective ceil_mode flag; False when ceil mode has no effect).
    if dilation is None:
        dilation = [1] * len(padding)

    x_out = FloorDiv(
        x + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) + (stride[i] - 1),
        stride[i],
    )

    if ceil_mode:
        x_alt = FloorDiv(
            x
            + 2 * padding[i]
            - dilation[i] * (kernel_size[i] - 1)
            + 2 * (stride[i] - 1),
            stride[i],
        )
        if V.graph.sizevars.guard_or_false(
            sympy.Ge((x_alt - 1) * stride[i] - x - padding[i], 0)
        ):
            # Sliding windows must start within the input or left padding
            x_alt -= 1  # type: ignore[assignment]
        if V.graph.sizevars.guard_or_false(sympy.Eq(x_out, x_alt)):
            # ceil mode is actually a no-op, lets guard on that
            ceil_mode = False
        else:
            x_out = x_alt
    return x_out, ceil_mode


def should_fallback_max_pool_with_indices(kernel_size, *, n_dim):
    # [FUNCTION DOC] HELPER: True if the max-pool window (product of kernel sizes) exceeds 25 elements, in which case the eager fallback is preferred.
    kernel_size = pad_listlike(kernel_size, n_dim)
    window_size = functools.reduce(operator.mul, kernel_size)
    return window_size > 25


def max_pool_checks(
    x, kernel_size, stride, padding, dilation, n_dim, *, assert_fallback=None
):
    # [FUNCTION DOC] HELPER: Normalize max-pool parameters (scalar defaults expanded, stride defaulting to kernel size, list-like padded to n_dim) and validate ranks (raises an error on mismatch). Returns
    # (kernel_size, stride, padding, dilation, use_fallback). If assert_fallback is given, raises an error if the computed fallback decision differs.
    if padding == 0:
        padding = [0] * n_dim
    if dilation == 1:
        dilation = [1] * n_dim
    if not stride:
        stride = kernel_size

    kernel_size = pad_listlike(kernel_size, n_dim)
    stride = pad_listlike(stride, n_dim)
    padding = pad_listlike(padding, n_dim)
    dilation = pad_listlike(dilation, n_dim)

    if not (isinstance(x, TensorBox)):
        raise AssertionError("expected: isinstance(x, TensorBox)")
    if len(kernel_size) != n_dim:
        raise AssertionError("expected: len(kernel_size) == n_dim")
    if len(stride) != n_dim:
        raise AssertionError("expected: len(stride) == n_dim")
    if len(padding) != n_dim:
        raise AssertionError("expected: len(padding) == n_dim")
    if len(dilation) != n_dim:
        raise AssertionError("expected: len(dilation) == n_dim")
    if len(x.get_size()) not in (n_dim + 1, n_dim + 2):
        raise AssertionError("expected: len(x.get_size()) in (n_dim + 1, n_dim + 2)")

    use_fallback = should_fallback_max_pool_with_indices(kernel_size, n_dim=n_dim)
    if assert_fallback is not None:
        if use_fallback != assert_fallback:
            raise AssertionError("expected: use_fallback == assert_fallback")

    return kernel_size, stride, padding, dilation, use_fallback


def _max_pool_with_offsets(
    x,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
    *,
    n_dim,
):
    # [FUNCTION DOC] HELPER: Compute max pooling as two Reductions over the pooling window: max values and argmax window offsets (int64). Uses boundary masks with the dtype minimum when padding, ceil mode or dilation apply.
    # Realizes each result only if the reduction was not unrolled. Returns (values, offsets).
    x.realize_hint()
    batch = x.shape[:-n_dim]
    dhw = x.shape[-n_dim:]

    dhw_out, ceil_mode = zip(
        *[
            pooling_size(
                dhw[d], d, kernel_size, stride, padding, ceil_mode, dilation=dilation
            )
            for d in range(n_dim)
        ]
    )

    dtype = x.dtype
    min_value = (
        False
        if dtype is torch.bool
        else (float("-inf") if dtype.is_floating_point else torch.iinfo(dtype).min)
    )

    new_size = list(batch) + list(dhw_out)
    if any(padding) or any(ceil_mode) or any(d > 1 for d in dilation):
        x_loader = constant_boundary_condition(x, min_value, dim=n_dim)
    else:
        x_loader = x.make_loader()

    def fn_inner(idx, reduction_idx):
        # [FUNCTION DOC] HELPER: Map output coordinates plus window offsets to input coordinates (stride, dilation, padding) and load.
        prefix = idx[:-n_dim]
        bh = idx[-n_dim:]
        ih = [
            (bh[i] * stride[i]) + (reduction_idx[i] * dilation[i]) - padding[i]
            for i in range(n_dim)
        ]
        return x_loader([*prefix, *ih])

    result = Reduction.create(
        reduction_type="max",
        input_node=x,
        device=x.get_device(),
        dst_dtype=dtype,
        src_dtype=dtype,
        inner_fn=fn_inner,
        ranges=new_size,
        reduction_ranges=kernel_size,
    )
    offsets = Reduction.create(
        reduction_type="argmax",
        input_node=x,
        device=x.get_device(),
        dst_dtype=torch.int64,
        src_dtype=dtype,
        inner_fn=fn_inner,
        ranges=new_size,
        reduction_ranges=kernel_size,
    )
    if isinstance(result.data.data, Reduction):  # type: ignore[attr-defined, union-attr]
        # Only realize if reduction isn't unrolled
        result.realize()
    if isinstance(offsets.data.data, Reduction):  # type: ignore[attr-defined, union-attr]
        # Only realize if reduction isn't unrolled
        offsets.realize()

    return result, offsets


@register_lowering(prims._low_memory_max_pool_with_offsets, type_promotion_kind=None)
def _low_memory_max_pool_with_offsets(
    x,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode=False,
):
    # [FUNCTION DOC] Lowering for the low-memory max pool prim. Verifies via max_pool_checks that the non-fallback path applies, computes values and offsets with reduction unrolling up to 25 elements,
    # and returns (values, offsets cast to int8).
    n_dim = len(kernel_size)

    # assert we are not on a fallback path, the inductor decomp should have guaranteed this
    kernel_size, stride, padding, dilation, _ = max_pool_checks(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        n_dim,
        assert_fallback=False,
    )

    with config.patch(unroll_reductions_threshold=25):
        result, offsets = _max_pool_with_offsets(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            n_dim=n_dim,
        )
        return result, to_dtype(offsets, torch.int8)


def _pool_offsets_to_indices(
    offsets: TensorBox,
    kernel_size: Sequence[int | torch.SymInt],
    input_size: Sequence[int | torch.SymInt],
    increments_to_index: Callable[
        [Sequence[int | torch.SymInt], Sequence[int | torch.SymInt]],
        torch._inductor.virtualized.OpsValue,
    ],
) -> TensorBox:
    # [FUNCTION DOC] HELPER: Convert pooling window offsets (flat position within the window) to flattened input indices: unflatten the offset to window coordinates, map to input coordinates with increments_to_index, and flatten against the trailing input size. Returns an int64 Pointwise.
    n_dim = len(kernel_size)
    offsets_loader = offsets.make_loader()
    window_size = sympy.sympify(functools.reduce(operator.mul, kernel_size))

    def offsets_to_indices(idx):
        # [FUNCTION DOC] HELPER: Convert one loaded offset to a flattened input index.
        offset = offsets_loader(idx)
        offset_sympy = ops.indirect_indexing(offset, window_size)
        reduction_idx = inductor_prims._flattened_index_to_nd(offset_sympy, kernel_size)
        idhw = increments_to_index(idx, reduction_idx)
        return ops.index_expr(
            inductor_prims._flatten_index(idhw, input_size[-n_dim:]), torch.int64
        )

    indices = Pointwise.create(
        device=offsets.get_device(),
        dtype=torch.int64,
        inner_fn=offsets_to_indices,
        ranges=offsets.get_size(),
    )
    return indices


@register_lowering(
    prims._low_memory_max_pool_offsets_to_indices, type_promotion_kind=None
)
def _low_memory_max_pool_offsets_to_indices(
    offsets, kernel_size, input_size, stride, padding, dilation
):
    # [FUNCTION DOC] Lowering for the max-pool offsets-to-indices prim: computes input indices from window offsets using stride, padding and dilation.
    # TODO: Generalize to other max pooling flavors
    n_dim = len(kernel_size)

    def increments_to_index(idx, reduction_idx):
        # [FUNCTION DOC] HELPER: Input coordinates for output coordinates plus window offsets.
        bh = idx[-n_dim:]
        return [
            (bh[i] * stride[i]) + (reduction_idx[i] * dilation[i]) - padding[i]
            for i in range(n_dim)
        ]

    return _pool_offsets_to_indices(
        offsets, kernel_size, input_size, increments_to_index
    )


def _max_pool_with_indices(
    x,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
    n_dim,
):
    # [FUNCTION DOC] HELPER: Max pooling returning (values, indices): validates arguments, computes values and offsets, then converts offsets to flat input indices.
    kernel_size, stride, padding, dilation, _ = max_pool_checks(
        x, kernel_size, stride, padding, dilation, n_dim=n_dim
    )

    out, offsets = _max_pool_with_offsets(
        x, kernel_size, stride, padding, dilation, ceil_mode, n_dim=n_dim
    )

    indices = _low_memory_max_pool_offsets_to_indices(
        offsets,
        kernel_size,
        x.shape[-n_dim:],
        stride,
        padding,
        dilation,
    )

    return out, indices


# Fallback when we do not decompose to the low-memory path.
@register_lowering(aten.max_pool2d_with_indices, type_promotion_kind=None)
def max_pool2d_with_indices(
    x,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    # [FUNCTION DOC] Lowering for aten.max_pool2d_with_indices (used when the low-memory decomposition is not applied): 2-D max pool returning (values, indices).
    return _max_pool_with_indices(
        x, kernel_size, stride, padding, dilation, ceil_mode, n_dim=2
    )


# Fallback when we do not decompose to the low-memory path.
@register_lowering(aten.max_pool3d_with_indices, type_promotion_kind=None)
def max_pool3d_with_indices(
    x,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    # [FUNCTION DOC] Lowering for aten.max_pool3d_with_indices (used when the low-memory decomposition is not applied): 3-D max pool returning (values, indices).
    return _max_pool_with_indices(
        x, kernel_size, stride, padding, dilation, ceil_mode, n_dim=3
    )


fallback_max_pool2d_with_indices_backward = fallback_handler(
    aten.max_pool2d_with_indices_backward.default,
    add_to_fallback_set=False,
)


@register_lowering(aten.max_pool2d_with_indices_backward, type_promotion_kind=None)
def max_pool2d_with_indices_backward(
    grad_output, x, kernel_size, stride, padding, dilation, ceil_mode, indices
):
    # [FUNCTION DOC] Lowering for aten.max_pool2d_with_indices_backward. Uses the eager fallback on XPU, for dilation other than 1, or when the number of pooled windows covering an input element exceeds 25. Otherwise normalizes parameters,
    # validates ranks (raises an error), and builds a Pointwise over the input shape that, for each input element, scans the bounded set of covering output windows and sums grad_output where the stored index equals the element's flat position.
    # Result is forced channels-last if x or grad_output is channels-last.
    if x.get_device().type == "xpu":
        return fallback_max_pool2d_with_indices_backward(
            grad_output, x, kernel_size, stride, padding, dilation, ceil_mode, indices
        )

    if padding == 0:
        padding = [0, 0]
    if dilation == 1:
        dilation = [1, 1]
    if not stride:
        stride = kernel_size

    if not (isinstance(x, TensorBox)):
        raise AssertionError("expected: isinstance(x, TensorBox)")
    if len(kernel_size) != 2:
        raise AssertionError("expected: len(kernel_size) == 2")
    if len(stride) != 2:
        raise AssertionError("expected: len(stride) == 2")
    if len(padding) != 2:
        raise AssertionError("expected: len(padding) == 2")
    if len(dilation) != 2:
        raise AssertionError("expected: len(dilation) == 2")
    if len(x.get_size()) not in (3, 4):
        raise AssertionError("expected: len(x.get_size()) in (3, 4)")

    # we will read this many times, so make sure it is computed
    grad_output.realize_hint()
    gO_stride = grad_output.maybe_get_stride()
    x_stride: Sequence[Any] | None
    if isinstance(x, TensorBox) and isinstance(x.data.data, Pointwise):  # type: ignore[attr-defined]
        data = x.data.data  # type: ignore[attr-defined]
        device = data.get_device()
        if device is None:
            raise AssertionError("expected: device is not None")
        x_buffer = ir.ComputedBuffer(
            name=None,
            layout=ir.FlexibleLayout(
                device=device,
                dtype=data.get_dtype(),
                size=data.get_size(),
            ),
            data=data,
        )
        x_buffer.decide_layout()
        x_stride = x_buffer.get_stride()
    else:
        x_stride = x.maybe_get_stride()

    is_channels_last = (x_stride is not None and x_stride[1] == 1) or (
        gO_stride is not None and gO_stride[1] == 1
    )
    if any(d != 1 for d in dilation):
        # dilation NYI
        return fallback_max_pool2d_with_indices_backward(
            grad_output, x, kernel_size, stride, padding, dilation, ceil_mode, indices
        )

    *_batch, _height, width = x.get_size()
    *_, pooled_height, pooled_width = grad_output.get_size()

    indices_loader = indices.make_loader()
    grad_loader = grad_output.make_loader()
    new_size = list(x.get_size())

    h_window_size = max(
        max(FloorDiv(h, stride[0]) - max(0, FloorDiv(h - kernel_size[0], stride[0])), 1)
        for h in range(kernel_size[0] * 2)
    )
    w_window_size = max(
        max(FloorDiv(w, stride[1]) - max(0, FloorDiv(w - kernel_size[1], stride[1])), 1)
        for w in range(kernel_size[1] * 2)
    )

    window_size = h_window_size * w_window_size

    if window_size > 25:
        # Kernel size too big. Results in hard-to-optimize Triton code. Use fallback.
        return fallback_max_pool2d_with_indices_backward(
            grad_output, x, kernel_size, stride, padding, dilation, ceil_mode, indices
        )

    indices_size = indices.get_size()

    def fn(idx):
        # [FUNCTION DOC] HELPER: For one input position, loop over the fixed window grid of candidate output positions, clamp their indices, and accumulate gradient where the pooled index matches the position and the window is in range.
        *prefix, h, w = idx
        index_test = ops.index_expr(h * width + w, torch.int32)
        h = h + padding[0]
        w = w + padding[1]
        phstart = ops.index_expr(
            FloorDiv(h - kernel_size[0] + stride[0], stride[0]), torch.int32
        )
        pwstart = ops.index_expr(
            FloorDiv(w - kernel_size[1] + stride[1], stride[1]), torch.int32
        )
        phend = ops.index_expr(FloorDiv(h, stride[0]) + 1, torch.int32)
        pwend = ops.index_expr(FloorDiv(w, stride[1]) + 1, torch.int32)

        phstart = ops.maximum(phstart, ops.constant(0, torch.int32))
        pwstart = ops.maximum(pwstart, ops.constant(0, torch.int32))
        phend = ops.minimum(phend, ops.index_expr(pooled_height, torch.int32))
        pwend = ops.minimum(pwend, ops.index_expr(pooled_width, torch.int32))

        gradient = None
        for ph_ in range(h_window_size):
            for pw_ in range(w_window_size):
                ph = ops.add(phstart, ops.constant(ph_, torch.int32))
                pw = ops.add(pwstart, ops.constant(pw_, torch.int32))
                grad_index = [
                    *prefix,
                    ops.indirect_indexing(
                        ops.minimum(ph, ops.sub(phend, ops.constant(1, torch.int32))),
                        indices_size[-2],
                        check=False,
                    ),
                    ops.indirect_indexing(
                        ops.minimum(pw, ops.sub(pwend, ops.constant(1, torch.int32))),
                        indices_size[-1],
                        check=False,
                    ),
                ]

                index_actual = indices_loader(grad_index)
                grad_part = grad_loader(grad_index)
                check = ops.eq(index_actual, index_test)

                if gradient is None:
                    # don't need mask for 0, 0
                    gradient = ops.where(
                        check, grad_part, ops.constant(0.0, torch.float32)
                    )
                else:
                    mask = ops.and_(
                        ops.and_(
                            ops.lt(ph, phend),
                            ops.lt(pw, pwend),
                        ),
                        check,
                    )
                    gradient = ops.where(mask, ops.add(gradient, grad_part), gradient)
        if gradient is None:
            raise AssertionError("expected: gradient is not None")
        return gradient

    out = Pointwise.create(
        device=grad_output.get_device(),
        dtype=grad_output.get_dtype(),
        inner_fn=fn,
        ranges=new_size,
    )
    if is_channels_last:
        return ir.ExternKernel.require_channels_last(out)
    else:
        return out


def pad_adaptive_loader(x, pad_val=0.0):
    # [FUNCTION DOC] HELPER: Return a loader for adaptive pooling windows that yields x at start plus increment, or pad_val when that position is past the window end.
    x_loader = x.make_loader()

    def load(prefix, increments, start_indices, end_indices):
        # [FUNCTION DOC] HELPER: Masked load of one window element given the prefix index, increments and the window start and end indices per spatial dim.
        ih, iw = increments
        h_start_index, w_start_index = start_indices
        h_end_index, w_end_index = end_indices

        mask = ops.and_(
            ops.lt(
                ops.index_expr(h_start_index + ih, torch.int64),
                ops.index_expr(h_end_index, torch.int64),
            ),
            ops.lt(
                ops.index_expr(w_start_index + iw, torch.int64),
                ops.index_expr(w_end_index, torch.int64),
            ),
        )

        return ops.masked(
            mask,
            lambda: x_loader([*prefix, h_start_index + ih, w_start_index + iw]),
            pad_val,
        )

    return load


def compute_indices_adaptive_pooling(start_index, end_index, h_in, w_in, h_out, w_out):
    # [FUNCTION DOC] HELPER: Bind output and input dims to the start/end index functions for height and width; returns (h start fn, h end fn, w start fn, w end fn).
    h_start_index = functools.partial(start_index, out_dim=h_out, inp_dim=h_in)
    h_end_index = functools.partial(end_index, out_dim=h_out, inp_dim=h_in)

    w_start_index = functools.partial(start_index, out_dim=w_out, inp_dim=w_in)
    w_end_index = functools.partial(end_index, out_dim=w_out, inp_dim=w_in)

    return h_start_index, h_end_index, w_start_index, w_end_index


def _adaptive_pooling_fn(
    start_index, end_index, kernel_maxes, in_sizes, out_sizes, pooling_fn
):
    # [FUNCTION DOC] HELPER: Build an index function for 2-D adaptive pooling that, for an output position, loops over the maximum window size, loads each element via loader and combines them with pooling_fn (add, maximum). Returns that function of (idx, loader).
    h_in, w_in = in_sizes
    h_out, w_out = out_sizes

    (
        h_start_index_fn,
        h_end_index_fn,
        w_start_index_fn,
        w_end_index_fn,
    ) = compute_indices_adaptive_pooling(
        start_index, end_index, h_in, w_in, h_out, w_out
    )

    def fn(idx, loader):
        # [FUNCTION DOC] HELPER: Reduce all window elements for output position idx with the pooling function.
        *prefix, bh, bw = idx

        h_start_index = h_start_index_fn(bh)
        h_end_index = h_end_index_fn(bh)

        w_start_index = w_start_index_fn(bw)
        w_end_index = w_end_index_fn(bw)

        result = None
        for ih, iw in itertools.product(range(kernel_maxes[0]), range(kernel_maxes[1])):
            val = loader(
                prefix,
                [ih, iw],
                [h_start_index, w_start_index],
                [h_end_index, w_end_index],
            )
            if result is None:
                result = val
            else:
                result = pooling_fn(val, result)
        return result

    return fn


def _adaptive_pooling_fn_with_idx(
    start_index, end_index, kernel_maxes, in_sizes, out_sizes, pooling_fn
# [FUNCTION DOC] HELPER: Like the adaptive pooling function builder but tracks the flat input index of the maximum element and returns that index (int64) instead of the value.
):
    h_in, w_in = in_sizes
    h_out, w_out = out_sizes

    (
        h_start_index_fn,
        h_end_index_fn,
        w_start_index_fn,
        w_end_index_fn,
    ) = compute_indices_adaptive_pooling(
        start_index, end_index, h_in, w_in, h_out, w_out
    )

    def fn(idx, loader):
        # [FUNCTION DOC] HELPER: Return the flat index of the maximum element in the window for output position idx.
        *prefix, bh, bw = idx

        h_start_index = h_start_index_fn(bh)
        h_end_index = h_end_index_fn(bh)

        w_start_index = w_start_index_fn(bw)
        w_end_index = w_end_index_fn(bw)

        maxval = None
        maxindex = None
        for ih, iw in itertools.product(range(kernel_maxes[0]), range(kernel_maxes[1])):
            val = loader(
                prefix,
                [ih, iw],
                [h_start_index, w_start_index],
                [h_end_index, w_end_index],
            )

            index = ops.index_expr(
                (h_start_index + ih) * w_in + w_start_index + iw, torch.int64
            )

            if maxindex is None:
                maxindex = index
            else:
                maxindex = ops.where(ops.gt(val, maxval), index, maxindex)

            if maxval is None:
                maxval = val
            else:
                maxval = pooling_fn(val, maxval)

        return maxindex

    return fn


fallback_adaptive_avg_pool2d = fallback_handler(
    aten._adaptive_avg_pool2d.default, add_to_fallback_set=False
)


@register_lowering(aten._adaptive_avg_pool2d)
def _adaptive_avg_pool2d(x, output_size):
    # [FUNCTION DOC] Lowering for aten._adaptive_avg_pool2d. Requires a TensorBox and a 2-element output size (raises an error otherwise). Empty outputs give an empty tensor; int64 raises RuntimeError; equal sizes clone;
    # evenly divisible sizes use avg_pool2d; windows larger than 25 elements use the eager fallback; otherwise a Pointwise computes window sum divided by the count of in-range elements.
    if not (isinstance(x, TensorBox)):
        raise AssertionError("expected: isinstance(x, TensorBox)")
    if len(output_size) != 2:
        raise AssertionError("expected: len(output_size) == 2")
    x.realize_hint()

    *batch, h_in, w_in = x.get_size()

    h_in = V.graph.sizevars.guard_int(h_in)
    w_in = V.graph.sizevars.guard_int(w_in)

    h_out, w_out = output_size

    if h_out == 0 or w_out == 0:
        o_size = [*batch, h_out, w_out]
        return empty(o_size, dtype=x.get_dtype(), device=x.get_device())

    if x.get_dtype() == torch.int64:
        # not supported in eager
        raise RuntimeError("'adaptive_avg_pool2d' not implemented for 'Long'")

    # no-op if the same input and output
    if h_in == h_out and w_in == w_out:
        return clone(x)

    if h_in % h_out == 0 and w_in % w_out == 0:
        kernel_size = [FloorDiv(h_in, h_out), FloorDiv(w_in, w_out)]
        return avg_pool2d(x, kernel_size)

    h_kernel_max = ceildiv((h_in + h_out - 1), h_out)
    w_kernel_max = ceildiv((w_in + w_out - 1), w_out)

    new_size = list(batch) + [h_out, w_out]
    dtype = x.get_dtype()

    window_size = h_kernel_max * w_kernel_max
    if V.graph.sizevars.guard_or_true(sympy.Gt(window_size, 25)):
        # Kernel size too big. Results in hard-to-optimize Triton code. Use fallback.
        return fallback_adaptive_avg_pool2d(x, output_size)

    h_kernel_max = V.graph.sizevars.guard_int(h_kernel_max)
    w_kernel_max = V.graph.sizevars.guard_int(w_kernel_max)

    def start_index(index, out_dim, inp_dim):
        # [FUNCTION DOC] HELPER: Start of the adaptive window for an output index: floor(index * inp / out).
        return FloorDiv((index * inp_dim), out_dim)

    def end_index(index, out_dim, inp_dim):
        # [FUNCTION DOC] HELPER: End of the adaptive window for an output index: ceil((index + 1) * inp / out).
        return FloorDiv((index + 1) * inp_dim + out_dim - 1, out_dim)

    fn_sum = _adaptive_pooling_fn(
        start_index=start_index,
        end_index=end_index,
        kernel_maxes=[h_kernel_max, w_kernel_max],
        in_sizes=[h_in, w_in],
        out_sizes=[h_out, w_out],
        pooling_fn=ops.add,
    )

    ones_loader = pad_adaptive_loader(ones_like(x))

    def fn(idx):
        # [FUNCTION DOC] HELPER: Average = window sum over input divided by window sum over ones (valid element count).
        return ops.truediv(
            fn_sum(idx, pad_adaptive_loader(x)), fn_sum(idx, ones_loader)
        )

    rv = Pointwise.create(
        device=x.get_device(),
        dtype=dtype,
        inner_fn=fn,
        ranges=new_size,
    )
    # TODO: should we force these to be realized?
    return rv


fallback_adaptive_max_pool2d = fallback_handler(
    aten.adaptive_max_pool2d.default, add_to_fallback_set=False
)


@register_lowering(aten.adaptive_max_pool2d)
def adaptive_max_pool2d(x, output_size):
    # [FUNCTION DOC] Lowering for aten.adaptive_max_pool2d returning (values, indices). int64 raises RuntimeError; empty outputs return empty tensors; evenly divisible sizes raise ValueError (a decomposition handles them);
    # windows larger than 25 use the eager fallback; otherwise two Pointwise nodes compute the max value and flat argmax index over adaptive windows with negative-infinity padding.
    if x.get_dtype() == torch.int64:
        # not supported in eager
        raise RuntimeError("adaptive_max_pool2d not implemented for Long")
    if not (isinstance(x, TensorBox)):
        raise AssertionError("expected: isinstance(x, TensorBox)")
    if len(output_size) != 2:
        raise AssertionError("expected: len(output_size) == 2")
    x.realize_hint()

    *batch, h_in, w_in = x.get_size()

    h_in = V.graph.sizevars.guard_int(h_in)
    w_in = V.graph.sizevars.guard_int(w_in)

    h_out, w_out = output_size

    if h_out == 0 or w_out == 0:
        o_size = [*batch, h_out, w_out]
        return empty(o_size, dtype=x.get_dtype(), device=x.get_device()), empty(
            o_size, dtype=torch.int64, device=x.get_device()
        )

    if h_in % h_out == 0 and w_in % w_out == 0:
        # This is handled by a decomposition
        raise ValueError

    h_kernel_max = ceildiv((h_in + h_out - 1), h_out)
    w_kernel_max = ceildiv((w_in + w_out - 1), w_out)

    new_size = list(batch) + [h_out, w_out]
    dtype = x.get_dtype()

    window_size = h_kernel_max * w_kernel_max
    if window_size > 25:
        # Kernel size too big. Results in hard-to-optimize Triton code. Use fallback.
        return fallback_adaptive_max_pool2d(x, output_size)

    def start_index(index, out_dim, inp_dim):
        # [FUNCTION DOC] HELPER: Start of the adaptive window: floor(index * inp / out).
        return FloorDiv((index * inp_dim), out_dim)

    def end_index(index, out_dim, inp_dim):
        # [FUNCTION DOC] HELPER: End of the adaptive window: ceil((index + 1) * inp / out).
        return FloorDiv((index + 1) * inp_dim + out_dim - 1, out_dim)

    inner_func_max_val = _adaptive_pooling_fn(
        start_index=start_index,
        end_index=end_index,
        kernel_maxes=[h_kernel_max, w_kernel_max],
        in_sizes=[h_in, w_in],
        out_sizes=[h_out, w_out],
        pooling_fn=ops.maximum,
    )

    inner_func_max_idx = _adaptive_pooling_fn_with_idx(
        start_index=start_index,
        end_index=end_index,
        kernel_maxes=[h_kernel_max, w_kernel_max],
        in_sizes=[h_in, w_in],
        out_sizes=[h_out, w_out],
        pooling_fn=ops.maximum,
    )

    def inner_fn_max_val(idx):
        # [FUNCTION DOC] HELPER: Max value over the adaptive window at idx.
        return inner_func_max_val(idx, pad_adaptive_loader(x, float("-inf")))

    def inner_fn_max_idx(idx):
        # [FUNCTION DOC] HELPER: Flat index of the max element over the adaptive window at idx.
        return inner_func_max_idx(idx, pad_adaptive_loader(x, float("-inf")))

    rv = Pointwise.create(
        device=x.get_device(),
        dtype=dtype,
        inner_fn=inner_fn_max_val,
        ranges=new_size,
    )
    ri = Pointwise.create(
        device=x.get_device(),
        dtype=torch.int64,
        inner_fn=inner_fn_max_idx,
        ranges=new_size,
    )
    return rv, ri


def _fractional_pooling_offsets(samples, in_sz, out_sz, kernel_sz, dim, ndims):
    # [FUNCTION DOC] HELPER: Build a loader giving, for output position i along spatial dim, the start offset of the fractional pooling window computed from the random samples:
    # trunc((i + sample) * alpha) - trunc(sample * alpha) with alpha = (in - kernel) / (out - 1), and the last output position pinned at in - kernel. Indexes the samples tensor by batch and channel from the prefix.
    out_sz = out_sz[dim]
    in_sz = in_sz[dim]
    kernel_sz = kernel_sz[dim]
    samples_loader = samples.make_loader()

    def load(prefix, i):
        # [FUNCTION DOC] HELPER: Look up the random sample for this prefix and dim (handling 3-D and 4-D input prefix layouts) and compute the window start as an indirect index bounded by the input size.
        # Handle indexing for samples tensor correctly for different input dimensions
        # samples tensor always has shape (N, C, 2) for fractional_max_pool2d where:
        # - N=1 for 3D inputs (C,H,W), N=batch_size for 4D inputs (N,C,H,W)
        # - C=num_channels
        # - 2 for the two spatial dimensions (height, width)
        samples_shape = samples.get_size()

        if len(samples_shape) == 3:  # Expected: (N, C, 2)
            if len(prefix) == 1:
                # 3D input case: prefix=(channel,), samples=(1, C, 2)
                # Access: samples[0, channel, dim]
                sample = samples_loader([0, prefix[0], ndims - 1 - dim])
            elif len(prefix) >= 2:
                # 4D+ input case: prefix=(batch, channel, ...), samples=(batch, C, 2)
                # Access: samples[batch, channel, dim]
                sample = samples_loader([prefix[0], prefix[1], ndims - 1 - dim])
            else:
                # Edge case - shouldn't happen for valid fractional pooling
                sample = samples_loader([0, 0, ndims - 1 - dim])
        else:
            # Fallback for unexpected tensor shapes
            sample = samples_loader([*prefix, ndims - 1 - dim])
        i_expr = ops.index_expr(i, samples.get_dtype())
        diff = ops.index_expr(in_sz - kernel_sz, torch.int64)
        out_sz_expr = ops.index_expr(out_sz - 1, torch.int64)
        alpha = ops.truediv(
            ops.to_dtype(diff, torch.float64), ops.to_dtype(out_sz_expr, torch.float64)
        )
        alpha = ops.where(ops.eq(out_sz_expr, 0), 0, alpha)
        seq_i = ops.trunc((i_expr + sample) * alpha) - ops.trunc(sample * alpha)
        seq_i = ops.to_dtype(seq_i, torch.int64)
        mask = ops.lt(i_expr, out_sz_expr)
        return ops.indirect_indexing(ops.where(mask, seq_i, diff), sympy.sympify(in_sz))

    return load


@register_lowering(aten.fractional_max_pool2d)
def fractional_max_pool2d(x, kernel_size, output_size, random_samples):
    # [FUNCTION DOC] Lowering for aten.fractional_max_pool2d: delegates to the n-dimensional implementation with two spatial dims.
    return _fractional_max_pool(x, kernel_size, output_size, random_samples, n_dim=2)


@register_lowering(aten.fractional_max_pool3d)
def fractional_max_pool3d(x, kernel_size, output_size, random_samples):
    # [FUNCTION DOC] Lowering for aten.fractional_max_pool3d: delegates to the n-dimensional implementation with three spatial dims.
    return _fractional_max_pool(x, kernel_size, output_size, random_samples, n_dim=3)


def _fractional_max_pool(x, kernel_size, output_size, random_samples, n_dim):
    # [FUNCTION DOC] HELPER: Fractional max pooling over the last n_dim dims. Window starts come from random samples; builds max and argmax Reductions over the kernel window (unrolled up to 25 elements), realizing each if still a reduction, and converts the
    # argmax offsets to flat input indices. Returns (values, indices).
    x.realize_hint()
    batch, inp_dhw = x.shape[:-n_dim], x.shape[-n_dim:]

    with config.patch(unroll_reductions_threshold=25):
        dhw_index_fn = [
            _fractional_pooling_offsets(
                samples=random_samples,
                in_sz=inp_dhw,
                out_sz=output_size,
                kernel_sz=kernel_size,
                ndims=n_dim,
                dim=d,
            )
            for d in range(n_dim)
        ]

        x_loader = x.make_loader()

        def fn_inner(idx, reduction_idx):
            # [FUNCTION DOC] HELPER: Load the input element for an output position plus window offsets.
            prefix = idx[:-n_dim]
            return x_loader([*prefix, *increments_to_index(idx, reduction_idx)])

        def increments_to_index(idx, reduction_idx):
            # [FUNCTION DOC] HELPER: Map output coordinates and window offsets to input coordinates using the fractional window start offsets.
            prefix = idx[:-n_dim]
            bdhw = idx[-n_dim:]
            return [
                dhw_index_fn[d](prefix, bdhw[d]) + reduction_idx[d]
                for d in range(n_dim)
            ]

        new_size = list(batch) + list(output_size)
        dtype = x.get_dtype()
        result = Reduction.create(
            reduction_type="max",
            input_node=x,
            device=x.get_device(),
            dst_dtype=dtype,
            src_dtype=dtype,
            inner_fn=fn_inner,
            ranges=new_size,
            reduction_ranges=kernel_size,
        )
        offsets = Reduction.create(
            reduction_type="argmax",
            input_node=x,
            device=x.get_device(),
            dst_dtype=torch.int64,
            src_dtype=dtype,
            inner_fn=fn_inner,
            ranges=new_size,
            reduction_ranges=kernel_size,
        )
        if not (isinstance(result, TensorBox)):
            raise AssertionError(result)
        if isinstance(result.data.data, Reduction):  # type: ignore[attr-defined]
            # Only realize if reduction isn't unrolled
            result.realize()
        if not (isinstance(offsets, TensorBox)):
            raise AssertionError(offsets)
        if isinstance(offsets.data.data, Reduction):  # type: ignore[attr-defined]
            # Only realize if reduction isn't unrolled
            offsets.realize()

        indices = _pool_offsets_to_indices(
            offsets, kernel_size, x.shape, increments_to_index
        )
        return result, indices


@register_lowering(aten.upsample_nearest2d_backward.default)
def upsample_nearest2d_backward(
    x, output_size=None, input_size=None, scales_h=None, scales_w=None
):
    # [FUNCTION DOC] Lowering for aten.upsample_nearest2d_backward. Divisible sizes reduce to avg_pool2d with divisor 1 (a sum); otherwise a Pointwise over the input shape sums, with ceil-based window bounds, all
    # output-gradient elements mapped to each input pixel using the adaptive pooling machinery with zero padding. output_size, scales_h and scales_w are not used; input_size is required.
    x.realize_hint()

    *_batch, inp_h, inp_w = x.get_size()
    inp_h = V.graph.sizevars.guard_int(inp_h)
    inp_w = V.graph.sizevars.guard_int(inp_w)

    # pyrefly: ignore [not-iterable]
    *_batch, out_h, out_w = input_size

    if inp_h % out_h == 0 and inp_w % out_w == 0:
        return avg_pool2d(
            x, [FloorDiv(inp_h, out_h), FloorDiv(inp_w, out_w)], divisor_override=1
        )

    h_kernel_max = ceildiv(inp_h, out_h)
    w_kernel_max = ceildiv(inp_w, out_w)

    def start_index(index, out_dim, inp_dim):
        # [FUNCTION DOC] HELPER: Window start for an output index: ceil(index * inp / out).
        return CeilDiv(index * inp_dim, sympy.sympify(out_dim))

    def end_index(index, out_dim, inp_dim):
        # [FUNCTION DOC] HELPER: Window end for an output index, equal to the start of the next index.
        return start_index((index + 1), out_dim, inp_dim)

    fn_sum = _adaptive_pooling_fn(
        start_index=start_index,
        end_index=end_index,
        kernel_maxes=[h_kernel_max, w_kernel_max],
        in_sizes=[inp_h, inp_w],
        out_sizes=[out_h, out_w],
        pooling_fn=ops.add,
    )

    def fn(idx):
        # [FUNCTION DOC] HELPER: Sum the gradient window for an input position.
        return fn_sum(idx, pad_adaptive_loader(x))

    rv = Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=fn,
        # pyrefly: ignore [bad-argument-type, no-matching-overload]
        ranges=list(input_size),
    )

    return rv


@register_lowering(aten.avg_pool2d, type_promotion_kind=None)
def avg_pool2d(
    x,
    kernel_size,
    stride=(),
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    # [FUNCTION DOC] Lowering for aten.avg_pool2d: delegates to the n-d average pooling with two spatial dims.
    return _avg_poolnd(
        x,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        dim=2,
    )


@register_lowering(aten.avg_pool3d, type_promotion_kind=None)
def avg_pool3d(
    x,
    kernel_size,
    stride=(),
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    # [FUNCTION DOC] Lowering for aten.avg_pool3d: delegates to the n-d average pooling with three spatial dims.
    return _avg_poolnd(
        x,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        dim=3,
    )


fallbacks_avg_poolnd = [
    fallback_handler(aten.avg_pool1d.default, add_to_fallback_set=False),
    fallback_handler(aten.avg_pool2d.default, add_to_fallback_set=False),
    fallback_handler(aten.avg_pool3d.default, add_to_fallback_set=False),
]


def _avg_poolnd(
    x,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
    dim,
):
    # [FUNCTION DOC] HELPER: Average pooling over the last dim dims. Stride defaults to the kernel size and padding to zero. Validates ranks (raises an error). Uses the eager fallback for big windows (over 25 elements) whose stride differs from the kernel.
    # Otherwise sums the window with a Reduction (unrolled for windows up to 25), using a zero boundary condition if padding or ceil mode applies, then divides by the divisor override, the window size, or a per-position valid count (honoring
    # count_include_pad). Computation is in the compute dtype; result cast to the input dtype.
    if not stride:
        stride = kernel_size
    if not padding:
        padding = [0] * dim
    kernel_size = pad_listlike(kernel_size, dim)
    stride = pad_listlike(stride, dim)
    padding = pad_listlike(padding, dim)

    if not (isinstance(x, TensorBox)):
        raise AssertionError("expected: isinstance(x, TensorBox)")
    if len(kernel_size) != dim:
        raise AssertionError("expected: len(kernel_size) == dim")
    if len(stride) != dim:
        raise AssertionError("expected: len(stride) == dim")
    if len(padding) != dim:
        raise AssertionError("expected: len(padding) == dim")
    if len(x.get_size()) not in (dim + 1, dim + 2):
        raise AssertionError("expected: len(x.get_size()) in (dim + 1, dim + 2)")

    x.realize_hint()
    batch = x.get_size()[:-dim]
    h = x.get_size()[-dim:]

    h_out, ceil_modes = zip(
        *[
            pooling_size(h[i], i, kernel_size, stride, padding, ceil_mode)
            for i in range(dim)
        ]
    )

    if any(padding) or any(ceil_modes):
        x_loader = constant_boundary_condition(x, 0.0, dim=dim)
        had_padding = True
    else:
        x_loader = x.make_loader()
        had_padding = False

    new_size = list(batch) + list(h_out)
    dtype = x.get_dtype()
    # compute in higher-precision until scaling
    output_dtype = get_promoted_dtype(
        x,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
        return_compute_dtype=True,
    )

    def fn_inner(idx, reduction_idx):
        # [FUNCTION DOC] HELPER: Map output coordinates plus window offsets to input coordinates and load.
        prefix = idx[:-dim]
        bh = idx[-dim:]
        ih = reduction_idx
        ih = [bh[i] * stride[i] + ih[i] - padding[i] for i in range(dim)]
        return x_loader([*prefix, *ih])

    window_size = functools.reduce(operator.mul, kernel_size)

    if window_size > 25 and any(
        V.graph.sizevars.statically_known_true(sympy.Ne(k, s))
        for k, s in zip(kernel_size, stride)
    ):
        fallback = fallbacks_avg_poolnd[dim - 1]
        return fallback(
            x,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )

    # TODO: remove this when #100331 is merged. We only do this
    # for window_size <=25 to avoid performance regressions compared
    # to the previous algorithm which unrolled manually for <=25
    context = (
        config.patch(unroll_reductions_threshold=25)
        if window_size <= 25
        else contextlib.nullcontext()
    )

    device = x.get_device()
    if device is None:
        raise AssertionError("expected: device is not None")

    with context:
        rv = Reduction.create(
            reduction_type="sum",
            input_node=x,
            device=device,
            dst_dtype=output_dtype,
            src_dtype=dtype,
            inner_fn=fn_inner,
            ranges=new_size,
            reduction_ranges=kernel_size,
        )
    if hasattr(rv.data, "data") and isinstance(rv.data.data, Reduction):
        # Only realize if reduction isn't unrolled
        rv.realize()

    if not had_padding or divisor_override:
        divisor = divisor_override if divisor_override else window_size
        result = div_prim(rv, divisor)
    else:

        def fn_count(idx):
            # [FUNCTION DOC] HELPER: Number of elements in the pooling window at an output position (clipped to padded or real bounds depending on count_include_pad), as a product over dims.
            bh = idx[-dim:]

            divide_factors = []
            for i in range(dim):
                hstart = bh[i] * stride[i] - padding[i]
                hend = Min(hstart + kernel_size[i], h[i] + padding[i])
                if not count_include_pad:
                    hstart = Max(hstart, 0)
                    hend = Min(hend, h[i])
                factor = ops.index_expr(hend - hstart, torch.int32)
                divide_factors.append(factor)
            return functools.reduce(ops.mul, divide_factors)

        divide_factor = Pointwise.create(
            device=x.get_device(),
            dtype=dtype,
            inner_fn=fn_count,
            ranges=new_size,
        )
        result = div_prim(rv, divide_factor)

    return to_dtype(result, dtype)


fallback_avg_pool2d_backward = fallback_handler(
    aten.avg_pool2d_backward.default, add_to_fallback_set=False
)


@register_lowering(aten.avg_pool2d_backward, type_promotion_kind=None)
def avg_pool2d_backward(
    grad_output,
    x,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override=None,
):
    # [FUNCTION DOC] Lowering for aten.avg_pool2d_backward. Validates arguments (zero divisor or wrong ranks raise an error), falls back to eager when the number of windows covering an input element exceeds 25, otherwise builds a Pointwise over the input
    # shape that sums grad_output values of covering windows each divided by its scale (divisor override, kernel area, or valid count when padding is excluded), masked to in-range windows.
    if divisor_override is not None and divisor_override == 0:
        raise AssertionError("divisor must be not zero")
    if not stride:
        stride = kernel_size
    if not padding:
        padding = [0, 0]

    if not (isinstance(grad_output, TensorBox)):
        raise AssertionError("expected: isinstance(grad_output, TensorBox)")
    if not (isinstance(x, TensorBox)):
        raise AssertionError("expected: isinstance(x, TensorBox)")
    if len(kernel_size) != 2:
        raise AssertionError("expected: len(kernel_size) == 2")
    if len(stride) != 2:
        raise AssertionError("expected: len(stride) == 2")
    if len(padding) != 2:
        raise AssertionError("expected: len(padding) == 2")
    if len(x.get_size()) not in (3, 4):
        raise AssertionError("expected: len(x.get_size()) in (3, 4)")

    grad_output.realize_hint()  # we will read this many times, so make sure it is computed

    *_, height, width = x.get_size()

    _h_out, ceil_mode1 = pooling_size(
        height, 0, kernel_size, stride, padding, ceil_mode
    )
    _w_out, ceil_mode2 = pooling_size(width, 1, kernel_size, stride, padding, ceil_mode)

    grad_loader = grad_output.make_loader()

    had_padding = padding[0] or padding[1] or ceil_mode1 or ceil_mode2

    *_, pooled_height, pooled_width = grad_output.get_size()
    new_size = list(x.get_size())
    dtype = x.get_dtype()

    h_window_size = max(
        max(FloorDiv(h, stride[0]) - max(0, FloorDiv(h - kernel_size[0], stride[0])), 1)
        for h in range(kernel_size[0] * 2)
    )
    w_window_size = max(
        max(FloorDiv(w, stride[1]) - max(0, FloorDiv(w - kernel_size[1], stride[1])), 1)
        for w in range(kernel_size[1] * 2)
    )

    window_size = h_window_size * w_window_size
    if window_size > 25:
        # Kernel size too big. Results in hard-to-optimize Triton code. Use fallback.
        return fallback_avg_pool2d_backward(
            grad_output,
            x,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )

    def compute_pool_size_without_padding(ph, pw):
        # [FUNCTION DOC] HELPER: Valid (non-padding) element count of the pooling window at output position (ph, pw), as an ops value.
        """
        This computes the scaling factor that we will divide an element
        by when `count_include_pad=False`
        """
        stride_h = ops.constant(stride[0], torch.int32)
        stride_w = ops.constant(stride[1], torch.int32)
        pad_h = ops.constant(padding[0], torch.int32)
        pad_w = ops.constant(padding[1], torch.int32)
        kernel_h = ops.constant(kernel_size[0], torch.int32)
        kernel_w = ops.constant(kernel_size[1], torch.int32)
        hstart = ops.sub(ops.mul(ph, stride_h), pad_h)
        wstart = ops.sub(ops.mul(pw, stride_w), pad_w)
        hend = ops.minimum(
            ops.add(hstart, kernel_h),
            ops.add(ops.index_expr(height, torch.int32), pad_h),
        )
        wend = ops.minimum(
            ops.add(wstart, kernel_w),
            ops.add(ops.index_expr(width, torch.int32), pad_w),
        )
        hstart = ops.maximum(hstart, ops.constant(0, torch.int32))
        wstart = ops.maximum(wstart, ops.constant(0, torch.int32))
        hend = ops.minimum(hend, ops.index_expr(height, torch.int32))
        wend = ops.minimum(wend, ops.index_expr(width, torch.int32))
        divide_factor = ops.mul(ops.sub(hend, hstart), ops.sub(wend, wstart))
        return divide_factor

    def fn(idx):
        # [FUNCTION DOC] HELPER: Accumulate the scaled gradient contributions of all windows covering input position (h, w).
        *prefix, h, w = idx
        h = h + padding[0]
        w = w + padding[1]
        phstart = ops.index_expr(
            FloorDiv(h - kernel_size[0] + stride[0], stride[0]), torch.int32
        )
        pwstart = ops.index_expr(
            FloorDiv(w - kernel_size[1] + stride[1], stride[1]), torch.int32
        )
        phend = ops.index_expr(FloorDiv(h, stride[0]) + 1, torch.int32)
        pwend = ops.index_expr(FloorDiv(w, stride[1]) + 1, torch.int32)

        phstart = ops.maximum(phstart, ops.constant(0, torch.int32))
        pwstart = ops.maximum(pwstart, ops.constant(0, torch.int32))
        phend = ops.minimum(phend, ops.index_expr(pooled_height, torch.int32))
        pwend = ops.minimum(pwend, ops.index_expr(pooled_width, torch.int32))

        gradient = None
        for ph_ in range(h_window_size):
            for pw_ in range(w_window_size):
                ph = ops.add(phstart, ops.constant(ph_, torch.int32))
                pw = ops.add(pwstart, ops.constant(pw_, torch.int32))

                if divisor_override is not None:
                    scale = divisor_override
                elif count_include_pad or not had_padding:
                    scale = kernel_size[0] * kernel_size[1]
                else:
                    scale = compute_pool_size_without_padding(ph, pw)

                part = ops.truediv(
                    grad_loader(
                        [
                            *prefix,
                            ops.indirect_indexing(
                                ops.minimum(
                                    ph, ops.sub(phend, ops.constant(1, torch.int32))
                                ),
                                pooled_height,
                                check=False,
                            ),
                            ops.indirect_indexing(
                                ops.minimum(
                                    pw, ops.sub(pwend, ops.constant(1, torch.int32))
                                ),
                                pooled_width,
                                check=False,
                            ),
                        ]
                    ),
                    scale,
                )

                mask = ops.and_(
                    ops.lt(ph, phend),
                    ops.lt(pw, pwend),
                )
                if gradient is None:
                    gradient = ops.where(mask, part, ops.constant(0.0, torch.float32))
                else:
                    gradient = ops.where(mask, ops.add(gradient, part), gradient)
        if gradient is None:
            raise AssertionError("expected: gradient is not None")
        return gradient

    rv = Pointwise.create(
        device=grad_output.get_device(),
        dtype=dtype,
        inner_fn=fn,
        ranges=new_size,
    )
    return rv


fallback_avg_pool3d_backward = fallback_handler(
    aten.avg_pool3d_backward.default, add_to_fallback_set=False
)


@register_lowering(aten.avg_pool3d_backward, type_promotion_kind=None)
def avg_pool3d_backward(
    grad_output,
    x,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override=None,
):
    # [FUNCTION DOC] Lowering for aten.avg_pool3d_backward. Same approach as the 2-D version over depth, height and width: validates arguments (raises an error), falls back to eager when more than 125 windows may cover an element,
    # otherwise builds a Pointwise summing scaled grad_output over covering windows.
    if divisor_override is not None and divisor_override == 0:
        raise AssertionError("divisor must be not zero")
    if not stride:
        stride = kernel_size
    if not padding:
        padding = [0, 0, 0]

    if not (isinstance(grad_output, TensorBox)):
        raise AssertionError("expected: isinstance(grad_output, TensorBox)")
    if not (isinstance(x, TensorBox)):
        raise AssertionError("expected: isinstance(x, TensorBox)")
    if len(kernel_size) != 3:
        raise AssertionError("expected: len(kernel_size) == 3")
    if len(stride) != 3:
        raise AssertionError("expected: len(stride) == 3")
    if len(padding) != 3:
        raise AssertionError("expected: len(padding) == 3")
    if len(x.get_size()) not in (4, 5):
        raise AssertionError("expected: len(x.get_size()) in (4, 5)")

    grad_output.realize_hint()

    *_batch, depth, height, width = x.get_size()

    _d_out, ceil_mode_d = pooling_size(
        depth, 0, kernel_size, stride, padding, ceil_mode
    )
    _h_out, ceil_mode_h = pooling_size(
        height, 1, kernel_size, stride, padding, ceil_mode
    )
    _w_out, ceil_mode_w = pooling_size(
        width, 2, kernel_size, stride, padding, ceil_mode
    )

    grad_loader = grad_output.make_loader()
    had_padding = any(padding) or ceil_mode_d or ceil_mode_h or ceil_mode_w

    *_, pooled_depth, pooled_height, pooled_width = grad_output.get_size()
    new_size = list(x.get_size())
    dtype = x.get_dtype()

    d_window_size, h_window_size, w_window_size = (
        max(
            max(d // stride[i] - max(0, (d - kernel_size[i]) // stride[i]), 1)
            for d in range(kernel_size[i] * 2)
        )
        for i in range(3)
    )

    window_size = d_window_size * h_window_size * w_window_size
    if window_size > 125:
        # Kernel size too big. Results in hard-to-optimize Triton code.
        return fallback_avg_pool3d_backward(
            grad_output,
            x,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )

    def compute_pool_size_without_padding(pd, ph, pw):
        # [FUNCTION DOC] HELPER: Valid (non-padding) element count of the 3-D pooling window at output position (pd, ph, pw), as an ops value.
        stride_d, stride_h, stride_w = (ops.constant(s, torch.int32) for s in stride)
        pad_d, pad_h, pad_w = (ops.constant(p, torch.int32) for p in padding)
        kernel_d, kernel_h, kernel_w = (
            ops.constant(k, torch.int32) for k in kernel_size
        )

        dstart, hstart, wstart = (
            ops.sub(ops.mul(p, s), pad)
            for p, s, pad in zip(
                [pd, ph, pw], [stride_d, stride_h, stride_w], [pad_d, pad_h, pad_w]
            )
        )
        dend, hend, wend = (
            ops.minimum(
                ops.add(start, k), ops.add(ops.index_expr(dim, torch.int32), pad)
            )
            for start, k, dim, pad in zip(
                [dstart, hstart, wstart],
                [kernel_d, kernel_h, kernel_w],
                [depth, height, width],
                [pad_d, pad_h, pad_w],
            )
        )
        dstart, hstart, wstart = (
            ops.maximum(start, ops.constant(0, torch.int32))
            for start in [dstart, hstart, wstart]
        )
        dend, hend, wend = (
            ops.minimum(end, ops.index_expr(dim, torch.int32))
            for end, dim in zip([dend, hend, wend], [depth, height, width])
        )
        divide_factor = ops.mul(
            ops.mul(ops.sub(dend, dstart), ops.sub(hend, hstart)), ops.sub(wend, wstart)
        )
        return divide_factor

    def fn(idx):
        # [FUNCTION DOC] HELPER: Accumulate the scaled gradient contributions of all windows covering input position (d, h, w).
        *prefix, d, h, w = idx
        d, h, w = (v + pad for v, pad in zip([d, h, w], padding))

        pdstart, phstart, pwstart = (
            ops.index_expr(FloorDiv(v - k + s, s), torch.int32)
            for v, k, s in zip([d, h, w], kernel_size, stride)
        )

        pdend, phend, pwend = (
            ops.index_expr(FloorDiv(v, s) + 1, torch.int32)
            for v, s in zip([d, h, w], stride)
        )

        pdstart, phstart, pwstart = (
            ops.maximum(pstart, ops.constant(0, torch.int32))
            for pstart in [pdstart, phstart, pwstart]
        )
        pdend, phend, pwend = (
            ops.minimum(pend, ops.index_expr(pooled_dim, torch.int32))
            for pend, pooled_dim in zip(
                [pdend, phend, pwend], [pooled_depth, pooled_height, pooled_width]
            )
        )

        gradient = None
        # Iterate over the 3D region to accumulate gradients
        for pd_ in range(d_window_size):
            for ph_ in range(h_window_size):
                for pw_ in range(w_window_size):
                    pd, ph, pw = (
                        ops.add(pstart, ops.constant(p_, torch.int32))
                        for pstart, p_ in zip(
                            [pdstart, phstart, pwstart], [pd_, ph_, pw_]
                        )
                    )

                    if divisor_override is not None:
                        scale = divisor_override
                    elif count_include_pad or not had_padding:
                        scale = kernel_size[0] * kernel_size[1] * kernel_size[2]
                    else:
                        scale = compute_pool_size_without_padding(pd, ph, pw)

                    part = ops.truediv(
                        grad_loader(
                            [
                                *prefix,
                                ops.indirect_indexing(
                                    ops.minimum(
                                        pd, ops.sub(pdend, ops.constant(1, torch.int32))
                                    ),
                                    pooled_depth,
                                    check=False,
                                ),
                                ops.indirect_indexing(
                                    ops.minimum(
                                        ph, ops.sub(phend, ops.constant(1, torch.int32))
                                    ),
                                    pooled_height,
                                    check=False,
                                ),
                                ops.indirect_indexing(
                                    ops.minimum(
                                        pw, ops.sub(pwend, ops.constant(1, torch.int32))
                                    ),
                                    pooled_width,
                                    check=False,
                                ),
                            ]
                        ),
                        scale,
                    )

                    mask = ops.and_(
                        ops.and_(ops.lt(pd, pdend), ops.lt(ph, phend)),
                        ops.lt(pw, pwend),
                    )
                    if gradient is None:
                        gradient = ops.where(
                            mask, part, ops.constant(0.0, torch.float32)
                        )
                    else:
                        gradient = ops.where(mask, ops.add(gradient, part), gradient)
        if gradient is None:
            raise AssertionError("expected: gradient is not None")
        return gradient

    rv = Pointwise.create(
        device=grad_output.get_device(),
        dtype=dtype,
        inner_fn=fn,
        ranges=new_size,
    )
    return rv


def _validate_reduction_axis(x, axis):
    # [FUNCTION DOC] HELPER: Normalize a reduction axis spec (int, sequence, or empty meaning all dims) to a list of non-negative dims for x. For 0-d inputs only empty, 0 or -1 are allowed and an empty list is returned. Raises an error for invalid or duplicate axes.
    # KNOWN ISSUE: the second part of the bounds check mentioning a zero-length size is unreachable because 0-d inputs return earlier.
    size = x.get_size()
    if isinstance(axis, int):
        axis = [axis]
    elif not axis:
        axis = range(len(size))
    if len(size) == 0:
        if tuple(axis) not in [(), (0,), (-1,)]:
            raise AssertionError(f"invalid axis: {axis}")
        return []
    axis = list(axis)
    for i in range(len(axis)):
        if axis[i] < 0:
            axis[i] += len(size) if len(size) else 1
        if not (0 <= axis[i] < len(size) or (len(size) == 0 and axis[i] == 0)):
            raise AssertionError(
                "expected: 0 <= axis[i] < len(size) or (len(size) == 0 and axis..."
            )
    if len(OrderedSet(axis)) != len(axis):
        raise AssertionError("reduction axis not unique")
    return axis


def _make_reduction_inner(
    x, *, axis, keepdims, dtype, override_return_dtype, reduction_type=None
):
    # [FUNCTION DOC] HELPER: Compute the keyword arguments for creating a Reduction over x: optionally casts x to dtype, validates the axes, splits dims into kept and reduced, and builds a loader mapping (kept index, reduction index) to the input index
    # (keepdims inserts size-one output dims). For arg-reductions over several dims on supported backends with transposed or non-contiguous layouts the loader returns a (value, logical row-major index) pair. Returns a mapping with device, dst_dtype,
    # src_dtype, inner_fn, ranges and reduction_ranges.
    if dtype is not None:
        x = to_dtype(x, dtype)
    size = x.get_size()
    axis = OrderedSet[int](_validate_reduction_axis(x, axis))

    kept_sizes = []
    kept_idx = []
    reduced_sizes = []
    reduced_idx = []
    for i in range(len(size)):
        if i in axis:
            reduced_idx.append(i)
            reduced_sizes.append(size[i])
        else:
            kept_idx.append(i)
            kept_sizes.append(size[i])

    # For argmax/argmin compute logical indices when the tensor has non-contiguous layout.
    should_compute_logical_index = False
    supports_logical_index_argreduce = is_triton(x) or (
        ir.get_device_type(x) == "cpu" and config.cpu_backend == "cpp"
    )
    if (
        reduction_type
        in (
            "argmax",
            "argmin",
            "argmax_value",
            "argmin_value",
            "argmax_with_value",
            "argmin_with_value",
        )
        and len(reduced_sizes) > 1
        and supports_logical_index_argreduce
    ):
        if isinstance(x.data, PermuteView):
            should_compute_logical_index = True
        elif isinstance(x.data, ir.ReinterpretView) or (
            isinstance(x.data, ir.StorageBox) and isinstance(x.data.data, ir.Buffer)
        ):
            layout = x.get_layout()
            should_compute_logical_index = (
                layout.is_transposed() or not layout.is_contiguous()
            )

    def loader(index, reduction_index):
        # [FUNCTION DOC] HELPER: Reassemble the full input index from kept and reduced coordinates, load the value, and for logical-index arg-reductions also return the row-major flat index over the reduced dims. Raises an error on length mismatches.
        if len(reduction_index) != len(reduced_idx):
            raise AssertionError("expected: len(reduction_index) == len(reduced_idx)")
        if keepdims:
            if len(index) != len(size):
                raise AssertionError("expected: len(index) == len(size)")
            index = [index[i] for i in kept_idx]
        if len(index) != len(kept_idx):
            raise AssertionError("expected: len(index) == len(kept_idx)")
        new_index = [None] * (len(index) + len(reduction_index))
        for idx, var in itertools.chain(
            zip(kept_idx, index), zip(reduced_idx, reduction_index)
        ):
            new_index[idx] = var
        value = inner_loader(new_index)

        # For argmax/argmin reductions, return tuple with logical linear index if needed
        if should_compute_logical_index:
            rindex = [sympy.expand(i) for i in reduction_index]

            # Compute linear index in row-major order
            # For reduction_ranges = [4, 6]: linear_index = r0 * 6 + r1
            linear_idx = rindex[0]
            for i in range(1, len(rindex)):
                linear_idx = linear_idx * reduced_sizes[i] + rindex[i]

            return (value, ops.index_expr(linear_idx, torch.int64))

        return value

    if keepdims:
        new_size = list(size)
        for i in reduced_idx:
            new_size[i] = sympy.S.One
    else:
        new_size = kept_sizes

    inner_loader = x.make_loader()
    return dict(
        device=x.get_device(),
        dst_dtype=override_return_dtype or x.get_dtype(),
        src_dtype=x.get_dtype(),
        inner_fn=loader,
        ranges=new_size,
        reduction_ranges=reduced_sizes,
    )


def make_reduction(
    reduction_type: ReductionType,
    override_return_dtype=None,
    *,
    strict_sum: bool = False,
) -> Callable[..., TensorBox]:
    # [FUNCTION DOC] Create a reduction lowering for reduction_type, optionally forcing an output dtype and strict_sum. The lowering takes (x, axis, keepdims, dtype), converts bool inputs of argmax/argmin to int32, builds a Reduction node from
    # _make_reduction_inner and realizes it if it was not unrolled. Returns the lowering function.
    def inner(x, axis=None, keepdims=False, *, dtype=None) -> TensorBox:
        # [FUNCTION DOC] HELPER: The lowering body: reduce x along axis with the configured reduction type and return the resulting TensorBox.
        # For argmax/argmin on boolean tensors, cast to int32 first to ensure
        # correct comparison. Boolean comparisons can produce incorrect indices
        # on multiple backends (Triton, CPU, etc.).
        # See https://github.com/pytorch/pytorch/issues/174069
        # and https://github.com/pytorch/pytorch/issues/184893
        if reduction_type in ("argmax", "argmin") and x.get_dtype() == torch.bool:
            x = to_dtype(x, torch.int32)
        kwargs = _make_reduction_inner(
            x,
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            override_return_dtype=override_return_dtype,
            reduction_type=reduction_type,
        )
        result = Reduction.create(
            reduction_type=reduction_type,
            input_node=x,
            strict_sum=strict_sum,
            **kwargs,
        )
        if isinstance(
            result.data.data,  # type: ignore[attr-defined, attr-type, union-attr]
            Reduction,
        ):  # Only realize if reduction isn't unrolled
            result.realize()
        return result

    return inner


def make_arg_with_value_reduction(
    reduction_type: ReductionType,
) -> Callable[..., Sequence[TensorBox]]:
    # [FUNCTION DOC] Create a lowering for reductions that return both an arg index and a value (argmax/argmin with value). The result comes from ArgReduction.create using the shared reduction inner function. Returns the lowering function.
    def inner(x, axis=None, keepdims=False) -> Sequence[TensorBox]:
        # [FUNCTION DOC] HELPER: The lowering body: reduce x along axis and return the sequence of outputs.
        kwargs = _make_reduction_inner(
            x,
            axis=axis,
            keepdims=keepdims,
            dtype=None,
            override_return_dtype=x.get_dtype(),
            reduction_type=reduction_type,
        )
        return ir.ArgReduction.create(
            reduction_type=reduction_type, input_node=x, **kwargs
        )

    return inner


def _make_scan_inner(x, *, axis, dtype):
    # [FUNCTION DOC] HELPER: Compute keyword arguments for a scan over x along axis: optionally casts to dtype, validates the dim, and returns device, dtypes, inner function loaders, size and axis.
    if dtype is not None:
        x = to_dtype(x, dtype)
    axis = _validate_dim(x, axis)

    return dict(
        device=x.get_device(),
        dtypes=(x.get_dtype(),),
        inner_fns=(x.make_loader(),),
        size=x.get_size(),
        axis=axis,
    )


@register_lowering(aten.mean)
def mean(x, axis=None, keepdim=False, *, dtype=None):
    # [FUNCTION DOC] Lowering for aten.mean: optional dtype cast, computes in float32 for half types, sums over the axes and divides by the number of reduced elements (a symbolic constant expanded to the output shape), then casts back to the original dtype.
    if dtype is not None:
        x = to_dtype(x, dtype)
    size = x.get_size()
    axis = _validate_reduction_axis(x, axis)
    # compute in higher-precision until end of mean lowering
    output_dtype = x.get_dtype()
    if output_dtype in (torch.float16, torch.bfloat16):
        x = to_dtype(x, torch.float)
    sum_result = sum_(x, axis, keepdim)
    denom = sympy_product(size[i] for i in axis)
    denom = ir.IndexingConstant(index=denom, dtype=x.get_dtype(), device=x.get_device())
    denom = ExpandView.create(denom, list(sum_result.get_size()))
    return to_dtype(div(sum_result, denom), output_dtype)


def var_mean_sum_(x, axis, correction, keepdim, return_mean):
    # [FUNCTION DOC] HELPER: Two-pass variance: computes the mean, sums squared deviations, and divides by max(count - correction, 0) (correction defaults to 1). Returns a tuple of (variance,) or (variance, mean); the mean is realized if returned.
    if correction is None:
        correction = 1

    size = x.get_size()
    axis = _validate_reduction_axis(x, axis)
    x_mean = mean(x, axis, keepdim=True)
    if return_mean:
        x_mean.realize()

    diffs = square(sub(x, x_mean))
    sum_result = sum_(diffs, axis, keepdim)

    denom = sympy_product(size[i] for i in axis)
    if correction:
        denom = Max(denom - correction, 0)
    denom = ir.IndexingConstant(index=denom, dtype=x.get_dtype(), device=x.get_device())
    denom = ExpandView.create(denom, list(sum_result.get_size()))
    x_var = div(sum_result, denom)
    if not return_mean:
        return (x_var,)

    x_mean = x_mean if keepdim else squeeze(x_mean, axis)
    return x_var, x_mean


def use_two_step_variance(x, axis, keepdim, input_dtype):
    # [FUNCTION DOC] HELPER: Decide whether to use the two-step variance algorithm instead of Welford. Requires a statically known reduction size not larger than a backend-specific threshold (CPU config, CUDA triton for fp16/bf16 with a minimum size
    # and without reduction splitting, otherwise the unroll threshold) and more than one output element. Returns a bool.
    # The two-step algorithm can be faster for non-split reductions because
    # Welford does more work per element. Preserve the old tiny-reduction
    # two-step path, keep Welford for the rest of the small reductions where
    # the speedup is limited and training gradients are more sensitive to the
    # different accumulation order, and keep Welford for larger or split
    # reductions where avoiding another full pass over the data is profitable.
    axis = _validate_reduction_axis(x, axis)
    kwargs = _make_reduction_inner(
        x, axis=axis, keepdims=keepdim, dtype=None, override_return_dtype=None
    )

    ranges = kwargs["ranges"]
    reduction_numel = sympy_product(kwargs["reduction_ranges"])
    device = x.get_device()
    check_for_split = False
    min_numel = 0
    is_cuda_two_step_dtype = input_dtype in (torch.bfloat16, torch.float16)
    if device and device.type == "cpu":
        # 1024 is a default value to pass all the UTs about accuracy.
        # A larger threshold can still get performance benefits.
        threshold = config.cpp.use_two_step_variance_threshold
    elif device and device.type == "cuda" and is_triton(x) and is_cuda_two_step_dtype:
        min_numel = config.triton.use_two_step_variance_min_numel
        threshold = config.triton.use_two_step_variance_threshold
        check_for_split = True
    else:
        threshold = config.unroll_reductions_threshold

    if not isinstance(reduction_numel, sympy.Integer):
        return False

    reduction_numel = int(reduction_numel)
    if reduction_numel > threshold or sympy_product(ranges) == 1:
        return False

    if min_numel and config.unroll_reductions_threshold < reduction_numel < min_numel:
        return False

    if not check_for_split:
        return True

    _, split = ir.Reduction.num_splits(
        reduction_numel=reduction_numel,
        reduction_type="sum",
        **kwargs,
    )
    return V.graph.sizevars.statically_known_leq(split, 1)


def var_mean_welford_(x, axis, *, correction, keepdim, return_mean):
    # [FUNCTION DOC] HELPER: Variance via a Welford reduction. Computes mean and M2, realizes M2, and scales it by 1 / max(count - correction, 0) (correction defaults to 1) with a pointwise node. Returns (variance,) or (variance, mean).
    if correction is None:
        correction = 1

    kwargs = _make_reduction_inner(
        x, axis=axis, keepdims=keepdim, dtype=None, override_return_dtype=None
    )
    loader = kwargs.pop("inner_fn")
    kwargs.pop("dst_dtype")
    kwargs.pop("src_dtype")

    mean, m2, _ = ir.WelfordReduction.create(
        inner_fns=(loader,),
        reduction_type="welford_reduce",
        dtype=x.get_dtype(),
        **kwargs,
    )
    m2.realize()

    dtype = x.get_dtype()
    size = x.get_size()
    axis = _validate_reduction_axis(x, axis)
    rnumel = sympy_product(size[i] for i in axis)

    def get_constant_or_index_expr(x, dtype):
        # [FUNCTION DOC] HELPER: Return an ops constant for numbers, or an index expression converted to dtype for non-numeric sympy expressions.
        if isinstance(x, sympy.Expr) and not x.is_number:
            return ops.to_dtype(ops.index_expr(x, torch.int64), dtype)
        return ops.constant(x, dtype)

    def scale_fn(data):
        # [FUNCTION DOC] HELPER: Divide M2 by max(0, N - correction).
        c = get_constant_or_index_expr(correction, dtype)
        N = get_constant_or_index_expr(rnumel, dtype)
        zero = ops.constant(0, dtype)
        return data / ops.maximum(zero, N - c)

    var = make_pointwise(scale_fn)(m2)

    if return_mean:
        mean.realize()
        return var, mean
    return (var,)


def var_mean_helper_(x, *, axis, correction, keepdim, return_mean):
    # [FUNCTION DOC] HELPER: Shared implementation of var and var_mean: computes in the compute dtype, chooses the two-step or Welford algorithm (Welford unless MTIA disables it or the heuristic prefers two-step), and casts outputs back to the input dtype.
    # Returns the variance alone or (variance, mean) depending on return_mean.
    out_dtype = x.get_dtype()
    compute_dtype = get_computation_dtype(out_dtype)
    x = to_dtype(x, compute_dtype, copy=False)
    kwargs = dict(
        x=x,
        axis=axis,
        correction=correction,
        keepdim=keepdim,
        return_mean=return_mean,
    )
    output = (
        var_mean_sum_(**kwargs)
        if (
            config.mtia.disable_welford_reduction
            or use_two_step_variance(
                x, axis=axis, keepdim=keepdim, input_dtype=out_dtype
            )
        )
        else var_mean_welford_(**kwargs)
    )
    output = tuple(to_dtype(x, out_dtype, copy=False) for x in output)
    return output[0] if not return_mean else output


@register_lowering([aten.var, prims.var])
def var_(x, axis=None, *, correction=None, keepdim=False):
    # [FUNCTION DOC] Lowering for aten.var and prims.var: variance without the mean.
    return var_mean_helper_(
        x, axis=axis, correction=correction, keepdim=keepdim, return_mean=False
    )


@register_lowering(aten.var_mean)
def var_mean(x, axis=None, *, correction=None, keepdim=False):
    # [FUNCTION DOC] Lowering for aten.var_mean: returns (variance, mean).
    return var_mean_helper_(
        x, axis=axis, correction=correction, keepdim=keepdim, return_mean=True
    )


def pow_recursive(x, y, dtype):
    # [FUNCTION DOC] HELPER: Raise ops value x to an integer power y using repeated squaring; negative powers take the reciprocal first, zero gives a constant one.
    if y < 0:
        return pow_recursive(ops.reciprocal(x), -y, dtype)
    if y == 0:
        return ops.constant(1, dtype)
    if y == 1:
        return x

    result = pow_recursive(x, y // 2, dtype)
    result = ops.mul(result, result)
    if (y % 2) == 1:
        result = ops.mul(result, x)
    return result


@make_pointwise
def pow_native(a, b):
    # [FUNCTION DOC] Pointwise body for a native pow op.
    return ops.pow(a, b)


fallback_pow_tensor_tensor = fallback_handler(
    aten.pow.Tensor_Tensor, add_to_fallback_set=False
)
fallback_pow_scalar = fallback_handler(aten.pow.Scalar, add_to_fallback_set=False)
fallback_pow_tensor_scalar = fallback_handler(
    aten.pow.Tensor_Scalar, add_to_fallback_set=False
)


@register_lowering(aten.pow, broadcast=True)
def pow(a, b):
    # [FUNCTION DOC] Lowering for aten.pow. Integer-valued float exponents are converted to ints, 0.5 uses sqrt, 1 clones. Small fixed integer exponents (magnitude below 32, or non-negative for integer dtypes) are expanded into multiplications. A base of 1
    # gives ones and base 2 with float exponent uses exp2. Integer dtypes otherwise use eager fallbacks; the rest use a native pow Pointwise.
    if isinstance(b, float) and b.is_integer():
        return pow(a, int(b))
    elif isinstance(b, float) and b == 0.5:
        return sqrt(a)
    elif isinstance(b, int) and b == 1:
        return clone(a)

    # Type promotion ensures all tensor arguments have the same type
    dtype = next(x.get_dtype() for x in (a, b) if isinstance(x, ir.TensorBox))
    is_integer_pow = is_integer_dtype(dtype)

    # Optimize away small fixed powers, or for integers avoid falling back to ATen
    embed_exponent = isinstance(b, int) and (
        -32 < b < 32 or (is_integer_pow and b >= 0)
    )
    if embed_exponent:
        loader = a.make_loader()

        def fn(idx):
            # [FUNCTION DOC] HELPER: Load the base and raise it to the fixed integer exponent.
            return pow_recursive(loader(idx), b, a.get_dtype())

        return Pointwise.create(
            device=a.get_device(),
            dtype=a.get_dtype(),
            inner_fn=fn,
            ranges=a.get_size(),
        )

    if isinstance(a, Number):
        if a == 1:
            return full_like(b, 1)

        if a == 2 and is_float_dtype(b.get_dtype()):
            return exp2(b)

    if is_integer_pow:
        # ops.pow doesn't work for integers
        if isinstance(a, Number):
            return fallback_pow_scalar(a, b)
        elif isinstance(b, Number):
            return fallback_pow_tensor_scalar(a, b)
        else:
            return fallback_pow_tensor_tensor(a, b)

    return pow_native(a, b)


def mutate_to(changed, val, unsafe_alias=False):
    # [FUNCTION DOC] Make changed take the value val (in-place mutation semantics). Non-storage values are first copied through a Pointwise. If changed is a plain computed storage (not an input, module buffer, or no-op kernel), its data pointer is swung to val
    # (realized). Otherwise val is realized into the changed buffer through a mutation layout, optionally permitting aliasing. Returns changed.
    if isinstance(changed, TensorBox):
        changed_data = changed.data
    else:
        changed_data = changed
    if isinstance(val, TensorBox):
        val = val.data

    if not isinstance(val, ir.StorageBox):
        # introduce a copy to handle views
        node = Pointwise.create(
            device=changed.get_device(),
            dtype=changed.get_dtype(),
            inner_fn=val.make_loader(),
            ranges=changed.get_size(),
        )
        if not (isinstance(node, (BaseView, MutableBox))):
            raise AssertionError("expected: isinstance(node, (BaseView, MutableBox))")
        val = node.data
        if not (isinstance(val, ir.StorageBox)):
            raise AssertionError("expected: isinstance(val, ir.StorageBox)")

    if isinstance(changed_data, ir.StorageBox) and not (
        changed_data.is_input_buffer()
        # In AOTI, module parameters and buffers are not lifted as graph inputs
        or changed_data.is_module_buffer()
        or isinstance(changed_data.data, ir.NopKernel)
    ):
        # Fast path, just swing the data pointer
        val.realize()
        changed_data.data = val.data
        return changed

    ir.MutationLayoutSHOULDREMOVE.realize_into(
        val, changed_data, unsafe_alias=unsafe_alias
    )
    return changed


@register_lowering(aten.fill_)
def fill_(x, fill_value):
    # [FUNCTION DOC] Lowering for aten.fill_: mutate x to a tensor filled with fill_value.
    return mutate_to(x, full_like(x, fill_value))


@register_lowering(aten.copy_, type_promotion_kind=None)
def copy_(dst, src, non_blocking=False):
    # [FUNCTION DOC] Lowering for aten.copy_: returns dst unchanged if src is dst; otherwise moves src to dst's device and dtype, expands to dst's shape, and mutates dst to hold it.
    if dst is src:
        # dst.copy_(dst) can happen from the reinplacing pass
        return dst
    src = to_device(src, dst.get_device())
    src = to_dtype(src, dst.get_dtype())
    src = expand(src, dst.get_size())
    return mutate_to(dst, src)


@make_pointwise
def floordiv(a, b):
    # [FUNCTION DOC] Pointwise body for floor division on integer values.
    return ops.floordiv(a, b)


@make_pointwise
def truncdiv(a, b):
    # [FUNCTION DOC] Pointwise body for truncating division on integer values.
    return ops.truncdiv(a, b)


@make_pointwise
def _div_rn(a, b):
    # [FUNCTION DOC] Pointwise body for IEEE round-to-nearest division.
    return ops.div_rn(a, b)


def _floor_div_floating(a, b):
    # [FUNCTION DOC] HELPER: Floor division for floating operands with eager-matching edge cases: floor of the round-nearest quotient, NaN when a is infinite and b nonzero, and -1 when a is finite nonzero and b is infinite with differing signs. Either operand may be a python scalar.
    # Either operand may be a python scalar (e.g. torch.floor_divide(scalar,
    # tensor)); constant_like needs a tensor for dtype/device/size, so seed the
    # constants from whichever operand is a tensor.
    ref = a if isinstance(a, (TensorBox, IRNode)) else b
    nan = constant_like(float("nan"))(ref)
    neg_one = constant_like(-1.0)(ref)
    zero = constant_like(0.0)(ref)

    def fn(a, b, nan, neg_one, zero):
        # [FUNCTION DOC] HELPER: Pointwise body implementing the floating floor division with infinity corrections.
        quotient = ops.div_rn(a, b)
        result = ops.floor(quotient)
        a_is_inf = ops.isinf(a)
        a_is_finite = ops.logical_and(
            ops.logical_not(a_is_inf), ops.logical_not(ops.isnan(a))
        )
        # Eager floor division uses a fmod-based implementation. For +/-inf
        # divided by a nonzero divisor, fmod produces NaN. Keep divisor == 0
        # unchanged because eager returns a / b directly.
        a_inf_nonzero_b = ops.logical_and(a_is_inf, ops.ne(b, zero))

        # For finite nonzero a divided by +/-inf, floor(a / b) can produce
        # signed zero. Eager applies Python-style floor-division sign correction,
        # so negative zero direction results become -1.
        finite_nonzero_a_inf_b = ops.logical_and(
            ops.logical_and(a_is_finite, ops.ne(a, zero)),
            ops.logical_and(ops.isinf(b), ops.ne(ops.lt(a, zero), ops.lt(b, zero))),
        )
        result = ops.where(finite_nonzero_a_inf_b, neg_one, result)
        return ops.where(a_inf_nonzero_b, nan, result)

    return make_pointwise(fn)(a, b, nan, neg_one, zero)


@register_lowering(aten.div, broadcast=True)
def div_mode(a, b, rounding_mode=None):
    # [FUNCTION DOC] Lowering for aten.div with rounding_mode. floor: integer operands use floor division, otherwise the floating version; trunc: integer truncating division or trunc of the true quotient; none: true division. Both-boolean operands raise an error for floor/trunc.
    both_integer = is_integer_type(a) and is_integer_type(b)
    both_boolean = is_boolean_type(a) and is_boolean_type(b)

    # floordiv and truncdiv need special handling for integer tensors on Triton,
    # see the discussion at https://github.com/triton-lang/triton/issues/605
    if rounding_mode == "floor":
        if both_boolean:
            raise AssertionError(
                "floordiv operands can not be boolean at the same time"
            )
        # Use div_rn (IEEE round-to-nearest) instead of truediv here because
        # Triton's default division uses an approximate reciprocal, which can
        # produce a result slightly below the true quotient and cause floor()
        # to round down by one.
        return floordiv(a, b) if both_integer else _floor_div_floating(a, b)
    if rounding_mode == "trunc":
        if both_boolean:
            raise AssertionError(
                "truncdiv operands can not be boolean at the same time"
            )
        return truncdiv(a, b) if both_integer else trunc(div(a, b))
    return div(a, b)


@register_lowering([aten.mul], broadcast=True)
def mul(a, b):
    # [FUNCTION DOC] Lowering for aten.mul: logical and for two booleans, otherwise a pointwise multiply.
    both_bool = is_boolean_type(a) and is_boolean_type(b)
    if both_bool:
        return logical_and(a, b)
    else:
        fn = ops_wrapper(aten.mul.__name__)
        return make_pointwise(fn)(a, b)


def get_constant_value(x: ir.IRNode) -> ir.Constant | None:
    # [FUNCTION DOC] HELPER: Try to resolve an IR node to a Constant: unwrap boxes and views; for pointwise-like Loops evaluate the inner function with a constant-extracting handler and return the constant if the output is one. Returns None otherwise.
    """Try convert an arbitrary IR node into an ir.Constant value"""

    # First try unwrapping the IRNode to see if it is already an ir.Constant
    # Optional step, but avoids unnecessary inner_fn evaluation.
    if isinstance(x, ir.MutableBox):
        return get_constant_value(x.data)
    if isinstance(x, ir.BaseView):
        return get_constant_value(x.unwrap_view())
    if isinstance(x, ir.Constant):
        return x

    # If the unwrapped node is not an ir.Constant, try evaluating inner_fn
    # to see if the returned value is from an `ops.constant` call
    if not isinstance(x, ir.Loops):
        return None

    handler = torch._inductor.ops_handler.ExtractConstantsHandler(x.get_device())
    with (
        V.set_ops_handler(handler),
        patch.object(ir.FlexibleLayout, "allow_indexing", True),
    ):
        out = x.inner_fn(*x.inner_fn_args())

    if not (isinstance(out, torch._inductor.virtualized.OpsValue)):
        raise AssertionError(
            "expected: isinstance(out, torch._inductor.virtualized.OpsValue)"
        )
    if isinstance(out.value, ir.Constant):
        return out.value
    return None


# NOTE: prims.div maps to a / b in C, so performs truncation division on
#   integer inputs and true division for floating and complex inputs.
@register_lowering([prims.div], broadcast=True)
def div_prim(a, b):
    # [FUNCTION DOC] Lowering for prims.div (C-style division): integer or bool operands use truncating division. Otherwise, for non-CPU devices, division by a constant becomes multiplication by the reciprocal (infinity with sign for zero); else true division pointwise.
    is_integral = all(is_boolean_type(x) or is_integer_type(x) for x in [a, b])

    if is_integral:
        return truncdiv(a, b)

    # Disable CPU optimization to avoid precision issues.
    # see https://github.com/pytorch/pytorch/issues/157959
    if (divisor := get_constant_value(b)) is not None and a.get_device().type != "cpu":
        # Replace divide by constant with multiply by reciprocal

        if divisor.value == 0:
            reciprocal = math.copysign(float("inf"), divisor.value)
        else:
            reciprocal = 1.0 / divisor.value
        return mul(a, reciprocal)

    def fn(*args):
        # [FUNCTION DOC] HELPER: Pointwise body for true division.
        return ops.truediv(*args)

    return make_pointwise(fn)(a, b)


@register_lowering(
    [aten.true_divide, aten.div.Tensor],
    broadcast=True,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def div(a, b):
    # [FUNCTION DOC] Lowering for aten.true_divide and div.Tensor: promotes scalar constants with int-to-float promotion and applies prims.div.
    a, b = promote_constants(
        (a, b), type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )
    return div_prim(a, b)


@register_lowering([aten.fmod, prims.fmod], broadcast=True)
def fmod(a, b):
    # [FUNCTION DOC] Lowering for fmod: integer or bool inputs use the integer modulus op, otherwise the floating fmod op, as a pointwise.
    is_integral = is_boolean_type(a) or is_integer_type(a)

    if is_integral:

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Pointwise body using the integer mod op.
            return ops.mod(a, b)

    else:

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Pointwise body using the floating fmod op.
            return ops.fmod(a, b)

    return make_pointwise(fn)(a, b)


def _strict_sum_layout_eligible(axis, dtype) -> bool:
    # [FUNCTION DOC] HELPER: Decide whether the current aten.sum.dim_IntList node should use the strict-ordered sum. Requires strict numerics config, triton reduction ordering support, no dtype override, a single reduce dim, a floating CUDA (non-ROCm) triton input,
    # and a statically known input layout where the reduced dim has unit stride (or is size one) and the remaining non-trivial dims have collapsible strides greater than one. Returns a bool; False when there is no current node.
    current_node = V.graph.current_node
    if (
        current_node is None
        or config.numerics != "strict"
        or current_node.target != aten.sum.dim_IntList
        or dtype is not None
        or axis is None
        or not has_triton_reduction_ordering()
    ):
        return False
    dims = [axis] if isinstance(axis, int) else list(axis)
    if len(dims) != 1:
        return False

    fx_input = current_node.args[0]
    if not isinstance(fx_input, torch.fx.Node):
        return False
    value = fx_input.meta.get("val")
    if not isinstance(value, torch.Tensor) or value.dtype not in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }:
        return False
    if (
        not value.is_cuda
        or torch.version.hip is not None
        or not is_triton(value.device)
    ):
        return False

    dim = dims[0] + value.ndim if dims[0] < 0 else dims[0]
    if not 0 <= dim < value.ndim:
        return False
    sizes = [convert_symint_to_expr(size) for size in value.shape]
    strides = [convert_symint_to_expr(stride) for stride in value.stride()]
    sizevars = V.graph.sizevars
    outer = [
        (size, stride)
        for i, (size, stride) in enumerate(zip(sizes, strides))
        if i != dim and not sizevars.statically_known_equals(size, 1)
    ]
    collapsible = all(
        sizevars.statically_known_equals(slow_stride, fast_stride * fast_size)
        for (_, slow_stride), (fast_size, fast_stride) in itertools.pairwise(outer)
    )
    if sizevars.statically_known_equals(sizes[dim], 1):
        if not outer:
            return sizevars.statically_known_equals(strides[0], 1) and all(
                sizevars.statically_known_equals(stride, 0)
                or sizevars.statically_known_equals(stride, 1)
                for stride in strides[1:]
            )
        return collapsible and sizevars.statically_known_equals(outer[-1][1], 1)
    if not sizevars.statically_known_equals(strides[dim], 1):
        return False
    if any(
        not sizevars.statically_known_true(sympy.Gt(stride, 1)) for _, stride in outer
    ):
        return False
    return collapsible


@register_lowering([aten.sum, prims.sum])
def sum_(x, axis=None, keepdims=False, *, dtype=None):
    # [FUNCTION DOC] Lowering for aten.sum/prims.sum: integer and bool inputs sum into int64 when no dtype is given; uses a sum reduction with the strict-order flag from the layout check. Returns the reduced TensorBox.
    strict_sum = _strict_sum_layout_eligible(axis, dtype)
    if (
        is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
    ) and dtype is None:
        dtype = torch.int64

    fn = make_reduction("sum", override_return_dtype=dtype, strict_sum=strict_sum)
    return fn(x, axis, keepdims, dtype=dtype)


fallback_cumsum = fallback_handler(aten.cumsum.default)
fallback_cumprod = fallback_handler(aten.cumprod.default)
fallback_logcumsumexp = fallback_handler(aten.logcumsumexp.default)
fallback_cummax = fallback_handler(aten.cummax.default)
fallback_cummin = fallback_handler(aten.cummin.default)


@register_lowering(aten.cumsum)
def cumsum(x, axis=None, dtype=None):
    # [FUNCTION DOC] Lowering for aten.cumsum: integer/bool default to int64; 0-d input only allows axis 0 or -1 (else raises an error) and returns a cast copy. Otherwise a Scan with an add combine; if the scan cannot be generated, falls back to the eager kernel.
    if (
        is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
    ) and dtype is None:
        dtype = torch.int64

    if len(x.get_size()) == 0:
        if axis not in [0, -1]:
            raise AssertionError("expected: axis in [0, -1]")
        dtype = dtype or x.get_dtype()
        return to_dtype(x, dtype, copy=True)

    def combine_fn(a_tuple, b_tuple):
        # [FUNCTION DOC] HELPER: Scan combine function: elementwise add of the single carried values.
        (a,) = a_tuple
        (b,) = b_tuple
        return (ops.add(a, b),)

    kwargs = _make_scan_inner(x, axis=axis, dtype=dtype)
    (result,) = ir.Scan.create(**kwargs, combine_fn=combine_fn)
    if result is None:
        return fallback_cumsum(x, dim=axis, dtype=dtype)
    return result


@register_lowering(aten.cumprod)
def cumprod(x, axis=None, dtype=None):
    # [FUNCTION DOC] Lowering for aten.cumprod: same structure as cumsum with a multiply combine function and eager fallback.
    if (
        is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
    ) and dtype is None:
        dtype = torch.int64

    if len(x.get_size()) == 0:
        if axis not in [0, -1]:
            raise AssertionError("expected: axis in [0, -1]")
        dtype = dtype or x.get_dtype()
        return to_dtype(x, dtype, copy=True)

    def combine_fn(a_tuple, b_tuple):
        # [FUNCTION DOC] HELPER: Scan combine function: elementwise multiply of the single carried values.
        (a,) = a_tuple
        (b,) = b_tuple
        return (ops.mul(a, b),)

    kwargs = _make_scan_inner(x, axis=axis, dtype=dtype)
    (result,) = ir.Scan.create(**kwargs, combine_fn=combine_fn)
    if result is None:
        return fallback_cumprod(x, dim=axis, dtype=dtype)
    return result


@register_lowering(aten.logcumsumexp)
def logcumsumexp(x, dim):
    # [FUNCTION DOC] Lowering for aten.logcumsumexp: 0-d input clones (dim must be 0 or -1); otherwise a Scan with a numerically stable log-add-exp combine, falling back to eager if the scan is unavailable.
    def log_add_exp_helper(a_tuple, b_tuple):
        # [FUNCTION DOC] HELPER: Scan combine function computing log(exp(a) + exp(b)) stably using the maximum and log1p, handling equal infinities.
        (a,) = a_tuple
        (b,) = b_tuple
        min_v = ops.minimum(a, b)
        max_v = ops.maximum(a, b)
        mask = (min_v != max_v) | (~ops.isinf(min_v))
        return (ops.where(mask, ops.log1p(ops.exp(min_v - max_v)) + max_v, a),)

    dtype = x.get_dtype()
    if len(x.get_size()) == 0:
        if dim not in [0, -1]:
            raise AssertionError("expected: dim in [0, -1]")
        return clone(x)

    kwargs = _make_scan_inner(x, axis=dim, dtype=dtype)
    (result,) = ir.Scan.create(**kwargs, combine_fn=log_add_exp_helper)
    if result is None:
        return fallback_logcumsumexp(x, dim=dim)
    return result


@register_lowering(aten.cummax, type_promotion_kind=None)
def cummax(x, axis=None):
    # [FUNCTION DOC] Lowering for aten.cummax returning (values, indices). 0-d input returns a clone and an empty index tensor. Otherwise a two-output Scan carrying value and int64 index with an argmax combine (ties go to the later element); eager fallback if the scan is unavailable.
    if len(x.get_size()) == 0:
        if axis not in [0, -1]:
            raise AssertionError("expected: axis in [0, -1]")
        return clone(x), empty_like(x, dtype=torch.int64)

    dtype = x.get_dtype()
    combine_fn = ir.get_reduction_combine_fn(
        "argmax", dtype=dtype, arg_break_ties_left=False
    )

    kwargs = _make_scan_inner(x, axis=axis, dtype=dtype)
    kwargs["dtypes"] = (dtype, torch.int64)
    kwargs["inner_fns"] = (
        x.make_loader(),
        lambda idx: ops.index_expr(idx[axis], torch.int64),
    )
    values, indices = ir.Scan.create(**kwargs, combine_fn=combine_fn)  # type: ignore[arg-type]
    if values is None:
        return fallback_cummax(x, dim=axis)
    return values, indices


@register_lowering(aten.cummin, type_promotion_kind=None)
def cummin(x, axis=None):
    # [FUNCTION DOC] Lowering for aten.cummin returning (values, indices): same as cummax with an argmin combine.
    if len(x.get_size()) == 0:
        if axis not in [0, -1]:
            raise AssertionError("expected: axis in [0, -1]")
        return clone(x), empty_like(x, dtype=torch.int64)

    dtype = x.get_dtype()
    combine_fn = ir.get_reduction_combine_fn(
        "argmin", dtype=dtype, arg_break_ties_left=False
    )

    kwargs = _make_scan_inner(x, axis=axis, dtype=dtype)
    kwargs["dtypes"] = (dtype, torch.int64)
    kwargs["inner_fns"] = (
        x.make_loader(),
        lambda idx: ops.index_expr(idx[axis], torch.int64),
    )
    values, indices = ir.Scan.create(**kwargs, combine_fn=combine_fn)  # type: ignore[arg-type]
    if values is None:
        return fallback_cummin(x, dim=axis)
    return values, indices


@register_lowering(aten.prod)
def prod(x, axis=None, keepdims=False, *, dtype=None):
    # [FUNCTION DOC] Lowering for aten.prod: integer/bool inputs accumulate into int64; uses a prod reduction.
    if (
        is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
    ) and dtype is None:
        dtype = torch.int64

    fn = make_reduction("prod", override_return_dtype=dtype)
    return fn(x, axis, keepdims, dtype=dtype)


def _current_node_uses_all_tuple_outputs(indices: tuple[int, ...]) -> bool:
    # [FUNCTION DOC] HELPER: True if every listed tuple output index of the current node is consumed by a getitem user (or if usage cannot be determined, such as no current node or a non-getitem user).
    current_node = V.graph.current_node
    if current_node is None:
        return True

    used_indices: OrderedSet[int] = OrderedSet()
    for user in current_node.users:
        if (
            user.op == "call_function"
            and user.target is operator.getitem
            and len(user.args) >= 2
            and isinstance(user.args[1], int)
        ):
            if len(user.users) > 0:
                used_indices.add(user.args[1])
        else:
            return True

    return all(index in used_indices for index in indices)


@register_lowering(aten.any)
def reduce_any(x, dim=None, keepdim=False):
    # [FUNCTION DOC] Lowering for aten.any: casts to bool and applies an any reduction.
    x = to_dtype(x, torch.bool)
    return make_reduction("any")(x, axis=dim, keepdims=keepdim)


@register_lowering(aten.max, type_promotion_kind=None)
def reduce_max(x, dim=None, keepdim=False):
    # [FUNCTION DOC] Lowering for aten.max. With no dim returns the full max reduction. With a dim returns (values, indices): on triton for non-bool inputs it uses a fused arg-with-value reduction when both outputs are used, or separate value and arg reductions; otherwise separate max and argmax reductions.
    if dim is not None:
        if is_triton(x) and x.get_dtype() != torch.bool:
            if _current_node_uses_all_tuple_outputs((0, 1)):
                return reduce_argmax_with_value(x, axis=dim, keepdims=keepdim)
            return (
                reduce_argmax_value(x, axis=dim, keepdims=keepdim),
                reduce_argmax(x, axis=dim, keepdims=keepdim),
            )
        return (
            reduce_amax(x, axis=dim, keepdims=keepdim),
            reduce_argmax(x, axis=dim, keepdims=keepdim),
        )

    return reduce_amax(x, axis=None, keepdims=keepdim)


@register_lowering(aten.min, type_promotion_kind=None)
def reduce_min(x, dim=None, keepdim=False):
    # [FUNCTION DOC] Lowering for aten.min: same structure as max using min and argmin reductions.
    if dim is not None:
        if is_triton(x) and x.get_dtype() != torch.bool:
            if _current_node_uses_all_tuple_outputs((0, 1)):
                return reduce_argmin_with_value(x, axis=dim, keepdims=keepdim)
            return (
                reduce_argmin_value(x, axis=dim, keepdims=keepdim),
                reduce_argmin(x, axis=dim, keepdims=keepdim),
            )
        return (
            reduce_amin(x, axis=dim, keepdims=keepdim),
            reduce_argmin(x, axis=dim, keepdims=keepdim),
        )

    return reduce_amin(x, axis=None, keepdims=keepdim)


register_lowering(prims.xor_sum)(make_reduction("xor_sum"))
reduce_amax = register_lowering(aten.amax)(make_reduction("max"))
reduce_amin = register_lowering(aten.amin)(make_reduction("min"))
reduce_argmax = register_lowering(aten.argmax)(
    make_reduction("argmax", override_return_dtype=torch.int64)
)
reduce_argmin = register_lowering(aten.argmin)(
    make_reduction("argmin", override_return_dtype=torch.int64)
)
reduce_argmax_value = make_reduction("argmax_value")
reduce_argmin_value = make_reduction("argmin_value")
reduce_argmax_with_value = make_arg_with_value_reduction("argmax_with_value")
reduce_argmin_with_value = make_arg_with_value_reduction("argmin_with_value")

add = register_pointwise(
    aten.add,
    allow_alpha=True,
    use_fma_for_alpha=True,
    override_fn_when_input_bool="logical_or",
    round_scalars_to_tensor_dtype=True,
)

sort_fallback = fallback_handler(aten.sort.stable, add_to_fallback_set=False)


@register_lowering(aten.sort.stable, type_promotion_kind=None)
def sort_stable(x, *, stable=None, dim=-1, descending=False):
    # [FUNCTION DOC] Lowering for aten.sort.stable returning (values, int64 indices). 0-d input returns a clone with zero index. Uses an index iota of int16 (int32 if decompose_sort_ops is on) broadcast along the sort dim; falls back to the eager sort when the dim
    # may be too large for the index dtype or the sort IR node cannot be generated. Otherwise creates a Sort node.
    if stable is None:
        stable = False

    shape = x.get_size()
    device = x.get_device()
    dim = canonicalize_dim(len(shape), dim)
    if len(shape) == 0:
        return clone(x), _full(0, device, torch.int64, shape)

    dim_size = shape[dim] if len(shape) else 1
    # Use int32 indices when decompose_sort_ops is enabled, allowing sort
    # dimensions up to 2^31-1.  Default int16 keeps register pressure low
    # on GPU where the bitonic network holds all indices in-block.
    if config.triton.decompose_sort_ops:
        idx_dtype = torch.int32
    else:
        idx_dtype = torch.int16
    if not V.graph.sizevars.guard_or_false(
        sympy.Lt(dim_size, torch.iinfo(idx_dtype).max)
    ):
        return sort_fallback(x, stable=stable, dim=dim, descending=descending)

    indices = iota(
        dim_size, start=0, step=1, dtype=idx_dtype, device=device, requires_grad=False
    )
    view_shape = [1] * len(shape)
    if len(shape):
        view_shape[dim] = dim_size
    indices = view(indices, view_shape)
    indices = expand(indices, shape)

    values, indices = ir.Sort.create(
        device=device,
        dtypes=(x.dtype, indices.dtype),
        inner_fns=(x.make_loader(), indices.make_loader()),
        size=shape,
        axis=dim,
        stable=stable,
        descending=descending,
    )
    if values is None:
        return sort_fallback(x, stable=stable, dim=dim, descending=descending)

    if indices is None:
        raise AssertionError("expected: indices is not None")
    return values, to_dtype(indices, torch.int64)


@register_lowering(aten.sort.default, type_promotion_kind=None)
def sort(x, dim=-1, descending=False):
    # [FUNCTION DOC] Lowering for aten.sort.default: unstable sort via sort_stable.
    return sort_stable(x, stable=False, dim=dim, descending=descending)


# Sort-based op lowerings
# When config.triton.decompose_sort_ops is enabled, decompose into sort-based
# ops so Inductor generates Triton kernels via ir.Sort.
# Otherwise, fall back to ATen eager.
topk_fallback = fallback_handler(aten.topk.default, add_to_fallback_set=False)
kthvalue_fallback = fallback_handler(aten.kthvalue.default, add_to_fallback_set=False)
median_fallback = fallback_handler(aten.median.default, add_to_fallback_set=False)
median_dim_fallback = fallback_handler(aten.median.dim, add_to_fallback_set=False)
mode_fallback = fallback_handler(aten.mode.default, add_to_fallback_set=False)

# sort/sort.stable already have register_lowering above (sort_stable, sort).
# They use ir.Sort directly and fall back when the dimension is too large.
# When decompose_sort_ops is enabled, the size limit is lifted (int32 indices).


@register_lowering(aten.median.default, type_promotion_kind=None)
def median_default(self):
    # [FUNCTION DOC] Lowering for aten.median.default: uses the eager fallback unless sort decomposition is enabled, in which case flattens, sorts, and selects the element at position (numel - 1) // 2.
    if not config.triton.decompose_sort_ops:
        return median_fallback(self)
    size = self.get_size()
    numel = functools.reduce(operator.mul, size, sympy.Integer(1))
    flat = view(self, [numel])
    sorted_vals, _ = sort_stable(flat, dim=0)
    k = (numel - 1) // 2
    return select(sorted_vals, 0, k)


@register_lowering(aten.median.dim, type_promotion_kind=None)
def median_dim(self, dim, keepdim=False):
    # [FUNCTION DOC] Lowering for aten.median.dim returning (values, indices): eager fallback unless sort decomposition is enabled; otherwise stable sort along dim and select position (n - 1) // 2 (lower median), optionally keeping the dim.
    if not config.triton.decompose_sort_ops:
        return median_dim_fallback(self, dim, keepdim)
    shape = self.get_size()
    ndim = len(shape)
    if ndim == 0:
        return clone(self), _full(0, self.get_device(), torch.int64, shape)
    dim = canonicalize_dim(ndim, dim)
    sorted_vals, sorted_idxs = sort_stable(self, stable=True, dim=dim)
    n = shape[dim]
    k = (n - 1) // 2
    values = select(sorted_vals, dim, k)
    indices = select(sorted_idxs, dim, k)
    if keepdim:
        values = unsqueeze(values, dim)
        indices = unsqueeze(indices, dim)
    return values, indices


@register_lowering(aten.mode.default, type_promotion_kind=None)
def mode_default(self, dim=-1, keepdim=False):
    # [FUNCTION DOC] Lowering for aten.mode: eager fallback unless sort decomposition is enabled. Otherwise sorts along dim, marks run boundaries where the sorted value changes, propagates the last boundary with a running maximum to get each position's run length,
    # and picks the position of the first longest run (argmax) to gather mode value and index. 0-d input returns a clone and zero index. Returns (values, indices), squeezed unless keepdim.
    """Lower aten.mode via sort-based decomposition or fallback."""
    if not config.triton.decompose_sort_ops:
        return mode_fallback(self, dim, keepdim)
    shape = self.get_size()
    ndim = len(shape)
    device = self.get_device()
    if ndim == 0:
        return clone(self), _full(0, device, torch.int64, shape)
    dim = canonicalize_dim(ndim, dim)
    sorted_vals, sorted_idxs = sort_stable(self, stable=True, dim=dim)
    n = shape[dim]

    # Position indices along dim: [0, 1, ..., n-1]
    positions = iota(
        n, start=0, step=1, dtype=torch.int64, device=device, requires_grad=False
    )
    pos_view_shape = [sympy.Integer(1)] * ndim
    pos_view_shape[dim] = n
    positions = view(positions, pos_view_shape)
    positions = expand(positions, shape)

    # Shift positions by -1, clamp to 0 for position 0
    positions_loader0 = positions.make_loader()

    def prev_pos_fn(idx):
        # [FUNCTION DOC] HELPER: Position minus one clamped at zero, used to look at the previous sorted element.
        return ops.maximum(
            ops.sub(positions_loader0(idx), ops.constant(1, torch.int64)),
            ops.constant(0, torch.int64),
        )

    prev_positions = Pointwise.create(
        device=decode_device(device),
        dtype=torch.int64,
        inner_fn=prev_pos_fn,
        ranges=shape,
    )

    # Gather shifted values and compare for run boundaries
    shifted_vals = gather(sorted_vals, dim, prev_positions)

    sorted_loader = sorted_vals.make_loader()
    shifted_loader = shifted_vals.make_loader()
    positions_loader = positions.make_loader()

    # is_boundary = (sorted != shifted) | (position == 0)
    def is_boundary_fn(idx):
        # [FUNCTION DOC] HELPER: True where the sorted value differs from the previous one or at position zero.
        return ops.or_(
            ops.ne(sorted_loader(idx), shifted_loader(idx)),
            ops.eq(positions_loader(idx), ops.constant(0, torch.int64)),
        )

    is_boundary = Pointwise.create(
        device=decode_device(device),
        dtype=torch.bool,
        inner_fn=is_boundary_fn,
        ranges=shape,
    )

    # boundary_pos = where(is_boundary, position, -1)
    is_boundary_loader = is_boundary.make_loader()
    positions_loader2 = positions.make_loader()

    def boundary_pos_fn(idx):
        # [FUNCTION DOC] HELPER: Position where a run boundary occurs, else -1.
        return ops.where(
            is_boundary_loader(idx),
            positions_loader2(idx),
            ops.constant(-1, torch.int64),
        )

    boundary_pos = Pointwise.create(
        device=decode_device(device),
        dtype=torch.int64,
        inner_fn=boundary_pos_fn,
        ranges=shape,
    )

    # Propagate boundary positions forward with cummax
    last_boundary, _ = cummax(boundary_pos, dim)

    # run_len = position - last_boundary + 1
    positions_loader3 = positions.make_loader()
    last_boundary_loader = last_boundary.make_loader()

    def run_len_fn(idx):
        # [FUNCTION DOC] HELPER: Run length so far: position minus last boundary position plus one.
        return ops.add(
            ops.sub(positions_loader3(idx), last_boundary_loader(idx)),
            ops.constant(1, torch.int64),
        )

    run_len = Pointwise.create(
        device=decode_device(device),
        dtype=torch.int64,
        inner_fn=run_len_fn,
        ranges=shape,
    )

    # argmax returns first maximum -> end of leftmost longest run
    max_pos = reduce_argmax(run_len, axis=dim, keepdims=True)
    mode_vals = gather(sorted_vals, dim, max_pos)
    mode_idxs = gather(sorted_idxs, dim, max_pos)

    if not keepdim:
        mode_vals = squeeze(mode_vals, dim)
        mode_idxs = squeeze(mode_idxs, dim)

    return mode_vals, mode_idxs


@register_lowering(aten.topk.default, type_promotion_kind=None)
def topk(self, k, dim=-1, largest=True, sorted=True):
    # [FUNCTION DOC] Lowering for aten.topk: eager fallback unless sort decomposition is enabled; otherwise a stable sort along dim (descending if largest) and the first k entries of values and indices. 0-d input returns a clone and zero index. The sorted flag is ignored.
    if not config.triton.decompose_sort_ops:
        return topk_fallback(self, k, dim, largest, sorted)
    shape = self.get_size()
    ndim = len(shape)
    if ndim == 0:
        return clone(self), _full(0, self.get_device(), torch.int64, shape)
    dim = canonicalize_dim(ndim, dim)
    sorted_vals, sorted_idxs = sort_stable(
        self, stable=True, dim=dim, descending=largest
    )
    values = slice_(sorted_vals, dim, 0, k)
    indices = slice_(sorted_idxs, dim, 0, k)
    return values, indices


@register_lowering(aten.kthvalue.default, type_promotion_kind=None)
def kthvalue(self, k, dim=-1, keepdim=False):
    # [FUNCTION DOC] Lowering for aten.kthvalue: eager fallback unless sort decomposition is enabled; otherwise stable sort along dim and select the (k-1)th entry (k is 1-based) of values and indices, optionally keeping the dim.
    if not config.triton.decompose_sort_ops:
        return kthvalue_fallback(self, k, dim, keepdim)
    shape = self.get_size()
    ndim = len(shape)
    if ndim == 0:
        return clone(self), _full(0, self.get_device(), torch.int64, shape)
    dim = canonicalize_dim(ndim, dim)
    sorted_vals, sorted_idxs = sort_stable(self, stable=True, dim=dim)
    # k is 1-based
    values = select(sorted_vals, dim, k - 1)
    indices = select(sorted_idxs, dim, k - 1)
    if keepdim:
        values = unsqueeze(values, dim)
        indices = unsqueeze(indices, dim)
    return values, indices


def register_pointwise_numeric(op, name=None, triton_fallback=None):
    # [FUNCTION DOC] HELPER: Register a pointwise lowering with int-to-float type promotion and an optional triton fallback.
    return register_pointwise(
        op,
        name=name,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        triton_fallback=triton_fallback,
    )


def register_pointwise_numeric_ldf64(op: torch._ops.OpOverloadPacket):
    # [FUNCTION DOC] HELPER: Register an int-to-float pointwise lowering for an op that needs libdevice fp64 support; records that requirement for the op name.
    register_op_requires_libdevice_fp64(op.__name__)
    return register_pointwise(
        op,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    )


rsqrt = register_pointwise_numeric(aten.rsqrt)
exp = register_pointwise_numeric_ldf64(aten.exp)
exp2 = register_pointwise_numeric(aten.exp2)
expm1 = register_pointwise_numeric(aten.expm1)
relu = register_pointwise(aten.relu)
sigmoid = register_pointwise_numeric_ldf64(aten.sigmoid)
sqrt = register_pointwise_numeric_ldf64(aten.sqrt)
square = register_pointwise(aten.square)
sub = register_pointwise(aten.sub, allow_alpha=True, round_scalars_to_tensor_dtype=True)


@register_lowering(aten.addcmul, broadcast=True)
def addcmul(self, tensor1, tensor2, *, value=1):
    # [FUNCTION DOC] Lowering for aten.addcmul: self + value * tensor1 * tensor2 as a Pointwise in the promoted dtype. On CUDA/XPU floating types uses fused multiply-add matching eager order (product rounded first with mul_rn, then fma with value);
    # value of one uses a single fma. Other dtypes use plain multiply and add. value may be a number or a sympy expression.
    """
    Computes self + value * tensor1 * tensor2 using FMA for better precision.

    Matches eager CUDA kernel order: self + value * (tensor1 * tensor2)
    This is computed as: fma(value, tensor1 * tensor2, self)

    Note: FMA is used for floating-point types on CUDA and ROCm (via tl.fma). For
    integer types we fall back to regular arithmetic since FMA doesn't support integers.

    For floating-point types, we use mul_rn (round-to-nearest multiplication)
    to force rounding of the product before the FMA. This prevents Triton's
    compiler from fusing the multiplication with the FMA, matching eager's
    rounding behavior. On ROCm, mul_rn falls back to regular multiplication since
    libdevice.mul_rn is not available; the FMA itself (tl.fma) is still used.
    """
    dtype = get_promoted_dtype(
        self,
        tensor1,
        tensor2,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    )

    self_loader = self.make_loader()
    t1_loader = tensor1.make_loader()
    t2_loader = tensor2.make_loader()

    device = self.get_device()
    use_fma = (
        dtype.is_floating_point
        and device is not None
        and device.type in ["cuda", "xpu"]
    )

    def inner_fn(idx):
        # [FUNCTION DOC] HELPER: Compute self + value * (t1 * t2) for one element using the fma or plain arithmetic path.
        self_val = self_loader(idx)
        t1_val = t1_loader(idx)
        t2_val = t2_loader(idx)

        if value == 1 and use_fma:
            return ops.fma(t1_val, t2_val, self_val)

        # Match eager order: self + value * (tensor1 * tensor2)
        # Compute tensor1 * tensor2 first
        if use_fma:
            # Use mul_rn to force rounding of the product, preventing Triton
            # from fusing t1*t2 with the subsequent FMA
            t1_times_t2 = ops.mul_rn(t1_val, t2_val)
        else:
            t1_times_t2 = ops.mul(t1_val, t2_val)

        # Use index_expr for sympy expressions (e.g., from .item()), constant otherwise
        if isinstance(value, sympy.Basic):
            value_expr = ops.index_expr(value, dtype)
        else:
            value_expr = ops.constant(value, dtype)

        if use_fma:
            # Use FMA for floating-point types for better precision
            return ops.fma(value_expr, t1_times_t2, self_val)
        else:
            # Fall back to regular arithmetic for integer types
            return ops.add(self_val, ops.mul(value_expr, t1_times_t2))

    return Pointwise.create(
        device=self.get_device(),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=self.get_size(),
    )


@register_lowering(aten.addcdiv, broadcast=True)
def addcdiv(self, tensor1, tensor2, *, value=1):
    # [FUNCTION DOC] Lowering for aten.addcdiv: self + value * (tensor1 / tensor2) in the int-to-float promoted dtype. Uses round-nearest division on CUDA/XPU floating types and fma when value is not one; value of one adds the quotient; other dtypes use true division and plain arithmetic.
    """
    Computes self + value * (tensor1 / tensor2) using FMA for better precision.

    Matches eager CUDA kernel order: self + value * (tensor1 / tensor2)
    This is computed as: fma(value, tensor1 / tensor2, self)

    For value=1: self + tensor1 / tensor2 (no FMA needed, just add the division)
    For value!=1: fma(value, div_rn(tensor1, tensor2), self)

    Note: FMA is used for floating-point types on CUDA and ROCm (via tl.fma). For
    integer types we fall back to regular arithmetic since FMA doesn't support integers.

    We use div_rn (round-to-nearest division) to force proper rounding, preventing
    Triton from fusing operations in ways that change the rounding behavior.
    div_rn emits triton.language.div_rn on both CUDA and ROCm (unlike mul_rn, which
    falls back to regular multiplication on ROCm where libdevice.mul_rn is unavailable).
    """
    dtype = get_promoted_dtype(
        self,
        tensor1,
        tensor2,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    )

    self_loader = self.make_loader()
    t1_loader = tensor1.make_loader()
    t2_loader = tensor2.make_loader()

    device = self.get_device()
    use_fma = (
        dtype.is_floating_point
        and device is not None
        and device.type in ["cuda", "xpu"]
    )

    def inner_fn(idx):
        # [FUNCTION DOC] HELPER: Compute self + value * (t1 / t2) for one element using the fma or plain arithmetic path.
        self_val = self_loader(idx)
        t1_val = t1_loader(idx)
        t2_val = t2_loader(idx)

        if use_fma:
            t1_div_t2 = ops.div_rn(t1_val, t2_val)
        else:
            t1_div_t2 = ops.truediv(t1_val, t2_val)

        if value == 1:
            return ops.add(self_val, t1_div_t2)

        # Use index_expr for sympy expressions (e.g., from .item()), constant otherwise
        if isinstance(value, sympy.Basic):
            value_expr = ops.index_expr(value, dtype)
        else:
            value_expr = ops.constant(value, dtype)

        if use_fma:
            return ops.fma(value_expr, t1_div_t2, self_val)
        else:
            return ops.add(self_val, ops.mul(value_expr, t1_div_t2))

    return Pointwise.create(
        device=self.get_device(),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=self.get_size(),
    )


_foreach_addcmul_scalar = register_foreach_pointwise(
    aten._foreach_addcmul.Scalar, addcmul, allow_alpha=True, scalar_kwarg="value"
)
_foreach_addcdiv_scalar = register_foreach_pointwise(
    aten._foreach_addcdiv.Scalar, addcdiv, allow_alpha=True, scalar_kwarg="value"
)


register_pointwise_numeric_ldf64(aten.cos)
register_pointwise_numeric_ldf64(aten.sin)
abs = register_pointwise(aten.abs)
bitwise_and = register_pointwise(aten.bitwise_and)
bitwise_left_shift = register_pointwise(aten.bitwise_left_shift)
bitwise_not = register_pointwise(
    aten.bitwise_not, override_fn_when_input_bool="logical_not"
)
bitwise_or = register_pointwise(aten.bitwise_or)
bitwise_right_shift = register_pointwise(aten.bitwise_right_shift)
bitwise_xor = register_pointwise(aten.bitwise_xor)
register_pointwise_numeric(aten.lgamma)
erf = register_pointwise_numeric(aten.erf)
register_lowering(
    aten.special_erf, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)(erf)

register_pointwise_numeric(aten.log1p)
register_pointwise_numeric(aten.tan)
register_pointwise_numeric(aten.tanh)
register_pointwise_numeric_ldf64(aten.log)
logical_and = register_pointwise(
    aten.logical_and,
    type_promotion_kind=None,
    convert_input_to_bool=True,
    override_return_dtype=torch.bool,
)
logical_not = register_pointwise(
    aten.logical_not,
    type_promotion_kind=None,
    convert_input_to_bool=True,
    override_return_dtype=torch.bool,
)
logical_or = register_pointwise(
    aten.logical_or,
    type_promotion_kind=None,
    convert_input_to_bool=True,
    override_return_dtype=torch.bool,
)
logical_xor = register_pointwise(
    aten.logical_xor,
    type_promotion_kind=None,
    convert_input_to_bool=True,
    override_return_dtype=torch.bool,
)
maximum = register_pointwise(aten.maximum)
minimum = register_pointwise(aten.minimum)
register_lowering(aten.clamp_min)(maximum)
register_lowering(aten.clamp_max)(minimum)
register_op_dtype_propagation_rules(
    "fmaximum",
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    override_return_dtype=None,
)
neg = register_pointwise(aten.neg)
abs = register_pointwise(aten.abs)
reciprocal = register_pointwise_numeric(aten.reciprocal)

register_op_dtype_propagation_rules(
    "remainder",
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    override_return_dtype=None,
)


@register_lowering(aten.remainder, broadcast=True)
def remainder(a, b):
    # [FUNCTION DOC] Lowering for aten.remainder: promotes scalar constants (rounded to low-precision tensor dtype) and applies the pointwise remainder op.
    a, b = promote_constants((a, b), round_scalar_constants=True)
    fn = ops_wrapper("remainder")
    return make_pointwise(fn)(a, b)


sign = register_pointwise(aten.sign, override_fn_when_input_bool="identity")
register_pointwise(aten.ceil)


@register_lowering(aten.signbit)
def signbit(x):
    # [FUNCTION DOC] Lowering for aten.signbit: all-False bool tensor for bool and unsigned integer dtypes, otherwise a pointwise signbit producing bool.
    if x.get_dtype() in (
        torch.bool,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    ):
        return full_like(x, False, dtype=torch.bool)
    fn = ops_wrapper("signbit")
    return make_pointwise(fn, override_return_dtype=torch.bool)(x)


register_lowering(aten._neg_view)(neg)

register_pointwise(aten.le, override_return_dtype=torch.bool)
register_pointwise(aten.lt, override_return_dtype=torch.bool)
register_pointwise(aten.ge, override_return_dtype=torch.bool)
gt = register_pointwise(aten.gt, override_return_dtype=torch.bool)
register_pointwise(aten.eq, override_return_dtype=torch.bool)
register_pointwise(aten.ne, override_return_dtype=torch.bool)

register_pointwise_numeric(aten.cosh)
register_pointwise_numeric(aten.sinh)
register_pointwise_numeric(aten.acos)
register_pointwise_numeric(aten.acosh)
register_pointwise_numeric(aten.asin)
register_pointwise_numeric(aten.asinh)
register_pointwise_numeric(aten.atan2)
register_pointwise_numeric(aten.atan)
register_pointwise_numeric(aten.atanh)
register_pointwise_numeric(aten.copysign)
register_pointwise_numeric(aten.erfc)
register_pointwise_numeric(aten.erfinv)
register_pointwise_numeric(aten.hypot)
register_pointwise_numeric(aten.log10)
register_pointwise_numeric(aten.log2)
register_pointwise_numeric(aten.nextafter)

from .codegen.common import BackendFeature, pointwise_overrides_data


def _get_pointwise_overrides(ns, name):
    # [FUNCTION DOC] HELPER: Generator yielding (op, type promotion kind, triton fallback) triples for the pointwise override named name in namespace ns, one per overload of a packet. Yields nothing if the op does not exist in the namespace.
    data = pointwise_overrides_data[name]
    op = getattr(ns, data.name, None)
    if op is None:
        return

    def make_triton_fallback(op):
        # [FUNCTION DOC] HELPER: Return an eager fallback handler for the op when the override data has no triton implementation; otherwise returns None (no fallback).
        if data.triton is None:
            return fallback_handler(op)

    if isinstance(op, torch._ops.OpOverloadPacket):
        for ol in op.op_overloads():
            yield ol, data.type_promotion_kind, make_triton_fallback(ol)
    else:
        yield op, data.type_promotion_kind, make_triton_fallback(op)


for name in pointwise_overrides_data:
    for op, type_promotion_kind, triton_fallback in _get_pointwise_overrides(
        aten, name
    ):
        register_pointwise(
            op,
            name=name,
            type_promotion_kind=type_promotion_kind,
            triton_fallback=triton_fallback,
        )

    for op, type_promotion_kind, triton_fallback in _get_pointwise_overrides(
        prims, name
    ):
        register_pointwise(
            op,
            name=name,
            type_promotion_kind=type_promotion_kind,
            triton_fallback=triton_fallback,
        )


foreach_add_list = register_foreach_pointwise(
    aten._foreach_add.List, add, allow_alpha=True
)
foreach_add_scalar = register_foreach_pointwise(
    aten._foreach_add.Scalar, add, allow_alpha=True
)
register_foreach_pointwise(aten._foreach_add.Tensor, add, allow_alpha=True)
foreach_mul_list = register_foreach_pointwise(aten._foreach_mul.List, mul)
register_foreach_pointwise(aten._foreach_mul.Tensor, mul)
foreach_mul_scalar = register_foreach_pointwise(aten._foreach_mul.Scalar, mul)
register_foreach_pointwise(aten._foreach_sub.List, sub, allow_alpha=True)
register_foreach_pointwise(aten._foreach_sub.Scalar, sub, allow_alpha=True)
register_foreach_pointwise(aten._foreach_neg.default, neg)
register_foreach_pointwise(aten._foreach_abs.default, abs)
register_foreach_pointwise(aten._foreach_pow.Scalar, pow)
register_foreach_pointwise(aten._foreach_pow.List, pow)
register_foreach_pointwise(aten._foreach_pow.ScalarAndTensor, pow)
foreach_div_list = register_foreach_pointwise(aten._foreach_div.List, div)
register_foreach_pointwise(aten._foreach_div.Tensor, div)
foreach_div_scalar = register_foreach_pointwise(aten._foreach_div.Scalar, div)
register_foreach_pointwise(aten._foreach_sqrt, sqrt)
register_foreach_pointwise(aten._foreach_rsqrt, rsqrt)
register_foreach_pointwise(aten._foreach_maximum.List, maximum)
register_foreach_pointwise(aten._foreach_maximum.Scalar, maximum)
register_foreach_pointwise(aten._foreach_minimum.List, minimum)
register_foreach_pointwise(aten._foreach_minimum.Scalar, minimum)
register_foreach_pointwise(aten._foreach_clamp_min.List, maximum)
register_foreach_pointwise(aten._foreach_clamp_min.Scalar, maximum)
register_foreach_pointwise(aten._foreach_clamp_max.List, minimum)
register_foreach_pointwise(aten._foreach_clamp_max.Scalar, minimum)
register_foreach_pointwise(aten._foreach_reciprocal, reciprocal)
register_foreach_pointwise(aten._foreach_sign, sign)
register_foreach_pointwise(aten._foreach_clone, clone)
foreach_copy = register_foreach_pointwise(aten._foreach_copy, copy)


# these are only encountered as outputs of the graph
# reinplacing epilogue copies improves compile time
# by removing extra buffers sent to the scheduler.
def register_foreach_inplace(aten_op, outplace_aten_op, outplace_op):
    # [FUNCTION DOC] HELPER: Register the in-place foreach op aten_op as the out-of-place lowering followed by mutation of each list element in place (aliasing allowed). Records the in-place/out-of-place pairing in global tables and registers the foreach lowering.
    inplaceable_foreach_ops[outplace_aten_op] = aten_op
    inplace_foreach_ops.add(aten_op)

    def fn(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Run the out-of-place foreach lowering and mutate each input list element with its result; returns the mutated list.
        results = outplace_op(*args, **kwargs)
        mut_results = []
        for arg, result in zip(args[0], results):
            mut_results.append(mutate_to(arg, result, unsafe_alias=True))

        return mut_results

    _register_foreach_lowering(aten_op, fn)


register_foreach_inplace(
    aten._foreach_add_.List, aten._foreach_add.List, foreach_add_list
)
register_foreach_inplace(
    aten._foreach_add_.Scalar, aten._foreach_add.Scalar, foreach_add_scalar
)
register_foreach_inplace(
    aten._foreach_mul_.List, aten._foreach_mul.List, foreach_mul_list
)
register_foreach_inplace(
    aten._foreach_mul_.Scalar, aten._foreach_mul.Scalar, foreach_mul_scalar
)
register_foreach_inplace(
    aten._foreach_div_.List, aten._foreach_div.List, foreach_div_list
)
register_foreach_inplace(
    aten._foreach_div_.Scalar, aten._foreach_div.Scalar, foreach_div_scalar
)
register_foreach_inplace(
    aten._foreach_copy_.default, aten._foreach_copy.default, foreach_copy
)
register_foreach_inplace(
    aten._foreach_addcmul_.Scalar,
    aten._foreach_addcmul.Scalar,
    _foreach_addcmul_scalar,
)
register_foreach_inplace(
    aten._foreach_addcdiv_.Scalar,
    aten._foreach_addcdiv.Scalar,
    _foreach_addcdiv_scalar,
)


def register_inplace(aten_op, outplace_op):
    # [FUNCTION DOC] HELPER: Register an in-place lowering for aten_op: computes the out-of-place result, casts it to the first argument's dtype, mutates the first argument with it. Returns the lowering.
    @register_lowering(aten_op, type_promotion_kind=None)
    def fn(*args, **kwargs):
        # [FUNCTION DOC] HELPER: In-place lowering body: compute the result with the out-of-place op, cast to the destination dtype, mutate and return args[0].
        result = outplace_op(*args, **kwargs)
        result = to_dtype(result, args[0].get_dtype())
        return mutate_to(args[0], result)

    return fn


register_inplace(aten.add_, add)
register_inplace(aten.bitwise_and_, bitwise_and)
register_inplace(aten.bitwise_left_shift_, bitwise_left_shift)
register_inplace(aten.bitwise_not_, bitwise_not)
register_inplace(aten.bitwise_or_, bitwise_or)
register_inplace(aten.bitwise_right_shift_, bitwise_right_shift)
register_inplace(aten.bitwise_xor_, bitwise_xor)
register_inplace(aten.mul_, mul)
register_inplace(aten.div_.Tensor, div)
register_inplace(aten.div_.Tensor_mode, div_mode)
register_inplace(aten.logical_and_, logical_and)
register_inplace(aten.logical_not_, logical_not)
register_inplace(aten.logical_or_, logical_or)
register_inplace(aten.logical_xor_, logical_xor)
register_inplace(aten.sub_, sub)
register_inplace(aten.relu_, relu)
register_inplace(aten.sigmoid_, sigmoid)


register_lowering(aten.__and__)(bitwise_and)
register_lowering(aten.__lshift__)(bitwise_left_shift)
register_lowering(aten.__or__)(bitwise_or)
register_lowering(aten.__rshift__)(bitwise_right_shift)
register_lowering(aten.__xor__)(bitwise_xor)

register_inplace(aten.__iand__, aten.__and__)
register_inplace(aten.__ilshift__, aten.__lshift__)
register_inplace(aten.__ior__, aten.__or__)
register_inplace(aten.__irshift__, aten.__rshift__)
register_inplace(aten.__ixor__, aten.__xor__)


@register_lowering(aten.sym_constrain_range)
def sym_constrain_range(a, min=None, max=None):
    # [FUNCTION DOC] Lowering for aten.sym_constrain_range: no-op returning None.
    return None


def _record_symbolic_input_source(
    tensor: Any,
    dim: int,
    expr: sympy.Expr,
    kind: Literal["size", "stride"],
) -> None:
    # [FUNCTION DOC] HELPER: If expr is a plain sympy symbol and tensor is a graph input buffer, remember on the graph (first registration wins) which input name, kind (size or stride) and dim define that symbol. Otherwise does nothing.
    if not isinstance(expr, sympy.Symbol) or not isinstance(tensor, TensorBox):
        return

    if not isinstance(tensor.data, ir.StorageBox) or not isinstance(
        tensor.data.data, ir.InputBuffer
    ):
        return

    name = tensor.get_name()
    if name not in V.graph.graph_inputs:
        return

    V.graph.symbolic_input_sources.setdefault(expr, (name, kind, int(dim)))


@register_lowering(aten.sym_size.int)
def sym_size(a, dim):
    # [FUNCTION DOC] Lowering for aten.sym_size.int: returns the sympy expression of the node's symbolic value (recording the symbolic input source) or the int when static.
    val = V.graph.current_node.meta["val"]
    if isinstance(val, torch.SymInt):
        expr = val.node.expr
        _record_symbolic_input_source(a, dim, expr, "size")
        return expr
    else:
        return int(val)


@register_lowering(aten.sym_stride.int)
def sym_stride(a, dim):
    # [FUNCTION DOC] Lowering for aten.sym_stride.int: returns the sympy expression of the node's symbolic value (recording the source) or the int when static.
    val = V.graph.current_node.meta["val"]
    if isinstance(val, torch.SymInt):
        expr = val.node.expr
        _record_symbolic_input_source(a, dim, expr, "stride")
        return expr
    else:
        return int(val)


@register_lowering(aten.sym_numel)
def sym_numel(a):
    # [FUNCTION DOC] Lowering for aten.sym_numel: returns the element count of the tensor.
    return a.get_numel()


def _unwrap_symbolic_magic_arg(x):
    # [FUNCTION DOC] HELPER: Convert symbolic values to their sympy expressions and python numbers to sympy numbers; leave other values unchanged.
    if isinstance(x, SymTypes):
        return x.node.expr
    if isinstance(x, (int, float, bool)):
        return sympy.sympify(x)
    return x


for method, func in magic_methods.items():

    @register_lowering(method_to_operator(method))
    def wrapped(*args, _func=func, **kwargs):
        # [FUNCTION DOC] HELPER: Lowering for a symbolic magic method (arithmetic or comparison on symbolic numbers). Returns the node's metadata symbolic expression when present; if any argument is symbolic or sympy, applies the method to unwrapped sympy values;
        # otherwise applies it to the plain values.
        node = V.graph.current_node
        meta_val = node.meta.get("val") if node is not None else None
        if isinstance(meta_val, SymTypes):
            return meta_val.node.expr
        if any(
            isinstance(x, (SymTypes, sympy.Basic))
            for x in itertools.chain(args, kwargs.values())
        ):
            return _func(
                *(_unwrap_symbolic_magic_arg(x) for x in args),
                **{k: _unwrap_symbolic_magic_arg(v) for k, v in kwargs.items()},
            )
        return _func(*args, **kwargs)


@register_lowering(torch.sym_sum)
def sym_sum(*args):
    # [FUNCTION DOC] Lowering for torch.sym_sum: accepts one list/tuple or varargs and returns the sympy sum.
    # sym_sum can be called as sym_sum([a, b]) or sym_sum(a, b).
    # Normalize to a flat list before summing.
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        args = args[0]
    return sympy.Add(*args)


@register_lowering(aten._foobar)
def foobar(self, *args, **kwargs):
    # [FUNCTION DOC] HELPER: Debug lowering that always raises NotImplementedError.
    raise NotImplementedError("Helpful for debugging")


@register_lowering(torch.ops._inductor_test.realize)
def _realize(x):
    # [FUNCTION DOC] HELPER: Test lowering that forces realization of x and returns a clone.
    x.realize()
    return clone(x)


@register_lowering(torch.ops.inductor.resize_storage_bytes_)
def resize_storage_bytes_(variable, new_size):
    # [FUNCTION DOC] Lowering for inductor resize_storage_bytes_: realizes the variable, creates a ResizeStorageBytes node, and returns the variable.
    variable.realize()
    ir.ResizeStorageBytes(variable, new_size)
    return variable


@register_lowering(torch.ops.aten.set_.source_Tensor)
def set__source_tensor(self, source_tensor):
    # [FUNCTION DOC] Lowering for aten.set_.source_Tensor: realizes both tensors and returns a SetSourceTensorKernel result.
    self.realize()
    source_tensor.realize()
    return TensorBox.create(ir.SetSourceTensorKernel(self, source_tensor))


@register_lowering(torch.ops.aten.shallow_copy_data_.default)
def shallow_copy_data_(self, source_tensor):
    # [FUNCTION DOC] Lowering for aten.shallow_copy_data_: realizes both tensors and returns a ShallowCopyDataKernel result.
    self.realize()
    source_tensor.realize()
    return TensorBox.create(ir.ShallowCopyDataKernel(self, source_tensor))


if hasattr(torch.ops.fsdp, "copy_"):

    @register_lowering(torch.ops.fsdp.copy_.default)
    def fsdp_copy_(dst, src):
        # [FUNCTION DOC] Lowering for fsdp.copy_: like copy_, returns dst if src is dst; otherwise moves src to dst's device, dtype and shape and mutates dst. Defined only if the op exists.
        if dst is src:
            # dst.copy_(dst) can happen from the reinplacing pass
            return dst
        src = to_device(src, dst.get_device())
        src = to_dtype(src, dst.get_dtype())
        src = expand(src, dst.get_size())
        return mutate_to(dst, src)


@register_lowering(torch.ops.aten.resize)
def resize(x, size, *, memory_format=None):
    # [FUNCTION DOC] Lowering for aten.resize. Requires a TensorBox and list/tuple size (raises an error otherwise); preserve_format raises RuntimeError and channels-last needs 4 or 5 dims. Existing elements are kept in storage order (a flat view; overlapping inputs use a
    # logical flat view); new elements are filled with an uninitialized-memory value (NaN, max int or True under deterministic fill mode, else zero). An empty input yields a filled tensor. Returns a Pointwise with the requested memory format strides.
    if not (isinstance(x, TensorBox)):
        raise AssertionError("expected: isinstance(x, TensorBox)")
    if not (isinstance(size, (list, tuple))):
        raise AssertionError("expected: isinstance(size, (list, tuple))")

    if memory_format is None:
        memory_format = torch.contiguous_format
    if memory_format == torch.preserve_format:
        raise RuntimeError(f"unsupported memory format: {memory_format}")

    if memory_format == torch.channels_last:
        if len(size) != 4:
            raise AssertionError("expected: len(size) == 4")
    if memory_format == torch.channels_last_3d:
        if len(size) != 5:
            raise AssertionError("expected: len(size) == 5")

    old_numel = x.get_numel()
    dtype = x.get_dtype()
    device = x.get_device_or_error()

    if (
        torch.are_deterministic_algorithms_enabled()
        and torch.utils.deterministic.fill_uninitialized_memory  # type: ignore[attr-defined]
    ):
        if is_float_dtype(dtype):
            uninitialized_val = float("nan")
        elif is_integer_dtype(dtype):
            uninitialized_val = torch.iinfo(dtype).max
        else:
            uninitialized_val = True
    else:
        # using zero as that is what empty does
        uninitialized_val = 0.0

    if V.graph.sizevars.statically_known_equals(old_numel, 0):  # type: ignore[arg-type]
        return full(size, uninitialized_val, dtype=dtype, device=device)

    strides = x.maybe_get_stride()
    has_overlapping = strides is not None and any(
        V.graph.sizevars.statically_known_equals(s, 0) for s in strides
    )
    if has_overlapping:
        # overlapping: provide a contiguous logical view
        x_flat = view(x, [old_numel])
    else:
        # non-overlapping: keep storage order
        if isinstance(x.data, ir.BaseView):
            x.data = x.data.unwrap_view()
        x_flat = as_strided(x, [old_numel], [1])
    flat_loader = x_flat.make_loader()
    out_stride = ir.FlexibleLayout.stride_ordered_for_memory_format(size, memory_format)
    out_indexer = ir.FixedLayout(device, dtype, size, out_stride).make_indexer()

    def inner_fn(idx):
        # [FUNCTION DOC] HELPER: Load the flat source element if the output position is within the old element count, else the uninitialized fill value.
        flat_index = out_indexer(idx)
        flat_index_expr = ops.index_expr(flat_index, torch.int64)
        limit = ops.index_expr(old_numel, torch.int64)
        mask = ops.lt(flat_index_expr, limit)
        return ops.masked(mask, lambda: flat_loader([flat_index]), uninitialized_val)

    out = Pointwise.create(
        device=device, dtype=dtype, inner_fn=inner_fn, ranges=list(size)
    )
    return out


from torch._higher_order_ops.auto_functionalize import auto_functionalized


make_fallback(auto_functionalized)


@register_lowering(triton_kernel_wrapper_mutation)
def triton_kernel_wrap_(
    *,
    kernel_idx,
    constant_args_idx,
    grid,
    tma_descriptor_metadata,
    kwargs,
    launch_kwargs=None,
):
    # [FUNCTION DOC] Lowering for the triton kernel wrapper mutation higher-order op: fetches constant args from the kernel side table and registers a UserDefinedTritonKernel node with the grid, descriptors and merged kernel arguments. Returns a mapping of the tensor arguments that the kernel may mutate.
    from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

    constant_args = kernel_side_table.get_constant_args(constant_args_idx)
    ir.UserDefinedTritonKernel(
        kernel_idx=kernel_idx,
        grid=grid,
        tma_descriptor_metadata=tma_descriptor_metadata,
        kernel_args={**kwargs, **constant_args},
        launch_kwargs=() if launch_kwargs is None else launch_kwargs,
    )
    return {key: val for key, val in kwargs.items() if isinstance(val, TensorBox)}


@register_lowering(torch.ops.higher_order.cond, type_promotion_kind=None)
def cond(
    pred, true_fn, false_fn, operands
) -> list[ir.TensorBox | ir.ShapeAsConstantBuffer]:
    # [FUNCTION DOC] Lowering for torch.cond: if any operand is on triton, records a reason that disables cudagraphs. Creates a Switch node over [false_fn, true_fn] (the predicate converts to index 0 or 1) and returns the outputs as TensorBoxes.
    # TODO: when graph_partition is enabled, skip - partitioning handles control flow
    # we run into memory cleanup issue
    if any(isinstance(x, IRNode) and is_triton(x) for x in [pred, *operands]):
        msg = "control flow operator: torch.cond."
        if stack_trace := V.graph.current_node.meta.get("stack_trace", None):
            msg = f"{msg} Found from : \n {stack_trace}"
        V.graph.disable_cudagraphs_reason = msg

    # The branches are reordered to [false_fn, true_fn]
    # because during codegen the pred is converted to an integer with True -> 1 and False -> 0.
    # When iterating over the branches the false_fn is associated index 0.
    result = ir.Switch.create(pred, [false_fn, true_fn], operands, is_cond=True)
    return list(map(TensorBox.create, result))  # pyrefly: ignore no-matching-overload


@register_lowering(torch.ops.higher_order.switch, type_promotion_kind=None)
def switch(index, branches, operands) -> list[ir.TensorBox | ir.ShapeAsConstantBuffer]:
    # [FUNCTION DOC] Lowering for torch.switch: always records a reason that disables cudagraphs; creates a Switch node over the branches and returns the outputs as TensorBoxes.
    # TODO: cudagraph support for torch.switch is not yet implemented; always disable.
    msg = "control flow operator: torch.switch."
    if stack_trace := V.graph.current_node.meta.get("stack_trace", None):
        msg = f"{msg} Found from : \n {stack_trace}"
    V.graph.disable_cudagraphs_reason = msg

    result = ir.Switch.create(index, branches, operands)
    return list(map(TensorBox.create, result))  # pyrefly: ignore no-matching-overload


@register_lowering(torch.ops.higher_order.while_loop, type_promotion_kind=None)
def while_loop(cond_fn, body_fn, carried_inputs, additional_inputs, stack_output=False):
    # [FUNCTION DOC] Lowering for torch.while_loop: when graph partitioning is off and any input is on triton, records a reason disabling cudagraphs. Creates a WhileLoop node (optionally stacking outputs) and returns the wrapped outputs; raises an error if the result is not a sequence.
    # TODO: when graph_partition is enabled, skip - partitioning handles control flow
    # we run into memory cleanup issue
    if not config.graph_partition and any(
        isinstance(x, IRNode) and is_triton(x)
        for x in carried_inputs + additional_inputs
    ):
        msg = "control flow operator: torch.while_loop."
        if stack_trace := V.graph.current_node.meta.get("stack_trace", None):
            msg = f"{msg} Found from : \n {stack_trace}"
        V.graph.disable_cudagraphs_reason = msg

    result = ir.WhileLoop.create(
        cond_fn, body_fn, carried_inputs, additional_inputs, stack_output
    )
    if not (isinstance(result, Sequence)):
        raise AssertionError("expected: isinstance(result, Sequence)")
    return list(map(ir.WhileLoop._maybe_wrap_as_tensor_box, result))


register_lowering(
    torch.ops.higher_order.while_loop_stack_output, type_promotion_kind=None
)(functools.partial(while_loop, stack_output=True))


@register_lowering(torch.ops.higher_order.invoke_subgraph, type_promotion_kind=None)
def invoke_subgraph(subgraph_fn: ir.Subgraph, identifier: str, *operands):
    # [FUNCTION DOC] Lowering for invoke_subgraph: creates an InvokeSubgraph node for the subgraph and operands and returns outputs as TensorBoxes. The identifier is unused.
    result = ir.InvokeSubgraph.create(subgraph_fn, *operands)
    return list(map(TensorBox.create, result))  # type: ignore[call-overload]


_MISSING = object()


def process_subgraph_nodes(graph_module: torch.fx.GraphModule, args: list[Any]):
    # [FUNCTION DOC] HELPER: Execute a subgraph's FX nodes through the current graph lowering. Placeholders bind to the given args, other nodes run via run_node (temporarily setting the current node and restoring it), and the output node's value is returned.
    # Raises an error if a node is already in the environment, and RuntimeError if there is no output node.
    """Process nodes from a FX graph by executing them through V.graph.

    This is a common pattern for executing a subgraph's nodes:
    - Placeholder nodes are mapped to the provided args
    - Output nodes return their result
    - Other nodes are executed via V.graph.run_node

    """
    output = _MISSING

    for i, node in enumerate(graph_module.graph.nodes):
        if node.op == "placeholder":
            if node in V.graph.env:
                raise AssertionError("expected: node not in V.graph.env")
            V.graph.env[node] = args[i]
            continue
        elif node.op == "output":
            output_args, kwargs = V.graph.fetch_args_kwargs_from_env(node)
            output = torch.fx.Interpreter.output(V.graph, node, output_args, kwargs)
        else:
            if node in V.graph.env:
                raise AssertionError("expected: node not in V.graph.env")
            # Track current node for error diagnostics; restore after run_node to handle nested calls correctly
            saved_current_node = V.graph.current_node
            try:
                V.graph.current_node = node
                V.graph.env[node] = V.graph.run_node(node)
            finally:
                V.graph.current_node = saved_current_node

    if output is _MISSING:
        raise RuntimeError("No output node found in graph")

    return output


# Import the control_deps_op HOP for lowering
from torch._inductor.fx_passes.control_dependencies import control_deps


@register_lowering(control_deps, type_promotion_kind=None)
def control_deps_op_lowering(additional_deps, subgraph_fn, *args):
    # [FUNCTION DOC] Lowering for control_deps. Realizes all dependency IR nodes (and recorded names of void ops from earlier control_deps nodes), runs the subgraph via process_subgraph_nodes, records names of void-layout ops created, adds each dependency as an additional buffer
    # dependency of every operation created in the subgraph, and for pass-through values (inputs also returned) adds an ordering barrier depending on the subgraph's buffers. Returns the subgraph output. Raises an error if there are no dependencies or arg counts mismatch.
    """
    Lower control_deps_op by ensuring dependencies are realized and tracking them.

    The control_deps_op HOP makes dependencies explicit in the graph. During lowering:
    1. Realize all additional dependencies to ensure they're computed
    2. Execute the target operation normally
    3. Track the dependencies for the scheduler
    """
    # Pair lowered deps with their original FX nodes so we can handle void ops
    # (e.g. record_event) that lower to None but still create operations that
    # subsequent control_deps nodes (e.g. wait_event) must be ordered after.
    original_dep_nodes = V.graph.current_node.args[0]
    if not (isinstance(original_dep_nodes, tuple)):
        raise AssertionError("expected: isinstance(original_dep_nodes, tuple)")

    dep_names: list[str] = []
    for dep, orig_node in zip(additional_deps, original_dep_nodes, strict=True):
        dep_ir_nodes = [
            dep_leaf
            for dep_leaf in pytree.tree_leaves(dep)
            if isinstance(dep_leaf, IRNode)
        ]
        for dep_ir_node in dep_ir_nodes:
            dep_ir_node.realize()
            dep_names.append(dep_ir_node.get_name())
        if isinstance(orig_node, torch.fx.Node):
            found = V.graph._void_ctrl_dep_op_names.get(orig_node, [])
            dep_names.extend(found)

    original_args = V.graph.current_node.args
    arg_offset = 2  # first two args (additional_deps, subgraph)
    if len(args) + arg_offset != len(original_args):
        raise AssertionError("expected: len(args) + arg_offset == len(original_args)")

    operation_len = len(V.graph.operations)
    if len(subgraph_fn.graph_module.graph.find_nodes(op="placeholder")) != len(args):
        raise AssertionError(
            'expected: len(subgraph_fn.graph_module.graph.find_nodes(op="pl...'
        )

    # Process subgraph nodes using the shared helper
    output = process_subgraph_nodes(subgraph_fn.graph_module, list(args))

    if not (additional_deps):
        raise AssertionError("expected: additional_deps")

    new_ops = V.graph.operations[operation_len:]

    # Store buffer names of void ops (e.g. record_event has NoneLayout) so that
    # subsequent control_deps nodes (e.g. wait_event) can depend on them even
    # though void ops don't produce a usable tensor value.  We detect void ops
    # by NoneLayout rather than checking `output is None`, because the overall
    # control_deps output may be a passthrough tuple (not None) even when the
    # subgraph contains a void op.
    #
    # Skip ops that are not themselves named buffers (op.name is None).  This
    # excludes multi-output kernels like UserDefinedTritonKernel, whose buffer
    # identity lives on their MutationOutput children rather than the op: the
    # op is registered only as an Operation, and Buffer.get_name() asserts on
    # the missing self.name.  Downstream ordering for those ops is already
    # captured through the named MutationOutput buffers they produce.
    void_names = [
        op.get_name()
        for op in new_ops
        if isinstance(op, ir.Buffer)
        and op.name is not None
        and isinstance(op.layout, ir.NoneLayout)
    ]
    if void_names:
        V.graph._void_ctrl_dep_op_names[V.graph.current_node] = void_names

    # some operators, like wait_tensor, just return their input,
    # so its more robust to add dep to the operation itself,
    # otherwise you can have a cycle of
    # a = coll
    # b = control_deps(a, mm, ...)
    # c = control_deps(b, wait, ...)
    # if c == a, then you have a cycle.
    subgraph_ops = V.graph.operations[operation_len:]
    for op in subgraph_ops:
        for dep_name in dep_names:
            op_name = op.operation_name
            if op_name is None:
                raise AssertionError("expected: op_name is not None")
            V.graph.additional_buffer_deps[op_name].add(dep_name)

    # Pass-through versioning: inputs that are also outputs get an
    # OrderingOutput so future readers are ordered after all subgraph ops.
    # TODO: if control_deps ever wraps ops that truly mutate a pass-through
    # (not just ordering), this needs MutationOutput instead of OrderingOutput.
    input_names = OrderedSet()
    for a in args:
        if isinstance(a, IRNode):
            a.realize()
            input_names.add(a.get_name())
    output_leaves = list(pytree.tree_leaves(output)) if output is not None else []
    passthrough_vals = []
    for v in output_leaves:
        if isinstance(v, IRNode):
            v.realize()
            if v.get_name() in input_names:
                passthrough_vals.append(v)
    for val in passthrough_vals:
        barrier = ir.OrderingBarrier(val)
        barrier_op = barrier.get_operation_name()
        for op in subgraph_ops:
            op_name = op.operation_name
            if op_name is not None:
                buf_name = op.get_buffer_name()
                if buf_name is not None:
                    V.graph.additional_buffer_deps[barrier_op].add(buf_name)

    return output


@register_lowering(torch._higher_order_ops.invoke_quant, type_promotion_kind=None)
def invoke_quant_tracer(subgraph_fn: ir.Subgraph, *operands, scheme=None):
    # [FUNCTION DOC] Lowering for the invoke_quant higher-order op. Requires quant_options on the current node (raises an error otherwise). Runs the subgraph nodes through the graph lowering (placeholders bound to operands); at the output node realizes every output,
    # records their operation names as invoke_quant ops (and as low-precision codegen ops when the options request it), and returns the output.
    output = None
    quant_options = V.graph.current_node.meta.get("quant_options", None)
    if quant_options is None:
        raise AssertionError("expected: quant_options is not None")

    for i, node in enumerate(subgraph_fn.graph_module.graph.nodes):
        if node.op == "placeholder":
            V.graph.env[node] = operands[i]
            continue
        # todo getattr
        elif node.op == "output":
            args, kwargs = V.graph.fetch_args_kwargs_from_env(node)

            for v in itertools.chain(args, kwargs.values()):
                v.realize()

                if quant_options.codegen_low_precision:
                    V.graph.low_precision_codegen_ops.add(v.get_operation_name())

                V.graph.invoke_quant_ops.add(v.get_operation_name())

            output = torch.fx.Interpreter.output(V.graph, node, args, kwargs)
        else:
            V.graph.env[node] = V.graph.run_node(node)

    return output


@register_lowering(associative_scan_op, type_promotion_kind=None)
def associative_scan(combine_fn: ir.Subgraph, xs, additional_inputs: tuple[Any, ...]):
    # [FUNCTION DOC] Lowering for the associative_scan higher-order op (pointwise combine mode). Raises RuntimeError if any input's device lacks scan support, if lifted additional inputs do not match or are unsupported (non-numeric or used by the combine graph). Lowers the combine
    # subgraph to a pointwise function over leaf values of both operands plus additional inputs, then creates a Scan over the first dim of the inputs with no eager fallback. Returns the scan outputs; raises RuntimeError if code cannot be generated.
    from .subgraph_lowering import InputDescriptor, lower_pointwise_subgraph

    # combine_mode="pointwise" codegen requires a backend with scan support
    # (e.g. Triton on CUDA/XPU). The eager wrapper no longer enforces a device
    # requirement, so check every leaf here -- before lower_pointwise_subgraph
    # runs -- to raise a clear device-specific error. ir.Scan.create re-checks
    # the same feature on xs[0] below and otherwise fails with a generic "Unable
    # to generate code" error. See https://github.com/pytorch/pytorch/issues/186594.
    for x in xs:
        device = x.get_device()
        if not V.graph.has_feature(device, BackendFeature.SCAN):
            device_str = device.type if device is not None else "unknown device"
            raise RuntimeError(
                "associative_scan with combine_mode='pointwise' is not supported "
                f"on {device_str}. Try to use combine_mode='generic'."
            )

    num_scan_inputs = 2 * len(xs)
    placeholders = [
        node for node in combine_fn.graph_module.graph.nodes if node.op == "placeholder"
    ]
    additional_placeholders = placeholders[num_scan_inputs:]
    if len(additional_placeholders) != len(additional_inputs):
        raise RuntimeError("Unable to generate code for associative_scan op")

    # Dynamic shapes can lift symbols from the scan element metadata into the
    # combine graph even when the combine function does not read them.
    unsupported_inputs = []
    for arg, placeholder in zip(additional_inputs, additional_placeholders):
        if not isinstance(arg, (int, sympy.Basic)) or placeholder.users:
            unsupported_inputs.append(arg)
    if unsupported_inputs:
        raise RuntimeError(
            "Unable to generate code for associative_scan op, because there are "
            "unsupported lifted arguments"
        )

    subgraph_inputs = [
        InputDescriptor(dtype=x.get_dtype(), device=x.get_device())
        for x in itertools.chain(xs, xs)
    ]
    subgraph_inputs.extend(additional_inputs)
    lowered_combine_fn = lower_pointwise_subgraph(combine_fn, subgraph_inputs)

    def wrapped_combine_fn(lhs, rhs):
        # [FUNCTION DOC] HELPER: Combine two carried tuples by flattening them and calling the lowered subgraph with the additional inputs.
        return lowered_combine_fn(
            *pytree.tree_leaves(lhs),
            *pytree.tree_leaves(rhs),
            *additional_inputs,
        )

    kwargs = _make_scan_inner(xs[0], axis=0, dtype=None)
    kwargs["dtypes"] = tuple(x.get_dtype() for x in xs)
    kwargs["inner_fns"] = tuple(x.make_loader() for x in xs)
    result = ir.Scan.create(
        combine_fn=wrapped_combine_fn,
        can_fallback_to_aten=False,
        **kwargs,
    )
    if result[0] is None:
        raise RuntimeError("Unable to generate code for associative_scan op")
    return result


@register_lowering(torch.ops.prims._sink_tokens.default)
def _sink_tokens(tokens):
    # [FUNCTION DOC] Lowering for prims._sink_tokens: no-op returning None.
    return None


@register_lowering(torch.ops.prims._make_token.default)
def _make_token():
    # [FUNCTION DOC] Lowering for prims._make_token: no-op returning None.
    return None


@register_lowering(torch.ops.higher_order.with_effects, type_promotion_kind=None)
def with_effects(token, op, *args, **kwargs):
    # [FUNCTION DOC] Lowering for with_effects: lowers the wrapped op directly (using its registered lowering, realizing results, or an eager fallback kernel), then marks every newly created operation as having side effects and chains it after the previous operation with the same effect type
    # (resolving the effect of invoke_subgraph from the tracing cache). Raises an error if no operations were created or multiple effects are found. Returns the token followed by the op results, shaped by the op schema (a warning is logged and a list output assumed if the schema is unavailable).
    """
    We lower the operator directly, and then we add StarDep dependencies to all
    the newly created nodes in the graph.
    """
    from torch._higher_order_ops.effects import _get_effect, _get_schema

    # Get effect type
    effect_type = _get_effect(op)
    if effect_type is None and op is torch.ops.higher_order.invoke_subgraph:
        from torch._guards import InvokeSubgraphCache, TracingContext

        tracing_ctx = TracingContext.try_get()
        if tracing_ctx:
            invoke_subgraph_cache = tracing_ctx.hop_dispatch_set_cache.get_cache(
                torch.ops.higher_order.invoke_subgraph
            )
            if invoke_subgraph_cache:
                if not (isinstance(invoke_subgraph_cache, InvokeSubgraphCache)):
                    raise AssertionError(
                        "expected: isinstance(invoke_subgraph_cache, InvokeSubgraphCache)"
                    )
                # args[1] is identifier
                effects = invoke_subgraph_cache.get_effects(args[1])
                if effects:
                    if len(effects) != 1:
                        raise AssertionError("Multiple effects NYI")
                    effect_type = next(iter(effects))

    # Track operations before
    operation_len = len(V.graph.operations)

    # Lower the op
    if op in lowerings:
        result = lowerings[op](*args, **kwargs)
        # Realize so that we can get the ops to show up in V.graph.operations
        pytree.tree_map_only(TensorBox, lambda a: a.realize(), result)
    else:

        def wrap_tensors(x):
            # [FUNCTION DOC] HELPER: Wrap IR nodes for lowering; pass other values through.
            return x.wrap_for_lowering() if isinstance(x, ir.IRNode) else x

        result = pytree.tree_map(
            wrap_tensors, ir.FallbackKernel.create(op, *args, **kwargs)
        )

    # Get all the operations created during the lowering above, and add StarDeps
    # to the previous node with the same effect
    if len(V.graph.operations[operation_len:]) <= 0:
        raise AssertionError(
            f"No operation nodes were generated when lowering effectful operator {op}."
        )
    if effect_type:
        prev_effect_buffer = V.graph.effectful_ops.get(effect_type)
        for new_op in V.graph.operations[operation_len:]:
            # Patch has_side_effects to return True
            new_op.has_side_effects = lambda: True  # pyrefly: ignore[missing-attribute]
            if prev_effect_buffer:
                op_name = new_op.get_name()  # pyrefly: ignore[missing-attribute]
                V.graph.additional_star_deps[op_name].add(prev_effect_buffer.get_name())
        # Update the effectful ops chain to point to the latest operation
        V.graph.effectful_ops[effect_type] = (
            new_op  # pyrefly: ignore[unsupported-operation]
        )

    try:

        def convert_ir_to_value(a):
            # [FUNCTION DOC] HELPER: Convert an IR value to a concrete value for schema lookup: torch-bind objects to their value, tensor boxes to their example tensor when available, otherwise unchanged.
            if isinstance(a, ir.TorchBindObject):
                return a.get_value()
            elif isinstance(a, TensorBox):
                # TensorBox wraps StorageBox, which wraps the actual buffer
                # We need to get the example tensor from the inner buffer
                try:
                    storage = a.data
                    if hasattr(storage, "data") and hasattr(
                        storage.data, "get_example"
                    ):
                        return storage.data.get_example()
                except (AttributeError, NotImplementedError):
                    pass
                # Fall back to returning the TensorBox itself if get_example fails
                return a
            return a

        schema_args, schema_kwargs = pytree.tree_map(
            convert_ir_to_value, (args, kwargs)
        )
        schema = _get_schema(op, schema_args, schema_kwargs)
    except RuntimeError as e:
        error_msg = str(e)
        log.warning(
            "Failed to get schema for %s: %s. Assuming list output", op, error_msg
        )
        if isinstance(result, (tuple, list)):
            return (token, *result)
        else:
            return (token, result)

    if len(schema.returns) == 0:
        return (token, result)
    elif len(schema.returns) == 1:
        return (token, result)
    else:
        return (token, *result)


from .comm_lowering import register_comm_lowerings, register_symm_mem_lowerings


register_comm_lowerings()
register_symm_mem_lowerings()


@register_lowering(inductor_prims.prepare_softmax_online, type_promotion_kind=None)
def prepare_softmax_online(x, dim):
    # [FUNCTION DOC] Lowering for prepare_softmax_online: for a statically known reduction size at least the unroll threshold, creates an online softmax reduction returning (max, sum of exp). Otherwise computes amax, the exponentials, and their sum with ordinary reductions. Returns (max, sum) keeping the reduced dim.
    """
    Lowering inductor_prims.prepare_softmax_online to compute max/sum with
    online softmax reductions when profitable.
    """
    kwargs = _make_reduction_inner(
        x, axis=dim, keepdims=True, dtype=None, override_return_dtype=None
    )

    rnumel = V.graph.sizevars.simplify(sympy_product(kwargs["reduction_ranges"]))

    if V.graph.sizevars.statically_known_geq(
        rnumel, config.unroll_reductions_threshold
    ):
        max_tensor, sum_tensor = OnlineSoftmaxReduction.create(
            input_node=x, num_output=2, **kwargs
        )
        return max_tensor, sum_tensor
    else:
        amax = reduce_amax(x, dim, keepdims=True)
        exp = lowerings[aten.exp](sub(x, amax))
        xsum = sum_(exp, dim, keepdims=True)
        return amax, xsum


@register_lowering(inductor_prims.cvt_e8m0_rceil, type_promotion_kind=None)
def cvt_e8m0_rceil_lowering(inp):
    # [FUNCTION DOC] Lowering for cvt_e8m0_rceil: converts float values to the e8m0 format with round-toward-positive using inline PTX. Requires NVIDIA SM100 or later (NotImplementedError otherwise) and float32, float16 or bfloat16 input (ValueError otherwise); lower precision is upcast to float32.
    # Returns a uint8 tensor.
    """
    Lowering for cvt_e8m0_rceil. Uses PTX cvt.rp.satfinite.ue8m0x2.f32 on NVIDIA SM100+.

    The PTX instruction takes 2 float32 and outputs 2 e8m0 packed in uint16.
    Currently we pass 0.0 as the second input and only use the low byte result.
    """
    # TODO: Optimize to process pairs (pack=2) by creating a custom Pointwise
    # that loads adjacent elements, applies PTX to both, and uses a follow-up
    # kernel to extract the packed uint16 results as uint8.
    if not is_nvidia_sm100_or_later():
        raise NotImplementedError(
            "cvt_e8m0_rceil requires NVIDIA SM100+ (Blackwell) for PTX instruction support"
        )

    dtype = inp.get_dtype()
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(
            f"cvt_e8m0_rceil requires float32, float16, or bfloat16 input, got {dtype}"
        )

    # Upcast bf16/fp16 to float32 for PTX instruction
    if dtype != torch.float32:
        inp = to_dtype(inp, torch.float32)

    fn = functools.partial(
        ops.inline_asm_elementwise,
        asm="cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;",
        constraints="=h,r",
        dtype=torch.uint16,
        is_pure=True,
        pack=1,
    )
    result = make_pointwise(fn)(inp)
    return to_dtype(result, torch.uint8)


@register_lowering(
    torch._higher_order_ops.inline_asm_elementwise, type_promotion_kind=None
)
def lower_inline_asm_elementwise(
    *inputs, asm_str, constraints, dtype, is_pure=True, pack=1
):
    # [FUNCTION DOC] Lowering for the inline_asm_elementwise higher-order op: broadcasts the inputs and builds a Pointwise that applies the inline assembly string with the given constraints, output dtype, purity and packing, upcasting fp16/bf16 results to float32 for fusion.
    inputs = broadcast_tensors(*inputs)

    input_dtypes = tuple(inp.get_dtype() for inp in inputs)
    loaders = [inp.make_loader() for inp in inputs]

    def inner_fn(idx):
        # [FUNCTION DOC] HELPER: Load all inputs at idx and apply the inline assembly elementwise op.
        vals = tuple(loader(idx) for loader in loaders)
        result = ops.inline_asm_elementwise(
            *vals,
            asm=asm_str,
            constraints=constraints,
            dtype=dtype,
            is_pure=is_pure,
            pack=pack,
            input_dtypes=input_dtypes,
        )
        # Inductor computes in fp32 for bf16/fp16. Upcast so fused downstream
        # ops (reductions, etc.) see fp32 values. The Pointwise's storage dtype
        # handles the final downcast on store.
        if dtype in (torch.float16, torch.bfloat16):
            result = ops.to_dtype(result, torch.float32)
        return result

    return ir.Pointwise.create(
        device=inputs[0].get_device(),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=list(inputs[0].get_size()),
    )


# populate lowerings defined in kernel/*
from . import kernel


import_submodule(kernel)

from . import (
    jagged_lowerings,
    mkldnn_lowerings,  # noqa: F401  # registers oneDNN fusion ops on import
    quantized_lowerings,
)


jagged_lowerings.register_jagged_ops()
quantized_lowerings.register_quantized_ops()
quantized_lowerings.register_woq_mm_ops()


@contextlib.contextmanager
def force_fallback(op: torch._ops.OpOverload):
    # [FUNCTION DOC] HELPER: Context manager (for tests) that forces op to use an eager fallback kernel lowering. Requires an OpOverload (raises an error otherwise); on exit restores the previous lowering or removes the registered one.
    """
    A context manager to force fallback an op. Used in unit test
    for FallbackKernel.
    """
    if not (isinstance(op, torch._ops.OpOverload)):
        raise AssertionError("Only OpOverload to make the clean up easier")
    old_handler = lowerings.get(op)
    try:
        register_lowering(op)(fallback_handler(op))
        yield
    finally:
        if old_handler:
            lowerings[op] = old_handler
        else:
            lowerings.pop(op)
